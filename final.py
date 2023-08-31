# Importing required libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, PowerTransformer
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import learning_curve
import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import ParameterGrid


# Reading training and test data
train = pd.read_csv('Train.csv')
test = pd.read_csv('Test.csv')
uber_data = pd.read_csv('Travel_Times_Daily.csv')

# Aggregating the training data based on specific columns and renaming the size column
train_aggregated = train.groupby(['ride_id', 'travel_date', 'travel_time', 'travel_from', 'travel_to', 'car_type', 'max_capacity']).size().reset_index(name='size')

# Converting the 'travel_date' column to datetime format
train_aggregated['travel_date'] = pd.to_datetime(train_aggregated['travel_date'], format='%d-%m-%y')
test['travel_date'] = pd.to_datetime(test['travel_date'])

# Feature Engineering

# Adding 'day_of_week' and 'is_weekend' columns to both training and test data
train_aggregated['day_of_week'] = train_aggregated['travel_date'].dt.dayofweek
test['day_of_week'] = test['travel_date'].dt.dayofweek
train_aggregated['is_weekend'] = (train_aggregated['day_of_week'] >= 5).astype(int)
test['is_weekend'] = (test['day_of_week'] >= 5).astype(int)

# Converting 'travel_time' to a float representing the hour of the day
train_aggregated['travel_time'] = pd.to_datetime(train_aggregated['travel_time']).dt.hour + pd.to_datetime(train_aggregated['travel_time']).dt.minute / 60
test['travel_time'] = pd.to_datetime(test['travel_time']).dt.hour + pd.to_datetime(test['travel_time']).dt.minute / 60

# Adding a 'peak_hour' column based on the travel time
train_aggregated['peak_hour'] = ((train_aggregated['travel_time'] >= 6) & (train_aggregated['travel_time'] <= 9) |
                                (train_aggregated['travel_time'] >= 16) & (train_aggregated['travel_time'] <= 19)).astype(int)
test['peak_hour'] = ((test['travel_time'] >= 6) & (test['travel_time'] <= 9) |
                    (test['travel_time'] >= 16) & (test['travel_time'] <= 19)).astype(int)

# Label encoding for categorical variables
label_encoders = {}
for column in ['travel_from', 'car_type', 'travel_to']:
    le = LabelEncoder()
    train_aggregated[column] = le.fit_transform(train_aggregated[column])
    test[column] = le.transform(test[column])
    label_encoders[column] = le

# Feature selection using RFE (Recursive Feature Elimination)
selected_features = ['travel_time', 'travel_from', 'car_type', 'max_capacity', 'is_weekend', 'peak_hour']
selector = RFE(estimator=GradientBoostingRegressor(), n_features_to_select=5, step=1)
selector = selector.fit(train_aggregated[selected_features], train_aggregated['size'])
selected_features = np.array(selected_features)[selector.support_]

# Separating features and target variable
X = train_aggregated[selected_features].values
y = train_aggregated['size'].values

# Transforming the target variable to handle skewness
y_transformer = PowerTransformer(method='box-cox')
y_transformed = y_transformer.fit_transform(y.reshape(-1, 1)).flatten()

# Hyperparameter tuning using RandomizedSearchCV for XGBoost model
param_distributions = {
    'learning_rate': [0.01, 0.03, 0.05],
    'max_depth': [6, 8, 10],
    'n_estimators': [500, 1000, 1500],
    'subsample': [0.8, 0.9, 1]
}
xgb_model = xgb.XGBRegressor(random_state=42, reg_alpha=0.1, reg_lambda=1)
random_search = RandomizedSearchCV(estimator=xgb_model, param_distributions=param_distributions, 
                                   scoring='neg_mean_squared_error', cv=3, verbose=1, n_jobs=-1, n_iter=10)
random_search.fit(X, y_transformed)
best_xgb_model = random_search.best_estimator_

# Splitting the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y_transformed, test_size=0.2, random_state=42)

# Creating LightGBM datasets
lgb_train = lgb.Dataset(X_train, y_train)
lgb_eval = lgb.Dataset(X_val, y_val, reference=lgb_train)

# Hyperparameter tuning for LightGBM using ParameterGrid
lgb_params_grid = {
    'objective': ['regression'],
    'metric': [{'l2', 'l1'}],
    'num_leaves': [10, 31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'feature_fraction': [0.8, 0.9, 1.0],
    'lambda_l1': [0, 0.1, 0.5],
    'lambda_l2': [0, 0.1, 0.5]
}
best_score = float('inf')
best_params = {}
# Continue with hyperparameter tuning for LightGBM
for params in ParameterGrid(lgb_params_grid):
    temp_model = lgb.train(params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, verbose_eval=False)
    temp_pred = temp_model.predict(X_val, num_iteration=temp_model.best_iteration)
    temp_score = mean_squared_error(y_val, temp_pred)
    if temp_score < best_score:
        best_score = temp_score
        best_params = params

# Training the final LightGBM model with the best parameters
lgb_model = lgb.train(best_params, lgb_train, num_boost_round=1000, valid_sets=lgb_eval, early_stopping_rounds=50)

# Making predictions with both models
xgb_predictions = best_xgb_model.predict(test[selected_features].values)
lgb_predictions = lgb_model.predict(test[selected_features].values, num_iteration=lgb_model.best_iteration)

# Converting the predictions back to the original scale
xgb_predictions_original = y_transformer.inverse_transform(xgb_predictions.reshape(-1, 1)).flatten()
lgb_predictions_original = y_transformer.inverse_transform(lgb_predictions.reshape(-1, 1)).flatten()

# Stacking the models using a meta-model (Linear Regression)
meta_model = LinearRegression()
meta_X = np.column_stack((best_xgb_model.predict(X_train), lgb_model.predict(X_train, num_iteration=lgb_model.best_iteration)))
meta_model.fit(meta_X, y_train)

# Making final predictions using the stacked model
meta_predictions = meta_model.predict(np.column_stack((xgb_predictions, lgb_predictions)))
meta_predictions_original = y_transformer.inverse_transform(meta_predictions.reshape(-1, 1)).flatten()

# Creating and saving the submission file
original_test = pd.read_csv('SampleSubmission.csv')
original_test['number_of_ticket'] = meta_predictions_original
submission = original_test[['ride_id', 'number_of_ticket']]
submission.to_csv('stacked_submission.csv', index=False)

print("Prediction completed and saved as 'stacked_submission.csv'")

# Function for plotting the learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Cross-validation score")
    plt.legend(loc="best")
    plt.show()

# Plotting the learning curve for the best XGBoost model
plot_learning_curve(best_xgb_model, "Learning Curve (XGBoost)", X, y_transformed, cv=3)
