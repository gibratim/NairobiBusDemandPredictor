# Code Overview and Methodology for Transportation Prediction

## Challenge 

Nairobi is one of the most heavily congested cities in Africa. Each day thousands of Kenyans make the trip into Nairobi from towns such as Kisii, Keroka, and beyond for work, business, or to visit friends and family. The journey can be long, and the final approach into the city can impact the length of the trip significantly depending on traffic. How do traffic patterns influence people’s decisions to come into the city by bus and which bus to take? Does knowing the traffic patterns in Nairobi help anticipate the demand for particular routes at particular times?

Create a predictive model using traffic data provided from Uber Movement and historic bus ticket sales data from Mobiticket to predict the number of tickets that will be sold for buses into Nairobi from cities in "up country" Kenya.

The data used to train the model will be historic hourly traffic patterns in Nairobi and historic ticket purchasing data for 14 bus routes into Nairobi from October 2017 to April 2018, and includes the place or origin, the scheduled time of departure, the channel used for the purchase, the type of vehicle, the capacity of the vehicle, and the assigned seat number. 

This resulting model can be used to anticipate customer demand for certain rides, to manage resources and vehicles more efficiently, to offer promotions and sell other services more effectively, such as micro-insurance, or even improve customer service by being able to send alerts and other useful information to customers.

## Summary

The code aims to predict transportation-related variables, such as the number of tickets sold for each ride. The code uses Python and several machine learning libraries like scikit-learn, XGBoost, and LightGBM. The pipeline includes data preprocessing, feature engineering, hyperparameter tuning, model training, and ensembling to produce the most accurate predictions. 

## Data Preprocessing and Feature Engineering

The datasets undergoe several preprocessing steps. Initially, the training data is aggregated based on specific columns like 'ride_id', 'travel_date', 'travel_time', 'travel_from', 'travel_to', 'car_type', and 'max_capacity'. This helps in reducing redundancy and makes the dataset easier to manage. 

## New Features
-**Day of the Week**: Added to capture weekly patterns, which could affect ticket sales.

-**Is Weekend**: Created to distinguish between weekdays and weekends. Weekends could see different travel patterns.

-**Peak Hour**: Identified using the 'travel_time' feature to categorize busy travel hours. Peak hours often experience increased demand.

These are created to help the model understand patterns relating to the day of the week or weekends. The 'travel_time' is converted into a continuous numerical variable representing the hour of the day to capture patterns related to time. A 'peak_hour' feature is also created to categorize the busy hours of travel. 

## Label Encoding
Categorical variables like 'travel_from', 'car_type', and 'travel_to' are label-encoded to convert them into numerical form. 

## Model Selection and Hyperparameter Tuning

Two models, XGBoost and LightGBM, are selected for this task. Both models are known for their high performance on tabular data and are widely used. 

## Hyperparameter Tuning
- **XGBoost**: RandomizedSearchCV was used to tune parameters like 'learning_rate', 'max_depth', and 'n_estimators'. RandomizedSearchCV is computationally more efficient than GridSearchCV and provides a good trade-off between performance and computational time.

- **LightGBM**: A custom loop using ParameterGrid was used to iterate through various combinations of hyperparameters like 'num_leaves', 'learning_rate', and 'lambda_l1'.

## Ensembling and Stacking

To improve the model's accuracy, the predictions from both XGBoost and LightGBM are stacked using a meta-model (Linear Regression). This technique captures the strengths of both models, producing a more robust prediction.

## Model Evaluation

The models' performances are evaluated using Mean Squared Error (MSE) as the metric, which is standard for regression problems. The final predictions are inverse-transformed to bring them back to the original scale and are then saved in a CSV file for submission.

## Findings and Conclusions

- **Feature Engineering**: The newly created features like 'day_of_week', 'is_weekend', and 'peak_hour' contributed to capturing more complex relationships in the data.
  
- **Model Selection**: XGBoost and LightGBM are both gradient boosting algorithms that perform well on a variety of datasets. They handle missing values, categorical features, and irrelevant variables efficiently.
  
- **Hyperparameter Tuning**: Instead of using default parameters, which may not be optimal, RandomizedSearchCV and ParameterGrid were used to find the best parameters, improving the model's performance.
  
- **Ensembling**: The stacking method successfully combined the predictive power of two high-performing models, likely resulting in a more accurate prediction.


# Instructions to Run NairobiBusDemandPredictor Code

## Prerequisites
Ensure you have Python 3.x installed on your machine. If not, you can download it from (https://www.python.org/downloads/).

## Libraries Needed
The following Python libraries are required to run the code:
- Pandas
- NumPy
- scikit-learn
- XGBoost
- LightGBM
- Matplotlib

## Installation

You can install these libraries using `pip`. Open your terminal and run:

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib
```

Or, if you are using a `requirements.txt` file, navigate to the directory where the `requirements.txt` is located and run:

```bash
pip install -r requirements.txt
```

## Download the Code

1. Clone the GitHub repository to your local machine:

    ```bash
    git clone https://github.com/your_username/NairobiBusDemandPredictor.git
    ```
  
2. Navigate to the cloned directory:

    ```bash
    cd NairobiBusDemandPredictor
    ```

## Running the Code

1. Open the terminal and navigate to the directory containing the code file (e.g., `final.py`).

2. Run the code:

    ```bash
    python final.py
    ```

    Or, if you are using a Jupyter Notebook (e.g., `final.ipynb`), launch Jupyter Notebook by running:

    ```bash
    jupyter notebook
    ```

    Then, navigate to the notebook file and run all cells.

## Output
Once the code is successfully executed, you should see predictions for the number of bus tickets that will be sold for each route. A CSV file named `stacked_submission.csv` will also be generated, containing the predictions.
