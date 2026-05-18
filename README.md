# Implementation of Random Forest Algorithm for Weather Prediction
## AIM:
To write a program to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data using Random Forest Algorithm.

## Problem Statement and Dataset
The objective of this experiment is to develop a Random Forest Regression Model to predict:
Temperature
PM2.5 pollution level
Energy generation
using environmental sensor data collected from a weather monitoring station.

The model uses input features such as:

Humidity
Atmospheric Pressure
Wind Speed
Date and Time information

The performance of the model is evaluated using:

R² Score
Mean Absolute Error (MAE)
Root Mean Squared Error (RMSE)
Cross-validation is also performed to ensure the reliability and stability of the prediction model.

# Dataset
## Dataset Name
weather-station-eee-block_2024_07_13.csv
## Dataset Description

The dataset contains environmental sensor readings collected from a weather station.

## Important Attributes Used
Attribute|	Description
time     |	Date and time of sensor reading
hum	     | Humidity
pressure | Atmospheric pressure
wind_speed |	Wind speed
tem      |	Temperature
pm2_5	PM2.5| pollution level
tsr	Energy | solar radiation value
## Input Features
Humidity (hum)
Pressure (pressure)
Wind Speed (wind_speed)
Day
Month
Year
Hour
## Target Variables
Temperature (tem)
PM2.5 (pm2_5)
Energy (tsr)


## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the dataset, check for missing values, and extract useful date information such as day, month, year, and hour.
2. Select humidity, pressure, wind speed, and date-related columns as input features and temperature, PM2.5, and energy as target variables.
3. Create the Random Forest Regressor model and train it using the training dataset.
4. Predict output values and calculate R² score, Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and perform cross-validation.

## Program:
```
/*
Program to implement the Random Forest Algorithm to predict daily temperature , PM2.5 pollution level and Energy based on environmental sensor data.
Developed by: 
RegisterNumber:  
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split,
    cross_val_score
)
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
data = pd.read_csv("weather-station-eee-block_2024_07_13.csv")
print("FIRST 5 ROWS OF DATASET:\n")
print(data.head())
print("\nMISSING VALUES:\n")
print(data.isnull().sum())
data["time"] = pd.to_datetime(data["time"])
data["day"] = data["time"].dt.day
data["month"] = data["time"].dt.month
data["year"] = data["time"].dt.year
data["hour"] = data["time"].dt.hour
X = data[[
    "hum",
    "pressure",
    "wind_speed",
    "day",
    "month",
    "year",
    "hour"
]]
y = data[[
    "tem",
    "pm2_5",
    "tsr"
]]
X = X.fillna(X.mean())
y = y.fillna(y.mean())
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\nTRAINING DATA SIZE:", X_train.shape)
print("TESTING DATA SIZE:", X_test.shape)
rf = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

print("\nMODEL TRAINED SUCCESSFULLY")
y_pred = rf.predict(X_test)

print("\nFIRST 5 PREDICTED VALUES:\n")
print(y_pred[:5])
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print("\n==============================")
print("MODEL EVALUATION METRICS")
print("==============================")
print("\nR2 SCORE:")
print(r2)
print("\nMEAN ABSOLUTE ERROR (MAE):")
print(mae)
print("\nROOT MEAN SQUARED ERROR (RMSE):")
print(rmse)
cv_scores = cross_val_score(
    rf,
    X,
    y,
    cv=5,
    scoring='r2'
)
print("\n==============================")
print("CROSS VALIDATION")
print("==============================")
print("\nCross Validation Scores:")
print(cv_scores)
print("\nAverage Cross Validation Score:")
print(cv_scores.mean())
print("\n==============================")
print("RESULT ANALYSIS")
print("==============================")
if r2 > 0.8:
    print("\nThe model shows HIGH prediction accuracy.")
elif r2 > 0.5:
    print("\nThe model shows MODERATE prediction accuracy.")
else:
    print("\nThe model shows LOW prediction accuracy.")

print("\nLower MAE and RMSE values indicate better prediction performance.")

print("\nCross-validation scores close to each other indicate that the model is stable and reliable.")
```

## Output:
<img width="1110" height="694" alt="image" src="https://github.com/user-attachments/assets/38eda52b-b218-4237-b4bc-2c101b5bd9fa" />
<img width="1089" height="691" alt="image" src="https://github.com/user-attachments/assets/0b053823-e1f9-4b15-aa66-a4b629379c4d" />



## Result:
The Random Forest Regression model was successfully trained using environmental sensor data. The model predicted Temperature,PM2.5 pollution level,Energy values with good accuracy.
