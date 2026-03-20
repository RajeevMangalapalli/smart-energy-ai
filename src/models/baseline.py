#ARIMA as the baseline model for forecasting
from pathlib import Path
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

#importing the data
FILE_PATH = Path("smart-energy-ai/data/processed/DE_load_daily_features_time_features.csv")
df = pd.read_csv(FILE_PATH, parse_dates=["timestamp"], index_col="timestamp")

#Splitting the data into train and test sets
train = df[df.index < "2020-01-01"]
test = df[df.index >= "2020-01-01"]

#Fitting the ARIMA model

model = ARIMA(train["load"], order=(5, 1, 0)) #Training the ARIMA model
model_fit = model.fit() # Fitting the model to the training data

#Making predictions
predictions = model_fit.forecast(steps=len(test)) # Forecasting the load for the test period

#Evaluating the model(MAE, RMSE)
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
mae = mean_absolute_error(test["load"], predictions)
rmse = root_mean_squared_error(test["load"], predictions, squared=False)
print(f"Mean Absolute Error: {mae}")
print(f"Root Mean Squared Error: {rmse}")


