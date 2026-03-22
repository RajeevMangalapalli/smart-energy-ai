import pandas as pd
import json
import joblib
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from pathlib import Path

FILE_PATH = Path("smart-energy-ai/data/processed/DE_load_daily_features_time_features.csv")
df = pd.read_csv(FILE_PATH, parse_dates=["timestamp"])
df = df.set_index("timestamp")
df.index = pd.DatetimeIndex(df.index.values, freq = "D")
df.dropna(inplace=True)


# Split the data into train and test sets
train = df[df.index < "2020-01-01"]
test = df[df.index >= "2020-01-01"]

train_load = train["load_mw"]
test_load = test["load_mw"]

# Fit the ARIMA model
model = ARIMA(train_load, order=(0,0,3), seasonal_order=(0,0,2,7))  # (p, d, q) parameters taken from the eda notebook
model_fit = model.fit()
print(model_fit.summary())


# Forecast the test set
forecast = model_fit.forecast(steps=len(test_load))

# Evaluate the model
mse = mean_squared_error(test_load, forecast)
rmse = root_mean_squared_error(test_load, forecast)

"""
                                      SARIMAX Results
============================================================================================
Dep. Variable:                              load_mw   No. Observations:                 1796
Model:             ARIMA(0, 0, 3)x(0, 0, [1, 2], 7)   Log Likelihood              -17217.186
Date:                              Sun, 22 Mar 2026   AIC                          34448.371
Time:                                      22:35:16   BIC                          34486.825
Sample:                                  01-31-2015   HQIC                         34462.568
                                       - 12-31-2019
Covariance Type:                                opg
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.584e+04    384.752    145.134      0.000    5.51e+04    5.66e+04
ma.L1          0.6737      0.023     29.754      0.000       0.629       0.718
ma.L2          0.2162      0.025      8.653      0.000       0.167       0.265
ma.L3          0.1087      0.023      4.720      0.000       0.064       0.154
ma.S.L7        0.6766      0.016     42.516      0.000       0.645       0.708
ma.S.L14       0.4111      0.018     22.992      0.000       0.376       0.446
sigma2      1.239e+07      0.062   1.99e+08      0.000    1.24e+07    1.24e+07
===================================================================================
Ljung-Box (L1) (Q):                   0.06   Jarque-Bera (JB):               224.02
Prob(Q):                              0.81   Prob(JB):                         0.00
Heteroskedasticity (H):               1.07   Skew:                             0.05
Prob(H) (two-sided):                  0.42   Kurtosis:                         4.73
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
[2] Covariance matrix is singular or near-singular, with condition number 1.12e+24. Standard errors may be unstable.
ARIMA Model - MSE: 52545114.64, RMSE: 7248.80

"""


# Baseline results
print(f"Mean load      : {train_load.mean():.0f} MW") #55842 MW
print(f"RMSE           : 7248 MW") 
print(f"RMSE % of mean : {7248 / train_load.mean() * 100:.1f}%") #13.0%


#Saving the model predictions
forecast = pd.DataFrame({
    "timestamp": test.index,
    "actual": test_load.values,
    "sARIMA_predicted": forecast.values
})

forecast.to_csv("smart-energy-ai/data/processed/sARIMA_predictions.csv", index = False)

#Saving the metrics
metrics = {
    "model" : "SARIMA (0,0,3)(0,0,2,7)",
    "rmse" : round(rmse,2),
    "mse" : round(mse,2),
    "rmse_pct_of_mean": rmse/train_load.mean() * 100
}

with open("smart-energy-ai/data/processed/sarima_metrics.json","w") as f:
    json.dump(metrics, f, indent = 2)


joblib.dump(model, "smart-energy-ai/src/data/s-arima.joblib")



