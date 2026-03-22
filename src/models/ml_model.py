import pandas as pd
import numpy as np
import json
from xgboost import XGBRegressor
import joblib
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from pathlib import Path

FILE_PATH = Path("smart-energy-ai/data/processed/DE_load_daily_features_time_features.csv")
df = pd.read_csv(FILE_PATH, parse_dates=["timestamp"])
df = df.set_index("timestamp")
df.index = pd.DatetimeIndex(df.index.values, freq="D")


df["day_of_week"] = df.index.dayofweek
df["month"]       = df.index.month
df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)

# Features and target
features = ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30",
            "day_of_week", "month", "is_weekend"]
target = "load_mw"

df = df.dropna(subset=features + [target])

# Split
train = df[df.index < "2020-01-01"]
test  = df[df.index >= "2020-01-01"]

# Train
model = XGBRegressor(n_estimators=500, max_depth=5, random_state=42)
model.fit(train[features], train[target])

# Predictions
predictions = model.predict(test[features])

# Evaluation
mse  = mean_squared_error(test[target], predictions)
rmse = root_mean_squared_error(test[target], predictions)
rmse_pct = rmse / test[target].mean() * 100

print(f"MSE            : {mse:.2f}")
print(f"RMSE           : {rmse:.2f} MW")
print(f"RMSE % of mean : {rmse_pct:.1f}%")

# Save predictions
pred_df = pd.DataFrame({
    "timestamp"      : test.index,
    "actual"         : test[target].values,
    "xgb_predicted"  : predictions
})
pred_df.to_csv("smart-energy-ai/data/processed/xgb_predictions.csv", index=False)

# Save metrics
metrics = {
    "model"          : "XGBoost",
    "features"       : features,
    "rmse"           : round(rmse, 2),
    "mse"            : round(mse, 2),
    "rmse_pct_mean"  : round(rmse_pct, 1)
}
with open("smart-energy-ai/data/processed/xgb_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

print("XGBoost predictions and metrics saved.")

joblib.dump(model, Path("smart-energy-ai/src/data/xgboost.joblib"))
