#Evaluation file to compare both the models and decide which one to use
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path

BASEPATH = Path("smart-energy-ai/data/processed")

df = pd.read_csv("smart-energy-ai/data/processed/DE_load_daily_features.csv", parse_dates=["timestamp"])
sarima_df = pd.read_csv("smart-energy-ai/data/processed/sARIMA_predictions.csv", parse_dates=["timestamp"])
xgb_df = pd.read_csv("smart-energy-ai/data/processed/xgb_predictions.csv", parse_dates=["timestamp"])


with open(BASEPATH / "sarima_metrics.json") as f:
    sarima_metrics = json.load(f)
with open(BASEPATH / "xgb_metrics.json") as f:
    xgb_metrics = json.load(f)

df = pd.merge(sarima_df, xgb_df, on="timestamp", suffixes=("_sarima", "_xgb"))
df = df.sort_values("timestamp")



#Forecast comparision plot
fig, axes = plt.subplots(2,1,figsize=(16,10))

#Top plot
axes[0].plot(df["timestamp"], df["actual_sarima"], label="Actual",
             color="black", linewidth=1.2)
axes[0].plot(df["timestamp"], df["sarima_predicted"], label=f"SARIMA  RMSE%={sarima_metrics['rmse_pct_of_mean']}%",
             color="steelblue", linestyle="--", linewidth=1)
axes[0].plot(df["timestamp"], df["xgb_predicted"], label=f"XGBoost RMSE%={xgb_metrics['rmse_pct_mean']}%",
             color="darkorange", linestyle="--", linewidth=1)
axes[0].set_title("Forecast Comparison — SARIMA vs XGBoost vs Actual")
axes[0].set_ylabel("Load (MW)")
axes[0].legend()
axes[0].grid(True, alpha=0.3)

#Bottom plot
sarima_error = df["actual_sarima"] - df["sarima_predicted"]
xgb_error    = df["actual_sarima"] - df["xgb_predicted"]

axes[1].plot(df["timestamp"], sarima_error,  label="SARIMA error",
             color="steelblue", linewidth=0.8, alpha=0.8)
axes[1].plot(df["timestamp"], xgb_error, label="XGBoost error",
             color="darkorange", linewidth=0.8, alpha=0.8)
axes[1].axhline(y=0, color="black", linewidth=0.8, linestyle="-")
axes[1].set_title("Residuals (Actual - Predicted)")
axes[1].set_ylabel("Error (MW)")
axes[1].set_xlabel("Date")
axes[1].legend()
axes[1].grid(True, alpha=0.3)


#Metrics comparinison
plt.tight_layout()
plt.savefig(BASEPATH / "forecast_comparison.png", dpi=150)
plt.show()

fig, ax = plt.subplots(figsize=(7, 2))
ax.axis("off")

table_data = [
    ["Model",                    "RMSE (MW)",                      "RMSE % of Mean"],
    ["SARIMA(0,0,3)(0,0,2,7)",   f"{sarima_metrics['rmse']:.0f}",  f"{sarima_metrics['rmse_pct_of_mean']}%"],
    ["XGBoost",                  f"{xgb_metrics['rmse']:.0f}",     f"{xgb_metrics['rmse_pct_mean']}%"],
]

table = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                 loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.4, 2)

# Highlight XGBoost row in light green
for col in range(3):
    table[2, col].set_facecolor("#d4edda")

plt.title("Model Evaluation Summary", fontsize=13, pad=20)
plt.tight_layout()
plt.savefig(BASEPATH / "metrics_table.png", dpi=150)
plt.show()




######
print("\n── Evaluation Summary ────────────────────────────────────────")
print(f"{'Model':<30} {'RMSE':>10} {'RMSE %':>10}")
print(f"{'SARIMA(0,0,3)(0,0,2,7)':<30} {sarima_metrics['rmse']:>10.2f} {sarima_metrics['rmse_pct_of_mean']:>9}%")
print(f"{'XGBoost':<30} {xgb_metrics['rmse']:>10.2f} {xgb_metrics['rmse_pct_mean']:>9}%")
improvement = (1 - xgb_metrics["rmse_pct_mean"] / sarima_metrics["rmse_pct_of_mean"]) * 100
print(f"\nXGBoost improved over SARIMA by {improvement:.1f}%")