import pandas as pd
import numpy as np
from pathlib import Path
from xgboost import XGBRegressor
from datetime import timedelta
import sys

sys.path.append(str(Path(__file__).parent))
from retrieve import ask

DATA_PATH    = Path("smart-energy-ai/data/processed/DE_load_daily_features_time_features.csv")
METRICS_PATH = Path("smart-energy-ai/data/processed/xgb_metrics.json")

FEATURES = ["lag_1", "lag_7", "rolling_mean_7", "rolling_mean_30",
            "day_of_week", "month", "is_weekend"]
TARGET = "load_mw"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.set_index("timestamp")
    df = df.resample("D").mean()
    df = df.ffill()
    df["day_of_week"] = df.index.dayofweek
    df["month"]       = df.index.month
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)
    df = df.dropna(subset=FEATURES + [TARGET])
    return df


def train_model(df: pd.DataFrame) -> XGBRegressor:
    model = XGBRegressor(n_estimators=100, max_depth=5, random_state=42)
    model.fit(df[FEATURES], df[TARGET])
    print(f"Model trained on {len(df)} days of data")
    return model


def forecast_next_n_days(model: XGBRegressor, df: pd.DataFrame, n: int = 7) -> pd.DataFrame:
    recent_values = list(df[TARGET].values)
    last_date     = df.index[-1]
    forecasts     = []

    for step in range(n):
        next_date        = last_date + timedelta(days=step + 1)
        lag_1            = recent_values[-1]
        lag_7            = recent_values[-7]
        rolling_mean_7   = np.mean(recent_values[-7:])
        rolling_mean_30  = np.mean(recent_values[-30:])
        day_of_week      = next_date.dayofweek
        month            = next_date.month
        is_weekend       = int(next_date.dayofweek >= 5)

        row = pd.DataFrame([{
            "lag_1"          : lag_1,
            "lag_7"          : lag_7,
            "rolling_mean_7" : rolling_mean_7,
            "rolling_mean_30": rolling_mean_30,
            "day_of_week"    : day_of_week,
            "month"          : month,
            "is_weekend"     : is_weekend
        }])

        prediction = model.predict(row[FEATURES])[0]
        forecasts.append({
            "date"       : next_date.strftime("%Y-%m-%d"),
            "day"        : next_date.strftime("%A"),
            "forecast_mw": round(float(prediction), 1)
        })
        recent_values.append(prediction)

    return pd.DataFrame(forecasts)


def format_forecast_summary(forecast_df: pd.DataFrame) -> str:
    lines = ["Forecast for the next 7 days (German electricity demand):"]
    for _, row in forecast_df.iterrows():
        lines.append(f"  - {row['day']} {row['date']}: {row['forecast_mw']:,.0f} MW")

    avg  = forecast_df["forecast_mw"].mean()
    high = forecast_df.loc[forecast_df["forecast_mw"].idxmax()]
    low  = forecast_df.loc[forecast_df["forecast_mw"].idxmin()]

    lines.append(f"\nWeekly average : {avg:,.0f} MW")
    lines.append(f"Peak day       : {high['day']} at {high['forecast_mw']:,.0f} MW")
    lines.append(f"Lowest day     : {low['day']} at {low['forecast_mw']:,.0f} MW")
    return "\n".join(lines)


def run_pipeline(user_question: str = None, n_days: int = 7):
    print("\n" + "═" * 55)
    print(" SMART ENERGY AI — FORECAST + EXPLANATION PIPELINE")
    print("═" * 55)

    df          = load_data()
    model       = train_model(df)
    forecast_df = forecast_next_n_days(model, df, n=n_days)
    summary     = format_forecast_summary(forecast_df)

    print(f"\n{summary}")
    print("\n" + "─" * 55)
    print("EXPLANATION FROM KNOWLEDGE BASE")
    print("─" * 55)

    if not user_question:
        user_question = "Why does electricity demand vary across days of the week?"

    result = ask(query=user_question, context=summary)

    print("\n" + "═" * 55)
    print("FINAL REPORT")
    print("═" * 55)
    print(f"\nForecast:\n{summary}")
    print(f"\nExplanation:\n{result['answer']}")
    print(f"\nSources: {', '.join(result['sources']) if result['sources'] else 'None'}")

    return {
        "forecast" : forecast_df.to_dict(orient="records"),
        "summary"  : summary,
        "answer"   : result["answer"],
        "sources"  : result["sources"]
    }


if __name__ == "__main__":
    question = input("Enter your question about the forecast: ").strip()
    if not question:
        question = None
    run_pipeline(user_question=question)