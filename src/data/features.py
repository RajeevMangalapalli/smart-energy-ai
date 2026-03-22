# File to add the features lags and rolling means to the data
"""
timestamp | load_mw | lag_1 | lag_7 | rolling_mean_7 | rolling_mean_30

"""
import pandas as pd
from ingestion import load_data
from pathlib import Path
FILE_PATH = Path("smart-energy-ai/data/processed/DE_load_daily.csv")
OUTPUT_PATH = Path("smart-energy-ai/data/processed/DE_load_daily_features_time_features.csv")
df = load_data(FILE_PATH)
df = df.dropna().reset_index(drop=True)

def create_time_features(df: pd.DataFrame)-> pd.DataFrame:
    #Lag features
    df["lag_1"] = df["load_mw"].shift(1)
    df["lag_7"] = df["load_mw"].shift(7)

    #Rolling means
    df["rolling_mean_7"] = df["load_mw"].shift(1).rolling(window=7).mean()
    df["rolling_mean_30"] = df["load_mw"].shift(1).rolling(window = 30).mean()

    return df

create_time_features(df).to_csv(OUTPUT_PATH, index=False)


