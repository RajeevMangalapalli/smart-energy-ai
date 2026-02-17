#ingestion.py
import pandas as pd
from pathlib import Path

FILE_PATH = Path("smart-energy-ai/data/raw/time_series_60min_singleindex.csv")
OUTPUT_PATH = Path("smart-energy-ai/data/processed/DE_load_daily.csv")


def load_data(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df


def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    df_selected = df[["utc_timestamp", "DE_load_actual_entsoe_transparency"]]

    #Rename the columns
    df_selected = df_selected.rename(columns={
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load_mw"
    })
    
    #Convert timestamp to datetime format
    df_selected["timestamp"] = pd.to_datetime(df_selected["timestamp"])
    
    # Set timestamp as index
    df_selected.set_index("timestamp", inplace=True)

    return df_selected.sort_index()

def preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df["load_mw"] = df["load_mw"].interpolate(method="time")

    #Resample to daily
    df_daily = df.resample("D").mean()
    return df_daily


if __name__ == "__main__":
    df = load_data(FILE_PATH)
    df_selected = feature_selection(df)
    df_daily = preprocessing(df_selected)
    df_daily.to_csv(OUTPUT_PATH)
    

    print(f"Data has been processed and saved to {OUTPUT_PATH}")



