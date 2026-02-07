import pandas as pd
FILE_PATH = r"C:\Users\rajee\Desktop\VS code\smart-energy-ai\data\raw\time_series_60min_singleindex.csv"


def load_data(file_path : str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    return df

df = load_data(FILE_PATH)


# Function to select timestamp and DE load columns
"""
The following features are selected (All Deutschland related features):

DE_load_actual_entsoe_transparency
timestamp

"""
def feature_selection(df: pd.DataFrame) -> pd.DataFrame:
    df_selected = df[["utc_timestamp", "DE_load_actual_entsoe_transparency"]]

    #Rename the columns
    df_selected = df_selected.rename(columns={
        "utc_timestamp": "timestamp",
        "DE_load_actual_entsoe_transparency": "load_mw"
    })

    return df_selected

df_selected = feature_selection(df)
print(df_selected.head())
