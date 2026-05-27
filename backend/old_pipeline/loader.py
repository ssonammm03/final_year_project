import os
import numpy as np
import pandas as pd


def get_company_files(data_folder: str):
    files = {}

    if not os.path.exists(data_folder):
        return files

    for file in os.listdir(data_folder):
        if file.endswith("_daily_featured.csv"):
            company = file.split("_")[0].upper()
            files[company] = os.path.join(data_folder, file)

    return dict(sorted(files.items()))


def load_processed_data(path: str):
    df = pd.read_csv(path)

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.sort_values("Date").reset_index(drop=True)

    return df


def latest_value(df, col):
    if col not in df.columns:
        return None

    vals = df[col].replace([np.inf, -np.inf], np.nan).dropna()

    if vals.empty:
        return None

    value = vals.iloc[-1]

    if pd.isna(value):
        return None

    return value


def clean_value(value):
    if pd.isna(value):
        return None

    if isinstance(value, float):
        if np.isnan(value) or np.isinf(value):
            return None

    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y-%m-%d")

    return value


def dataframe_to_records(df):
    df = df.copy()

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(
            df["Date"],
            errors="coerce"
        ).dt.strftime("%Y-%m-%d")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.astype(object).where(pd.notnull(df), None)

    records = df.to_dict(orient="records")

    cleaned_records = []
    for row in records:
        cleaned_row = {}
        for key, value in row.items():
            cleaned_row[key] = clean_value(value)
        cleaned_records.append(cleaned_row)

    return cleaned_records