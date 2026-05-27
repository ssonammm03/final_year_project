import os
import pandas as pd
import numpy as np


def extract_company_name_from_filename(filename):
    base = os.path.basename(filename)
    name_without_ext = os.path.splitext(base)[0]
    return name_without_ext.split("_")[0].upper()


def normalize_columns(df):
    df.columns = [str(c).strip() for c in df.columns]
    return df


def find_date_column(df):
    candidates = [c for c in df.columns if "date" in c.lower()]
    if not candidates:
        raise ValueError(f"No date column found. Available columns: {list(df.columns)}")
    return candidates[0]


def find_close_column(df):
    candidates = [c for c in df.columns if "close" in c.lower()]
    if not candidates:
        raise ValueError(f"No close column found. Available columns: {list(df.columns)}")
    return candidates[0]


def process_raw_dataframe(df, company_name="UPLOADED"):
    df = normalize_columns(df)

    date_col = find_date_column(df)
    close_col = find_close_column(df)

    df = df[[date_col, close_col]].copy()
    df = df.rename(columns={date_col: "Date", close_col: "Close"})

    df["Date"] = pd.to_datetime(df["Date"], dayfirst=True, errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    # Since raw file has no time column, keep one daily close
    df["Company"] = company_name
    df = (
        df.groupby(["Company", "Date"], as_index=False)
        .agg({"Close": "last"})
        .sort_values(["Company", "Date"])
        .reset_index(drop=True)
    )

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Differences
    df["diff_1"] = df["Close"].diff(1)
    df["diff_2"] = df["Close"].diff(2)
    df["diff_3"] = df["Close"].diff(3)

    # Returns
    df["return_1"] = df["Close"].pct_change(1)
    df["return_2"] = df["Close"].pct_change(2)
    df["return_3"] = df["Close"].pct_change(3)

    # Log returns
    df["log_return_1"] = np.log(df["Close"] / df["Close"].shift(1))
    df["log_return_2"] = np.log(df["Close"] / df["Close"].shift(2))

    # Moving averages
    for window in [3, 5, 10]:
        df[f"sma_{window}"] = df["Close"].rolling(window).mean()
        df[f"ema_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()

    # Volatility
    for window in [3, 5, 10]:
        df[f"volatility_{window}"] = df["return_1"].rolling(window).std()

    # Momentum and ROC
    for n in [3, 5, 10]:
        df[f"momentum_{n}"] = df["Close"] - df["Close"].shift(n)
        df[f"roc_{n}"] = (df["Close"] - df["Close"].shift(n)) / df["Close"].shift(n)

    # Rolling min/max
    for window in [3, 5, 10]:
        df[f"roll_min_{window}"] = df["Close"].rolling(window).min()
        df[f"roll_max_{window}"] = df["Close"].rolling(window).max()

    # Range position
    for window in [3, 5, 10]:
        min_col = f"roll_min_{window}"
        max_col = f"roll_max_{window}"
        df[f"range_position_{window}"] = (
            (df["Close"] - df[min_col]) /
            (df[max_col] - df[min_col]).replace(0, np.nan)
        )

    # Date features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day

    # Target
    df["future_close"] = df["Close"].shift(-1)
    df["future_diff"] = df["future_close"] - df["Close"]

    def make_label(x):
        if pd.isna(x):
            return np.nan
        if x > 0:
            return "UP"
        if x < 0:
            return "DOWN"
        return "SAME"

    df["target"] = df["future_diff"].apply(make_label)
    target_map = {"DOWN": 0, "SAME": 1, "UP": 2}
    df["target_class"] = df["target"].map(target_map)

    return df