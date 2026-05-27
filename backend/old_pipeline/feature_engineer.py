import numpy as np
import pandas as pd


def load_and_engineer_data(csv_path, company_name="UNKNOWN"):
    df = pd.read_csv(csv_path)

    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    for col in df.columns:
        if "date" in col.lower():
            date_col = col
            break

    if date_col is None:
        raise ValueError("No date column found.")

    close_col = None
    for col in df.columns:
        if "close" in col.lower():
            close_col = col
            break

    if close_col is None:
        raise ValueError("No close column found.")

    df = df[[date_col, close_col]].copy()
    df.columns = ["Date", "Close"]

    df["Date"] = pd.to_datetime(
        df["Date"],
        errors="coerce"
    )

    df["Close"] = pd.to_numeric(
        df["Close"],
        errors="coerce"
    )

    df = df.dropna(subset=["Date", "Close"]).copy()
    df = df.sort_values("Date").reset_index(drop=True)

    df["Company"] = company_name

    # Intraday session because data is collected around 5 times per day
    df["session"] = np.arange(len(df)) % 5

    # Lag features
    for lag in [1, 2, 3, 5, 10]:
        df[f"lag_{lag}"] = df["Close"].shift(lag)

    # Returns
    for n in [1, 2, 3, 5]:
        df[f"return_{n}"] = df["Close"].pct_change(n)

    # Log returns
    for n in [1, 2, 3]:
        df[f"log_return_{n}"] = np.log(
            df["Close"] / df["Close"].shift(n)
        )

    # Momentum
    for n in [3, 5, 10]:
        df[f"momentum_{n}"] = df["Close"] - df["Close"].shift(n)

    # Velocity and acceleration
    df["velocity"] = df["Close"] - df["Close"].shift(1)
    df["acceleration"] = df["velocity"] - df["velocity"].shift(1)

    # Shorter rolling windows to preserve smaller company datasets
    for w in [3, 5, 10]:
        df[f"sma_{w}"] = df["Close"].rolling(w, min_periods=1).mean()
        df[f"ema_{w}"] = df["Close"].ewm(span=w, adjust=False).mean()

    df["dist_sma_5"] = df["Close"] - df["sma_5"]
    df["dist_ema_5"] = df["Close"] - df["ema_5"]

    # Volatility
    for w in [3, 5, 10]:
        df[f"volatility_{w}"] = (
            df["return_1"]
            .rolling(w, min_periods=2)
            .std()
        )

    # Rolling min/max
    for w in [3, 5, 10]:
        df[f"roll_min_{w}"] = df["Close"].rolling(w, min_periods=1).min()
        df[f"roll_max_{w}"] = df["Close"].rolling(w, min_periods=1).max()

    # Range position
    for w in [3, 5, 10]:
        min_col = f"roll_min_{w}"
        max_col = f"roll_max_{w}"

        denominator = (df[max_col] - df[min_col]).replace(0, np.nan)

        df[f"range_position_{w}"] = (
            (df["Close"] - df[min_col]) / denominator
        )

    # Z-score
    for w in [3, 5, 10]:
        rolling_mean = df["Close"].rolling(w, min_periods=2).mean()
        rolling_std = df["Close"].rolling(w, min_periods=2).std()

        df[f"zscore_{w}"] = (
            (df["Close"] - rolling_mean) / rolling_std
        )

    # =========================
    # RSI 7
    # =========================
    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(7, min_periods=2).mean()
    avg_loss = loss.rolling(7, min_periods=2).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_7"] = 100 - (100 / (1 + rs))

    # Fix missing RSI values safely
    df.loc[df["rsi_7"].isna() & (avg_gain > 0), "rsi_7"] = 100
    df.loc[df["rsi_7"].isna() & (avg_gain <= 0), "rsi_7"] = 50
    df["rsi_7"] = df["rsi_7"].fillna(50)

    # MACD
    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands using 10-window instead of 20-window
    bb_window = 10

    rolling_mean = df["Close"].rolling(bb_window, min_periods=2).mean()
    rolling_std = df["Close"].rolling(bb_window, min_periods=2).std()

    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std

    bb_denominator = (df["bb_upper"] - df["bb_lower"]).replace(0, np.nan)

    df["bb_position"] = (
        (df["Close"] - df["bb_lower"]) / bb_denominator
    )

    # Date features
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["month"] = df["Date"].dt.month

    # Cyclical encoding
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

    # Market regime
    volatility_threshold = df["volatility_10"].median()

    df["market_regime"] = np.where(
        df["volatility_10"] > volatility_threshold,
        1,
        0
    )

    # Future targets
    df["future_close_1"] = df["Close"].shift(-1)
    df["future_close_5"] = df["Close"].shift(-5)
    df["future_close_10"] = df["Close"].shift(-10)

    future_diff = df["future_close_1"] - df["Close"]

    def make_label(x):
        if pd.isna(x):
            return np.nan
        if x > 0:
            return "UP"
        if x < 0:
            return "DOWN"
        return "SAME"

    df["target"] = future_diff.apply(make_label)

    target_map = {
        "DOWN": 0,
        "SAME": 1,
        "UP": 2
    }

    df["target_class"] = df["target"].map(target_map)

    # Clean infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Fill non-critical feature NaNs instead of deleting many rows
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        if col not in [
            "future_close_1",
            "future_close_5",
            "future_close_10",
            "target_class"
        ]:
            df[col] = df[col].ffill().bfill()

    required_cols = [
        "Date",
        "Close",
        "return_1",
        "sma_5",
        "ema_5",
        "future_close_1",
        "future_close_5",
        "future_close_10",
        "target",
        "target_class"
    ]

    df = df.dropna(subset=required_cols).reset_index(drop=True)

    return df