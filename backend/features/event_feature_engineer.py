import os
import numpy as np
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVENT_FILE = os.path.join(BASE_DIR, "data", "events", "dividend_events.csv")

SENTIMENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "events",
    "announcement_sentiment.csv"
)


def clean_stock_data(df):
    df.columns = [str(c).strip() for c in df.columns]

    date_col = None
    close_col = None

    for col in df.columns:
        if "date" in col.lower():
            date_col = col
        if "close" in col.lower():
            close_col = col

    if date_col is None:
        raise ValueError("Date column not found")

    if close_col is None:
        raise ValueError("Close column not found")

    df = df[[date_col, close_col]].copy()
    df.columns = ["Date", "Close"]

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")

    df = df.dropna(subset=["Date", "Close"])
    df = df[df["Close"] > 0]
    df = df.sort_values("Date").reset_index(drop=True)

    df["session"] = df.groupby("Date").cumcount()

    df["daily_frequency"] = (
        df.groupby("Date")["Close"]
        .transform("count")
    )

    df["intra_change"] = (
        df["Close"]
        - df.groupby("Date")["Close"].transform("first")
    )

    df["intra_return"] = (
        df.groupby("Date")["Close"]
        .pct_change()
    )

    df = df.replace([np.inf, -np.inf], np.nan)
    df["intra_return"] = df["intra_return"].fillna(0)

    return df


def load_event_data():
    if not os.path.exists(EVENT_FILE):
        raise ValueError(f"Event file not found: {EVENT_FILE}")

    events = pd.read_csv(EVENT_FILE)

    events["announcement_date"] = pd.to_datetime(
        events["announcement_date"],
        errors="coerce"
    )

    events["dividend_percent"] = pd.to_numeric(
        events["dividend_percent"],
        errors="coerce"
    )

    return events


def load_sentiment_data():
    if not os.path.exists(SENTIMENT_FILE):
        return pd.DataFrame(columns=[
            "company",
            "announcement_date",
            "sentiment_score",
            "sentiment_confidence",
            "event_importance_score"
        ])

    sentiment = pd.read_csv(SENTIMENT_FILE)

    sentiment["announcement_date"] = pd.to_datetime(
        sentiment["announcement_date"],
        errors="coerce"
    )

    sentiment["sentiment_score"] = pd.to_numeric(
        sentiment["sentiment_score"],
        errors="coerce"
    ).fillna(0.0)

    sentiment["sentiment_confidence"] = pd.to_numeric(
        sentiment.get("sentiment_confidence", 0.0),
        errors="coerce"
    ).fillna(0.0)

    sentiment["event_importance_score"] = pd.to_numeric(
        sentiment["event_importance_score"],
        errors="coerce"
    ).fillna(0.0)

    return sentiment


def merge_event_features(df, company):
    events = load_event_data()
    sentiment_data = load_sentiment_data()

    company_events = events[
        events["company"] == company
    ].copy()

    company_sentiment = sentiment_data[
        sentiment_data["company"] == company
    ].copy()

    df["dividend_event"] = 0
    df["dividend_percent"] = 0.0
    df["bonus_event"] = 0
    df["agm_event"] = 0

    df["days_to_dividend"] = -1
    df["days_after_dividend"] = -1

    df["same_day_dividend_event"] = 0
    df["post_dividend_7day_window"] = 0
    df["days_since_dividend_announcement"] = -1
    df["strong_dividend"] = 0

    # FinBERT sentiment features
    df["announcement_sentiment_score"] = 0.0
    df["announcement_sentiment_confidence"] = 0.0
    df["finbert_event_importance_score"] = 0.0

    for _, event in company_events.iterrows():
        event_date = event["announcement_date"]

        if pd.isna(event_date):
            continue

        event_mask = df["Date"].dt.date == event_date.date()

        dividend_percent = 0.0
        if not pd.isna(event["dividend_percent"]):
            dividend_percent = float(event["dividend_percent"])

        title = str(event["title"]).lower()

        sentiment_match = company_sentiment[
            company_sentiment["announcement_date"].dt.date == event_date.date()
        ]

        sentiment_score = 0.0
        sentiment_confidence = 0.0
        event_importance_score = 0.0

        if not sentiment_match.empty:
            sentiment_score = float(
                sentiment_match["sentiment_score"].iloc[0]
            )

            sentiment_confidence = float(
                sentiment_match["sentiment_confidence"].iloc[0]
            )

            event_importance_score = float(
                sentiment_match["event_importance_score"].iloc[0]
            )

        df.loc[event_mask, "dividend_event"] = 1
        df.loc[event_mask, "same_day_dividend_event"] = 1
        df.loc[event_mask, "days_to_dividend"] = 0
        df.loc[event_mask, "days_after_dividend"] = 0
        df.loc[event_mask, "days_since_dividend_announcement"] = 0

        df.loc[event_mask, "announcement_sentiment_score"] = sentiment_score
        df.loc[event_mask, "announcement_sentiment_confidence"] = sentiment_confidence
        df.loc[event_mask, "finbert_event_importance_score"] = event_importance_score

        if dividend_percent > 0:
            df.loc[event_mask, "dividend_percent"] = dividend_percent

        if dividend_percent >= 10:
            df.loc[event_mask, "strong_dividend"] = 1

        if pd.notna(event["bonus_ratio"]):
            df.loc[event_mask, "bonus_event"] = 1

        if "agm" in title:
            df.loc[event_mask, "agm_event"] = 1

        pre_window = 14

        pre_mask = (
            (df["Date"] < event_date)
            &
            (df["Date"] >= event_date - pd.Timedelta(days=pre_window))
        )

        df.loc[pre_mask, "days_to_dividend"] = (
            event_date - df.loc[pre_mask, "Date"]
        ).dt.days

        post_window = 7

        post_mask = (
            (df["Date"] > event_date)
            &
            (df["Date"] <= event_date + pd.Timedelta(days=post_window))
        )

        df.loc[post_mask, "days_after_dividend"] = (
            df.loc[post_mask, "Date"] - event_date
        ).dt.days

        df.loc[post_mask, "post_dividend_7day_window"] = 1

        df.loc[post_mask, "days_since_dividend_announcement"] = (
            df.loc[post_mask, "Date"] - event_date
        ).dt.days

        df.loc[post_mask, "announcement_sentiment_score"] = sentiment_score
        df.loc[post_mask, "announcement_sentiment_confidence"] = sentiment_confidence
        df.loc[post_mask, "finbert_event_importance_score"] = event_importance_score

        if dividend_percent > 0:
            df.loc[post_mask, "dividend_percent"] = dividend_percent

        if dividend_percent >= 10:
            df.loc[post_mask, "strong_dividend"] = 1

        if pd.notna(event["bonus_ratio"]):
            df.loc[post_mask, "bonus_event"] = 1

        if "agm" in title:
            df.loc[post_mask, "agm_event"] = 1

    return df


def engineer_features(df, company):
    df["Company"] = company

    for n in [1, 2, 3, 5]:
        df[f"return_{n}"] = df["Close"].pct_change(n)

    for n in [1, 2, 3]:
        df[f"log_return_{n}"] = np.log(
            df["Close"] / df["Close"].shift(n)
        )

    for w in [3, 5, 10]:
        df[f"sma_{w}"] = (
            df["Close"]
            .rolling(w, min_periods=1)
            .mean()
        )

        df[f"ema_{w}"] = (
            df["Close"]
            .ewm(span=w, adjust=False)
            .mean()
        )

    for n in [3, 5, 10]:
        df[f"momentum_{n}"] = df["Close"] - df["Close"].shift(n)

    df["velocity"] = df["Close"] - df["Close"].shift(1)
    df["acceleration"] = df["velocity"] - df["velocity"].shift(1)

    for w in [3, 5, 10]:
        df[f"volatility_{w}"] = (
            df["return_1"]
            .rolling(w, min_periods=2)
            .std()
        )

    delta = df["Close"].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(7).mean()
    avg_loss = loss.rolling(7).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)

    df["rsi_7"] = 100 - (100 / (1 + rs))
    df["rsi_7"] = df["rsi_7"].fillna(50)

    ema12 = df["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df["Close"].ewm(span=26, adjust=False).mean()

    df["macd"] = ema12 - ema26

    df["macd_signal"] = (
        df["macd"]
        .ewm(span=9, adjust=False)
        .mean()
    )

    rolling_mean = (
        df["Close"]
        .rolling(10, min_periods=2)
        .mean()
    )

    rolling_std = (
        df["Close"]
        .rolling(10, min_periods=2)
        .std()
    )

    df["bb_upper"] = rolling_mean + 2 * rolling_std
    df["bb_lower"] = rolling_mean - 2 * rolling_std

    df["month"] = df["Date"].dt.month
    df["day_of_week"] = df["Date"].dt.dayofweek

    df["dividend_season"] = np.where(
        df["month"].isin([3, 4, 5]),
        1,
        0
    )

    df["high_activity_day"] = np.where(
        df["daily_frequency"] >= 4,
        1,
        0
    )

    df["positive_sentiment_event"] = np.where(
        df["announcement_sentiment_score"] > 0,
        1,
        0
    )

    df["negative_sentiment_event"] = np.where(
        df["announcement_sentiment_score"] < 0,
        1,
        0
    )

    df["sentiment_weighted_importance"] = (
        df["announcement_sentiment_score"]
        * df["finbert_event_importance_score"]
    )

    df["event_activity_score"] = (
        df["daily_frequency"]
        * (
            1
            + df["same_day_dividend_event"]
            + df["post_dividend_7day_window"]
        )
    )
    
    df["event_impact_score"] = (

    # Dividend strength
    (df["dividend_percent"].fillna(0) / 100)

    # Strong actual market reaction
    + (0.30 * df["same_day_dividend_event"])

    # Continued momentum after announcement
    + (0.25 * df["post_dividend_7day_window"])

    # Strong dividend announcements
    + (0.20 * df["strong_dividend"])

    # Market activity increase
    + (
        0.10
        * (
            df["event_activity_score"]
            .clip(0, 1)
        )
    )

    # FinBERT contextual importance
    + (0.08 * df["finbert_event_importance_score"])

    # Positive sentiment influence
    + (
        0.07
        * (
            df["announcement_sentiment_score"]
            .clip(lower=0)
        )
    )

    # Bonus share influence
    + (0.05 * df["bonus_event"])

    # AGM effect
    + (0.03 * df["agm_event"])
)

    df["event_impact_score"] = df["event_impact_score"].clip(0, 1)

    df["post_dividend_momentum"] = (
        df["return_1"].fillna(0)
        * df["post_dividend_7day_window"]
    )

    df["event_volatility"] = (
        df["volatility_5"].fillna(0)
        * (
            df["same_day_dividend_event"]
            + df["post_dividend_7day_window"]
        )
    )

    

    df["dividend_reaction_strength"] = (
        df["event_impact_score"]
        * df["event_activity_score"]
    )

    df["sentiment_event_momentum"] = (
        df["announcement_sentiment_score"]
        * df["post_dividend_momentum"]
    )

    df["future_close_week"] = df["Close"].shift(-10)
    df["future_close_month"] = df["Close"].shift(-20)

    future_diff = df["future_close_week"] - df["Close"]

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

    df = df.replace([np.inf, -np.inf], np.nan)

    numeric_cols = df.select_dtypes(include=[np.number]).columns

    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()

    required_cols = [
        "Close",
        "future_close_week",
        "future_close_month"
    ]

    df = (
        df.dropna(subset=required_cols)
        .reset_index(drop=True)
    )

    return df


def process_company_file(csv_path, company):
    print("=" * 60)
    print(f"PROCESSING {company}")
    print("=" * 60)

    raw_df = pd.read_csv(csv_path)

    clean_df = clean_stock_data(raw_df)

    clean_df = merge_event_features(
        clean_df,
        company
    )

    final_df = engineer_features(
        clean_df,
        company
    )

    print(final_df.head())

    return final_df