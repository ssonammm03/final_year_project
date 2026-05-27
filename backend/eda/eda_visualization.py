import os
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data", "processed_event")
OUTPUT_DIR = os.path.join(BASE_DIR, "eda", "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]


# CLEAN PROFESSIONAL COLORS
PRICE = "#0077B6"
EMA = "#06D6A0"
SMA = "#FFB703"
EVENT = "#EF476F"
VOL = "#9B5DE5"
RSI = "#F15BB5"
POSITIVE = "#2A9D8F"
NEGATIVE = "#E63946"
GRID = "#D9D9D9"


plt.style.use("default")

plt.rcParams["figure.figsize"] = (14, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"


def load_company(company):

    path = os.path.join(
        DATA_DIR,
        f"{company}_event_featured.csv"
    )

    df = pd.read_csv(path)

    df["Date"] = pd.to_datetime(df["Date"])

    # SORT BY DATE
    df = df.sort_values("Date")

    # AGGREGATE MULTIPLE ENTRIES PER DAY
    aggregation = {}

    if "Close" in df.columns:
        aggregation["Close"] = "last"

    if "return_1" in df.columns:
        aggregation["return_1"] = "mean"

    if "volatility_5" in df.columns:
        aggregation["volatility_5"] = "mean"

    if "rsi_7" in df.columns:
        aggregation["rsi_7"] = "mean"

    if "same_day_dividend_event" in df.columns:
        aggregation["same_day_dividend_event"] = "max"

    if "announcement_sentiment_score" in df.columns:
        aggregation["announcement_sentiment_score"] = "mean"

    if "sma_5" in df.columns:
        aggregation["sma_5"] = "mean"

    if "ema_5" in df.columns:
        aggregation["ema_5"] = "mean"

    if "event_impact_score" in df.columns:
        aggregation["event_impact_score"] = "mean"

    # GROUP BY DATE
    df = (
        df.groupby("Date")
        .agg(aggregation)
        .reset_index()
    )

    return df


def style_plot():

    ax = plt.gca()

    ax.grid(
        True,
        linestyle="--",
        alpha=0.25,
        color=GRID
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def save_figure(path):

    plt.tight_layout()

    plt.savefig(
        path,
        dpi=500,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {path}")


def plot_price_trend(company, df):

    plt.figure()

    style_plot()

    plt.plot(
        df["Date"],
        df["Close"],
        color=PRICE,
        linewidth=2.5,
        label="Close Price"
    )

    if "sma_5" in df.columns:

        plt.plot(
            df["Date"],
            df["sma_5"],
            color=SMA,
            linewidth=2,
            label="SMA 5"
        )

    if "ema_5" in df.columns:

        plt.plot(
            df["Date"],
            df["ema_5"],
            color=EMA,
            linewidth=2,
            label="EMA 5"
        )

    plt.fill_between(
        df["Date"],
        df["Close"],
        color=PRICE,
        alpha=0.06
    )

   

    plt.xlabel("Date")
    plt.ylabel("Close Price")

    plt.legend()

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_price_trend.png"
        )
    )


def plot_volatility(company, df):

    plt.figure()

    style_plot()

    plt.plot(
        df["Date"],
        df["volatility_5"],
        color=VOL,
        linewidth=2.5
    )

    plt.fill_between(
        df["Date"],
        df["volatility_5"],
        color=VOL,
        alpha=0.12
    )


    plt.xlabel("Date")
    plt.ylabel("5-Day Volatility")

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_volatility.png"
        )
    )


def plot_dividend_events(company, df):

    plt.figure()

    style_plot()

    plt.plot(
        df["Date"],
        df["Close"],
        color=PRICE,
        linewidth=2.2
    )

    dividend_points = df[
        df["same_day_dividend_event"] == 1
    ]

    plt.scatter(
        dividend_points["Date"],
        dividend_points["Close"],
        color=EVENT,
        marker="^",
        s=120,
        label="Dividend Event"
    )


    plt.xlabel("Date")
    plt.ylabel("Close Price")

    plt.legend()

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_dividend_events.png"
        )
    )


def plot_rsi(company, df):

    plt.figure()

    style_plot()

    plt.plot(
        df["Date"],
        df["rsi_7"],
        color=RSI,
        linewidth=2.5
    )

    plt.axhline(
        70,
        linestyle="--",
        color=NEGATIVE,
        alpha=0.7
    )

    plt.axhline(
        30,
        linestyle="--",
        color=POSITIVE,
        alpha=0.7
    )

    plt.fill_between(
        df["Date"],
        70,
        100,
        color=NEGATIVE,
        alpha=0.05
    )

    plt.fill_between(
        df["Date"],
        0,
        30,
        color=POSITIVE,
        alpha=0.05
    )

    plt.ylim(0, 100)


    plt.xlabel("Date")
    plt.ylabel("RSI")

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_rsi.png"
        )
    )


def plot_sentiment_distribution(company, df):

    if "announcement_sentiment_score" not in df.columns:
        return

    plt.figure(figsize=(8, 5))

    style_plot()

    positive = (
        df["announcement_sentiment_score"] > 0
    ).sum()

    neutral = (
        df["announcement_sentiment_score"] == 0
    ).sum()

    negative = (
        df["announcement_sentiment_score"] < 0
    ).sum()

    labels = [
        "Positive",
        "Neutral",
        "Negative"
    ]

    values = [
        positive,
        neutral,
        negative
    ]

    colors = [
    "#43AA8B",   # Positive
    "#A8B2C1",   # Neutral
    "#E76F51"    # Negative
]

    bars = plt.bar(
        labels,
        values,
        color=colors,
        width=0.55
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
    bar.get_x() + bar.get_width() / 2,
    height,
    f"{int(height)}",
    ha="center",
    va="bottom",
    fontsize=14,
    fontweight="bold"
)


    plt.ylabel("Records")

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_sentiment_distribution.png"
        )
    )


def plot_return_distribution(company, df):

    if "return_1" not in df.columns:
        return

    plt.figure()

    style_plot()

    returns = df["return_1"].dropna()

    plt.hist(
        returns,
        bins=30,
        color="#F4A261",
        edgecolor="white",
        alpha=0.85
    )

    plt.axvline(
        returns.mean(),
        color=EVENT,
        linestyle="--",
        linewidth=2,
        label="Mean Return"
    )


    plt.xlabel("1-Day Return")
    plt.ylabel("Frequency")

    plt.legend()

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_return_distribution.png"
        )
    )


def plot_cumulative_return(company, df):

    if "return_1" not in df.columns:
        return

    plt.figure()

    style_plot()

    cumulative = (
        1 + df["return_1"].fillna(0)
    ).cumprod()

    plt.plot(
        df["Date"],
        cumulative,
        color=POSITIVE,
        linewidth=2.8
    )

    plt.fill_between(
        df["Date"],
        cumulative,
        color=POSITIVE,
        alpha=0.08
    )

    plt.xlabel("Date")
    plt.ylabel("Growth")

    save_figure(
        os.path.join(
            OUTPUT_DIR,
            f"{company}_cumulative_return.png"
        )
    )


def run_eda():

    for company in COMPANIES:

        print("=" * 60)
        print(f"RUNNING EDA FOR {company}")
        print("=" * 60)

        try:

            df = load_company(company)

            plot_price_trend(company, df)
            plot_volatility(company, df)
            plot_dividend_events(company, df)
            plot_rsi(company, df)
            plot_sentiment_distribution(company, df)
            plot_return_distribution(company, df)
            plot_cumulative_return(company, df)

        except Exception as e:

            print(f"[ERROR] {company}: {e}")


if __name__ == "__main__":

    run_eda()