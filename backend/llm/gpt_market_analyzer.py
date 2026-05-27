import os
import json
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# =========================================================
# LOAD ENV
# =========================================================

load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_DIR = os.path.join(
    BASE_DIR,
    "data",
    "processed_event"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "llm_analysis"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# =========================================================
# COMPANIES
# =========================================================

COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]


# =========================================================
# LOAD COMPANY DATA
# =========================================================

def load_company_data(company):

    path = os.path.join(
        PROCESSED_DIR,
        f"{company}_event_featured.csv"
    )

    if not os.path.exists(path):
        raise ValueError(f"Missing file: {path}")

    df = pd.read_csv(path)

    return df


# =========================================================
# CREATE MARKET SUMMARY
# =========================================================

def create_market_summary(df, company):

    recent = df.tail(50).copy()

    latest_close = recent["Close"].iloc[-1]

    avg_return = recent["return_1"].mean()

    avg_volatility = recent["volatility_5"].mean()

    latest_rsi = recent["rsi_7"].iloc[-1]

    dividend_events = int(
        recent["dividend_event"].sum()
    )

    bonus_events = int(
        recent["bonus_event"].sum()
    )

    agm_events = int(
        recent["agm_event"].sum()
    )

    high_activity_days = int(
        recent["high_activity_day"].sum()
    )

    avg_daily_frequency = (
        recent["daily_frequency"].mean()
    )

    momentum = recent["momentum_5"].iloc[-1]

    velocity = recent["velocity"].iloc[-1]

    acceleration = recent["acceleration"].iloc[-1]

    summary = f"""
Company: {company}

Latest Close Price: {latest_close:.2f}

Average Return: {avg_return:.4f}

Average Volatility: {avg_volatility:.4f}

Latest RSI: {latest_rsi:.2f}

Dividend Events Detected: {dividend_events}

Bonus Share Events: {bonus_events}

AGM Events: {agm_events}

High Activity Trading Sessions: {high_activity_days}

Average Daily Capture Frequency: {avg_daily_frequency:.2f}

Latest Momentum: {momentum:.4f}

Latest Velocity: {velocity:.4f}

Latest Acceleration: {acceleration:.4f}

Bhutan stock market behavior is highly dividend-driven.
Please analyze the company behavior considering:
- dividend announcements
- investor accumulation
- low liquidity market
- seasonal dividend movement
- trading activity
- momentum
- volatility
"""

    return summary


# =========================================================
# GPT ANALYSIS
# =========================================================

def analyze_with_gpt(company, summary):

    prompt = f"""
You are a Bhutan stock market analyst.

Analyze the following stock behavior carefully.

Provide:
1. Market sentiment (Bullish/Bearish/Neutral)
2. Confidence score (0-100)
3. Short-term outlook
4. Dividend impact
5. Trading activity interpretation
6. Investor behavior interpretation
7. Final recommendation

Keep response professional and concise.

DATA:
{summary}
"""

    response = client.chat.completions.create(

        model="gpt-4o-mini",

        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert financial analyst "
                    "specializing in Bhutan stock markets."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],

        temperature=0.3
    )

    analysis = response.choices[0].message.content

    return analysis


# =========================================================
# SAVE ANALYSIS
# =========================================================

def save_analysis(company, analysis):

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{company}_analysis.json"
    )

    result = {
        "company": company,
        "analysis": analysis
    }

    with open(output_path, "w", encoding="utf-8") as f:

        json.dump(
            result,
            f,
            indent=4,
            ensure_ascii=False
        )

    print(f"Saved analysis: {output_path}")


# =========================================================
# PROCESS SINGLE COMPANY
# =========================================================

def process_company(company):

    print("=" * 60)
    print(f"GPT ANALYSIS: {company}")
    print("=" * 60)

    df = load_company_data(company)

    summary = create_market_summary(
        df,
        company
    )

    analysis = analyze_with_gpt(
        company,
        summary
    )

    print("\nGPT ANALYSIS:")
    print("-" * 40)
    print(analysis)

    save_analysis(
        company,
        analysis
    )


# =========================================================
# PROCESS ALL COMPANIES
# =========================================================

def process_all_companies():

    for company in COMPANIES:

        try:

            process_company(company)

        except Exception as e:

            print(f"[ERROR] {company}: {e}")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    process_all_companies()