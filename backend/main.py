import os
import re
import json
import hashlib
from unittest import result
import numpy as np
import pandas as pd

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import time

from scraper.downloader import download_stock_data

from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

from openai import OpenAI
from dotenv import load_dotenv

from financial_agent.financial_agent import FinancialAgent

from old_pipeline.loader import (
    get_company_files,
    load_processed_data,
    latest_value,
    dataframe_to_records
)

from old_pipeline.processor import (
    process_raw_dataframe,
    extract_company_name_from_filename
)

from prediction.predict_best_model import predict_company
from prediction.prediction_history import (
    save_prediction_history,
    get_prediction_history
)
from recommendation.hybrid_recommendation_engine import (
    hybrid_recommend_one_stock,
    hybrid_recommend_top_stocks
)
from recommendation.user_activity_logger import log_user_activity, get_user_activity_history
import recommendation.hybrid_recommendation_engine

import requests
from bs4 import BeautifulSoup

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FOLDER = os.path.join(BASE_DIR, "data")

FINANCIAL_DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "financial_db",
    "financial_records.csv"
)

CHAT_HISTORY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "financial_chat_history.json"
)

CACHE_DIR = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI(
    title="RSEB AI Investor Assistant API",
    description="Forecasting, recommendation, financial agent, RAG chatbot, and investor assistance API.",
    version="3.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

financial_agent = FinancialAgent()


# =====================================================
# COMMON HELPERS
# =====================================================

def safe_float(value, decimals=4):
    if value is None:
        return None

    try:
        if pd.isna(value) or np.isinf(value):
            return None

        return round(float(value), decimals)

    except Exception:
        return None


def format_currency(value):
    if value is None:
        return None

    try:
        return f"Nu. {float(value):,.2f}"
    except Exception:
        return value


def format_ratio_value(name, value):
    if value is None:
        return None

    try:
        numeric = float(value)

        if name == "Earnings Per Share":
            return format_currency(numeric)

        if name == "Current Ratio":
            return f"{numeric:.2f}x"

        if name in [
            "Net Profit Margin",
            "Return on Assets",
            "Return on Equity",
            "Debt Ratio",
            "Debt to Equity",
            "Operating Cash Ratio"
        ]:
            return f"{numeric * 100:.2f}%"

        return f"{numeric:.4f}"

    except Exception:
        return value


def format_ratio_dict(ratios):
    return {
        k: format_ratio_value(k, v)
        for k, v in ratios.items()
    }


def format_currency_dict(values):
    formatted = {}

    for key, value in values.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            formatted[key] = format_currency(value)
        else:
            formatted[key] = value

    return formatted


def filter_by_period(df: pd.DataFrame, period: str):
    if "Date" not in df.columns:
        return df

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date")

    period = period.upper()

    if period == "ALL":
        return df

    latest_date = df["Date"].max()

    if period == "1M":
        start_date = latest_date - pd.DateOffset(months=1)
    elif period == "3M":
        start_date = latest_date - pd.DateOffset(months=3)
    elif period == "6M":
        start_date = latest_date - pd.DateOffset(months=6)
    elif period == "1Y":
        start_date = latest_date - pd.DateOffset(years=1)
    else:
        return df

    return df[df["Date"] >= start_date]


def build_metrics(company: str, df: pd.DataFrame):
    close = latest_value(df, "Close")
    trend = latest_value(df, "target")
    ret = latest_value(df, "return_1")
    vol = latest_value(df, "volatility_5")

    first_close = None
    last_close = None
    total_change = None
    total_return_pct = None

    valid = df.dropna(subset=["Close"]).copy()

    if not valid.empty:
        first_close = valid["Close"].iloc[0]
        last_close = valid["Close"].iloc[-1]
        total_change = last_close - first_close

        if first_close != 0:
            total_return_pct = (total_change / first_close) * 100

    return {
        "latest_close": safe_float(close, 2),
        "return": safe_float(ret, 4),
        "volatility": safe_float(vol, 4),
        "trend": trend,
        "first_close": safe_float(first_close, 2),
        "last_close": safe_float(last_close, 2),
        "total_change": safe_float(total_change, 2),
        "total_return_pct": safe_float(total_return_pct, 2),
    }


def company_summary(company: str, df: pd.DataFrame):
    metrics = build_metrics(company, df)

    return {
        "company": company,
        "latest_close": metrics["latest_close"],
        "return": metrics["return"],
        "volatility": metrics["volatility"],
        "trend": metrics["trend"],
        "total_change": metrics["total_change"],
        "total_return_pct": metrics["total_return_pct"],
    }


# =====================================================
# GPT INSIGHTS
# =====================================================

def generate_ai_insight(company: str, df: pd.DataFrame, metrics: dict):
    try:
        latest_close = metrics.get("latest_close")
        total_return = metrics.get("total_return_pct")
        volatility = metrics.get("volatility")
        trend = metrics.get("trend")

        cache_key = f"{company}_{latest_close}_{total_return}_{volatility}_{trend}"
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"insight_{cache_hash}.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)["insight"]

        recent = df.tail(30).copy()
        latest_sma = latest_value(recent, "sma_5")
        latest_ema = latest_value(recent, "ema_5")

        recent_prices = recent["Close"].dropna().tail(10).tolist()
        highest_price = recent["Close"].max() if "Close" in recent.columns else None
        lowest_price = recent["Close"].min() if "Close" in recent.columns else None

        prompt = f"""
You are a professional stock market analyst.

Company: {company}
Latest Close: Nu. {latest_close}
Total Return: {total_return}%
Volatility: {volatility}
Trend: {trend}
SMA 5: {latest_sma}
EMA 5: {latest_ema}
Highest Recent Price: Nu. {highest_price}
Lowest Recent Price: Nu. {lowest_price}
Recent Prices: {recent_prices}

Write a concise stock insight in maximum 90 words.
Use Nu. for currency.
Do not use bullets.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial analyst. Use Nu. for currency."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=180
        )

        insight = response.choices[0].message.content

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"insight": insight}, f, indent=4)

        return insight

    except Exception as e:
        return f"AI insight generation failed: {str(e)}"


def generate_forecast_insight(prediction_result):
    try:
        company = prediction_result.get("company")
        current_price = prediction_result.get("current_price")
        predicted_week_price = prediction_result.get("predicted_week_price")
        predicted_month_price = prediction_result.get("predicted_month_price")
        predicted_direction = prediction_result.get("predicted_direction")
        direction_confidence = prediction_result.get("direction_confidence")
        week_change_percent = prediction_result.get("week_change_percent")
        month_change_percent = prediction_result.get("month_change_percent")

        cache_key = (
            f"{company}_{current_price}_{predicted_week_price}_"
            f"{predicted_month_price}_{predicted_direction}_"
            f"{direction_confidence}_{week_change_percent}_{month_change_percent}"
        )

        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        cache_file = os.path.join(CACHE_DIR, f"forecast_{cache_hash}.json")

        if os.path.exists(cache_file):
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)["insight"]

        prompt = f"""
You are an institutional financial analyst.

Company: {company}
Current Price: Nu. {current_price}
Predicted 1-Week Price: Nu. {predicted_week_price}
Predicted 1-Month Price: Nu. {predicted_month_price}
Predicted Direction: {predicted_direction}
Direction Confidence: {direction_confidence}%
1-Week Change: {week_change_percent}%
1-Month Change: {month_change_percent}%

Explain the forecast clearly for a retail investor.
Mention short-term and monthly outlook.
Use Nu. for currency.
Maximum 120 words.
No bullets.
"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a professional financial analyst. Use Nu. for currency and avoid investment guarantees."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.4,
            max_tokens=220
        )

        insight = response.choices[0].message.content

        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump({"insight": insight}, f, indent=4)

        return insight

    except Exception as e:
        return f"Forecast insight generation failed: {str(e)}"


# =====================================================
# FINANCIAL AGENT HELPERS
# =====================================================

def load_financial_records():
    if not os.path.exists(FINANCIAL_DB_PATH):
        raise HTTPException(
            status_code=404,
            detail="financial_records.csv not found. Run financial database builder first."
        )

    df = pd.read_csv(FINANCIAL_DB_PATH)

    if df.empty:
        raise HTTPException(
            status_code=404,
            detail="financial_records.csv exists but has no records."
        )

    df["company"] = df["company"].astype(str).str.upper().str.strip()
    df["item"] = df["item"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    df = df[(df["year"] >= 2000) & (df["year"] <= 2035)]

    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    df = df.dropna(subset=["value"])

    if "raw_item" not in df.columns:
        df["raw_item"] = ""

    if "report_type" not in df.columns:
        df["report_type"] = ""

    df["raw_item"] = df["raw_item"].astype(str)
    df["report_type"] = df["report_type"].astype(str)

    return df


def save_financial_chat_history(record):
    os.makedirs(os.path.dirname(CHAT_HISTORY_PATH), exist_ok=True)

    history = []

    if os.path.exists(CHAT_HISTORY_PATH):
        try:
            with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
                history = json.load(f)
        except Exception:
            history = []

    history.append(record)

    with open(CHAT_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=4)


def get_financial_chat_history(company=None, limit=100):
    if not os.path.exists(CHAT_HISTORY_PATH):
        return []

    try:
        with open(CHAT_HISTORY_PATH, "r", encoding="utf-8") as f:
            history = json.load(f)
    except Exception:
        return []

    if company:
        company = company.upper()
        history = [
            item for item in history
            if item.get("company") == company
        ]

    return history[-int(limit):]


def get_latest_raw_csv(company):
    company = company.upper()

    possible_dirs = [
        os.path.join(BASE_DIR, "data", "raw", company),
        os.path.join(BASE_DIR, "data", "raw"),
        os.path.join(BASE_DIR, "data"),
    ]

    csv_files = []

    for folder in possible_dirs:
        if not os.path.exists(folder):
            continue

        for root, dirs, files in os.walk(folder):
            for file in files:
                if file.lower().endswith(".csv") and company.lower() in file.lower():
                    csv_files.append(os.path.join(root, file))

    if not csv_files:
        return None

    return max(csv_files, key=os.path.getmtime)


# =====================================================
# BASIC ROUTES
# =====================================================

@app.get("/")
def home():
    return {
        "message": "RSEB AI Investor Assistant API is running",
        "status": "success",
        "version": "3.0.0"
    }


@app.get("/companies")
def get_companies():
    files = get_company_files(DATA_FOLDER)

    return {
        "status": "success",
        "companies": list(files.keys()),
        "count": len(files)
    }


# =====================================================
# STOCK DASHBOARD ROUTES
# =====================================================

@app.get("/market-summary")
def get_market_summary(period: str = Query("ALL")):
    files = get_company_files(DATA_FOLDER)
    summaries = []

    for company, path in files.items():
        try:
            df = load_processed_data(path)
            filtered_df = filter_by_period(df, period)

            if filtered_df.empty:
                continue

            summaries.append(company_summary(company, filtered_df))

        except Exception:
            continue

    summaries = sorted(
        summaries,
        key=lambda x: x["total_return_pct"] if x["total_return_pct"] is not None else -999999,
        reverse=True
    )

    return {
        "status": "success",
        "period": period.upper(),
        "count": len(summaries),
        "companies": summaries
    }


@app.get("/dashboard/{company}")
def get_dashboard_data(
    company: str,
    period: str = Query("ALL"),
    user_id: str = Query("demo_user")
):
    company = company.upper()
    files = get_company_files(DATA_FOLDER)

    if company not in files:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for company: {company}"
        )

    df = load_processed_data(files[company])
    filtered_df = filter_by_period(df, period)

    if filtered_df.empty:
        raise HTTPException(
            status_code=404,
            detail=f"No data available for {company} in selected period: {period}"
        )

    try:
        log_user_activity(
            user_id=user_id,
            company=company,
            action_type="overview_view",
            source="dashboard"
        )
    except Exception:
        pass

    metrics = build_metrics(company, filtered_df)
    ai_insight = generate_ai_insight(company, filtered_df, metrics)

    return {
        "status": "success",
        "company": company,
        "period": period.upper(),
        "metrics": metrics,
        "chart_data": dataframe_to_records(filtered_df),
        "latest_insight": {
            "text": ai_insight
        }
    }


@app.get("/analytics/{company}")
def get_analytics_data(company: str, period: str = Query("ALL")):
    company = company.upper()
    files = get_company_files(DATA_FOLDER)

    if company not in files:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for company: {company}"
        )

    df = load_processed_data(files[company])
    filtered_df = filter_by_period(df, period)

    trend_counts = (
        filtered_df["target"]
        .value_counts(dropna=False)
        .reset_index()
    )

    trend_counts.columns = ["trend", "count"]

    return {
        "status": "success",
        "company": company,
        "period": period.upper(),
        "return_data": dataframe_to_records(
            filtered_df[["Date", "return_1"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        ),
        "volatility_data": dataframe_to_records(
            filtered_df[["Date", "volatility_5"]]
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        ),
        "trend_distribution": dataframe_to_records(trend_counts)
    }


@app.get("/data/{company}")
def get_full_data(company: str):
    company = company.upper()
    files = get_company_files(DATA_FOLDER)

    if company not in files:
        raise HTTPException(
            status_code=404,
            detail=f"No data found for company: {company}"
        )

    df = load_processed_data(files[company])

    return {
        "status": "success",
        "company": company,
        "rows": len(df),
        "data": dataframe_to_records(df)
    }


@app.post("/upload")
async def upload_raw_csv(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Only CSV files are allowed"
        )

    try:
        raw = pd.read_csv(file.file)
        company = extract_company_name_from_filename(file.filename)
        processed_df = process_raw_dataframe(raw, company)

        metrics = build_metrics(company, processed_df)
        ai_insight = generate_ai_insight(company, processed_df, metrics)

        return {
            "status": "success",
            "company": company,
            "rows": len(processed_df),
            "metrics": metrics,
            "chart_data": dataframe_to_records(processed_df),
            "data": dataframe_to_records(processed_df),
            "latest_insight": {
                "text": ai_insight
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

PROFILE_CACHE = {}
@app.get("/company-profile/{company}")
def company_profile(company: str):

    company = company.upper()

    if company in PROFILE_CACHE:
        return PROFILE_CACHE[company]

    url = f"https://rsebl.org.bt/stocks/{company}"

    try:

        chrome_options = Options()

        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(
            service=Service(
                ChromeDriverManager().install()
            ),
            options=chrome_options
        )

        driver.get(url)

        time.sleep(5)

        body_text = driver.find_element(By.TAG_NAME, "body").text

        driver.quit()

        lines = body_text.split("\n")

        extra_info = {
            "Established": "N/A",
            "Sector": "N/A",
            "Address": "N/A",
            "Paid-up Shares": "N/A",
            "Website": "N/A"
        }

        for line in lines:

            line = line.strip()

            if "Established:" in line:
                extra_info["Established"] = (
                    line.replace("Established:", "").strip()
                )

            elif "Sector:" in line:
                extra_info["Sector"] = (
                    line.replace("Sector:", "").strip()
                )

            elif "Address:" in line:
                extra_info["Address"] = (
                    line.replace("Address:", "").strip()
                )

            elif "Paid-up Shares:" in line:
                extra_info["Paid-up Shares"] = (
                    line.replace("Paid-up Shares:", "").strip()
                )

            elif "Website:" in line:
                extra_info["Website"] = (
                    line.replace("Website:", "").strip()
                )

        return {
            "status": "success",
            "company": company,
            "url": url,
            "extra_info": extra_info
        }
        
        PROFILE_CACHE[company] = result

        return result

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
    
@app.get("/rseb-stock-info/{company}")
def rseb_stock_info(company: str):
    company = company.upper()
    url = f"https://rsebl.org.bt/stocks/{company}"

    try:
        response = requests.get(url, timeout=15)
        soup = BeautifulSoup(response.text, "html.parser")
        text = soup.get_text(" ", strip=True)

        metrics = {
            "Open": None,
            "High": None,
            "Low": None,
            "Vol": None,
            "P/E": None,
            "Mkt cap": None,
            "52W H": None,
            "52W L": None,
            "Avg Vol": None,
            "Div yield": None,
            "Book Val": None,
            "EPS": None,
        }

        for key in metrics.keys():
            pattern = rf"{key}\s+([^\s]+)"
            match = re.search(pattern, text)
            if match:
                metrics[key] = match.group(1)

        return {
            "status": "success",
            "company": company,
            "source": url,
            "metrics": metrics
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# FORECASTING ROUTES
# =====================================================

@app.get("/predict/{company}")
def predict_stock(
    company: str,
    scrape_latest: bool = Query(True),
    headless: bool = Query(True),
    user_id: str = Query("demo_user")
):
    try:
        company = company.upper()

        if scrape_latest:
            csv_path = download_stock_data(
                company=company,
                headless=headless
            )

            if csv_path is None:
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to scrape latest data for {company}"
                )

        else:
            csv_path = get_latest_raw_csv(company)

            if csv_path is None:
                raise HTTPException(
                    status_code=404,
                    detail=f"No raw CSV file found for {company}"
                )

        result = predict_company(
            company=company,
            csv_path=csv_path
        )

        result["forecast_insight"] = generate_forecast_insight(result)

        save_prediction_history(result)

        try:
            log_user_activity(
                user_id=user_id,
                company=company,
                action_type="prediction_view",
                source="forecast"
            )
        except Exception:
            pass

        return {
            "status": "success",
            "data": result
        }

    except HTTPException:
        raise

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/prediction-history")
def prediction_history(company: str = Query(None)):
    try:
        history = get_prediction_history(company)

        return {
            "status": "success",
            "history": history
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# NEW HYBRID RECOMMENDATION ROUTES
# =====================================================

@app.get("/recommend/{user_id}")
def recommend(user_id: str, top_k: int = 5):

    try:

        recommendations = hybrid_recommend_top_stocks(
            user_id=user_id,
            top_k=top_k
        )

        return {
            "status": "success",
            "user_id": user_id,
            "recommendations": recommendations
        }

    except Exception as e:

        print("Recommendation Error:", e)

        return {
            "status": "error",
            "user_id": user_id,
            "recommendations": [],
            "message": str(e)
        }

@app.get("/stock-suggestions")
def stock_suggestions(
    user_id: str = Query("demo_user"),
    limit: int = Query(1)
):
    try:
        result = recommendation.hybrid_recommendation_engine.hybrid_recommend_one_stock(user_id)

        return {
            "status": "success",
            "count": 1,
            "suggestions": [result],
            "logic": {
                "personalization": "Based on BERT4Rec user behaviour sequence.",
                "forecast": "Based on weekly and monthly prediction output.",
                "direction": "Based on XGBoost direction confidence.",
                "event": "Based on dividend/event and sentiment-aware features.",
                "decision_rule": "Hybrid decision-support recommendation, not financial advice."
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


@app.get("/user-activity/{user_id}")
def user_activity(user_id: str):
    try:
        history = get_user_activity_history(user_id=user_id)

        return {
            "status": "success",
            "user_id": user_id,
            "count": len(history),
            "history": history
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# =====================================================
# FINANCIAL AGENT ROUTES
# =====================================================

@app.get("/financial-agent/companies")
def financial_agent_companies():
    df = load_financial_records()
    companies = sorted(df["company"].dropna().unique().tolist())

    return {
        "status": "success",
        "companies": companies,
        "count": len(companies)
    }


@app.get("/financial-agent/years/{company}")
def financial_agent_years(company: str):
    df = load_financial_records()
    company = company.upper()

    years = sorted(
        df[df["company"] == company]["year"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "status": "success",
        "company": company,
        "years": [int(y) for y in years]
    }


@app.get("/financial-agent/items/{company}")
def financial_agent_items(company: str):
    df = load_financial_records()
    company = company.upper()

    items = sorted(
        df[df["company"] == company]["item"]
        .dropna()
        .unique()
        .tolist()
    )

    return {
        "status": "success",
        "company": company,
        "items": items
    }


@app.post("/financial-agent/chat")
def financial_agent_chat(payload: dict):
    question = payload.get("question", "")

    if not question:
        raise HTTPException(
            status_code=400,
            detail="question is required"
        )

    try:
        result = financial_agent.answer_user_question(question)

        response_payload = {
            "status": "success",
            "question": question,
            "type": result.get("type"),
            "company": result.get("company"),
            "item": result.get("item"),
            "answer": result.get("answer"),
            "explanation": result.get("explanation"),
            "data": result.get("data"),
        }

        save_financial_chat_history(response_payload)

        try:
            if result.get("company"):
                log_user_activity(
                    user_id=payload.get("user_id", "demo_user"),
                    company=result.get("company"),
                    action_type="chat_query",
                    source="financial_agent"
                )
        except Exception:
            pass

        return response_payload

    except Exception as e:
        import traceback
        print("\n========== FINANCIAL AGENT CHAT ERROR ==========")
        traceback.print_exc()
        print("===============================================\n")

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

@app.get("/financial-agent/chat-history")
def financial_agent_chat_history(
    company: str = Query(None),
    limit: int = Query(100)
):
    history = get_financial_chat_history(
        company=company,
        limit=limit
    )

    return {
        "status": "success",
        "count": len(history),
        "history": history
    }


@app.post("/chat")
def chat_with_agent(payload: dict):
    question = payload.get("question", "")

    if not question:
        raise HTTPException(
            status_code=400,
            detail="Question is required"
        )

    try:
        result = financial_agent.answer_user_question(question)

        try:
            if result.get("company"):
                log_user_activity(
                    user_id=payload.get("user_id", "demo_user"),
                    company=result.get("company"),
                    action_type="chat_query",
                    source="chatbot"
                )
        except Exception:
            pass

        return {
            "status": "success",
            **result
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )