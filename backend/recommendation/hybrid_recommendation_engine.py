import os
import sys
import json
import glob
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from recommendation.recommend_stocks import recommend_stocks
from prediction.predict_best_model import predict_company

from recommendation.recommend_stocks import recommend_stocks
from prediction.predict_best_model import predict_company
from prediction.predict_best_model import predict_company

RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_EVENT_DIR = os.path.join(BASE_DIR, "data", "processed_event")


def get_latest_raw_csv(company):
    company_dir = os.path.join(RAW_DIR, company)
    files = glob.glob(os.path.join(company_dir, "*.csv"))

    if not files:
        return None

    return max(files, key=os.path.getmtime)


def get_latest_event_features(company):
    path = os.path.join(
        PROCESSED_EVENT_DIR,
        f"{company}_event_featured.csv"
    )

    if not os.path.exists(path):
        return {}

    df = pd.read_csv(path)
    latest = df.iloc[-1]

    return {
        "dividend_event": int(latest.get("dividend_event", 0)),
        "dividend_percent": float(latest.get("dividend_percent", 0)),
        "bonus_event": int(latest.get("bonus_event", 0)),
        "agm_event": int(latest.get("agm_event", 0)),
        "dividend_season": int(latest.get("dividend_season", 0)),
        "high_activity_day": int(latest.get("high_activity_day", 0)),
        "current_price": float(latest.get("Close", 0)),
        "recent_min": float(df["Close"].tail(100).min()),
        "recent_max": float(df["Close"].tail(100).max())
    }


def calculate_price_attractiveness(current_price, recent_min, recent_max):
    if recent_max == recent_min:
        return 0.5

    score = 1 - ((current_price - recent_min) / (recent_max - recent_min))
    return max(0, min(1, score))


def clamp_forecast_output(forecast, max_return=25):
    forecast = forecast.copy()

    raw_week = forecast.get("week_change_percent", 0)
    raw_month = forecast.get("month_change_percent", 0)

    clamped_week = max(min(raw_week, max_return), -max_return)
    clamped_month = max(min(raw_month, max_return), -max_return)

    current_price = forecast.get("current_price", 0)

    forecast["raw_week_change_percent"] = raw_week
    forecast["raw_month_change_percent"] = raw_month

    forecast["week_change_percent"] = round(clamped_week, 2)
    forecast["month_change_percent"] = round(clamped_month, 2)

    if current_price > 0:
        forecast["predicted_week_price"] = round(
            current_price * (1 + clamped_week / 100),
            2
        )

        forecast["predicted_month_price"] = round(
            current_price * (1 + clamped_month / 100),
            2
        )

    forecast["forecast_note"] = (
        "Forecast return was capped to avoid unrealistic recommendation scoring."
        if raw_week != clamped_week or raw_month != clamped_month
        else "Forecast return is within acceptable range."
    )

    return forecast


def calculate_final_score(bert_score, forecast, event_features):
    week_return = forecast.get("week_change_percent", 0)
    month_return = forecast.get("month_change_percent", 0)
    confidence = forecast.get("direction_confidence", 0)

    avg_return = (week_return + month_return) / 2

    forecast_score = (avg_return + 25) / 50
    forecast_score = max(0, min(1, forecast_score))

    confidence_score = confidence / 100

    dividend_score = 0

    if event_features.get("dividend_event", 0) == 1:
        dividend_score += 0.4

    if event_features.get("dividend_percent", 0) > 0:
        dividend_score += min(event_features["dividend_percent"] / 50, 0.4)

    if event_features.get("bonus_event", 0) == 1:
        dividend_score += 0.1

    if event_features.get("dividend_season", 0) == 1:
        dividend_score += 0.1

    dividend_score = min(dividend_score, 1)

    activity_score = 1 if event_features.get("high_activity_day", 0) == 1 else 0.3

    price_score = calculate_price_attractiveness(
        event_features.get("current_price", 0),
        event_features.get("recent_min", 0),
        event_features.get("recent_max", 0)
    )

    final_score = (
        0.25 * bert_score
        + 0.25 * forecast_score
        + 0.20 * confidence_score
        + 0.15 * dividend_score
        + 0.10 * price_score
        + 0.05 * activity_score
    )

    return round(final_score, 4)


def hybrid_recommend_one_stock(user_id):
    bert_recommendations = recommend_stocks(
        user_id=user_id,
        top_k=5
    )

    if not bert_recommendations:
        return {
            "message": "No recommendation available. User history is too limited."
        }

    candidates = []

    for item in bert_recommendations:
        company = item["company"]
        bert_score = item["score"]

        csv_path = get_latest_raw_csv(company)

        if csv_path is None:
            continue

        try:
            forecast = predict_company(
                 company=company,
                 csv_path=get_latest_raw_csv(company)
            )

            # =====================================================
            # FIX EXTREME FORECAST VALUES
            # =====================================================

            forecast = clamp_forecast_output(forecast)

            week_change = float(
                 forecast.get("week_change_percent", 0)
            )

            # prevent unrealistic values
            week_change = max(min(week_change, 15), -15)

            forecast["week_change_percent"] = round(
                 week_change,
                 2
            )

            final_score = (
    float(item.get("score", 0))
    + float(forecast.get("direction_confidence", 0)) / 100
    + max(week_change, 0) / 100
) / 3

            event_features = get_latest_event_features(company)

            final_score = calculate_final_score(
                bert_score=bert_score,
                forecast=forecast,
                event_features=event_features
            )

            candidates.append({
                "company": company,
                "final_score": final_score,
                "bert_score": bert_score,
                "forecast": forecast,
                "event_features": event_features
            })

        except Exception as e:
            print(f"[ERROR] {company}: {e}")

    if not candidates:
        return {
            "message": "No valid recommendation could be generated."
        }

    best = sorted(
        candidates,
        key=lambda x: x["final_score"],
        reverse=True
    )[0]

    result = {
        "recommended_company": best["company"],
        "final_score": best["final_score"],
        "reason": {
            "personalized_interest_score": best["bert_score"],
            "predicted_direction": best["forecast"]["predicted_direction"],
            "direction_confidence": best["forecast"]["direction_confidence"],
            "week_change_percent": best["forecast"]["week_change_percent"],
            "month_change_percent": best["forecast"]["month_change_percent"],
            "raw_week_change_percent": best["forecast"].get("raw_week_change_percent"),
            "raw_month_change_percent": best["forecast"].get("raw_month_change_percent"),
            "dividend_percent": best["event_features"].get("dividend_percent", 0),
            "dividend_season": best["event_features"].get("dividend_season", 0),
            "price_attractiveness": calculate_price_attractiveness(
                best["event_features"].get("current_price", 0),
                best["event_features"].get("recent_min", 0),
                best["event_features"].get("recent_max", 0)
            )
        },
        "forecast": best["forecast"]
    }

    return result

def hybrid_recommend_top_stocks(
    user_id,
    top_k=5
):
    base_recommendations = recommend_stocks(
        user_id=user_id,
        top_k=top_k
    )

    results = []

    for item in base_recommendations:
        company = item.get("company")

        try:
            forecast = predict_company(
                company=company,
                csv_path=get_latest_raw_csv(company)
            )

            # FIX: cap unrealistic week/month percentage
            forecast = clamp_forecast_output(
                forecast,
                max_return=15
            )

            week_change = float(
                forecast.get("week_change_percent", 0)
            )

            month_change = float(
                forecast.get("month_change_percent", 0)
            )

            week_change = max(
                min(week_change, 15),
                -15
            )

            month_change = max(
                min(month_change, 15),
                -15
            )

            forecast["week_change_percent"] = round(
                week_change,
                2
            )

            forecast["month_change_percent"] = round(
                month_change,
                2
            )

            final_score = (
                float(item.get("score", 0))
                + float(forecast.get("direction_confidence", 0)) / 100
                + max(week_change, 0) / 100
            ) / 3

            results.append({
                "recommended_company": company,
                "final_score": round(final_score, 4),
                "reason": {
                    "personalized_interest_score": item.get("score", 0),
                    "predicted_direction": forecast.get("predicted_direction"),
                    "direction_confidence": forecast.get("direction_confidence"),
                    "week_change_percent": forecast.get("week_change_percent"),
                    "month_change_percent": forecast.get("month_change_percent"),
                },
                "forecast": forecast
            })

        except Exception as e:
            print(f"ERROR recommending {company}: {e}")

    ranked = sorted(
        results,
        key=lambda x: x["final_score"],
        reverse=True
    )

    return ranked[:top_k]

def get_latest_raw_csv(company):
    import os

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    company = company.upper()

    raw_company_dir = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        company
    )

    if not os.path.exists(raw_company_dir):
        return None

    csv_files = [
        os.path.join(raw_company_dir, f)
        for f in os.listdir(raw_company_dir)
        if f.endswith(".csv")
    ]

    if not csv_files:
        return None

    return max(csv_files, key=os.path.getmtime)

    return ranked[:top_k]
if __name__ == "__main__":
    result = hybrid_recommend_top_stocks("demo_user", top_k=5)
    print(json.dumps(result, indent=4))