import math
from collections import Counter


def safe_float(value, default=0):
    try:
        if value is None:
            return default
        if isinstance(value, str):
            value = value.replace("%", "").replace(",", "")
        value = float(value)
        if math.isnan(value):
            return default
        return value
    except Exception:
        return default


def get_latest_prediction_for_company(company, prediction_history):
    company = company.upper()

    records = [
        r for r in prediction_history
        if str(r.get("company", "")).upper() == company
    ]

    if not records:
        return None

    return records[-1]


def calculate_forecast_score(prediction):
    if not prediction:
        return 40

    week = prediction.get("one_week", {})
    month = prediction.get("one_month", {})

    week_return = safe_float(week.get("return_pct"))
    month_return = safe_float(month.get("return_pct"))

    week_direction = str(week.get("direction", "")).upper()
    month_direction = str(month.get("direction", "")).upper()

    score = 50

    if week_direction == "UP":
        score += 15
    elif week_direction == "DOWN":
        score -= 12

    if month_direction == "UP":
        score += 20
    elif month_direction == "DOWN":
        score -= 15

    score += min(max(week_return, -20), 20) * 0.6
    score += min(max(month_return, -30), 30) * 0.5

    return max(0, min(100, score))


def calculate_market_score(summary):
    total_return = safe_float(summary.get("total_return_pct"))
    volatility = safe_float(summary.get("volatility"))
    trend = str(summary.get("trend", "")).upper()

    score = 50

    if trend == "UP":
        score += 15
    elif trend == "DOWN":
        score -= 10

    score += min(max(total_return, -30), 30) * 0.7

    if volatility > 0:
        score -= min(volatility * 10, 12)

    return max(0, min(100, score))


def calculate_behavior_score(company, prediction_history, chat_history):
    company = company.upper()

    prediction_count = sum(
        1 for r in prediction_history
        if str(r.get("company", "")).upper() == company
    )

    chat_count = sum(
        1 for r in chat_history
        if str(r.get("company", "")).upper() == company
    )

    score = 30 + min(prediction_count * 8, 35) + min(chat_count * 5, 35)

    return max(0, min(100, score))


def generate_stock_suggestions(
    market_summary,
    prediction_history,
    chat_history,
    limit=8
):
    suggestions = []

    for summary in market_summary:
        company = str(summary.get("company", "")).upper()

        latest_prediction = get_latest_prediction_for_company(
            company,
            prediction_history
        )

        forecast_score = calculate_forecast_score(latest_prediction)
        market_score = calculate_market_score(summary)
        behavior_score = calculate_behavior_score(
            company,
            prediction_history,
            chat_history
        )

        final_score = (
            forecast_score * 0.50 +
            market_score * 0.30 +
            behavior_score * 0.20
        )

        action = "BUY" if final_score >= 70 else "HOLD"

        suggestions.append({
            "company": company,
            "action": action,
            "score": round(final_score, 2),
            "logo": f"/logos/{company}.png",
            "forecast_score": round(forecast_score, 2),
            "market_score": round(market_score, 2),
            "behavior_score": round(behavior_score, 2)
        })

    suggestions = sorted(
        suggestions,
        key=lambda x: x["score"],
        reverse=True
    )

    return suggestions[:limit]