import os
import json
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)


def explain_forecast(forecast_result):
    prompt = f"""
You are a Bhutan stock market analyst.

Explain this AI stock forecast in simple, professional language.

Forecast result:
{json.dumps(forecast_result, indent=2)}

Important context:
- Bhutan stock prices often move strongly during dividend season.
- Many stocks remain flat for long periods.
- The price model predicts future price.
- The direction classifier predicts UP / DOWN / SAME.
- If confidence is low, explain that the signal is uncertain.

Provide:
1. Summary
2. Forecast interpretation
3. Direction confidence explanation
4. Dividend/event-aware comment
5. Investor-friendly conclusion

Do not give guaranteed financial advice.
Use cautious language.
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You explain AI stock forecasts clearly and cautiously for Bhutanese retail investors."
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0.3
    )

    return response.choices[0].message.content


if __name__ == "__main__":
    sample_forecast = {
        "company": "STCB",
        "current_price": 38.5,
        "predicted_week_price": 40.59,
        "predicted_month_price": 40.55,
        "predicted_direction": "SAME",
        "direction_confidence": 37.64,
        "week_change_percent": 5.43,
        "month_change_percent": 5.33
    }

    explanation = explain_forecast(sample_forecast)

    print(explanation)