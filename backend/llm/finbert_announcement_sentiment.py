import os
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

EVENT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "events",
    "dividend_events.csv"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "events",
    "announcement_sentiment.csv"
)


MODEL_NAME = "ProsusAI/finbert"

LABEL_SCORE = {
    "positive": 1.0,
    "neutral": 0.0,
    "negative": -1.0
}


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)

    model.eval()

    return tokenizer, model


def analyze_sentiment(text, tokenizer, model):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)[0]

    label_id = torch.argmax(probs).item()

    label = model.config.id2label[label_id].lower()

    sentiment_score = LABEL_SCORE.get(label, 0.0)

    confidence = float(probs[label_id])

    event_importance_score = min(
        1.0,
        max(
            0.3,
            confidence
        )
    )

    return {
        "sentiment_label": label,
        "sentiment_score": sentiment_score,
        "sentiment_confidence": round(confidence, 4),
        "event_importance_score": round(event_importance_score, 4)
    }


def run_finbert_sentiment():
    df = pd.read_csv(EVENT_FILE)

    tokenizer, model = load_model()

    results = []

    for _, row in df.iterrows():
        company = row.get("company")
        title = row.get("title")
        raw_text = row.get("raw_text")

        text = f"""
Company: {company}
Title: {title}
Dividend Percent: {row.get("dividend_percent")}
Bonus Ratio: {row.get("bonus_ratio")}
Announcement: {raw_text}
"""

        print(f"Analyzing: {company} - {title}")

        sentiment = analyze_sentiment(
            text,
            tokenizer,
            model
        )

        results.append({
            "company": company,
            "announcement_date": row.get("announcement_date"),
            "title": title,
            "sentiment_label": sentiment["sentiment_label"],
            "sentiment_score": sentiment["sentiment_score"],
            "sentiment_confidence": sentiment["sentiment_confidence"],
            "event_importance_score": sentiment["event_importance_score"]
        })

    out_df = pd.DataFrame(results)

    out_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_finbert_sentiment()