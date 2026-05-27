import os
import json
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HISTORY_DIR = os.path.join(BASE_DIR, "prediction_history")
HISTORY_FILE = os.path.join(HISTORY_DIR, "prediction_history.json")

os.makedirs(HISTORY_DIR, exist_ok=True)


def save_prediction_history(prediction_result):
    record = {
        "saved_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        **prediction_result
    }

    history = []

    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, "r") as f:
            try:
                history = json.load(f)
            except Exception:
                history = []

    history.append(record)

    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

    return record


def get_prediction_history(company=None):
    if not os.path.exists(HISTORY_FILE):
        return []

    with open(HISTORY_FILE, "r") as f:
        try:
            history = json.load(f)
        except Exception:
            history = []

    if company:
        company = company.upper()

        history = [
            item for item in history
            if item.get("company") == company
        ]

    return history