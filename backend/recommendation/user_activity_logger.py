import os
import json
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ACTIVITY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "user_activity.json"
)


def load_activity():
    if not os.path.exists(ACTIVITY_PATH):
        return []

    try:
        with open(ACTIVITY_PATH, "r", encoding="utf-8") as f:
            return json.load(f)

    except Exception:
        return []


def save_activity(activity):
    os.makedirs(os.path.dirname(ACTIVITY_PATH), exist_ok=True)

    with open(ACTIVITY_PATH, "w", encoding="utf-8") as f:
        json.dump(activity, f, indent=4)


def log_user_activity(
    user_id,
    company,
    action_type,
    source="system"
):
    history = load_activity()

    history.append({
        "user_id": user_id,
        "company": company,
        "action_type": action_type,
        "source": source,
        "timestamp": datetime.now().isoformat()
    })

    save_activity(history)


def get_user_activity(user_id=None):
    history = load_activity()

    if user_id:
        history = [
            h for h in history
            if h.get("user_id") == user_id
        ]

    return history


def get_user_activity_history(
    user_id=None,
    company=None,
    limit=100
):
    history = load_activity()

    if user_id:
        history = [
            h for h in history
            if h.get("user_id") == user_id
        ]

    if company:
        company = company.upper()

        history = [
            h for h in history
            if h.get("company") == company
        ]

    return history[-int(limit):]