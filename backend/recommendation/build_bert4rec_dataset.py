import os
import json
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

HISTORY_FILE = os.path.join(
    BASE_DIR,
    "data",
    "user_history",
    "user_activity.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "bert4rec"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)

SEQUENCES_FILE = os.path.join(OUTPUT_DIR, "user_sequences.json")
ITEM_MAP_FILE = os.path.join(OUTPUT_DIR, "company_item_map.json")
TRAIN_FILE = os.path.join(OUTPUT_DIR, "bert4rec_train.csv")


COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]


ACTION_WEIGHTS = {
    "overview_view": 1,
    "company_search": 2,
    "prediction_view": 3,
    "chat_query": 4,
    "watchlist_add": 5
}


def load_activity():
    if not os.path.exists(HISTORY_FILE):
        raise ValueError(f"Missing activity file: {HISTORY_FILE}")

    with open(HISTORY_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def build_item_map():
    return {
        company: idx + 1
        for idx, company in enumerate(COMPANIES)
    }


def build_sequences():
    activity = load_activity()

    df = pd.DataFrame(activity)

    if df.empty:
        raise ValueError("No user activity found.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["company"] = df["company"].str.upper()
    df["weight"] = df["action_type"].map(ACTION_WEIGHTS).fillna(1)

    df = df[df["company"].isin(COMPANIES)]
    df = df.dropna(subset=["timestamp"])

    item_map = build_item_map()

    df["item_id"] = df["company"].map(item_map)

    df = df.sort_values(["user_id", "timestamp"])

    user_sequences = {}

    train_rows = []

    for user_id, group in df.groupby("user_id"):
        sequence = []

        for _, row in group.iterrows():
            repeat_count = int(row["weight"])

            for _ in range(repeat_count):
                sequence.append(int(row["item_id"]))

        user_sequences[str(user_id)] = sequence

        for pos, item_id in enumerate(sequence):
            train_rows.append({
                "user_id": str(user_id),
                "position": pos,
                "item_id": item_id
            })

    with open(SEQUENCES_FILE, "w", encoding="utf-8") as f:
        json.dump(user_sequences, f, indent=4)

    with open(ITEM_MAP_FILE, "w", encoding="utf-8") as f:
        json.dump(item_map, f, indent=4)

    train_df = pd.DataFrame(train_rows)
    train_df.to_csv(TRAIN_FILE, index=False)

    print("Saved:")
    print(SEQUENCES_FILE)
    print(ITEM_MAP_FILE)
    print(TRAIN_FILE)

    return user_sequences, item_map


if __name__ == "__main__":
    sequences, item_map = build_sequences()

    print("\nUser sequences:")
    print(json.dumps(sequences, indent=4))

    print("\nItem map:")
    print(json.dumps(item_map, indent=4))