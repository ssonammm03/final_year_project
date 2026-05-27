import os
import json
import torch
import torch.nn as nn
import numpy as np

import sys

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from recommendation.user_activity_logger import (
    get_user_activity
)

# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "bert4rec_model.pt"
)

COMPANY_MAP_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bert4rec",
    "company_item_map.json"
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# LOAD COMPANY MAP
# =========================================================

with open(COMPANY_MAP_PATH, "r", encoding="utf-8") as f:
    COMPANY_MAP = json.load(f)

ID_TO_COMPANY = {
    v: k for k, v in COMPANY_MAP.items()
}


# =========================================================
# MODEL CONFIG
# =========================================================

MAX_LEN = 20
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2

PAD_ID = 0
MASK_ID = 18


# =========================================================
# BERT4REC MODEL
# =========================================================

class BERT4Rec(nn.Module):

    def __init__(self, num_items):
        super().__init__()

        self.item_embedding = nn.Embedding(
            num_items + 2,
            EMBED_DIM,
            padding_idx=PAD_ID
        )

        self.position_embedding = nn.Embedding(
            MAX_LEN,
            EMBED_DIM
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM,
            nhead=NUM_HEADS,
            dim_feedforward=128,
            dropout=DROPOUT,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=NUM_LAYERS
        )

        self.output = nn.Linear(
            EMBED_DIM,
            num_items + 1
        )

    def forward(self, x):

        batch_size, seq_len = x.shape

        positions = torch.arange(
            seq_len,
            device=x.device
        ).unsqueeze(0).repeat(batch_size, 1)

        x = (
            self.item_embedding(x)
            + self.position_embedding(positions)
        )

        encoded = self.encoder(x)

        logits = self.output(encoded)

        return logits


# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

num_items = checkpoint["num_items"]

model = BERT4Rec(num_items)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()


# =========================================================
# BUILD USER SEQUENCE
# =========================================================

ACTION_WEIGHTS = {
    "overview_view": 1,
    "company_search": 2,
    "prediction_view": 3,
    "chat_query": 4,
    "watchlist_add": 5
}


def build_user_sequence(user_id):

    activity = get_user_activity(user_id)

    activity = sorted(
        activity,
        key=lambda x: x["timestamp"]
    )

    sequence = []

    for item in activity:

        company = item["company"].upper()

        if company not in COMPANY_MAP:
            continue

        item_id = COMPANY_MAP[company]

        weight = ACTION_WEIGHTS.get(
            item["action_type"],
            1
        )

        for _ in range(weight):
            sequence.append(item_id)

    return sequence


# =========================================================
# PAD SEQUENCE
# =========================================================

def pad_sequence(seq):

    seq = seq[-MAX_LEN:]

    padding = [PAD_ID] * (MAX_LEN - len(seq))

    return padding + seq


# =========================================================
# RECOMMEND
# =========================================================

def recommend_stocks(user_id, top_k=5):

    sequence = build_user_sequence(user_id)

    if not sequence:
        return []

    padded = pad_sequence(sequence)

    padded[-1] = MASK_ID

    x = torch.tensor(
        [padded],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        logits = model(x)

        scores = logits[0, -1]

        probs = torch.softmax(
            scores,
            dim=0
        ).cpu().numpy()

    ranked = np.argsort(probs)[::-1]

    recommendations = []

    already_seen = set(sequence)

    for item_id in ranked:

        if item_id in already_seen:
            continue

        if item_id == PAD_ID:
            continue

        if item_id == MASK_ID:
            continue

        if item_id not in ID_TO_COMPANY:
            continue

        recommendations.append({
            "company": ID_TO_COMPANY[item_id],
            "score": round(float(probs[item_id]), 4)
        })

        if len(recommendations) >= top_k:
            break

    return recommendations


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    recs = recommend_stocks(
        user_id="demo_user",
        top_k=5
    )

    print(json.dumps(recs, indent=4))