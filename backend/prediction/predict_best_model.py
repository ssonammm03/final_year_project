import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

import sys


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

sys.path.append(BASE_DIR)

from features.event_feature_engineer import process_company_file


# =========================================================
# PATHS
# =========================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "best_event_global_model.pt")
SCALER_PATH = os.path.join(MODEL_DIR, "event_global_scaler.pkl")
DIRECTION_MODEL_PATH = os.path.join(MODEL_DIR, "direction_classifier.pkl")
DIRECTION_SCALER_PATH = os.path.join(MODEL_DIR, "direction_classifier_scaler.pkl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# TiDE MODEL
# =========================================================

class TiDEModel(nn.Module):

    def __init__(
        self,
        input_dim,
        num_companies,
        seq_len=30,
        output_dim=2
    ):
        super().__init__()

        self.company_embedding = nn.Embedding(
            num_companies,
            8
        )

        self.model = nn.Sequential(

            nn.Linear(
                input_dim * seq_len + 8,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.15),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Linear(64, output_dim)
        )

    def forward(self, x, company_id):

        x = x.flatten(start_dim=1)

        company_emb = self.company_embedding(company_id)

        x = torch.cat([x, company_emb], dim=1)

        return self.model(x)


# =========================================================
# LOAD MODEL
# =========================================================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE
)

feature_cols = checkpoint["feature_cols"]

company_map = checkpoint["company_map"]

seq_len = checkpoint["seq_len"]

input_dim = checkpoint["input_dim"]

num_companies = checkpoint["num_companies"]

model = TiDEModel(
    input_dim=input_dim,
    num_companies=num_companies,
    seq_len=seq_len,
    output_dim=2
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)

model.eval()

# =========================================================
# LOAD SCALERS
# =========================================================

scaler = joblib.load(SCALER_PATH)

direction_model = joblib.load(
    DIRECTION_MODEL_PATH
)

direction_scaler = joblib.load(
    DIRECTION_SCALER_PATH
)


# =========================================================
# LABEL MAP
# =========================================================

LABEL_MAP = {
    0: "DOWN",
    1: "SAME",
    2: "UP"
}


# =========================================================
# PREDICT
# =========================================================

def predict_company(csv_path, company):

    df = process_company_file(
        csv_path,
        company
    )

    latest_df = df.tail(seq_len).copy()

    X = latest_df[feature_cols]

    X = (
        X.replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # =====================================================
    # PRICE FORECAST
    # =====================================================

    X_scaled = scaler.transform(X.values)

    X_tensor = torch.tensor(
        X_scaled,
        dtype=torch.float32
    ).unsqueeze(0).to(DEVICE)

    company_id = torch.tensor(
        [company_map[company]],
        dtype=torch.long
    ).to(DEVICE)

    with torch.no_grad():

        preds = model(
            X_tensor,
            company_id
        ).cpu().numpy()[0]

    week_price = float(preds[0])

    month_price = float(preds[1])

    # =====================================================
    # DIRECTION CLASSIFIER
    # =====================================================

    direction_input = latest_df[
        feature_cols
    ].iloc[-1:]

    direction_input = (
        direction_input
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    direction_scaled = direction_scaler.transform(
        direction_input.values
    )

    direction_class = direction_model.predict(
        direction_scaled
    )[0]

    direction_probs = direction_model.predict_proba(
        direction_scaled
    )[0]

    confidence = float(
        np.max(direction_probs) * 100
    )

    direction_label = LABEL_MAP[
        direction_class
    ]

    # =====================================================
    # CURRENT PRICE
    # =====================================================

    current_price = float(
        latest_df["Close"].iloc[-1]
    )

    # =====================================================
    # DIRECTION ADJUSTMENT
    # =====================================================
    week_change = (
        (week_price - current_price)
        / current_price
        ) * 100

    if week_change > 1:
        direction_label = "UP"

    elif week_change < -1:
        direction_label = "DOWN"

    else:
        direction_label = "SAME"

    # =====================================================
    # RESULT
    # =====================================================

    result = {

        "company": company,

        "current_price": round(
            current_price,
            2
        ),

        "predicted_week_price": round(
            week_price,
            2
        ),

        "predicted_month_price": round(
            month_price,
            2
        ),

        "predicted_direction": direction_label,

        "direction_confidence": round(
            confidence,
            2
        ),

        "week_change_percent": round(
            (
                (week_price - current_price)
                / current_price
            ) * 100,
            2
        ),

        "month_change_percent": round(
            (
                (month_price - current_price)
                / current_price
            ) * 100,
            2
        )
    }

    return result


# =========================================================
# MAIN TEST
# =========================================================

if __name__ == "__main__":

    company = "STCB"

    csv_path = os.path.join(
        BASE_DIR,
        "data",
        "raw",
        company,
        os.listdir(
            os.path.join(
                BASE_DIR,
                "data",
                "raw",
                company
            )
        )[-1]
    )

    result = predict_company(
        csv_path,
        company
    )

    print(json.dumps(result, indent=4))