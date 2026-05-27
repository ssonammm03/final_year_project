import os
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    f1_score
)
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "processed_event",
    "all_companies_event_featured.csv"
)

MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(MODEL_DIR, "direction_classifier.pkl")
BEST_SCALER_PATH = os.path.join(MODEL_DIR, "direction_classifier_scaler.pkl")
INFO_PATH = os.path.join(MODEL_DIR, "direction_classifier_info.json")
COMPARISON_CSV = os.path.join(MODEL_DIR, "direction_classifier_comparison.csv")
COMPARISON_JSON = os.path.join(MODEL_DIR, "direction_classifier_comparison.json")

CLASS_NAMES = ["DOWN", "SAME", "UP"]


def load_data():
    df = pd.read_csv(DATA_PATH)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    df = df.dropna(
        subset=[
            "Date",
            "Company",
            "Close",
            "target_class"
        ]
    )

    df["target_class"] = df["target_class"].astype(int)

    return df


def get_feature_columns(df):
    exclude_cols = [
        "Date",
        "Company",
        "target",
        "target_class",
        "future_close_week",
        "future_close_month"
    ]

    return [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]


def chronological_split(df):
    train_frames = []
    test_frames = []

    for company in df["Company"].unique():
        company_df = (
            df[df["Company"] == company]
            .sort_values("Date")
            .reset_index(drop=True)
        )

        split_idx = int(len(company_df) * 0.8)

        train_frames.append(company_df.iloc[:split_idx])
        test_frames.append(company_df.iloc[split_idx:])

    train_df = pd.concat(train_frames, ignore_index=True)
    test_df = pd.concat(test_frames, ignore_index=True)

    return train_df, test_df


def get_models():
    return {
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            random_state=42
        ),

        "XGBoost": XGBClassifier(
            n_estimators=400,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            objective="multi:softprob",
            num_class=3,
            eval_metric="mlogloss",
            random_state=42,
            n_jobs=-1
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=400,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            objective="multiclass",
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        )
    }


def train_best_direction_classifier():
    print("=" * 70)
    print("BEST DIRECTION CLASSIFIER SELECTION STARTED")
    print("=" * 70)

    df = load_data()

    train_df, test_df = chronological_split(df)

    feature_cols = get_feature_columns(train_df)

    X_train = (
        train_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    X_test = (
        test_df[feature_cols]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    y_train = train_df["target_class"]
    y_test = test_df["target_class"]

    models = get_models()

    results = []
    trained_objects = {}

    for model_name, classifier in models.items():
        print("\n" + "=" * 70)
        print(f"TRAINING {model_name}")
        print("=" * 70)

        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        classifier.fit(X_train_scaled, y_train)

        preds = classifier.predict(X_test_scaled)

        accuracy = accuracy_score(y_test, preds)
        macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
        weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        report = classification_report(
            y_test,
            preds,
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0
        )

        matrix = confusion_matrix(y_test, preds).tolist()

        record = {
            "model": model_name,
            "accuracy": round(float(accuracy), 4),
            "accuracy_percent": round(float(accuracy * 100), 2),
            "macro_f1": round(float(macro_f1), 4),
            "weighted_f1": round(float(weighted_f1), 4),
            "down_f1": round(float(report["DOWN"]["f1-score"]), 4),
            "same_f1": round(float(report["SAME"]["f1-score"]), 4),
            "up_f1": round(float(report["UP"]["f1-score"]), 4),
            "selection_score": round(float((macro_f1 * 0.6) + (weighted_f1 * 0.4)), 4)
        }

        results.append(record)

        trained_objects[model_name] = {
            "model": classifier,
            "scaler": scaler,
            "report": report,
            "confusion_matrix": matrix,
            "predictions": preds
        }

        print(f"Accuracy: {record['accuracy_percent']}%")
        print(f"Macro F1: {record['macro_f1']}")
        print(f"Weighted F1: {record['weighted_f1']}")
        print(f"Selection Score: {record['selection_score']}")

    comparison_df = pd.DataFrame(results)

    best_idx = comparison_df["selection_score"].idxmax()
    best_model_name = comparison_df.loc[best_idx, "model"]

    comparison_df["selected"] = "No"
    comparison_df.loc[best_idx, "selected"] = "Yes"

    best_obj = trained_objects[best_model_name]

    joblib.dump(best_obj["model"], BEST_MODEL_PATH)
    joblib.dump(best_obj["scaler"], BEST_SCALER_PATH)

    comparison_df.to_csv(COMPARISON_CSV, index=False)

    with open(COMPARISON_JSON, "w", encoding="utf-8") as f:
        json.dump(
            comparison_df.to_dict(orient="records"),
            f,
            indent=4
        )

    best_info = {
        "model": best_model_name,
        "accuracy": round(float(comparison_df.loc[best_idx, "accuracy"]), 4),
        "accuracy_percent": round(float(comparison_df.loc[best_idx, "accuracy_percent"]), 2),
        "macro_f1": round(float(comparison_df.loc[best_idx, "macro_f1"]), 4),
        "weighted_f1": round(float(comparison_df.loc[best_idx, "weighted_f1"]), 4),
        "selection_metric": "0.6 * macro_f1 + 0.4 * weighted_f1",
        "selection_score": round(float(comparison_df.loc[best_idx, "selection_score"]), 4),
        "feature_columns": feature_cols,
        "class_mapping": {
            "DOWN": 0,
            "SAME": 1,
            "UP": 2
        },
        "classification_report": best_obj["report"],
        "confusion_matrix": best_obj["confusion_matrix"],
        "comparison_csv": COMPARISON_CSV,
        "comparison_json": COMPARISON_JSON,
        "model_path": BEST_MODEL_PATH,
        "scaler_path": BEST_SCALER_PATH,
        "description": "Best selected UP/SAME/DOWN classifier using GradientBoosting, XGBoost, and LightGBM on event-aware dividend-reaction features."
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(best_info, f, indent=4)

    print("\n" + "=" * 70)
    print(f"BEST DIRECTION CLASSIFIER SELECTED: {best_model_name}")
    print("=" * 70)
    print(comparison_df)

    print("\nSaved:")
    print(BEST_MODEL_PATH)
    print(BEST_SCALER_PATH)
    print(INFO_PATH)
    print(COMPARISON_CSV)

    return best_info


if __name__ == "__main__":
    result = train_best_direction_classifier()

    print("\nTRAINING COMPLETE")
    print(json.dumps(result, indent=4))