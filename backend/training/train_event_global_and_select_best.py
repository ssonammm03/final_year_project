import os
import json
import glob
import argparse
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import joblib

from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

PROCESSED_EVENT_DIR = os.path.join(BASE_DIR, "data", "processed_event")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

COMPANIES = [
    "BBPL", "BCCL", "BFAL", "BIL", "BNBL",
    "BPCL", "BTCL", "DFAL", "DPL", "DPNB",
    "DWAL", "GICB", "KCL", "PCAL", "RICB",
    "STCB", "TBL"
]

SEQ_LEN = 30
BATCH_SIZE = 32
EPOCHS = 30
LR = 0.001
PATIENCE = 5

COMPANY_EMBED_DIM = 8

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class GlobalStockDataset(Dataset):
    def __init__(self, X, y_week, y_month, close_values, company_ids, seq_len):
        self.X = X
        self.y_week = y_week
        self.y_month = y_month
        self.close_values = close_values
        self.company_ids = company_ids
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len

    def __getitem__(self, idx):
        x = self.X[idx:idx + self.seq_len]
        y_week = self.y_week[idx + self.seq_len]
        y_month = self.y_month[idx + self.seq_len]
        last_close = self.close_values[idx + self.seq_len - 1]
        company_id = self.company_ids[idx + self.seq_len - 1]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor([y_week, y_month], dtype=torch.float32),
            torch.tensor(last_close, dtype=torch.float32),
            torch.tensor(company_id, dtype=torch.long)
        )


class PatchTSTModel(nn.Module):
    def __init__(self, input_dim, num_companies, seq_len=30, patch_size=5, d_model=96, n_heads=4, num_layers=2, output_dim=2):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = seq_len // patch_size

        self.company_embedding = nn.Embedding(num_companies, COMPANY_EMBED_DIM)

        self.patch_embedding = nn.Linear(input_dim * patch_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=192,
            dropout=0.15,
            batch_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.head = nn.Sequential(
            nn.LayerNorm(d_model + COMPANY_EMBED_DIM),
            nn.Linear(d_model + COMPANY_EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, company_id):
        batch_size, seq_len, features = x.shape

        usable_len = self.num_patches * self.patch_size
        x = x[:, :usable_len, :]

        x = x.reshape(
            batch_size,
            self.num_patches,
            self.patch_size * features
        )

        x = self.patch_embedding(x)
        x = self.encoder(x)
        x = x.mean(dim=1)

        company_emb = self.company_embedding(company_id)

        x = torch.cat([x, company_emb], dim=1)

        return self.head(x)


class TiDEModel(nn.Module):
    def __init__(self, input_dim, num_companies, seq_len=30, output_dim=2):
        super().__init__()

        self.company_embedding = nn.Embedding(num_companies, COMPANY_EMBED_DIM)

        self.model = nn.Sequential(
            nn.Linear(input_dim * seq_len + COMPANY_EMBED_DIM, 256),
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


class TFTModel(nn.Module):
    def __init__(self, input_dim, num_companies, hidden_dim=64, output_dim=2):
        super().__init__()

        self.company_embedding = nn.Embedding(num_companies, COMPANY_EMBED_DIM)

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.15
        )

        self.attention = nn.Linear(hidden_dim, 1)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim + COMPANY_EMBED_DIM, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, output_dim)
        )

    def forward(self, x, company_id):
        output, _ = self.lstm(x)

        weights = torch.softmax(
            self.attention(output),
            dim=1
        )

        context = torch.sum(weights * output, dim=1)

        company_emb = self.company_embedding(company_id)

        x = torch.cat([context, company_emb], dim=1)

        return self.head(x)


def load_event_data():
    frames = []

    for company in COMPANIES:
        path = os.path.join(
            PROCESSED_EVENT_DIR,
            f"{company}_event_featured.csv"
        )

        if not os.path.exists(path):
            print(f"[SKIPPED] Missing file for {company}")
            continue

        df = pd.read_csv(path)

        df["Company"] = company
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

        df = df.dropna(subset=["Date", "Close", "future_close_week", "future_close_month"])

        frames.append(df)

        print(f"[LOADED] {company}: {len(df)} rows")

    if not frames:
        raise ValueError("No processed event data found.")

    combined = pd.concat(frames, ignore_index=True)

    combined_path = os.path.join(
        PROCESSED_EVENT_DIR,
        "all_companies_event_featured.csv"
    )

    combined.to_csv(combined_path, index=False)

    return combined, combined_path


def chronological_train_test_split(df):
    train_frames = []
    test_frames = []

    for company in df["Company"].unique():
        company_df = df[df["Company"] == company].sort_values("Date").reset_index(drop=True)

        split_idx = int(len(company_df) * 0.8)

        train_frames.append(company_df.iloc[:split_idx])
        test_frames.append(company_df.iloc[split_idx:])

    return (
        pd.concat(train_frames, ignore_index=True),
        pd.concat(test_frames, ignore_index=True)
    )


def get_feature_columns(df):
    exclude_cols = [
        "Date",
        "Company",
        "target",
        "target_class",
        "future_close_week",
        "future_close_month"
    ]

    feature_cols = [
        col for col in df.columns
        if col not in exclude_cols
        and pd.api.types.is_numeric_dtype(df[col])
    ]

    return feature_cols


def prepare_ml_arrays(df, feature_cols, scaler=None, fit_scaler=False, company_map=None):
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0).values

    y_week = df["future_close_week"].values
    y_month = df["future_close_month"].values
    close_values = df["Close"].values

    if company_map is None:
        company_codes = sorted(df["Company"].unique())
        company_map = {company: idx for idx, company in enumerate(company_codes)}

    company_ids = df["Company"].map(company_map).values

    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, y_week, y_month, close_values, company_ids, scaler, company_map


def train_one_model(model_name, model, train_loader, test_loader):
    model = model.to(DEVICE)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=1e-5
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    loss_fn = nn.MSELoss()

    best_test_loss = float("inf")
    best_state = None
    patience_counter = 0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for X_batch, y_batch, _, company_id_batch in train_loader:
            X_batch = X_batch.to(DEVICE)
            y_batch = y_batch.to(DEVICE)
            company_id_batch = company_id_batch.to(DEVICE)

            optimizer.zero_grad()

            preds = model(X_batch, company_id_batch)

            loss = loss_fn(preds, y_batch)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        test_losses = []

        with torch.no_grad():
            for X_batch, y_batch, _, company_id_batch in test_loader:
                X_batch = X_batch.to(DEVICE)
                y_batch = y_batch.to(DEVICE)
                company_id_batch = company_id_batch.to(DEVICE)

                preds = model(X_batch, company_id_batch)

                loss = loss_fn(preds, y_batch)

                test_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        test_loss = float(np.mean(test_losses))

        scheduler.step(test_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "test_loss": test_loss,
            "learning_rate": current_lr
        })

        print(
            f"{model_name} | Epoch {epoch + 1}/{EPOCHS} "
            f"| Train Loss: {train_loss:.6f} "
            f"| Test Loss: {test_loss:.6f} "
            f"| LR: {current_lr:.6f}"
        )

        if test_loss < best_test_loss:
            best_test_loss = test_loss
            best_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered for {model_name}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, history


def evaluate_model(model, loader):
    model.eval()

    preds = []
    actuals = []
    last_closes = []

    with torch.no_grad():
        for X_batch, y_batch, last_close_batch, company_id_batch in loader:
            X_batch = X_batch.to(DEVICE)
            company_id_batch = company_id_batch.to(DEVICE)

            batch_preds = model(X_batch, company_id_batch).cpu().numpy()

            preds.extend(batch_preds)
            actuals.extend(y_batch.numpy())
            last_closes.extend(last_close_batch.numpy())

    preds = np.array(preds)
    actuals = np.array(actuals)
    last_closes = np.array(last_closes)

    week_preds = preds[:, 0]
    month_preds = preds[:, 1]

    week_actuals = actuals[:, 0]
    month_actuals = actuals[:, 1]

    week_mae = mean_absolute_error(week_actuals, week_preds)
    week_rmse = np.sqrt(mean_squared_error(week_actuals, week_preds))

    month_mae = mean_absolute_error(month_actuals, month_preds)
    month_rmse = np.sqrt(mean_squared_error(month_actuals, month_preds))

    week_direction_accuracy = (
        np.sign(week_actuals - last_closes)
        ==
        np.sign(week_preds - last_closes)
    ).mean() * 100

    month_direction_accuracy = (
        np.sign(month_actuals - last_closes)
        ==
        np.sign(month_preds - last_closes)
    ).mean() * 100

    combined_score = (
        week_direction_accuracy * 0.40
        + month_direction_accuracy * 0.25
        - week_rmse * 0.20
        - month_rmse * 0.15
    )

    return {
        "week_mae": round(float(week_mae), 6),
        "week_rmse": round(float(week_rmse), 6),
        "week_direction_accuracy": round(float(week_direction_accuracy), 2),
        "month_mae": round(float(month_mae), 6),
        "month_rmse": round(float(month_rmse), 6),
        "month_direction_accuracy": round(float(month_direction_accuracy), 2),
        "combined_score": round(float(combined_score), 6)
    }


def train_event_global_and_select_best():
    print("=" * 70)
    print("EVENT-AWARE GLOBAL MODEL TRAINING STARTED")
    print("=" * 70)

    combined_df, combined_path = load_event_data()

    train_df, test_df = chronological_train_test_split(combined_df)

    feature_cols = get_feature_columns(train_df)

    company_codes = sorted(combined_df["Company"].unique())
    company_map = {company: idx for idx, company in enumerate(company_codes)}

    X_train, y_week_train, y_month_train, close_train, company_ids_train, scaler, company_map = prepare_ml_arrays(
        train_df,
        feature_cols,
        fit_scaler=True,
        company_map=company_map
    )

    X_test, y_week_test, y_month_test, close_test, company_ids_test, _, _ = prepare_ml_arrays(
        test_df,
        feature_cols,
        scaler=scaler,
        fit_scaler=False,
        company_map=company_map
    )

    train_dataset = GlobalStockDataset(
        X_train,
        y_week_train,
        y_month_train,
        close_train,
        company_ids_train,
        SEQ_LEN
    )

    test_dataset = GlobalStockDataset(
        X_test,
        y_week_test,
        y_month_test,
        close_test,
        company_ids_test,
        SEQ_LEN
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    input_dim = X_train.shape[1]
    num_companies = len(company_map)

    models = {
        "PatchTST": PatchTSTModel(
            input_dim=input_dim,
            num_companies=num_companies,
            seq_len=SEQ_LEN,
            patch_size=5,
            d_model=96,
            n_heads=4,
            num_layers=2,
            output_dim=2
        ),

        "TiDE": TiDEModel(
            input_dim=input_dim,
            num_companies=num_companies,
            seq_len=SEQ_LEN,
            output_dim=2
        ),

        "TFT": TFTModel(
            input_dim=input_dim,
            num_companies=num_companies,
            hidden_dim=64,
            output_dim=2
        )
    }

    all_metrics = []
    histories = {}
    trained_models = {}

    for model_name, model in models.items():
        print("\n" + "=" * 70)
        print(f"TRAINING {model_name}")
        print("=" * 70)

        trained_model, history = train_one_model(
            model_name,
            model,
            train_loader,
            test_loader
        )

        test_metrics = evaluate_model(
            trained_model,
            test_loader
        )

        record = {
            "model": model_name,
            "test_week_mae": test_metrics["week_mae"],
            "test_week_rmse": test_metrics["week_rmse"],
            "test_week_direction_accuracy": test_metrics["week_direction_accuracy"],
            "test_month_mae": test_metrics["month_mae"],
            "test_month_rmse": test_metrics["month_rmse"],
            "test_month_direction_accuracy": test_metrics["month_direction_accuracy"],
            "test_combined_score": test_metrics["combined_score"],
            "selected": "No"
        }

        all_metrics.append(record)
        histories[model_name] = history
        trained_models[model_name] = trained_model

    metrics_df = pd.DataFrame(all_metrics)

    best_idx = metrics_df["test_combined_score"].idxmax()
    best_model_name = metrics_df.loc[best_idx, "model"]
    metrics_df.loc[best_idx, "selected"] = "Yes"

    best_model = trained_models[best_model_name]

    print("\n" + "=" * 70)
    print(f"BEST EVENT-AWARE MODEL SELECTED: {best_model_name}")
    print("=" * 70)

    best_model_path = os.path.join(MODEL_DIR, "best_event_global_model.pt")
    scaler_path = os.path.join(MODEL_DIR, "event_global_scaler.pkl")
    comparison_csv = os.path.join(MODEL_DIR, "event_model_comparison.csv")
    comparison_json = os.path.join(MODEL_DIR, "event_model_comparison.json")
    history_path = os.path.join(MODEL_DIR, "event_training_history.json")
    info_path = os.path.join(MODEL_DIR, "best_event_model_info.json")

    torch.save(
        {
            "model_name": best_model_name,
            "model_state_dict": best_model.state_dict(),
            "input_dim": input_dim,
            "num_companies": num_companies,
            "seq_len": SEQ_LEN,
            "feature_cols": feature_cols,
            "company_map": company_map,
            "company_embed_dim": COMPANY_EMBED_DIM,
            "model_config": {
                "patchtst": {
                    "patch_size": 5,
                    "d_model": 96,
                    "n_heads": 4,
                    "num_layers": 2
                },
                "tide": {},
                "tft": {
                    "hidden_dim": 64
                }
            }
        },
        best_model_path
    )

    joblib.dump(scaler, scaler_path)

    metrics_df.to_csv(comparison_csv, index=False)

    with open(comparison_json, "w") as f:
        json.dump(metrics_df.to_dict(orient="records"), f, indent=4)

    with open(history_path, "w") as f:
        json.dump(histories, f, indent=4)

    best_info = {
        "best_model": best_model_name,
        "model_path": best_model_path,
        "scaler_path": scaler_path,
        "comparison_csv": comparison_csv,
        "comparison_json": comparison_json,
        "training_history": history_path,
        "processed_event_data": combined_path,
        "feature_columns": feature_cols,
        "company_map": company_map,
        "sequence_length": SEQ_LEN,
        "selected_model_metrics": metrics_df.loc[best_idx].to_dict(),
        "description": "Event-aware global model using technical, intraday, dividend, AGM, bonus-share, and market activity features."
    }

    with open(info_path, "w") as f:
        json.dump(best_info, f, indent=4)

    print("\nSaved:")
    print(best_model_path)
    print(scaler_path)
    print(comparison_csv)
    print(info_path)

    return best_info


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS
    )

    args = parser.parse_args()

    EPOCHS = args.epochs

    result = train_event_global_and_select_best()

    print("\nTRAINING COMPLETE")
    print(json.dumps(result, indent=4))