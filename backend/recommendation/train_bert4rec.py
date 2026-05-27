import os
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

BERT_DIR = os.path.join(BASE_DIR, "data", "bert4rec")
MODEL_DIR = os.path.join(BASE_DIR, "models")

os.makedirs(MODEL_DIR, exist_ok=True)

SEQUENCES_FILE = os.path.join(BERT_DIR, "user_sequences.json")
ITEM_MAP_FILE = os.path.join(BERT_DIR, "company_item_map.json")

MODEL_PATH = os.path.join(MODEL_DIR, "bert4rec_model.pt")
INFO_PATH = os.path.join(MODEL_DIR, "bert4rec_info.json")


MAX_LEN = 20
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
DROPOUT = 0.2
EPOCHS = 50
LR = 0.001

PAD_ID = 0
MASK_ID = 18

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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

        x = self.item_embedding(x) + self.position_embedding(positions)

        encoded = self.encoder(x)

        logits = self.output(encoded)

        return logits


def load_data():
    with open(SEQUENCES_FILE, "r", encoding="utf-8") as f:
        sequences = json.load(f)

    with open(ITEM_MAP_FILE, "r", encoding="utf-8") as f:
        item_map = json.load(f)

    return sequences, item_map


def pad_sequence(seq):
    seq = seq[-MAX_LEN:]

    padding = [PAD_ID] * (MAX_LEN - len(seq))

    return padding + seq


def create_training_samples(sequences):
    samples = []

    for user_id, seq in sequences.items():
        if len(seq) < 2:
            continue

        padded = pad_sequence(seq)

        for i in range(len(padded)):
            if padded[i] == PAD_ID:
                continue

            input_seq = padded.copy()
            target = input_seq[i]
            input_seq[i] = MASK_ID

            samples.append((input_seq, i, target))

    return samples


def train_bert4rec():
    print("=" * 70)
    print("BERT4REC TRAINING STARTED")
    print("=" * 70)

    sequences, item_map = load_data()

    num_items = len(item_map)

    samples = create_training_samples(sequences)

    if not samples:
        raise ValueError("Not enough user activity data to train BERT4Rec.")

    model = BERT4Rec(num_items).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LR)

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):
        random.shuffle(samples)

        losses = []

        model.train()

        for input_seq, mask_pos, target in samples:
            x = torch.tensor(
                [input_seq],
                dtype=torch.long
            ).to(DEVICE)

            y = torch.tensor(
                [target],
                dtype=torch.long
            ).to(DEVICE)

            optimizer.zero_grad()

            logits = model(x)

            masked_logits = logits[:, mask_pos, :]

            loss = loss_fn(masked_logits, y)

            loss.backward()

            optimizer.step()

            losses.append(loss.item())

        avg_loss = sum(losses) / len(losses)

        print(f"Epoch {epoch + 1}/{EPOCHS} | Loss: {avg_loss:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_items": num_items,
            "item_map": item_map,
            "max_len": MAX_LEN,
            "mask_id": MASK_ID,
            "pad_id": PAD_ID,
            "embed_dim": EMBED_DIM
        },
        MODEL_PATH
    )

    info = {
        "model": "BERT4Rec",
        "model_path": MODEL_PATH,
        "num_items": num_items,
        "max_len": MAX_LEN,
        "description": "Personalized stock recommender trained on user dashboard, prediction, watchlist, and chat behavior sequences."
    }

    with open(INFO_PATH, "w", encoding="utf-8") as f:
        json.dump(info, f, indent=4)

    print("\nSaved:")
    print(MODEL_PATH)
    print(INFO_PATH)


if __name__ == "__main__":
    train_bert4rec()