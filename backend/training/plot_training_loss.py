import os
import json
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

HISTORY_PATH = os.path.join(
    MODEL_DIR,
    "event_training_history.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training",
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


plt.rcParams["figure.figsize"] = (10, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


def save_plot(filename):

    save_path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")


def plot_model_history(model_name, history):

    epochs = [x["epoch"] for x in history]

    train_loss = [x["train_loss"] for x in history]

    test_loss = [x["test_loss"] for x in history]

    learning_rate = [x["learning_rate"] for x in history]

    # =====================================================
    # TRAIN VS TEST LOSS
    # =====================================================

    plt.figure()

    plt.plot(
        epochs,
        train_loss,
        linewidth=1.8,
        label="Training Loss"
    )

    plt.plot(
        epochs,
        test_loss,
        linewidth=1.8,
        linestyle="--",
        label="Validation Loss"
    )

    plt.title(
        f"{model_name} Training vs Validation Loss",
        fontsize=14,
        pad=12
    )

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.grid(
        True,
        axis="y",
        alpha=0.25
    )

    plt.legend(frameon=False)

    save_plot(
        f"{model_name.lower()}_training_loss.png"
    )

    # =====================================================
    # LEARNING RATE
    # =====================================================

    plt.figure()

    plt.plot(
        epochs,
        learning_rate,
        linewidth=1.8
    )

    plt.title(
        f"{model_name} Learning Rate Schedule",
        fontsize=14,
        pad=12
    )

    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")

    plt.grid(
        True,
        axis="y",
        alpha=0.25
    )

    save_plot(
        f"{model_name.lower()}_learning_rate.png"
    )


def main():

    with open(HISTORY_PATH, "r") as f:
        histories = json.load(f)

    for model_name, history in histories.items():

        print("=" * 60)
        print(f"PLOTTING {model_name}")
        print("=" * 60)

        plot_model_history(
            model_name,
            history
        )


if __name__ == "__main__":
    main()