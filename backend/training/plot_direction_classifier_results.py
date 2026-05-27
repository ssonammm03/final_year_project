import os
import json
import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "models")

INFO_PATH = os.path.join(
    MODEL_DIR,
    "direction_classifier_info.json"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "training",
    "plots"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


plt.rcParams["figure.figsize"] = (8, 6)
plt.rcParams["font.size"] = 11
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.spines.right"] = False


CLASS_NAMES = ["DOWN", "SAME", "UP"]


def save_plot(filename):
    save_path = os.path.join(OUTPUT_DIR, filename)

    plt.tight_layout()

    plt.savefig(
        save_path,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()

    print(f"Saved: {save_path}")


def load_info():
    with open(INFO_PATH, "r") as f:
        return json.load(f)


def plot_confusion_matrix(info):

    matrix = np.array(info["confusion_matrix"])

    plt.figure(figsize=(7, 6))

    # Aesthetic colormap
    cmap = plt.cm.Blues

    plt.imshow(
        matrix,
        cmap=cmap
    )

    plt.title(
        "Direction Classifier Confusion Matrix",
        fontsize=14,
        pad=12
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")

    plt.xticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    plt.yticks(
        range(len(CLASS_NAMES)),
        CLASS_NAMES
    )

    # Dynamic text color
    threshold = matrix.max() / 2

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):

            color = (
                "white"
                if matrix[i, j] > threshold
                else "black"
            )

            plt.text(
                j,
                i,
                f"{matrix[i, j]}",
                ha="center",
                va="center",
                fontsize=11,
                color=color
            )

    cbar = plt.colorbar()

    cbar.set_label(
        "Number of Records",
        rotation=90
    )

    save_plot(
        "direction_classifier_confusion_matrix.png"
    )


def plot_class_metrics(info):

    report = info["classification_report"]

    precision = [report[c]["precision"] for c in CLASS_NAMES]
    recall = [report[c]["recall"] for c in CLASS_NAMES]
    f1 = [report[c]["f1-score"] for c in CLASS_NAMES]

    x = np.arange(len(CLASS_NAMES))
    width = 0.24

    plt.figure(figsize=(9, 6))

    # Aesthetic pastel colors
    precision_color = "#4C78A8"   # soft blue
    recall_color = "#72B7B2"      # teal
    f1_color = "#F4A261"          # pastel orange

    bars1 = plt.bar(
        x - width,
        precision,
        width,
        label="Precision",
        color=precision_color,
        edgecolor="white",
        linewidth=1
    )

    bars2 = plt.bar(
        x,
        recall,
        width,
        label="Recall",
        color=recall_color,
        edgecolor="white",
        linewidth=1
    )

    bars3 = plt.bar(
        x + width,
        f1,
        width,
        label="F1-score",
        color=f1_color,
        edgecolor="white",
        linewidth=1
    )

    # Add values on top
    for bars in [bars1, bars2, bars3]:

        for bar in bars:

            height = bar.get_height()

            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    plt.xticks(
        x,
        CLASS_NAMES,
        fontsize=11
    )

    plt.yticks(fontsize=10)

    plt.ylim(0, 1.05)

    plt.xlabel(
        "Class",
        fontsize=12
    )

    plt.ylabel(
        "Score",
        fontsize=12
    )

    plt.legend(
        frameon=False,
        fontsize=10
    )

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    save_plot(
        "direction_classifier_class_metrics.png"
    )


def plot_class_distribution(info):

    report = info["classification_report"]

    support = [report[c]["support"] for c in CLASS_NAMES]

    plt.figure(figsize=(8, 5))

    colors = [
        "#EF476F",   # DOWN
        "#B8C0C8",   # SAME
        "#06D6A0"    # UP
    ]

    bars = plt.bar(
        CLASS_NAMES,
        support,
        width=0.55,
        color=colors,
        edgecolor="white",
        linewidth=1.5
    )

    for bar in bars:

        height = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold"
        )

    plt.xlabel(
        "Direction Class",
        fontsize=12
    )

    plt.ylabel(
        "Number of Records",
        fontsize=12
    )

    plt.xticks(fontsize=11)
    plt.yticks(fontsize=10)

    plt.grid(
        True,
        axis="y",
        linestyle="--",
        alpha=0.25
    )

    save_plot(
        "direction_classifier_class_distribution.png"
    )


def main():
    info = load_info()

    plot_confusion_matrix(info)
    plot_class_metrics(info)
    plot_class_distribution(info)


if __name__ == "__main__":
    main()