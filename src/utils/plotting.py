from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt


def plot_training_history(
    history: Dict[str, List[float]],
    model_name: str,
    output_dir: str | Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    epochs = list(range(1, len(history["train_loss"]) + 1))

    # Loss curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(f"{model_name} - Loss Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Accuracy curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title(f"{model_name} - Accuracy Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / f"{model_name}_accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()