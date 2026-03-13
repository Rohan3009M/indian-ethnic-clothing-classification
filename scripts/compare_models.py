import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "outputs" / "reports"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures" / "model_comparison"

MODELS = ["resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121"]


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for model_name in MODELS:
        metrics_path = REPORTS_DIR / f"{model_name}_test_metrics.json"
        if not metrics_path.exists():
            print(f"Skipping missing metrics file: {metrics_path}")
            continue

        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)

        rows.append(metrics)

    if not rows:
        raise ValueError("No metrics files found. Run evaluate_all_models.py first.")

    df = pd.DataFrame(rows)
    df = df.sort_values(by="test_accuracy", ascending=False).reset_index(drop=True)

    summary_csv = REPORTS_DIR / "model_comparison_summary.csv"
    df.to_csv(summary_csv, index=False)
    print(f"Saved comparison summary to: {summary_csv}")

    # Accuracy plot
    plt.figure(figsize=(9, 6))
    plt.bar(df["model_name"], df["test_accuracy"])
    plt.xlabel("Model")
    plt.ylabel("Test Accuracy")
    plt.title("Model Comparison - Test Accuracy")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_accuracy_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Macro F1 plot
    plt.figure(figsize=(9, 6))
    plt.bar(df["model_name"], df["macro_f1"])
    plt.xlabel("Model")
    plt.ylabel("Macro F1")
    plt.title("Model Comparison - Macro F1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_macro_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Weighted F1 plot
    plt.figure(figsize=(9, 6))
    plt.bar(df["model_name"], df["weighted_f1"])
    plt.xlabel("Model")
    plt.ylabel("Weighted F1")
    plt.title("Model Comparison - Weighted F1")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_weighted_f1_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nFinal comparison:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()