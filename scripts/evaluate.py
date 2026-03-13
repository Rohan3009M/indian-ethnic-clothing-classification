import sys
from pathlib import Path
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch

from src.data.dataloaders import get_dataloaders
from src.models.model_factory import get_model
from src.utils.device import get_device, print_device_info
from src.evaluation.evaluator import (
    evaluate_model,
    save_classification_report,
    save_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate trained image classification model")
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        choices=["resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121"],
        help="Model architecture to evaluate",
    )
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Use if the model was trained with frozen backbone",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print_device_info()
    device = get_device()

    _, _, test_loader, classes = get_dataloaders(
        data_dir="data/processed/dataset_subset",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = get_model(
        model_name=args.model_name,
        num_classes=len(classes),
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{args.model_name}_best.pth"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader, device)

    print(f"\nModel: {args.model_name}")
    print(f"Test Accuracy: {results['accuracy']:.4f}")

    report_path = PROJECT_ROOT / "outputs" / "reports" / f"{args.model_name}_classification_report.txt"
    cm_path = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrices" / f"{args.model_name}_confusion_matrix.png"
    metrics_path = PROJECT_ROOT / "outputs" / "reports" / f"{args.model_name}_test_metrics.json"

    report_dict = save_classification_report(
        results["y_true"],
        results["y_pred"],
        classes,
        report_path,
    )

    save_confusion_matrix(
        results["y_true"],
        results["y_pred"],
        classes,
        cm_path,
    )

    metrics = {
        "model_name": args.model_name,
        "test_accuracy": results["accuracy"],
        "macro_precision": report_dict["macro avg"]["precision"],
        "macro_recall": report_dict["macro avg"]["recall"],
        "macro_f1": report_dict["macro avg"]["f1-score"],
        "weighted_f1": report_dict["weighted avg"]["f1-score"],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Classification report saved to: {report_path}")
    print(f"Confusion matrix saved to: {cm_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()




# import sys
# from pathlib import Path
# import json

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# import torch

# from src.data.dataloaders import get_dataloaders
# from src.models.model_factory import get_model
# from src.utils.device import get_device, print_device_info
# from src.evaluation.evaluator import (
#     evaluate_model,
#     save_classification_report,
#     save_confusion_matrix,
# )


# def main():
#     model_name = "resnet50"
#     num_classes = 15
#     batch_size = 32
#     image_size = 224
#     num_workers = 2

#     print_device_info()
#     device = get_device()

#     _, _, test_loader, classes = get_dataloaders(
#         data_dir="data/processed/dataset_subset",
#         image_size=image_size,
#         batch_size=batch_size,
#         num_workers=num_workers,
#     )

#     model = get_model(
#         model_name=model_name,
#         num_classes=num_classes,
#         freeze_backbone=True,
#     ).to(device)

#     checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{model_name}_best.pth"
#     checkpoint = torch.load(checkpoint_path, map_location=device)
#     model.load_state_dict(checkpoint["model_state_dict"])

#     results = evaluate_model(model, test_loader, device)

#     print(f"\nTest Accuracy: {results['accuracy']:.4f}")

#     report_path = PROJECT_ROOT / "outputs" / "reports" / f"{model_name}_classification_report.txt"
#     cm_path = PROJECT_ROOT / "outputs" / "figures" / "confusion_matrices" / f"{model_name}_confusion_matrix.png"
#     metrics_path = PROJECT_ROOT / "outputs" / "reports" / f"{model_name}_test_metrics.json"

#     save_classification_report(
#         results["y_true"],
#         results["y_pred"],
#         classes,
#         report_path,
#     )

#     save_confusion_matrix(
#         results["y_true"],
#         results["y_pred"],
#         classes,
#         cm_path,
#     )

#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump({"test_accuracy": results["accuracy"]}, f, indent=2)

#     print(f"Classification report saved to: {report_path}")
#     print(f"Confusion matrix saved to: {cm_path}")
#     print(f"Metrics saved to: {metrics_path}")


# if __name__ == "__main__":
#     main()