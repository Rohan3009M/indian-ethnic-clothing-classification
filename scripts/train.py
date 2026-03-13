import sys
from pathlib import Path
import json
import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import StepLR

from src.data.dataloaders import get_dataloaders
from src.models.model_factory import get_model
from src.training.trainer import train_model
from src.utils.device import get_device, print_device_info
from src.utils.plotting import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train image classification model")
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        choices=["resnet50", "mobilenet_v2", "efficientnet_b0", "densenet121"],
        help="Model architecture to train",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--image_size", type=int, default=224, help="Input image size")
    parser.add_argument("--num_workers", type=int, default=2, help="Number of dataloader workers")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--freeze_backbone",
        action="store_true",
        help="Freeze pretrained backbone",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    num_classes = 15

    print_device_info()
    device = get_device()

    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir="data/processed/dataset_subset",
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nClasses ({len(classes)}): {classes}")
    print(f"Training model: {args.model_name}")
    print(f"Freeze backbone: {args.freeze_backbone}")

    model = get_model(
        model_name=args.model_name,
        num_classes=num_classes,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

    checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{args.model_name}_best.pth"

    trained_model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=args.epochs,
        checkpoint_path=checkpoint_path,
    )

    history_path = PROJECT_ROOT / "outputs" / "logs" / args.model_name / "history.json"
    history_path.parent.mkdir(parents=True, exist_ok=True)

    with open(history_path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    figures_dir = PROJECT_ROOT / "outputs" / "figures" / "training_curves"
    plot_training_history(
        history=history,
        model_name=args.model_name,
        output_dir=figures_dir,
    )

    print(f"\nTraining history saved to: {history_path}")
    print(f"Best model checkpoint saved to: {checkpoint_path}")
    print(f"Training curves saved to: {figures_dir}")


if __name__ == "__main__":
    main()

# Example Run Command: python .\scripts\train.py --model_name resnet50 --epochs 5 --freeze_backbone













# import sys
# from pathlib import Path
# import json

# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# sys.path.append(str(PROJECT_ROOT))

# import torch
# import torch.nn as nn
# from torch.optim import AdamW
# from torch.optim.lr_scheduler import StepLR

# from src.data.dataloaders import get_dataloaders
# from src.models.model_factory import get_model
# from src.training.trainer import train_model
# from src.utils.device import get_device, print_device_info
# from src.utils.plotting import plot_training_history


# def main():
#     model_name = "resnet50"   # change later: mobilenet_v2, efficientnet_b0, densenet121
#     num_classes = 15
#     batch_size = 32
#     image_size = 224
#     num_workers = 2
#     num_epochs = 15   # started with 5, but increased to 15 for better performance
#     learning_rate = 1e-3
#     weight_decay = 1e-4

#     print_device_info()
#     device = get_device()

#     train_loader, val_loader, test_loader, classes = get_dataloaders(
#         data_dir="data/processed/dataset_subset",
#         image_size=image_size,
#         batch_size=batch_size,
#         num_workers=num_workers,
#     )

#     print(f"\nClasses ({len(classes)}): {classes}")

#     model = get_model(
#         model_name=model_name,
#         num_classes=num_classes,
#         freeze_backbone=True,
#     ).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = AdamW(
#         model.parameters(),
#         lr=learning_rate,
#         weight_decay=weight_decay,
#     )
#     scheduler = StepLR(optimizer, step_size=3, gamma=0.1)

#     checkpoint_path = PROJECT_ROOT / "outputs" / "checkpoints" / f"{model_name}_best.pth"

#     trained_model, history = train_model(
#         model=model,
#         train_loader=train_loader,
#         val_loader=val_loader,
#         criterion=criterion,
#         optimizer=optimizer,
#         scheduler=scheduler,
#         device=device,
#         num_epochs=num_epochs,
#         checkpoint_path=checkpoint_path,
#     )

#     history_path = PROJECT_ROOT / "outputs" / "logs" / model_name / "history.json"
#     history_path.parent.mkdir(parents=True, exist_ok=True)

#     with open(history_path, "w", encoding="utf-8") as f:
#         json.dump(history, f, indent=2)

#     figures_dir = PROJECT_ROOT / "outputs" / "figures" / "training_curves"
#     plot_training_history(
#         history=history,
#         model_name=model_name,
#         output_dir=figures_dir,
#     )

#     print(f"Training curves saved to: {figures_dir}")
#     print(f"\nTraining history saved to: {history_path}")
#     print(f"Best model checkpoint saved to: {checkpoint_path}")


# if __name__ == "__main__":
#     main()