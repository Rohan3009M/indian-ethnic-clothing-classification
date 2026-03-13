import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch
from src.data.dataloaders import get_dataloaders


def main():
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        data_dir="data/processed/dataset_subset",
        image_size=224,
        batch_size=32,
        num_workers=2,
    )

    print("Classes:")
    print(classes)
    print(f"Number of classes: {len(classes)}")

    images, labels = next(iter(train_loader))

    print("\nOne batch check:")
    print("Images shape:", images.shape)
    print("Labels shape:", labels.shape)
    print("Label dtype:", labels.dtype)
    print("CUDA available:", torch.cuda.is_available())


if __name__ == "__main__":
    main()