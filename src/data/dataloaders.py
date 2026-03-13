from pathlib import Path
from typing import Tuple, List

from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from src.data.transforms import get_train_transforms, get_val_test_transforms


def get_dataloaders(
    data_dir: str = "data/processed/dataset_subset",
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    data_path = Path(data_dir)

    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    train_dataset = ImageFolder(
        root=str(train_dir),
        transform=get_train_transforms(image_size=image_size),
    )

    val_dataset = ImageFolder(
        root=str(val_dir),
        transform=get_val_test_transforms(image_size=image_size),
    )

    test_dataset = ImageFolder(
        root=str(test_dir),
        transform=get_val_test_transforms(image_size=image_size),
    )

    classes = train_dataset.classes

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader, classes