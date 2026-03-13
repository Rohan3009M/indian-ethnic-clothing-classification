import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import torch

from src.models.model_factory import get_model
from src.utils.device import get_device, print_device_info


def count_trainable_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model):
    return sum(p.numel() for p in model.parameters())


def main():
    print_device_info()
    device = get_device()

    model_names = [
        "resnet50",
        "mobilenet_v2",
        "efficientnet_b0",
        "densenet121",
    ]

    num_classes = 15

    for model_name in model_names:
        print("\n" + "=" * 60)
        print(f"Testing model: {model_name}")

        model = get_model(
            model_name=model_name,
            num_classes=num_classes,
            freeze_backbone=True,
        )
        model = model.to(device)

        dummy_input = torch.randn(4, 3, 224, 224).to(device)
        output = model(dummy_input)

        print("Output shape:", output.shape)
        print("Total params:", count_total_params(model))
        print("Trainable params:", count_trainable_params(model))

        assert output.shape == (4, num_classes), f"Unexpected output shape for {model_name}"

    print("\nAll models loaded successfully.")


if __name__ == "__main__":
    main()