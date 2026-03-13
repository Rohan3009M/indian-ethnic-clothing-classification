import torch.nn as nn
from torchvision import models


def _freeze_backbone(model):
    for param in model.parameters():
        param.requires_grad = False
    return model


def _build_classifier_head(in_features: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
    return nn.Sequential(
        nn.Linear(in_features, hidden_dim),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, num_classes),
    )


def get_model(model_name: str, num_classes: int, freeze_backbone: bool = True):
    model_name = model_name.lower()

    if model_name == "resnet50":
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)

        if freeze_backbone:
            _freeze_backbone(model)

        in_features = model.fc.in_features
        model.fc = _build_classifier_head(in_features, num_classes)

    elif model_name == "mobilenet_v2":
        weights = models.MobileNet_V2_Weights.DEFAULT
        model = models.mobilenet_v2(weights=weights)

        if freeze_backbone:
            _freeze_backbone(model)

        in_features = model.classifier[1].in_features
        model.classifier[1] = _build_classifier_head(in_features, num_classes)

    elif model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT
        model = models.efficientnet_b0(weights=weights)

        if freeze_backbone:
            _freeze_backbone(model)

        in_features = model.classifier[1].in_features
        model.classifier[1] = _build_classifier_head(in_features, num_classes)

    elif model_name == "densenet121":
        weights = models.DenseNet121_Weights.DEFAULT
        model = models.densenet121(weights=weights)

        if freeze_backbone:
            _freeze_backbone(model)

        in_features = model.classifier.in_features
        model.classifier = _build_classifier_head(in_features, num_classes)

    else:
        raise ValueError(
            f"Unsupported model_name: {model_name}. "
            f"Choose from: resnet50, mobilenet_v2, efficientnet_b0, densenet121"
        )

    return model