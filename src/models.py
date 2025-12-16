import torch.nn as nn
import torchvision.models as models


def _freeze_all(model):
    for p in model.parameters():
        p.requires_grad = False


def create_model(model_name: str, num_classes: int, freeze_backbone: bool = False):
    """
    Creates an ImageNet-pretrained model and replaces the classification head.
    Supported: resnet50, mobilenet_v3_small, densenet121
    """
    model_name = model_name.lower().strip()

    if model_name == "resnet50":
        try:
            weights = models.ResNet50_Weights.IMAGENET1K_V1
        except Exception:
            weights = "IMAGENET1K_V1"
        model = models.resnet50(weights=weights)

        if freeze_backbone:
            _freeze_all(model)

        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)

        # ensure head is trainable
        for p in model.fc.parameters():
            p.requires_grad = True

    elif model_name == "mobilenet_v3_small":
        try:
            weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
        except Exception:
            weights = "IMAGENET1K_V1"
        model = models.mobilenet_v3_small(weights=weights)

        if freeze_backbone:
            _freeze_all(model)

        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, num_classes)

        for p in model.classifier[3].parameters():
            p.requires_grad = True

    elif model_name == "densenet121":
        try:
            weights = models.DenseNet121_Weights.IMAGENET1K_V1
        except Exception:
            weights = "IMAGENET1K_V1"
        model = models.densenet121(weights=weights)

        if freeze_backbone:
            _freeze_all(model)

        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)

        for p in model.classifier.parameters():
            p.requires_grad = True

    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    return model
