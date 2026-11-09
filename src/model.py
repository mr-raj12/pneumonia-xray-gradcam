"""Model construction: ImageNet-pretrained EfficientNet-B0 with a 2-class head."""
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models

from .data import CLASSES, ROOT

CHECKPOINT = ROOT / "models" / "best_model.pt"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.efficientnet_b0(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3),
        nn.Linear(in_features, len(CLASSES)),
    )
    return model


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    # Last conv block of EfficientNet-B0 (1280 channels, 7x7 spatial at 224px input).
    return model.features[-1]


def load_trained(path: Path = CHECKPOINT, device: str = "cpu") -> tuple[nn.Module, dict]:
    """Returns (model in eval mode, checkpoint metadata such as threshold/epoch/val_auc)."""
    model = build_model(pretrained=False)
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state.pop("model"))
    state.setdefault("threshold", 0.5)
    return model.to(device).eval(), state
