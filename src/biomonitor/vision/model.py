"""Modelo del clasificador de imagen (Módulo 1) — transfer learning con timm.

EfficientNet-B3 (o ViT) preentrenado en ImageNet, con cabeza nueva para las especies.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_image_model(
    backbone: str = "efficientnet_b3",
    num_classes: int = 53,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """CNN/ViT preentrenada en ImageNet con cabeza de `num_classes` (3 canales RGB)."""
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
