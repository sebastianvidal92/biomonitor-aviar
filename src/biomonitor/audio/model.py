"""Modelos del clasificador de cantos (Módulo 1).

Baseline: CNN preentrenada (timm) que toma el espectrograma log-mel de 1 canal
como si fuera una imagen. timm adapta la primera convolución a in_chans=1.
"""

from __future__ import annotations

import timm
import torch.nn as nn


def build_audio_model(
    backbone: str = "efficientnet_b0",
    num_classes: int = 55,
    in_chans: int = 1,
    pretrained: bool = True,
    drop_rate: float = 0.3,
    drop_path_rate: float = 0.2,
) -> nn.Module:
    """Crea la CNN baseline para clasificar espectrogramas.

    El espectrograma entra como [B, 1, n_mels, frames]; EfficientNet usa pooling
    adaptativo, así que admite el ancho temporal variable. `drop_rate` (dropout del
    clasificador) y `drop_path_rate` (stochastic depth) regularizan contra el
    sobreajuste observado en el baseline.
    """
    return timm.create_model(
        backbone,
        pretrained=pretrained,
        num_classes=num_classes,
        in_chans=in_chans,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
