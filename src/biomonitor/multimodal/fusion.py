"""Estrategias de fusión multimodal (Módulo 1) — late, early e intermediate.

Las tres operan sobre los embeddings de los backbones ya entrenados (congelados), de modo
que la ÚNICA diferencia es la arquitectura de fusión (comparación controlada).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LateFusion(nn.Module):
    """Un clasificador por modalidad; se promedian las probabilidades (fusión de decisión)."""

    def __init__(self, d_audio: int, d_image: int, n_classes: int):
        super().__init__()
        self.head_a = nn.Linear(d_audio, n_classes)
        self.head_i = nn.Linear(d_image, n_classes)

    def forward(self, ea, ei):
        return self.head_a(ea), self.head_i(ei)  # dos logits (se combinan al evaluar)


class EarlyFusion(nn.Module):
    """Concatena los embeddings a la entrada de una única cabeza (fusión temprana)."""

    def __init__(self, d_audio: int, d_image: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_audio + d_image, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes))

    def forward(self, ea, ei):
        return self.net(torch.cat([ea, ei], dim=1))


class IntermediateFusion(nn.Module):
    """Proyecta cada modalidad a un espacio común, concatena y clasifica (fusión intermedia)."""

    def __init__(self, d_audio: int, d_image: int, n_classes: int, hidden: int = 256):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(d_audio, hidden), nn.ReLU())
        self.proj_i = nn.Sequential(nn.Linear(d_image, hidden), nn.ReLU())
        self.head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_classes))

    def forward(self, ea, ei):
        return self.head(torch.cat([self.proj_a(ea), self.proj_i(ei)], dim=1))
