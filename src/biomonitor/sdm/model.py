"""Modelo de distribución espaciotemporal (SDM) con deep learning — Módulo 2.

Red neuronal CONJUNTA multi-especie: las covariables ambientales + estacionales pasan
por un MLP, y cada especie aporta un embedding aprendido. Al compartir parámetros entre
especies, aprovecha patrones comunes (ventaja sobre un MaxEnt por especie), especialmente
para especies con pocos registros.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class DeepSDM(nn.Module):
    def __init__(self, n_env: int, n_species: int, emb_dim: int = 8, hidden: int = 64):
        super().__init__()
        self.emb = nn.Embedding(n_species, emb_dim)
        self.net = nn.Sequential(
            nn.Linear(n_env + emb_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_env: torch.Tensor, species: torch.Tensor) -> torch.Tensor:
        h = torch.cat([x_env, self.emb(species)], dim=1)
        return self.net(h).squeeze(1)  # logits
