"""SDM multimodal: covariables tabulares + parche satelital (CNN) + embedding de especie.

La rama CNN procesa el parche de cobertura de suelo (imagen no estructurada) que MaxEnt
no puede usar; su salida se concatena con las covariables tabulares y el embedding de
especie. Es el mecanismo por el que el deep learning puede superar a MaxEnt.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class MultimodalSDM(nn.Module):
    def __init__(self, n_env: int, n_species: int, n_lc: int = 11,
                 emb_dim: int = 8, hidden: int = 64, img_dim: int = 32):
        super().__init__()
        self.sp_emb = nn.Embedding(n_species, emb_dim)
        # BatchNorm estabiliza la rama de imagen (evita que sus features dominen).
        self.cnn = nn.Sequential(
            nn.Conv2d(n_lc, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.AdaptiveAvgPool2d(4),
            nn.Flatten(), nn.Linear(32 * 4 * 4, img_dim),
        )
        # LayerNorm sobre el vector fusionado: normaliza a la misma escala tabular+imagen+especie.
        self.head = nn.Sequential(
            nn.LayerNorm(n_env + emb_dim + img_dim),
            nn.Linear(n_env + emb_dim + img_dim, hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_env: torch.Tensor, species: torch.Tensor,
                patch: torch.Tensor) -> torch.Tensor:
        img = self.cnn(patch)
        h = torch.cat([x_env, self.sp_emb(species), img], dim=1)
        return self.head(h).squeeze(1)  # logits
