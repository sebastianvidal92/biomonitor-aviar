"""Particiona el dataset de audio en train/val/test.

Clave metodológica: la partición es a nivel de GRABACIÓN (no de clip). Todos los
clips de 5 s que luego se extraigan de una grabación quedan en el mismo conjunto,
evitando fuga de información (data leakage) entre train/val/test.

Estratificado por especie y reproducible (semilla fija).

    uv run python scripts/05_split_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import data_path  # noqa: E402

SEED = 42
RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}


def assign_splits(n: int, rng: np.random.Generator) -> list[str]:
    """Asigna n grabaciones a splits respetando ratios, con ≥1 en val y test."""
    idx = rng.permutation(n)
    n_val = max(1, round(n * RATIOS["val"])) if n >= 3 else 0
    n_test = max(1, round(n * RATIOS["test"])) if n >= 3 else 0
    labels = np.array(["train"] * n, dtype=object)
    labels[idx[:n_val]] = "val"
    labels[idx[n_val:n_val + n_test]] = "test"
    return labels.tolist()


def main() -> None:
    m = pd.read_csv(data_path("interim", "audio_manifest.csv"))
    rng = np.random.default_rng(SEED)

    m["split"] = ""
    for sp, grp in m.groupby("scientific_name"):
        m.loc[grp.index, "split"] = assign_splits(len(grp), rng)

    out = data_path("interim", "audio_manifest_split.csv")
    m.to_csv(out, index=False)

    print("\n== Partición train/val/test (a nivel de grabación) ==\n")
    tab = m.groupby(["split"]).size()
    total = len(m)
    for split in ["train", "val", "test"]:
        n = int(tab.get(split, 0))
        print(f"  {split:<6} {n:>5} grabaciones  ({n/total:.0%})")
    # Verificación: toda especie presente en train.
    sp_per_split = m.groupby("split")["scientific_name"].nunique()
    print(f"\n  Especies presentes por split: {sp_per_split.to_dict()}")
    print(f"  → {out.relative_to(out.parents[2])}\n")


if __name__ == "__main__":
    main()
