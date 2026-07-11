"""Particiona el dataset de imágenes en train/val/test.

Filtra las especies con muy pocas fotos (quedan solo-audio, cubiertas por la fusión
multimodal) y estratifica por especie. Cada foto es una observación independiente, así
que la partición es a nivel de foto (sin el problema de leakage del audio).

    uv run python scripts/11_split_images.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, data_path  # noqa: E402

SEED = 42
RATIOS = {"val": 0.15, "test": 0.15}
MIN_FOR_CLASSIFIER = CONFIG["image"]["min_for_classifier"]


def assign_splits(n: int, rng: np.random.Generator) -> list[str]:
    idx = rng.permutation(n)
    n_val = max(1, round(n * RATIOS["val"]))
    n_test = max(1, round(n * RATIOS["test"]))
    labels = np.array(["train"] * n, dtype=object)
    labels[idx[:n_val]] = "val"
    labels[idx[n_val:n_val + n_test]] = "test"
    return labels.tolist()


def main() -> None:
    m = pd.read_csv(data_path("interim", "image_manifest.csv"))
    counts = m.groupby("scientific_name").size()

    keep = counts[counts >= MIN_FOR_CLASSIFIER].index
    excluded = sorted(set(counts.index) - set(keep))
    m = m[m["scientific_name"].isin(keep)].copy()

    rng = np.random.default_rng(SEED)
    m["split"] = ""
    for sp, grp in m.groupby("scientific_name"):
        m.loc[grp.index, "split"] = assign_splits(len(grp), rng)

    out = data_path("interim", "image_manifest_split.csv")
    m.to_csv(out, index=False)

    print("\n== Partición de imágenes ==\n")
    print(f"  Especies en el clasificador: {m['scientific_name'].nunique()} (≥{MIN_FOR_CLASSIFIER} fotos)")
    print(f"  Excluidas (solo-audio): {len(excluded)} → {', '.join(excluded) if excluded else '—'}")
    tab = m.groupby("split").size()
    for s in ("train", "val", "test"):
        print(f"  {s:<6} {int(tab.get(s, 0)):>5} fotos")
    print(f"  → {out.relative_to(out.parents[2])}\n")


if __name__ == "__main__":
    main()
