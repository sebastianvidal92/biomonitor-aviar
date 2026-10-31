"""Mapas de distribución estimada (SDM, Random Forest) para la tesis → tesis/figuras/.

Usa los Random Forest por especie ya entrenados para la demo (models/sdm_rf.pkl, script 23)
y la grilla de covariables de Chile (data/processed/chile_grid.csv, 0,1°) con la máscara
de Chile continental (chile_grid_mask.csv). Produce:
  - fig_sdm_mapas.pdf: probabilidad de presencia en enero para tres especies de
    distribución contrastante, con los avistamientos (GBIF) superpuestos.
  - fig_sdm_estacional.pdf: una especie migratoria en enero y en julio.

    uv run python scripts/33_sdm_maps.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biomonitor.config import PROJECT_ROOT  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

tf = import_module("32_thesis_figures")  # estilo, paleta y silueta de Chile

DATA = PROJECT_ROOT / "data"
ESPECIES = [("Scelorchilus rubecula", "chucao"), ("Cyanoliseus patagonus", "tricahue (EN)"),
            ("Conirostrum tamarugense", "comesebo de los tamarugales (EN)")]
MIGRATORIA = ("Limosa haemastica", "zarapito de pico recto (VU)")


def predict(models: dict, grid: pd.DataFrame, species: str, month: int) -> np.ndarray:
    X = grid[["ndvi", "temp", "precip", "elev"]].copy()
    X["sin_m"], X["cos_m"] = np.sin(2 * np.pi * month / 12), np.cos(2 * np.pi * month / 12)
    return models[species].predict_proba(X[["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]])[:, 1]


def draw(ax, grid, prob, occ, species, title):
    tf.chile_outline(ax, fill=False)
    sc = ax.scatter(grid["lon"], grid["lat"], c=prob, s=1.3, marker="s", cmap="Blues", vmin=0, vmax=1, linewidths=0, zorder=2)
    o = occ[occ["scientific_name"] == species].sample(min(600, (occ["scientific_name"] == species).sum()), random_state=42)
    ax.scatter(o["lon"], o["lat"], s=1.2, color=tf.C2, alpha=0.55, linewidths=0, zorder=3)
    ax.set_title(title, fontsize=8); ax.set_xlabel(""); ax.set_ylabel("")
    ax.set_xticks([-74, -70]); ax.tick_params(labelsize=7)
    return sc


def main() -> None:
    bundle = joblib.load(PROJECT_ROOT / "models" / "sdm_rf.pkl")
    models = bundle["models"]
    grid = pd.read_csv(DATA / "processed" / "chile_grid.csv").merge(pd.read_csv(DATA / "processed" / "chile_grid_mask.csv"), on=["lat", "lon"])
    grid = grid[grid["in_chile"] == 1].reset_index(drop=True)
    occ = pd.read_csv(DATA / "interim" / "occurrences.csv")

    fig, axes = plt.subplots(1, 3, figsize=(tf.TEXTW, 5.0), gridspec_kw={"wspace": 0.12})
    for ax, (sp, nombre) in zip(axes, ESPECIES):
        sc = draw(ax, grid, predict(models, grid, sp, 1), occ, sp, f"$\\it{{{sp.replace(' ', '\\ ')}}}$\n{nombre}")
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Latitud")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.025, pad=0.07, shrink=0.6)
    cb.set_label("Probabilidad de presencia estimada (enero)"); cb.outline.set_visible(False)
    axes[0].scatter([], [], s=8, color=tf.C2, label="Avistamientos (GBIF)"); axes[0].legend(loc="lower left", fontsize=7)
    tf.finish(fig, "fig_sdm_mapas")

    sp, nombre = MIGRATORIA
    fig, axes = plt.subplots(1, 2, figsize=(tf.TEXTW * 0.62, 5.0), gridspec_kw={"wspace": 0.12})
    for ax, (mes, etiqueta) in zip(axes, [(1, "enero (verano austral)"), (7, "julio (invierno austral)")]):
        sc = draw(ax, grid, predict(models, grid, sp, mes), occ, sp, f"$\\it{{{sp.replace(' ', '\\ ')}}}$\n{etiqueta}")
    axes[1].set_yticklabels([]); axes[0].set_ylabel("Latitud")
    cb = fig.colorbar(sc, ax=axes, orientation="horizontal", fraction=0.025, pad=0.07, shrink=0.7)
    cb.set_label("Probabilidad de presencia estimada"); cb.outline.set_visible(False)
    tf.finish(fig, "fig_sdm_estacional")
    print("figuras:", ", ".join(tf.saved))


if __name__ == "__main__":
    main()
