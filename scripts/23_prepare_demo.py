"""Prepara los insumos de la demo web (Hito 5).

1. Entrena y guarda un Random Forest por especie (el mejor SDM) → models/sdm_rf.pkl.
2. Extrae covariables ambientales en una grilla de Chile (GEE) → data/processed/chile_grid.csv,
   para dibujar el mapa de distribución.

    uv run python scripts/23_prepare_demo.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.gee import init_ee  # noqa: E402

MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = CONFIG["train_sdm"]["background_per_species"]
BBOX = CONFIG["region"]["bbox"]
BASE = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]
GRID_STEP = 0.1  # grados
SEED = 42


def train_rf_models() -> None:
    df = pd.read_csv(data_path("interim", "sdm_dataset.csv"))
    df["sin_m"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)
    presence = df[df["label"] == 1]
    background = df[df["label"] == 0]
    species = sorted(presence["species"].value_counts().loc[lambda c: c >= MIN].index)

    models = {}
    for s in tqdm(species, desc="RF por especie"):
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(BG_PER, len(background)), random_state=SEED)])
        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    n_jobs=-1, random_state=SEED)
        models[s] = rf.fit(sub[BASE], sub["label"])

    out = PROJECT_ROOT / CONFIG["paths"]["models"] / "sdm_rf.pkl"
    joblib.dump({"models": models, "features": BASE, "static_feats": BASE[:4]}, out)
    print(f"  {len(models)} modelos RF guardados → {out.relative_to(PROJECT_ROOT)}")


def extract_grid() -> None:
    init_ee()
    from biomonitor.sdm.covariates import covariate_image, sample_covariates

    lats = np.arange(BBOX["lat_min"], BBOX["lat_max"], GRID_STEP)
    lons = np.arange(BBOX["lon_min"], BBOX["lon_max"], GRID_STEP)
    grid = [{"lat": float(la), "lon": float(lo)} for la in lats for lo in lons]
    print(f"  Muestreando covariables en {len(grid)} celdas de la grilla (GEE)...")

    cov = covariate_image()
    parts = []
    for i in tqdm(range(0, len(grid), 2000), desc="Grilla"):
        parts.append(sample_covariates(grid[i:i + 2000], cov, scale=1000, batch=500))
    g = pd.concat(parts, ignore_index=True).dropna(subset=["ndvi", "temp", "precip", "elev"])
    out = data_path("processed", "chile_grid.csv")
    g[["lat", "lon", "ndvi", "temp", "precip", "elev"]].to_csv(out, index=False)
    print(f"  Grilla con {len(g)} celdas terrestres → {out.relative_to(PROJECT_ROOT)}")


def main() -> None:
    print("== Preparando insumos de la demo ==")
    train_rf_models()
    extract_grid()
    print("\nListo. Corre la demo con: uv run streamlit run app/app.py\n")


if __name__ == "__main__":
    main()
