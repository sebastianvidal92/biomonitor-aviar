"""Extrae covariables ambientales (GEE) para presencia + fondo del SDM (Módulo 2).

- Presencia: puntos de avistamientos (occurrences.csv), adelgazados a
  `sdm.presence_cap_per_species` por especie.
- Fondo: `sdm.background_points` puntos aleatorios en Chile continental (mes aleatorio).
- Para todos: NDVI + temperatura + precipitación + elevación (muestreo por punto en GEE).

Guarda data/interim/sdm_dataset.csv (label 1=presencia, 0=fondo). Reanudable.

    uv run python scripts/16_extract_covariates.py

NOTA: hace muchas consultas a GEE (varios minutos) → lo corre el usuario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.gee import init_ee  # noqa: E402

CAP = CONFIG["sdm"]["presence_cap_per_species"]
N_BG = CONFIG["sdm"]["background_points"]
SEED = 42


def main() -> None:
    init_ee()
    from biomonitor.sdm import covariates as cov  # importa tras init_ee

    # --- presencia: adelgazar a CAP por especie ---
    occ = pd.read_csv(data_path("interim", "occurrences.csv"))
    occ = occ.dropna(subset=["lat", "lon"])
    pres = pd.concat(
        [g.sample(min(len(g), CAP), random_state=SEED) for _, g in occ.groupby("scientific_name")],
        ignore_index=True,
    )
    pres_points = [
        {"lat": r.lat, "lon": r.lon, "species": r.scientific_name,
         "month": int(r.month) if not pd.isna(r.month) else 0, "label": 1}
        for r in pres.itertuples(index=False)
    ]

    # --- fondo: target-group background (muestra de avistamientos de todas las especies) ---
    bg_pool = occ.sample(min(N_BG, len(occ)), random_state=SEED)
    bg = [
        {"lat": r.lat, "lon": r.lon, "species": "background",
         "month": int(r.month) if not pd.isna(r.month) else 0, "label": 0}
        for r in bg_pool.itertuples(index=False)
    ]

    all_points = pres_points + bg
    print(f"Muestreando covariables en {len(all_points):,} puntos "
          f"({len(pres_points):,} presencia + {len(bg):,} fondo)...")
    cov_img = cov.covariate_image()
    df = cov.sample_covariates(all_points, cov_img)

    # limpieza: descarta puntos sin covariables (ej. sobre agua)
    before = len(df)
    df = df.dropna(subset=["ndvi", "temp", "precip", "elev"])
    out = data_path("interim", "sdm_dataset.csv")
    df.to_csv(out, index=False)

    print(f"\n== Dataset del SDM ==")
    print(f"  Puntos con covariables: {len(df):,} (descartados {before - len(df)} sin datos)")
    print(f"  Presencia: {int((df['label'] == 1).sum()):,} | Fondo: {int((df['label'] == 0).sum()):,}")
    print(f"  Especies con presencia: {df[df['label'] == 1]['species'].nunique()}")
    print(f"  → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
