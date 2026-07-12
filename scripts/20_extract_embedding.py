"""Extrae el embedding satelital (64-dim, GEE) para cada punto del SDM (Módulo 2).

Muestrea las 64 bandas del Satellite Embedding de Google en cada punto de
sdm_dataset.csv y las guarda alineadas por índice. Es la representación de la imagen
satelital por un foundation model: datos ricos que el DL puede explotar y MaxEnt no.
Reanudable.

    uv run python scripts/20_extract_embedding.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.gee import init_ee  # noqa: E402

BATCH = 500


def main() -> None:
    init_ee()
    from biomonitor.sdm.covariates import embedding_image, sample_covariates

    df = pd.read_csv(data_path("interim", "sdm_dataset.csv")).reset_index(drop=True)
    df["idx"] = range(len(df))
    out = data_path("interim", "sdm_embedding.csv")

    done = pd.read_csv(out) if out.exists() else pd.DataFrame(columns=["idx"])
    todo_df = df[~df["idx"].isin(done["idx"])]
    points = [{"lat": r.lat, "lon": r.lon, "idx": int(r.idx)} for r in todo_df.itertuples(index=False)]
    if not points:
        print("Nada que extraer (ya completo).")
        return
    print(f"Extrayendo embedding en {len(points)} puntos (batches de {BATCH})...")

    emb_img = embedding_image()
    parts = [done]
    for i in tqdm(range(0, len(points), BATCH * 4), desc="Bloques"):
        chunk = points[i:i + BATCH * 4]
        parts.append(sample_covariates(chunk, emb_img, scale=10, batch=BATCH))
        pd.concat(parts, ignore_index=True).to_csv(out, index=False)  # guardado incremental

    res = pd.concat(parts, ignore_index=True)
    res.to_csv(out, index=False)
    bands = [c for c in res.columns if c.startswith("A")]
    print(f"\n== Embedding extraído ==")
    print(f"  Puntos: {len(res)}/{len(df)} | dimensiones: {len(bands)}")
    print(f"  → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
