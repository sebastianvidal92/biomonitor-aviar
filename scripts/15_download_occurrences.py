"""Descarga avistamientos (puntos de presencia) para el SDM (Módulo 2).

Por cada especie baja hasta `sdm.max_occurrences_per_species` ocurrencias con
coordenadas dentro de Chile continental y las guarda en un único CSV con especie,
lat, lon, año y mes. No requiere credenciales (búsqueda pública de GBIF).

    uv run python scripts/15_download_occurrences.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.occurrences import species_occurrences  # noqa: E402

MAX = CONFIG["sdm"]["max_occurrences_per_species"]
MIN = CONFIG["sdm"]["min_occurrences"]
BBOX = CONFIG["region"]["bbox"]


def main() -> None:
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    out = data_path("interim", "occurrences.csv")

    # Reanudable: conserva las especies ya descargadas, solo pide las que faltan.
    existing = pd.read_csv(out) if out.exists() else pd.DataFrame()
    done = set(existing["scientific_name"].unique()) if len(existing) else set()
    todo = [sp for sp in species if sp not in done]
    if done:
        print(f"Reanudando: {len(done)} especies ya descargadas, faltan {len(todo)}.")

    rows = []
    for sp in tqdm(todo, desc="Especies"):
        try:
            recs = species_occurrences(sp, MAX, BBOX)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  ⚠️  {sp}: {exc}")
            continue
        for r in recs:
            r["scientific_name"] = sp
        rows.extend(recs)

    df = pd.concat([existing, pd.DataFrame(rows)], ignore_index=True) if len(existing) else pd.DataFrame(rows)
    df.to_csv(out, index=False)

    per_sp = df.groupby("scientific_name").size()
    low = per_sp[per_sp < MIN]
    print(f"\n== Avistamientos descargados (Chile continental) ==")
    print(f"  Puntos totales: {len(df):,}")
    print(f"  Especies: {per_sp.size}/{len(species)}")
    print(f"  Por especie: min={per_sp.min()} · mediana={int(per_sp.median())} · max={per_sp.max()}")
    print(f"  Especies con <{MIN} puntos (difíciles de modelar): {len(low)}"
          + (f" → {', '.join(low.index)}" if len(low) else ""))
    print(f"  Rango temporal: {int(df['year'].min())}-{int(df['year'].max())}")
    print(f"  → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
