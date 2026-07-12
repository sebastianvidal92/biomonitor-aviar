"""Extrae parches de cobertura de suelo (GEE) para cada punto del SDM (Módulo 2).

Lee data/interim/sdm_dataset.csv y extrae un parche 33×33 (ESA WorldCover, 10 m) por
punto. Guarda un .npz alineado con las filas (parches + índice) para el SDM multimodal.
Reanudable: si ya existe el .npz, solo extrae los puntos que falten.

    uv run python scripts/18_extract_patches.py

NOTA: hace muchas consultas a GEE (varios minutos) → lo corre el usuario.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.gee import init_ee  # noqa: E402

BATCH = 200
RADIUS = 16  # parche 33×33


def main() -> None:
    init_ee()
    from biomonitor.sdm.patches import sample_patches  # tras init_ee

    df = pd.read_csv(data_path("interim", "sdm_dataset.csv")).reset_index(drop=True)
    df["idx"] = range(len(df))
    out_npz = data_path("interim", "sdm_patches.npz")

    done: dict[int, np.ndarray] = {}
    if out_npz.exists():  # reanudable
        z = np.load(out_npz)
        done = {int(i): p for i, p in zip(z["idx"], z["patches"])}
        print(f"Reanudando: {len(done)} parches ya extraídos.")

    todo = [{"lat": r.lat, "lon": r.lon, "idx": int(r.idx)}
            for r in df.itertuples(index=False) if int(r.idx) not in done]
    print(f"Extrayendo {len(todo)} parches (batches de {BATCH})...")

    for i in tqdm(range(0, len(todo), BATCH * 5), desc="Bloques"):
        block = todo[i:i + BATCH * 5]
        done.update(sample_patches(block, radius=RADIUS, batch=BATCH))
        # guardado incremental (por si se corta)
        idxs = sorted(done)
        np.savez_compressed(out_npz, idx=np.array(idxs),
                            patches=np.stack([done[k] for k in idxs]))

    idxs = sorted(done)
    patches = np.stack([done[k] for k in idxs])
    np.savez_compressed(out_npz, idx=np.array(idxs), patches=patches)
    df.iloc[idxs].drop(columns=["idx"]).to_csv(data_path("interim", "sdm_patches_rows.csv"), index=False)

    print(f"\n== Parches extraídos ==")
    print(f"  Parches: {len(idxs)}/{len(df)} puntos | forma {patches.shape}")
    print(f"  → {out_npz.relative_to(PROJECT_ROOT)} + sdm_patches_rows.csv\n")


if __name__ == "__main__":
    main()
