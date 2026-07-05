"""Enriquece las especies en foco (RCE) con disponibilidad de datos.

Para cada especie consulta nº de grabaciones (Xeno-canto, global y Chile) y nº de
fotos en GBIF-Chile. Guarda data/external/species_focus_enriched.csv y resume
cuántas son entrenables para audio (≥ umbral de grabaciones).

    uv run python scripts/02_enrich_availability.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import data_path  # noqa: E402
from biomonitor.data import availability as av  # noqa: E402

MIN_RECORDINGS = 30  # umbral de entrenabilidad para audio (config.yaml: audio)


def main() -> None:
    df = pd.read_csv(data_path("external", "species_list_candidates.csv"))
    focus = df[df["in_focus"] & df["continental"]].copy().reset_index(drop=True)

    rows = []
    for _, r in tqdm(focus.iterrows(), total=len(focus), desc="Consultando"):
        sp = r["scientific_name"]
        try:
            xc_g = av.xc_recordings(sp)
            xc_cl = av.xc_recordings(sp, country="Chile")
        except Exception:
            xc_g = xc_cl = -1
        try:
            ph_cl = av.gbif_photos(sp, country="CL")
        except Exception:
            ph_cl = -1
        rows.append({"xc_global": xc_g, "xc_chile": xc_cl, "gbif_photos_cl": ph_cl})
        time.sleep(0.5)

    focus = pd.concat([focus, pd.DataFrame(rows)], axis=1)
    focus["audio_trainable"] = focus["xc_global"] >= MIN_RECORDINGS

    out = data_path("external", "species_focus_enriched.csv")
    focus.to_csv(out, index=False)

    print("\n== Disponibilidad de datos (especies en foco) ==\n")
    for grp in ["terrestrial", "coastal_shorebird", "pelagic"]:
        g = focus[focus["habitat_group"] == grp]
        trainable = g["audio_trainable"].sum()
        print(f"  {grp:<18} {len(g):>2} especies  |  con ≥{MIN_RECORDINGS} grabaciones: {trainable}")
    print()
    core = focus[(focus["habitat_group"] == "terrestrial") & focus["audio_trainable"]]
    print(f"  → NÚCLEO entrenable (terrestres en foco con audio): {len(core)} especies")
    print(f"  → faltarían ~{max(0, 50 - len(core))} especies comunes de relleno para llegar a 50")
    print(f"  → guardado en {out.relative_to(out.parents[2])}\n")


if __name__ == "__main__":
    main()
