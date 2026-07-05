"""Construye la lista FINAL de especies del clasificador (≥50).

Composición (decidida con el usuario):
  1. Núcleo de conservación: especies en foco (terrestres/costeras) con ≥ umbral
     de grabaciones (config.audio.min_recordings_per_species).
  2. Endémicas de Chile continental (aunque no estén amenazadas).
  3. Relleno: aves chilenas comunes (más grabadas en Xeno-canto) hasta llegar a ≥50.

Barre Xeno-canto (grp:birds cnt:Chile) para rankear las comunes por nº de
grabaciones. Guarda data/external/species_final.csv.

    uv run python scripts/03_build_final_list.py
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, data_path  # noqa: E402
from biomonitor.data import xenocanto  # noqa: E402
from biomonitor.data.conservation import CONTINENTAL_ENDEMICS  # noqa: E402

TARGET = 55  # objetivo con margen sobre el mínimo de 50
MIN_REC = CONFIG["audio"]["min_recordings_per_species"]


def crawl_chile_species() -> pd.DataFrame:
    """Cuenta grabaciones por especie en Chile barriendo Xeno-canto."""
    counts: dict[tuple[str, str], int] = defaultdict(int)
    names: dict[tuple[str, str], str] = {}
    # Placeholders de Xeno-canto para grabaciones sin identificar (no son especies).
    non_species = {"mystery", "soundscape", "identity"}
    print("Barriendo Xeno-canto (grp:birds cnt:Chile)...")
    for rec in xenocanto.iter_search("grp:birds cnt:Chile"):
        gen, sp = rec.get("gen", ""), rec.get("sp", "")
        if gen.lower() in non_species or sp.lower() in non_species:
            continue
        key = (gen, sp)
        counts[key] += 1
        names.setdefault(key, rec.get("en", ""))
    rows = [
        {"scientific_name": f"{g} {s}".strip(), "en_name": names[(g, s)], "xc_chile": n}
        for (g, s), n in counts.items()
        if g and s
    ]
    df = pd.DataFrame(rows).sort_values("xc_chile", ascending=False).reset_index(drop=True)
    print(f"  {len(df)} especies de aves con grabaciones en Chile.\n")
    return df


def main() -> None:
    enriched = pd.read_csv(data_path("external", "species_focus_enriched.csv"))

    # 1) Núcleo de conservación entrenable (no pelágico, audio suficiente).
    core = enriched[
        enriched["habitat_group"].isin(["terrestrial", "coastal_shorebird"])
        & (enriched["xc_global"] >= MIN_REC)
    ].copy()
    core["role"] = "conservacion"

    chile = crawl_chile_species()
    chile_by_name = chile.set_index("scientific_name")["xc_chile"].to_dict()
    selected = {r["scientific_name"] for _, r in core.iterrows()}

    # 2) Endémicas continentales con datos (las que aún no estén en el núcleo).
    extra_rows = []
    for sp in sorted(CONTINENTAL_ENDEMICS):
        if sp in selected:
            continue
        n = chile_by_name.get(sp, 0)
        if n >= MIN_REC:
            extra_rows.append({"scientific_name": sp, "xc_chile": n, "role": "endemica"})
            selected.add(sp)

    # 3) Relleno con comunes hasta llegar al objetivo.
    for _, r in chile.iterrows():
        if len(selected) >= TARGET:
            break
        sp = r["scientific_name"]
        if sp in selected or r["xc_chile"] < MIN_REC:
            continue
        extra_rows.append({"scientific_name": sp, "xc_chile": r["xc_chile"], "role": "comun"})
        selected.add(sp)

    extra = pd.DataFrame(extra_rows)
    extra["endemic"] = extra["scientific_name"].isin(CONTINENTAL_ENDEMICS)

    cols = ["scientific_name", "common_name", "category_primary", "endemic",
            "xc_global", "xc_chile", "gbif_photos_cl", "habitat_group", "role"]
    core_out = core.reindex(columns=cols)
    extra_out = extra.reindex(columns=cols)
    final = pd.concat([core_out, extra_out], ignore_index=True)

    out = data_path("external", "species_final.csv")
    final.to_csv(out, index=False)

    print("== Lista FINAL de especies del clasificador ==\n")
    print(f"  Total: {len(final)}  (objetivo ≥50)")
    print(f"  Por rol: {final['role'].value_counts().to_dict()}")
    print(f"  Endémicas: {int(final['endemic'].sum())}")
    print(f"  → guardado en {out.relative_to(out.parents[2])}\n")


if __name__ == "__main__":
    main()
