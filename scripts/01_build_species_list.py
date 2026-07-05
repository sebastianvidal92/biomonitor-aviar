"""Construye la lista candidata de especies objetivo desde el RCE del MMA.

Genera data/external/species_list_candidates.csv y un resumen por consola.
No requiere API keys.

    uv run python scripts/01_build_species_list.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import data_path  # noqa: E402
from biomonitor.data.conservation import THREATENED, load_rce_birds  # noqa: E402


def main() -> None:
    birds = load_rce_birds()
    out_path = data_path("external", "species_list_candidates.csv")
    birds.to_csv(out_path, index=False)

    focus = birds[birds["in_focus"] & birds["continental"]]
    threatened = birds[birds["category_primary"].isin(THREATENED)]

    print(f"\n== Lista candidata de especies (RCE MMA) ==\n")
    print(f"  Aves clasificadas en RCE:        {len(birds)}")
    print(f"  Continentales:                   {birds['continental'].sum()}")
    print(f"  Endémicas de Chile:              {birds['endemic'].sum()}")
    print(f"  Amenazadas (CR/EN/VU):           {len(threatened)}")
    print(f"  EN FOCO (endémica o amenazada,")
    print(f"           y continental):         {len(focus)}")
    print()
    print("  Distribución por categoría (foco):")
    print(focus["category_primary"].value_counts().to_string().replace("\n", "\n    "))
    print(f"\n  → guardado en {out_path.relative_to(out_path.parents[2])}")
    print(f"  → {len(focus)} candidatas en foco (objetivo del entregable: ≥50)\n")


if __name__ == "__main__":
    main()
