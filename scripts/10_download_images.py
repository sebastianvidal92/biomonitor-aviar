"""Descarga fotos (iNaturalist Research-grade vía GBIF) para las 55 especies.

Por especie baja hasta `image.max_per_species` fotos a data/raw/images/<slug>/ y
registra todo en un manifiesto (con licencia y geolocalización). Reanudable.

    uv run python scripts/10_download_images.py

NOTA: descarga masiva → la corre el usuario. Reanudable (salta lo ya bajado).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data import gbif_images as gi  # noqa: E402

MAX = CONFIG["image"]["max_per_species"]
MIN = CONFIG["image"]["min_per_species"]


def main() -> None:
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    img_root = PROJECT_ROOT / CONFIG["paths"]["raw"] / "images"

    manifest_rows = []
    for sp in tqdm(species, desc="Especies"):
        slug = sp.replace(" ", "_")
        dest_dir = img_root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            recs = gi.image_occurrences(sp, MAX, min_records=MIN)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  ⚠️  {sp}: error consultando GBIF ({exc})")
            continue

        for rec in recs:
            dest = dest_dir / f"{rec['gbif_id']}.jpg"
            if not dest.exists():
                try:
                    gi.download_image(rec["url"], dest)
                    time.sleep(0.1)  # cortesía
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"  ⚠️  {sp} {rec['gbif_id']}: descarga falló ({exc})")
                    continue
            manifest_rows.append({
                "scientific_name": sp, "slug": slug, "gbif_id": rec["gbif_id"],
                "file": str(dest.relative_to(PROJECT_ROOT)),
                "license": rec["license"], "country": rec["country"],
                "lat": rec["lat"], "lon": rec["lon"], "source": rec["source"],
            })

    manifest = pd.DataFrame(manifest_rows)
    man_path = data_path("interim", "image_manifest.csv")
    manifest.to_csv(man_path, index=False)

    per_sp = manifest.groupby("scientific_name").size()
    print(f"\n== Descarga de imágenes completada ==")
    print(f"  Especies con fotos: {manifest['scientific_name'].nunique()}/{len(species)}")
    print(f"  Fotos totales: {len(manifest)}")
    print(f"  Por especie: min={per_sp.min()} · mediana={int(per_sp.median())} · max={per_sp.max()}")
    print(f"  Manifiesto: {man_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
