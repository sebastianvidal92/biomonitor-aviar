"""Descarga audio de Xeno-canto para las especies de species_final.csv.

Por especie: selecciona hasta `max_recordings_per_species` grabaciones de calidad
A/B/C, las baja a data/raw/audio/<Genus_species>/XC<id>.<ext> y registra todo en
un manifiesto. Es reanudable: salta archivos ya descargados.

    uv run python scripts/04_download_audio.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data import xenocanto  # noqa: E402

MAX = CONFIG["audio"]["max_recordings_per_species"]
QUALITIES = CONFIG["audio"]["qualities"]
MAX_LEN = CONFIG["audio"]["max_length_seconds"]


def main() -> None:
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    audio_root = PROJECT_ROOT / CONFIG["paths"]["raw"] / "audio"

    manifest_rows = []
    for sp in tqdm(species, desc="Especies"):
        slug = sp.replace(" ", "_")
        dest_dir = audio_root / slug
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            recs = xenocanto.select_recordings(sp, QUALITIES, MAX, max_length_seconds=MAX_LEN)
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  ⚠️  {sp}: error consultando ({exc})")
            continue

        for rec in recs:
            xc_id = rec["id"]
            ext = Path(rec.get("file-name", f"{xc_id}.mp3")).suffix or ".mp3"
            dest = dest_dir / f"XC{xc_id}{ext}"
            if not dest.exists():
                try:
                    xenocanto.download_audio(rec["file"], dest)
                    time.sleep(0.2)  # cortesía
                except Exception as exc:  # noqa: BLE001
                    tqdm.write(f"  ⚠️  {sp} XC{xc_id}: descarga falló ({exc})")
                    continue
            manifest_rows.append({
                "scientific_name": sp, "slug": slug, "xc_id": xc_id,
                "file": str(dest.relative_to(PROJECT_ROOT)),
                "quality": rec.get("q"), "length": rec.get("length"),
                "type": rec.get("type"), "country": rec.get("cnt"),
                "license": rec.get("lic"), "lat": rec.get("lat"), "lon": rec.get("lon"),
            })

    manifest = pd.DataFrame(manifest_rows)
    man_path = data_path("interim", "audio_manifest.csv")
    manifest.to_csv(man_path, index=False)

    print(f"\n== Descarga de audio completada ==")
    print(f"  Especies con audio: {manifest['scientific_name'].nunique()}/{len(species)}")
    print(f"  Grabaciones totales: {len(manifest)}")
    print(f"  Manifiesto: {man_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
