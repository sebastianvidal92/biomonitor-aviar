"""Resamplea y cachea todo el audio a 32 kHz mono (.flac) para entrenamiento.

Lee data/interim/audio_manifest_split.csv, cachea cada grabación en
data/processed/audio32/<slug>/XC<id>.flac y escribe un manifiesto con la duración
y el split. Reanudable y auto-reparable: salta lo ya cacheado (con frames > 0).

    uv run python scripts/06_preprocess_audio.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import soundfile as sf
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.preprocess import TARGET_SR, resample_and_cache  # noqa: E402
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool

def main() -> None:
    m = pd.read_csv(data_path("interim", "audio_manifest_split.csv"))
    cache_root = PROJECT_ROOT / CONFIG["paths"]["processed"] / "audio32"

    rows = []
    for r in tqdm(m.itertuples(index=False), total=len(m), desc="Cacheando"):
        src = PROJECT_ROOT / r.file
        dst = cache_root / r.slug / f"XC{r.xc_id}.flac"
        try:
            # Auto-reparable: re-cachea si no existe o si quedó con 0 frames.
            if dst.exists() and sf.info(str(dst)).frames > 0:
                dur = sf.info(str(dst)).duration
            else:
                with ProcessPoolExecutor(max_workers=1) as ex:
                    dur = ex.submit(resample_and_cache, src, dst).result()
        except BrokenProcessPool:
            tqdm.write(f"  💥 crash nativo en {r.slug}/XC{r.xc_id}, saltando")
            continue
        except Exception as exc:  # noqa: BLE001
            tqdm.write(f"  ⚠️  {r.slug}/XC{r.xc_id}: {exc}")
            continue

        rows.append({
            "scientific_name": r.scientific_name, "slug": r.slug, "xc_id": r.xc_id,
            "cached_file": str(dst.relative_to(PROJECT_ROOT)),
            "duration_sec": round(dur, 2), "split": r.split,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        print("No se cacheó ninguna grabación — revisa data/interim/audio_manifest_split.csv")
        return
    man = data_path("processed", "audio32_manifest.csv")
    out.to_csv(man, index=False)

    size_gb = sum(f.stat().st_size for f in cache_root.rglob("*.flac")) / 1e9
    print(f"\n== Audio cacheado a {TARGET_SR} Hz mono (.flac) ==")
    print(f"  Grabaciones cacheadas: {len(out)}/{len(m)}")
    print(f"  Duración total: {out['duration_sec'].sum()/3600:.1f} h")
    print(f"  Tamaño cache: {size_gb:.2f} GB  (raw eran 7.3 GB)")
    print(f"  Manifiesto: {man.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
