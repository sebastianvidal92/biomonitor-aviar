"""Fusión multimodal tardía audio + imagen (Módulo 1).

Combina las probabilidades del clasificador de audio (55 clases) y el de imagen
(53 clases) para la predicción final. Como los datasets NO están pareados (audio de
Xeno-canto, fotos de iNaturalist), la fusión se evalúa con PAREO POR ESPECIE: se
emparejan un audio y una foto de la misma especie (simula un observador que graba y
fotografía el mismo individuo). Reporta audio-solo vs imagen-solo vs fusión sobre las
mismas instancias pareadas.

    uv run python scripts/13_multimodal_fusion.py
    uv run python scripts/13_multimodal_fusion.py --audio-backbone efficientnet_b0 --w-audio 0.5

NOTA: inferencia sobre ambos test sets → la corre el usuario (unos minutos).
"""

from __future__ import annotations

import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR, build_label_map  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402
from biomonitor.vision.dataset import _eval_tf  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402

PAIRS_PER_SPECIES = 30
SEED = 42


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--audio-backbone", default=CONFIG["train_audio"]["backbone"])
    p.add_argument("--image-backbone", default=CONFIG["train_image"]["backbone"])
    p.add_argument("--w-audio", type=float, default=0.5, help="peso del audio en la fusión")
    p.add_argument("--max-windows", type=int, default=12)
    return p.parse_args()


# ---------- inferencia de audio (agregada por grabación) ----------
def _audio_windows(path: str, n: int, max_windows: int) -> np.ndarray:
    total = sf.info(path).frames
    if total <= n:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        reps = int(np.ceil(n / max(len(y), 1)))
        return np.tile(y, reps)[:n][None, :]
    starts = np.linspace(0, total - n, num=min(max_windows, total // n), dtype=int)
    wins = []
    for s in starts:
        y, _ = sf.read(path, start=int(s), frames=n, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        wins.append(np.pad(y, (0, max(0, n - len(y))))[:n])
    return np.stack(wins)


def audio_probs_by_species(model, manifest, device, max_windows) -> dict[str, list]:
    melspec = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    out: dict[str, list] = defaultdict(list)
    with torch.no_grad():
        for r in manifest.itertuples(index=False):
            wins = _audio_windows(str(PROJECT_ROOT / r.cached_file), CLIP_SAMPLES, max_windows)
            mel = to_db(melspec(torch.from_numpy(wins)))
            mel = (mel - mel.mean(dim=(1, 2), keepdim=True)) / (mel.std(dim=(1, 2), keepdim=True) + 1e-6)
            probs = model(mel.unsqueeze(1).to(device)).softmax(1).mean(0).cpu().numpy()
            out[r.scientific_name].append(probs)
    return out


# ---------- inferencia de imagen (por foto) ----------
def image_probs_by_species(model, manifest, device, img_to_audio, n_audio) -> dict[str, list]:
    tf = _eval_tf()
    out: dict[str, list] = defaultdict(list)
    with torch.no_grad():
        for r in manifest.itertuples(index=False):
            try:
                img = Image.open(PROJECT_ROOT / r.file).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            p = model(tf(img).unsqueeze(0).to(device)).softmax(1)[0].cpu().numpy()
            p55 = np.zeros(n_audio, dtype=np.float32)
            for j, val in enumerate(p):
                p55[img_to_audio[j]] = val
            out[r.scientific_name].append(p55)
    return out


def main() -> None:
    args = get_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    models_dir = PROJECT_ROOT / CONFIG["paths"]["models"]

    # --- audio (55 clases) ---
    audio_species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    audio_lm = build_label_map(audio_species)
    a_model = build_audio_model(args.audio_backbone, num_classes=len(audio_lm), pretrained=False)
    a_model.load_state_dict(torch.load(models_dir / f"audio_{args.audio_backbone}.pt", map_location=device))
    a_model.eval().to(device)
    a_manifest = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    a_test = a_manifest[a_manifest["split"] == "test"]

    # --- imagen (53 clases) ---
    i_manifest = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    image_species = sorted(i_manifest["scientific_name"].unique())
    image_lm = build_label_map(image_species)
    img_to_audio = {i: audio_lm[sp] for sp, i in image_lm.items()}  # 53 → 55
    i_model = build_image_model(args.image_backbone, num_classes=len(image_lm), pretrained=False)
    i_model.load_state_dict(torch.load(models_dir / f"image_{args.image_backbone}.pt", map_location=device))
    i_model.eval().to(device)
    i_test = i_manifest[i_manifest["split"] == "test"]

    print("Calculando probabilidades de audio (test)...")
    a_probs = audio_probs_by_species(a_model, a_test, device, args.max_windows)
    print("Calculando probabilidades de imagen (test)...")
    i_probs = image_probs_by_species(i_model, i_test, device, img_to_audio, len(audio_lm))

    # --- evaluación por pareo (especies presentes en ambas modalidades) ---
    common = sorted(set(a_probs) & set(i_probs))
    rng = random.Random(SEED)
    wa, wi = args.w_audio, 1 - args.w_audio
    y_true, y_a, y_i, y_f = [], [], [], []
    for sp in common:
        target = audio_lm[sp]
        for _ in range(PAIRS_PER_SPECIES):
            Pa = rng.choice(a_probs[sp])
            Pi = rng.choice(i_probs[sp])
            y_true.append(target)
            y_a.append(int(Pa.argmax()))
            y_i.append(int(Pi.argmax()))
            y_f.append(int((wa * Pa + wi * Pi).argmax()))

    def m(y):
        return accuracy_score(y_true, y), f1_score(y_true, y, average="macro", zero_division=0)

    aa, af1 = m(y_a); ia, if1 = m(y_i); fa, ff1 = m(y_f)
    print(f"\n== Fusión multimodal (pareo por especie, {len(common)} especies, "
          f"{len(y_true)} pares) ==\n")
    print(f"  {'modalidad':<22}{'accuracy':>10}{'F1 macro':>10}")
    print(f"  {'audio solo':<22}{aa:>10.3f}{af1:>10.3f}")
    print(f"  {'imagen sola':<22}{ia:>10.3f}{if1:>10.3f}")
    print(f"  {'FUSIÓN (w_audio=%.2f)' % wa:<22}{fa:>10.3f}{ff1:>10.3f}")
    print(f"\n  meta del entregable: accuracy > 0.85\n")


if __name__ == "__main__":
    main()
