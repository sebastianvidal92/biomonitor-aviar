"""Evalúa el clasificador de audio con AGREGACIÓN por grabación.

En vez de una sola ventana de 5 s, promedia las probabilidades sobre varias
ventanas de cada grabación (test-time aggregation) → refleja el desempeño real de
despliegue y suele mejorar la métrica sin reentrenar. Reporta accuracy, F1 macro,
F1 por especie (guardado a CSV) y las peores clases.

    uv run python scripts/08_evaluate_audio.py
    uv run python scripts/08_evaluate_audio.py --backbone efficientnet_b0 --max-windows 12
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import accuracy_score, classification_report, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR, build_label_map  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=CONFIG["train_audio"]["backbone"])
    p.add_argument("--max-windows", type=int, default=12)
    p.add_argument("--tag", default="", help="sufijo del checkpoint (ej. 'sinreg'); vacío = modelo principal")
    return p.parse_args()


def recording_windows(path: str, n: int, max_windows: int) -> np.ndarray:
    """Ventanas de n muestras que cubren la grabación (hasta max_windows)."""
    total = sf.info(path).frames
    if total <= 0:
        return np.zeros((1, n), dtype=np.float32)
    if total <= n:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        reps = int(np.ceil(n / max(len(y), 1)))
        return np.tile(y, reps)[:n][None, :]
    n_win = min(max_windows, total // n)
    starts = np.linspace(0, total - n, num=n_win, dtype=int)
    wins = []
    for s in starts:
        y, _ = sf.read(path, start=int(s), frames=n, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        if len(y) < n:  # seeking impreciso en algunos formatos: fuerza largo exacto
            y = np.pad(y, (0, n - len(y)))
        wins.append(y[:n])
    return np.stack(wins)


def main() -> None:
    args = get_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    manifest = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    label_map = build_label_map(species)
    idx2sp = {v: k for k, v in label_map.items()}
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)

    model = build_audio_model(args.backbone, num_classes=len(label_map), pretrained=False)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = PROJECT_ROOT / CONFIG["paths"]["models"] / f"audio_{args.backbone}{suffix}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)

    melspec = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def logmel(wav: torch.Tensor) -> torch.Tensor:  # [W, n] -> [W, 1, n_mels, frames]
        mel = to_db(melspec(wav))
        mu = mel.mean(dim=(1, 2), keepdim=True)
        sd = mel.std(dim=(1, 2), keepdim=True) + 1e-6
        return ((mel - mu) / sd).unsqueeze(1)

    y_true, y_pred_agg, y_pred_center = [], [], []
    with torch.no_grad():
        for r in test.itertuples(index=False):
            wins = recording_windows(str(PROJECT_ROOT / r.cached_file), CLIP_SAMPLES, args.max_windows)
            x = logmel(torch.from_numpy(wins)).to(device)
            probs = model(x).softmax(dim=1)          # [W, C]
            y_true.append(label_map[r.scientific_name])
            y_pred_agg.append(int(probs.mean(0).argmax()))          # agregado
            y_pred_center.append(int(probs[len(wins) // 2].argmax()))  # ventana central

    acc_c = accuracy_score(y_true, y_pred_center)
    acc_a = accuracy_score(y_true, y_pred_agg)
    f1_c = f1_score(y_true, y_pred_center, average="macro", zero_division=0)
    f1_a = f1_score(y_true, y_pred_agg, average="macro", zero_division=0)

    print("\n== Evaluación en TEST ==")
    print(f"  ventana central:   accuracy {acc_c:.3f} | F1 macro {f1_c:.3f}")
    print(f"  AGREGADO ({args.max_windows} vent.): accuracy {acc_a:.3f} | F1 macro {f1_a:.3f}  "
          f"({acc_a - acc_c:+.3f})")
    print(f"  meta del entregable: accuracy > 0.85")

    # F1 por especie (con el agregado) → CSV + peores clases.
    rep = classification_report(
        [idx2sp[i] for i in y_true], [idx2sp[i] for i in y_pred_agg],
        output_dict=True, zero_division=0,
    )
    per_class = pd.DataFrame(rep).T.loc[sorted(label_map)][["precision", "recall", "f1-score", "support"]]
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    out = reports_dir / f"audio_{args.backbone}{suffix}_per_class.csv"
    per_class.to_csv(out)
    print(f"\n  Peores 8 especies por F1 (agregado):")
    for sp, row in per_class.sort_values("f1-score").head(8).iterrows():
        print(f"    {row['f1-score']:.2f}  {sp:<28} (n_test={int(row['support'])})")
    print(f"\n  F1 por especie → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
