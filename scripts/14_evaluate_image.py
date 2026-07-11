"""Evalúa el clasificador de imagen con Test-Time Augmentation (TTA).

Promedia la predicción sobre 10 vistas de cada foto (4 esquinas + centro, + sus
volteos horizontales) → mejora la métrica sin reentrenar. Reporta single-crop vs TTA,
F1 por especie (a CSV) y las peores clases.

    uv run python scripts/14_evaluate_image.py
    uv run python scripts/14_evaluate_image.py --backbone efficientnet_b3
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import build_label_map  # noqa: E402
from biomonitor.vision.dataset import IMAGENET_MEAN, IMAGENET_STD, SIZE  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=CONFIG["train_image"]["backbone"])
    return p.parse_args()


_resize = transforms.Resize(int(SIZE * 1.15))
_tencrop = transforms.TenCrop(SIZE)
_center = transforms.CenterCrop(SIZE)
_norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])


def views(img: Image.Image) -> tuple[torch.Tensor, torch.Tensor]:
    """Devuelve (vista central [1,3,H,W], 10 vistas TTA [10,3,H,W])."""
    r = _resize(img)
    center = _norm(_center(r)).unsqueeze(0)
    tta = torch.stack([_norm(c) for c in _tencrop(r)])
    return center, tta


def main() -> None:
    args = get_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    manifest = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    species = sorted(manifest["scientific_name"].unique())
    label_map = build_label_map(species)
    idx2sp = {v: k for k, v in label_map.items()}
    test = manifest[manifest["split"] == "test"].reset_index(drop=True)

    model = build_image_model(args.backbone, num_classes=len(label_map), pretrained=False)
    ckpt = PROJECT_ROOT / CONFIG["paths"]["models"] / f"image_{args.backbone}.pt"
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval().to(device)

    y_true, y_center, y_tta = [], [], []
    with torch.no_grad():
        for r in test.itertuples(index=False):
            try:
                img = Image.open(PROJECT_ROOT / r.file).convert("RGB")
            except Exception:  # noqa: BLE001
                continue
            center, tta = views(img)
            p_center = model(center.to(device)).softmax(1)[0]
            p_tta = model(tta.to(device)).softmax(1).mean(0)
            y_true.append(label_map[r.scientific_name])
            y_center.append(int(p_center.argmax()))
            y_tta.append(int(p_tta.argmax()))

    acc_c = accuracy_score(y_true, y_center)
    acc_t = accuracy_score(y_true, y_tta)
    f1_c = f1_score(y_true, y_center, average="macro", zero_division=0)
    f1_t = f1_score(y_true, y_tta, average="macro", zero_division=0)

    print("\n== Clasificador de imagen — evaluación TEST ==")
    print(f"  single-crop:   accuracy {acc_c:.3f} | F1 macro {f1_c:.3f}")
    print(f"  TTA (10-crop): accuracy {acc_t:.3f} | F1 macro {f1_t:.3f}  ({acc_t - acc_c:+.3f})")
    print(f"  meta del entregable: accuracy > 0.85")

    rep = classification_report([idx2sp[i] for i in y_true], [idx2sp[i] for i in y_tta],
                                output_dict=True, zero_division=0)
    per_class = pd.DataFrame(rep).T.loc[species][["precision", "recall", "f1-score", "support"]]
    out = PROJECT_ROOT / "reports" / f"image_{args.backbone}_per_class.csv"
    out.parent.mkdir(exist_ok=True)
    per_class.to_csv(out)
    print(f"\n  Peores 8 especies por F1 (TTA):")
    for sp, row in per_class.sort_values("f1-score").head(8).iterrows():
        print(f"    {row['f1-score']:.2f}  {sp:<28} (n_test={int(row['support'])})")
    print(f"\n  F1 por especie → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
