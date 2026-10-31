"""Entrena el clasificador de imagen (EfficientNet-B3 / ViT, transfer learning).

Usa las fotos particionadas (data/interim/image_manifest_split.csv). Cada foto es un
ejemplo independiente → evaluación por foto (sin agregación). Corre en MPS.

    uv run python scripts/12_train_image.py
    uv run python scripts/12_train_image.py --backbone vit_small_patch16_224.augreg_in21k_ft_in1k \
        --size 224 --lr 1e-4 --epochs 30                          # transformer de visión (ViT-S/16)
    uv run python scripts/12_train_image.py --backbone vit_base_patch16_384 --epochs 20

NOTA: entrenamiento = cómputo pesado → lo corre el usuario.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.vision.dataset import image_datasets_from_manifest  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402


def get_args() -> argparse.Namespace:
    c = CONFIG["train_image"]
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=c["backbone"])
    p.add_argument("--batch-size", type=int, default=c["batch_size"])
    p.add_argument("--epochs", type=int, default=c["epochs"])
    p.add_argument("--lr", type=float, default=c["lr"])
    p.add_argument("--weight-decay", type=float, default=c["weight_decay"])
    p.add_argument("--num-workers", type=int, default=c["num_workers"])
    p.add_argument("--mixup-alpha", type=float, default=c["mixup_alpha"])
    p.add_argument("--resume", action="store_true", help="continuar desde el último checkpoint")
    p.add_argument("--init-from", default=None, help="warm-start: cargar pesos de un .pt")
    p.add_argument("--size", type=int, default=CONFIG["image"]["size"], help="lado de la imagen de entrada (ViT: 224)")
    return p.parse_args()


def class_weights(manifest, label_map) -> torch.Tensor:
    counts = manifest[manifest["split"] == "train"]["scientific_name"].map(label_map).value_counts()
    w = np.ones(len(label_map), dtype=np.float32)
    for idx, n in counts.items():
        w[idx] = 1.0 / n
    return torch.tensor(w / w.mean())


def mixup(x, y, alpha):
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y[idx], lam


def run_epoch(model, loader, device, criterion, optimizer=None, mixup_alpha=0.0):
    train = optimizer is not None
    model.train(train)
    losses, y_true, y_pred = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            if train and mixup_alpha > 0:
                x_in, y_b, lam = mixup(x, y, mixup_alpha)
                logits = model(x_in)
                loss = lam * criterion(logits, y) + (1 - lam) * criterion(logits, y_b)
            else:
                logits = model(x)
                loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        losses.append(loss.item())
        y_true.extend(y.cpu().tolist())
        y_pred.extend(logits.argmax(1).cpu().tolist())
    return (float(np.mean(losses)),
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average="macro", zero_division=0))


def main() -> None:
    args = get_args()
    # Evita "Too many open files" con DataLoader workers en macOS.
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    manifest = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    species = sorted(manifest["scientific_name"].unique())
    train_ds, val_ds, test_ds, label_map = image_datasets_from_manifest(manifest, species, size=args.size)
    print(f"Dispositivo: {device} | backbone: {args.backbone} | entrada: {args.size} px | clases: {len(label_map)}\n")

    dl = lambda ds, shuf, drop=False: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=shuf,
        num_workers=args.num_workers, drop_last=drop)
    train_dl = dl(train_ds, True, True)
    val_dl, test_dl = dl(val_ds, False), dl(test_ds, False)

    model = build_image_model(args.backbone, num_classes=len(label_map)).to(device)
    ls = CONFIG["train_image"]["label_smoothing"]
    criterion = nn.CrossEntropyLoss(weight=class_weights(manifest, label_map).to(device),
                                    label_smoothing=ls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    models_dir = PROJECT_ROOT / CONFIG["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / f"image_{args.backbone}.pt"
    ckpt_last = models_dir / f"image_{args.backbone}_last.pt"
    best_f1, start_epoch = 0.0, 1

    if args.init_from:  # warm-start: solo pesos (ej. continuar un modelo ya entrenado)
        model.load_state_dict(torch.load(args.init_from, map_location=device))
        print(f"Warm-start desde {args.init_from}\n")
    if args.resume and ckpt_last.exists():  # reanudar estado completo tras un crash
        st = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(st["model"]); optimizer.load_state_dict(st["opt"])
        scheduler.load_state_dict(st["sched"]); best_f1 = st["best_f1"]; start_epoch = st["epoch"] + 1
        print(f"Reanudando desde época {start_epoch} (best_f1={best_f1:.3f})\n")

    log_dir = PROJECT_ROOT / "reports" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_csv = log_dir / f"image_{args.backbone}.csv"
    history = pd.read_csv(log_csv).to_dict("records") if (args.resume and log_csv.exists()) else []

    print(f"{'época':>6} {'tr_loss':>8} {'val_acc':>8} {'val_f1':>8}")
    for epoch in range(start_epoch, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_dl, device, criterion, optimizer, args.mixup_alpha)
        _, val_acc, val_f1 = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            flag = "★ best"
        torch.save({"model": model.state_dict(), "opt": optimizer.state_dict(),
                    "sched": scheduler.state_dict(), "epoch": epoch, "best_f1": best_f1}, ckpt_last)
        history.append(dict(ep=epoch, tr_loss=tr_loss, val_acc=val_acc, val_f1=val_f1))
        pd.DataFrame(history).to_csv(log_csv, index=False)  # incremental (sobrevive reanudación)
        print(f"{epoch:>6} {tr_loss:>8.3f} {val_acc:>8.3f} {val_f1:>8.3f}  {flag}  ({time.time()-t0:.0f}s)")

    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc, test_f1 = run_epoch(model, test_dl, device, criterion)
    (models_dir / f"image_{args.backbone}_labels.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2))

    print(f"\n== Clasificador de imagen — TEST ==")
    print(f"  accuracy {test_acc:.3f} | F1 macro {test_f1:.3f}  ({len(label_map)} especies)")
    print(f"  modelo: {ckpt.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
