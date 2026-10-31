"""Entrena el clasificador de cantos (baseline CNN sobre espectrogramas).

Usa el audio cacheado (data/processed/audio32_manifest.csv). Corre en la GPU de
Apple (MPS) si está disponible. Guarda el mejor modelo en models/ y evalúa en test.

Ejemplos:
    uv run python scripts/07_train_audio.py                 # usa config.yaml
    uv run python scripts/07_train_audio.py --epochs 30 --batch-size 48
    uv run python scripts/07_train_audio.py --backbone efficientnet_b3

NOTA: entrenamiento = cómputo pesado → lo corre el usuario en su Mac.
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
from biomonitor.audio.dataset import datasets_from_manifest  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402


def get_args() -> argparse.Namespace:
    c = CONFIG["train_audio"]
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", default=c["backbone"])
    p.add_argument("--batch-size", type=int, default=c["batch_size"])
    p.add_argument("--epochs", type=int, default=c["epochs"])
    p.add_argument("--lr", type=float, default=c["lr"])
    p.add_argument("--weight-decay", type=float, default=c["weight_decay"])
    p.add_argument("--num-workers", type=int, default=c["num_workers"])
    p.add_argument("--mixup-alpha", type=float, default=c["mixup_alpha"])
    p.add_argument("--label-smoothing", type=float, default=c["label_smoothing"])
    p.add_argument("--drop-rate", type=float, default=0.3, help="dropout del clasificador")
    p.add_argument("--drop-path", type=float, default=0.2, help="profundidad estocástica")
    p.add_argument("--no-augment", action="store_true", help="sin SpecAugment, ganancia ni ruido")
    p.add_argument("--tag", default="", help="sufijo del checkpoint y del log (ej. 'sinreg'); vacío = modelo principal")
    return p.parse_args()


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def class_weights(manifest: pd.DataFrame, label_map: dict[str, int]) -> torch.Tensor:
    """Peso inverso a la frecuencia por clase (para el leve desbalance)."""
    counts = manifest[manifest["split"] == "train"]["scientific_name"].map(label_map).value_counts()
    w = np.ones(len(label_map), dtype=np.float32)
    for idx, n in counts.items():
        w[idx] = 1.0 / n
    w = w / w.mean()
    return torch.tensor(w)


def mixup(x: torch.Tensor, y: torch.Tensor, alpha: float):
    """Mezcla convexa de pares del batch. Devuelve x_mix, y_b, lam (y_a es y original)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y[idx], lam


def run_epoch(model, loader, device, criterion, optimizer=None,
              mixup_alpha: float = 0.0) -> tuple[float, float, float]:
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
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return float(np.mean(losses)), acc, f1


def main() -> None:
    args = get_args()
    torch.multiprocessing.set_sharing_strategy("file_system")  # evita 'Too many open files' en macOS
    device = pick_device()
    print(f"Dispositivo: {device} | backbone: {args.backbone}\n")

    manifest = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    train_ds, val_ds, test_ds, label_map = datasets_from_manifest(manifest, species, augment=not args.no_augment)

    dl = lambda ds, shuf, drop=False: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=shuf,
        num_workers=args.num_workers, drop_last=drop,
    )
    train_dl = dl(train_ds, True, True)  # drop_last=True: estabiliza BatchNorm y mixup
    val_dl, test_dl = dl(val_ds, False), dl(test_ds, False)

    model = build_audio_model(args.backbone, num_classes=len(label_map),
                              drop_rate=args.drop_rate, drop_path_rate=args.drop_path).to(device)
    ls = args.label_smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights(manifest, label_map).to(device), label_smoothing=ls
    )
    print(f"regularización: mixup_alpha={args.mixup_alpha} | label_smoothing={ls} | dropout={args.drop_rate} | "
          f"drop_path={args.drop_path} | augment={not args.no_augment}\n")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    models_dir = PROJECT_ROOT / CONFIG["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.tag}" if args.tag else ""
    ckpt = models_dir / f"audio_{args.backbone}{suffix}.pt"
    best_f1 = 0.0
    history = []

    print(f"{'época':>6} {'tr_loss':>8} {'val_loss':>9} {'val_acc':>8} {'val_f1':>8}  {'':>4}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_dl, device, criterion, optimizer,
                                  mixup_alpha=args.mixup_alpha)
        val_loss, val_acc, val_f1 = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            flag = "★ best"
        history.append(dict(ep=epoch, tr_loss=tr_loss, val_loss=val_loss,
                            val_acc=val_acc, val_f1=val_f1))
        print(f"{epoch:>6} {tr_loss:>8.3f} {val_loss:>9.3f} {val_acc:>8.3f} "
              f"{val_f1:>8.3f}  {flag:>6}  ({time.time()-t0:.0f}s)")

    log_dir = PROJECT_ROOT / "reports" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_dir / f"audio_{args.backbone}{suffix}.csv", index=False)

    # Evaluación final en test con el mejor checkpoint.
    model.load_state_dict(torch.load(ckpt, map_location=device))
    _, test_acc, test_f1 = run_epoch(model, test_dl, device, criterion)
    label_map_path = models_dir / f"audio_{args.backbone}{suffix}_labels.json"
    label_map_path.write_text(json.dumps(label_map, ensure_ascii=False, indent=2))

    print(f"\n== Resultado final (mejor modelo en TEST) ==")
    print(f"  accuracy: {test_acc:.3f}  |  F1 macro: {test_f1:.3f}")
    print(f"  meta del entregable: accuracy > 0.85")
    print(f"  modelo: {ckpt.relative_to(PROJECT_ROOT)}")
    print(f"  etiquetas: {label_map_path.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
