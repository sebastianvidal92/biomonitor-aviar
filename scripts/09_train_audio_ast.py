"""Entrena el AST (Audio Spectrogram Transformer) — modelo avanzado del Módulo 1.

Fine-tuning de AST preentrenado en AudioSet sobre las 55 especies. Reporta test con
agregación por grabación (comparable al CNN). Corre en MPS.

    uv run python scripts/09_train_audio_ast.py
    uv run python scripts/09_train_audio_ast.py --epochs 15 --batch-size 12

NOTA: cómputo pesado (AST ~86M params) → lo corre el usuario. Más lento que el CNN.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.ast_dataset import (  # noqa: E402
    AST_SR, CLIP_SECONDS, SRC_WINDOW, ast_datasets_from_manifest, feature_extractor,
)
from biomonitor.audio.ast_model import build_ast_model  # noqa: E402
from biomonitor.audio.dataset import SR as SRC_SR  # noqa: E402


def get_args() -> argparse.Namespace:
    c = CONFIG["train_ast"]
    p = argparse.ArgumentParser()
    p.add_argument("--batch-size", type=int, default=c["batch_size"])
    p.add_argument("--epochs", type=int, default=c["epochs"])
    p.add_argument("--lr", type=float, default=c["lr"])
    p.add_argument("--weight-decay", type=float, default=c["weight_decay"])
    p.add_argument("--num-workers", type=int, default=c["num_workers"])
    p.add_argument("--max-windows", type=int, default=c["max_windows_eval"])
    return p.parse_args()


def class_weights(manifest, label_map) -> torch.Tensor:
    counts = manifest[manifest["split"] == "train"]["scientific_name"].map(label_map).value_counts()
    w = np.ones(len(label_map), dtype=np.float32)
    for idx, n in counts.items():
        w[idx] = 1.0 / n
    return torch.tensor(w / w.mean())


def run_epoch(model, loader, device, criterion, optimizer=None):
    train = optimizer is not None
    model.train(train)
    losses, y_true, y_pred = [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(input_values=x).logits
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


def recording_windows_16k(path: str, max_windows: int) -> np.ndarray:
    """Varias ventanas de ~10 s resampleadas a 16 kHz que cubren la grabación."""
    total = sf.info(path).frames
    n = SRC_WINDOW
    if total <= n:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        wins = [y]
    else:
        starts = np.linspace(0, total - n, num=min(max_windows, total // n), dtype=int)
        wins = []
        for s in starts:
            y, _ = sf.read(path, start=int(s), frames=n, dtype="float32", always_2d=False)
            if y.ndim > 1:
                y = y.mean(axis=1)
            wins.append(y)
    return [torchaudio.functional.resample(torch.from_numpy(w), SRC_SR, AST_SR).numpy() for w in wins]


@torch.no_grad()
def evaluate_test(model, test_df, label_map, device, fe, max_windows):
    """Accuracy/F1 en test: ventana central y agregado por grabación."""
    idx = {v: k for k, v in label_map.items()}
    y_true, y_agg, y_center = [], [], []
    for r in test_df.itertuples(index=False):
        wins = recording_windows_16k(str(PROJECT_ROOT / r.cached_file), max_windows)
        feats = fe(wins, sampling_rate=AST_SR, return_tensors="pt")["input_values"].to(device)
        probs = model(input_values=feats).logits.softmax(dim=1)
        y_true.append(label_map[r.scientific_name])
        y_agg.append(int(probs.mean(0).argmax()))
        y_center.append(int(probs[len(wins) // 2].argmax()))
    return (accuracy_score(y_true, y_center), f1_score(y_true, y_center, average="macro", zero_division=0),
            accuracy_score(y_true, y_agg), f1_score(y_true, y_agg, average="macro", zero_division=0))


def main() -> None:
    args = get_args()
    torch.multiprocessing.set_sharing_strategy("file_system")  # evita 'Too many open files' en macOS
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Dispositivo: {device} | modelo: AST ({CLIP_SECONDS}s @ {AST_SR}Hz)\n")

    manifest = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    train_ds, val_ds, test_ds, label_map = ast_datasets_from_manifest(manifest, species)

    dl = lambda ds, shuf, drop=False: DataLoader(  # noqa: E731
        ds, batch_size=args.batch_size, shuffle=shuf,
        num_workers=args.num_workers, drop_last=drop)
    train_dl, val_dl = dl(train_ds, True, True), dl(val_ds, False)

    model = build_ast_model(len(label_map)).to(device)
    ls = CONFIG["train_ast"]["label_smoothing"]
    criterion = nn.CrossEntropyLoss(weight=class_weights(manifest, label_map).to(device),
                                    label_smoothing=ls)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    models_dir = PROJECT_ROOT / CONFIG["paths"]["models"]
    models_dir.mkdir(parents=True, exist_ok=True)
    ckpt = models_dir / "audio_ast.pt"
    best_f1 = 0.0
    history = []

    print(f"{'época':>6} {'tr_loss':>8} {'val_acc':>8} {'val_f1':>8}")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, _, _ = run_epoch(model, train_dl, device, criterion, optimizer)
        _, val_acc, val_f1 = run_epoch(model, val_dl, device, criterion)
        scheduler.step()
        flag = ""
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), ckpt)
            flag = "★ best"
        history.append(dict(ep=epoch, tr_loss=tr_loss, val_acc=val_acc, val_f1=val_f1))
        print(f"{epoch:>6} {tr_loss:>8.3f} {val_acc:>8.3f} {val_f1:>8.3f}  {flag}  ({time.time()-t0:.0f}s)")

    log_dir = PROJECT_ROOT / "reports" / "training_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(history).to_csv(log_dir / "audio_ast.csv", index=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    acc_c, f1_c, acc_a, f1_a = evaluate_test(
        model, test_ds.df, label_map, device, feature_extractor(), args.max_windows)
    (models_dir / "audio_ast_labels.json").write_text(
        json.dumps(label_map, ensure_ascii=False, indent=2))

    print(f"\n== AST — Resultado en TEST ==")
    print(f"  ventana central:  accuracy {acc_c:.3f} | F1 macro {f1_c:.3f}")
    print(f"  AGREGADO:         accuracy {acc_a:.3f} | F1 macro {f1_a:.3f}")
    print(f"  (comparar con CNN EfficientNet-B0: 0.858 agregado)")
    print(f"  modelo: {ckpt.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
