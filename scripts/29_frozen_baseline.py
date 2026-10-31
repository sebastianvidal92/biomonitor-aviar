"""Baseline de bajo costo: clasificador lineal sobre EMBEDDINGS CONGELADOS (Módulo 1).

Protocolo de Ghani et al. (2023): se extraen las representaciones de un modelo
preentrenado SIN ajustarlo y se entrena encima un clasificador lineal (regresión logística
multinomial, C elegido en validación). Sirve como referencia de "cómo se abordaría el
problema sin entrenar una red": si el fine-tuning completo (scripts 07/09/12) no supera
claramente a esta referencia, la red no está aportando.

Modelos congelados:
  - audio:  EfficientNet-B0 preentrenado en ImageNet (1 canal, 1.280-d) sobre log-mel;
            AST preentrenado en AudioSet (pooler 768-d) sobre fbank 16 kHz.
  - imagen: EfficientNet-B3 preentrenado en ImageNet (1.536-d), recorte central 384 px.

    uv run python scripts/29_frozen_baseline.py
    uv run python scripts/29_frozen_baseline.py --skip-ast        # el AST es el más lento
    uv run python scripts/29_frozen_baseline.py --ast-windows 4   # ventanas por grabación (AST)

Salida: reports/frozen_baseline.csv. Embeddings cacheados en data/interim/frozen_*.npz
(reanudable). ~5 min (CNN + imagen) + ~15 min (AST) en MPS.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR, build_label_map  # noqa: E402
from biomonitor.evaluation import LogMel, auc_macro, recording_windows, recording_windows_16k  # noqa: E402
from biomonitor.vision.dataset import _eval_tf  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
C_GRID = [0.01, 0.1, 1.0, 10.0]


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ast", action="store_true")
    p.add_argument("--cnn-windows", type=int, default=8)
    p.add_argument("--ast-windows", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="smoke test: N grabaciones/fotos, sin guardar")
    return p.parse_args()


# ------------------------------ extracción ----------------------------------
@torch.no_grad()
def embed_audio_cnn(manifest: pd.DataFrame, max_windows: int, cache: Path, save: bool) -> tuple[np.ndarray, np.ndarray]:
    """[N_ventanas, 1280] + índice de grabación por ventana."""
    if cache.exists() and save:
        z = np.load(cache)
        return z["emb"], z["rec"]
    model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0, in_chans=1).eval().to(DEVICE)
    logmel = LogMel(SR, N_FFT, HOP, N_MELS)
    embs, recs = [], []
    for i, r in enumerate(manifest.itertuples(index=False)):
        wins = recording_windows(str(PROJECT_ROOT / r.cached_file), CLIP_SAMPLES, max_windows)
        e = model(logmel(torch.from_numpy(wins)).to(DEVICE)).cpu().numpy()
        embs.append(e); recs.extend([i] * len(e))
        if (i + 1) % 250 == 0:
            print(f"    {i + 1}/{len(manifest)}")
    emb, rec = np.concatenate(embs).astype(np.float32), np.array(recs)
    if save:
        np.savez_compressed(cache, emb=emb, rec=rec)
    return emb, rec


@torch.no_grad()
def embed_audio_ast(manifest: pd.DataFrame, max_windows: int, cache: Path, save: bool) -> tuple[np.ndarray, np.ndarray]:
    """[N_ventanas, 768] del AST de AudioSet congelado (salida pooler)."""
    if cache.exists() and save:
        z = np.load(cache)
        return z["emb"], z["rec"]
    from transformers import ASTModel
    from biomonitor.audio.ast_dataset import AST_SR, SRC_WINDOW, feature_extractor
    from biomonitor.audio.ast_model import AST_CHECKPOINT

    model = ASTModel.from_pretrained(AST_CHECKPOINT).eval().to(DEVICE)
    fe = feature_extractor()
    embs, recs = [], []
    for i, r in enumerate(manifest.itertuples(index=False)):
        wins = recording_windows_16k(str(PROJECT_ROOT / r.cached_file), SRC_WINDOW, max_windows, SR, AST_SR)
        feats = fe(wins, sampling_rate=AST_SR, return_tensors="pt")["input_values"].to(DEVICE)
        e = model(input_values=feats).pooler_output.cpu().numpy()
        embs.append(e); recs.extend([i] * len(e))
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(manifest)}")
    emb, rec = np.concatenate(embs).astype(np.float32), np.array(recs)
    if save:
        np.savez_compressed(cache, emb=emb, rec=rec)
    return emb, rec


@torch.no_grad()
def embed_images(manifest: pd.DataFrame, cache: Path, save: bool) -> tuple[np.ndarray, np.ndarray]:
    """[N_fotos, 1536] (recorte central) + índice de foto (las ilegibles se omiten)."""
    if cache.exists() and save:
        z = np.load(cache)
        return z["emb"], z["rec"]
    model = timm.create_model("efficientnet_b3", pretrained=True, num_classes=0).eval().to(DEVICE)
    tf = _eval_tf()
    embs, recs = [], []
    for i in range(0, len(manifest), 32):
        xs, ok = [], []
        for j, r in enumerate(manifest.iloc[i:i + 32].itertuples(index=False)):
            try:
                xs.append(tf(Image.open(PROJECT_ROOT / r.file).convert("RGB"))); ok.append(i + j)
            except Exception:  # noqa: BLE001
                pass
        if xs:
            embs.append(model(torch.stack(xs).to(DEVICE)).cpu().numpy()); recs.extend(ok)
        if (i // 32 + 1) % 25 == 0:
            print(f"    {min(i + 32, len(manifest))}/{len(manifest)}")
    emb, rec = np.concatenate(embs).astype(np.float32), np.array(recs)
    if save:
        np.savez_compressed(cache, emb=emb, rec=rec)
    return emb, rec


# ------------------------------ clasificador --------------------------------
def linear_probe(emb: np.ndarray, rec: np.ndarray, manifest: pd.DataFrame, label_map: dict,
                 aggregate: bool) -> dict:
    """Regresión logística sobre embeddings; C elegido por F1 macro en validación.

    Si `aggregate`, en val/test se promedian las probabilidades de las ventanas de cada
    grabación (mismo protocolo "agregado" que los modelos ajustados)."""
    split = manifest["split"].values[rec]
    y_all = manifest["scientific_name"].map(label_map).values[rec]
    scaler = StandardScaler().fit(emb[split == "train"])
    X = scaler.transform(emb)

    def predict(clf, mask):
        probs = clf.predict_proba(X[mask])
        y = y_all[mask]
        if aggregate:
            ids = rec[mask]
            uniq = np.unique(ids)
            probs = np.stack([probs[ids == u].mean(0) for u in uniq])
            y = np.array([y_all[mask][ids == u][0] for u in uniq])
        full = np.zeros((len(probs), len(label_map)), dtype=np.float32)
        full[:, clf.classes_] = probs
        return y, full

    best, best_f1 = None, -1
    for c in C_GRID:
        clf = LogisticRegression(C=c, max_iter=3000, class_weight="balanced")
        clf.fit(X[split == "train"], y_all[split == "train"])
        yv, pv = predict(clf, split == "val")
        f1 = f1_score(yv, pv.argmax(1), average="macro", zero_division=0)
        if f1 > best_f1:
            best, best_f1, best_c = clf, f1, c
    yt, pt = predict(best, split == "test")
    auc, _ = auc_macro(yt, pt)
    return {"C": best_c, "val_f1": float(best_f1), "n_test": int(len(yt)),
            "accuracy": float(accuracy_score(yt, pt.argmax(1))),
            "f1_macro": float(f1_score(yt, pt.argmax(1), average="macro", zero_division=0)),
            "auc_macro": float(auc)}


def main() -> None:
    args = get_args()
    save = args.limit is None
    print(f"Dispositivo: {DEVICE}" + ("" if save else f" | SMOKE TEST ({args.limit} muestras, sin guardar)") + "\n")
    cache_dir = data_path("interim", "frozen_cache.txt").parent

    a_man = pd.read_csv(data_path("processed", "audio32_manifest.csv")).reset_index(drop=True)
    a_species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    a_lm = build_label_map(a_species)
    i_man = pd.read_csv(data_path("interim", "image_manifest_split.csv")).reset_index(drop=True)
    i_lm = build_label_map(sorted(i_man["scientific_name"].unique()))
    if not save:  # smoke test: 3 especies × hasta N/9 muestras por partición (≥2 clases por split)
        k = max(2, args.limit // 9)
        a_sub = sorted(a_man["scientific_name"].unique())[:3]
        a_man = a_man[a_man["scientific_name"].isin(a_sub)].groupby(["split", "scientific_name"]).head(k).reset_index(drop=True)
        i_sub = sorted(i_man["scientific_name"].unique())[:3]
        i_man = i_man[i_man["scientific_name"].isin(i_sub)].groupby(["split", "scientific_name"]).head(k).reset_index(drop=True)

    rows = []
    t0 = time.time()
    print(f"[1/3] Audio — EfficientNet-B0 (ImageNet) congelado, {len(a_man)} grabaciones")
    emb, rec = embed_audio_cnn(a_man, args.cnn_windows, cache_dir / "frozen_audio_effb0.npz", save)
    r = linear_probe(emb, rec, a_man, a_lm, aggregate=True)
    rows.append({"modalidad": "audio", "embedding": "EfficientNet-B0 (ImageNet, congelado)", "dim": emb.shape[1], **r})
    print(f"    acc {r['accuracy']:.3f} | F1 {r['f1_macro']:.3f} | AUC {r['auc_macro']:.3f} (C={r['C']}) "
          f"[{time.time() - t0:.0f} s]")

    if not args.skip_ast:
        t0 = time.time()
        print(f"[2/3] Audio — AST (AudioSet) congelado, {len(a_man)} grabaciones (lento)")
        emb, rec = embed_audio_ast(a_man, args.ast_windows, cache_dir / "frozen_audio_ast.npz", save)
        r = linear_probe(emb, rec, a_man, a_lm, aggregate=True)
        rows.append({"modalidad": "audio", "embedding": "AST (AudioSet, congelado)", "dim": emb.shape[1], **r})
        print(f"    acc {r['accuracy']:.3f} | F1 {r['f1_macro']:.3f} | AUC {r['auc_macro']:.3f} (C={r['C']}) "
              f"[{time.time() - t0:.0f} s]")
    else:
        print("[2/3] AST omitido (--skip-ast)")

    t0 = time.time()
    print(f"[3/3] Imagen — EfficientNet-B3 (ImageNet) congelado, {len(i_man)} fotos")
    emb, rec = embed_images(i_man, cache_dir / "frozen_image_effb3.npz", save)
    r = linear_probe(emb, rec, i_man, i_lm, aggregate=False)
    rows.append({"modalidad": "imagen", "embedding": "EfficientNet-B3 (ImageNet, congelado)", "dim": emb.shape[1], **r})
    print(f"    acc {r['accuracy']:.3f} | F1 {r['f1_macro']:.3f} | AUC {r['auc_macro']:.3f} (C={r['C']}) "
          f"[{time.time() - t0:.0f} s]")

    res = pd.DataFrame(rows)
    if save:
        out = PROJECT_ROOT / "reports" / "frozen_baseline.csv"
        res.to_csv(out, index=False)
        print(f"\n  → {out.relative_to(PROJECT_ROOT)}\n")
    print("\nReferencia (modelos ajustados, protocolo agregado/TTA): audio CNN 0,858 · AST 0,868 · imagen 0,848")


if __name__ == "__main__":
    main()
