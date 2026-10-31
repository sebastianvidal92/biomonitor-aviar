"""Utilidades de evaluación compartidas por los scripts del informe final (27-31).

- Ventanas de audio por grabación (32 kHz para la CNN, 16 kHz para el AST).
- Métricas de clasificación: accuracy, F1 macro, AUC-ROC macro (uno contra el resto),
  top-5, matriz de confusión.
- Grupo de conservación de cada especie (Amenazada / Endémica / Común), igual que en el
  informe (Tabla 4.1): amenazada = CR/EN/VU; endémica = endémica no amenazada (incluye
  el choroy, LC); común = el resto.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score

from biomonitor.config import data_path

THREATENED = {"CR", "EN", "VU"}
GROUPS = ("Amenazada", "Endémica", "Común")


def species_groups() -> dict[str, str]:
    """Especie → grupo del informe (Amenazada / Endémica / Común)."""
    sp = pd.read_csv(data_path("external", "species_final.csv"))
    out = {}
    for r in sp.itertuples(index=False):
        cat = str(r.category_primary) if not pd.isna(r.category_primary) else ""
        if cat in THREATENED:
            out[r.scientific_name] = "Amenazada"
        elif bool(r.endemic):
            out[r.scientific_name] = "Endémica"
        else:
            out[r.scientific_name] = "Común"
    return out


# ------------------------------- audio -------------------------------------
def recording_windows(path: str, n: int, max_windows: int) -> np.ndarray:
    """Hasta `max_windows` ventanas de `n` muestras que cubren la grabación → [W, n]."""
    total = sf.info(path).frames
    if total <= 0:
        return np.zeros((1, n), dtype=np.float32)
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


def recording_windows_16k(path: str, n_src: int, max_windows: int, src_sr: int,
                          dst_sr: int = 16000) -> list[np.ndarray]:
    """Ventanas (del cache a `src_sr`) resampleadas a 16 kHz para el AST."""
    wins = recording_windows(path, n_src, max_windows)
    return [torchaudio.functional.resample(torch.from_numpy(w), src_sr, dst_sr).numpy()
            for w in wins]


class LogMel:
    """Espectrograma log-mel normalizado por ventana, igual que en entrenamiento."""

    def __init__(self, sr: int, n_fft: int, hop: int, n_mels: int):
        self.melspec = torchaudio.transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop,
                                                            n_mels=n_mels)
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:  # [W, n] -> [W, 1, mels, T]
        mel = self.to_db(self.melspec(wav))
        mu = mel.mean(dim=(1, 2), keepdim=True)
        sd = mel.std(dim=(1, 2), keepdim=True) + 1e-6
        return ((mel - mu) / sd).unsqueeze(1)


# ------------------------------ métricas -----------------------------------
def auc_macro(y_true: np.ndarray, probs: np.ndarray) -> tuple[float, int]:
    """AUC-ROC uno-contra-el-resto promediado sobre las clases presentes en y_true."""
    aucs = []
    for k in range(probs.shape[1]):
        pos = (np.asarray(y_true) == k)
        if pos.any() and (~pos).any():
            aucs.append(roc_auc_score(pos.astype(int), probs[:, k]))
    return float(np.mean(aucs)), len(aucs)


def topk_accuracy(y_true: np.ndarray, probs: np.ndarray, k: int = 5) -> float:
    topk = np.argsort(-probs, axis=1)[:, :k]
    return float(np.mean([y in row for y, row in zip(y_true, topk)]))


def summarize(y_true: np.ndarray, probs: np.ndarray) -> dict:
    y_pred = probs.argmax(1)
    auc, n_auc = auc_macro(y_true, probs)
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_macro": auc,
        "top5": topk_accuracy(y_true, probs, 5),
        "classes_auc": n_auc,
    }


def confusion(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))
    return pd.DataFrame(cm, index=classes, columns=classes)


def per_class_table(y_true, y_pred, classes: list[str]) -> pd.DataFrame:
    """precision / recall / F1 / soporte por clase (mismo formato que los CSV previos)."""
    from sklearn.metrics import classification_report

    rep = classification_report([classes[i] for i in y_true], [classes[i] for i in y_pred],
                                labels=classes, output_dict=True, zero_division=0)
    return pd.DataFrame(rep).T.loc[classes][["precision", "recall", "f1-score", "support"]]


def by_group(per_class: pd.DataFrame, groups: dict[str, str]) -> pd.DataFrame:
    """F1 medio, mínimo y accuracy (recall ponderado) por grupo de conservación."""
    d = per_class.copy()
    d["grupo"] = [groups.get(s, "Común") for s in d.index]
    rows = []
    for g in GROUPS:
        sub = d[d["grupo"] == g]
        if not len(sub):
            continue
        rows.append({
            "grupo": g, "especies": int(len(sub)),
            "f1_medio": float(sub["f1-score"].mean()),
            "f1_min": float(sub["f1-score"].min()),
            "accuracy": float((sub["recall"] * sub["support"]).sum() / sub["support"].sum()),
            "n_test": int(sub["support"].sum()),
        })
    return pd.DataFrame(rows)


def count_params(model) -> int:
    return int(sum(p.numel() for p in model.parameters()))
