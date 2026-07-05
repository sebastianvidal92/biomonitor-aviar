"""Dataset para el AST — entrega los input_values [1024, 128] que espera el modelo.

Carga el audio cacheado (32 kHz FLAC), toma una ventana de ~10 s (aleatoria en train,
central en eval), la resamplea a 16 kHz y la pasa por el ASTFeatureExtractor (fbank
+ normalización AudioSet). Reutiliza el split y el label_map del pipeline de audio.
"""

from __future__ import annotations

import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset
from transformers import ASTFeatureExtractor

from biomonitor.config import CONFIG, PROJECT_ROOT
from biomonitor.audio.ast_model import AST_CHECKPOINT
from biomonitor.audio.dataset import SR as SRC_SR
from biomonitor.audio.dataset import build_label_map

AST_SR = 16000
CLIP_SECONDS = CONFIG["train_ast"]["clip_seconds"]
SRC_WINDOW = SRC_SR * CLIP_SECONDS  # muestras a leer del cache 32 kHz

_FE: ASTFeatureExtractor | None = None


def feature_extractor() -> ASTFeatureExtractor:
    """ASTFeatureExtractor cacheado (fbank 128 mel, 1024 frames, norm AudioSet)."""
    global _FE
    if _FE is None:
        _FE = ASTFeatureExtractor.from_pretrained(AST_CHECKPOINT)
    return _FE


def load_window_16k(path: str, random_crop: bool, n_src: int = SRC_WINDOW) -> np.ndarray:
    """Ventana de audio (del cache 32 kHz) resampleada a 16 kHz mono para el AST."""
    total = sf.info(path).frames
    if total <= 0:
        return np.zeros(AST_SR * CLIP_SECONDS, dtype=np.float32)
    if total <= n_src:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
    else:
        start = random.randint(0, total - n_src) if random_crop else (total - n_src) // 2
        y, _ = sf.read(path, start=start, frames=n_src, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return torchaudio.functional.resample(torch.from_numpy(y), SRC_SR, AST_SR).numpy()


class ASTDataset(Dataset):
    def __init__(self, manifest, label_map: dict[str, int], train: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.label_map = label_map
        self.train = train
        self.fe = feature_extractor()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        y16 = load_window_16k(str(PROJECT_ROOT / row["cached_file"]), self.train)
        feats = self.fe(y16, sampling_rate=AST_SR, return_tensors="pt")
        x = feats["input_values"][0]  # [1024, 128]
        return x, self.label_map[row["scientific_name"]]


def ast_datasets_from_manifest(manifest, species: list[str]):
    """(train_ds, val_ds, test_ds, label_map) para el AST — mismo split que el CNN."""
    label_map = build_label_map(species)
    parts = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    return (
        ASTDataset(parts["train"], label_map, train=True),
        ASTDataset(parts["val"], label_map, train=False),
        ASTDataset(parts["test"], label_map, train=False),
        label_map,
    )
