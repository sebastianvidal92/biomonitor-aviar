"""Dataset de PyTorch para el clasificador de cantos (Módulo 1).

Lee el audio cacheado (.flac 32 kHz mono, manifiesto data/processed/audio32_manifest.csv),
toma una ventana de 5 s (aleatoria en train, central en val/test), calcula el
espectrograma log-mel on-the-fly y aplica augmentation en train.

    from biomonitor.audio.dataset import datasets_from_manifest, build_label_map
"""

from __future__ import annotations

import random

import numpy as np
import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset

from biomonitor.config import CONFIG, PROJECT_ROOT

SR = CONFIG["audio"]["sample_rate"]
CLIP_SAMPLES = int(CONFIG["audio"]["clip_seconds"] * SR)
N_MELS = CONFIG["audio"]["n_mels"]
N_FFT = 1024
HOP = SR // 100  # 100 frames por segundo


def build_label_map(species: list[str]) -> dict[str, int]:
    """Mapa especie → índice de clase, ordenado para reproducibilidad."""
    return {sp: i for i, sp in enumerate(sorted(set(species)))}


def _load_window(path: str, n_samples: int, random_crop: bool) -> np.ndarray:
    """Carga una ventana de n_samples del audio; pad por repetición si es corto."""
    total = sf.info(path).frames
    if total <= 0:  # clip vacío/corrupto: devuelve silencio (defensivo)
        return np.zeros(n_samples, dtype=np.float32)
    if total <= n_samples:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
        reps = int(np.ceil(n_samples / max(len(y), 1)))
        y = np.tile(y, reps)[:n_samples]
    else:
        start = random.randint(0, total - n_samples) if random_crop else (total - n_samples) // 2
        y, _ = sf.read(path, start=start, frames=n_samples, dtype="float32", always_2d=False)
        if y.ndim > 1:
            y = y.mean(axis=1)
    return y


class BirdSpectrogramDataset(Dataset):
    def __init__(self, manifest, label_map: dict[str, int], train: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.label_map = label_map
        self.train = train
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=24)
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=48)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        path = str(PROJECT_ROOT / row["cached_file"])
        y = _load_window(path, CLIP_SAMPLES, random_crop=self.train)
        wav = torch.from_numpy(y)

        if self.train:  # augmentation a nivel de audio
            wav = wav * (10 ** (random.uniform(-6, 6) / 20))      # ganancia ±6 dB
            if random.random() < 0.5:
                wav = wav + 0.005 * torch.randn_like(wav)          # ruido gaussiano

        mel = self.to_db(self.melspec(wav))                        # [n_mels, frames]
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)              # normalización

        if self.train:                                             # SpecAugment (2 máscaras)
            mel = mel.unsqueeze(0)
            for _ in range(2):
                mel = self.time_mask(self.freq_mask(mel))
            mel = mel.squeeze(0)

        label = self.label_map[row["scientific_name"]]
        return mel.unsqueeze(0), label                             # [1, n_mels, frames], int


def datasets_from_manifest(manifest, species: list[str]):
    """Devuelve (train_ds, val_ds, test_ds, label_map) a partir del manifiesto cacheado."""
    label_map = build_label_map(species)
    parts = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    return (
        BirdSpectrogramDataset(parts["train"], label_map, train=True),
        BirdSpectrogramDataset(parts["val"], label_map, train=False),
        BirdSpectrogramDataset(parts["test"], label_map, train=False),
        label_map,
    )
