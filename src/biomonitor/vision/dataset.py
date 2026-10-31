"""Dataset de PyTorch para el clasificador de imagen (Módulo 1).

Carga las fotos (JPEG), redimensiona a `image.size` y aplica augmentation en train.
Normaliza con estadísticas de ImageNet (los backbones son preentrenados en ImageNet).
"""

from __future__ import annotations

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from biomonitor.config import CONFIG, PROJECT_ROOT
from biomonitor.audio.dataset import build_label_map  # reutiliza el mapeo especie→índice

SIZE = CONFIG["image"]["size"]
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _train_tf(size: int = SIZE):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_tf(size: int = SIZE):
    return transforms.Compose([
        transforms.Resize(int(size * 1.15)),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BirdImageDataset(Dataset):
    def __init__(self, manifest, label_map: dict[str, int], train: bool = False, size: int = SIZE):
        self.df = manifest.reset_index(drop=True)
        self.label_map = label_map
        self.size = size
        self.tf = _train_tf(size) if train else _eval_tf(size)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = Image.open(PROJECT_ROOT / row["file"]).convert("RGB")
        except Exception:  # noqa: BLE001  imagen corrupta → placeholder gris
            img = Image.fromarray(np.zeros((self.size, self.size, 3), dtype=np.uint8))
        return self.tf(img), self.label_map[row["scientific_name"]]


def image_datasets_from_manifest(manifest, species: list[str], size: int = SIZE):
    """(train_ds, val_ds, test_ds, label_map) para el clasificador de imagen."""
    label_map = build_label_map(species)
    parts = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    return (
        BirdImageDataset(parts["train"], label_map, train=True, size=size),
        BirdImageDataset(parts["val"], label_map, train=False, size=size),
        BirdImageDataset(parts["test"], label_map, train=False, size=size),
        label_map,
    )
