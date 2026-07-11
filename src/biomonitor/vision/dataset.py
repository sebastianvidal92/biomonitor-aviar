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


def _train_tf():
    return transforms.Compose([
        transforms.RandomResizedCrop(SIZE, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def _eval_tf():
    return transforms.Compose([
        transforms.Resize(int(SIZE * 1.15)),
        transforms.CenterCrop(SIZE),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class BirdImageDataset(Dataset):
    def __init__(self, manifest, label_map: dict[str, int], train: bool = False):
        self.df = manifest.reset_index(drop=True)
        self.label_map = label_map
        self.tf = _train_tf() if train else _eval_tf()

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        try:
            img = Image.open(PROJECT_ROOT / row["file"]).convert("RGB")
        except Exception:  # noqa: BLE001  imagen corrupta → placeholder gris
            img = Image.fromarray(np.zeros((SIZE, SIZE, 3), dtype=np.uint8))
        return self.tf(img), self.label_map[row["scientific_name"]]


def image_datasets_from_manifest(manifest, species: list[str]):
    """(train_ds, val_ds, test_ds, label_map) para el clasificador de imagen."""
    label_map = build_label_map(species)
    parts = {s: manifest[manifest["split"] == s] for s in ("train", "val", "test")}
    return (
        BirdImageDataset(parts["train"], label_map, train=True),
        BirdImageDataset(parts["val"], label_map, train=False),
        BirdImageDataset(parts["test"], label_map, train=False),
        label_map,
    )
