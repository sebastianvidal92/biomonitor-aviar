"""Parches de imagen satelital (cobertura de suelo) para el SDM multimodal — Módulo 2.

Extrae de GEE un parche NxN de ESA WorldCover (cobertura de suelo, 10 m) centrado en
cada punto. Es una imagen estática (barata de muestrear) que captura el mosaico de
hábitat alrededor del avistamiento: dato espacial NO estructurado que una CNN puede
procesar y MaxEnt no.
"""

from __future__ import annotations

import time

import ee
import numpy as np

# Clases de ESA WorldCover v200 → índices 0..10 (para one-hot en la CNN).
WORLDCOVER_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
CLASS_TO_IDX = {c: i for i, c in enumerate(WORLDCOVER_CLASSES)}
N_CLASSES = len(WORLDCOVER_CLASSES)


def landcover_image() -> ee.Image:
    return ee.Image("ESA/WorldCover/v200/2021").select("Map")


def sample_patches(points: list[dict], radius: int = 16, scale: int = 10,
                   batch: int = 200) -> dict[int, np.ndarray]:
    """Extrae un parche (2*radius+1)² por punto. Devuelve {idx: array int16}.

    `points`: lista de dicts con 'lat', 'lon', 'idx'.
    """
    arr = landcover_image().neighborhoodToArray(ee.Kernel.square(radius=radius, units="pixels"))
    out: dict[int, np.ndarray] = {}
    for i in range(0, len(points), batch):
        chunk = points[i:i + batch]
        feats = [ee.Feature(ee.Geometry.Point([p["lon"], p["lat"]]), {"idx": int(p["idx"])})
                 for p in chunk]
        sampled = arr.sampleRegions(collection=ee.FeatureCollection(feats), scale=scale,
                                    geometries=False)
        for attempt in range(4):  # reintentos ante timeouts transitorios
            try:
                info = sampled.getInfo()
                break
            except Exception:  # noqa: BLE001
                if attempt == 3:
                    raise
                time.sleep(5 * (attempt + 1))
        for f in info["features"]:
            out[int(f["properties"]["idx"])] = np.array(f["properties"]["Map"], dtype=np.int16)
    return out


def one_hot(patches: np.ndarray) -> np.ndarray:
    """(N, H, W) de clases WorldCover → (N, n_clases, H, W) one-hot float32."""
    idx = np.vectorize(lambda c: CLASS_TO_IDX.get(int(c), 0))(patches)
    oh = np.zeros((patches.shape[0], N_CLASSES, *patches.shape[1:]), dtype=np.float32)
    n, h, w = patches.shape
    ii, hh, ww = np.meshgrid(np.arange(n), np.arange(h), np.arange(w), indexing="ij")
    oh[ii, idx, hh, ww] = 1.0
    return oh


def one_hot_single(patch: np.ndarray) -> np.ndarray:
    """(H, W) de clases → (n_clases, H, W) one-hot float32 (para el Dataset on-the-fly)."""
    idx = np.vectorize(lambda c: CLASS_TO_IDX.get(int(c), 0))(patch)
    oh = np.zeros((N_CLASSES, *patch.shape), dtype=np.float32)
    hh, ww = np.meshgrid(np.arange(patch.shape[0]), np.arange(patch.shape[1]), indexing="ij")
    oh[idx, hh, ww] = 1.0
    return oh
