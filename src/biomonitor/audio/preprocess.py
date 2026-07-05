"""Preprocesamiento de audio: carga (mp3/wav), mono, resampleo y cacheo a .flac.

Estrategia elegida: cachear el audio resampleado a 32 kHz mono comprimido (.flac,
lossless), y calcular ventana + espectrograma mel on-the-fly en el Dataset de
entrenamiento. Usa torchaudio (resampleo, soporta MPS) y soundfile (I/O), sin librosa.

Nota: se usa FLAC en vez de OGG/Vorbis porque el codificador Vorbis de libsndfile
crashea (segfault) al escribir algunos WAV grandes de alta tasa; FLAC es robusto.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

from biomonitor.config import CONFIG

TARGET_SR = CONFIG["audio"]["sample_rate"]


def load_mono(path: str | Path) -> tuple[np.ndarray, int]:
    """Carga un archivo de audio (mp3/wav/ogg) como mono float32."""
    y, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    return np.ascontiguousarray(y), sr


def resample_and_cache(src: str | Path, dst: str | Path, target_sr: int = TARGET_SR) -> float:
    """Resamplea `src` a `target_sr` mono y lo escribe como .flac en `dst`.

    Devuelve la duración en segundos. Lanza si el audio está vacío/corrupto.
    """
    y, sr = load_mono(src)
    if y.size == 0:
        raise ValueError("audio vacío")
    t = torch.from_numpy(y)
    if sr != target_sr:
        t = torchaudio.functional.resample(t, sr, target_sr)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    out = np.ascontiguousarray(t.numpy(), dtype=np.float32)
    if out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError("audio no finito tras resamplear")
    np.clip(out, -1.0, 1.0, out=out)
    sf.write(str(dst), out, target_sr, format="FLAC", subtype="PCM_16")
    return len(out) / target_sr
