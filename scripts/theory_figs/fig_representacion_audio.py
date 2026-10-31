"""Figura del marco teórico: del audio al espectrograma log-mel y aumentaciones.

Seis paneles (2x3): (a) forma de onda, (b) espectrograma lineal |STFT|^2 en dB,
(c) banco de filtros mel, (d) log-mel de 128 bandas normalizado (entrada de la CNN),
(e) SpecAugment (2 máscaras de frecuencia + 2 de tiempo) y (f) mixup con otra especie.

    uv run python scripts/theory_figs/fig_representacion_audio.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
tf = importlib.import_module("32_thesis_figures")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import soundfile as sf  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # noqa: E402

from biomonitor.audio.dataset import HOP, N_FFT, N_MELS, SR  # noqa: E402

mpl.rcParams["mathtext.fontset"] = "stix"
CLIP = 5 * SR                      # ventana de 5 s
SEED = 4920                        # semilla fija para SpecAugment (máscaras anchas, separadas y fuera del canto)
XC_A, XC_B = 954086, 1080172       # T. falcklandii (A) y Z. capensis (B)
CMAP = "magma"
FS_NOTA = 7                        # anotaciones dentro de los paneles
FS_TIT = 8                         # títulos de panel (van en set_title: con titlelocation=left el título es _left_title)
dec = FuncFormatter(lambda v, _: f"{v:g}".replace(".", ","))


def window(xc_id: int) -> np.ndarray:
    """Primeros 5 s de la grabación cacheada (FLAC 32 kHz mono)."""
    man = pd.read_csv(ROOT / "data" / "processed" / "audio32_manifest.csv")
    row = man[man["xc_id"] == xc_id].iloc[0]
    y, sr = sf.read(ROOT / row["cached_file"], frames=CLIP, dtype="float32", always_2d=False)
    assert sr == SR and len(y) == CLIP and row["split"] == "train", (sr, len(y), row["split"])
    return y


def logmel(y: np.ndarray) -> np.ndarray:
    """Log-mel normalizado por ventana, igual que en biomonitor.audio.dataset."""
    melspec = torchaudio.transforms.MelSpectrogram(sample_rate=SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)
    m = to_db(melspec(torch.from_numpy(y)))
    m = (m - m.mean()) / (m.std() + 1e-6)
    return m.numpy()


def specaugment(m: np.ndarray, seed: int) -> tuple[np.ndarray, list[tuple[str, int, int]]]:
    """Dos máscaras de frecuencia (<= 24 bandas) y dos de tiempo (<= 48 cuadros), como torchaudio.

    Relleno con el valor medio del espectrograma (0 tras la normalización). Devuelve las
    máscaras (eje, inicio, ancho) para anotarlas.
    """
    rng = np.random.default_rng(seed)
    out = m.copy()
    masks = []
    for axis, param in (("f", 24), ("t", 48), ("f", 24), ("t", 48)):
        n = out.shape[0] if axis == "f" else out.shape[1]
        w = int(rng.uniform(0, param)); s = int(rng.uniform(0, n - w))
        if axis == "f":
            out[s:s + w, :] = m.mean()
        else:
            out[:, s:s + w] = m.mean()
        masks.append((axis, s, w))
    return out, masks


def side_colorbar(fig, ax, mappable, label: str):
    """Barra de color adosada al borde derecho de `ax` (sin robarle espacio al panel)."""
    cax = inset_axes(ax, width="4.5%", height="100%", loc="center left",
                     bbox_to_anchor=(1.03, 0.0, 1.0, 1.0), bbox_transform=ax.transAxes, borderpad=0)
    cb = fig.colorbar(mappable, cax=cax)
    cb.outline.set_visible(False)
    cb.set_label(label, fontsize=7.5, labelpad=2)
    cb.ax.tick_params(labelsize=7, length=2, pad=1.5)
    cb.ax.yaxis.set_major_formatter(dec)
    return cb


def main() -> None:
    ya, yb = window(XC_A), window(XC_B)
    t = np.arange(CLIP) / SR

    # (b) espectrograma de potencia lineal en dB (513 bins x 501 cuadros), relativo al máximo
    spec = torchaudio.transforms.Spectrogram(n_fft=N_FFT, hop_length=HOP, power=2.0)
    S = torchaudio.transforms.AmplitudeToDB("power", top_db=80)(spec(torch.from_numpy(ya))).numpy()
    S -= S.max()
    # (c) banco de filtros mel
    fb = torchaudio.functional.melscale_fbanks(n_freqs=N_FFT // 2 + 1, f_min=0.0, f_max=SR / 2,
                                               n_mels=N_MELS, sample_rate=SR).numpy()  # [513, 128]
    freqs = np.linspace(0, SR / 2, N_FFT // 2 + 1) / 1000
    # (d)-(f) log-mel, SpecAugment y mixup
    ma, mb = logmel(ya), logmel(yb)
    aug, masks = specaugment(ma, SEED)
    lam = 0.6
    mix = lam * ma + (1 - lam) * mb
    vmin, vmax = np.percentile(np.concatenate([ma.ravel(), mix.ravel()]), [0.5, 99.8])
    n_frames = ma.shape[1]
    ext = (0, n_frames / 100, 0, N_MELS)

    # Grilla 2x3. El espacio entre columnas (wspace) aloja la barra de color de (b) más el eje y de (c);
    # los márgenes dejan sitio al eje y de (a)/(d) y a la barra compartida de la fila inferior.
    fig = plt.figure(figsize=(tf.TEXTW, 5.2))
    gs = fig.add_gridspec(2, 3, left=0.085, right=0.905, top=0.96, bottom=0.075, hspace=0.42, wspace=0.76)
    axs = [[fig.add_subplot(gs[r, c]) for c in range(3)] for r in range(2)]
    (aa, ab, ac), (ad, ae, af) = axs

    # (a) forma de onda: envolvente min/max por bloques de 32 muestras (vector liviano)
    blk = 32
    yy = ya[: len(ya) // blk * blk].reshape(-1, blk)
    tb = t[: len(ya) // blk * blk].reshape(-1, blk).mean(1)
    aa.fill_between(tb, yy.min(1), yy.max(1), color=tf.C1, lw=0.3, edgecolor=tf.C1)
    aa.set_xlim(0, 5); amp = float(np.abs(ya).max()) * 1.15
    aa.set_ylim(-amp, amp); aa.yaxis.set_major_formatter(dec)
    aa.set_xlabel("Tiempo (s)"); aa.set_ylabel("Amplitud", labelpad=2)
    aa.set_title("(a) Forma de onda", fontsize=FS_TIT)

    # (b) espectrograma lineal, con su barra de color adosada
    imb = ab.imshow(S, origin="lower", aspect="auto", cmap=CMAP, extent=(0, n_frames / 100, 0, SR / 2000),
                    vmin=-80, vmax=0, interpolation="nearest")
    ab.set_xlabel("Tiempo (s)"); ab.set_ylabel("Frecuencia (kHz)", labelpad=1)
    ab.set_yticks([0, 4, 8, 12, 16]); ab.grid(False)
    ab.tick_params(axis="y", pad=1.5)
    ab.set_title("(b) Espectrograma (dB)", fontsize=FS_TIT)
    side_colorbar(fig, ab, imb, "Potencia relativa (dB)").set_ticks([0, -20, -40, -60, -80])

    # (c) banco de filtros mel: 10 de los 128 filtros triangulares
    idx = np.linspace(6, N_MELS - 1, 10).round().astype(int)
    for i in idx:
        ac.fill_between(freqs, 0, fb[:, i], color=tf.C1, alpha=0.18, lw=0)
        ac.plot(freqs, fb[:, i], color=tf.C1, lw=0.7)
    ac.set_xlim(0, 16); ac.set_ylim(0, 1.45)
    ac.set_xticks([0, 4, 8, 12, 16]); ac.set_yticks([0, 0.5, 1]); ac.yaxis.set_major_formatter(dec)
    ac.set_xlabel("Frecuencia (kHz)"); ac.set_ylabel("Ganancia del filtro", labelpad=1)
    ac.tick_params(axis="y", pad=1.5, length=2)
    ac.text(0.97, 0.96, "$\\mathrm{mel}(f) =$\n$2595\\,\\log_{10}(1 + f/700)$",
            transform=ac.transAxes, ha="right", va="top", fontsize=FS_NOTA, linespacing=1.45)
    ac.set_title("(c) Filtros mel (10 de 128)", fontsize=FS_TIT)

    # (d)-(f) log-mel y aumentaciones
    for ax, M, ttl in ((ad, ma, "(d) Log-mel (128 bandas)"),
                       (ae, aug, "(e) SpecAugment"),
                       (af, mix, "(f) mixup, $\\lambda$ = 0,6")):
        im = ax.imshow(M, origin="lower", aspect="auto", cmap=CMAP, extent=ext, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xlabel("Tiempo (s)"); ax.set_yticks([0, 32, 64, 96, 128]); ax.grid(False)
        ax.set_title(ttl, fontsize=FS_TIT)
    ad.set_ylabel("Banda mel", labelpad=2)
    # anotaciones de las máscaras de SpecAugment (en la más ancha de cada eje)
    widest = {ax_: max((m for m in masks if m[0] == ax_), key=lambda m: m[2]) for ax_ in ("f", "t")}
    _, s, w = widest["f"]
    ae.text(2.5, s + w / 2, "máscara de frecuencia", ha="center", va="center", fontsize=FS_NOTA, color="white")
    _, s, w = widest["t"]
    ae.text((s + w / 2) / 100, 64, "máscara de tiempo", ha="center", va="center", fontsize=FS_NOTA,
            color="white", rotation=90)
    af.text(0.04, 0.965, "0,6 · A + 0,4 · B\nA: $\\mathit{T.\\,falcklandii}$\nB: $\\mathit{Z.\\,capensis}$",
            transform=af.transAxes, ha="left", va="top", fontsize=FS_NOTA, color="white", linespacing=1.45)
    side_colorbar(fig, af, im, "Log-mel normalizado (z)")
    tf.finish(fig, "fig_representacion_audio")
    print("máscaras (eje, inicio, ancho):", masks)
    print("A =", XC_A, "(T. falcklandii), B =", XC_B, "(Z. capensis)")


if __name__ == "__main__":
    main()
