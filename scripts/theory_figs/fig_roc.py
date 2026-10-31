"""Figura teórica: curva ROC y AUC (uno-contra-el-resto) de la CNN de audio.

A partir de reports/probs/audio_efficientnet_b0_test.npz dibuja las curvas ROC de tres
especies del conjunto de prueba (>= 6 grabaciones): la de mayor AUC, una cercana a la
mediana y la de menor AUC. Salida: tesis/figuras/fig_roc.pdf.

    uv run python scripts/theory_figs/fig_roc.py
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
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

# Cursiva solo en el nombre de especie (mathtext) con la misma Times del resto del texto.
mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic"})

MIN_TEST = 6  # grabaciones de prueba mínimas para que la especie sea elegible


def coma(x: float, nd: int) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def elegir_especies(P: np.ndarray, y: np.ndarray, classes: np.ndarray) -> list[tuple[int, float]]:
    """Índices (mayor AUC, cercana a la mediana, menor AUC) entre especies con >= MIN_TEST grabaciones.

    Los empates en AUC = 1 se deshacen con el margen de separación (min score positivo - max negativo).
    """
    n = np.bincount(y, minlength=len(classes))
    rows = []
    for k in range(len(classes)):
        if n[k] < MIN_TEST:
            continue
        auc = roc_auc_score((y == k).astype(int), P[:, k])
        margen = P[y == k, k].min() - P[y != k, k].max()
        rows.append((auc, margen, k))
    rows.sort()
    aucs = np.array([r[0] for r in rows])
    med = np.median(aucs)
    i_med = int(np.argmin(np.abs(aucs - med)))
    return [(rows[-1][2], rows[-1][0]), (rows[i_med][2], rows[i_med][0]), (rows[0][2], rows[0][0])]


def main() -> None:
    d = np.load(tf.REP / "probs" / "audio_efficientnet_b0_test.npz", allow_pickle=True)
    P, y, classes = d["probs_agg"], d["y_true"], d["classes"]
    sel = elegir_especies(P, y, classes)

    fig, (ax, axz) = plt.subplots(1, 2, figsize=(tf.TEXTW, 3.15), gridspec_kw={"wspace": 0.32})
    ZOOM = (0.0, 0.08, 0.80, 1.0)  # x0, x1, y0, y1 de la esquina ampliada
    DX, DY = 0.010, 0.015
    ax.plot([0, 1], [0, 1], color=tf.GRIS, lw=0.8, ls=":", label="azar (AUC = 0,5)", zorder=1)
    for (k, auc), col in zip(sel, (tf.C1, tf.C2, tf.C3)):
        fpr, tpr, _ = roc_curve((y == k).astype(int), P[:, k])
        gen, esp = tf.abbreviate(str(classes[k])).split(" ")
        for a in (ax, axz):
            a.plot(fpr, tpr, color=col, lw=1.4, drawstyle="steps-post", solid_joinstyle="miter",
                   label=rf"$\it{{{gen}}}$ $\it{{{esp}}}$, AUC = {coma(auc, 3)}", zorder=3)
        print(f"{classes[k]:28s} AUC = {auc:.4f}  n_test = {(y == k).sum()}")
    # panel izquierdo: curva completa con el recuadro que se amplía a la derecha
    ax.add_patch(plt.Rectangle((ZOOM[0] - DX, ZOOM[2]), ZOOM[1] - ZOOM[0] + DX, ZOOM[3] - ZOOM[2] + DY,
                               fill=False, edgecolor=tf.GRIS, lw=0.7, zorder=2))
    ax.set_xlabel("Tasa de falsos positivos (1 − especificidad)")
    ax.set_ylabel("Tasa de verdaderos positivos (sensibilidad)")
    ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02); ax.set_aspect("equal")
    ax.set_title("(a) Curva ROC completa", fontsize=9)
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_locator(MultipleLocator(0.2))
    # panel derecho: detalle de la esquina superior izquierda
    axz.set_xlim(ZOOM[0] - DX, ZOOM[1]); axz.set_ylim(ZOOM[2], ZOOM[3] + DY)
    axz.set_xticks([0, 0.02, 0.04, 0.06, 0.08]); axz.set_yticks([0.8, 0.85, 0.9, 0.95, 1.0])
    axz.set_xlabel("Tasa de falsos positivos"); axz.set_ylabel("Tasa de verdaderos positivos")
    axz.set_title("(b) Detalle del recuadro de (a)", fontsize=9)
    fmt = FuncFormatter(lambda v, _: str(int(v)) if float(v).is_integer() else f"{v:g}".replace(".", ","))
    for axis in (ax.xaxis, ax.yaxis, axz.xaxis, axz.yaxis):
        axis.set_major_formatter(fmt)
    ax.legend(loc="lower right", handlelength=1.8, borderaxespad=0.6, fontsize=7.5)
    tf.finish(fig, "fig_roc")


if __name__ == "__main__":
    main()
