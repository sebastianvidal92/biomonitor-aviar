"""Figura teórica: semivariograma esquemático con efecto pepita, meseta y rango.

Modelo esférico gamma(d) = c0 + c * (1,5 d/a - 0,5 (d/a)^3) para d <= a y c0 + c para d > a,
con semivarianzas empíricas simuladas alrededor del modelo. Salida:
tesis/figuras/fig_semivariograma.pdf.

    uv run python scripts/theory_figs/fig_semivariograma.py
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

mpl.rcParams.update({"mathtext.fontset": "custom", "mathtext.rm": "Times New Roman",
                     "mathtext.it": "Times New Roman:italic"})

C0, C, A = 0.18, 0.72, 200.0   # pepita, meseta parcial y rango (km)


def esferico(d: np.ndarray) -> np.ndarray:
    r = np.clip(d / A, 0, 1)
    return C0 + C * (1.5 * r - 0.5 * r**3)


def main() -> None:
    rng = np.random.default_rng(7)
    d = np.linspace(0, 340, 400)
    d_emp = np.arange(12, 331, 16)
    g_emp = esferico(d_emp) + rng.normal(0, 0.025, d_emp.size)

    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(d, esferico(d), color=tf.C1, lw=1.6, label="modelo esférico ajustado", zorder=3)
    ax.scatter(d_emp, g_emp, s=14, color=tf.C2, zorder=4, label="semivarianzas empíricas")
    # meseta y rango
    ax.axhline(C0 + C, color=tf.GRIS, lw=0.8, ls="--", zorder=1)
    ax.axvline(A, color=tf.GRIS, lw=0.8, ls="--", zorder=1)
    ax.annotate("", xy=(A, 0), xytext=(0, 0), arrowprops={"arrowstyle": "<->", "color": tf.TINTA2, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax.text(A / 2, 0.025, "rango ($\\it{range}$)", ha="center", va="bottom", fontsize=8, color=tf.TINTA2)
    ax.text(335, C0 + C + 0.015, "meseta ($\\it{sill}$)", ha="right", va="bottom", fontsize=8, color=tf.TINTA2)
    # pepita
    ax.annotate("", xy=(0, C0), xytext=(0, 0), arrowprops={"arrowstyle": "<->", "color": tf.TINTA2, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax.annotate("efecto pepita ($\\it{nugget}$)", xy=(2, C0 / 2), xytext=(38, 0.30), fontsize=8, color=tf.TINTA2,
                arrowprops={"arrowstyle": "-", "color": tf.TINTA2, "lw": 0.6}, ha="left", va="center")
    # meseta parcial
    ax.annotate("", xy=(260, C0 + C), xytext=(260, C0), arrowprops={"arrowstyle": "<->", "color": tf.TINTA2, "lw": 0.8, "shrinkA": 0, "shrinkB": 0})
    ax.text(266, C0 + C / 2, "meseta parcial\n($\\it{partial\\ sill}$)", ha="left", va="center", fontsize=7.5, color=tf.TINTA2)
    ax.axhline(C0, xmin=0, xmax=0.78, color=tf.GRIS, lw=0.6, ls=":", zorder=1)

    ax.set_xlim(0, 345); ax.set_ylim(0, 1.05)
    ax.set_xlabel("Distancia entre pares de puntos $d$ (km)")
    ax.set_ylabel("Semivarianza $\\gamma(d)$")
    ax.set_xticks([0, 100, 200, 300]); ax.set_yticks([])
    ax.legend(loc="lower right", fontsize=7.5, frameon=False)
    tf.finish(fig, "fig_semivariograma")


if __name__ == "__main__":
    main()
