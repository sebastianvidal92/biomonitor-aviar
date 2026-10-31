"""Figura: embudo de selección del conjunto de 55 especies objetivo (capítulo 4).

Cuatro etapas de filtrado (147 -> 133 -> 59 -> 14) y la composición final
(14 de conservación + 6 endémicas + 35 comunes = 55). Salida:
tesis/figuras/fig_seleccion_especies.pdf.

    uv run python scripts/theory_figs/fig_funnel_especies.py
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
tf = importlib.import_module("32_thesis_figures")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Polygon, Rectangle  # noqa: E402

AZUL = "#5B9BD5"; AZUL_CLARO = "#cfe0f3"; AMARILLO = "#FFC000"; GRIS = "#d9d9d9"; TINTA = "#404040"

STAGES = [  # (n, título, filtro que lleva a la etapa siguiente)
    (147, "aves clasificadas por el RCE", "sin islas oceánicas"),
    (133, "de distribución continental", "CR, EN, VU o endémica"),
    (59, "de interés de conservación", "≥ 20 grabaciones; sin pelágicas ni insulares"),
    (14, "con datos suficientes", ""),
]
FINAL = [(14, "conservación", AZUL), (6, "endémicas", AMARILLO), (35, "comunes", AZUL_CLARO)]


def main() -> None:
    fig, ax = plt.subplots(figsize=(tf.TEXTW, 3.5))
    ax.set_xlim(-0.02, 1.62); ax.set_ylim(-1.25, 4.05); ax.axis("off")
    bh, cx, full = 0.56, 0.5, 1.0
    y_top = 3.5
    ys = [y_top - i for i in range(len(STAGES))]
    for i, (n, label, filtro) in enumerate(STAGES):
        w = full * n / STAGES[0][0]; x0 = cx - w / 2; y = ys[i]
        ax.add_patch(Rectangle((x0, y - bh / 2), w, bh, facecolor=AZUL if i < 3 else "#2f6db3", edgecolor="white", lw=1))
        if i + 1 < len(STAGES):   # conector en forma de embudo hacia la etapa siguiente
            w2 = full * STAGES[i + 1][0] / STAGES[0][0]; y2 = ys[i + 1]
            ax.add_patch(Polygon([(x0, y - bh / 2), (x0 + w, y - bh / 2), (cx + w2 / 2, y2 + bh / 2), (cx - w2 / 2, y2 + bh / 2)],
                                 closed=True, facecolor=GRIS, edgecolor="none", alpha=0.7, zorder=0))
            ax.text(cx, (y - bh / 2 + y2 + bh / 2) / 2, filtro, ha="center", va="center", fontsize=7, color=TINTA, style="italic")
        txt = f"{n} {label}"
        if w > 0.45:
            ax.text(cx, y, txt, ha="center", va="center", fontsize=9, color="white", fontweight="bold")
        else:
            ax.text(cx + w / 2 + 0.03, y, txt, ha="left", va="center", fontsize=9, color=TINTA, fontweight="bold")
    # composición final
    y5 = ys[-1] - 1.15; w5 = full * 55 / STAGES[0][0]; x = cx - w5 / 2
    ax.annotate("", xy=(cx, y5 + bh / 2 + 0.03), xytext=(cx, ys[-1] - bh / 2 - 0.03),
                arrowprops={"arrowstyle": "-|>", "color": TINTA, "lw": 1.0})
    ax.text(cx + 0.03, (y5 + ys[-1]) / 2, "+ 6 endémicas continentales\n+ 35 comunes con más grabaciones chilenas",
            ha="left", va="center", fontsize=7.5, color=TINTA)
    for n, label, col in FINAL:
        w = full * n / STAGES[0][0]
        ax.add_patch(Rectangle((x, y5 - bh / 2), w, bh, facecolor=col, edgecolor="white", lw=1))
        ax.text(x + w / 2, y5, str(n), ha="center", va="center", fontsize=8.5, fontweight="bold",
                color="white" if col != AZUL_CLARO else TINTA)
        x += w
    ax.text(cx + w5 / 2 + 0.03, y5, "55 especies objetivo", ha="left", va="center", fontsize=9, color=TINTA, fontweight="bold")
    ax.text(cx + w5 / 2 + 0.03, y5 - 0.42, "conservación, endémicas, comunes", ha="left", va="center", fontsize=7, color=TINTA)
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    tf.finish(fig, "fig_seleccion_especies")


if __name__ == "__main__":
    main()
