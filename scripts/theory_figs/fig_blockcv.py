"""Figura del marco teórico: validación cruzada espacial por bloques (fig_blockcv).

Mapa de Chile con los puntos de data/interim/sdm_dataset.csv (presencias y fondo)
coloreados según el fold espacial que les asigna el protocolo del script 28
(bloques de 200 km, 5 folds), más las fronteras de la grilla de bloques.

    uv run python scripts/theory_figs/fig_blockcv.py   →  tesis/figuras/fig_blockcv.pdf
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts")); sys.path.insert(0, str(ROOT / "src"))
tf = importlib.import_module("32_thesis_figures")
cv = importlib.import_module("28_sdm_spatial_cv")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

BLOCK_KM, K, N_SAMPLE = 200.0, 5, 8000
FOLD_COLORS = [tf.C1, tf.C2, tf.C3, tf.C4, tf.GRIS]


def main() -> None:
    # 1. Folds sobre el DataFrame COMPLETO (misma convención y semilla que el script 28).
    df = pd.read_csv(ROOT / "data" / "interim" / "sdm_dataset.csv").reset_index(drop=True)
    df["block"] = cv.assign_blocks(df, BLOCK_KM)
    df["fold"] = cv.assign_folds(df["block"], K)
    lat0, lon0 = df["lat"].min(), df["lon"].min()
    n_blocks = df["block"].nunique()
    print(f"{len(df):,} puntos, {n_blocks} bloques de {BLOCK_KM:.0f} km, {K} folds")
    print("registros por fold:", df["fold"].value_counts().sort_index().to_dict())
    print("bloques por fold:  ", df.drop_duplicates("block")["fold"].value_counts().sort_index().to_dict())

    # 2. Muestra para dibujar.
    s = df.sample(N_SAMPLE, random_state=42)

    fig, ax = plt.subplots(figsize=(tf.TEXTW * 0.55, 6.0))
    tf.chile_outline(ax, fill=True)

    # 3. Grilla de bloques. Convención del script 28: y_km = (lat - lat0)·111,
    #    x_km = (lon - lon0)·111·cos(lat). Las fronteras horizontales son latitudes
    #    constantes; las verticales son las curvas lon = lon0 + i·B / (111·cos lat).
    deg_lat = BLOCK_KM / 111.0
    n_rows = int(np.ceil((df["lat"].max() - lat0) / deg_lat))
    lat_top = lat0 + n_rows * deg_lat
    xlim, ylim = ax.get_xlim(), ax.get_ylim()
    grid_kw = dict(color=tf.GRIS_CLARO, lw=0.4, zorder=0.5, solid_capstyle="butt")
    for j in range(n_rows + 1):
        ax.plot(xlim, [lat0 + j * deg_lat] * 2, **grid_kw)
    lats = np.linspace(lat0, lat_top, 400)
    i = 0
    while True:
        lons = lon0 + i * BLOCK_KM / (111.0 * np.cos(np.radians(lats)))
        if lons.min() > xlim[1]:
            break
        ax.plot(lons, lats, **grid_kw)
        i += 1

    # 4. Puntos coloreados por fold (orden aleatorio de la muestra: ningún fold tapa a otro).
    ax.scatter(s["lon"], s["lat"], s=2.0, c=[FOLD_COLORS[f] for f in s["fold"]],
               linewidths=0, zorder=2, rasterized=False)
    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xticks([-74, -70]); ax.set_yticks(range(-55, -19, 5))
    ax.tick_params(labelsize=7)

    handles = [Line2D([], [], marker="o", ls="", color=c, markersize=4.5, label=f"Fold {k + 1}")
               for k, c in enumerate(FOLD_COLORS)]
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.04, 1.0), borderaxespad=0,
              handletextpad=0.4, labelspacing=0.7, fontsize=8)
    tf.finish(fig, "fig_blockcv")
    print("guardado:", tf.FIG / "fig_blockcv.pdf")


if __name__ == "__main__":
    main()
