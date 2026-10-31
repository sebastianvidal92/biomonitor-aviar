"""Figura teórica: presencias vs. pseudo-ausencias (fondo uniforme y fondo de grupo objetivo).

Tres mapas de Chile continental: (a) 500 presencias de Scelorchilus rubecula (GBIF),
(b) 1500 puntos de fondo uniforme sobre la grilla terrestre (alternativa no usada) y
(c) 1500 puntos del fondo de grupo objetivo empleado en los SDM (data/interim/sdm_dataset.csv),
muestreado al azar de los avistamientos de todas las especies del conjunto (16_extract_covariates.py).

    uv run python scripts/theory_figs/fig_fondo_objetivo.py  →  tesis/figuras/fig_fondo_objetivo.pdf
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

mpl.rcParams["mathtext.fontset"] = "stix"  # itálicas compatibles con Times
DATA = ROOT / "data"
ESPECIE = "Scelorchilus rubecula"
N_PRES, N_FONDO, SEED = 500, 1500, 42
# Notas en el océano frente al norte, entre las marcas de 20°S y 30°S y separadas del eje y.
X_NOTA, Y_ESPECIE, Y_N, FS_NOTA = -75.6, -21.4, -23.0, 6.0


def _medir_notas(fig, ax, textos, mask: pd.DataFrame) -> None:
    """Imprime la holgura (pt) de cada nota respecto del eje y (izquierda) y de la costa (derecha)."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    pt_por_px = 72.0 / fig.dpi
    for t in textos:
        bb = t.get_window_extent(r)
        (lon0, lat0), (lon1, lat1) = ax.transData.inverted().transform([[bb.x0, bb.y0], [bb.x1, bb.y1]])
        x_eje = ax.transData.transform([[ax.get_xlim()[0], 0]])[0, 0]
        franja = mask[(mask["in_chile"] == 1) & (mask["lat"] >= lat0 - 0.1) & (mask["lat"] <= lat1 + 0.1)]
        costa = franja["lon"].min() - 0.05  # borde oeste de la celda costera más occidental (contorno a 0,5)
        x_costa = ax.transData.transform([[costa, 0]])[0, 0]
        print(f"  '{t.get_text()}': lon {lon0:.2f}..{lon1:.2f}, lat {lat1:.1f}..{lat0:.1f} | "
              f"eje y → {(bb.x0 - x_eje) * pt_por_px:.1f} pt | costa ({costa:.2f}) → {(x_costa - bb.x1) * pt_por_px:.1f} pt")


def main() -> None:
    occ = pd.read_csv(DATA / "interim" / "occurrences.csv")
    pres = occ[occ["scientific_name"] == ESPECIE].sample(N_PRES, random_state=SEED)

    mask = pd.read_csv(DATA / "processed" / "chile_grid_mask.csv")
    unif = mask[mask["in_chile"] == 1].sample(N_FONDO, random_state=SEED).copy()
    rng = np.random.default_rng(SEED)
    unif["lat"] += rng.uniform(-0.05, 0.05, len(unif))
    unif["lon"] += rng.uniform(-0.05, 0.05, len(unif))

    sdm = pd.read_csv(DATA / "interim" / "sdm_dataset.csv")
    tgb = sdm[sdm["label"] == 0].sample(N_FONDO, random_state=SEED)

    # (b) usa TINTA2 (más oscuro que el contorno en GRIS) para que silueta y puntos no se fundan en los fiordos.
    paneles = [
        ("(a) Presencias de una especie", pres, tf.C2, 4.0, (tf.abbreviate(ESPECIE), N_PRES)),
        ("(b) Fondo uniforme (no usado)", unif, tf.TINTA2, 2.5, (None, N_FONDO)),
        ("(c) Fondo de grupo objetivo (usado)", tgb, tf.C1, 2.5, (None, N_FONDO)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(tf.TEXTW, 4.8), sharey=True,
                             gridspec_kw={"wspace": 0.0, "left": 0.08, "right": 0.99, "top": 0.95, "bottom": 0.08})
    notas = []
    for k, (ax, (titulo, df, color, s, nota)) in enumerate(zip(axes, paneles)):
        tf.chile_outline(ax, fill=True)
        ax.scatter(df["lon"], df["lat"], s=s, c=color, alpha=0.75, linewidths=0, zorder=3, rasterized=False)
        ax.set_title(titulo, fontsize=8, pad=4)
        especie, n = nota  # misma x/y en los tres paneles para mantener la alineación
        if especie:
            notas.append(ax.text(X_NOTA, Y_ESPECIE, especie, fontsize=FS_NOTA, fontstyle="italic",
                                 color=tf.TINTA2, ha="left", va="top"))
        notas.append(ax.text(X_NOTA, Y_N, f"$n$ = {n}", fontsize=FS_NOTA, color=tf.TINTA2, ha="left", va="top"))
        if k > 0:  # ejes solo en el panel izquierdo
            ax.set_xlabel(""); ax.set_ylabel("")
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            for sp in ax.spines.values():
                sp.set_visible(False)
        else:
            ax.set_xticks([-74, -70, -66]); ax.set_yticks([-20, -30, -40, -50])
            ax.set_xticklabels(["74°O", "70°O", "66°O"]); ax.set_yticklabels(["20°S", "30°S", "40°S", "50°S"])
    print("Holgura de las notas del panel (a):")
    _medir_notas(fig, axes[0], notas[:2], mask)
    tf.finish(fig, "fig_fondo_objetivo")
    print("tesis/figuras/fig_fondo_objetivo.pdf")


if __name__ == "__main__":
    main()
