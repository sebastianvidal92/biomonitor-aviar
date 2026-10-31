"""Figuras de la tesis (vectoriales, estilo sobrio) → tesis/figuras/*.pdf.

A diferencia de `26_report_figures.py` (PNG con títulos incrustados, para la presentación),
estas figuras van sin título (el pie de figura lo pone LaTeX), con tipografía Times, un
solo eje por panel y una paleta categórica validada de ≤4 tonos. Cada figura se genera
solo si sus insumos existen, así el script puede correrse antes y después de los scripts
27-31:

    uv run python scripts/32_thesis_figures.py

Insumos: data/interim/*.csv, data/processed/chile_grid*.csv, reports/*.csv y, si existen,
reports/eval_extended.*, fusion_*.csv, sdm_spatial_cv.csv, frozen_baseline.csv,
llm_baseline.csv, confusion_audio_cnn.csv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biomonitor.config import PROJECT_ROOT  # noqa: E402
from biomonitor.evaluation import species_groups  # noqa: E402

FIG = PROJECT_ROOT / "tesis" / "figuras"
REP = PROJECT_ROOT / "reports"
DATA = PROJECT_ROOT / "data"

# Paleta categórica validada (orden fijo) + neutros. Secuenciales: un solo tono.
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
GRIS, GRIS_CLARO, TINTA, TINTA2 = "#8a8a8a", "#d9d9d9", "#1a1a1a", "#555555"
GRUPO_COLOR = {"Amenazada": C2, "Endémica": C3, "Común": C1}
TEXTW = 6.5  # ancho de texto (pulgadas): carta con márgenes de 2,5 cm

mpl.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "STIXGeneral", "DejaVu Serif"],
    "font.size": 9, "axes.labelsize": 9, "axes.titlesize": 9.5, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "axes.edgecolor": TINTA2, "axes.linewidth": 0.6,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5, "grid.linestyle": "-",
    "xtick.color": TINTA2, "ytick.color": TINTA2, "text.color": TINTA, "axes.labelcolor": TINTA,
    "legend.frameon": False, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42, "axes.titleweight": "normal", "axes.titlelocation": "left",
})
saved: list[str] = []


def finish(fig, name: str) -> None:
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    plt.close(fig)
    saved.append(name)


def exists(*paths: Path) -> bool:
    return all(p.exists() for p in paths)


def chile_outline(ax, fill: bool = True) -> None:
    """Silueta de Chile continental a partir de la máscara de la grilla (0,1°)."""
    m = pd.read_csv(DATA / "processed" / "chile_grid_mask.csv")
    lats, lons = np.sort(m["lat"].unique()), np.sort(m["lon"].unique())
    Z = m.pivot(index="lat", columns="lon", values="in_chile").reindex(index=lats, columns=lons).fillna(0).values
    if fill:
        ax.contourf(lons, lats, Z, levels=[0.5, 1.5], colors=["#f2f2f2"], zorder=0)
    ax.contour(lons, lats, Z, levels=[0.5], colors=[GRIS], linewidths=0.4, zorder=1)
    ax.set_aspect(1 / np.cos(np.radians(37)))
    ax.set_xlim(-76, -66); ax.set_ylim(-56.2, -17.3)
    ax.grid(False)
    ax.set_xlabel("Longitud"); ax.set_ylabel("Latitud")


def abbreviate(name: str) -> str:
    g, s = name.split(" ", 1)
    return f"{g[0]}. {s}"


# ============================ Capítulo 4 — datos ============================
def fig_datos_por_especie() -> None:
    a = pd.read_csv(DATA / "interim" / "audio_manifest_split.csv")
    i = pd.read_csv(DATA / "interim" / "image_manifest_split.csv")
    groups = species_groups()
    chile = a[a["country"] == "Chile"]["scientific_name"].value_counts()
    otros = a[a["country"] != "Chile"]["scientific_name"].value_counts()
    fotos = i["scientific_name"].value_counts()
    sp = sorted(a["scientific_name"].unique(),
                key=lambda s: (["Amenazada", "Endémica", "Común"].index(groups[s]), -(chile.get(s, 0) + otros.get(s, 0))))
    y = np.arange(len(sp))[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 8.2), sharey=True, gridspec_kw={"width_ratios": [1, 1], "wspace": 0.08})
    ch = np.array([chile.get(s, 0) for s in sp]); ot = np.array([otros.get(s, 0) for s in sp])
    a1.barh(y, ch, color=C1, height=0.72, label="Grabadas en Chile")
    a1.barh(y, ot, left=ch, color="#a9c9ef", height=0.72, label="Otros países")
    a1.set_yticks(y); a1.set_yticklabels([abbreviate(s) for s in sp], fontstyle="italic", fontsize=6.8)
    a1.set_xlabel("Grabaciones (Xeno-canto)"); a1.legend(loc="lower right")
    a1.set_xlim(0, 55)
    a2.barh(y, [fotos.get(s, 0) for s in sp], color=C1, height=0.72)
    a2.set_xlabel("Fotografías (iNaturalist, grado de investigación)")
    a2.set_xlim(0, 160)
    for s, yy in zip(sp, y):  # franja de grupo a la izquierda
        a1.plot([-2.5, -2.5], [yy - 0.4, yy + 0.4], color=GRUPO_COLOR[groups[s]], lw=2.2, clip_on=False)
    for s, yy in zip(sp, y):
        if fotos.get(s, 0) == 0:
            a2.text(2, yy, "< 15 fotos: solo audio", fontsize=6.3, color=TINTA2, va="center")
    handles = [plt.Line2D([], [], color=col, lw=3, label=g) for g, col in GRUPO_COLOR.items()]
    handles += [plt.Rectangle((0, 0), 1, 1, color=C1, label="Grabadas en Chile"),
                plt.Rectangle((0, 0), 1, 1, color="#a9c9ef", label="Grabadas en otros países")]
    a1.get_legend().remove()
    fig.legend(handles=handles, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 0.935), fontsize=7.5)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    finish(fig, "fig_datos_por_especie")


def fig_avistamientos() -> None:
    from matplotlib.colors import LinearSegmentedColormap, LogNorm
    occ = pd.read_csv(DATA / "interim" / "occurrences.csv")
    blues = plt.get_cmap("Blues")
    cmap = LinearSegmentedColormap.from_list("blues_dark", blues(np.linspace(0.3, 1.0, 256)))
    fig = plt.figure(figsize=(TEXTW, 7.4))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.75, 1], hspace=0.32, wspace=0.12)
    panels = [("Todos los registros", occ), ("Verano austral (dic.--feb.)", occ[occ["month"].isin([12, 1, 2])]),
              ("Invierno austral (jun.--ago.)", occ[occ["month"].isin([6, 7, 8])])]
    xe, ye = np.arange(-76, -65.9, 0.5), np.arange(-56.5, -17.0, 0.5)
    H0, _, _ = np.histogram2d(occ["lon"], occ["lat"], bins=[xe, ye]); vmax = H0.max()
    for j, (title, d) in enumerate(panels):
        ax = fig.add_subplot(gs[0, j]); chile_outline(ax)
        H, _, _ = np.histogram2d(d["lon"], d["lat"], bins=[xe, ye])
        hb = ax.pcolormesh(xe, ye, np.ma.masked_less(H.T, 1), norm=LogNorm(vmin=1, vmax=vmax), cmap=cmap,
                           shading="flat", zorder=2, rasterized=True)
        ax.set_title(f"{title}\n(n = {len(d):,})".replace(",", "."), fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("Latitud" if j == 0 else ""); ax.set_xticks([-74, -70]); ax.tick_params(labelsize=7)
        if j:
            ax.set_yticklabels([])
    cb = fig.colorbar(hb, ax=fig.axes[:3], orientation="horizontal", fraction=0.03, pad=0.12, shrink=0.5)
    cb.set_label("Registros por celda de 0,5° (escala logarítmica)"); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
    ay = fig.add_subplot(gs[1, :2]); amo = fig.add_subplot(gs[1, 2])
    # Años individuales desde 2015; los anteriores (2000--2014) se agrupan en una sola barra.
    y_all = occ[occ["year"].between(2000, 2026)]["year"].astype(int)
    labels_yr = ["2000–2014"] + [str(a) for a in range(2015, int(y_all.max()) + 1)]
    counts_yr = [int((y_all < 2015).sum())] + [int((y_all == a).sum()) for a in range(2015, int(y_all.max()) + 1)]
    ay.bar(range(len(labels_yr)), np.array(counts_yr) / 1000, color=[C2] + [C1] * (len(labels_yr) - 1), width=0.72)
    ay.set_xticks(range(len(labels_yr))); ay.set_xticklabels(labels_yr, rotation=45, ha="right", fontsize=7)
    ay.set_xlabel("Año"); ay.set_ylabel("Registros (miles)"); ay.grid(axis="x", visible=False)
    mo = occ["month"].dropna().astype(int).value_counts().reindex(range(1, 13), fill_value=0)
    amo.bar(range(1, 13), mo.values / 1000, color=C1, width=0.72)
    amo.set_xticks(range(1, 13)); amo.set_xticklabels(list("EFMAMJJASOND")); amo.set_xlabel("Mes"); amo.grid(axis="x", visible=False)
    finish(fig, "fig_avistamientos")


def fig_covariables() -> None:
    g = pd.read_csv(DATA / "processed" / "chile_grid.csv")
    m = pd.read_csv(DATA / "processed" / "chile_grid_mask.csv")
    g = g.merge(m, on=["lat", "lon"]); g = g[g["in_chile"] == 1]
    panels = [("ndvi", "NDVI\n(MODIS)", "Greens", None), ("temp", "Temperatura media\nanual (°C)", "Oranges", None),
              ("precip", "Precipitación\nanual (mm)", "Blues", 3000), ("elev", "Elevación\n(m)", "Purples", None)]
    fig, axes = plt.subplots(1, 4, figsize=(TEXTW, 4.6), gridspec_kw={"wspace": 0.15})
    for ax, (col, label, cmap, vmax) in zip(axes, panels):
        chile_outline(ax, fill=False)
        sc = ax.scatter(g["lon"], g["lat"], c=g[col], s=1.1, marker="s", cmap=cmap, vmin=g[col].min(),
                        vmax=vmax or g[col].max(), linewidths=0, zorder=2)
        cb = fig.colorbar(sc, ax=ax, orientation="horizontal", fraction=0.04, pad=0.06)
        cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel(""); ax.set_ylabel("")
        ax.set_xticks([-74, -70]); ax.tick_params(labelsize=7)
        if ax is not axes[0]:
            ax.set_yticklabels([])
    axes[0].set_ylabel("Latitud")
    finish(fig, "fig_covariables")


def fig_corpus_especies() -> None:
    """Censo de grabaciones de Xeno-canto con localidad en Chile (Tabla 4.1), por especie."""
    d = pd.read_csv(PROJECT_ROOT / "tesis" / "datos" / "corpus_resumen.csv")
    groups = species_groups()
    d["g"] = d["scientific_name"].map(groups)
    d = d.sort_values(["xc_chile"], ascending=True).reset_index(drop=True)
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(TEXTW, 7.0))
    ax.barh(y, d["xc_chile"], color=[GRUPO_COLOR[g] for g in d["g"]], height=0.72)
    labels = []
    for r in d.itertuples(index=False):
        cat = r.cat if isinstance(r.cat, str) and r.cat not in ("NE", "nan") else ""
        labels.append(abbreviate(r.scientific_name) + (f" ({cat})" if cat else ""))
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6.8, fontstyle="italic")
    for yy, v in zip(y, d["xc_chile"]):
        ax.text(v + 1.5, yy, str(int(v)), va="center", fontsize=6.3, color=TINTA2)
    ax.set_xlabel("Grabaciones de Xeno-canto con localidad en Chile"); ax.set_xlim(0, 200)
    ax.margins(y=0.01); ax.grid(axis="y", visible=False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=g) for g, col in GRUPO_COLOR.items()]
    ax.legend(handles=handles, loc="lower right")
    finish(fig, "fig_corpus_especies")


def fig_llm_zero_few() -> None:
    """Antes/después del few-shot: F1 por grupo, zero-shot vs few-shot vs clasificador propio."""
    groups = species_groups()
    z = REP / "llm_zeroshot_per_class.csv"; fw = REP / "llm_fewshot_per_class.csv"
    if not (z.exists() and fw.exists()):
        return
    def grp_f1(csv):
        d = pd.read_csv(csv).rename(columns={"Unnamed: 0": "sp", "f1-score": "f1"}); d["g"] = d["sp"].map(groups)
        return [d[d["g"] == g]["f1"].mean() for g in ["Amenazada", "Endémica", "Común"]] + [d["f1"].mean()]
    zs, fs = grp_f1(z), grp_f1(fw)
    b3c = REP / "image_efficientnet_b3_tta_per_class.csv"
    b3 = grp_f1(b3c if b3c.exists() else REP / "image_efficientnet_b3_per_class.csv")
    labels = ["Amenazadas", "Endémicas", "Comunes", "Todas (53)"]
    x, w = np.arange(4), 0.26
    fig, ax = plt.subplots(figsize=(5.2, 2.6))
    b1 = ax.bar(x - w, zs, w, color="#f5c4a9", label="IA generativa, zero-shot")
    b2 = ax.bar(x, fs, w, color=C2, label="IA generativa, few-shot")
    b3_ = ax.bar(x + w, b3, w, color=C1, label="EfficientNet-B3 ajustada (TTA)")
    for bars in (b1, b2, b3_):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{b.get_height():.2f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=6.5)
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("F1 medio por especie"); ax.set_ylim(0, 1.0)
    ax.legend(fontsize=7, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False); ax.grid(axis="x", visible=False)
    finish(fig, "fig_llm_zero_few")


# ========================== Capítulo 5 — resultados =========================
CURVES = {  # épocas clave registradas en docs/MODULO_AUDIO.md y MODULO_IMAGEN.md
    "Audio: CNN\n(EfficientNet-B0)": dict(
        ep=[1, 5, 10, 13, 19, 23, 38, 50, 60, 67, 81, 90, 100],
        tr_loss=[4.043, 2.621, 2.248, 2.080, 1.984, 1.827, 1.691, 1.591, 1.399, 1.411, 1.357, 1.233, 1.241],
        val_acc=[0.175, 0.558, 0.680, 0.723, 0.749, 0.777, 0.794, 0.812, 0.815, 0.815, 0.815, 0.812, 0.812],
        val_f1=[0.147, 0.539, 0.675, 0.721, 0.748, 0.775, 0.793, 0.809, 0.810, 0.811, 0.811, 0.809, 0.808], best=67),
    "Audio: AST\n(transformer)": dict(
        ep=list(range(1, 21)),
        tr_loss=[2.844, 1.480, 1.112, 0.983, 0.927, 0.893, 0.867, 0.841, 0.843, 0.814, 0.801, 0.802, 0.788, 0.788,
                 0.791, 0.777, 0.776, 0.777, 0.775, 0.774],
        val_acc=[0.675, 0.779, 0.802, 0.805, 0.812, 0.827, 0.827, 0.843, 0.850, 0.838, 0.860, 0.850, 0.840, 0.850,
                 0.840, 0.848, 0.845, 0.843, 0.840, 0.843],
        val_f1=[0.668, 0.781, 0.799, 0.802, 0.819, 0.829, 0.822, 0.846, 0.850, 0.842, 0.860, 0.851, 0.842, 0.853,
                0.843, 0.850, 0.843, 0.842, 0.839, 0.841], best=11),
    "Imagen:\nEfficientNet-B3": dict(
        ep=[1, 5, 6, 13, 18, 23, 26, 27, 31, 32, 36, 38, 48, 50],
        tr_loss=[3.218, 1.699, 1.697, 1.497, 1.465, 1.465, 1.425, 1.418, 1.389, 1.367, 1.332, 1.401, 1.292, 1.360],
        val_acc=[0.660, 0.794, 0.807, 0.821, 0.827, 0.828, 0.825, 0.835, 0.835, 0.837, 0.842, 0.855, 0.857, 0.852],
        val_f1=[0.626, 0.761, 0.779, 0.784, 0.792, 0.800, 0.800, 0.801, 0.804, 0.805, 0.814, 0.828, 0.824, 0.823], best=38),
}


def fig_curvas_entrenamiento() -> None:
    fig, axes = plt.subplots(2, 3, figsize=(TEXTW, 4.4), gridspec_kw={"hspace": 0.5, "wspace": 0.3, "height_ratios": [1.3, 1]})
    for j, (title, c) in enumerate(CURVES.items()):
        ax, al = axes[0, j], axes[1, j]
        ax.plot(c["ep"], c["val_acc"], "-", color=C1, lw=1.4, label="Accuracy (val.)")
        ax.plot(c["ep"], c["val_f1"], "-", color=C2, lw=1.4, label="F1 macro (val.)")
        ax.axvline(c["best"], color=GRIS, lw=0.8, ls="-")
        bi = c["ep"].index(c["best"])
        ax.plot([c["best"]], [c["val_acc"][bi]], "o", color=C1, ms=5, mec="white", mew=0.8, zorder=5)
        ax.plot([c["best"]], [c["val_f1"][bi]], "o", color=C2, ms=5, mec="white", mew=0.8, zorder=5)
        ax.text(c["best"], 0.03, f" mejor época: {c['best']} ", color=TINTA2, fontsize=7, va="bottom",
                ha="right" if c["best"] > max(c["ep"]) / 2 else "left")
        ax.set_ylim(0, 1); ax.set_title(title); ax.set_xlabel("")
        al.plot(c["ep"], c["tr_loss"], "-", color=C3, lw=1.4)
        al.set_xlabel("Época"); al.set_ylim(0, max(c["tr_loss"]) * 1.1)
        if j == 0:
            ax.set_ylabel("Métrica de validación"); al.set_ylabel("Pérdida de\nentrenamiento")
            fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), fontsize=8)
    finish(fig, "fig_curvas_entrenamiento")


def fig_sobreajuste() -> None:
    """Curvas de pérdida de la CNN sin regularizar (log del script 07 --tag sinreg) vs. regularizada."""
    p = REP / "training_logs" / "audio_efficientnet_b0_sinreg.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    reg = CURVES["Audio: CNN\n(EfficientNet-B0)"]
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    ax.plot(d["ep"], d["tr_loss"], "-", color=C2, lw=1.4, label="Sin regularizar: entrenamiento")
    if "val_loss" in d:
        ax.plot(d["ep"], d["val_loss"], "--", color=C2, lw=1.4, label="Sin regularizar: validación")
    ax.plot(reg["ep"], reg["tr_loss"], "-", color=C1, lw=1.4, label="Regularizada: entrenamiento")
    val_reg = [3.514, 2.246, 1.861, 1.763, 1.673, 1.583, 1.573, 1.511, 1.485, 1.524, 1.499, 1.472, 1.473]
    ax.plot(reg["ep"], val_reg, "--", color=C1, lw=1.4, label="Regularizada: validación")
    ax.set_xlabel("Época"); ax.set_ylabel("Pérdida (entropía cruzada)"); ax.set_ylim(0, None)
    ax.legend(fontsize=7, loc="upper right")
    finish(fig, "fig_sobreajuste")


def fig_audio_cnn_vs_ast() -> None:
    single, agg = [0.812, 0.832], [0.858, 0.870]  # valores de la tabla 5.2 (cap5_resultados.tex)
    x, w = np.arange(2), 0.34
    fig, ax = plt.subplots(figsize=(3.6, 2.7))
    b1 = ax.bar(x - w / 2, single, w, color="#a9c9ef", label="Ventana única (5 s)")
    b2 = ax.bar(x + w / 2, agg, w, color=C1, label="Agregado por grabación")
    for b in list(b1) + list(b2):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.004, f"{b.get_height():.3f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7.5, color=TINTA)
    ax.axhline(0.85, color=GRIS, lw=0.8); ax.text(1.68, 0.852, "meta 0,85", color=TINTA2, fontsize=7, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(["CNN (EfficientNet-B0)", "AST (transformer)"]); ax.set_ylabel("Accuracy (test)")
    ax.set_xlim(-0.5, 1.7); ax.set_ylim(0.70, 0.90); ax.grid(axis="x", visible=False)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=7)
    finish(fig, "fig_audio_cnn_vs_ast")


def fig_f1_por_especie() -> None:
    groups = species_groups()
    a = pd.read_csv(REP / "audio_efficientnet_b0_per_class.csv").rename(columns={"Unnamed: 0": "sp", "f1-score": "f1"})
    img_csv = REP / "image_efficientnet_b3_tta_per_class.csv"
    i = pd.read_csv(img_csv if img_csv.exists() else REP / "image_efficientnet_b3_per_class.csv").rename(
        columns={"Unnamed: 0": "sp", "f1-score": "f1"})
    a = a[a["sp"].isin(groups)].sort_values("f1"); imap = dict(zip(i["sp"], i["f1"]))
    y = np.arange(len(a))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 8.2), sharey=True, gridspec_kw={"wspace": 0.08})
    cols = [GRUPO_COLOR[groups[s]] for s in a["sp"]]
    a1.barh(y, a["f1"], color=cols, height=0.72)
    a1.set_yticks(y); a1.set_yticklabels([abbreviate(s) for s in a["sp"]], fontstyle="italic", fontsize=6.8)
    a1.set_xlabel("F1 por especie — audio (CNN, agregado)"); a1.set_xlim(0, 1)
    a1.axvline(a["f1"].mean(), color=GRIS, lw=0.8); a1.text(a["f1"].mean(), -0.6, f" media {a['f1'].mean():.2f}".replace(".", ","), fontsize=7, color=TINTA2, va="top")
    vals = [imap.get(s, np.nan) for s in a["sp"]]
    a2.barh(y, [0 if np.isnan(v) else v for v in vals], color=cols, height=0.72)
    for yy, v in zip(y, vals):
        if np.isnan(v):
            a2.text(0.01, yy, "solo audio", fontsize=6.5, color=TINTA2, va="center")
    a2.set_xlabel("F1 por especie — imagen (EfficientNet-B3, TTA)"); a2.set_xlim(0, 1)
    mi = np.nanmean(vals); a2.axvline(mi, color=GRIS, lw=0.8); a2.text(mi, -0.6, f" media {mi:.2f}".replace(".", ","), fontsize=7, color=TINTA2, va="top")
    handles = [plt.Rectangle((0, 0), 1, 1, color=col, label=g) for g, col in GRUPO_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.935), fontsize=7.5)
    for ax in (a1, a2):
        ax.grid(axis="y", visible=False)
    finish(fig, "fig_f1_por_especie")


def fig_fusion() -> None:
    labels = ["Audio solo", "Imagen sola", "Fusión tardía"]
    acc, sd = [0.867, 0.825, 0.950], [0, 0, 0]
    fs = REP / "fusion_summary.csv"
    if fs.exists():
        d = pd.read_csv(fs); d = d[d["rama_audio"].str.startswith("CNN")]
        sel = [d[d["modalidad"] == m].iloc[0] for m in ["Audio solo", "Imagen sola", "Fusión tardía (α=0,5)"]]
        acc, sd = [float(r["accuracy_media"]) for r in sel], [float(r["accuracy_sd"]) for r in sel]
    st = pd.read_csv(REP / "fusion_strategies.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 2.8), gridspec_kw={"wspace": 0.35})
    bars = a1.bar(labels, acc, color=["#a9c9ef", "#a9c9ef", C1], width=0.6, yerr=sd if any(sd) else None,
                  error_kw={"elinewidth": 0.8, "capsize": 3, "ecolor": TINTA2})
    for b, v in zip(bars, acc):
        a1.text(b.get_x() + b.get_width() / 2, v + (max(sd) if any(sd) else 0) + 0.005, f"{v:.3f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7.5)
    a1.axhline(0.85, color=GRIS, lw=0.8, label="meta 0,85"); a1.legend(loc="upper left", fontsize=7, frameon=False)
    a1.set_ylim(0.7, 1.0); a1.set_ylabel("Accuracy (pares por especie)"); a1.grid(axis="x", visible=False)
    a1.set_title("Fusión frente a cada modalidad")
    x, w = np.arange(len(st)), 0.36
    short = ["Tardía\n(decisión)", "Temprana\n(concatenación)", "Intermedia\n(proyección)"]
    b1 = a2.bar(x - w / 2, st["accuracy"], w, color=C1, label="Accuracy")
    b2 = a2.bar(x + w / 2, st["f1_macro"], w, color="#a9c9ef", label="F1 macro")
    for b in list(b1) + list(b2):
        a2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}".replace(".", ","),
                ha="center", va="bottom", fontsize=7)
    a2.set_xticks(x); a2.set_xticklabels(short[:len(st)]); a2.set_ylim(0.85, 0.96)
    a2.legend(loc="upper right", fontsize=7); a2.grid(axis="x", visible=False)
    a2.set_title("Estrategias de fusión (mismas condiciones)")
    finish(fig, "fig_fusion")


def fig_fusion_alpha() -> None:
    p = REP / "fusion_alpha_sweep.csv"
    if not p.exists():
        return
    d = pd.read_csv(p); d = d[d["fusion"] == "aritmética"]
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    for (name, col) in zip(sorted(d["rama_audio"].unique()), [C1, C2]):
        s = d[d["rama_audio"] == name].sort_values("alpha")
        ax.plot(s["alpha"], s["accuracy_media"], "-o", color=col, lw=1.4, ms=3, label=f"Rama de audio: {name}")
        ax.fill_between(s["alpha"], s["accuracy_media"] - s["accuracy_sd"], s["accuracy_media"] + s["accuracy_sd"],
                        color=col, alpha=0.15, lw=0)
    ax.set_xlabel("Peso del audio α (0 = solo imagen, 1 = solo audio)"); ax.set_ylabel("Accuracy (media ± d.e.)")
    ax.set_xticks(np.arange(0, 1.01, 0.2)); ax.legend(loc="lower center", fontsize=7)
    finish(fig, "fig_fusion_alpha")


def fig_sdm_modelos() -> None:
    ml = pd.read_csv(REP / "sdm_ml_auc.csv"); dl = pd.read_csv(REP / "sdm_auc_por_especie.csv")
    labels = ["MaxEnt", "Red neuronal\nconjunta", "Gradient\nBoosting", "Random\nForest"]
    rand = [ml["maxent"].mean(), dl["auc_dl"].mean(), ml["hgb_base"].mean(), ml["rf_base"].mean()]
    sp = REP / "sdm_spatial_cv.csv"
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    x = np.arange(4)
    if sp.exists():
        s = pd.read_csv(sp); s = s[s["block_km"] == s["block_km"].min()]
        spat = [s["maxent"].mean(), s["dl"].mean(), s["hgb"].mean(), s["rf"].mean()]
        w = 0.36
        b1 = ax.bar(x - w / 2, rand, w, color="#a9c9ef", label="Partición aleatoria")
        b2 = ax.bar(x + w / 2, spat, w, color=C1, label=f"Bloques espaciales ({int(s['block_km'].iloc[0])} km)")
        for b in list(b1) + list(b2):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=6.8)
        ax.legend(loc="upper left", fontsize=7)
        ax.set_ylim(min(rand + spat) - 0.03, max(rand + spat) + 0.03)
    else:
        cols = [GRIS, "#a9c9ef", "#a9c9ef", C1]
        b1 = ax.bar(x, rand, 0.6, color=cols)
        for b in b1:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.002, f"{b.get_height():.3f}".replace(".", ","),
                    ha="center", va="bottom", fontsize=7.5)
        ax.set_ylim(0.80, 0.90)
    ax.axhline(rand[0], color=GRIS, lw=0.7, ls=(0, (3, 3)))
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylabel("AUC-ROC media (54 especies)"); ax.grid(axis="x", visible=False)
    finish(fig, "fig_sdm_modelos")


def fig_sdm_satelite() -> None:
    ml = pd.read_csv(REP / "sdm_ml_auc.csv"); t = pd.read_csv(REP / "sdm_temporal_auc.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(TEXTW, 2.8), gridspec_kw={"wspace": 0.3})
    l1 = ["RF\n(base)", "RF +\nembedding\nsatelital", "Red +\nembedding\nsatelital", "Red + parche\nde cobertura\nde suelo"]
    v1 = [ml["rf_base"].mean(), ml["rf_emb"].mean(), 0.767, 0.613]
    b = a1.bar(l1, v1, color=[C1, "#a9c9ef", "#a9c9ef", "#a9c9ef"], width=0.6)
    for bb in b:
        a1.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.004, f"{bb.get_height():.3f}".replace(".", ","), ha="center", va="bottom", fontsize=7.5)
    a1.axhline(v1[0], color=GRIS, lw=0.7, ls=(0, (3, 3))); a1.set_ylim(0.55, 0.93); a1.set_ylabel("AUC-ROC media")
    a1.set_title("Variables satelitales no estructuradas"); a1.grid(axis="x", visible=False); a1.tick_params(axis="x", labelsize=7)
    l2 = ["MaxEnt", "RF\n(base)", "RF + satélite\nfijo (2023)", "RF + satélite\npareado al año"]
    v2 = [t["maxent"].mean(), t["rf_base"].mean(), t["rf_fijo"].mean(), t["rf_pareado"].mean()]
    b = a2.bar(l2, v2, color=[GRIS, C1, "#a9c9ef", "#a9c9ef"], width=0.6)
    for bb in b:
        a2.text(bb.get_x() + bb.get_width() / 2, bb.get_height() + 0.001, f"{bb.get_height():.3f}".replace(".", ","), ha="center", va="bottom", fontsize=7.5)
    a2.axhline(v2[1], color=GRIS, lw=0.7, ls=(0, (3, 3))); a2.set_ylim(0.85, 0.90)
    a2.set_title("Pareo temporal (observaciones 2017-2025)"); a2.grid(axis="x", visible=False); a2.tick_params(axis="x", labelsize=7)
    finish(fig, "fig_sdm_satelite")


def fig_sdm_auc_especie() -> None:
    groups = species_groups()
    d = pd.read_csv(REP / "sdm_ml_auc.csv").sort_values("rf_base")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(TEXTW * 0.75, 7.6))
    ax.hlines(y, d["maxent"], d["rf_base"], color=GRIS_CLARO, lw=1.6, zorder=1)
    ax.scatter(d["maxent"], y, color=GRIS, s=14, label="MaxEnt", zorder=2)
    ax.scatter(d["rf_base"], y, color=C1, s=16, label="Random Forest", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels([abbreviate(s) for s in d["species"]], fontstyle="italic", fontsize=6.8)
    for yy, s in zip(y, d["species"]):
        ax.plot([0.495, 0.495], [yy - 0.4, yy + 0.4], color=GRUPO_COLOR[groups.get(s, "Común")], lw=2.2, clip_on=False)
    ax.set_xlabel("AUC-ROC (test, partición aleatoria)"); ax.set_xlim(0.5, 1.0)
    ax.legend(loc="lower right"); ax.grid(axis="y", visible=False)
    finish(fig, "fig_sdm_auc_especie")


def fig_confusion_audio() -> None:
    p = REP / "confusion_audio_cnn.csv"
    if not p.exists():
        return
    cm = pd.read_csv(p, index_col=0)
    rec = cm.values.diagonal() / cm.values.sum(1).clip(min=1)
    worst = [s for s, r in zip(cm.index, rec) if r < 0.8]
    # especies con las que se confunden (columnas con errores)
    cols = set(worst)
    for s in worst:
        row = cm.loc[s].drop(s)
        cols |= set(row[row > 0].sort_values(ascending=False).head(2).index)
    sel = sorted(cols)
    sub = cm.loc[sel, sel].astype(float)
    norm = sub.div(cm.loc[sel].sum(1).clip(lower=1), axis=0)
    fig, ax = plt.subplots(figsize=(TEXTW * 0.9, TEXTW * 0.8))
    im = ax.imshow(norm.values, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(sel))); ax.set_yticks(range(len(sel)))
    ax.set_xticklabels([abbreviate(s) for s in sel], rotation=90, fontstyle="italic", fontsize=7)
    ax.set_yticklabels([abbreviate(s) for s in sel], fontstyle="italic", fontsize=7)
    for i in range(len(sel)):
        for j in range(len(sel)):
            v = int(sub.values[i, j])
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=6.5, color="white" if norm.values[i, j] > 0.55 else TINTA)
    ax.set_xlabel("Especie predicha"); ax.set_ylabel("Especie verdadera"); ax.grid(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Fracción de la fila"); cb.outline.set_visible(False)
    finish(fig, "fig_confusion_audio")


def main() -> None:
    fig_corpus_especies()
    fig_llm_zero_few()
    fig_datos_por_especie()
    fig_avistamientos()
    if exists(DATA / "processed" / "chile_grid.csv", DATA / "processed" / "chile_grid_mask.csv"):
        fig_covariables()
    fig_curvas_entrenamiento()
    fig_sobreajuste()
    fig_audio_cnn_vs_ast()
    fig_f1_por_especie()
    fig_fusion()
    fig_fusion_alpha()
    fig_sdm_modelos()
    fig_sdm_satelite()
    fig_sdm_auc_especie()
    fig_confusion_audio()
    print(f"{len(saved)} figuras en tesis/figuras/: " + ", ".join(saved))


if __name__ == "__main__":
    main()
