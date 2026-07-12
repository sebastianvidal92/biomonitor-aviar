"""Figuras del informe — todo el proyecto en gráficos (identidad visual UC).

Genera en `reports/figures/*.png`:
  Curvas de entrenamiento (val_acc / val_f1 / loss por época)
    01 audio CNN · 02 audio AST · 03 imagen B3
  Comparaciones
    04 audio (single vs agregado) · 05 multimodal · 06 estrategias de fusión
    07 SDM (modelos) · 08 SDM (¿ayuda el satélite?) · 09 SDM temporal (pareo)
  Datos y desempeño por especie
    10 audio/especie · 11 fotos/especie · 12 avistamientos (año/mes)
    13 F1 por especie (audio) · 14 F1 por especie (imagen) · 15 AUC por especie (SDM)

Datos por época = épocas clave registradas en docs/MODULO_AUDIO.md y MODULO_IMAGEN.md
(el entrenamiento las imprimía a stdout; ahora los scripts 07/09/12 además las guardan en
reports/training_logs/*.csv — si esos CSV existen, se usan en vez de los valores embebidos).

    uv run python scripts/26_report_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

mpl.use("Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biomonitor.config import PROJECT_ROOT  # noqa: E402

# --- Identidad UC ---
UC = dict(celeste="#0176DE", azul="#173F8A", azul_osc="#03122E", amarillo="#FEC60D",
          verde="#0DA400", rojo="#F24F4F", gris="#707070", gris_cl="#F6F6F6", borde="#E3E8EF")
mpl.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 13,
    "axes.titleweight": "bold", "axes.labelsize": 11, "axes.edgecolor": UC["gris"],
    "axes.linewidth": 0.8, "axes.grid": True, "grid.color": UC["borde"],
    "grid.linewidth": 0.8, "figure.dpi": 130, "savefig.dpi": 150,
    "savefig.bbox": "tight", "axes.titlecolor": UC["azul_osc"],
})

FIG = PROJECT_ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
LOGS = PROJECT_ROOT / "reports" / "training_logs"
DATA = PROJECT_ROOT / "data" / "interim"
REP = PROJECT_ROOT / "reports"
saved: list[str] = []


def finish(fig, name: str, title: str) -> None:
    fig.suptitle(title, color=UC["azul_osc"], fontweight="bold", fontsize=14, y=1.0)
    for ax in fig.axes:
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_axisbelow(True)
    fig.tight_layout()
    p = FIG / f"{name}.png"
    fig.savefig(p)
    plt.close(fig)
    saved.append(name)


def barlabels(ax, bars, fmt="{:.3f}", dy=0.004, color=None):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + dy, fmt.format(b.get_height()),
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
                color=color or UC["azul_osc"])


# ============================ Curvas de entrenamiento ============================
# Épocas clave (docs). Si existe el CSV de historial completo, se prefiere.
CURVES = {
    "audio_cnn": dict(
        ep=[1, 5, 10, 13, 19, 23, 38, 50, 60, 67, 81, 90, 100],
        tr_loss=[4.043, 2.621, 2.248, 2.080, 1.984, 1.827, 1.691, 1.591, 1.399, 1.411, 1.357, 1.233, 1.241],
        val_loss=[3.514, 2.246, 1.861, 1.763, 1.673, 1.583, 1.573, 1.511, 1.485, 1.524, 1.499, 1.472, 1.473],
        val_acc=[0.175, 0.558, 0.680, 0.723, 0.749, 0.777, 0.794, 0.812, 0.815, 0.815, 0.815, 0.812, 0.812],
        val_f1=[0.147, 0.539, 0.675, 0.721, 0.748, 0.775, 0.793, 0.809, 0.810, 0.811, 0.811, 0.809, 0.808],
        best=67),
    "audio_ast": dict(
        ep=list(range(1, 21)),
        tr_loss=[2.844, 1.480, 1.112, 0.983, 0.927, 0.893, 0.867, 0.841, 0.843, 0.814,
                 0.801, 0.802, 0.788, 0.788, 0.791, 0.777, 0.776, 0.777, 0.775, 0.774],
        val_acc=[0.675, 0.779, 0.802, 0.805, 0.812, 0.827, 0.827, 0.843, 0.850, 0.838,
                 0.860, 0.850, 0.840, 0.850, 0.840, 0.848, 0.845, 0.843, 0.840, 0.843],
        val_f1=[0.668, 0.781, 0.799, 0.802, 0.819, 0.829, 0.822, 0.846, 0.850, 0.842,
                0.860, 0.851, 0.842, 0.853, 0.843, 0.850, 0.843, 0.842, 0.839, 0.841],
        best=11),
    "imagen_b3": dict(
        ep=[1, 5, 6, 13, 18, 23, 26, 27, 31, 32, 36, 38, 48, 50],
        tr_loss=[3.218, 1.699, 1.697, 1.497, 1.465, 1.465, 1.425, 1.418, 1.389, 1.367, 1.332, 1.401, 1.292, 1.360],
        val_acc=[0.660, 0.794, 0.807, 0.821, 0.827, 0.828, 0.825, 0.835, 0.835, 0.837, 0.842, 0.855, 0.857, 0.852],
        val_f1=[0.626, 0.761, 0.779, 0.784, 0.792, 0.800, 0.800, 0.801, 0.804, 0.805, 0.814, 0.828, 0.824, 0.823],
        best=38),
}


def get_curve(key: str) -> dict:
    csv = LOGS / f"{key}.csv"
    if csv.exists():
        d = pd.read_csv(csv)
        c = {k: d[k].tolist() for k in d.columns if k in ("ep", "tr_loss", "val_loss", "val_acc", "val_f1")}
        c["best"] = int(d.loc[d["val_f1"].idxmax(), "ep"]) if "val_f1" in d else CURVES[key]["best"]
        return c
    return CURVES[key]


def plot_curve(key: str, title: str) -> None:
    c = get_curve(key)
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    ax.plot(c["ep"], c["val_acc"], "-o", color=UC["celeste"], lw=2.2, ms=5, label="Val accuracy")
    ax.plot(c["ep"], c["val_f1"], "-o", color=UC["azul"], lw=2.2, ms=5, label="Val F1 macro")
    ax.axvline(c["best"], color=UC["verde"], ls="--", lw=1.4, alpha=0.8)
    ax.text(c["best"], 0.965, f" mejor época ({c['best']})", color=UC["verde"],
            fontsize=9, fontweight="bold", transform=ax.get_xaxis_transform(), va="top")
    ax.set_xlabel("Época")
    ax.set_ylabel("Métrica de validación")
    ax.set_ylim(0, 1)

    ax2 = ax.twinx()
    ax2.plot(c["ep"], c["tr_loss"], "-", color=UC["amarillo"], lw=1.8, alpha=0.9, label="Train loss")
    if "val_loss" in c:
        ax2.plot(c["ep"], c["val_loss"], "--", color=UC["gris"], lw=1.6, alpha=0.9, label="Val loss")
    ax2.set_ylabel("Pérdida (loss)", color=UC["gris"])
    ax2.grid(False)
    ax2.spines["top"].set_visible(False)

    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=9, framealpha=0.95)
    finish(fig, f"0{list(CURVES).index(key) + 1}_curva_{key}", title)


# =============================== Barras comparativas ===============================
def fig_audio_compare() -> None:
    models = ["CNN\n(EfficientNet-B0)", "AST\n(transformer)"]
    single = [0.812, 0.832]
    agg = [0.858, 0.868]
    x = np.arange(len(models))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - w / 2, single, w, color=UC["celeste"], label="Ventana individual (5 s)")
    b2 = ax.bar(x + w / 2, agg, w, color=UC["azul"], label="Agregado por grabación")
    barlabels(ax, b1)
    barlabels(ax, b2)
    ax.axhline(0.85, color=UC["rojo"], ls="--", lw=1.4)
    ax.text(1.45, 0.852, "meta 0.85", color=UC["rojo"], fontsize=9, ha="right", va="bottom")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylabel("Accuracy (test)")
    ax.set_ylim(0.7, 0.92)
    ax.legend(fontsize=9, loc="lower right")
    finish(fig, "04_comparacion_audio", "Audio — CNN vs. AST · efecto de la agregación por grabación")


def fig_multimodal() -> None:
    labels = ["Audio\nsolo", "Imagen\nsola", "Fusión\nmultimodal"]
    acc = [0.867, 0.825, 0.950]
    cols = [UC["celeste"], UC["amarillo"], UC["azul"]]
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    bars = ax.bar(labels, acc, color=cols, width=0.62)
    barlabels(ax, bars)
    ax.axhline(0.85, color=UC["rojo"], ls="--", lw=1.4)
    ax.text(2.4, 0.852, "meta 0.85", color=UC["rojo"], fontsize=9, ha="right", va="bottom")
    ax.set_ylabel("Accuracy (pares por especie)")
    ax.set_ylim(0.7, 1.0)
    finish(fig, "05_multimodal", "Módulo 1 — La fusión multimodal supera a cada modalidad")


def fig_fusion_strategies() -> None:
    d = pd.read_csv(REP / "fusion_strategies.csv")
    short = ["Late\n(decisión)", "Early\n(concatenación)", "Intermediate\n(proyección)"]
    x = np.arange(len(d))
    w = 0.36
    fig, ax = plt.subplots(figsize=(7, 4.6))
    b1 = ax.bar(x - w / 2, d["accuracy"], w, color=UC["azul"], label="Accuracy")
    b2 = ax.bar(x + w / 2, d["f1_macro"], w, color=UC["celeste"], label="F1 macro")
    barlabels(ax, b1)
    barlabels(ax, b2)
    ax.set_xticks(x)
    ax.set_xticklabels(short[:len(d)])
    ax.set_ylabel("Métrica (pares por especie)")
    ax.set_ylim(0.85, 0.95)
    ax.legend(fontsize=9, loc="lower left")
    finish(fig, "06_estrategias_fusion", "Módulo 1 — Estrategias de fusión (mismas condiciones)")


def fig_sdm_models() -> None:
    ml = pd.read_csv(REP / "sdm_ml_auc.csv")
    dl = pd.read_csv(REP / "sdm_auc_por_especie.csv")
    labels = ["MaxEnt\n(clásico)", "Red neuronal\n(DL conjunto)", "Gradient\nBoosting", "Random\nForest"]
    vals = [ml["maxent"].mean(), dl["auc_dl"].mean(), ml["hgb_base"].mean(), ml["rf_base"].mean()]
    cols = [UC["gris"], UC["amarillo"], UC["celeste"], UC["azul"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    bars = ax.bar(labels, vals, color=cols, width=0.64)
    barlabels(ax, bars)
    ax.axhline(vals[0], color=UC["gris"], ls=":", lw=1.3)
    ax.set_ylabel("AUC-ROC media (54 especies)")
    ax.set_ylim(0.80, 0.90)
    finish(fig, "07_sdm_modelos", "Módulo 2 — Los árboles superan a MaxEnt; el DL no")


def fig_sdm_satellite() -> None:
    ml = pd.read_csv(REP / "sdm_ml_auc.csv")
    labels = ["RF base\n(sin satélite)", "RF + embedding\n(satélite)",
              "DL + embedding\n(satélite)", "DL + parche\ncobertura suelo"]
    vals = [ml["rf_base"].mean(), ml["rf_emb"].mean(), 0.767, 0.613]
    cols = [UC["azul"], UC["celeste"], UC["amarillo"], UC["rojo"]]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    bars = ax.bar(labels, vals, color=cols, width=0.64)
    barlabels(ax, bars)
    ax.axhline(vals[0], color=UC["azul"], ls=":", lw=1.3)
    ax.set_ylabel("AUC-ROC media")
    ax.set_ylim(0.55, 0.92)
    finish(fig, "08_sdm_satelite", "Módulo 2 — Agregar imágenes satelitales NO mejora el SDM")


def fig_sdm_temporal() -> None:
    d = pd.read_csv(REP / "sdm_temporal_auc.csv")
    labels = ["MaxEnt\n(base)", "RF base\n(sin satélite)", "RF + satélite\nFIJO (2023)",
              "RF + satélite\nPAREADO (año obs.)"]
    vals = [d["maxent"].mean(), d["rf_base"].mean(), d["rf_fijo"].mean(), d["rf_pareado"].mean()]
    cols = [UC["gris"], UC["azul"], UC["amarillo"], UC["celeste"]]
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    bars = ax.bar(labels, vals, color=cols, width=0.64)
    barlabels(ax, bars)
    ax.axhline(vals[1], color=UC["azul"], ls=":", lw=1.3)
    ax.set_ylabel("AUC-ROC media (53 especies, 2017-2025)")
    ax.set_ylim(0.84, 0.90)
    finish(fig, "09_sdm_temporal", "Módulo 2 — Parear el satélite al año de la observación")


# ============================ Datos y desempeño por especie ============================
def _counts_fig(counts: pd.Series, name: str, title: str, xlabel: str, color: str) -> None:
    counts = counts.sort_values(ascending=True)
    fig, ax = plt.subplots(figsize=(8.4, max(6, len(counts) * 0.22)))
    ax.barh(range(len(counts)), counts.values, color=color, height=0.72)
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts.index, fontsize=7.2, fontstyle="italic")
    ax.set_xlabel(xlabel)
    ax.margins(y=0.005)
    med = counts.median()
    ax.axvline(med, color=UC["gris"], ls="--", lw=1.2)
    ax.text(med, len(counts) - 1, f" mediana {med:.0f}", color=UC["gris"], fontsize=8, va="top")
    finish(fig, name, title)


def fig_data_audio() -> None:
    d = pd.read_csv(DATA / "audio_manifest_split.csv")
    _counts_fig(d["scientific_name"].value_counts(), "10_datos_audio_especie",
                f"Datos de audio — grabaciones por especie (n={len(d)}, Xeno-canto)",
                "Grabaciones", UC["celeste"])


def fig_data_image() -> None:
    d = pd.read_csv(DATA / "image_manifest_split.csv")
    _counts_fig(d["scientific_name"].value_counts(), "11_datos_imagen_especie",
                f"Datos de imagen — fotos por especie (n={len(d)}, iNaturalist RG)",
                "Fotos", UC["azul"])


def fig_occurrences() -> None:
    d = pd.read_csv(DATA / "occurrences.csv")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    yr = d[d["year"].between(2000, 2025)]["year"].value_counts().sort_index()
    a1.bar(yr.index, yr.values, color=UC["azul"], width=0.8)
    a1.set_xlabel("Año")
    a1.set_ylabel("Avistamientos")
    a1.set_title("Distribución temporal (por año)", fontsize=11)
    mo = d["month"].dropna().astype(int).value_counts().reindex(range(1, 13), fill_value=0)
    meses = ["E", "F", "M", "A", "M", "J", "J", "A", "S", "O", "N", "D"]
    a2.bar(range(1, 13), mo.values, color=UC["celeste"], width=0.72)
    a2.set_xticks(range(1, 13))
    a2.set_xticklabels(meses)
    a2.set_xlabel("Mes")
    a2.set_ylabel("Avistamientos")
    a2.set_title("Estacionalidad (por mes)", fontsize=11)
    finish(fig, "12_avistamientos", f"Módulo 2 — Avistamientos GBIF (n={len(d):,}, eBird+iNaturalist)")


def _f1_fig(csv: str, name: str, title: str) -> None:
    d = pd.read_csv(REP / csv).rename(columns={"Unnamed: 0": "sp", "f1-score": "f1"})
    d = d[d["sp"] != "accuracy"].dropna(subset=["f1"]).sort_values("f1")
    cols = [UC["rojo"] if v < 0.6 else UC["amarillo"] if v < 0.8 else UC["celeste"] for v in d["f1"]]
    fig, ax = plt.subplots(figsize=(8.4, max(6, len(d) * 0.22)))
    ax.barh(range(len(d)), d["f1"], color=cols, height=0.72)
    ax.set_yticks(range(len(d)))
    ax.set_yticklabels(d["sp"], fontsize=7.2, fontstyle="italic")
    ax.set_xlabel("F1-score (test)")
    ax.set_xlim(0, 1)
    ax.axvline(d["f1"].mean(), color=UC["azul_osc"], ls="--", lw=1.3)
    ax.text(d["f1"].mean(), 0, f" media {d['f1'].mean():.3f}", color=UC["azul_osc"], fontsize=8, va="bottom")
    finish(fig, name, title)


def fig_sdm_species() -> None:
    d = pd.read_csv(REP / "sdm_ml_auc.csv").sort_values("rf_base")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8.4, max(6, len(d) * 0.22)))
    ax.hlines(y, d["maxent"], d["rf_base"], color=UC["borde"], lw=2, zorder=1)
    ax.scatter(d["maxent"], y, color=UC["gris"], s=22, label="MaxEnt", zorder=2)
    ax.scatter(d["rf_base"], y, color=UC["azul"], s=26, label="Random Forest", zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(d["species"], fontsize=7.2, fontstyle="italic")
    ax.set_xlabel("AUC-ROC (test)")
    ax.set_xlim(0.5, 1.0)
    ax.legend(fontsize=9, loc="lower left")
    finish(fig, "15_sdm_auc_especie", "Módulo 2 — AUC por especie: Random Forest vs. MaxEnt")


def main() -> None:
    plot_curve("audio_cnn", "Audio — Curva de entrenamiento CNN (EfficientNet-B0 regularizado)")
    plot_curve("audio_ast", "Audio — Curva de entrenamiento AST (transformer preentrenado)")
    plot_curve("imagen_b3", "Imagen — Curva de entrenamiento EfficientNet-B3")
    fig_audio_compare()
    fig_multimodal()
    fig_fusion_strategies()
    fig_sdm_models()
    fig_sdm_satellite()
    fig_sdm_temporal()
    fig_data_audio()
    fig_data_image()
    fig_occurrences()
    _f1_fig("audio_efficientnet_b0_per_class.csv", "13_f1_especie_audio",
            "Audio — F1 por especie (CNN EfficientNet-B0)")
    _f1_fig("image_efficientnet_b3_per_class.csv", "14_f1_especie_imagen",
            "Imagen — F1 por especie (EfficientNet-B3)")
    fig_sdm_species()
    print(f"\n{len(saved)} figuras en reports/figures/:")
    for n in saved:
        print(f"  ✓ {n}.png")


if __name__ == "__main__":
    main()
