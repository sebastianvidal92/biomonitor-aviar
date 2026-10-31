"""Tablas de la tesis generadas desde los CSV de resultados → tesis/datos/tab_*.tex.

Evita transcribir números a mano. Cada tabla se incluye en LaTeX con \\input{datos/tab_...}.
  - tab_anexoC_especies.tex : métricas por especie (F1 audio CNN, F1 imagen B3, AUC MaxEnt/RF)
  - tab_por_grupo.tex       : desempeño por grupo de conservación (audio, imagen, SDM)
  - tab_sdm_modelos.tex     : resumen de modelos SDM (partición aleatoria y, si existe, espacial)
  - tab_eval_extended.tex   : accuracy / F1 / AUC / top-5 / parámetros / latencia (si existe el script 27)
  - tab_fusion_summary.tex  : media ± d.e. del pareo repetido (si existe el script 31)
  - tab_baselines.tex       : \\textit{frozen embeddings} e IA generativa (si existen los scripts 29/30)

    uv run python scripts/34_thesis_tables.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biomonitor.config import PROJECT_ROOT  # noqa: E402
from biomonitor.evaluation import species_groups  # noqa: E402

REP = PROJECT_ROOT / "reports"
OUT = PROJECT_ROOT / "tesis" / "datos"
GROUP_ORDER = ["Amenazada", "Endémica", "Común"]


def f(x, nd=3) -> str:
    """Número con coma decimal (es-CL); vacío si NaN."""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "--"
    return f"{x:.{nd}f}".replace(".", ",")


def write(name: str, lines: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  ✓ {name}")


def tab_anexoC() -> None:
    groups = species_groups()
    a = pd.read_csv(REP / "audio_efficientnet_b0_per_class.csv").rename(columns={"Unnamed: 0": "sp"})
    img_csv = REP / "image_efficientnet_b3_tta_per_class.csv"
    i = pd.read_csv(img_csv if img_csv.exists() else REP / "image_efficientnet_b3_per_class.csv").rename(columns={"Unnamed: 0": "sp"})
    ast_csv = REP / "audio_ast_per_class.csv"
    ast = pd.read_csv(ast_csv).rename(columns={"Unnamed: 0": "sp"}) if ast_csv.exists() else None
    s = pd.read_csv(REP / "sdm_ml_auc.csv")
    af1, asup = dict(zip(a["sp"], a["f1-score"])), dict(zip(a["sp"], a["support"]))
    if1, isup = dict(zip(i["sp"], i["f1-score"])), dict(zip(i["sp"], i["support"]))
    astf1 = dict(zip(ast["sp"], ast["f1-score"])) if ast is not None else {}
    me, rf, npres = dict(zip(s["species"], s["maxent"])), dict(zip(s["species"], s["rf_base"])), dict(zip(s["species"], s["n_presence"]))
    sp = sorted(groups, key=lambda x: (GROUP_ORDER.index(groups[x]), x))
    has_ast = ast is not None
    cols = "l l r r" + (" r" if has_ast else "") + " r r r r r"
    head = ("Especie & Grupo & $n_\\mathrm{test}$ & F1 audio (CNN)" + (" & F1 audio (AST)" if has_ast else "")
            + " & $n_\\mathrm{test}$ & F1 imagen & $n_\\mathrm{pres}$ & AUC MaxEnt & AUC RF \\\\")
    lines = ["\\setlength{\\tabcolsep}{2.5pt}", "{\\scriptsize", f"\\begin{{longtable}}{{@{{}}{cols}@{{}}}}",
             "  \\caption{Métricas por especie en los conjuntos de prueba: F1 del clasificador de audio "
             "(CNN EfficientNet-B0" + (" y AST" if has_ast else "") + ", agregado por grabación), F1 del clasificador de imagen "
             "(EfficientNet-B3 con TTA) y AUC-ROC del modelo de distribución (MaxEnt y Random Forest, partición aleatoria). "
             "``--'' indica que la especie no participa de ese modelo.}",
             "  \\label{tab:anexoC}\\\\", "  \\toprule", "  " + head, "  \\midrule", "  \\endfirsthead",
             "  \\multicolumn{" + str(len(cols.split())) + "}{l}{\\footnotesize\\textit{Tabla C.1 (continuación)}}\\\\",
             "  \\toprule", "  " + head, "  \\midrule", "  \\endhead", "  \\bottomrule", "  \\endlastfoot"]
    for x in sp:
        g, sname = x.split(" ", 1)
        row = [f"\\textit{{{g[0]}. {sname}}}", groups[x], f(asup.get(x), 0), f(af1.get(x))]
        if has_ast:
            row.append(f(astf1.get(x)))
        row += [f(isup.get(x), 0) if x in isup else "--", f(if1.get(x)) if x in if1 else "--",
                f(npres.get(x), 0) if x in npres else "--", f(me.get(x)) if x in me else "--", f(rf.get(x)) if x in rf else "--"]
        lines.append("    " + " & ".join(row) + " \\\\")
    # promedios
    row = ["\\textbf{Media}", "", "", f"\\textbf{{{f(a['f1-score'].mean())}}}"]
    if has_ast:
        row.append(f"\\textbf{{{f(ast['f1-score'].mean())}}}")
    row += ["", f"\\textbf{{{f(i['f1-score'].mean())}}}", "", f"\\textbf{{{f(s['maxent'].mean())}}}", f"\\textbf{{{f(s['rf_base'].mean())}}}"]
    lines += ["  \\midrule", "    " + " & ".join(row) + " \\\\", "\\end{longtable}}"]
    write("tab_anexoC_especies.tex", lines)


def tab_por_grupo() -> None:
    groups = species_groups()
    a = pd.read_csv(REP / "audio_efficientnet_b0_per_class.csv").rename(columns={"Unnamed: 0": "sp"})
    img_csv = REP / "image_efficientnet_b3_tta_per_class.csv"
    i = pd.read_csv(img_csv if img_csv.exists() else REP / "image_efficientnet_b3_per_class.csv").rename(columns={"Unnamed: 0": "sp"})
    s = pd.read_csv(REP / "sdm_ml_auc.csv")
    fus = REP / "fusion_by_group.csv"
    fg = pd.read_csv(fus) if fus.exists() else None
    if fg is not None:
        fg = fg[fg["rama_audio"].str.startswith("CNN")]
    a["g"], i["g"], s["g"] = a["sp"].map(groups), i["sp"].map(groups), s["species"].map(groups)
    has_f = fg is not None
    lines = ["\\begin{table}[!htb]", "  \\centering\\small",
             "  \\caption{Desempeño por grupo de conservación. F1 medio por especie en el conjunto de prueba (audio: CNN, agregado por "
             "grabación; imagen: EfficientNet-B3 con TTA" + ("; fusión: pareo por especie" if has_f else "") + ") y AUC-ROC medio del SDM.}",
             "  \\label{tab:por-grupo}",
             "  \\fitwidth{\\begin{tabular}{@{}l r r r r" + (" r" if has_f else "") + " r r@{}}", "    \\toprule",
             "    Grupo & Especies & F1 audio & Especies & F1 imagen" + (" & F1 fusión" if has_f else "") + " & AUC MaxEnt & AUC RF \\\\",
             "    \\midrule"]
    for g in GROUP_ORDER:
        ag, ig, sg = a[a["g"] == g], i[i["g"] == g], s[s["g"] == g]
        row = [g, str(len(ag)), f(ag["f1-score"].mean()), str(len(ig)), f(ig["f1-score"].mean())]
        if has_f:
            row.append(f(float(fg[fg["grupo"] == g]["f1_medio"].iloc[0])) if (fg["grupo"] == g).any() else "--")
        row += [f(sg["maxent"].mean()), f(sg["rf_base"].mean())]
        lines.append("    " + " & ".join(row) + " \\\\")
    row = ["Todas", str(len(a)), f(a["f1-score"].mean()), str(len(i)), f(i["f1-score"].mean())]
    if has_f:
        row.append(f(float((fg["f1_medio"] * fg["especies"]).sum() / fg["especies"].sum())))
    row += [f(s["maxent"].mean()), f(s["rf_base"].mean())]
    lines += ["    \\midrule", "    " + " & ".join(row) + " \\\\", "    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("tab_por_grupo.tex", lines)


def tab_sdm_modelos() -> None:
    ml = pd.read_csv(REP / "sdm_ml_auc.csv"); dl = pd.read_csv(REP / "sdm_auc_por_especie.csv")
    sp_csv = REP / "sdm_spatial_cv.csv"
    sp = pd.read_csv(sp_csv) if sp_csv.exists() else None
    if sp is not None:
        sp = sp[sp["block_km"] == sp["block_km"].min()]
    n = len(ml)
    rows = [("MaxEnt (regresión logística penalizada)", ml["maxent"], sp["maxent"] if sp is not None else None),
            ("Red neuronal conjunta", dl["auc_dl"], sp["dl"] if sp is not None else None),
            ("Gradient boosting", ml["hgb_base"], sp["hgb"] if sp is not None else None),
            ("Random Forest", ml["rf_base"], sp["rf"] if sp is not None else None)]
    has_sp = sp is not None
    lines = ["\\begin{table}[!htb]", "  \\centering\\small",
             "  \\caption{Modelos de distribución comparados: AUC-ROC medio sobre las " + str(n) + " especies con al menos 20 presencias y "
             "número de especies en que cada modelo supera a MaxEnt. Partición aleatoria estratificada 70/30"
             + (f" y validación cruzada espacial por bloques de {int(sp['block_km'].iloc[0])} km ({int(sp['rf_folds'].max())} \\textit{{folds}})." if has_sp else ".") + "}",
             "  \\label{tab:sdm-modelos}", "  \\footnotesize\\setlength{\\tabcolsep}{4pt}",
             "  \\fitwidth{\\begin{tabular}{@{}l r r" + (" r r" if has_sp else "") + "@{}}", "    \\toprule",
             "    Modelo & AUC (aleatoria) & Gana a MaxEnt" + (" & AUC (espacial) & Gana a MaxEnt" if has_sp else "") + " \\\\",
             "    \\midrule"]
    for name, rnd, spa in rows:
        wins = int((rnd.values > ml["maxent"].values).sum()) if "MaxEnt" not in name else None
        row = [name, f(rnd.mean()), f"{wins}/{n}" if wins is not None else "--"]
        if has_sp:
            wins_s = int((spa.values > sp["maxent"].values).sum()) if "MaxEnt" not in name else None
            row += [f(spa.mean()), f"{wins_s}/{len(sp)}" if wins_s is not None else "--"]
        lines.append("    " + " & ".join(row) + " \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}}", "\\end{table}"]
    write("tab_sdm_modelos.tex", lines)


def tab_eval_extended() -> None:
    p = REP / "eval_extended.json"
    if not p.exists():
        return
    d = json.loads(p.read_text())
    names = {"audio_cnn": "Audio: CNN (agregado)", "audio_ast": "Audio: AST (agregado)", "image_b3": "Imagen: B3 (TTA)"}
    lines = ["\\begin{table}[!htb]", "  \\centering\\small",
             "  \\caption{Métricas extendidas de los clasificadores unimodales (solo audio o solo imagen) en el conjunto de prueba (protocolo operacional). "
             "AUC-ROC macro (cada especie frente a las demás); top-5: la especie correcta está entre las cinco más probables; latencia mediana por muestra "
             "en Apple M4 Pro (MPS), incluido el cálculo de características.}",
             "  \\label{tab:eval-extended}", "  \\footnotesize\\setlength{\\tabcolsep}{4pt}", "  \\begin{tabular}{@{}l r r r r r r@{}}", "    \\toprule",
             "    Modelo & \\textit{Accuracy} & F1 macro & AUC macro & Top-5 & Parámetros & Latencia (ms) \\\\", "    \\midrule"]
    for k, name in names.items():
        if k in d:
            s = d[k]
            lines.append(f"    {name} & {f(s['accuracy'])} & {f(s['f1_macro'])} & {f(s['auc_macro'])} & {f(s['top5'])} & "
                         f"{s['n_params'] / 1e6:.1f} M".replace(".", ",") + f" & {s['latency_ms_median']:.0f} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("tab_eval_extended.tex", lines)


def tab_fusion_summary() -> None:
    p = REP / "fusion_summary.csv"
    if not p.exists():
        return
    d = pd.read_csv(p)
    lines = ["\\begin{table}[!htb]", "  \\centering\\small",
             "  \\caption{\\textit{Late fusion} evaluada con pareo por especie repetido: media y desviación estándar sobre "
             f"{int(d['repeticiones'].iloc[0])} repeticiones de {int(d['pares'].iloc[0])} pares ({int(d['especies'].iloc[0])} especies), "
             "para las dos ramas de audio disponibles.}",
             "  \\label{tab:fusion-summary}", "  \\footnotesize", "  \\begin{tabular}{@{}l l r r@{}}", "    \\toprule",
             "    Rama de audio & Modalidad & \\textit{Accuracy} & F1 macro \\\\", "    \\midrule"]
    for r in d.itertuples(index=False):
        modal = r.modalidad.replace("Fusión tardía", "\\textit{Late fusion}").replace("α=0.5", "$\\alpha=0{,}5$").replace("α=0,5", "$\\alpha=0{,}5$").replace("α", "$\\alpha$")
        lines.append(f"    {r.rama_audio} & {modal} & {f(r.accuracy_media)} $\\pm$ {f(r.accuracy_sd)} & "
                     f"{f(r.f1_media)} $\\pm$ {f(r.f1_sd)} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    write("tab_fusion_summary.tex", lines)


def tab_baselines() -> None:
    fz, llm = REP / "frozen_baseline.csv", REP / "llm_baseline.csv"
    if not fz.exists() and not llm.exists():
        return
    lines = ["\\begin{table}[!htb]", "  \\centering\\small",
             "  \\caption{\\textit{Baselines} de referencia frente a los modelos ajustados (conjuntos de prueba). \\textit{Frozen embeddings}: "
             "regresión logística sobre las representaciones de un modelo preentrenado sin ajustar. IA generativa: modelo multimodal de "
             "propósito general consultado con la fotografía y la lista cerrada de especies.}",
             "  \\label{tab:baselines}", "  \\footnotesize", "  \\begin{tabularx}{\\textwidth}{@{}l >{\\raggedright\\arraybackslash}X r r r@{}}", "    \\toprule",
             "    Modalidad & Método & \\textit{Accuracy} & F1 macro & AUC macro \\\\", "    \\midrule"]
    if fz.exists():
        for r in pd.read_csv(fz).itertuples(index=False):
            emb = r.embedding.replace(", congelado", "")
            lines.append(f"    {r.modalidad.capitalize()} & \\textit{{Frozen embeddings}} de {emb} + regresión logística & "
                         f"{f(r.accuracy)} & {f(r.f1_macro)} & {f(r.auc_macro)} \\\\")
    if llm.exists():
        for r in pd.read_csv(llm).itertuples(index=False):
            modo = "\\textit{zero-shot}" if r.modo == "zeroshot" else "\\textit{few-shot}"
            lines.append(f"    Imagen & IA generativa ({r.modelo}, {modo}, esfuerzo \\textit{{{r.esfuerzo}}}) & {f(r.accuracy)} & {f(r.f1_macro)} & -- \\\\")
    ext = REP / "eval_extended.json"
    e = json.loads(ext.read_text()) if ext.exists() else {}
    def fitted(key, default):
        d = e.get(key); return (f(d["accuracy"]), f(d["f1_macro"]), f(d["auc_macro"])) if d else default
    cnn, ast, b3 = fitted("audio_cnn", ("0,858", "0,858", "--")), fitted("audio_ast", ("0,870", "0,864", "--")), fitted("image_b3", ("0,848", "0,808", "--"))
    lines += ["    \\midrule",
              f"    Audio & \\textbf{{CNN EfficientNet-B0 ajustada (este trabajo)}} & {cnn[0]} & {cnn[1]} & {cnn[2]} \\\\",
              f"    Audio & \\textbf{{AST ajustado (este trabajo)}} & {ast[0]} & {ast[1]} & {ast[2]} \\\\",
              f"    Imagen & \\textbf{{EfficientNet-B3 ajustada + TTA (este trabajo)}} & {b3[0]} & {b3[1]} & {b3[2]} \\\\",
              "    \\bottomrule", "  \\end{tabularx}", "\\end{table}"]
    write("tab_baselines.tex", lines)


def main() -> None:
    print("Tablas → tesis/datos/")
    tab_anexoC(); tab_por_grupo(); tab_sdm_modelos(); tab_eval_extended(); tab_fusion_summary(); tab_baselines()


if __name__ == "__main__":
    main()
