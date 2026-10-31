"""Intervalos de confianza y tests pareados para los resultados de la tesis.

Clasificadores (accuracy y F1 macro): intervalo bootstrap percentil al 95 % sobre las
muestras de prueba (2.000 remuestreos), a partir de las probabilidades guardadas por el
script 27 (reports/probs/*.npz) y de las respuestas del modelo generativo (reports/llm/).
Fusión: bootstrap sobre las grabaciones y fotografías de prueba (300 remuestreos), con el
pareo por especie recalculado en cada remuestreo. Baselines congelados: intervalo normal.
SDM: intervalo bootstrap de la AUC media sobre las especies y test de Wilcoxon de rangos
con signo (pareado por especie) para cada comparación entre modelos.

    uv run python scripts/35_confidence_intervals.py

Salidas: reports/stats_ci.csv, reports/stats_tests_sdm.csv, tesis/datos/tab_ic.tex,
tesis/datos/tab_tests_sdm.tex.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from biomonitor.config import PROJECT_ROOT  # noqa: E402

REP = PROJECT_ROOT / "reports"
OUT = PROJECT_ROOT / "tesis" / "datos"
B, B_FUS, SEED = 2000, 300, 42
rng = np.random.default_rng(SEED)


def f(x, nd=3) -> str:
    return f"{x:.{nd}f}".replace(".", ",")


def boot_ci(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    n = len(y_true)
    acc = np.empty(B); f1 = np.empty(B)
    for b in range(B):
        idx = rng.integers(0, n, n)
        acc[b] = accuracy_score(y_true[idx], y_pred[idx])
        f1[b] = f1_score(y_true[idx], y_pred[idx], average="macro", labels=range(n_classes), zero_division=0)
    return {"n": n, "accuracy": accuracy_score(y_true, y_pred), "acc_lo": np.percentile(acc, 2.5), "acc_hi": np.percentile(acc, 97.5),
            "f1": f1_score(y_true, y_pred, average="macro", labels=range(n_classes), zero_division=0),
            "f1_lo": np.percentile(f1, 2.5), "f1_hi": np.percentile(f1, 97.5)}


def normal_ci(p: float, n: int) -> tuple[float, float]:
    h = 1.96 * np.sqrt(p * (1 - p) / n)
    return p - h, p + h


def load_npz(name: str) -> dict:
    z = np.load(REP / "probs" / name, allow_pickle=True)
    return {k: z[k] for k in z.files}


def fusion_boot(audio: dict, image: dict, pairs: int = 30) -> dict:
    """Bootstrap sobre grabaciones y fotos de prueba; pareo por especie en cada remuestreo."""
    common = sorted(set(image["classes"]) & set(audio["classes"]))
    K = len(common)
    ai = {c: i for i, c in enumerate(audio["classes"])}; ii = {c: i for i, c in enumerate(image["classes"])}
    Pa = audio["probs_agg"][:, [ai[c] for c in common]]; Pi = image["probs_agg"][:, [ii[c] for c in common]]
    ya_all = np.array([c for c in audio["classes"][audio["y_true"]]]); keep = np.isin(ya_all, common)
    Pa, ya = Pa[keep], np.array([common.index(c) for c in ya_all[keep]])
    yi = np.array([common.index(c) for c in image["classes"][image["y_true"]]])

    def metrics(idx_a, idx_i, r):
        acc = {}
        ia = np.concatenate([r.choice(idx_a[ya[idx_a] == k], pairs) for k in range(K)])
        ii_ = np.concatenate([r.choice(idx_i[yi[idx_i] == k], pairs) for k in range(K)])
        y = np.repeat(np.arange(K), pairs); pa, pi = Pa[ia], Pi[ii_]
        for name, p in [("audio", pa), ("imagen", pi), ("fusion", 0.5 * pa + 0.5 * pi),
                        ("fusion_geo", 0.5 * np.log(pa + 1e-9) + 0.5 * np.log(pi + 1e-9))]:
            acc[name] = accuracy_score(y, p.argmax(1))
        return acc

    out = {k: [] for k in ["audio", "imagen", "fusion", "fusion_geo"]}
    for b in range(B_FUS):
        r = np.random.default_rng(1000 + b)
        # remuestreo estratificado por especie (garantiza al menos una muestra por clase)
        idx_a = np.concatenate([r.choice(np.flatnonzero(ya == k), (ya == k).sum()) for k in range(K)])
        idx_i = np.concatenate([r.choice(np.flatnonzero(yi == k), (yi == k).sum()) for k in range(K)])
        m = metrics(idx_a, idx_i, r)
        for k in out:
            out[k].append(m[k])
    return {k: (float(np.mean(v)), float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in out.items()}


def main() -> None:
    rows = []
    # --- clasificadores de modalidad única ---
    for name, file, label in [("audio_cnn", "audio_efficientnet_b0_test.npz", "Audio: CNN EfficientNet-B0 (agregado)"),
                              ("audio_ast", "audio_ast_test.npz", "Audio: AST (agregado)"),
                              ("image_b3", "image_efficientnet_b3_test.npz", "Imagen: EfficientNet-B3 (TTA)")]:
        d = load_npz(file)
        r = boot_ci(d["y_true"], d["probs_agg"].argmax(1), len(d["classes"]))
        rows.append({"modelo": label, "metodo": "bootstrap", **r})
        print(f"{label}: acc {r['accuracy']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]  F1 {r['f1']:.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]")

    # --- modelo generativo ---
    i_cls = sorted(pd.read_csv(PROJECT_ROOT / "data/interim/image_manifest_split.csv")["scientific_name"].unique())
    for mode, label in [("zeroshot", "IA generativa \\textit{zero-shot}"), ("fewshot", "IA generativa \\textit{few-shot}")]:
        p = REP / "llm" / f"{mode}_claude-opus-5-5.jsonl"
        if not p.exists():
            continue
        recs = {}
        for line in p.read_text().splitlines():
            r = json.loads(line)
            if (r.get("error") or "").startswith("conexion"):
                continue  # las demás filas con error cuentan como predicción incorrecta, como en la tesis
            recs[r["gbif_id"]] = r
        y = np.array([i_cls.index(r["true"]) for r in recs.values()])
        pred = np.array([i_cls.index(r["pred"]) if r.get("pred") in i_cls else (i_cls.index(r["true"]) + 1) % len(i_cls) for r in recs.values()])
        r = boot_ci(y, pred, len(i_cls)); rows.append({"modelo": label, "metodo": "bootstrap", **r})
        print(f"{label}: acc {r['accuracy']:.3f} [{r['acc_lo']:.3f}, {r['acc_hi']:.3f}]  F1 {r['f1']:.3f} [{r['f1_lo']:.3f}, {r['f1_hi']:.3f}]")

    # --- baselines congelados (solo métricas agregadas → intervalo normal) ---
    fz = REP / "frozen_baseline.csv"
    if fz.exists():
        for r in pd.read_csv(fz).itertuples(index=False):
            lo, hi = normal_ci(r.accuracy, r.n_test)
            rows.append({"modelo": f"\\textit{{Frozen embeddings}}: {r.embedding.replace(', congelado', '')}", "metodo": "normal",
                         "n": r.n_test, "accuracy": r.accuracy, "acc_lo": lo, "acc_hi": hi, "f1": r.f1_macro, "f1_lo": np.nan, "f1_hi": np.nan})

    # --- fusión ---
    fus = fusion_boot(load_npz("audio_efficientnet_b0_test.npz"), load_npz("image_efficientnet_b3_test.npz"))
    for k, label in [("audio", "Pareo: audio solo"), ("imagen", "Pareo: imagen sola"), ("fusion", "Pareo: \\textit{late fusion} ($\\alpha = 0{,}5$)"),
                     ("fusion_geo", "Pareo: \\textit{late fusion} geométrica")]:
        m, lo, hi = fus[k]
        rows.append({"modelo": label, "metodo": "bootstrap pares", "n": 1590, "accuracy": m, "acc_lo": lo, "acc_hi": hi,
                     "f1": np.nan, "f1_lo": np.nan, "f1_hi": np.nan})
        print(f"{label}: acc {m:.3f} [{lo:.3f}, {hi:.3f}]")
    ci = pd.DataFrame(rows); ci.to_csv(REP / "stats_ci.csv", index=False)

    # --- SDM: bootstrap de la media y Wilcoxon pareado por especie ---
    ml = pd.read_csv(REP / "sdm_ml_auc.csv"); dl = pd.read_csv(REP / "sdm_auc_por_especie.csv")
    ml = ml.merge(dl[["species", "auc_dl"]], on="species")
    comps = [("aleatoria", ml, "rf_base", "maxent", "Random Forest vs. MaxEnt"),
             ("aleatoria", ml, "hgb_base", "maxent", "Gradient boosting vs. MaxEnt"),
             ("aleatoria", ml, "auc_dl", "maxent", "Red neuronal vs. MaxEnt"),
             ("aleatoria", ml, "rf_emb", "rf_base", "RF + \\textit{embedding} satelital vs. RF")]
    sp = REP / "sdm_spatial_cv.csv"
    if sp.exists():
        s = pd.read_csv(sp); s = s[s["block_km"] == s["block_km"].min()].dropna(subset=["rf", "maxent"])
        comps += [("espacial", s, "rf", "maxent", "Random Forest vs. MaxEnt"),
                  ("espacial", s, "hgb", "maxent", "Gradient boosting vs. MaxEnt"),
                  ("espacial", s, "dl", "maxent", "Red neuronal vs. MaxEnt")]
    t = REP / "sdm_temporal_auc.csv"
    if t.exists():
        tt = pd.read_csv(t)
        comps += [("temporal 2017-2025", tt, "rf_base", "maxent", "RF vs. MaxEnt"),
                  ("temporal 2017-2025", tt, "rf_fijo", "rf_base", "RF + satélite fijo vs. RF"),
                  ("temporal 2017-2025", tt, "rf_pareado", "rf_base", "RF + satélite pareado vs. RF"),
                  ("temporal 2017-2025", tt, "rf_pareado", "rf_fijo", "RF + satélite pareado vs. fijo")]
    trows = []
    for proto, df, a, b, label in comps:
        x, y = df[a].values, df[b].values; d = x - y
        boots = [np.mean(d[rng.integers(0, len(d), len(d))]) for _ in range(B)]
        w = stats.wilcoxon(x, y, alternative="two-sided") if np.any(d != 0) else None
        trows.append({"protocolo": proto, "comparacion": label, "n_especies": len(d), "auc_a": x.mean(), "auc_b": y.mean(),
                      "dif_media": d.mean(), "dif_lo": np.percentile(boots, 2.5), "dif_hi": np.percentile(boots, 97.5),
                      "gana_a": int((d > 0).sum()), "p_wilcoxon": w.pvalue if w else np.nan})
        print(f"[{proto}] {label}: dif {d.mean():+.3f} [{np.percentile(boots, 2.5):+.3f}, {np.percentile(boots, 97.5):+.3f}]  "
              f"gana {int((d > 0).sum())}/{len(d)}  p = {w.pvalue if w else float('nan'):.3g}")
    tests = pd.DataFrame(trows); tests.to_csv(REP / "stats_tests_sdm.csv", index=False)

    # --- tablas LaTeX ---
    OUT.mkdir(parents=True, exist_ok=True)
    lines = ["\\begin{table}[!htb]", "  \\centering\\footnotesize",
             "  \\caption{Intervalos de confianza al \\SI{95}{\\percent} de las métricas principales de identificación. "
             "\\textit{Bootstrap} percentil con \\num{2000} remuestreos de las muestras de prueba (fusión: 300 remuestreos estratificados "
             "por especie de grabaciones y fotografías, con el pareo recalculado); los \\textit{baselines} de \\textit{frozen embeddings} "
             "usan el intervalo normal de una proporción.}",
             "  \\label{tab:ic}", "  \\begin{tabular}{@{}l r r r@{}}", "    \\toprule",
             "    Modelo & $n$ & \\textit{Accuracy} [IC 95\\,\\%] & F1 macro [IC 95\\,\\%] \\\\", "    \\midrule"]
    for r in ci.itertuples(index=False):
        f1s = f"{f(r.f1)} [{f(r.f1_lo)}; {f(r.f1_hi)}]" if not np.isnan(r.f1_lo) else (f(r.f1) if not np.isnan(r.f1) else "--")
        lines.append(f"    {r.modelo} & {int(r.n)} & {f(r.accuracy)} [{f(r.acc_lo)}; {f(r.acc_hi)}] & {f1s} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (OUT / "tab_ic.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")

    lines = ["\\begin{table}[!htb]", "  \\centering\\footnotesize",
             "  \\caption{Comparaciones pareadas por especie entre modelos de distribución: diferencia media de AUC-ROC con "
             "intervalo \\textit{bootstrap} al \\SI{95}{\\percent}, número de especies en que el primer modelo supera al segundo y valor $p$ "
             "del test de Wilcoxon de rangos con signo (bilateral).}",
             "  \\label{tab:tests-sdm}", "  \\begin{tabular}{@{}l l r r r r@{}}", "    \\toprule",
             "    Protocolo & Comparación & $n$ & $\\Delta$AUC [IC 95\\,\\%] & Gana & $p$ \\\\", "    \\midrule"]
    for r in tests.itertuples(index=False):
        pv = "$<$0,001" if r.p_wilcoxon < 0.001 else f(r.p_wilcoxon, 3)
        lines.append(f"    {r.protocolo} & {r.comparacion} & {r.n_especies} & {f(r.dif_media):>6} [{f(r.dif_lo)}; {f(r.dif_hi)}] & "
                     f"{r.gana_a}/{r.n_especies} & {pv} \\\\")
    lines += ["    \\bottomrule", "  \\end{tabular}", "\\end{table}"]
    (OUT / "tab_tests_sdm.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n  → reports/stats_ci.csv, reports/stats_tests_sdm.csv, tesis/datos/tab_ic.tex, tesis/datos/tab_tests_sdm.tex")


if __name__ == "__main__":
    main()
