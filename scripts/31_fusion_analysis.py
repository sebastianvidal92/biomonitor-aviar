"""Análisis extendido de la fusión multimodal (Módulo 1) a partir de probabilidades guardadas.

Usa las probabilidades de test que guarda el script 27 (reports/probs/*.npz), por lo que
NO vuelve a correr los modelos. Para cada rama de audio (CNN y AST) y la rama de imagen:
  1. Pareo por especie repetido R veces con semillas distintas → media ± desviación
     estándar de accuracy y F1 macro para audio solo, imagen sola y fusión (protocolo que
     el Capítulo 3 compromete).
  2. Barrido del peso α de la \\textit{late fusion} (0 = solo imagen, 1 = solo audio), con media
     aritmética y media geométrica de probabilidades.
  3. Análisis de complementariedad: en qué fracción de pares acierta solo una modalidad y
     cuántos de esos pares resuelve la fusión.
  4. Desempeño de la fusión por grupo de conservación.

    uv run python scripts/31_fusion_analysis.py
    uv run python scripts/31_fusion_analysis.py --repeats 50 --pairs 30

Salidas: reports/fusion_alpha_sweep.csv, fusion_summary.csv, fusion_complementarity.csv,
fusion_by_group.csv. Segundos de cómputo.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT  # noqa: E402
from biomonitor.evaluation import by_group, per_class_table, species_groups  # noqa: E402

REP = PROJECT_ROOT / "reports"
ALPHAS = np.round(np.arange(0.0, 1.01, 0.1), 2)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--probs-dir", default=str(REP / "probs"))
    p.add_argument("--repeats", type=int, default=20, help="repeticiones del pareo (semillas)")
    p.add_argument("--pairs", type=int, default=30, help="pares por especie y repetición")
    p.add_argument("--out-dir", default=str(REP))
    return p.parse_args()


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def to_common(probs: np.ndarray, classes: np.ndarray, common: list[str]) -> np.ndarray:
    """Reordena/recorta las columnas de probabilidad al espacio de clases común."""
    idx = {c: i for i, c in enumerate(classes)}
    return probs[:, [idx[c] for c in common]]


def pair_indices(y: np.ndarray, k: int, n_pairs: int, rng: np.random.Generator) -> np.ndarray:
    """n_pairs índices (con reemplazo) de muestras de la clase k."""
    cand = np.flatnonzero(y == k)
    return rng.choice(cand, size=n_pairs, replace=True)


def fuse(pa: np.ndarray, pi: np.ndarray, alpha: float, geometric: bool) -> np.ndarray:
    if geometric:
        return alpha * np.log(pa + 1e-9) + (1 - alpha) * np.log(pi + 1e-9)
    return alpha * pa + (1 - alpha) * pi


def main() -> None:
    args = get_args()
    probs_dir, out_dir = Path(args.probs_dir), Path(args.out_dir)
    img = load(probs_dir / "image_efficientnet_b3_test.npz")
    audio_branches = {"CNN (EfficientNet-B0)": "audio_efficientnet_b0_test.npz", "AST": "audio_ast_test.npz"}
    audio = {name: load(probs_dir / f) for name, f in audio_branches.items() if (probs_dir / f).exists()}
    if not audio:
        raise SystemExit("No hay probabilidades de audio en reports/probs — corre antes scripts/27_eval_extended.py")
    groups = species_groups()

    common = sorted(set(img["classes"]) & set(next(iter(audio.values()))["classes"]))
    K = len(common)
    Pi = to_common(img["probs_agg"], img["classes"], common)       # TTA
    yi = np.array([common.index(c) for c in img["classes"][img["y_true"]]])
    # (las especies solo-audio no tienen fotos: se excluyen del pareo, como en el script 13)

    sweep_rows, summary_rows, comp_rows, group_rows = [], [], [], []
    for name, a in audio.items():
        Pa = to_common(a["probs_agg"], a["classes"], common)
        ya_all = np.array([c for c in a["classes"][a["y_true"]]])
        keep = np.isin(ya_all, common)
        Pa, ya = Pa[keep], np.array([common.index(c) for c in ya_all[keep]])

        # ---- 1-2: pareo repetido + barrido de α ----
        metrics = {(alpha, geo): [] for alpha in ALPHAS for geo in (False, True)}
        comp = []
        y_pred_f_all, y_true_all = [], []
        for rep in range(args.repeats):
            rng = np.random.default_rng(1000 + rep)
            ia = np.concatenate([pair_indices(ya, k, args.pairs, rng) for k in range(K)])
            ii = np.concatenate([pair_indices(yi, k, args.pairs, rng) for k in range(K)])
            y = np.repeat(np.arange(K), args.pairs)
            pa, pi = Pa[ia], Pi[ii]
            for (alpha, geo), lst in metrics.items():
                pred = fuse(pa, pi, alpha, geo).argmax(1)
                lst.append((accuracy_score(y, pred), f1_score(y, pred, average="macro", zero_division=0)))
            ok_a, ok_i = pa.argmax(1) == y, pi.argmax(1) == y
            ok_f = fuse(pa, pi, 0.5, False).argmax(1) == y
            comp.append({"ambas": np.mean(ok_a & ok_i), "solo_audio": np.mean(ok_a & ~ok_i),
                         "solo_imagen": np.mean(~ok_a & ok_i), "ninguna": np.mean(~ok_a & ~ok_i),
                         "fusion_resuelve_discrepancias": np.mean(ok_f[ok_a ^ ok_i]) if np.any(ok_a ^ ok_i) else np.nan,
                         "fusion_falla_con_ambas_ok": np.mean(~ok_f[ok_a & ok_i]) if np.any(ok_a & ok_i) else np.nan})
            y_pred_f_all.append(fuse(pa, pi, 0.5, False).argmax(1)); y_true_all.append(y)

        for (alpha, geo), lst in metrics.items():
            arr = np.array(lst)
            sweep_rows.append({"rama_audio": name, "alpha": alpha, "fusion": "geométrica" if geo else "aritmética",
                               "accuracy_media": arr[:, 0].mean(), "accuracy_sd": arr[:, 0].std(),
                               "f1_media": arr[:, 1].mean(), "f1_sd": arr[:, 1].std()})
        for label, alpha in [("Audio solo", 1.0), ("Imagen sola", 0.0), ("Fusión tardía (α=0,5)", 0.5)]:
            arr = np.array(metrics[(alpha, False)])
            summary_rows.append({"rama_audio": name, "modalidad": label, "accuracy_media": arr[:, 0].mean(),
                                 "accuracy_sd": arr[:, 0].std(), "f1_media": arr[:, 1].mean(), "f1_sd": arr[:, 1].std(),
                                 "pares": K * args.pairs, "repeticiones": args.repeats, "especies": K})
        best = max(((al, g) for (al, g) in metrics), key=lambda t: np.mean([m[0] for m in metrics[t]]))
        arr = np.array(metrics[best])
        summary_rows.append({"rama_audio": name, "modalidad": f"Mejor fusión (α={best[0]}, {'geométrica' if best[1] else 'aritmética'})",
                             "accuracy_media": arr[:, 0].mean(), "accuracy_sd": arr[:, 0].std(),
                             "f1_media": arr[:, 1].mean(), "f1_sd": arr[:, 1].std(),
                             "pares": K * args.pairs, "repeticiones": args.repeats, "especies": K})
        c = pd.DataFrame(comp).mean().to_dict(); c["rama_audio"] = name; comp_rows.append(c)

        # ---- 4: por grupo de conservación (todas las repeticiones concatenadas) ----
        pc = per_class_table(np.concatenate(y_true_all), np.concatenate(y_pred_f_all), common)
        g = by_group(pc, groups); g.insert(0, "rama_audio", name); group_rows.extend(g.to_dict("records"))

    pd.DataFrame(sweep_rows).to_csv(out_dir / "fusion_alpha_sweep.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(out_dir / "fusion_summary.csv", index=False)
    pd.DataFrame(comp_rows).to_csv(out_dir / "fusion_complementarity.csv", index=False)
    pd.DataFrame(group_rows).to_csv(out_dir / "fusion_by_group.csv", index=False)

    print(f"== Fusión multimodal — pareo por especie ({K} especies × {args.pairs} pares × {args.repeats} repeticiones) ==")
    for r in summary_rows:
        print(f"  {r['rama_audio']:<22}{r['modalidad']:<40} acc {r['accuracy_media']:.3f} ± {r['accuracy_sd']:.3f}"
              f"   F1 {r['f1_media']:.3f} ± {r['f1_sd']:.3f}")
    print("\n== Complementariedad (fracción de pares) ==")
    for c in comp_rows:
        print(f"  {c['rama_audio']:<22} ambas {c['ambas']:.3f} | solo audio {c['solo_audio']:.3f} | solo imagen "
              f"{c['solo_imagen']:.3f} | ninguna {c['ninguna']:.3f} | la fusión resuelve {c['fusion_resuelve_discrepancias']:.1%} "
              f"de las discrepancias")
    print(f"\n  → {out_dir / 'fusion_alpha_sweep.csv'}, fusion_summary.csv, fusion_complementarity.csv, fusion_by_group.csv\n")


if __name__ == "__main__":
    main()
