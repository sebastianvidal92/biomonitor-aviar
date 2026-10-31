"""Validación cruzada ESPACIAL por bloques para el SDM (Módulo 2) — informe final.

Replica el protocolo de blockCV (Valavi et al., 2019; Roberts et al., 2017):
  1. Estima el rango de autocorrelación espacial de cada covariable con un semivariograma
     empírico (distancia a la que la semivarianza alcanza el 95 % de la varianza total).
  2. Divide el territorio en bloques cuadrados de `--block-km` y asigna los bloques a
     `--folds` folds al azar, eligiendo entre 200 asignaciones la que mejor equilibra el
     número de registros por fold.
  3. Para cada especie entrena MaxEnt, Random Forest y Gradient Boosting en k-1 folds y
     evalúa el AUC-ROC en el fold restante; la red neuronal conjunta (multi-especie) se
     entrena una vez por fold con todas las especies. Mismos puntos y mismo fondo
     (target-group, 1.500 por especie) que los scripts 17 y 22, de modo que la única
     diferencia con esos resultados es el protocolo de partición.

En Chile los rangos resultan muy largos (650-1.200 km) por el gradiente latitudinal de
clima y elevación; bloques de ese tamaño dejarían 4-6 bloques en todo el país. Por eso el
tamaño por defecto es 200 km (≈ 35 bloques), que elimina la autocorrelación de corto
alcance manteniendo folds con todas las zonas climáticas; `--block-km` admite varios
valores para analizar la sensibilidad.

    uv run python scripts/28_sdm_spatial_cv.py                       # bloques de 200 km
    uv run python scripts/28_sdm_spatial_cv.py --block-km 100 200 400  # sensibilidad

Salidas: reports/sdm_spatial_cv.csv (AUC por especie, modelo y tamaño de bloque),
reports/sdm_spatial_cv_summary.json (rangos, bloques, promedios, comparación con el
split aleatorio). ~10 min en CPU por tamaño de bloque.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.sdm.model import DeepSDM  # noqa: E402

C = CONFIG["train_sdm"]
MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = C["background_per_species"]
SEED = 42
BASE = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]
COV = ["ndvi", "temp", "precip", "elev"]
R_EARTH = 6371.0
MIN_TEST_PRES = 5  # presencias mínimas en el fold de prueba para calcular AUC


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--block-km", type=float, nargs="+", default=[200.0])
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--skip-dl", action="store_true", help="omite la red conjunta")
    p.add_argument("--limit", type=int, default=None, help="smoke test: N especies, sin guardar")
    return p.parse_args()


# ------------------------- 1. autocorrelación espacial -------------------------
def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def autocorrelation_ranges(df: pd.DataFrame, n: int = 2500, bin_km: float = 50, max_km: float = 2000) -> dict:
    """Rango empírico (km) por covariable: semivarianza ≥ 95 % de la varianza total."""
    s = df.sample(min(n, len(df)), random_state=SEED)
    lat, lon = s["lat"].values, s["lon"].values
    d = haversine_km(lat[:, None], lon[:, None], lat[None, :], lon[None, :])
    iu = np.triu_indices(len(s), 1)
    d = d[iu]
    bins = np.arange(0, max_km + bin_km, bin_km)
    idx = np.digitize(d, bins)
    out = {}
    for c in COV:
        v = s[c].values
        v = (v - v.mean()) / v.std()
        g = (0.5 * (v[:, None] - v[None, :]) ** 2)[iu]
        gamma = np.array([g[idx == i].mean() if (idx == i).sum() > 50 else np.nan for i in range(1, len(bins))])
        reach = [float(bins[i + 1]) for i, x in enumerate(gamma) if not np.isnan(x) and x >= 0.95]
        out[c] = {"rango_km": reach[0] if reach else None,
                  "semivarianza": {int(bins[i + 1]): float(x) for i, x in enumerate(gamma) if not np.isnan(x)}}
    return out


# ------------------------------ 2. bloques y folds ------------------------------
def assign_blocks(df: pd.DataFrame, block_km: float) -> pd.Series:
    lat0, lon0 = df["lat"].min(), df["lon"].min()
    y_km = (df["lat"] - lat0) * 111.0
    x_km = (df["lon"] - lon0) * 111.0 * np.cos(np.radians(df["lat"]))
    return (np.floor(y_km / block_km).astype(int) * 10000 + np.floor(x_km / block_km).astype(int))


def assign_folds(blocks: pd.Series, k: int, n_iter: int = 200) -> pd.Series:
    """Bloques → folds al azar, eligiendo la asignación más equilibrada en registros."""
    rng = np.random.default_rng(SEED)
    uniq = np.array(sorted(blocks.unique()))
    sizes = blocks.value_counts()
    best, best_score = None, np.inf
    for _ in range(n_iter):
        perm = rng.permutation(uniq)
        fold_of = {b: i % k for i, b in enumerate(perm)}
        counts = np.zeros(k)
        for b, n in sizes.items():
            counts[fold_of[b]] += n
        score = counts.std() if counts.min() > 0 else np.inf
        if score < best_score:
            best, best_score = fold_of, score
    return blocks.map(best)


# --------------------------------- 3. modelos ----------------------------------
def maxent():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced",
                                          random_state=SEED)


def train_dl(train: pd.DataFrame, n_species: int, device) -> tuple[DeepSDM, StandardScaler]:
    scaler = StandardScaler().fit(train[BASE])
    X = torch.tensor(scaler.transform(train[BASE]), dtype=torch.float32)
    sp = torch.tensor(train["sp_idx"].values, dtype=torch.long)
    y = torch.tensor(train["label"].values, dtype=torch.float32)
    pos_weight = torch.tensor([(y == 0).sum() / max((y == 1).sum(), 1)])
    model = DeepSDM(len(BASE), n_species, C["emb_dim"], C["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    dl = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X, sp, y), batch_size=C["batch_size"], shuffle=True)
    torch.manual_seed(SEED)
    model.train()
    for _ in range(C["epochs"]):
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb, sb), yb).backward()
            opt.step()
    model.eval()
    return model, scaler


@torch.no_grad()
def predict_dl(model, scaler, test: pd.DataFrame, device) -> np.ndarray:
    X = torch.tensor(scaler.transform(test[BASE]), dtype=torch.float32).to(device)
    sp = torch.tensor(test["sp_idx"].values, dtype=torch.long).to(device)
    return torch.sigmoid(model(X, sp)).cpu().numpy()


def run_block_size(df: pd.DataFrame, species: list[str], block_km: float, k: int, skip_dl: bool,
                   device) -> tuple[pd.DataFrame, dict]:
    presence, background = df[df["label"] == 1], df[df["label"] == 0]
    sp2idx = {s: i for i, s in enumerate(species)}

    # dataset por especie (mismos puntos/fondo que los scripts 17 y 22) con su fold espacial
    parts = []
    for s in species:
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(BG_PER, len(background)), random_state=SEED)],
                        ignore_index=True)
        sub["model_sp"], sub["sp_idx"] = s, sp2idx[s]
        parts.append(sub)
    data = pd.concat(parts, ignore_index=True)
    n_blocks = data["block"].nunique()
    fold_sizes = data.drop_duplicates(["lat", "lon", "label", "model_sp"])["fold"].value_counts().sort_index()

    rows = {s: {"species": s, "block_km": block_km,
                "n_presence": int((presence["species"] == s).sum())} for s in species}
    per_fold = {m: {s: [] for s in species} for m in ["maxent", "rf", "hgb", "dl"]}

    for f in range(k):
        t0 = time.time()
        tr, te = data[data["fold"] != f], data[data["fold"] == f]
        dl_model = None
        if not skip_dl and len(tr) and len(te):
            dl_model, scaler = train_dl(tr, len(species), device)
            te = te.assign(dl_prob=predict_dl(dl_model, scaler, te, device))
        for s in species:
            tr_s, te_s = tr[tr["model_sp"] == s], te[te["model_sp"] == s]
            if te_s["label"].sum() < MIN_TEST_PRES or (te_s["label"] == 0).sum() < MIN_TEST_PRES \
                    or tr_s["label"].sum() < MIN_TEST_PRES:
                continue
            per_fold["maxent"][s].append(roc_auc_score(
                te_s["label"], maxent().fit(tr_s[BASE], tr_s["label"]).predict_proba(te_s[BASE])[:, 1]))
            per_fold["rf"][s].append(roc_auc_score(
                te_s["label"], rf().fit(tr_s[BASE], tr_s["label"]).predict_proba(te_s[BASE])[:, 1]))
            per_fold["hgb"][s].append(roc_auc_score(
                te_s["label"], hgb().fit(tr_s[BASE], tr_s["label"]).predict_proba(te_s[BASE])[:, 1]))
            if dl_model is not None:
                per_fold["dl"][s].append(roc_auc_score(te_s["label"], te_s["dl_prob"]))
        print(f"    fold {f + 1}/{k}: test {len(te):,} filas ({time.time() - t0:.0f} s)")

    for m, d in per_fold.items():
        for s in species:
            v = d[s]
            rows[s][m] = float(np.mean(v)) if v else np.nan
            rows[s][f"{m}_sd"] = float(np.std(v)) if len(v) > 1 else np.nan
            rows[s][f"{m}_folds"] = len(v)
    res = pd.DataFrame(rows.values())
    info = {"block_km": block_km, "n_blocks": int(n_blocks), "folds": k,
            "registros_por_fold": {int(i): int(v) for i, v in fold_sizes.items()}}
    return res, info


def main() -> None:
    args = get_args()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    df = pd.read_csv(data_path("interim", "sdm_dataset.csv")).reset_index(drop=True)
    df["sin_m"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)
    counts = df[df["label"] == 1]["species"].value_counts()
    species = sorted(counts[counts >= MIN].index)
    save = args.limit is None
    if not save:
        species = species[: args.limit]
        print(f"SMOKE TEST: {len(species)} especies, no se guarda nada\n")

    print("Estimando rangos de autocorrelación espacial (semivariogramas)...")
    ranges = autocorrelation_ranges(df)
    for c, r in ranges.items():
        print(f"  {c:<7} rango (95 % de la varianza): {r['rango_km']} km")

    results, infos = [], []
    for bk in args.block_km:
        df["block"] = assign_blocks(df, bk)
        df["fold"] = assign_folds(df["block"], args.folds)
        print(f"\nBloques de {bk:.0f} km → {df['block'].nunique()} bloques, {args.folds} folds; "
              f"{len(species)} especies")
        res, info = run_block_size(df, species, bk, args.folds, args.skip_dl, device)
        results.append(res); infos.append(info)
        print(f"  AUC medio — MaxEnt {res['maxent'].mean():.3f} | RF {res['rf'].mean():.3f} | "
              f"GBM {res['hgb'].mean():.3f}" + ("" if args.skip_dl else f" | red {res['dl'].mean():.3f}"))

    res = pd.concat(results, ignore_index=True)
    summary = {"rangos_autocorrelacion": {c: r["rango_km"] for c, r in ranges.items()},
               "semivariogramas": {c: r["semivarianza"] for c, r in ranges.items()},
               "configuraciones": []}
    for info, r in zip(infos, results):
        cfg = dict(info)
        for m in ["maxent", "rf", "hgb", "dl"]:
            if r[m].notna().any():
                cfg[f"auc_{m}"] = float(r[m].mean())
                cfg[f"gana_a_maxent_{m}"] = int((r[m] > r["maxent"]).sum())
        cfg["especies_evaluadas"] = int(r["rf"].notna().sum())
        summary["configuraciones"].append(cfg)
    # referencia: split aleatorio (scripts 17/22)
    ml_csv, dl_csv = PROJECT_ROOT / "reports/sdm_ml_auc.csv", PROJECT_ROOT / "reports/sdm_auc_por_especie.csv"
    if ml_csv.exists():
        ml = pd.read_csv(ml_csv)
        summary["split_aleatorio"] = {"maxent": float(ml["maxent"].mean()), "rf": float(ml["rf_base"].mean()),
                                      "hgb": float(ml["hgb_base"].mean())}
        if dl_csv.exists():
            summary["split_aleatorio"]["dl"] = float(pd.read_csv(dl_csv)["auc_dl"].mean())

    if save:
        res.to_csv(PROJECT_ROOT / "reports/sdm_spatial_cv.csv", index=False)
        (PROJECT_ROOT / "reports/sdm_spatial_cv_summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False))
        print("\n  → reports/sdm_spatial_cv.csv, reports/sdm_spatial_cv_summary.json\n")
    else:
        print("\n", json.dumps({k: v for k, v in summary.items() if k != "semivariogramas"}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
