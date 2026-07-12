"""SDM con modelos de ML clásicos (Módulo 2): MaxEnt vs Random Forest vs Gradient Boosting.

Los ensambles de árboles (Boosted Regression Trees) son un método estándar y fuerte de
SDM, y manejan muchas features (incl. el embedding satelital de 64-dim) sin sobreajustar
como la red neuronal. Compara AUC-ROC por especie sobre covariables base y base+embedding.

Requiere: scripts/16 (sdm_dataset.csv) y scripts/20 (sdm_embedding.csv).

    uv run python scripts/22_train_sdm_ml.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402

MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = CONFIG["train_sdm"]["background_per_species"]
TEST_FRAC = CONFIG["train_sdm"]["test_frac"]
SEED = 42
BASE = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]


def maxent():
    return make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                         LogisticRegression(class_weight="balanced", max_iter=1000))


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                  n_jobs=-1, random_state=SEED)


def hgb():
    return HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05,
                                          class_weight="balanced", random_state=SEED)


def main() -> None:
    df = pd.read_csv(data_path("interim", "sdm_dataset.csv")).reset_index(drop=True)
    df["idx"] = range(len(df))
    emb = pd.read_csv(data_path("interim", "sdm_embedding.csv"))
    emb_cols = sorted(c for c in emb.columns if c.startswith("A"))
    df = df.merge(emb[["idx"] + emb_cols], on="idx", how="inner")
    df["sin_m"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)
    ALL = BASE + emb_cols

    presence = df[df["label"] == 1]
    background = df[df["label"] == 0]
    counts = presence["species"].value_counts()
    species = sorted(counts[counts >= MIN].index)
    print(f"Especies: {len(species)} | embedding: {len(emb_cols)}-dim\n")

    rows = []
    for s in species:
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(BG_PER, len(background)), random_state=SEED)],
                        ignore_index=True)
        tr, te = train_test_split(sub, test_size=TEST_FRAC, stratify=sub["label"], random_state=SEED)
        r = {"species": s, "n_presence": int((presence["species"] == s).sum())}
        r["maxent"] = roc_auc_score(te["label"], maxent().fit(tr[BASE], tr["label"]).predict_proba(te[BASE])[:, 1])
        for name, feats in [("base", BASE), ("emb", ALL)]:
            r[f"rf_{name}"] = roc_auc_score(
                te["label"], rf().fit(tr[feats], tr["label"]).predict_proba(te[feats])[:, 1])
            r[f"hgb_{name}"] = roc_auc_score(
                te["label"], hgb().fit(tr[feats], tr["label"]).predict_proba(te[feats])[:, 1])
        rows.append(r)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "sdm_ml_auc.csv"
    res.to_csv(out, index=False)

    print(f"== SDM — AUC-ROC promedio (test, {len(species)} especies) ==")
    cols = {"maxent": "MaxEnt (base)", "rf_base": "Random Forest (base)",
            "hgb_base": "Grad. Boosting (base)", "rf_emb": "Random Forest (base+emb)",
            "hgb_emb": "Grad. Boosting (base+emb)"}
    for c, name in cols.items():
        wins = int((res[c] > res["maxent"]).sum())
        print(f"  {name:<28} {res[c].mean():.3f}   (gana a MaxEnt en {wins}/{len(species)})")
    print(f"\n  → detalle: {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
