"""SDM con satélite PAREADO TEMPORALMENTE (Módulo 2) — observaciones recientes.

Responde: ¿usar la imagen satelital del AÑO del avistamiento (en vez de un compuesto fijo)
mejora el SDM? Se toman observaciones 2017-2025 (rango del embedding satelital) y se compara
Random Forest con: base, base+embedding FIJO (2023) y base+embedding PAREADO (año de la
observación). Métrica: AUC-ROC por especie.

Requiere: scripts/15 (occurrences.csv).

    uv run python scripts/25_temporal_sdm.py

NOTA: muchas consultas a GEE (por año) → varios minutos. Cachea el dataset extraído.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.data.gee import init_ee  # noqa: E402

MIN = CONFIG["sdm"]["min_occurrences"]
BASE = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]
YEARS = range(2017, 2026)
CAP, N_BG, FIXED_YEAR, SEED = 300, 5000, 2023, 42


def build_dataset() -> pd.DataFrame:
    cache = data_path("interim", "sdm_temporal.csv")
    if cache.exists():
        return pd.read_csv(cache)

    init_ee()
    from biomonitor.sdm import covariates as cov

    occ = pd.read_csv(data_path("interim", "occurrences.csv")).dropna(subset=["lat", "lon", "year"])
    occ = occ[occ["year"].between(YEARS.start, YEARS.stop - 1)]
    pres = pd.concat([g.sample(min(len(g), CAP), random_state=SEED)
                      for _, g in occ.groupby("scientific_name")], ignore_index=True)
    bg = occ.sample(min(N_BG, len(occ)), random_state=SEED).assign(scientific_name="background")
    df = pd.concat([pres.assign(label=1), bg.assign(label=0)], ignore_index=True)
    df["idx"] = range(len(df))
    df["year"] = df["year"].astype(int)
    meta = df[["idx", "scientific_name", "month", "year", "label"]].rename(
        columns={"scientific_name": "species"})

    pts = [{"lat": r.lat, "lon": r.lon, "idx": int(r.idx)} for r in df.itertuples(index=False)]

    # 1) estáticas (temp, precip, elev)
    print("Muestreando covariables estáticas...")
    stat = cov.sample_covariates(pts, cov.static_covariates())[["idx", "temp", "precip", "elev"]]

    # 2) matched: NDVI + embedding del AÑO de cada observación (por grupo de año)
    matched = []
    for yr in tqdm(YEARS, desc="Pareado por año"):
        grp = [p for p, y in zip(pts, df["year"]) if y == yr]
        if not grp:
            continue
        img = cov.ndvi_year(yr).addBands(cov.embedding_year(yr))
        matched.append(cov.sample_covariates(grp, img))
    matched = pd.concat(matched, ignore_index=True)
    emb_cols = sorted(c for c in matched.columns if c.startswith("A"))
    matched = matched[["idx", "ndvi"] + emb_cols]

    # 3) fixed: embedding de un año fijo (2023) para todos
    print(f"Muestreando embedding FIJO ({FIXED_YEAR})...")
    fixed = cov.sample_covariates(pts, cov.embedding_year(FIXED_YEAR))
    fixed = fixed[["idx"] + emb_cols].rename(columns={c: f"fx_{c}" for c in emb_cols})

    out = meta.merge(stat, on="idx").merge(matched, on="idx").merge(fixed, on="idx")
    out = out.dropna(subset=BASE[:4] + emb_cols)
    out.to_csv(cache, index=False)
    return out


def rf():
    return RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=SEED)


def main() -> None:
    df = build_dataset()
    df["sin_m"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)
    emb = sorted(c for c in df.columns if c.startswith("A") and not c.startswith("fx"))
    fx = [f"fx_{c}" for c in emb]

    presence = df[df["label"] == 1]
    background = df[df["label"] == 0]
    species = sorted(presence["species"].value_counts().loc[lambda c: c >= MIN].index)
    print(f"\nEspecies (≥{MIN} presencias recientes): {len(species)} | puntos: {len(df)} "
          f"(años {YEARS.start}-{YEARS.stop - 1})\n")

    rows = []
    for s in species:
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(len(background), 1500), random_state=SEED)])
        tr, te = train_test_split(sub, test_size=0.3, stratify=sub["label"], random_state=SEED)
        me = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                           LogisticRegression(class_weight="balanced", max_iter=1000))
        r = {"species": s,
             "maxent": roc_auc_score(te["label"], me.fit(tr[BASE], tr["label"]).predict_proba(te[BASE])[:, 1])}
        for name, feats in [("rf_base", BASE), ("rf_fijo", BASE + fx), ("rf_pareado", BASE + emb)]:
            r[name] = roc_auc_score(te["label"], rf().fit(tr[feats], tr["label"]).predict_proba(te[feats])[:, 1])
        rows.append(r)

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "sdm_temporal_auc.csv"
    res.to_csv(out, index=False)

    print(f"== SDM temporal — AUC-ROC promedio (test, {len(species)} especies, {len(df)} puntos) ==")
    for c, name in [("maxent", "MaxEnt (base)"), ("rf_base", "Random Forest (base, sin satélite)"),
                    ("rf_fijo", f"RF + embedding FIJO ({FIXED_YEAR})"),
                    ("rf_pareado", "RF + embedding PAREADO (año de la observación)")]:
        print(f"  {name:<46}{res[c].mean():.3f}")
    print(f"\n  → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
