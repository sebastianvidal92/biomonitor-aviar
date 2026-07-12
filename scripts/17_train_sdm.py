"""Entrena y compara el SDM (Módulo 2): deep learning conjunto vs MaxEnt por especie.

- MaxEnt (baseline clásico): regresión logística penalizada con features cuadráticas,
  ajustada por especie sobre presencia vs fondo.
- SDM deep learning: red neuronal conjunta multi-especie (embedding de especie).
- Métrica: AUC-ROC por especie sobre un test estratificado. Reporta el promedio y guarda
  el detalle por especie.

    uv run python scripts/17_train_sdm.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.sdm.model import DeepSDM  # noqa: E402

C = CONFIG["train_sdm"]
MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = C["background_per_species"]
TEST_FRAC = C["test_frac"]
SEED = 42
FEATS = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]


def add_time_feats(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sin_m"] = np.sin(2 * np.pi * df["month"].fillna(0) / 12)
    df["cos_m"] = np.cos(2 * np.pi * df["month"].fillna(0) / 12)
    return df


def main() -> None:
    df = add_time_feats(pd.read_csv(data_path("interim", "sdm_dataset.csv")))
    presence = df[df["label"] == 1]
    background = df[df["label"] == 0]
    counts = presence["species"].value_counts()
    species = sorted(counts[counts >= MIN].index)
    sp2idx = {s: i for i, s in enumerate(species)}
    print(f"Especies modeladas (≥{MIN} presencias): {len(species)}\n")

    # --- armar train/test por especie (presencia + fondo compartido) ---
    rng = np.random.default_rng(SEED)
    tr_parts, te_parts = [], []
    for s in species:
        pres_s = presence[presence["species"] == s]
        bg_s = background.sample(min(BG_PER, len(background)), random_state=SEED)
        sub = pd.concat([pres_s, bg_s], ignore_index=True)
        sub["model_sp"] = s
        sub["sp_idx"] = sp2idx[s]
        tr, te = train_test_split(sub, test_size=TEST_FRAC, stratify=sub["label"],
                                  random_state=SEED)
        tr_parts.append(tr); te_parts.append(te)
    train = pd.concat(tr_parts, ignore_index=True)
    test = pd.concat(te_parts, ignore_index=True)

    scaler = StandardScaler().fit(train[FEATS])

    # --- deep learning conjunto ---
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    Xtr = torch.tensor(scaler.transform(train[FEATS]), dtype=torch.float32)
    sptr = torch.tensor(train["sp_idx"].values, dtype=torch.long)
    ytr = torch.tensor(train["label"].values, dtype=torch.float32)
    pos_weight = torch.tensor([(ytr == 0).sum() / max((ytr == 1).sum(), 1)])

    model = DeepSDM(len(FEATS), len(species), C["emb_dim"], C["hidden"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    ds = torch.utils.data.TensorDataset(Xtr, sptr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=C["batch_size"], shuffle=True)

    print("Entrenando SDM deep learning (conjunto)...")
    model.train()
    for epoch in range(C["epochs"]):
        for xb, sb, yb in dl:
            xb, sb, yb = xb.to(device), sb.to(device), yb.to(device)
            opt.zero_grad()
            lossf(model(xb, sb), yb).backward()
            opt.step()

    # --- evaluación por especie: DL vs MaxEnt ---
    model.eval()
    Xte_all = scaler.transform(test[FEATS])
    with torch.no_grad():
        dl_prob = torch.sigmoid(model(
            torch.tensor(Xte_all, dtype=torch.float32).to(device),
            torch.tensor(test["sp_idx"].values, dtype=torch.long).to(device),
        )).cpu().numpy()
    test = test.assign(dl_prob=dl_prob)

    rows = []
    for s in species:
        tr_s, te_s = train[train["model_sp"] == s], test[test["model_sp"] == s]
        maxent = make_pipeline(
            StandardScaler(), PolynomialFeatures(2, include_bias=False),
            LogisticRegression(class_weight="balanced", max_iter=1000))
        maxent.fit(tr_s[FEATS], tr_s["label"])
        p_me = maxent.predict_proba(te_s[FEATS])[:, 1]
        rows.append({
            "species": s, "n_presence": int((presence["species"] == s).sum()),
            "auc_maxent": roc_auc_score(te_s["label"], p_me),
            "auc_dl": roc_auc_score(te_s["label"], te_s["dl_prob"]),
        })

    res = pd.DataFrame(rows)
    out = PROJECT_ROOT / "reports" / "sdm_auc_por_especie.csv"
    out.parent.mkdir(exist_ok=True)
    res.to_csv(out, index=False)

    print(f"\n== SDM — AUC-ROC (test, {len(species)} especies) ==")
    print(f"  MaxEnt (baseline):     {res['auc_maxent'].mean():.3f}")
    print(f"  Deep learning conjunto: {res['auc_dl'].mean():.3f}")
    wins = int((res['auc_dl'] > res['auc_maxent']).sum())
    print(f"  DL supera a MaxEnt en {wins}/{len(species)} especies")
    print(f"  → detalle por especie: {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
