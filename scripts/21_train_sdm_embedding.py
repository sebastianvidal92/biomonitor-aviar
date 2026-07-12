"""SDM con embedding satelital (Módulo 2): MaxEnt vs DL-tabular vs DL+embedding.

Prueba si darle al deep learning la representación satelital de un foundation model
(64-dim, datos ricos que MaxEnt no puede explotar) le permite superar a MaxEnt.
Reporta AUC-ROC por especie de los tres modelos.

Requiere: scripts/16 (sdm_dataset.csv) y scripts/20 (sdm_embedding.csv).

    uv run python scripts/21_train_sdm_embedding.py
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
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.sdm.model import DeepSDM  # noqa: E402

C = CONFIG["train_sdm"]
MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = C["background_per_species"]
TEST_FRAC = C["test_frac"]
SEED = 42
BASE = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def train_predict(feats, train, test, n_species, pos_weight):
    """Entrena un DeepSDM con las columnas `feats` y devuelve prob en test."""
    Xtr = torch.tensor(train[feats].values, dtype=torch.float32)
    Xte = torch.tensor(test[feats].values, dtype=torch.float32)
    sptr = torch.tensor(train["sp_idx"].values, dtype=torch.long)
    ytr = torch.tensor(train["label"].values, dtype=torch.float32)
    model = DeepSDM(len(feats), n_species, C["emb_dim"], C["hidden"]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    dl = DataLoader(TensorDataset(Xtr, sptr, ytr), batch_size=C["batch_size"], shuffle=True)
    model.train()
    for _ in range(C["epochs"]):
        for xb, sb, yb in dl:
            opt.zero_grad()
            lossf(model(xb.to(DEVICE), sb.to(DEVICE)), yb.to(DEVICE)).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        p = torch.sigmoid(model(Xte.to(DEVICE),
                                torch.tensor(test["sp_idx"].values, dtype=torch.long).to(DEVICE)))
    return p.cpu().numpy()


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
    sp2idx = {s: i for i, s in enumerate(species)}
    print(f"Especies: {len(species)} | embedding: {len(emb_cols)}-dim | dispositivo: {DEVICE}\n")

    tr_parts, te_parts = [], []
    for s in species:
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(BG_PER, len(background)), random_state=SEED)],
                        ignore_index=True)
        sub["model_sp"], sub["sp_idx"] = s, sp2idx[s]
        tr, te = train_test_split(sub, test_size=TEST_FRAC, stratify=sub["label"], random_state=SEED)
        tr_parts.append(tr); te_parts.append(te)
    train = pd.concat(tr_parts, ignore_index=True)
    test = pd.concat(te_parts, ignore_index=True)

    scaler = StandardScaler().fit(train[ALL])
    train[ALL] = scaler.transform(train[ALL])
    test[ALL] = scaler.transform(test[ALL])
    pos_weight = torch.tensor(
        [(train["label"] == 0).sum() / max((train["label"] == 1).sum(), 1)], dtype=torch.float32)

    print("Entrenando DL tabular (base)...")
    test["p_tab"] = train_predict(BASE, train, test, len(species), pos_weight)
    print("Entrenando DL + embedding (base + 64-dim)...")
    test["p_emb"] = train_predict(ALL, train, test, len(species), pos_weight)

    print("Ajustando MaxEnt por especie y evaluando...\n")
    res = []
    for s in species:
        tr_s, te_s = train[train["model_sp"] == s], test[test["model_sp"] == s]
        me = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                           LogisticRegression(class_weight="balanced", max_iter=1000))
        me.fit(tr_s[BASE], tr_s["label"])
        res.append({
            "species": s, "n_presence": int((presence["species"] == s).sum()),
            "auc_maxent": roc_auc_score(te_s["label"], me.predict_proba(te_s[BASE])[:, 1]),
            "auc_dl_tab": roc_auc_score(te_s["label"], te_s["p_tab"]),
            "auc_dl_emb": roc_auc_score(te_s["label"], te_s["p_emb"]),
        })
    rdf = pd.DataFrame(res)
    out = PROJECT_ROOT / "reports" / "sdm_embedding_auc.csv"
    rdf.to_csv(out, index=False)

    print(f"== SDM — AUC-ROC (test, {len(species)} especies) ==")
    print(f"  MaxEnt (tabular):            {rdf['auc_maxent'].mean():.3f}")
    print(f"  DL tabular:                  {rdf['auc_dl_tab'].mean():.3f}")
    print(f"  DL + embedding satelital:    {rdf['auc_dl_emb'].mean():.3f}")
    print(f"  DL-emb supera a MaxEnt en {int((rdf['auc_dl_emb'] > rdf['auc_maxent']).sum())}/{len(species)} especies")
    print(f"  → detalle: {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
