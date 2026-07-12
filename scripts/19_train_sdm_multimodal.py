"""SDM multimodal (Módulo 2): compara MaxEnt vs DL-tabular vs DL-tabular+imagen.

Prueba la hipótesis de que el deep learning supera a MaxEnt cuando incorpora datos NO
estructurados (un parche satelital de cobertura de suelo, vía CNN) que MaxEnt no puede
procesar. Reporta AUC-ROC por especie de los tres modelos.

Requiere: scripts/16 (sdm_dataset.csv) y scripts/18 (sdm_patches.npz + rows).

    uv run python scripts/19_train_sdm_multimodal.py
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
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.sdm.model import DeepSDM  # noqa: E402
from biomonitor.sdm.multimodal_model import MultimodalSDM  # noqa: E402
from biomonitor.sdm.patches import N_CLASSES, one_hot_single  # noqa: E402

C = CONFIG["train_sdm"]
MIN = CONFIG["sdm"]["min_occurrences"]
BG_PER = C["background_per_species"]
TEST_FRAC = C["test_frac"]
SEED = 42
FEATS = ["ndvi", "temp", "precip", "elev", "sin_m", "cos_m"]
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class PatchDataset(Dataset):
    def __init__(self, env, sp, y, rows, patches):
        self.env = torch.tensor(env, dtype=torch.float32)
        self.sp = torch.tensor(sp, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.rows, self.patches = rows, patches

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        patch = torch.from_numpy(one_hot_single(self.patches[self.rows[i]]))
        return self.env[i], self.sp[i], patch, self.y[i]


def train_dl(model, loader, multimodal, pos_weight, epochs):
    opt = torch.optim.AdamW(model.parameters(), lr=C["lr"], weight_decay=0.01)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(DEVICE))
    model.train()
    for _ in range(epochs):
        for batch in loader:
            if multimodal:
                env, sp, patch, y = (b.to(DEVICE) for b in batch)
                logits = model(env, sp, patch)
            else:
                env, sp, y = (b.to(DEVICE) for b in batch)
                logits = model(env, sp)
            opt.zero_grad(); lossf(logits, y).backward(); opt.step()


@torch.no_grad()
def predict_dl(model, loader, multimodal):
    model.eval()
    out = []
    for batch in loader:
        if multimodal:
            env, sp, patch, _ = (b.to(DEVICE) for b in batch)
            out.append(torch.sigmoid(model(env, sp, patch)).cpu())
        else:
            env, sp, _ = (b.to(DEVICE) for b in batch)
            out.append(torch.sigmoid(model(env, sp)).cpu())
    return torch.cat(out).numpy()


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    rows = pd.read_csv(data_path("interim", "sdm_patches_rows.csv")).reset_index(drop=True)
    rows["sin_m"] = np.sin(2 * np.pi * rows["month"].fillna(0) / 12)
    rows["cos_m"] = np.cos(2 * np.pi * rows["month"].fillna(0) / 12)
    rows["row"] = range(len(rows))
    patches = np.load(data_path("interim", "sdm_patches.npz"))["patches"]  # (M,33,33)

    presence = rows[rows["label"] == 1]
    background = rows[rows["label"] == 0]
    counts = presence["species"].value_counts()
    species = sorted(counts[counts >= MIN].index)
    sp2idx = {s: i for i, s in enumerate(species)}
    print(f"Especies: {len(species)} | dispositivo: {DEVICE}\n")

    tr_parts, te_parts = [], []
    for s in species:
        sub = pd.concat([presence[presence["species"] == s],
                         background.sample(min(BG_PER, len(background)), random_state=SEED)],
                        ignore_index=True)
        sub["model_sp"] = s
        sub["sp_idx"] = sp2idx[s]
        tr, te = train_test_split(sub, test_size=TEST_FRAC, stratify=sub["label"], random_state=SEED)
        tr_parts.append(tr); te_parts.append(te)
    train = pd.concat(tr_parts, ignore_index=True)
    test = pd.concat(te_parts, ignore_index=True)

    scaler = StandardScaler().fit(train[FEATS])
    Etr, Ete = scaler.transform(train[FEATS]), scaler.transform(test[FEATS])
    pos_weight = torch.tensor(
        [(train["label"] == 0).sum() / max((train["label"] == 1).sum(), 1)], dtype=torch.float32)

    # --- DL tabular ---
    print("Entrenando DL tabular...")
    tab = DeepSDM(len(FEATS), len(species), C["emb_dim"], C["hidden"]).to(DEVICE)
    tl = DataLoader(TensorDataset(torch.tensor(Etr, dtype=torch.float32),
                                  torch.tensor(train["sp_idx"].values),
                                  torch.tensor(train["label"].values, dtype=torch.float32)),
                    batch_size=C["batch_size"], shuffle=True)
    train_dl(tab, tl, False, pos_weight, C["epochs"])
    tab_te = DataLoader(TensorDataset(torch.tensor(Ete, dtype=torch.float32),
                                      torch.tensor(test["sp_idx"].values),
                                      torch.tensor(test["label"].values, dtype=torch.float32)),
                        batch_size=1024)
    test["p_tab"] = predict_dl(tab, tab_te, False)

    # --- DL multimodal (tabular + parche) ---
    print("Entrenando DL multimodal (tabular + imagen)...")
    mm = MultimodalSDM(len(FEATS), len(species), N_CLASSES, C["emb_dim"], C["hidden"]).to(DEVICE)
    ml = DataLoader(PatchDataset(Etr, train["sp_idx"].values, train["label"].values,
                                 train["row"].values, patches),
                    batch_size=C["batch_size"], shuffle=True, num_workers=2)
    train_dl(mm, ml, True, pos_weight, C["epochs"])
    mm_te = DataLoader(PatchDataset(Ete, test["sp_idx"].values, test["label"].values,
                                    test["row"].values, patches), batch_size=512, num_workers=2)
    test["p_mm"] = predict_dl(mm, mm_te, True)

    # --- MaxEnt por especie + AUC de los tres ---
    print("Ajustando MaxEnt por especie y evaluando...\n")
    res = []
    for s in species:
        tr_s, te_s = train[train["model_sp"] == s], test[test["model_sp"] == s]
        me = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False),
                           LogisticRegression(class_weight="balanced", max_iter=1000))
        me.fit(tr_s[FEATS], tr_s["label"])
        p_me = me.predict_proba(te_s[FEATS])[:, 1]
        res.append({
            "species": s, "n_presence": int((presence["species"] == s).sum()),
            "auc_maxent": roc_auc_score(te_s["label"], p_me),
            "auc_dl_tab": roc_auc_score(te_s["label"], te_s["p_tab"]),
            "auc_dl_mm": roc_auc_score(te_s["label"], te_s["p_mm"]),
        })
    df = pd.DataFrame(res)
    out = PROJECT_ROOT / "reports" / "sdm_multimodal_auc.csv"
    df.to_csv(out, index=False)

    print(f"== SDM — AUC-ROC (test, {len(species)} especies) ==")
    print(f"  MaxEnt (tabular):            {df['auc_maxent'].mean():.3f}")
    print(f"  DL tabular:                  {df['auc_dl_tab'].mean():.3f}")
    print(f"  DL multimodal (+imagen):     {df['auc_dl_mm'].mean():.3f}")
    print(f"  DL-mm supera a MaxEnt en {int((df['auc_dl_mm'] > df['auc_maxent']).sum())}/{len(species)} especies")
    print(f"  → detalle: {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
