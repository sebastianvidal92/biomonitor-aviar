"""Comparación de estrategias de fusión multimodal (Módulo 1): late vs early vs intermediate.

Extrae embeddings de los backbones ya entrenados (audio EfficientNet-B0, imagen
EfficientNet-B3), congelados, y entrena tres cabezas de fusión con configuración IDÉNTICA
(mismas épocas, datos pareados por especie, optimizador). Aísla el efecto de la estrategia
de fusión. Reporta accuracy y F1 macro en test.

    uv run python scripts/24_fusion_comparison.py

NOTA: la extracción de embeddings corre los backbones sobre todas las muestras (varios
minutos) → la cachea; reejecutar la comparación es instantáneo.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn as nn
import torchaudio
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402
from biomonitor.multimodal.fusion import EarlyFusion, IntermediateFusion, LateFusion  # noqa: E402
from biomonitor.vision.dataset import _eval_tf  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
EPOCHS, LR, BATCH = 40, 1e-3, 64
PAIRS_TRAIN, PAIRS_TEST = 150, 30
SEED = 42


def _emb(model, x):
    return model.forward_head(model.forward_features(x), pre_logits=True)


def _audio_windows(path, n, k=12):
    total = sf.info(path).frames
    if total <= n:
        y, _ = sf.read(path, dtype="float32", always_2d=False)
        y = y.mean(1) if y.ndim > 1 else y
        return np.tile(y, int(np.ceil(n / max(len(y), 1))))[:n][None, :]
    starts = np.linspace(0, total - n, num=min(k, total // n), dtype=int)
    out = []
    for s in starts:
        y, _ = sf.read(path, start=int(s), frames=n, dtype="float32", always_2d=False)
        y = y.mean(1) if y.ndim > 1 else y
        out.append(np.pad(y, (0, max(0, n - len(y))))[:n])
    return np.stack(out)


def extract_embeddings():
    cache = data_path("interim", "fusion_embeddings.npz")
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return (z["ae"], z["am"], z["ie"], z["im"])

    melf = torchaudio.transforms.MelSpectrogram(SR, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS)
    todb = torchaudio.transforms.AmplitudeToDB(top_db=80)
    al = json.loads((PROJECT_ROOT / "models/audio_efficientnet_b0_labels.json").read_text())
    il = json.loads((PROJECT_ROOT / "models/image_efficientnet_b3_labels.json").read_text())
    am = build_audio_model("efficientnet_b0", num_classes=len(al), pretrained=False)
    am.load_state_dict(torch.load(PROJECT_ROOT / "models/audio_efficientnet_b0.pt", map_location=DEVICE))
    am.eval().to(DEVICE)
    im = build_image_model("efficientnet_b3", num_classes=len(il), pretrained=False)
    im.load_state_dict(torch.load(PROJECT_ROOT / "models/image_efficientnet_b3.pt", map_location=DEVICE))
    im.eval().to(DEVICE)

    # audio (embedding agregado por grabación)
    am_manifest = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    ae, ameta = [], []
    print(f"Extrayendo embeddings de audio ({len(am_manifest)} grabaciones)...")
    with torch.no_grad():
        for r in am_manifest.itertuples(index=False):
            wins = _audio_windows(str(PROJECT_ROOT / r.cached_file), CLIP_SAMPLES)
            mel = todb(melf(torch.from_numpy(wins)))
            mel = (mel - mel.mean((1, 2), keepdim=True)) / (mel.std((1, 2), keepdim=True) + 1e-6)
            e = _emb(am, mel.unsqueeze(1).to(DEVICE)).mean(0).cpu().numpy()
            ae.append(e); ameta.append((r.scientific_name, r.split))

    # imagen (embedding por foto, en lotes)
    im_manifest = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    tf = _eval_tf()
    ie, imeta = [], []
    print(f"Extrayendo embeddings de imagen ({len(im_manifest)} fotos)...")
    with torch.no_grad():
        for i in range(0, len(im_manifest), 32):
            chunk = im_manifest.iloc[i:i + 32]
            xs, ok = [], []
            for r in chunk.itertuples(index=False):
                try:
                    xs.append(tf(Image.open(PROJECT_ROOT / r.file).convert("RGB")))
                    ok.append((r.scientific_name, r.split))
                except Exception:  # noqa: BLE001
                    pass
            if not xs:
                continue
            e = _emb(im, torch.stack(xs).to(DEVICE)).cpu().numpy()
            ie.extend(e); imeta.extend(ok)

    ae, ie = np.array(ae, dtype=np.float32), np.array(ie, dtype=np.float32)
    am_arr, im_arr = np.array(ameta, dtype=object), np.array(imeta, dtype=object)
    np.savez_compressed(cache, ae=ae, am=am_arr, ie=ie, im=im_arr)
    return ae, am_arr, ie, im_arr


def build_pairs(ae, am, ie, im):
    """Empareja audio+imagen de la misma especie por split. Devuelve tensores por split."""
    a_by, i_by = defaultdict(list), defaultdict(list)
    for e, (sp, sw) in zip(ae, am):
        a_by[(sp, sw)].append(e)
    for e, (sp, sw) in zip(ie, im):
        i_by[(sp, sw)].append(e)
    species = sorted({sp for sp, _ in i_by} & {sp for sp, _ in a_by})
    sp2i = {s: k for k, s in enumerate(species)}
    rng = np.random.default_rng(SEED)

    def make(split, k):
        A, I, Y = [], [], []
        for sp in species:
            al, il = a_by.get((sp, split), []), i_by.get((sp, split), [])
            if not al or not il:
                continue
            for _ in range(k):
                A.append(al[rng.integers(len(al))]); I.append(il[rng.integers(len(il))])
                Y.append(sp2i[sp])
        return np.array(A, np.float32), np.array(I, np.float32), np.array(Y)

    return make("train", PAIRS_TRAIN), make("test", PAIRS_TEST), len(species)


def train_eval(model, tr, te, n_classes, late=False):
    Atr, Itr, Ytr = tr
    Ate, Ite, Yte = te
    model = model.to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    ce = nn.CrossEntropyLoss()
    dl = DataLoader(TensorDataset(torch.tensor(Atr, dtype=torch.float32),
                                  torch.tensor(Itr, dtype=torch.float32),
                                  torch.tensor(Ytr, dtype=torch.long)),
                    batch_size=BATCH, shuffle=True)
    model.train()
    for _ in range(EPOCHS):
        for a, i, y in dl:
            a, i, y = a.to(DEVICE), i.to(DEVICE), y.to(DEVICE)
            opt.zero_grad()
            if late:
                la, li = model(a, i)
                loss = ce(la, y) + ce(li, y)
            else:
                loss = ce(model(a, i), y)
            loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        a = torch.tensor(Ate, dtype=torch.float32).to(DEVICE)
        i = torch.tensor(Ite, dtype=torch.float32).to(DEVICE)
        if late:
            la, li = model(a, i)
            prob = 0.5 * (la.softmax(1) + li.softmax(1))
        else:
            prob = model(a, i).softmax(1)
        pred = prob.argmax(1).cpu().numpy()
    return accuracy_score(Yte, pred), f1_score(Yte, pred, average="macro", zero_division=0)


def main() -> None:
    ae, am, ie, im = extract_embeddings()
    (Atr, Itr, Ytr), te, n_cls = build_pairs(ae, am, ie, im)

    # estandarizar cada modalidad (fit en train)
    sa, si = StandardScaler().fit(Atr), StandardScaler().fit(Itr)
    tr = (sa.transform(Atr), si.transform(Itr), Ytr)
    te = (sa.transform(te[0]), si.transform(te[1]), te[2])
    da, di = Atr.shape[1], Itr.shape[1]

    print(f"\nEspecies pareadas: {n_cls} | train {len(Ytr)} pares, test {len(te[2])} pares | {DEVICE}")
    print(f"Config idéntica: {EPOCHS} épocas, lr={LR}, batch={BATCH}\n")

    torch.manual_seed(SEED)
    results = [
        ("Late fusion (promedio de decisiones)", *train_eval(LateFusion(da, di, n_cls), tr, te, n_cls, late=True)),
        ("Early fusion (concatenación temprana)", *train_eval(EarlyFusion(da, di, n_cls), tr, te, n_cls)),
        ("Intermediate fusion (proyección + fusión)", *train_eval(IntermediateFusion(da, di, n_cls), tr, te, n_cls)),
    ]
    out = PROJECT_ROOT / "reports" / "fusion_strategies.csv"
    pd.DataFrame([(n, a, f) for n, a, f in results],
                 columns=["estrategia", "accuracy", "f1_macro"]).to_csv(out, index=False)

    print("== Comparación de estrategias de fusión (test) ==")
    print(f"  {'estrategia':<44}{'accuracy':>10}{'F1 macro':>10}")
    for name, acc, f1 in results:
        print(f"  {name:<44}{acc:>10.3f}{f1:>10.3f}")
    print(f"\n  → {out.relative_to(PROJECT_ROOT)}\n")


if __name__ == "__main__":
    main()
