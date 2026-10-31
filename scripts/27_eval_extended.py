"""Evaluación extendida del Módulo 1 (audio CNN + AST, imagen B3) para el informe final.

Guarda las PROBABILIDADES por muestra de test (reports/probs/*.npz) y calcula a partir
de ellas: accuracy, F1 macro, AUC-ROC macro (uno contra el resto), top-5, matriz de
confusión, desempeño por grupo de conservación, desempeño por origen de la grabación
(Chile vs. otros países: desplazamiento de dominio), latencia por muestra y número de
parámetros. Los .npz alimentan después el análisis de fusión (script 31) sin volver a
correr los modelos.

    uv run python scripts/27_eval_extended.py              # audio CNN + AST + imagen
    uv run python scripts/27_eval_extended.py --skip-ast   # sin el AST (el más lento)

Salidas (reports/): probs/*.npz, eval_extended.json, eval_extended.csv, eval_by_group.csv,
eval_audio_origin.csv, confusion_*.csv, audio_ast_per_class.csv,
image_efficientnet_b3_tta_per_class.csv.

NOTA: inferencia sobre los tres conjuntos de test en MPS (≈10-20 min) → la corre el usuario.
Reanudable: si un .npz ya existe, no vuelve a correr ese modelo.
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
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import CONFIG, PROJECT_ROOT, data_path  # noqa: E402
from biomonitor.audio.dataset import CLIP_SAMPLES, HOP, N_FFT, N_MELS, SR, build_label_map  # noqa: E402
from biomonitor.audio.model import build_audio_model  # noqa: E402
from biomonitor.evaluation import (  # noqa: E402
    LogMel, by_group, confusion, count_params, per_class_table, recording_windows,
    recording_windows_16k, species_groups, summarize,
)
from biomonitor.vision.dataset import IMAGENET_MEAN, IMAGENET_STD, SIZE  # noqa: E402
from biomonitor.vision.model import build_image_model  # noqa: E402

REP = PROJECT_ROOT / "reports"
PROBS = REP / "probs"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--skip-ast", action="store_true")
    p.add_argument("--max-windows", type=int, default=12, help="ventanas por grabación (CNN)")
    p.add_argument("--max-windows-ast", type=int, default=CONFIG["train_ast"]["max_windows_eval"])
    p.add_argument("--limit", type=int, default=None, help="smoke test: solo N muestras, sin guardar")
    return p.parse_args()


def _sync() -> None:
    if DEVICE.type == "mps":
        torch.mps.synchronize()


# ------------------------------- audio CNN ---------------------------------
@torch.no_grad()
def run_audio_cnn(test: pd.DataFrame, label_map: dict, max_windows: int, save: bool = True) -> dict:
    out = PROBS / "audio_efficientnet_b0_test.npz"
    if out.exists() and save:
        print("  (ya existe, se reutiliza)")
        z = np.load(out, allow_pickle=True)
        return dict(z)
    model = build_audio_model("efficientnet_b0", num_classes=len(label_map), pretrained=False)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models/audio_efficientnet_b0.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    logmel = LogMel(SR, N_FFT, HOP, N_MELS)

    agg, center, y, lat = [], [], [], []
    for i, r in enumerate(test.itertuples(index=False)):
        t0 = time.perf_counter()
        wins = recording_windows(str(PROJECT_ROOT / r.cached_file), CLIP_SAMPLES, max_windows)
        probs = model(logmel(torch.from_numpy(wins)).to(DEVICE)).softmax(1)
        _sync()
        lat.append(time.perf_counter() - t0)
        probs = probs.cpu().numpy()
        agg.append(probs.mean(0)); center.append(probs[len(wins) // 2]); y.append(label_map[r.scientific_name])
        if (i + 1) % 100 == 0:
            print(f"    {i + 1}/{len(test)}")
    res = dict(probs_agg=np.array(agg, dtype=np.float32), probs_center=np.array(center, dtype=np.float32),
               y_true=np.array(y), ids=test["xc_id"].astype(str).values, country=test["country"].values,
               latency_s=np.array(lat), n_params=count_params(model), classes=np.array(sorted(label_map)))
    if save:
        np.savez_compressed(out, **res)
    return res


# -------------------------------- audio AST --------------------------------
@torch.no_grad()
def run_audio_ast(test: pd.DataFrame, label_map: dict, max_windows: int, save: bool = True) -> dict:
    out = PROBS / "audio_ast_test.npz"
    if out.exists() and save:
        print("  (ya existe, se reutiliza)")
        return dict(np.load(out, allow_pickle=True))
    from biomonitor.audio.ast_dataset import AST_SR, SRC_WINDOW, feature_extractor
    from biomonitor.audio.ast_model import build_ast_model

    model = build_ast_model(len(label_map))
    model.load_state_dict(torch.load(PROJECT_ROOT / "models/audio_ast.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    fe = feature_extractor()

    agg, center, y, lat = [], [], [], []
    for i, r in enumerate(test.itertuples(index=False)):
        t0 = time.perf_counter()
        wins = recording_windows_16k(str(PROJECT_ROOT / r.cached_file), SRC_WINDOW, max_windows, SR, AST_SR)
        feats = fe(wins, sampling_rate=AST_SR, return_tensors="pt")["input_values"].to(DEVICE)
        probs = model(input_values=feats).logits.softmax(1)
        _sync()
        lat.append(time.perf_counter() - t0)
        probs = probs.cpu().numpy()
        agg.append(probs.mean(0)); center.append(probs[len(wins) // 2]); y.append(label_map[r.scientific_name])
        if (i + 1) % 50 == 0:
            print(f"    {i + 1}/{len(test)}")
    res = dict(probs_agg=np.array(agg, dtype=np.float32), probs_center=np.array(center, dtype=np.float32),
               y_true=np.array(y), ids=test["xc_id"].astype(str).values, country=test["country"].values,
               latency_s=np.array(lat), n_params=count_params(model), classes=np.array(sorted(label_map)))
    if save:
        np.savez_compressed(out, **res)
    return res


# --------------------------------- imagen ----------------------------------
@torch.no_grad()
def run_image(test: pd.DataFrame, label_map: dict, save: bool = True) -> dict:
    out = PROBS / "image_efficientnet_b3_test.npz"
    if out.exists() and save:
        print("  (ya existe, se reutiliza)")
        return dict(np.load(out, allow_pickle=True))
    model = build_image_model("efficientnet_b3", num_classes=len(label_map), pretrained=False)
    model.load_state_dict(torch.load(PROJECT_ROOT / "models/image_efficientnet_b3.pt", map_location=DEVICE))
    model.eval().to(DEVICE)
    resize = transforms.Resize(int(SIZE * 1.15))
    tencrop, ccrop = transforms.TenCrop(SIZE), transforms.CenterCrop(SIZE)
    norm = transforms.Compose([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

    tta, center, y, ids, country, lat_c, lat_t = [], [], [], [], [], [], []
    for i, r in enumerate(test.itertuples(index=False)):
        try:
            img = Image.open(PROJECT_ROOT / r.file).convert("RGB")
        except Exception:  # noqa: BLE001
            continue
        t0 = time.perf_counter()
        rs = resize(img)
        p_c = model(norm(ccrop(rs)).unsqueeze(0).to(DEVICE)).softmax(1)[0]
        _sync(); t1 = time.perf_counter()
        p_t = model(torch.stack([norm(c) for c in tencrop(rs)]).to(DEVICE)).softmax(1).mean(0)
        _sync(); t2 = time.perf_counter()
        lat_c.append(t1 - t0); lat_t.append(t2 - t1)
        center.append(p_c.cpu().numpy()); tta.append(p_t.cpu().numpy())
        y.append(label_map[r.scientific_name]); ids.append(str(r.gbif_id)); country.append(r.country)
        if (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(test)}")
    res = dict(probs_agg=np.array(tta, dtype=np.float32), probs_center=np.array(center, dtype=np.float32),
               y_true=np.array(y), ids=np.array(ids), country=np.array(country, dtype=object),
               latency_s=np.array(lat_t), latency_center_s=np.array(lat_c),
               n_params=count_params(model), classes=np.array(sorted(label_map)))
    if save:
        np.savez_compressed(out, **res)
    return res


# --------------------------------- reporte ---------------------------------
def report(name: str, res: dict, groups: dict, summary: dict, rows: list, grp_rows: list,
           origin_rows: list, per_class_csv: str | None, save: bool = True) -> None:
    classes = list(res["classes"])
    y = res["y_true"]
    for proto, key in [("agregado", "probs_agg"), ("ventana_central", "probs_center")]:
        s = summarize(y, res[key])
        s.update(model=name, protocol=proto)
        rows.append(s)
    probs = res["probs_agg"]
    y_pred = probs.argmax(1)
    pc = per_class_table(y, y_pred, classes)
    if per_class_csv and save:
        pc.to_csv(REP / per_class_csv)
    g = by_group(pc, groups); g.insert(0, "model", name); grp_rows.extend(g.to_dict("records"))
    if save:
        confusion(y, y_pred, classes).to_csv(REP / f"confusion_{name}.csv")
    lat = res["latency_s"] if len(res["latency_s"]) > 6 else np.concatenate([np.zeros(5), res["latency_s"]])
    summary[name] = {
        **summarize(y, probs), "n_params": int(res["n_params"]),
        "latency_ms_median": float(np.median(lat[5:]) * 1000), "latency_ms_mean": float(np.mean(lat[5:]) * 1000),
    }
    if "latency_center_s" in res:
        summary[name]["latency_center_ms_median"] = float(np.median(res["latency_center_s"][5:]) * 1000)
    # origen de la grabación (solo audio): Chile vs. otros países
    if name.startswith("audio"):
        country = np.array(res["country"], dtype=object)
        for label, mask in [("Chile", country == "Chile"), ("Otros países", country != "Chile")]:
            if mask.sum() == 0:
                continue
            s = summarize(y[mask], probs[mask]); s.update(model=name, origen=label)
            origin_rows.append(s)


def main() -> None:
    args = get_args()
    PROBS.mkdir(parents=True, exist_ok=True)
    print(f"Dispositivo: {DEVICE}\n")
    groups = species_groups()

    # audio: manifiesto con país (interim) + rutas cacheadas (processed)
    a_cache = pd.read_csv(data_path("processed", "audio32_manifest.csv"))
    a_meta = pd.read_csv(data_path("interim", "audio_manifest_split.csv"))[["xc_id", "country"]]
    a_test = a_cache[a_cache["split"] == "test"].merge(a_meta, on="xc_id", how="left").reset_index(drop=True)
    a_test["country"] = a_test["country"].fillna("desconocido")
    a_species = pd.read_csv(data_path("external", "species_final.csv"))["scientific_name"].tolist()
    a_lm = build_label_map(a_species)

    i_manifest = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    i_test = i_manifest[i_manifest["split"] == "test"].reset_index(drop=True)
    i_lm = build_label_map(sorted(i_manifest["scientific_name"].unique()))

    save = args.limit is None
    if not save:
        a_test, i_test = a_test.head(args.limit), i_test.head(args.limit)
        print(f"SMOKE TEST: {args.limit} muestras por modelo, no se guarda nada\n")
    summary, rows, grp_rows, origin_rows = {}, [], [], []

    print(f"[1/3] Audio CNN (EfficientNet-B0) — {len(a_test)} grabaciones de test")
    res = run_audio_cnn(a_test, a_lm, args.max_windows, save)
    report("audio_cnn", res, groups, summary, rows, grp_rows, origin_rows, None, save)

    if not args.skip_ast:
        print(f"[2/3] Audio AST — {len(a_test)} grabaciones de test (lento)")
        res = run_audio_ast(a_test, a_lm, args.max_windows_ast, save)
        report("audio_ast", res, groups, summary, rows, grp_rows, origin_rows, "audio_ast_per_class.csv", save)
    else:
        print("[2/3] AST omitido (--skip-ast)")

    print(f"[3/3] Imagen EfficientNet-B3 (TTA 10-crop) — {len(i_test)} fotos de test")
    res = run_image(i_test, i_lm, save)
    report("image_b3", res, groups, summary, rows, grp_rows, origin_rows, "image_efficientnet_b3_tta_per_class.csv", save)

    if save:
        pd.DataFrame(rows).to_csv(REP / "eval_extended.csv", index=False)
        pd.DataFrame(grp_rows).to_csv(REP / "eval_by_group.csv", index=False)
        pd.DataFrame(origin_rows).to_csv(REP / "eval_audio_origin.csv", index=False)
        (REP / "eval_extended.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("\n== Resumen (protocolo operacional: agregado por grabación / TTA) ==")
    print(f"  {'modelo':<12}{'acc':>8}{'F1':>8}{'AUC':>8}{'top5':>8}{'params':>10}{'ms/muestra':>12}")
    for m, s in summary.items():
        print(f"  {m:<12}{s['accuracy']:>8.3f}{s['f1_macro']:>8.3f}{s['auc_macro']:>8.3f}"
              f"{s['top5']:>8.3f}{s['n_params']/1e6:>9.1f}M{s['latency_ms_median']:>12.0f}")
    print("\n== Audio por origen de la grabación ==")
    for r in origin_rows:
        print(f"  {r['model']:<12}{r['origen']:<14} n={r['n']:<5} acc {r['accuracy']:.3f}  F1 {r['f1_macro']:.3f}")
    print("\n== Por grupo de conservación (F1 medio) ==")
    for r in grp_rows:
        print(f"  {r['model']:<12}{r['grupo']:<11} esp={r['especies']:<3} F1 {r['f1_medio']:.3f}  acc {r['accuracy']:.3f}")
    print(f"\n  → reports/eval_extended.{{json,csv}}, eval_by_group.csv, eval_audio_origin.csv, "
          f"confusion_*.csv, probs/*.npz\n")


if __name__ == "__main__":
    main()
