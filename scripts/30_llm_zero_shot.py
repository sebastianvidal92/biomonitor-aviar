"""Baseline de IA generativa multimodal de propósito general (OE4): zero-shot y few-shot.

Consulta un modelo de lenguaje multimodal (Claude, API de Anthropic) con cada fotografía
del conjunto de prueba de imagen y la lista cerrada de especies candidatas, con el mismo
protocolo de métricas que el clasificador propio (accuracy, F1 macro, por grupo de
conservación). Dos protocolos:
  - zero-shot: la foto + la lista de 53 especies (nombre científico y común).
  - few-shot:  además, una foto de referencia por especie (del conjunto de entrenamiento)
               en el prompt de sistema, servido desde la caché de prompts.

    uv run python scripts/30_llm_zero_shot.py --dry-run          # arma los prompts, sin API
    uv run python scripts/30_llm_zero_shot.py --limit 20         # prueba corta (≈US$0.1)
    uv run python scripts/30_llm_zero_shot.py                    # zero-shot, 1.040 fotos
    uv run python scripts/30_llm_zero_shot.py --few-shot         # few-shot (caché de prompts)

Requiere ANTHROPIC_API_KEY en .env. Reanudable: cada respuesta se anota en un JSONL y las
fotos ya consultadas se saltan. Imprime el costo acumulado (tokens × precio de lista).
Salidas: reports/llm/<modo>_<modelo>.jsonl, reports/llm_baseline.csv (resumen),
reports/llm_<modo>_per_class.csv.
"""

from __future__ import annotations

import argparse
import base64
import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from biomonitor.config import PROJECT_ROOT, data_path, get_env  # noqa: E402
from biomonitor.evaluation import by_group, per_class_table, species_groups  # noqa: E402

REP = PROJECT_ROOT / "reports"
DEFAULT_MODEL = "claude-opus-5-5"
# US$ por millón de tokens (lista de precios de la API, oct. 2026): entrada, salida,
# lectura de caché, escritura de caché.
PRICES = {
    "claude-opus-5-5": (4.0, 20.0, 0.20, 5.0),
    "claude-sonnet-5-5": (2.0, 10.0, 0.20, 2.5),
    "claude-haiku-5-5": (0.10, 0.50, 0.01, 0.125),
}
SEED = 42
REF_MAX_SIDE = 320  # lado mayor de las fotos de referencia (few-shot)


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--few-shot", action="store_true", help="una foto de referencia por especie")
    p.add_argument("--effort", default="low", choices=["low", "medium", "high"],
                   help="esfuerzo de razonamiento (los tokens de razonamiento se cobran como salida)")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--limit", type=int, default=None, help="solo N fotos de test (prueba)")
    p.add_argument("--dry-run", action="store_true", help="no llama a la API")
    return p.parse_args()


# ------------------------------- datos ---------------------------------------
def common_names() -> dict[str, str]:
    """Nombre común de cada especie, desde la tabla del Apéndice A de la tesis."""
    tex = PROJECT_ROOT / "tesis" / "anexoA.tex"
    out = {}
    if tex.exists():
        for m in re.finditer(r"\\textit\{([^}]+)\}\s*&\s*([^&]+?)\s*&", tex.read_text(encoding="utf-8")):
            out[m.group(1).strip()] = m.group(2).strip()
    sp = pd.read_csv(data_path("external", "species_final.csv"))
    for r in sp.itertuples(index=False):
        if r.scientific_name not in out and isinstance(r.common_name, str):
            out[r.scientific_name] = r.common_name.split(",")[0].strip()
    return out


def image_block(path: Path, max_side: int | None = None) -> dict:
    """Bloque de imagen base64. Se recodifica a JPEG con PIL porque algunos archivos .jpg
    de GBIF son en realidad PNG (la API rechaza el tipo declarado); `max_side` reduce las
    fotos de referencia del few-shot para abaratar tokens y subida."""
    import io
    from PIL import Image

    img = Image.open(path).convert("RGB")
    if max_side and max(img.size) > max_side:
        img.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=88)
    data = base64.standard_b64encode(buf.getvalue()).decode("utf-8")
    return {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": data}}


def build_system(species: list[str], names: dict[str, str]) -> list[dict]:
    listado = "\n".join(f"- {s}" + (f" ({names[s]})" if s in names else "") for s in species)
    intro = (
        "Eres un ornitólogo experto en la avifauna de Chile. Tu tarea es identificar la especie "
        "del ave que aparece en una fotografía, eligiendo exactamente una de las especies candidatas "
        "de la lista siguiente (todas son aves de Chile continental). Responde siempre con el nombre "
        "científico tal como aparece en la lista, aunque la foto sea difícil; si dudas, elige la "
        "especie más probable e indica una confianza baja. Responde en el formato JSON solicitado.\n\n"
        "Especies candidatas:\n" + listado
    )
    return [{"type": "text", "text": intro, "cache_control": {"type": "ephemeral"}}]


def build_examples(examples: list[tuple[str, Path]], names: dict[str, str]) -> list[dict]:
    """Turnos previos con una foto de referencia por especie (few-shot). La API solo admite
    texto en el prompt de sistema, así que las imágenes van en un primer turno de usuario,
    cacheado, seguido de un acuse del asistente."""
    content: list[dict] = [{"type": "text", "text": "Antes de la consulta, estas son fotografías de referencia "
                                                     "(conjunto de entrenamiento), una por especie, con su nombre científico:"}]
    for sp, path in examples:
        content.append(image_block(path, max_side=REF_MAX_SIDE))
        content.append({"type": "text", "text": f"Referencia: {sp}" + (f" ({names[sp]})" if sp in names else "")})
    content.append({"type": "text", "text": "Fin de las referencias. A continuación te enviaré la fotografía a identificar.",
                    "cache_control": {"type": "ephemeral"}})
    return [{"role": "user", "content": content},
            {"role": "assistant", "content": "Entendido. Identificaré la especie de la siguiente fotografía usando "
                                             "la lista de candidatas y las referencias."}]


def output_format(species: list[str]) -> dict:
    return {"format": {"type": "json_schema", "schema": {
        "type": "object",
        "properties": {
            "species": {"type": "string", "enum": species,
                        "description": "nombre científico elegido, exactamente como en la lista"},
            "confidence": {"type": "number", "description": "confianza entre 0 y 1"},
        },
        "required": ["species", "confidence"], "additionalProperties": False}}}


# ------------------------------- consulta ------------------------------------
def classify(client, model: str, effort: str, system: list[dict], history: list[dict], fmt: dict, path: Path) -> dict:
    import anthropic

    t0 = time.perf_counter()
    try:
        resp = client.messages.create(
            model=model, max_tokens=4096, system=system,  # holgura para el razonamiento adaptativo
            output_config={"effort": effort, **fmt},
            messages=history + [{"role": "user", "content": [
                image_block(path),
                {"type": "text", "text": "¿Qué especie de la lista es el ave de esta fotografía?"},
            ]}],
        )
    except anthropic.BadRequestError as e:  # p. ej. imagen rechazada: se registra como inválida
        return {"pred": None, "confidence": None, "error": f"bad_request: {e.message}", "latency_s": time.perf_counter() - t0}
    except (anthropic.APIConnectionError, anthropic.APIStatusError) as e:  # red o servidor: se reintenta en la próxima corrida
        return {"pred": None, "confidence": None, "error": f"conexion: {type(e).__name__}", "latency_s": time.perf_counter() - t0}
    out = {"latency_s": time.perf_counter() - t0, "stop_reason": resp.stop_reason,
           "usage": {"input": resp.usage.input_tokens, "output": resp.usage.output_tokens,
                     "cache_read": resp.usage.cache_read_input_tokens or 0,
                     "cache_write": resp.usage.cache_creation_input_tokens or 0}}
    if resp.stop_reason == "refusal":
        out.update(pred=None, confidence=None, error="refusal")
        return out
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        data = json.loads(text)
        out.update(pred=data.get("species"), confidence=data.get("confidence"))
    except json.JSONDecodeError:
        out.update(pred=None, confidence=None, error=f"json: {text[:120]}")
    return out


def cost_usd(model: str, usage: dict) -> float:
    pin, pout, pcache, pwrite = PRICES.get(model, PRICES[DEFAULT_MODEL])
    return (usage["input"] * pin + usage["output"] * pout + usage["cache_read"] * pcache
            + usage["cache_write"] * pwrite) / 1e6


def main() -> None:
    args = get_args()
    mode = "fewshot" if args.few_shot else "zeroshot"
    i_man = pd.read_csv(data_path("interim", "image_manifest_split.csv"))
    species = sorted(i_man["scientific_name"].unique())
    names = common_names()
    test = i_man[i_man["split"] == "test"].reset_index(drop=True)
    if args.limit:
        test = test.sample(min(args.limit, len(test)), random_state=SEED).reset_index(drop=True)

    system = build_system(species, names)
    history: list[dict] = []
    if args.few_shot:
        train = i_man[i_man["split"] == "train"]
        examples = [(sp, PROJECT_ROOT / g.sample(1, random_state=SEED)["file"].iloc[0])
                    for sp, g in train.groupby("scientific_name")]
        history = build_examples(examples, names)
    fmt = output_format(species)
    n_ref = sum(1 for m in history for b in (m["content"] if isinstance(m["content"], list) else []) if b["type"] == "image")
    ref_bytes = sum(len(b["source"]["data"]) for m in history for b in (m["content"] if isinstance(m["content"], list) else [])
                    if b["type"] == "image") * 3 // 4
    sys_chars = len(system[0]["text"])
    print(f"Modo: {mode} | modelo: {args.model} | esfuerzo: {args.effort} | fotos de test: {len(test)}")
    print(f"Prompt cacheado: {sys_chars:,} caracteres de instrucciones + {n_ref} fotos de referencia "
          f"({ref_bytes / 1e6:.1f} MB por consulta; ≈{sys_chars // 4 + n_ref * 95:,} tokens, servidos desde la caché)")
    pin = PRICES.get(args.model, PRICES[DEFAULT_MODEL])
    est = len(test) * (300 * pin[0] + (sys_chars // 4 + n_ref * 95) * pin[2] + 40 * pin[1]) / 1e6
    print(f"Costo estimado: ≈US${est:.2f} (más el razonamiento, que depende del esfuerzo)")
    if args.dry_run:
        print("\n--dry-run: no se llama a la API. Primer bloque del sistema:\n")
        print(system[0]["text"][:900] + "\n...")
        return

    import anthropic
    get_env("ANTHROPIC_API_KEY")  # error claro si falta
    client = anthropic.Anthropic(max_retries=5, timeout=180.0)

    out_dir = REP / "llm"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{mode}_{args.model}" + ("_prueba" if args.limit else "")
    jsonl = out_dir / f"{tag}.jsonl"
    done: dict[str, dict] = {}
    if jsonl.exists():  # reanudación: las filas con error (salvo rechazos del modelo) se reintentan
        for line in jsonl.read_text().splitlines():
            r = json.loads(line)
            if r.get("error") and r["error"] != "refusal":
                continue
            done[r["gbif_id"]] = r
    todo = test[~test["gbif_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"Ya consultadas: {len(done)} | pendientes: {len(todo)}\n")

    lock = threading.Lock()
    total_cost, n_done = 0.0, 0
    for r in done.values():
        if "usage" in r:
            total_cost += cost_usd(args.model, r["usage"])

    def work(row):
        res = classify(client, args.model, args.effort, system, history, fmt, PROJECT_ROOT / row.file)
        res.update(gbif_id=str(row.gbif_id), true=row.scientific_name, file=row.file)
        return res

    n_err = 0
    ex = ThreadPoolExecutor(max_workers=args.workers)
    try:
        with jsonl.open("a") as fh:
            rows_iter = list(todo.itertuples(index=False))
            if rows_iter and args.few_shot:  # la primera consulta escribe la caché; las demás la reutilizan
                first = work(rows_iter.pop(0))
                fh.write(json.dumps(first, ensure_ascii=False) + "\n"); fh.flush()
                done[first["gbif_id"]] = first; n_done += 1
                if "usage" in first:
                    total_cost += cost_usd(args.model, first["usage"])
                    print(f"  caché creada: {first['usage']['cache_write']:,} tokens escritos")
                elif first.get("error"):
                    raise SystemExit(f"La primera consulta falló: {first['error']}")
            futures = [ex.submit(work, row) for row in rows_iter]
            for fut in as_completed(futures):
                res = fut.result()
                with lock:
                    fh.write(json.dumps(res, ensure_ascii=False) + "\n"); fh.flush()
                    done[res["gbif_id"]] = res
                    n_done += 1
                    if "usage" in res:
                        total_cost += cost_usd(args.model, res["usage"])
                    if res.get("error"):
                        n_err += 1
                    if n_done % 25 == 0 or n_done == len(todo):
                        acc = np.mean([d.get("pred") == d["true"] for d in done.values()])
                        print(f"  {n_done}/{len(todo)} | accuracy parcial {acc:.3f} | errores {n_err} | "
                              f"costo acumulado US${total_cost:.2f}")
    except KeyboardInterrupt:
        print("\nInterrumpido: lo ya consultado quedó guardado; vuelve a correr para continuar.")
        ex.shutdown(wait=False, cancel_futures=True)
        return
    ex.shutdown(wait=True)
    if n_err:
        print(f"\n  {n_err} consultas con error (red o imagen); vuelve a correr el script para reintentarlas.")

    # ----------------------------- métricas -----------------------------
    rows = [done[g] for g in test["gbif_id"].astype(str) if g in done and not (done[g].get("error") or "").startswith("conexion")]
    y_true = np.array([species.index(r["true"]) for r in rows])
    y_pred = np.array([species.index(r["pred"]) if r.get("pred") in species else -1 for r in rows])
    invalid = int((y_pred < 0).sum())
    y_pred_eval = np.where(y_pred < 0, (y_true + 1) % len(species), y_pred)  # inválida = error
    acc = accuracy_score(y_true, y_pred_eval)
    f1 = f1_score(y_true, y_pred_eval, average="macro", labels=range(len(species)), zero_division=0)
    pc = per_class_table(y_true, y_pred_eval, species)
    grp = by_group(pc, species_groups())
    usage = {k: sum(r.get("usage", {}).get(k, 0) for r in rows) for k in ["input", "output", "cache_read", "cache_write"]}
    summary = {"modo": mode, "modelo": args.model, "esfuerzo": args.effort, "n": len(rows),
               "accuracy": float(acc), "f1_macro": float(f1), "invalidas": invalid,
               "latencia_s_mediana": float(np.median([r["latency_s"] for r in rows])),
               "costo_usd": round(cost_usd(args.model, usage), 2), **{f"tokens_{k}": v for k, v in usage.items()},
               **{f"f1_{g['grupo']}": round(g["f1_medio"], 3) for g in grp.to_dict("records")}}
    print(f"\n== {mode} · {args.model} · {len(rows)} fotos ==")
    print(f"  accuracy {acc:.3f} | F1 macro {f1:.3f} | respuestas inválidas/rechazos {invalid} | costo US${summary['costo_usd']}")
    for g in grp.to_dict("records"):
        print(f"  {g['grupo']:<10} esp={g['especies']:<3} F1 {g['f1_medio']:.3f}  acc {g['accuracy']:.3f}")
    if not args.limit:
        pc.to_csv(REP / f"llm_{mode}_per_class.csv")
        csv = REP / "llm_baseline.csv"
        prev = pd.read_csv(csv) if csv.exists() else pd.DataFrame()
        prev = prev[~((prev.get("modo") == mode) & (prev.get("modelo") == args.model))] if len(prev) else prev
        pd.concat([prev, pd.DataFrame([summary])], ignore_index=True).to_csv(csv, index=False)
        print(f"\n  → {csv.relative_to(PROJECT_ROOT)}, reports/llm_{mode}_per_class.csv, {jsonl.relative_to(PROJECT_ROOT)}\n")
    print("Referencia (clasificador propio, EfficientNet-B3 + TTA): accuracy 0,848 · F1 0,808")


if __name__ == "__main__":
    main()
