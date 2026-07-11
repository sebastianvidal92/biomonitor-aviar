# Módulo 1 — Clasificador de imagen

Documentación técnica del clasificador de especies a partir de fotografías. Cubre
datos, preprocesamiento, modelos, hiperparámetros, entrenamientos (época a época),
resultados y la **decisión de alcance** sobre la meta de accuracy.

Última actualización: 2026-07-05.

---

## Índice
1. [Datos](#1-datos)
2. [Partición](#2-partición)
3. [Preprocesamiento](#3-preprocesamiento)
4. [Modelos e hiperparámetros](#4-modelos-e-hiperparámetros)
5. [Resultados](#5-resultados)
6. [Registro de entrenamiento](#6-registro-de-entrenamiento)
7. [Análisis de errores](#7-análisis-de-errores)
8. [Decisión de alcance](#8-decisión-de-alcance)
9. [Reproducibilidad](#9-reproducibilidad)

---

## 1. Datos

**Fuente:** fotos de **iNaturalist Research-grade** (ID validado por la comunidad,
geolocalizadas, licencia Creative Commons) descargadas vía **GBIF** (`scripts/10`).

| Parámetro | Valor |
|---|---|
| Fotos totales | **7.068** (55 especies) |
| Tamaño (medium, ~500 px) | 1.0 GB |
| Máx. por especie | 150 |
| Fuente | 7.061 iNaturalist RG + 7 otros GBIF (fallback) |
| Geolocalizadas | 7.062 / 7.068 (reutilizables para el SDM) |

**Robustez de la descarga** (decisiones registradas en [DECISIONES.md](DECISIONES.md)):
- **Alias de taxonomía:** iNaturalist reclasifica o usa otra grafía → sin alias daban 0
  fotos. *Elaenia chilensis* → *E. albiceps* (150); *Sylviorthorhynchus desmurii* →
  *desmursii* (91).
- **Fallback filtrado:** para especies con pocas fotos en iNat RG, se amplía a todo
  GBIF pero solo **fotos de aves vivas** (`basisOfRecord=HUMAN_OBSERVATION`, JPEG,
  licencia CC), **excluyendo espectrogramas de Xeno-canto y especímenes de museo**
  (Arctos, cvcoll — lentos/forbidden).

**Especies escasas** (mismo patrón de escasez de datos que en audio): *Agriornis
albicauda* (15), *Rallus antarcticus* (6), *Charadrius nivosus* (2 — casi sin fotos de
aves vivas; el resto en GBIF eran especímenes de museo).

---

## 2. Partición

`scripts/11_split_images.py`. Cada foto es una observación independiente, así que la
partición es a nivel de foto (sin el problema de leakage del audio). Estratificada por
especie, semilla 42, 70/15/15. **Se excluyen las especies con < 15 fotos** (quedan
solo-audio, cubiertas por la fusión multimodal): *Charadrius nivosus*, *Rallus antarcticus*.

| | Valor |
|---|---|
| Especies en el clasificador | **53** (de 55) |
| train / val / test | 4.980 / 1.040 / 1.040 |

---

## 3. Preprocesamiento

`src/biomonitor/vision/dataset.py`. No requiere cache (las fotos ya son livianas):

- **Train:** `RandomResizedCrop(384, scale 0.7-1.0)` + flip horizontal + color jitter.
- **Val/test:** `Resize(441)` + `CenterCrop(384)`.
- Normalización con estadísticas de ImageNet (backbones preentrenados en ImageNet).
- **Evaluación con TTA** (`scripts/14`): 10-crop (4 esquinas + centro, + volteos).

---

## 4. Modelos e hiperparámetros

Transfer learning con timm. Se probaron dos variantes del mismo backbone.

| Hiperparámetro | Valor |
|---|---|
| Backbone | `efficientnet_b3` y `tf_efficientnet_b3.ns_jft_in1k` (NoisyStudent) |
| Entrada | 384×384 RGB |
| Optimizador | AdamW, lr=5e-4, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR |
| Batch size | 32 |
| Regularización | mixup (α=0.2), label_smoothing=0.1, drop_rate=0.3, drop_path=0.2 |
| Dispositivo | Apple M4 Pro (MPS) — ~250-500 s/época (con throttling térmico) |

---

## 5. Resultados

**Modelo adoptado: EfficientNet-B3 (transfer learning desde ImageNet), 50 épocas
completas.** Métrica por foto (cada foto es un ejemplo; no hay agregación como en audio,
salvo TTA de 10-crop).

| Modelo (EfficientNet-B3) | single-crop acc | single F1 | **TTA acc** | **TTA F1** |
|---|---|---|---|---|
| **adoptado** | 0.839 | 0.802 | **0.848** | 0.808 |

El TTA (10-crop) aporta +0.9 puntos. Se evaluó además la variante de pesos
**NoisyStudent** (`tf_efficientnet_b3.ns_jft_in1k`): rindió equivalente (TTA acc 0.845,
F1 0.820) y **no mejoró** el resultado, por lo que se conserva el B3 (ImageNet).

Métricas por especie: `reports/image_efficientnet_b3_per_class.csv`.

---

## 6. Registro de entrenamiento

<details><summary>EfficientNet-B3 (ImageNet) — 50 épocas · mejor val_f1 época 38</summary>

```
ep  tr_loss  val_acc  val_f1        ep  tr_loss  val_acc  val_f1
 1   3.218   0.660   0.626 ★        26  1.425   0.825   0.800
 5   1.699   0.794   0.761 ★        27  1.418   0.835   0.801 ★
 6   1.697   0.807   0.779 ★        31  1.389   0.835   0.804 ★
13   1.497   0.821   0.784 ★        32  1.367   0.837   0.805 ★
18   1.465   0.827   0.792 ★        36  1.332   0.842   0.814 ★
23   1.465   0.828   0.800 ★        38  1.401   0.855   0.828 ★ (mejor)
                                    48  1.292   0.857   0.824
                                    50  1.360   0.852   0.823
```
Progresión estable; plateau de val ~0.85 desde la época ~36. Test (checkpoint época 38):
single-crop 0.839 / TTA 0.848.
</details>

---

## 7. Análisis de errores

Peores especies por F1 (B3, TTA):

| F1 | Especie | n_test | Causa |
|---|---|---|---|
| 0.00 | *Agriornis albicauda* | 2 | Escasa (test minúsculo) |
| 0.00 | *Scytalopus fuscus* | 5 | *Scytalopus* casi idénticos entre sí (visualmente) |
| 0.44 | *Conirostrum tamarugense* | 3 | Escasa, endémica |
| 0.73 | *Pteroptochos castaneus* | 6 | Confusión con congéneres |
| 0.73-0.75 | *Leptasthenura*, *Diuca*, *Phrygilus* | 22 | Congéneres/similares |

**Patrón:** confusiones **intra-género** (visualmente muy parecidas) + especies **raras
con test minúsculo** (n=2-5) que fallan por completo y hunden el F1 macro. Es el mismo
límite (escasez + similitud) que en audio.

---

## 8. Decisión de alcance

**El clasificador de imagen se acepta en TTA 0.848 (B3), apenas bajo el 0.85.** Razones:

1. **La meta ">85% accuracy" del entregable es para el sistema MULTIMODAL**, no para
   cada modalidad por separado ("Modelo de clasificación multimodal … accuracy superior
   al 85%"). El **audio ya está en 0.858/0.868** y la **fusión** (ver
   [MODULO_FUSION.md](MODULO_FUSION.md)) los combina y supera 0.85 (fusión: **0.950**).
2. **La imagen-sola está en su techo** (~0.85) para este backbone y estos datos:
   convergió (plateau de 12+ épocas) y el límite son las especies difíciles.
3. Subir imagen-sola requeriría un backbone in22k (ConvNeXt/ViT), **inviable localmente**
   en el Mac (~8 h/época para ConvNeXt). Queda anotado que se podría hacer en **Colab/nube**
   (código portable) si se quiere un número de modalidad única más holgado.

---

## 9. Reproducibilidad

```bash
uv run python scripts/10_download_images.py      # descarga fotos (GBIF / iNat RG)
uv run python scripts/11_split_images.py          # split 70/15/15 (excluye <15 fotos)
uv run python scripts/12_train_image.py --epochs 50                              # B3
uv run python scripts/14_evaluate_image.py        # evaluación con TTA (10-crop)
```

**Robustez del entrenamiento:** `12_train_image.py` usa
`torch.multiprocessing.set_sharing_strategy("file_system")` (evita el límite de
descriptores de archivo de macOS con DataLoader workers en corridas largas) y admite
`--resume` / `--init-from` para continuar sin perder progreso.
Hiperparámetros en `config/config.yaml` (`image`, `train_image`).
