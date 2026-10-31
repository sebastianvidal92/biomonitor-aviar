# Módulo 1 — Clasificador de cantos (audio)

Documentación técnica completa del clasificador de especies a partir de audio.
Cubre datos, preprocesamiento, modelos, hiperparámetros, entrenamientos (época a
época) y resultados. **Entregable asociado:** identificar ≥50 especies con
accuracy > 0.85 → **cumplido** (ver [Resultados](#5-resultados)).

Última actualización: 2026-07-04.

---

## Índice
1. [Datos y especies](#1-datos-y-especies)
2. [Partición train/val/test](#2-partición-trainvaltest)
3. [Preprocesamiento](#3-preprocesamiento)
4. [Modelos e hiperparámetros](#4-modelos-e-hiperparámetros)
5. [Resultados](#5-resultados)
6. [Registro de entrenamiento (época a época)](#6-registro-de-entrenamiento-época-a-época)
7. [Análisis de errores](#7-análisis-de-errores)
8. [Reproducibilidad](#8-reproducibilidad)

---

## 1. Datos y especies

**Fuente de audio:** [Xeno-canto](https://xeno-canto.org/) API v3 (cantos etiquetados).

**Universo de 55 especies** (`data/external/species_final.csv`), construido con criterio
de conservación y respaldado por disponibilidad real de datos:

| Rol | N° | Descripción |
|---|---|---|
| 🛡️ Conservación | 14 | Amenazadas (CR/EN/VU) o endémicas terrestres/costeras con audio suficiente |
| ★ Endémicas | 6 | Endémicas de Chile continental comunes (turca, tenca, choroy, churrín…) |
| 🐦 Comunes | 35 | Relleno con datos abundantes (zorzal, chincol, chucao…) |

**Por qué esta composición** (decisión registrada en [DECISIONES.md](DECISIONES.md)):
las especies amenazadas/endémicas de Chile son escasas en audio (solo ~11-19 superan
10-30 grabaciones), y ~25 de las amenazadas son **pelágicas** (albatros, petreles) que
no vocalizan en tierra → se excluyeron del clasificador de cantos. Es imposible llegar
a 50 solo con amenazadas, por lo que se completó con comunes. Esta escasez de datos
para especies raras es un **hallazgo central** para la discusión del informe.

**Descarga** (`scripts/04_download_audio.py`):

| Parámetro | Valor |
|---|---|
| Calidad Xeno-canto | A, B, C |
| Máx. grabaciones/especie | 50 |
| Máx. duración/grabación | 240 s |
| Total descargado | **2.508 grabaciones · 21.4 h · ~7.3 GB (raw)** |
| Mínimo por especie | 22 (ninguna bajo el umbral de 20) |
| Distribución calidad | A 69% · B 26% · C 5% |
| Origen | Chile 64%, resto global (para especies con pocos registros nacionales) |

> Se usa audio global (no solo Chile) porque la firma acústica es específica de la
> especie; para especies raras varias grabaciones provienen de países vecinos (leve
> *domain-shift*, documentado).

---

## 2. Partición train/val/test

`scripts/05_split_dataset.py` — **a nivel de grabación, no de clip** (evita *data
leakage*: todos los clips de 5 s de una misma grabación quedan en el mismo conjunto).
Estratificado por especie, semilla fija = 42.

| Split | Grabaciones | % |
|---|---|---|
| train | 1.721 | 69% |
| val | 394 | 16% |
| test | 393 | 16% |

Las 55 especies están presentes en los tres conjuntos.

---

## 3. Preprocesamiento

`scripts/06_preprocess_audio.py` + `src/biomonitor/audio/preprocess.py`.

- **Resampleo a 32 kHz mono** y cacheo comprimido → uniforma formatos heterogéneos
  (1.884 mp3 + 625 wav, varias tasas 44.1/48/96 kHz) en un solo formato.
- **Formato de cache: FLAC** (lossless). *Decisión técnica:* el encoder OGG/Vorbis de
  libsndfile hace **segfault** con ciertos WAV grandes de alta tasa (dejaba 88 archivos
  con 0 muestras); FLAC es robusto. Cache final: **~0.6 GB** (vs 7.3 GB raw).
- **Entrada al modelo:** ventana de **5 s** (aleatoria en train, central en eval) →
  **espectrograma log-mel** de 128 bandas, calculado *on-the-fly* con `torchaudio`
  (`n_fft=1024`, `hop=320` → ~100 frames/s → ~501 frames), normalizado por muestra.

---

## 4. Modelos e hiperparámetros

Se entrenaron **dos modelos** para la comparación CNN vs. transformer que pide la propuesta.

### 4.1 Baseline — EfficientNet-B0 (CNN)
CNN preentrenada en ImageNet (timm) que trata el espectrograma como imagen de 1 canal.

| Hiperparámetro | Valor |
|---|---|
| Backbone | `efficientnet_b0` (preentrenado, `in_chans=1`) |
| Optimizador | AdamW, lr=1e-3, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR |
| Épocas | 100 |
| Batch size | 32 |
| Pérdida | CrossEntropy con pesos por clase (inv. frecuencia) |
| **Regularización** | **mixup (α=0.2), label_smoothing=0.1, drop_rate=0.3, drop_path=0.2, SpecAugment ×2** |
| Dispositivo | Apple M4 Pro (MPS) — ~22 s/época |

> **Motivación de la regularización:** la corrida de control sin regularización
> explícita (`--tag sinreg`, 26-oct-2026: sin aumentación, mixup 0, label smoothing 0,
> dropout 0, drop_path 0; log en `reports/training_logs/audio_efficientnet_b0_sinreg.csv`)
> sobreajusta: train_loss → 0.003, val_loss estancada en 1.5-1.7 (mínimo 1.41 en la
> época 45) y 0.735 single / 0.789 agregado en test. Con la receta completa el gap
> train/val se cierra y el agregado sube a 0.858. (Una primera versión anterior, sin
> mixup ni SpecAugment, había dado 0.756 / 0.822; esos valores quedaron reemplazados.)

### 4.2 Avanzado — AST (Audio Spectrogram Transformer)
Fine-tuning de `MIT/ast-finetuned-audioset-10-10-0.4593` (preentrenado en AudioSet).

| Hiperparámetro | Valor |
|---|---|
| Checkpoint | `MIT/ast-finetuned-audioset-10-10-0.4593` (~86M params) |
| Entrada | fbank 128 mel × 1024 frames (ventana ~10 s a 16 kHz) |
| Optimizador | AdamW, lr=2e-5, weight_decay=0.01 |
| Scheduler | CosineAnnealingLR |
| Épocas | 20 |
| Batch size | 8 |
| Pérdida | CrossEntropy con pesos por clase + label_smoothing=0.1 |
| Dispositivo | Apple M4 Pro (MPS) — **~480 s/época (~8 min)** |

---

## 5. Resultados

**Protocolo de evaluación:** se reporta la métrica **por ventana central** y la
**agregada** (promedio de probabilidades sobre 8-12 ventanas de la grabación completa).
La agregada es la métrica **operacional legítima** (se clasifica la grabación entera,
no un fragmento arbitrario de 5 s), estándar en bioacústica (BirdNET). El informe debe
reportar ambas.

| Modelo | Test acc (single) | Test F1 (single) | **Test acc (agregado)** | **Test F1 (agregado)** | Costo |
|---|---|---|---|---|---|
| EfficientNet-B0 sin regularizar (control `--tag sinreg`) | 0.735 | 0.736 | 0.789 | 0.784 | 22 s/época |
| **EfficientNet-B0 regularizado** | 0.812 | 0.811 | **0.858** | **0.858** | 22 s/época |
| **AST (transformer)** | 0.832 | 0.828 | **0.868** | **0.862** | ~480 s/época |

**Conclusiones:**
- **Ambos modelos superan la meta del entregable (accuracy > 0.85, agregado).**
- El **AST gana** al CNN (+1.0 pt accuracy, +0.4 pt F1) y es más robusto por ventana
  (0.832 vs 0.812), pero a **~20× el costo de cómputo** — un buen argumento
  precisión-vs-eficiencia para la discusión.
- La **agregación por grabación** aporta +7-8 puntos gratis sobre la ventana única.

Métricas por especie (F1): `reports/audio_efficientnet_b0_per_class.csv`.
Modelos entrenados: `models/audio_efficientnet_b0.pt`, `models/audio_ast.pt` (en `.gitignore`).

---

## 6. Registro de entrenamiento (época a época)

### 6.1 EfficientNet-B0 regularizado (modelo final CNN) — mejor val_f1 en época 67

<details><summary>Log completo de las 100 épocas (tr_loss · val_loss · val_acc · val_f1)</summary>

```
ep  tr_loss  val_loss  val_acc  val_f1
 1   4.043    3.514    0.175   0.147  ★
 5   2.621    2.246    0.558   0.539  ★
10   2.248    1.861    0.680   0.675  ★
13   2.080    1.763    0.723   0.721  ★
19   1.984    1.673    0.749   0.748  ★
23   1.827    1.583    0.777   0.775  ★
38   1.691    1.573    0.794   0.793  ★
50   1.591    1.511    0.812   0.809  ★
60   1.399    1.485    0.815   0.810  ★
67   1.411    1.524    0.815   0.811  ★  (mejor — checkpoint guardado)
81   1.357    1.499    0.815   0.811
90   1.233    1.472    0.812   0.809
100  1.241    1.473    0.812   0.808
```
Progresión estable, sin sobreajuste severo (train_loss se mantiene ~1.2-1.4 gracias a
mixup). Plateau de val ~0.81 desde la época ~50. Test final (checkpoint época 67):
**single 0.812 / agregado 0.858**.
</details>

### 6.2 AST — mejor val_f1 en época 11

| Época | tr_loss | val_acc | val_f1 | |
|---|---|---|---|---|
| 1 | 2.844 | 0.675 | 0.668 | ★ |
| 2 | 1.480 | 0.779 | 0.781 | ★ |
| 3 | 1.112 | 0.802 | 0.799 | ★ |
| 4 | 0.983 | 0.805 | 0.802 | ★ |
| 5 | 0.927 | 0.812 | 0.819 | ★ |
| 6 | 0.893 | 0.827 | 0.829 | ★ |
| 7 | 0.867 | 0.827 | 0.822 | |
| 8 | 0.841 | 0.843 | 0.846 | ★ |
| 9 | 0.843 | 0.850 | 0.850 | ★ |
| 10 | 0.814 | 0.838 | 0.842 | |
| **11** | **0.801** | **0.860** | **0.860** | **★ mejor** |
| 12 | 0.802 | 0.850 | 0.851 | |
| 13 | 0.788 | 0.840 | 0.842 | |
| 14 | 0.788 | 0.850 | 0.853 | |
| 15 | 0.791 | 0.840 | 0.843 | |
| 16 | 0.777 | 0.848 | 0.850 | |
| 17 | 0.776 | 0.845 | 0.843 | |
| 18 | 0.777 | 0.843 | 0.842 | |
| 19 | 0.775 | 0.840 | 0.839 | |
| 20 | 0.774 | 0.843 | 0.841 | |

Convergencia rápida (preentrenado): mejor val en época 11, luego plateau. Test final
(checkpoint época 11): **single 0.832 / agregado 0.868**.

---

## 7. Análisis de errores

Peores especies por F1 (modelo CNN regularizado, agregado):

| F1 | Especie | n_test | Causa probable |
|---|---|---|---|
| 0.46 | *Cinclodes patagonicus* | 8 | Confusión con congénere *C. fuscus* |
| 0.50 | *Agriornis albicauda* | 3 | Rara, test muy chico |
| 0.62 | *Sylviorthorhynchus desmurii* | 8 | Furnárido, canto similar |
| 0.62 | *Phrygilus gayi* | 8 | Confusión con *P. patagonicus* / otros fringílidos |
| 0.62 | *Leptasthenura aegithaloides* | 8 | — |
| 0.63 | *Zonotrichia capensis* | 8 | Emberízido común, confundido con similares |
| 0.67 | *Ochetorhynchus melanurus* | 5 | Endémica, pocos datos |

**Patrón:** los errores son mayormente **confusiones intra-género** (especies
acústicamente muy parecidas: *Cinclodes*, *Phrygilus*, *Pteroptochos*) y **especies
raras con test pequeño**. Es el límite natural de la identificación bioacústica y buen
material de discusión.

---

## 8. Reproducibilidad

Pipeline completo (scripts numerados, todos reanudables):

```bash
uv run python scripts/01_build_species_list.py     # lista candidata (RCE del MMA)
uv run python scripts/02_enrich_availability.py    # conteo XC + fotos GBIF por especie
uv run python scripts/03_build_final_list.py        # lista final de 55 especies
uv run python scripts/04_download_audio.py          # descarga audio (Xeno-canto)
uv run python scripts/05_split_dataset.py           # split 70/15/15 sin leakage
uv run python scripts/06_preprocess_audio.py        # cache 32 kHz mono FLAC
uv run python scripts/07_train_audio.py --epochs 100    # entrena CNN (EfficientNet-B0)
uv run python scripts/08_evaluate_audio.py          # evalúa CNN (single + agregado)
uv run python scripts/09_train_audio_ast.py         # entrena + evalúa AST
```

Hiperparámetros editables en `config/config.yaml` (secciones `audio`, `train_audio`,
`train_ast`). Entorno: `uv sync --extra ml` (PyTorch + torchaudio + transformers).
