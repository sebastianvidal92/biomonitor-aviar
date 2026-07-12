# Sistema Inteligente de Monitoreo de Biodiversidad Aviar en Chile
### Informe técnico completo — Proyecto de titulación

**Autor:** Sebastián Vidal · **Programa:** Magíster en Inteligencia Artificial, Pontificia
Universidad Católica de Chile · **Profesor:** Marcelo Mendoza · **Semestre:** 2026/1

> Este documento narra el proyecto de principio a fin: qué se hizo y por qué, de dónde
> salieron los datos y qué problemas hubo para obtenerlos, qué modelos se usaron, qué
> funcionó, qué **no** funcionó, y los hallazgos. Los detalles por módulo están en
> [MODULO_AUDIO](MODULO_AUDIO.md), [MODULO_IMAGEN](MODULO_IMAGEN.md),
> [MODULO_FUSION](MODULO_FUSION.md) y [MODULO_SDM](MODULO_SDM.md); las decisiones de diseño
> en [DECISIONES](DECISIONES.md).

---

## 1. Resumen ejecutivo

Se construyó un sistema de aprendizaje profundo **multimodal** para monitorear avifauna
chilena, con dos módulos:

- **Módulo 1 — Identificación de especies** desde el canto (audio) y/o una foto (imagen),
  con **fusión multimodal**.
- **Módulo 2 — Distribución espaciotemporal (SDM)**: predecir dónde y cuándo está presente
  cada especie a partir de avistamientos y variables ambientales.

**Resultados principales:**

| Componente | Métrica | Resultado | Meta |
|---|---|---|---|
| Audio — CNN (EfficientNet-B0) | accuracy | 0.858 | — |
| Audio — AST (transformer) | accuracy | **0.868** | — |
| Imagen — EfficientNet-B3 | accuracy | 0.848 | — |
| **Identificación multimodal (fusión)** | accuracy | **0.950** | **>0.85 ✅** |
| SDM — MaxEnt (baseline clásico) | AUC-ROC | 0.863 | — |
| **SDM — Random Forest** | AUC-ROC | **0.884** | **> MaxEnt ✅** |

Se entregó además un **prototipo web** (Streamlit) que integra los tres modelos.

**Dos hallazgos centrales, ambos honestos y algo contraintuitivos:**
1. En identificación, la **fusión multimodal supera con holgura** a cada modalidad por
   separado (0.950 vs. 0.867 audio / 0.825 imagen): sus errores son complementarios.
2. En distribución, el **deep learning NO superó a MaxEnt**; quienes lo superaron fueron
   los **ensambles de árboles (Random Forest / Boosted Regression Trees)**. Y agregar
   **datos satelitales no estructurados no ayudó** — resultaron redundantes con las
   covariables ambientales.

---

## 2. Motivación, pregunta de investigación e hipótesis

No existe hoy un sistema integrado que identifique aves chilenas de forma multimodal y, a
la vez, modele su distribución con deep learning. El problema es real y con impacto en
conservación (CONAF, SAG, MMA), ciencia ciudadana (eBird Chile, ROC) e investigación.

**Pregunta principal:** ¿Puede un sistema de deep learning multimodal identificar especies
de aves chilenas (audio y/o imagen) y modelar su distribución espaciotemporal mejor que los
enfoques de modalidad única y que los métodos clásicos (MaxEnt)?

**Hipótesis:** (H1) la fusión multimodal supera a cada modalidad sola; (H2) un SDM con deep
learning supera a MaxEnt. **H1 se confirmó; H2 no** — y ese resultado negativo, bien
fundamentado, es parte del aporte.

---

## 3. Alcance y selección de especies

**Alcance geográfico:** Chile continental (bounding box lat −17.5°…−56.0°, lon −66.4°…−75.7°).
Se **excluyó** el territorio antártico e insular oceánico (Rapa Nui, Juan Fernández,
Desventuradas), cuya avifauna sesgaría los modelos.

**Construcción de la lista de especies** (pipeline `scripts/01-03`):
1. Se descargó la **Nómina de Especies según Estado de Conservación** del Reglamento de
   Clasificación de Especies (RCE) del MMA (Excel oficial, procesos 1°–20°): **149 aves
   clasificadas**.
2. Foco en endémicas + amenazadas (CR/EN/VU): **59 candidatas continentales**.
3. **Problema/hallazgo clave:** ~25 de las amenazadas son **aves pelágicas** (albatros,
   petreles, golondrinas de mar, pingüinos). No sirven para el sistema porque (a) casi no
   "cantan" y no tienen grabaciones, y (b) su distribución es oceánica, incompatible con un
   SDM terrestre basado en NDVI. Se excluyeron del clasificador de cantos.
4. Se cruzó cada especie con su **disponibilidad real de datos** (grabaciones en Xeno-canto,
   fotos en GBIF). **Segundo hallazgo clave:** las especies amenazadas/endémicas tienen muy
   pocos datos (solo ~11-19 superan 10-30 grabaciones). *Es imposible llegar a 50 especies
   solo con amenazadas.*

**Lista final: 55 especies**, con una composición decidida explícitamente:
- **Núcleo de conservación (14):** amenazadas/endémicas terrestres con audio suficiente.
- **Endémicas comunes (6):** turca, tenca, choroy, churrín, etc.
- **Relleno de comunes (35):** aves con datos abundantes, para robustez.

La propuesta pedía "foco" en amenazadas, no exclusividad; esta composición cumple el mínimo
de 50 y es defendible. La **escasez de datos para especies raras** es un tema recurrente
del informe y el desafío central del monitoreo bioacústico.

---

## 4. Fuentes de datos y problemas de obtención

| Dato | Fuente | Volumen | Problemas y solución |
|---|---|---|---|
| Audio (cantos) | Xeno-canto API v3 | 2.508 grabaciones (21.4 h) | La API v2 fue retirada → se usó la v3 (requiere key). |
| Fotos | iNaturalist Research-grade vía GBIF | 7.068 fotos | Sinónimos de taxonomía y ruido (ver abajo). |
| Avistamientos | GBIF (eBird + iNaturalist) | 100.270 puntos | Se evitó el eBird EBD (aprobación humana con lead time) usando GBIF. |
| Clima | WorldClim BIO (vía GEE) | — | Plan original ERA5/DMC; ver §7. |
| NDVI | MODIS MOD13A2 (vía GEE) | — | Plan original Sentinel-2; ver §7. |
| Elevación | SRTM (vía GEE) | — | — |
| Estado conservación | MMA RCE (Excel) | 149 aves | Descarga directa, sin key. |

**Credenciales (5 servicios).** Se configuraron eBird, Xeno-canto, GBIF, Copernicus CDS y
Google Earth Engine, verificadas con `scripts/00_check_credentials.py`. El caso complejo fue
**Google Earth Engine**: el autenticador OAuth se **bloqueaba** ("Se bloqueó esta app",
por pedir permisos sensibles sin pantalla de consentimiento configurada). Se resolvió usando
una **cuenta de servicio** (JSON) con rol *Editor* en el proyecto GCP — sin navegador, sin
consentimiento, y reutilizable en la nube.

**Problemas específicos de obtención de datos:**
- **Taxonomía de fotos:** iNaturalist reclasifica o usa otra grafía. *Elaenia chilensis*
  daba 0 fotos hasta buscar como *E. albiceps* (subespecie); *Sylviorthorhynchus desmurii*
  requería la grafía *desmursii*. Se agregó un mapa de alias.
- **Ruido en el fallback de fotos:** al ampliar más allá de iNaturalist para especies
  escasas, GBIF devolvía **espectrogramas de Xeno-canto** y **especímenes de museo** (Arctos,
  con timeouts de 60 s). Se filtró a `basisOfRecord=HUMAN_OBSERVATION`, solo JPEG con licencia
  CC, excluyendo esos hosts.
- **Especies sin datos de imagen:** *Charadrius nivosus* (2 fotos de aves vivas; el resto eran
  especímenes) y *Rallus antarcticus* (6) quedaron **solo-audio** — cubiertas por la fusión.
- **Descargas interrumpidas:** los procesos largos en el entorno de asistencia se cortaban al
  cambiar de contexto → se ejecutaron localmente y se hicieron **reanudables** (saltan lo ya
  descargado). Un avistamiento (*Troglodytes aedon*) se cayó por un timeout de GBIF → se
  agregaron reintentos.

---

## 5. Infraestructura y decisiones técnicas

- **Entorno:** `uv` con **Python 3.12 fijado**. El Python del sistema era 3.14, demasiado
  nuevo: PyTorch/torchaudio aún no traen wheels estables para 3.14.
- **Cómputo:** local en Mac Apple M4 Pro (GPU vía Metal/MPS). Los entrenamientos y descargas
  pesadas los ejecutó el usuario.
- **Sin librosa:** se usó `torchaudio` para resampleo y espectrogramas. Instalar librosa
  arrastraba `numba`/`llvmlite` incompatibles con Python 3.12.
- **Estructura reproducible:** paquete `biomonitor` (data, audio, vision, multimodal, sdm),
  scripts numerados 00-23, configuración central en `config/config.yaml`, credenciales y
  datos/modelos fuera de git.

---

## 6. Módulo 1 — Identificación multimodal

### 6.1 Audio (cantos)
- **Descarga:** 2.508 grabaciones (calidad A/B/C, ≤50/especie, <4 min), formatos mezclados
  (1.884 mp3 + 625 wav a distintas tasas).
- **Problema — cache:** al uniformar a 32 kHz mono, el encoder **OGG/Vorbis de libsndfile
  hacía segfault** con ciertos WAV grandes, dejando 88 archivos con 0 muestras. Se cambió el
  formato de cache a **FLAC** (robusto). El audio quedó en 0.6 GB.
- **Partición sin fuga:** train/val/test **a nivel de grabación** (no de clip), para que
  clips de 5 s de una misma grabación no se filtren entre conjuntos.
- **Entrada:** ventana de 5 s → espectrograma **log-mel (128 bandas)** calculado on-the-fly.
- **Modelos y resultados** (accuracy en test):
  - **CNN baseline (EfficientNet-B0):** 0.756 → **0.858** tras agregar regularización
    (mixup, SpecAugment, label smoothing, dropout) y **agregación por grabación** (promediar
    varias ventanas; +8 puntos sobre una sola ventana).
  - **AST (Audio Spectrogram Transformer, preentrenado en AudioSet):** **0.868**, superando
    al CNN, a costa de ~20× más cómputo (8 min/época vs 22 s).
- **Errores:** confusiones **intra-género** (Cinclodes, Phrygilus, Pteroptochos) — el límite
  natural de la bioacústica.

### 6.2 Imagen (fotos)
- **Descarga:** 7.068 fotos de iNaturalist Research-grade (validadas por la comunidad,
  geolocalizadas, con licencia CC), tamaño medium (~1 GB). **53 especies** (2 quedaron
  solo-audio por escasez).
- **Modelo:** EfficientNet-B3 (transfer learning, 384 px) con augmentation + mixup.
- **Problema — crash de macOS:** el entrenamiento con la variante NoisyStudent murió por
  *"Too many open files"* (límite de descriptores de archivo con DataLoader en macOS). Se
  corrigió con `torch.multiprocessing.set_sharing_strategy("file_system")` + reanudación.
- **Resultado:** **0.848** con Test-Time Augmentation (10-crop). Se estancó en ~0.85 (techo
  del backbone para estos datos). **Decisión:** se aceptó, porque la meta >0.85 del entregable
  es del **sistema multimodal**, no de cada modalidad sola.

### 6.3 Fusión multimodal
- **Método:** fusión tardía — combinación ponderada de las probabilidades de ambos modelos.
- **Problema — datos no pareados:** el audio (Xeno-canto) y las fotos (iNaturalist) no vienen
  del mismo individuo. Se evaluó con **pareo por especie**: emparejar un audio y una foto de
  la misma especie (simula a un observador que graba y fotografía el mismo pájaro).
- **Resultado:** **fusión 0.950 > audio 0.867 > imagen 0.825**. **Confirma H1**: los errores
  de audio e imagen son complementarios, y la meta del entregable (>0.85) se cumple con
  holgura.
- **Comparación de estrategias de fusión** (late vs early vs intermediate, condiciones
  idénticas): **late 0.931 ≈ early 0.930 > intermediate 0.906**. La fusión simple (tardía o
  concatenación temprana) es óptima; la intermedia, más compleja, sobreajusta y no aporta.

---

## 7. Módulo 2 — Distribución espaciotemporal (SDM)

### 7.1 Datos y covariables
- **Presencia:** 100.270 avistamientos de GBIF (55 especies), adelgazados a ≤500/especie.
- **Fondo (pseudo-ausencias):** *target-group background* — muestra de los avistamientos de
  todas las especies. Corrige el **sesgo de muestreo** de la ciencia ciudadana.
- **Covariables (vía GEE):** NDVI (MODIS), temperatura y precipitación anuales (WorldClim
  BIO), elevación (SRTM), y **mes cíclico** (estacionalidad). Dataset final: **34.926 puntos**.

**Problema mayor — GEE se quedaba sin cómputo.** El plan original (inscripción) usaba **ERA5**
(clima) y **Sentinel-2** (NDVI). Pero muestrear la **mediana de Sentinel-2** (miles de
imágenes) o el **promedio de ERA5** (120 imágenes) en decenas de miles de puntos daba
*"Computation timed out"*. La solución fue usar capas **precalculadas y livianas**: **MODIS
NDVI** y **WorldClim** (climatologías estáticas) — sustitutos estándar en SDM. Además, generar
puntos de fondo con el polígono de Chile de GAUL era lentísimo → se resolvió con el
target-group background (que ya está en Chile).

### 7.2 Modelos y resultados (AUC-ROC por especie)

| Modelo | AUC | ¿Supera a MaxEnt? |
|---|---|---|
| MaxEnt (baseline clásico) | 0.863 | — |
| Red neuronal (deep learning conjunto) | 0.849 | **No** |
| **Random Forest** | **0.884** | **Sí (46/54 especies)** |
| Gradient Boosting | 0.883 | Sí (44/54) |
| DL + parche de cobertura de suelo (CNN) | 0.613 | No |
| DL + embedding satelital 64-dim | 0.767 | No |
| Random Forest + embedding | 0.878 | El embedding baja |

### 7.3 Lo que NO funcionó (y por qué es valioso)
- **El deep learning no superó a MaxEnt.** En datos tabulares con buenas covariables y
  muestras moderadas, los modelos regularizados/árboles suelen ganarle a las redes.
- **Los datos satelitales no estructurados no ayudaron.** Se probó, por iniciativa de dar al
  DL una modalidad que MaxEnt no puede usar:
  - *Parche de cobertura de suelo (ESA WorldCover) + CNN* → 0.613. Diagnóstico: los parches
    estaban **bien alineados**, pero la cobertura de suelo es **redundante** con NDVI/elevación
    (árbol↔NDVI alto, descubierto↔NDVI bajo/elevación alta), y la CNN agrega capacidad que
    **sobreajusta**.
  - *Embedding de foundation model (AlphaEarth, 64-dim)* → 0.767. También degradó, e incluso
    **bajó a los árboles** (0.884→0.878). Confirmación definitiva de que las features
    satelitales son redundantes/ruidosas aquí.
  - *Sentinel-2 RGB crudo:* se descartó — muestrear parches a escala es inviable localmente
    (timeout de GEE) y, dado lo anterior, casi seguro tampoco ayudaría. Queda como trabajo
    futuro en la nube.
- **Conclusión (refuta H2 con evidencia):** para SDM de avifauna chilena con covariables
  ambientales fuertes, un **Random Forest** es la mejor opción; la complejidad extra (redes,
  imágenes) no se justifica.

---

## 8. Prototipo web (demo)

Aplicación **Streamlit** (`app/app.py`) que integra los tres modelos: se sube un audio y/o
una foto, el sistema identifica la especie (con fusión si hay ambos, top-5 con confianza) y
dibuja su **mapa de distribución estimada** (Random Forest del SDM sobre una grilla de Chile,
con selector de mes). Insumos preparados con `scripts/23_prepare_demo.py`.

---

## 9. Anexo — Problemas técnicos y cómo se resolvieron

| Problema | Solución |
|---|---|
| Python 3.14 sin wheels de PyTorch | Fijar Python 3.12 con uv |
| `librosa` → numba/llvmlite incompatibles con 3.12 | Usar torchaudio |
| GEE OAuth "Se bloqueó esta app" | Cuenta de servicio (JSON) + rol IAM |
| eBird EBD con lead time de aprobación | Usar avistamientos de GBIF |
| Xeno-canto API v2 retirada | Migrar a la API v3 (con key) |
| OGG/Vorbis segfault con WAV grandes (88 archivos vacíos) | Cache en FLAC |
| Sinónimos/grafía de taxonomía en fotos | Mapa de alias (Elaenia, Sylviorthorhynchus) |
| Espectrogramas y especímenes de museo en fotos | Filtro HUMAN_OBSERVATION + JPEG/CC + host |
| "Too many open files" en macOS (DataLoader) | `set_sharing_strategy("file_system")` |
| ERA5/Sentinel-2: "Computation timed out" en GEE | MODIS + WorldClim (capas precalculadas) |
| Máscara de Chile (GAUL) lentísima en GEE | Target-group background |
| Datasets audio/imagen no pareados | Evaluación de fusión por pareo de especie |
| Procesos largos interrumpidos | Ejecución local + scripts reanudables |
| Bug pandas: groupby movía la columna a índice | Adelgazamiento con `pd.concat` explícito |
| MPS no soporta float64 | Forzar tensores a float32 |
| ConvNeXt ~8 h/época en el Mac | Descartado; NoisyStudent B3 (mismo costo) |
| `uv sync --extra app` quitaba las deps de `ml` | Sincronizar `--extra ml --extra app` juntos |

---

## 10. Conclusiones

1. **Se cumplió el entregable central:** identificación multimodal con **accuracy 0.950
   (>85%)** sobre 55 especies chilenas, validando que la fusión audio+imagen supera a cada
   modalidad.
2. **Comparación audio:** el transformer AST (0.868) supera levemente al CNN (0.858), a alto
   costo de cómputo — un trade-off precisión/eficiencia documentado.
3. **SDM:** se comparó rigurosamente MaxEnt, deep learning y ML de árboles. **Random Forest
   (0.884) superó a MaxEnt (0.863)**; el deep learning y los datos satelitales no aportaron.
   Es un resultado honesto que **refuta la hipótesis DL>MaxEnt** con evidencia.
4. **Hallazgo transversal:** la **escasez de datos para especies raras/amenazadas** limita el
   monitoreo bioacústico y de imagen — precisamente las especies de mayor interés de
   conservación son las de menos datos.

## 11. Limitaciones y trabajo futuro

- Solicitar el **eBird EBD** para un histórico de avistamientos más rico.
- Probar **Sentinel-2 RGB crudo** para el SDM en la nube (GPU + export asíncrono de GEE).
- Aumentar datos de especies raras (data augmentation avanzada, few-shot, o campañas de
  registro dirigidas).
- Llevar la demo a un ambiente productivo estable.
