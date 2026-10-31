# Informe de Actividad de Graduación — MIA UC

Proyecto LaTeX de la tesis (borrador del 30 de octubre e informe final del 30 de
noviembre de 2026). Compila en Overleaf (pdfLaTeX) o localmente con:

    latexmk -pdf main.tex

El PDF compilado más reciente queda en `Tesis_borrador_SebastianVidal.pdf`.

## Archivos

| Archivo | Contenido |
|---|---|
| `main.tex` | Documento principal (título, autor, fecha, orden de capítulos y anexos) |
| `preambulo.tex` | Paquetes, tipografía Times, márgenes, encabezados, marcas de revisión, bibliotecas TikZ |
| `portada.tex`, `ficha.tex` | Portada institucional y ficha del proyecto |
| `resumen.tex` | Resumen estructurado con los resultados finales e intervalos de confianza |
| `cap1_introduccion.tex` | 1. Introducción (figura de arquitectura, contribuciones) |
| `cap2_trabajo_relacionado.tex` | 2. Trabajo relacionado (tabla BirdNET, brecha) |
| `cap3_bases_teoricas.tex` | 3. Bases teóricas: PI/OE, redes neuronales y entrenamiento, regularización, CNN/EfficientNet, transformers (ViT/AST), transfer learning, audio → log-mel, protocolos de inferencia, fusión, SDM (MaxEnt, árboles, red conjunta), métricas, validación espacial, baselines y justificación |
| `cap4_datos.tex` | 4. Levantamiento de datos (especies, corpus, avistamientos, covariables, desviaciones del plan) |
| `cap5_resultados.tex` | 5. Resultados (audio, imagen, fusión, baselines, SDM, prototipo, síntesis) |
| `cap6_conclusiones.tex` | 6. Conclusiones, desafíos y trabajo futuro |
| `anexoA.tex` | A. Conjunto de especies objetivo |
| `anexoB.tex` | B. Hiperparámetros y detalles de entrenamiento |
| `datos/tab_*.tex` | Tablas generadas desde `reports/` por los scripts 34 y 35 (anexo C, por grupo, modelos SDM, baselines, intervalos de confianza, tests) |
| `figuras/fig_*.pdf` | Figuras vectoriales generadas por los scripts 32, 33 y `scripts/theory_figs/` (incluye `fig_seleccion_especies.pdf`, el embudo del cap. 4, y `fig_roc.pdf` en dos paneles) |
| `figuras/fig_arquitectura.png`, `fig_ast_gong2021.png`, `fig_fusion_huang2020.png`, `fig_tta_foto.jpg` | Imágenes: diagrama del sistema (draw.io del autor), figura 1 de Gong et al. (2021) renderizada desde el PDF de arXiv a 600 dpi, tres paneles de la figura 2 de Huang et al. (2020, CC BY 4.0) y una foto de chucao del corpus (iNaturalist, CC BY-NC 4.0, autor pmarinaraya) usada en la figura de TTA |
| `figuras/tikz_*.tex` | Diagramas TikZ del marco teórico que siguen en uso: CNN, ventanas/TTA, árboles y red SDM (se incluyen con `\input`). `tikz_efficientnet`, `tikz_ast`, `tikz_fusion` y `tikz_seleccion_especies` ya no se usan (reemplazados el 30-oct-2026 por imágenes de los papers, el embudo y texto) |
| `referencias.bib` | Bibliografía (natbib + `apalike-es`); los libros del marco teórico se citan con página |

## Marcas de revisión

`preambulo.tex` define `\revisionfalse` (entrega): los pendientes (`\pendiente{...}`)
y los comentarios al profesor (`\profe{...}`) no se imprimen. Cambiar a
`\revisiontrue` para verlos como cajas azules. Actualizar `\fechaTesis` en
`main.tex` para la entrega final.

## Colocación de figuras y tablas

Los flotantes usan `[!htb]` y el paquete `placeins` con la opción `section`,
de modo que cada figura o tabla queda en la página de su referencia o en la
siguiente y nunca cruza a otra sección. Las figuras pequeñas van con `[H]`.

## Regenerar figuras y tablas

Desde la raíz del repositorio:

    uv run python scripts/32_thesis_figures.py        # figuras de resultados → tesis/figuras/*.pdf
    uv run python scripts/33_sdm_maps.py              # mapas del SDM
    uv run python scripts/34_thesis_tables.py         # tablas → tesis/datos/tab_*.tex
    uv run python scripts/35_confidence_intervals.py  # IC bootstrap y tests de Wilcoxon → tab_ic, tab_tests_sdm
    for f in scripts/theory_figs/*.py; do uv run python "$f"; done   # figuras del marco teórico (ROC, bloques, fondo, log-mel, embudo de especies, semivariograma)

Los scripts 27 a 31 producen los resultados (evaluación extendida, validación
espacial, frozen embeddings, IA generativa, análisis de fusión); los scripts
32, 34 y 35 incorporan automáticamente esos CSV cuando existen. La figura
`fig_sobreajuste` (curvas con y sin regularización) solo se incluye si existe
`reports/training_logs/audio_efficientnet_b0_sinreg.csv`, que produce:

    uv run python scripts/07_train_audio.py --no-augment --mixup-alpha 0 --label-smoothing 0 --drop-rate 0 --drop-path 0 --epochs 100 --tag sinreg
