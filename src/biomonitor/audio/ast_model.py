"""Audio Spectrogram Transformer (AST) — modelo avanzado del Módulo 1 (audio).

Fine-tuning de AST preentrenado en AudioSet (HuggingFace) para las 55 especies.
Es el modelo "avanzado" que compara contra el baseline CNN (EfficientNet).
"""

from __future__ import annotations

from transformers import ASTForAudioClassification

from biomonitor.config import CONFIG

AST_CHECKPOINT = CONFIG["train_ast"]["checkpoint"]


def build_ast_model(num_classes: int, checkpoint: str = AST_CHECKPOINT):
    """AST preentrenado con una nueva cabeza de clasificación de `num_classes`."""
    return ASTForAudioClassification.from_pretrained(
        checkpoint,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,  # reemplaza la cabeza de AudioSet (527 clases)
    )
