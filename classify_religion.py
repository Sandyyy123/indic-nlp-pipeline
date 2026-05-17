"""
Stage 4 — Religion classification from normalized Indian names.
Wraps the client-specified ML algorithm with a clean batch interface,
confidence scores, and configurable threshold for uncertain cases.

Replace _load_model() with the actual algorithm once provided.
The interface (predict_batch) stays the same regardless of backend.
"""

import logging
from pathlib import Path
from typing import Optional

import polars as pl

logger = logging.getLogger(__name__)

LABELS = ["Hindu", "Muslim", "Sikh", "Christian", "Buddhist", "Jain", "Other", "Unknown"]
CONFIDENCE_THRESHOLD = 0.6  # below this -> flagged as "Uncertain"


# ---------------------------------------------------------------------------
# Model loading — replace body with actual algorithm
# ---------------------------------------------------------------------------

class _NameReligionClassifier:
    """
    Placeholder wrapper. Replace _load() and _predict_one() with
    the actual ML model (e.g. NamSor, Ethnea, custom LSTM/CNN, etc.).
    """

    def __init__(self, model_path: Optional[Path] = None):
        self.model = self._load(model_path)

    def _load(self, model_path: Optional[Path]):
        # TODO: load client-specified model weights here
        logger.info("Religion classifier: using stub (replace with real model)")
        return None

    def _predict_one(self, given_name: str, surname: str) -> tuple[str, float]:
        """
        Return (label, confidence) for a single name.
        Stub: returns 'Unknown' with 0.0 confidence.
        Replace with real inference call.
        """
        if self.model is None:
            return "Unknown", 0.0
        # --- real inference goes here ---
        raise NotImplementedError("Plug in real model inference")

    def predict_batch(
        self,
        given_names: list[str],
        surnames: list[str],
    ) -> list[tuple[str, float]]:
        """Batch inference — returns list of (label, confidence) tuples."""
        results = []
        for gn, sn in zip(given_names, surnames):
            label, conf = self._predict_one(gn, sn)
            if conf < CONFIDENCE_THRESHOLD:
                label = "Uncertain"
            results.append((label, round(conf, 4)))
        return results


_classifier: Optional[_NameReligionClassifier] = None


def load_classifier(model_path: Optional[Path] = None) -> None:
    global _classifier
    _classifier = _NameReligionClassifier(model_path)


def classify_chunk(
    df: pl.DataFrame,
    given_col: str = "given_name",
    surname_col: str = "surname",
    batch_size: int = 10_000,
) -> pl.DataFrame:
    """
    Add religion_label and religion_confidence columns to a Polars chunk.
    Processes in sub-batches to limit peak memory.
    """
    global _classifier
    if _classifier is None:
        load_classifier()

    given_names = df[given_col].to_list()
    surnames = df[surname_col].to_list()

    labels, confidences = [], []
    for i in range(0, len(given_names), batch_size):
        batch_results = _classifier.predict_batch(
            given_names[i : i + batch_size],
            surnames[i : i + batch_size],
        )
        for label, conf in batch_results:
            labels.append(label)
            confidences.append(conf)

    return df.with_columns([
        pl.Series("religion_label", labels, dtype=pl.Utf8),
        pl.Series("religion_confidence", confidences, dtype=pl.Float32),
    ])
