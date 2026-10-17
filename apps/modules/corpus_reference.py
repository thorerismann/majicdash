"""Metric columns shared by the CSV builder and live comparison."""

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REFERENCE_PATH = DATA_DIR / "corpus_reference.csv"
METRIC_KEYS = (
    "word_count", "paragraph_count", "avg_paragraph_words", "sentence_count",
    "avg_sentence_length", "syllable_count", "syllables_per_word", "char_count",
    "reading_ease", "fk_grade",
)


def comparison_values(metrics: dict, extra: dict) -> dict:
    """Use total paragraph count, never the six-paragraph display preview."""
    return {
        **{key: metrics[key] for key in METRIC_KEYS if key in metrics},
        "paragraph_count": extra["paragraph_count"],
        "avg_paragraph_words": metrics["word_count"] / extra["paragraph_count"] if extra["paragraph_count"] else 0.0,
        "syllables_per_word": metrics["syllable_count"] / metrics["word_count"] if metrics["word_count"] else 0.0,
    }
