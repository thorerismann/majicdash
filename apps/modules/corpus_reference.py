"""Metric columns shared by the CSV builder and live comparison."""

from pathlib import Path
from .pos_highlighting import SPACY_METRIC_KEYS
from .sentiment_analysis import SENTIMENT_METRIC_KEYS

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REFERENCE_PATH = DATA_DIR / "corpus_reference.csv"
BASE_METRIC_KEYS = (
    "word_count", "paragraph_count", "avg_paragraph_words", "sentence_count",
    "avg_sentence_length", "syllable_count", "syllables_per_word", "char_count",
    "reading_ease", "fk_grade",
)
STYLE_METRIC_KEYS = ("reading_minutes", "unique_words", "lexical_diversity", "passive_count",
                    "hedge_count", "hedges_per_100_words", "transition_count", "transitions_per_100_words",
                    "longest_paragraph", "long_sentence_ratio", "short_sentence_ratio")
METRIC_KEYS = BASE_METRIC_KEYS + STYLE_METRIC_KEYS + SENTIMENT_METRIC_KEYS + SPACY_METRIC_KEYS


def comparison_values(metrics: dict, extra: dict, sentiment: dict | None = None) -> dict:
    """Use total paragraph count, never the six-paragraph display preview."""
    return {
        **{key: metrics[key] for key in METRIC_KEYS if key in metrics},
        **{key: extra[key] for key in METRIC_KEYS if key in extra},
        **(sentiment or {}),
        "paragraph_count": extra["paragraph_count"],
        "avg_paragraph_words": metrics["word_count"] / extra["paragraph_count"] if extra["paragraph_count"] else 0.0,
        "syllables_per_word": metrics["syllable_count"] / metrics["word_count"] if metrics["word_count"] else 0.0,
    }
