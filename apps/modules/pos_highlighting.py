"""Local spaCy helpers for visualising parts of speech in submitted writing."""

from __future__ import annotations

import html
from collections import Counter
from functools import lru_cache
from typing import Any

try:
    import spacy
    from spacy.language import Language
except ImportError:
    spacy = None
    Language = Any


SUPPORTED_POS = {"VERB", "NOUN", "ADV", "ADJ"}
POS_COLORS = {
    "VERB": ("verb", "#ffe08a"),
    "NOUN": ("noun", "#a7f3d0"),
    "ADV": ("adverb", "#bfdbfe"),
    "ADJ": ("adjective", "#fbcfe8"),
}


@lru_cache(maxsize=2)
def load_pos_model(model_name: str) -> Language:
    """Load each language model once per Streamlit process."""
    if spacy is None:
        raise RuntimeError("spaCy is not installed")
    return spacy.load(model_name)


def highlight_parts_of_speech(text: str, model_name: str, selected_pos: set[str]) -> tuple[str, Counter[str]]:
    """Return escaped, safe HTML with the selected word classes highlighted."""
    unsupported_pos = selected_pos - SUPPORTED_POS
    if unsupported_pos:
        raise ValueError(f"Unsupported parts of speech: {unsupported_pos}")

    doc = load_pos_model(model_name)(text)
    pieces: list[str] = []
    counts: Counter[str] = Counter()

    for token in doc:
        escaped_token = html.escape(token.text)
        if token.pos_ in selected_pos:
            counts[token.pos_] += 1
            label, color = POS_COLORS[token.pos_]
            pieces.append(
                f'<mark title="{label}" style="background: {color}; border-radius: 0.2rem; '
                f'padding: 0.05rem 0.16rem;">{escaped_token}</mark>'
            )
        else:
            pieces.append(escaped_token)
        pieces.append(html.escape(token.whitespace_))

    return "".join(pieces), counts


def part_of_speech_counts(text: str, model_name: str) -> Counter[str]:
    """Count the four word classes shown by the interface."""
    doc = load_pos_model(model_name)(text)
    return Counter(token.pos_ for token in doc if token.pos_ in SUPPORTED_POS)
