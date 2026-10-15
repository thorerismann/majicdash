"""Local spaCy helpers for visualising parts of speech in submitted writing."""

from __future__ import annotations

import html
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

try:
    import spacy
    from spacy.language import Language
    from spacy.tokens import Doc
except ImportError:
    spacy = None
    Language = Any
    Doc = Any


POS_MODELS = {"en": "en_core_web_md", "fr": "fr_core_news_md"}
POS_LANGUAGE_NAMES = {"en": "English", "fr": "French"}
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


def parse_pos_text(text: str, model_name: str) -> Doc:
    """Normalize apostrophes for tagging without changing character offsets."""
    normalized = text.translate(str.maketrans({"’": "'", "ʼ": "'"}))
    return load_pos_model(model_name)(normalized)


def display_pos(pos: str) -> str:
    """The interface's Verb category includes spaCy's auxiliary verbs."""
    return "VERB" if pos == "AUX" else pos


@dataclass
class PosAnalysis:
    html: str
    counts: Counter[str]
    highlighted_counts: Counter[str]


def analyze_parts_of_speech(text: str, model_name: str, selected_pos: set[str]) -> PosAnalysis:
    """Tag once and use the same classification for highlighting and all counts."""
    unsupported_pos = selected_pos - SUPPORTED_POS
    if unsupported_pos:
        raise ValueError(f"Unsupported parts of speech: {unsupported_pos}")

    doc = parse_pos_text(text, model_name)
    pieces: list[str] = []
    counts: Counter[str] = Counter()
    highlighted_counts: Counter[str] = Counter()

    for token in doc:
        # Render the original spelling and punctuation using the aligned offsets.
        escaped_token = html.escape(text[token.idx:token.idx + len(token)])
        pos = display_pos(token.pos_)
        if pos in SUPPORTED_POS:
            counts[pos] += 1
        if pos in selected_pos:
            highlighted_counts[pos] += 1
            label, color = POS_COLORS[pos]
            pieces.append(
                f'<mark title="{label}" style="background: {color}; border-radius: 0.2rem; '
                f'padding: 0.05rem 0.16rem;">{escaped_token}</mark>'
            )
        else:
            pieces.append(escaped_token)
        pieces.append(html.escape(token.whitespace_))

    return PosAnalysis("".join(pieces), counts, highlighted_counts)


def highlight_parts_of_speech(text: str, model_name: str, selected_pos: set[str]) -> tuple[str, Counter[str]]:
    """Return escaped, safe HTML with the selected word classes highlighted."""
    result = analyze_parts_of_speech(text, model_name, selected_pos)
    return result.html, result.highlighted_counts


def part_of_speech_counts(text: str, model_name: str) -> Counter[str]:
    """Count the four word classes shown by the interface."""
    doc = parse_pos_text(text, model_name)
    return Counter(display_pos(token.pos_) for token in doc if display_pos(token.pos_) in SUPPORTED_POS)


def render_pos_highlighter(text: str, language_code: str = "en") -> None:
    import streamlit as st

    st.subheader("Parts of Speech")
    st.write("Select one or more word classes to highlight matching words in your submitted text. Verb includes auxiliary verbs.")
    pos_options = (("Verb", "VERB"), ("Noun", "NOUN"), ("Adverb", "ADV"), ("Adjective", "ADJ"))
    toggle_columns = st.columns(4)
    selected_pos = {
        pos
        for column, (label, pos) in zip(toggle_columns, pos_options)
        if column.checkbox(label, value=label == "Verb", key=f"en_pos_{pos}")
    }

    try:
        model_name = POS_MODELS[language_code]
        result = analyze_parts_of_speech(text, model_name, selected_pos)
        counts = result.counts
        highlighted_html = result.html
        highlighted_counts = result.highlighted_counts
    except RuntimeError:
        st.error("spaCy could not load. Run `python -m pip install -r requirements.txt` in the app's environment and restart the app.")
        return
    except OSError:
        st.error(f"The {POS_LANGUAGE_NAMES[language_code]} language model could not load. Run `python -m pip install -r requirements.txt` in the app's environment and restart the app.")
        return

    highlighted_total = sum(highlighted_counts.values())
    selected_labels = [label.lower() for label, pos in pos_options if pos in selected_pos]
    selection_summary = ", ".join(selected_labels) if selected_labels else "no word classes"
    st.caption(
        f"{highlighted_total} words highlighted ({selection_summary}). "
        f"Verbs: {counts['VERB']} · Nouns: {counts['NOUN']} · Adverbs: {counts['ADV']} · Adjectives: {counts['ADJ']}"
    )
    st.markdown(
        f'<div style="line-height: 1.8; padding: 1rem; border: 1px solid #d1d5db; border-radius: 0.5rem; white-space: pre-wrap;">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )
