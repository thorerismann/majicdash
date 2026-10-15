"""English/French style feedback and rendering for the Style tab."""

from __future__ import annotations

import re
from collections import Counter

import streamlit as st

from .text_utils import split_sentences, tokenize_words


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "he",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "that",
    "the",
    "to",
    "was",
    "were",
    "will",
    "with",
}
FRENCH_STOPWORDS = set(
    "le la les un une des du de à et ou mais donc car ni en dans par pour sur avec "
    "qui que quoi dont où je tu il elle on nous vous ils elles me te se y ce cet cette ces "
    "mon ton son notre votre leur leurs ne pas plus est sont a ont être avoir au aux "
    "c d j l m n s t qu était étaient été aussi comme tout tous toute toutes".split()
)


@st.cache_data(show_spinner=False)
def analyze_style(text: str, language_code: str = "en") -> dict[str, object]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    if language_code == "fr":
        words = [re.sub(r"^(?:[cdjlmnst]|qu)'", "", word) for word in words]
    stopwords = FRENCH_STOPWORDS if language_code == "fr" else STOPWORDS
    lowered = [word for word in words if word not in stopwords and len(word) > 3]
    frequencies = Counter(lowered)
    repeated_terms = [(word, count) for word, count in frequencies.most_common(8) if count > 2]

    long_sentences = [sentence for sentence in sentences if len(tokenize_words(sentence)) >= 28]
    short_sentences = [sentence for sentence in sentences if len(tokenize_words(sentence)) <= 5]
    exclamation_count = text.count("!")
    question_count = text.count("?")

    strengths: list[str] = []
    improvements: list[str] = []

    avg_sentence_length = len(words) / max(1, len(sentences))
    if 12 <= avg_sentence_length <= 22:
        strengths.append("Sentence length is generally in a readable range.")
    else:
        improvements.append("Sentence length is uneven; aim for a steadier rhythm across paragraphs.")

    if len(long_sentences) == 0:
        strengths.append("No overloaded long sentences were detected.")
    else:
        improvements.append("A few sentences are long enough to risk losing the reader.")

    if repeated_terms:
        repeated_labels = ", ".join(f"{word} ({count})" for word, count in repeated_terms[:5])
        improvements.append(f"Some words are repeated often: {repeated_labels}.")
    else:
        strengths.append("Word repetition is limited, which helps the draft feel less redundant.")

    if exclamation_count == 0 and question_count <= 2:
        strengths.append("Punctuation use reads as controlled rather than overly dramatic.")
    else:
        improvements.append("Punctuation may be adding extra intensity; review exclamation marks and rhetorical questions.")

    if short_sentences:
        improvements.append("Very short sentences appear in the draft; make sure they are intentional for emphasis.")

    return {
        "strengths": strengths[:3],
        "improvements": improvements[:4],
        "repeated_terms": repeated_terms[:8],
        "long_sentences": long_sentences[:5],
        "short_sentences": short_sentences[:5],
    }


def render_style(style: dict[str, object], show_examples: bool = True) -> None:
    st.subheader("Strengths")
    if style["strengths"]:
        for item in style["strengths"]:
            st.write(f"- {item}")
    else:
        st.write("- No standout strengths were identified by the local ruleset.")

    st.subheader("Revision Priorities")
    if style["improvements"]:
        for item in style["improvements"]:
            st.write(f"- {item}")
    else:
        st.write("- No major local revision flags were raised.")

    if style["repeated_terms"]:
        st.subheader("Repeated Terms")
        repeated_text = ", ".join(f"{word} ({count})" for word, count in style["repeated_terms"])
        st.write(repeated_text)

    if show_examples:
        if style["long_sentences"]:
            st.subheader("Long Sentences to Review")
            for sentence in style["long_sentences"]:
                st.code(sentence, language=None)
        if style["short_sentences"]:
            st.subheader("Very Short Sentences")
            for sentence in style["short_sentences"]:
                st.code(sentence, language=None)
