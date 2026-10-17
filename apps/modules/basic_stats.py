"""Local text counts, readability measures, and writing diagnostics.

analyze_basic_metrics returns counts, reading time, sentence length, and
language-specific readability scores. analyze_additional_tools adds paragraph
lengths, lexical diversity, hedges, transitions, and English passive-voice flags.
Vocabulary comes from language.py; tokenization comes from text_utils.py.
Results are cached by text and language. format_reading_time formats a translated
duration for the dashboard. This module does not render tabs or call an API.
"""

import streamlit as st
import re
from math import ceil
from textstat.textstat import textstatistics
from .language import DEFAULT_LANGUAGE, get_words, translator
from collections import Counter


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    # Keep Unicode letters (including œ) and normalize typographic apostrophes.
    return re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", text.lower().replace("’", "'"))


@st.cache_data(show_spinner=False)
def analyze_basic_metrics(text: str, language_code: str = DEFAULT_LANGUAGE) -> dict[str, float | int | str | None]:
    # Each analysis has its own language setting, including concurrent sessions.
    stats = textstatistics()
    stats.set_lang(language_code)
    words = tokenize_words(text)
    sentences = split_sentences(text)
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = stats.syllable_count(text)
    char_count = len(text)
    avg_sentence_length = len(words) / max(1, len(sentences))
    reading_minutes = len(words) / 200 if words else 0.0

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "char_count": char_count,
        "avg_sentence_length": avg_sentence_length,
        "reading_minutes": reading_minutes,
        "fk_grade": (stats.flesch_kincaid_grade(text) if text.strip() else 0.0) if language_code == "en" else None,
        "reading_ease": stats.flesch_reading_ease(text) if text.strip() else 0.0,
        "language_code": language_code,
    }


@st.cache_data(show_spinner=False)
def analyze_additional_tools(text: str, language_code: str = DEFAULT_LANGUAGE) -> dict[str, object]:
    words = tokenize_words(text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if words else 0.0
    vocabulary = get_words(language_code)["vocabulary"]
    auxiliaries = "|".join(re.escape(word) for word in vocabulary["passive_auxiliaries"])
    passive_hits = re.findall(r"\b(?:" + auxiliaries + r")\s+\w+(?:ed|en)\b", text.lower()) if auxiliaries else []
    hedges = vocabulary["hedges"]
    transitions = vocabulary["transitions"]
    lowered_text = text.lower().replace("’", "'")
    hedge_count = sum(len(re.findall(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lowered_text)) for phrase in hedges)

    transition_hits: list[str] = []
    for phrase in sorted(transitions):
        if re.search(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", lowered_text):
            transition_hits.append(phrase)

    paragraph_lengths = [len(tokenize_words(paragraph)) for paragraph in paragraphs]
    longest_paragraph = max(paragraph_lengths) if paragraph_lengths else 0

    return {
        "paragraph_count": len(paragraphs) if paragraphs else (1 if text.strip() else 0),
        "lexical_diversity": lexical_diversity,
        "unique_words": unique_words,
        "passive_count": len(passive_hits) if language_code == "en" else None,
        "hedge_count": hedge_count,
        "transition_hits": transition_hits[:8],
        "longest_paragraph": longest_paragraph,
        "paragraph_lengths": paragraph_lengths[:6],
    }

def format_reading_time(minutes: float, language_code: str = DEFAULT_LANGUAGE) -> str:
    t = translator(language_code)
    if minutes < 1:
        seconds = max(10, ceil(minutes * 60))
        return t("seconds", seconds=seconds)
    return t("minutes", minutes=minutes)


@st.cache_data(show_spinner=False)
def build_summary(metrics: dict[str, float | int | str | None], sentiment: dict[str, float | str]) -> list[str]:
    t = translator(str(metrics["language_code"]))
    grade = metrics["fk_grade"]
    average = float(metrics["avg_sentence_length"])
    return [
        t("summary_counts", words=int(metrics["word_count"]), sentences=int(metrics["sentence_count"])),
        t("summary_grade", grade=grade, average=average) if grade is not None else
        t("summary_ease", ease=float(metrics["reading_ease"]), average=average),
        t("summary_sentiment", label=t("sentiment_" + str(sentiment["label"]).lower()).lower()),
    ]


def render_basic_stats(metrics, sentiment, language_code):
    t = translator(language_code)
    st.subheader(t("summary"))
    summary = build_summary(metrics, sentiment)
    for sentence in summary:
        st.write(f"- {sentence}")

    stats_col1, stats_col2 = st.columns(2)
    with stats_col1:
        st.markdown(t("counts"))
        st.write(t("characters_count", count=int(metrics["char_count"])))
        st.write(t("syllables_count", count=int(metrics["syllable_count"])))
        st.write(t("reading_time_value", time=format_reading_time(float(metrics["reading_minutes"]), language_code)))
    with stats_col2:
        st.markdown(t("interpretation"))
        st.write(t("grade_help"))
        st.write(t("reading_help"))
        st.write(t("presence_help"))

def render_diagnostics(t, extra_tools):
        st.subheader(t("diagnostics"))
        st.write(
            t("diagnostics_help")
            )
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns(4)
        tool_col1.metric(t("paragraphs"), int(extra_tools["paragraph_count"]))
        tool_col2.metric(t("unique_words"), int(extra_tools["unique_words"]))
        tool_col3.metric(t("lexical_diversity"), f"{float(extra_tools['lexical_diversity']):.2f}")
        tool_col4.metric(t("passive_flags"), int(extra_tools["passive_count"]) if extra_tools["passive_count"] is not None else t("not_available"))

        tool_col5, tool_col6, tool_col7 = st.columns(3)
        tool_col5.metric(t("hedges"), int(extra_tools["hedge_count"]))
        tool_col6.metric(t("transitions"), len(extra_tools["transition_hits"]))
        tool_col7.metric(t("longest_paragraph"), t("word_count", count=int(extra_tools["longest_paragraph"])))

        st.markdown(t("diagnostics_meaning"))
        st.write(t("paragraph_help"))
        st.write(t("lexical_help"))
        st.write(t("passive_help"))
        st.write(t("hedge_help"))
        st.write(t("transition_help"))

        if extra_tools["transition_hits"]:
            st.markdown(t("detected_transitions"))
            st.write(", ".join(str(item) for item in extra_tools["transition_hits"]))

        if extra_tools["paragraph_lengths"]:
            st.markdown(t("paragraph_sizes"))
            st.write(", ".join(t("word_count", count=count) for count in extra_tools["paragraph_lengths"]))




@st.cache_data(show_spinner=False)
def analyze_style(text: str, language_code: str = DEFAULT_LANGUAGE) -> dict[str, object]:
    t = translator(language_code)
    words = tokenize_words(text)
    sentences = split_sentences(text)
    vocabulary = get_words(language_code)["vocabulary"]
    elisions = "|".join(re.escape(prefix) for prefix in vocabulary["elision_prefixes"])
    if elisions:
        words = [re.sub(r"^(?:" + elisions + r")'", "", word) for word in words]
    stopwords = set(vocabulary["stopwords"])
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
        strengths.append(t("style_readable"))
    else:
        improvements.append(t("style_rhythm"))

    if len(long_sentences) == 0:
        strengths.append(t("style_no_long"))
    else:
        improvements.append(t("style_long"))

    if repeated_terms:
        repeated_labels = ", ".join(f"{word} ({count})" for word, count in repeated_terms[:5])
        improvements.append(t("style_repetition", terms=repeated_labels))
    else:
        strengths.append(t("style_no_repetition"))

    if exclamation_count == 0 and question_count <= 2:
        strengths.append(t("style_punctuation"))
    else:
        improvements.append(t("style_intensity"))

    if short_sentences:
        improvements.append(t("style_short"))

    return {
        "strengths": strengths[:3],
        "improvements": improvements[:4],
        "repeated_terms": repeated_terms[:8],
        "long_sentences": long_sentences[:5],
        "short_sentences": short_sentences[:5],
    }


def render_style(style: dict[str, object], show_examples: bool = True, language_code: str = DEFAULT_LANGUAGE) -> None:
    t = translator(language_code)
    st.subheader(t("strengths"))
    if style["strengths"]:
        for item in style["strengths"]:
            st.write(f"- {item}")
    else:
        st.write(t("no_strengths"))

    st.subheader(t("revision_priorities"))
    if style["improvements"]:
        for item in style["improvements"]:
            st.write(f"- {item}")
    else:
        st.write(t("no_revision_flags"))

    if style["repeated_terms"]:
        st.subheader(t("repeated_terms"))
        repeated_text = ", ".join(f"{word} ({count})" for word, count in style["repeated_terms"])
        st.write(repeated_text)

    if show_examples:
        if style["long_sentences"]:
            st.subheader(t("long_sentences"))
            for sentence in style["long_sentences"]:
                st.code(sentence, language=None)
        if style["short_sentences"]:
            st.subheader(t("short_sentences"))
            for sentence in style["short_sentences"]:
                st.code(sentence, language=None)
