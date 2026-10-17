"""Local text counts, readability measures, and writing diagnostics.

analyze_basic_metrics returns counts, reading time, sentence length, and
language-specific readability scores (Flesch-Kandel for French, Flesch for English).
analyze_additional_tools adds paragraph lengths, lexical diversity, hedges,
transitions, and passive-voice flags. One cached spaCy analysis supplies tokens,
sentences, lemmas, and dependencies to all three analyses, reusing the models
loaded by pos_highlighting. Words are spaCy tokens containing letters (including
contractions and elisions); numbers and standalone punctuation do not count.
Repetition uses content-word lemmas (nouns, verbs, adjectives), while lexical
diversity uses surface forms. textstat supplies English readability and syllable
counts. This module also renders the basic statistics and diagnostics.
"""

import streamlit as st
import re
from math import ceil
from textstat.textstat import textstatistics
from .language import DEFAULT_LANGUAGE, get_words, translator
from collections import Counter
from spacy.matcher import PhraseMatcher
from .pos_highlighting import POS_MODELS, load_pos_model, parse_pos_text


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def _analyze_text(text: str, language_code: str) -> dict:
    """Cache small derived values, not a Doc and its model vocabulary."""
    if language_code not in POS_MODELS:
        raise ValueError(f"Unsupported analysis language: {language_code}")
    doc = parse_pos_text(text, POS_MODELS[language_code])
    words = [token for token in doc if any(char.isalpha() for char in token.text)]
    word_indices = {token.i for token in words}
    vocabulary = get_words(language_code)["vocabulary"]
    stopwords = set(vocabulary["stopwords"])
    lemmas = [
        token.lemma_.lower() for token in words
        if not token.is_stop and token.lower_ not in stopwords
        and token.lemma_.lower() not in stopwords
        and token.pos_ in {"NOUN", "VERB", "ADJ"}
    ]
    sentences = [
        (text[span.start_char:span.end_char].strip(),
         sum(token.i in word_indices for token in span))
        for span in doc.sents if span.text.strip()
    ]

    # Paragraph boundaries are formatting, not grammatical sentence boundaries.
    paragraph_lengths = []
    word_iter = iter(words)
    token = next(word_iter, None)
    start = 0
    ends = [(match.start(), match.end()) for match in re.finditer(r"\n\s*\n", text)]
    for end, next_start in ends + [(len(text), len(text))]:
        count = 0
        while token is not None and token.idx < end:
            count += 1
            token = next(word_iter, None)
        if text[start:end].strip():
            paragraph_lengths.append(count)
        start = next_start

    # Match the editorial phrase lists on spaCy token boundaries, ignoring case.
    nlp = load_pos_model(POS_MODELS[language_code])
    matcher = PhraseMatcher(doc.vocab, attr="LOWER")
    for category in ("hedges", "transitions"):
        for phrase in vocabulary[category]:
            normalized = phrase.translate(str.maketrans({"’": "'", "ʼ": "'"}))
            matcher.add(f"{category}:{phrase}", [nlp.make_doc(normalized)])
    matches = [doc.vocab.strings[match_id].split(":", 1) for match_id, _, _ in matcher(doc)]

    # Count passive predicates once, even if both subject and auxiliary flag it.
    passive_heads = {
        token.head.i for token in doc
        if token.dep_ in {"auxpass", "nsubjpass", "csubjpass", "aux:pass", "nsubj:pass", "csubj:pass"}
    }
    return {
        "words": [token.lower_ for token in words],
        "lemmas": lemmas,
        "sentences": sentences,
        "paragraph_lengths": paragraph_lengths,
        "passive_count": len(passive_heads),
        "hedge_count": sum(category == "hedges" for category, _ in matches),
        "transition_hits": sorted({phrase for category, phrase in matches if category == "transitions"}),
    }


def split_sentences(text: str, language_code: str = DEFAULT_LANGUAGE) -> list[str]:
    return [sentence for sentence, _ in _analyze_text(text, language_code)["sentences"]]


def tokenize_words(text: str, language_code: str = DEFAULT_LANGUAGE) -> list[str]:
    return _analyze_text(text, language_code)["words"]


@st.cache_data(show_spinner=False)
def analyze_basic_metrics(text: str, language_code: str = DEFAULT_LANGUAGE) -> dict[str, float | int | str | None]:
    # Each analysis has its own language setting, including concurrent sessions.
    stats = textstatistics()
    stats.set_lang(language_code)
    analysis = _analyze_text(text, language_code)
    words = analysis["words"]
    sentences = analysis["sentences"]
    word_count = len(words)
    sentence_count = len(sentences)
    syllable_count = stats.syllable_count(text) if text.strip() else 0
    char_count = len(text)
    avg_sentence_length = len(words) / max(1, len(sentences))
    reading_minutes = len(words) / 200 if words else 0.0

    if not text.strip():
        reading_ease = 0.0
    elif language_code == "fr":
        # Flesch-Kandel (de Landsheere) formula adapted for French:
        # 207 - (1.015 * avg_sentence_length) - (73.6 * (syllables / words))
        syllables_per_word = syllable_count / max(1, word_count)
        reading_ease = round(207.0 - (1.015 * avg_sentence_length) - (73.6 * syllables_per_word), 2)
    else:
        reading_ease = stats.flesch_reading_ease(text)

    return {
        "word_count": word_count,
        "sentence_count": sentence_count,
        "syllable_count": syllable_count,
        "char_count": char_count,
        "avg_sentence_length": avg_sentence_length,
        "reading_minutes": reading_minutes,
        "fk_grade": (stats.flesch_kincaid_grade(text) if text.strip() else 0.0) if language_code == "en" else None,
        "reading_ease": reading_ease,
        "language_code": language_code,
    }


@st.cache_data(show_spinner=False)
def analyze_additional_tools(text: str, language_code: str = DEFAULT_LANGUAGE) -> dict[str, object]:
    analysis = _analyze_text(text, language_code)
    words = analysis["words"]
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if words else 0.0
    paragraph_lengths = analysis["paragraph_lengths"]
    longest_paragraph = max(paragraph_lengths) if paragraph_lengths else 0

    return {
        "paragraph_count": len(paragraph_lengths),
        "lexical_diversity": lexical_diversity,
        "unique_words": unique_words,
        "passive_count": analysis["passive_count"],
        "hedge_count": analysis["hedge_count"],
        "transition_hits": analysis["transition_hits"][:8],
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
    analysis = _analyze_text(text, language_code)
    words = analysis["words"]
    sentences = analysis["sentences"]
    frequencies = Counter(analysis["lemmas"])
    repeated_terms = [(word, count) for word, count in frequencies.most_common(8) if count > 2]

    long_sentences = [sentence for sentence, length in sentences if length >= 28]
    short_sentences = [sentence for sentence, length in sentences if length <= 5]
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
