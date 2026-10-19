"""Local English/French VADER scoring and the dashboard's sentiment tab."""

from __future__ import annotations

import csv
import io
from collections import Counter
from dataclasses import dataclass
from typing import TypedDict

import streamlit as st


SENTIMENT_THRESHOLD = 0.05
LANGUAGE_NAMES = {"en": "English", "fr": "French"}


class SentimentScores(TypedDict):
    compound: float
    positive: float
    neutral: float
    negative: float
    label: str


class SentenceScores(SentimentScores):
    sentence: int
    text: str
    language: str


@dataclass
class SentimentReport:
    overall: SentimentScores
    sentences: list[SentenceScores]
    sentence_counts: Counter[str]
    mean_sentence_compound: float | None


def classify_sentiment(compound: float) -> str:
    if compound >= SENTIMENT_THRESHOLD:
        return "Positive"
    if compound <= -SENTIMENT_THRESHOLD:
        return "Negative"
    return "Neutral"


@st.cache_resource
def get_sentiment_analyzer(language_code: str = "en"):
    if language_code not in LANGUAGE_NAMES:
        raise ValueError(f"Unsupported sentiment language: {language_code}")
    try:
        if language_code == "fr":
            from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
        else:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            f"{LANGUAGE_NAMES[language_code]} sentiment support could not load. "
            "Run `python -m pip install -r requirements.txt` in the app's environment and restart the app."
        ) from exc
    return SentimentIntensityAnalyzer()


def _score(text: str, analyzer) -> SentimentScores:
    # Preserve emphasis and punctuation; normalize typographic contractions only.
    scores = analyzer.polarity_scores(text.translate(str.maketrans({"’": "'", "ʼ": "'"})))
    return {
        "compound": scores["compound"],
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"],
        "label": classify_sentiment(scores["compound"]),
    }


@st.cache_data(show_spinner=False)
def analyze_sentiment(text: str, language_code: str = "en") -> SentimentScores:
    return _score(text, get_sentiment_analyzer(language_code))


@st.cache_resource
def _sentence_splitter(language_code: str):
    import spacy

    # Language-specific tokenization handles abbreviations without loading a model.
    nlp = spacy.blank(language_code)
    nlp.add_pipe("sentencizer")
    return nlp


@st.cache_data(show_spinner=False)
def analyze_sentiment_report(text: str, language_code: str = "en") -> SentimentReport:
    analyzer = get_sentiment_analyzer(language_code)
    sentences: list[SentenceScores] = []
    for span in _sentence_splitter(language_code)(text).sents:
        sentence = span.text.strip()
        if sentence:
            sentences.append({
                **_score(sentence, analyzer),
                "sentence": len(sentences) + 1,
                "text": sentence,
                "language": language_code,
            })
    return SentimentReport(
        overall=analyze_sentiment(text, language_code),
        sentences=sentences,
        sentence_counts=Counter(row["label"] for row in sentences),
        mean_sentence_compound=(sum(row["compound"] for row in sentences) / len(sentences)) if sentences else None,
    )


def sentiment_csv(report: SentimentReport) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(
        output, fieldnames=["sentence", "language", "text", "label", "compound", "positive", "neutral", "negative"]
    )
    writer.writeheader()
    writer.writerows(report.sentences)
    return output.getvalue()


SENTIMENT_METRIC_KEYS = (
    "sentiment_compound", "sentiment_positive", "sentiment_neutral", "sentiment_negative",
    "sentiment_sentence_mean", "sentiment_sentence_min", "sentiment_sentence_max",
    "sentiment_positive_sentence_ratio", "sentiment_neutral_sentence_ratio", "sentiment_negative_sentence_ratio",
)


def sentiment_dimensions(text: str, language_code: str) -> dict:
    report = analyze_sentiment_report(text, language_code)
    compounds = [row["compound"] for row in report.sentences]
    return {
        **{f"sentiment_{key}": report.overall[key] for key in ("compound", "positive", "neutral", "negative")},
        "sentiment_sentence_mean": report.mean_sentence_compound,
        "sentiment_sentence_min": min(compounds) if compounds else None,
        "sentiment_sentence_max": max(compounds) if compounds else None,
        **{f"sentiment_{label.lower()}_sentence_ratio": report.sentence_counts[label] / len(compounds) if compounds else 0.0
           for label in ("Positive", "Neutral", "Negative")},
    }


def render_sentiment(text: str, language_code: str = "en") -> None:
    import altair as alt

    report = analyze_sentiment_report(text, language_code)
    scores = report.overall
    st.subheader("Sentiment")
    st.caption(f"{LANGUAGE_NAMES[language_code]} VADER · local analysis · positive ≥ +0.05, negative ≤ −0.05")
    if not report.sentences:
        st.info("Enter some text to see sentiment scores.")
        return

    overall, positive, neutral, negative = st.columns(4)
    overall.metric("Document compound", f"{scores['compound']:+.3f}")
    positive.metric("Positive share", f"{scores['positive']:.1%}")
    neutral.metric("Neutral share", f"{scores['neutral']:.1%}")
    negative.metric("Negative share", f"{scores['negative']:.1%}")
    st.write(f"Document classification: **{scores['label']}**")
    st.caption("Shares describe VADER's sentiment proportions, not confidence probabilities or exact percentages of words.")

    st.subheader("Sentiment across sentences")
    pos, neu, neg, mean = st.columns(4)
    pos.metric("Positive sentences", report.sentence_counts["Positive"])
    neu.metric("Neutral sentences", report.sentence_counts["Neutral"])
    neg.metric("Negative sentences", report.sentence_counts["Negative"])
    mean.metric("Mean sentence compound", f"{report.mean_sentence_compound:+.3f}")
    st.caption("The mean gives every sentence equal weight. It is separate from the document compound score.")
    if report.sentence_counts["Positive"] and report.sentence_counts["Negative"]:
        st.info("Both positive and negative sentences occur here. A single document score can hide that mixture.")

    chart = alt.Chart(alt.Data(values=report.sentences)).mark_line(point=True).encode(
        x=alt.X("sentence:Q", title="Sentence", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y("compound:Q", title="Compound sentiment", scale=alt.Scale(domain=[-1, 1])),
        tooltip=[alt.Tooltip("sentence:Q"), alt.Tooltip("compound:Q", format="+.3f"), "label:N", "text:N"],
    )
    st.altair_chart(chart, use_container_width=True)

    st.subheader("Passages to review")
    for column, label, reverse in zip(st.columns(2), ("Positive", "Negative"), (True, False)):
        with column:
            st.markdown(f"**Strongest {label.lower()} sentences**")
            ranked = sorted(
                (row for row in report.sentences if row["label"] == label),
                key=lambda row: row["compound"], reverse=reverse,
            )[:3]
            if not ranked:
                st.caption(f"No sentences classified as {label.lower()}.")
            for row in ranked:
                st.caption(f"Sentence {row['sentence']} · {row['compound']:+.3f}")
                st.text(row["text"])

    with st.expander("All sentence scores"):
        st.dataframe(report.sentences, hide_index=True, use_container_width=True)
    st.download_button(
        "Download sentence scores (CSV)", data=sentiment_csv(report),
        file_name=f"sentiment_{language_code}.csv", mime="text/csv", key="sentiment_csv",
    )
    with st.expander("How to read VADER"):
        st.write(
            "VADER uses a sentiment dictionary and rules for negation, intensifiers, capitals, and punctuation. "
            "Compound runs from −1 (negative) to +1 (positive). Near zero can mean little emotional wording "
            "or opposing signals that cancel; it does not prove objectivity."
        )
        st.write(
            "Use the sentence chart to locate shifts and review the actual passages. "
            "Quotes and descriptions of bad events can score negatively without the author taking a negative stance. "
            "Sarcasm, implied meaning, and domain-specific language can be missed. "
            "These scores do not measure writing quality, factual accuracy, or political bias."
        )
        st.write(
            "French uses a separate VADER adaptation with a translated lexicon. "
            "Its scores are not calibrated to be directly comparable with English scores."
        )
        st.markdown(
            "[VADER scoring documentation](https://github.com/cjhutto/vaderSentiment#about-the-scoring) · "
            "[French adaptation](https://github.com/thomas7lieues/vader_FR)"
        )
