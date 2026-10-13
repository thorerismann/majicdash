from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import textstat
from langdetect import DetectorFactory, LangDetectException, detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from modules.pos_highlighting import highlight_parts_of_speech, part_of_speech_counts
from modules.inference import request_angle_analysis


DetectorFactory.seed = 0

APP_TITLE = "Writer Dashboard"
ANGLE_MACHINE_PATH = Path(__file__).resolve().parents[1] / "assets" / "code_machine_angle.html"
POS_MODEL = "en_core_web_sm"
DEFAULT_TEXT = """Paste a student paragraph, memo, or draft here to analyze clarity, tone, and sentence flow.

This first version is English-first and local-only. It does not require any AI key."""
ANGLE_WHEELS: list[tuple[str, list[str]]] = [
    ("Temporalite", ["avant-papier", "Live (T0)", "T1", "T2", "T3", "Affaire", "Original"]),
    ("Fonction 1", ["Savoir (faits)", "Faire (action)", "Ressentir (emotion)", "Comprendre (contexte)"]),
    ("Fonction 2", ["Mise a jour", "Implication", "Aide", "Connexion", "Inspiration", "Divertissement", "Perspective", "Didactique"]),
    ("Perspective", ["Faits", "Bilan", "Precedents", "Causes", "Consequences", "Acteurs", "Reactions", "A cotes"]),
    ("Format/genre", ["Reportage", "Scrollytelling", "Dataviz", "Eclairage", "Compte rendu", "Synthese", "Sondage", "Enquete", "J'ai teste", "Portrait", "Interview", "Recit", "Etc."]),
    ("Dominante", ["Pratique", "Politique", "Societale", "Sociologique", "Ecologique", "Economique", "Philosophique", "Historique", "Intime", "Technologique", "De solution", "Etc."]),
]
ANGLE_FUNCTION_MAPPING = {
    "Savoir (faits)": ["Mise a jour", "Implication"],
    "Faire (action)": ["Aide", "Connexion"],
    "Ressentir (emotion)": ["Inspiration", "Divertissement"],
    "Comprendre (contexte)": ["Perspective", "Didactique"],
}
TRANSITION_WORDS = {
    "however",
    "therefore",
    "moreover",
    "meanwhile",
    "furthermore",
    "instead",
    "although",
    "because",
    "first",
    "second",
    "finally",
    "for example",
    "for instance",
    "in contrast",
    "as a result",
}
HEDGING_WORDS = {
    "maybe",
    "perhaps",
    "possibly",
    "somewhat",
    "rather",
    "fairly",
    "quite",
    "likely",
    "apparently",
    "seems",
    "suggests",
}
TOOL_LINKS = [
    ("Streamlit", "https://streamlit.io", "App framework"),
    ("textstat", "https://pypi.org/project/textstat/", "Readability metrics"),
    ("VADER Sentiment", "https://github.com/cjhutto/vaderSentiment", "Local sentiment scoring"),
    ("langdetect", "https://pypi.org/project/langdetect/", "Language detection"),
    ("ChatGPT", "https://chatgpt.com", "Optional external AI critique"),
    ("Claude", "https://claude.ai", "Optional external AI critique"),
    ("Gemini", "https://gemini.google.com", "Optional external AI critique"),
]
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


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"[A-Za-z][A-Za-z'-]*", text.lower())


def normalize_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


@st.cache_resource
def get_sentiment_analyzer() -> SentimentIntensityAnalyzer:
    return SentimentIntensityAnalyzer()


@st.cache_data(show_spinner=False)
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


@st.cache_data(show_spinner=False)
def analyze_basic_metrics(text: str) -> dict[str, float | int | str]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    word_count = textstat.lexicon_count(text, removepunct=True)
    sentence_count = max(1, textstat.sentence_count(text))
    syllable_count = textstat.syllable_count(text)
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
        "fk_grade": textstat.flesch_kincaid_grade(text) if text.strip() else 0.0,
        "reading_ease": textstat.flesch_reading_ease(text) if text.strip() else 0.0,
    }


@st.cache_data(show_spinner=False)
def analyze_sentiment(text: str) -> dict[str, float | str]:
    scores = get_sentiment_analyzer().polarity_scores(text)
    compound = scores["compound"]
    if compound >= 0.35:
        label = "Positive / encouraging"
    elif compound <= -0.35:
        label = "Negative / critical"
    else:
        label = "Neutral / measured"

    return {
        "compound": compound,
        "positive": scores["pos"],
        "neutral": scores["neu"],
        "negative": scores["neg"],
        "label": label,
    }


@st.cache_data(show_spinner=False)
def analyze_style(text: str) -> dict[str, object]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    lowered = [word for word in words if word not in STOPWORDS and len(word) > 3]
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


@st.cache_data(show_spinner=False)
def analyze_additional_tools(text: str) -> dict[str, object]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if words else 0.0
    passive_hits = re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", text.lower())
    hedge_hits = [word for word in words if word in HEDGING_WORDS]

    transition_hits: list[str] = []
    lowered_text = text.lower()
    for phrase in sorted(TRANSITION_WORDS):
        if phrase in lowered_text:
            transition_hits.append(phrase)

    paragraph_lengths = [len(tokenize_words(paragraph)) for paragraph in paragraphs]
    longest_paragraph = max(paragraph_lengths) if paragraph_lengths else 0

    return {
        "paragraph_count": len(paragraphs) if paragraphs else (1 if text.strip() else 0),
        "lexical_diversity": lexical_diversity,
        "unique_words": unique_words,
        "passive_count": len(passive_hits),
        "hedge_count": len(hedge_hits),
        "transition_hits": transition_hits[:8],
        "longest_paragraph": longest_paragraph,
        "paragraph_lengths": paragraph_lengths[:6],
    }


@st.cache_data(show_spinner=False)
def build_summary(text: str, metrics: dict[str, float | int | str], sentiment: dict[str, float | str]) -> list[str]:
    words = int(metrics["word_count"])
    sentences = int(metrics["sentence_count"])
    avg_sentence = float(metrics["avg_sentence_length"])
    grade = float(metrics["fk_grade"])
    sentiment_label = str(sentiment["label"])

    summary = [
        f"This draft is about {words} words across {sentences} sentences, which makes it a compact piece for classroom review.",
        f"The readability estimate is around grade {grade:.1f}, with an average sentence length of {avg_sentence:.1f} words.",
        f"The overall tone reads as {sentiment_label.lower()}, based on a local sentiment model rather than an external AI evaluation.",
    ]
    return summary


def executive_presence_score(metrics: dict[str, float | int | str], sentiment: dict[str, float | str], style: dict[str, object]) -> tuple[int, str]:
    readability = float(metrics["reading_ease"])
    avg_sentence = float(metrics["avg_sentence_length"])
    repeated_terms = len(style["repeated_terms"])
    long_sentences = len(style["long_sentences"])
    compound = abs(float(sentiment["compound"]))

    score = 72
    score -= round(normalize_score(abs(avg_sentence - 18), 0, 18) * 18)
    score -= round(normalize_score(repeated_terms, 0, 8) * 10)
    score -= round(normalize_score(long_sentences, 0, 5) * 10)
    score += round(normalize_score(readability, 35, 75) * 8)
    score -= round(normalize_score(compound, 0.55, 1.0) * 8)
    score = max(35, min(95, score))

    if score >= 82:
        band = "Strong"
    elif score >= 68:
        band = "Developing"
    else:
        band = "Needs revision"
    return score, band


def build_ai_prompt(text: str, language_code: str) -> str:
    return f"""You are helping a student improve a piece of writing.

Language of input: {language_code}
Expected response language: English

Please analyze the writing below and return:
1. A 3-sentence executive summary.
2. Top 3 strengths.
3. Top 3 weaknesses.
4. Notes on executive presence, clarity, and tone.
5. Concrete revision suggestions.

Writing:
\"\"\"
{text}
\"\"\""""


def format_reading_time(minutes: float) -> str:
    if minutes < 1:
        seconds = max(10, math.ceil(minutes * 60))
        return f"{seconds} sec"
    return f"{minutes:.1f} min"


@st.cache_data(show_spinner=False)
def load_angle_machine_html() -> str:
    if not ANGLE_MACHINE_PATH.exists():
        return ""
    return ANGLE_MACHINE_PATH.read_text(encoding="utf-8")


def build_angle_output(selections: dict[str, str]) -> dict[str, str]:
    angle_line = (
        f"{selections['Temporalite']} | {selections['Fonction 1']} -> {selections['Fonction 2']} | "
        f"{selections['Perspective']} | {selections['Format/genre']} | {selections['Dominante']}"
    )
    starter = (
        f"Build this as a {selections['Temporalite'].lower()} {selections['Format/genre'].lower()} piece. "
        f"Lead with a {selections['Dominante'].lower()} frame, using a {selections['Perspective'].lower()} perspective. "
        f"The target effect is {selections['Fonction 1'].lower()} through {selections['Fonction 2'].lower()}."
    )
    return {"angle_line": angle_line, "starter": starter}


def render_tool_links() -> None:
    for label, url, purpose in TOOL_LINKS:
        st.markdown(f"- [{label}]({url}): {purpose}")


def render_metric_cards(metrics: dict[str, float | int | str], score: int, band: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Words", f"{int(metrics['word_count']):,}")
    col2.metric("Sentences", int(metrics["sentence_count"]))
    col3.metric("Read Time", format_reading_time(float(metrics["reading_minutes"])))
    col4.metric("Exec Presence", f"{score}/100", band)

    col5, col6, col7 = st.columns(3)
    col5.metric("F-K Grade", f"{float(metrics['fk_grade']):.1f}")
    col6.metric("Reading Ease", f"{float(metrics['reading_ease']):.1f}")
    col7.metric("Avg Sentence", f"{float(metrics['avg_sentence_length']):.1f} words")


def render_pos_highlighter(text: str) -> None:
    st.subheader("Parts of Speech")
    st.write("Select one or more word classes to highlight every matching word in your submitted text.")
    pos_options = (("Verb", "VERB"), ("Noun", "NOUN"), ("Adverb", "ADV"), ("Adjective", "ADJ"))
    toggle_columns = st.columns(4)
    selected_pos = {
        pos
        for column, (label, pos) in zip(toggle_columns, pos_options)
        if column.checkbox(label, value=label == "Verb", key=f"en_pos_{pos}")
    }

    try:
        counts = part_of_speech_counts(text, POS_MODEL)
        highlighted_html, highlighted_counts = highlight_parts_of_speech(text, POS_MODEL, selected_pos)
    except RuntimeError:
        st.error("spaCy is not installed. Run `pip install -r requirements.txt` and restart the app.")
        return
    except OSError:
        st.error("The English language model is not installed. Run `python -m spacy download en_core_web_sm` and restart the app.")
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


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

    st.title(APP_TITLE)
    st.caption("Local-first writing feedback for quick classroom review.")

    st.sidebar.header("Settings")
    english_only = st.sidebar.toggle("English-first guidance", value=True)
    show_examples = st.sidebar.toggle("Show sentence flags", value=True)
    st.sidebar.markdown(
        "This version runs locally. No AI key is required, and the app stays useful even without external services."
    )

    st.sidebar.subheader("Tools Used")
    render_tool_links()

    text = st.text_area("Paste writing to analyze", value=DEFAULT_TEXT, height=320, placeholder="Paste up to a few pages of writing here.")
    analyze = st.button("Analyze Writing", type="primary", use_container_width=True)

    if analyze:
        st.session_state["en_analysis_requested"] = True

    if not st.session_state.get("en_analysis_requested", False):
        st.info("Paste text, then click `Analyze Writing`.")
        return

    if not text.strip():
        st.warning("Please paste some text before running the analysis.")
        return

    with st.spinner("Running local analysis..."):
        language_code = detect_language(text)
        metrics = analyze_basic_metrics(text)
        sentiment = analyze_sentiment(text)
        style = analyze_style(text)
        extra_tools = analyze_additional_tools(text)
        summary = build_summary(text, metrics, sentiment)
        score, band = executive_presence_score(metrics, sentiment, style)
        ai_prompt = build_ai_prompt(text, language_code)

    if language_code != "en":
        st.warning(
            f"Detected language: `{language_code}`. This version is English-first, so readability and sentiment signals may be less reliable."
        )
    elif english_only:
        st.success("Detected language: `en`. English-first scoring is enabled.")

    render_metric_cards(metrics, score, band)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
        ["Basic Stats", "Style", "Sentiment", "Parts of Speech", "AI Critique", "Angle Machine", "Writing Diagnostics", "Journalistic Angle"]
    )

    with tab1:
        st.subheader("Local Summary")
        for sentence in summary:
            st.write(f"- {sentence}")

        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown("**Counts**")
            st.write(f"Characters: {int(metrics['char_count']):,}")
            st.write(f"Syllables: {int(metrics['syllable_count']):,}")
            st.write(f"Estimated reading time: {format_reading_time(float(metrics['reading_minutes']))}")
        with stats_col2:
            st.markdown("**Interpretation**")
            st.write("Flesch-Kincaid grade estimates how difficult the text may feel to a reader.")
            st.write("Reading ease is higher when prose is easier to scan quickly.")
            st.write("Executive presence here is a heuristic score based on readability, repetition, and tonal stability.")

    with tab2:
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

    with tab3:
        st.subheader("Sentiment Snapshot")
        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
        sent_col1.metric("Overall", sentiment["label"])
        sent_col2.metric("Positive", f"{float(sentiment['positive']):.2f}")
        sent_col3.metric("Neutral", f"{float(sentiment['neutral']):.2f}")
        sent_col4.metric("Negative", f"{float(sentiment['negative']):.2f}")

        st.progress((float(sentiment["compound"]) + 1) / 2)
        st.caption(
            f"Compound sentiment score: {float(sentiment['compound']):.2f}. Values near zero usually indicate a more measured tone."
        )

        st.markdown("**Interpretation**")
        st.write(
            "This sentiment model is local and lightweight. It is useful for catching broad tonal drift, not for grading nuance or intent."
        )

    with tab4:
        render_pos_highlighter(text)

    with tab5:
        st.subheader("Use External AI Without Sharing Your Key")
        st.write(
            "This app does not call an external model. If you want richer critique, copy the prompt below into an AI tool of your choice."
        )
        st.info(
            "I have not wired in an automatic AI-authorship detector. The better-looking API options are paid or English-only, and I would not trust them enough to make this part feel authoritative for student evaluation."
        )
        st.code(ai_prompt, language="text")
        st.download_button(
            "Download AI Prompt",
            data=ai_prompt,
            file_name="writer_dashboard_prompt.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.markdown("**Useful links**")
        render_tool_links()

    with tab6:
        st.subheader("La Machine a Angler")
        st.write(
            "This is embedded from the standalone HTML tool so you can use it inside the dashboard without rewriting the interaction logic."
        )
        angle_html = load_angle_machine_html()
        if angle_html:
            components.html(angle_html, height=980, scrolling=False)
        else:
            st.warning("The angle-machine HTML file could not be found.")

        st.markdown("**Machine output**")
        st.write(
            "The embedded HTML widget does not expose its selected state back to Streamlit, so the dashboard includes a native companion output here."
        )

        angle_col1, angle_col2, angle_col3 = st.columns(3)
        with angle_col1:
            temporalite = st.selectbox("Temporalite", ANGLE_WHEELS[0][1], key="angle_temporalite")
            fonction_1 = st.selectbox("Fonction 1", ANGLE_WHEELS[1][1], key="angle_fonction_1")
        with angle_col2:
            fonction_2_options = ANGLE_FUNCTION_MAPPING[fonction_1]
            fonction_2 = st.selectbox("Fonction 2", fonction_2_options, key="angle_fonction_2")
            perspective = st.selectbox("Perspective", ANGLE_WHEELS[3][1], key="angle_perspective")
        with angle_col3:
            format_genre = st.selectbox("Format/genre", ANGLE_WHEELS[4][1], key="angle_format")
            dominante = st.selectbox("Dominante", ANGLE_WHEELS[5][1], key="angle_dominante")

        selections = {
            "Temporalite": temporalite,
            "Fonction 1": fonction_1,
            "Fonction 2": fonction_2,
            "Perspective": perspective,
            "Format/genre": format_genre,
            "Dominante": dominante,
        }
        angle_output = build_angle_output(selections)
        st.code(angle_output["angle_line"], language="text")
        st.caption(angle_output["starter"])

        st.caption(
            "Best fit: treat this as a planning companion for journalism/story framing, separate from the writing-quality analysis above."
        )

    with tab7:
        st.subheader("Writing Diagnostics")
        st.write(
            "These are extra revision checks. They are not separate apps or AI tools, just local signals that help you inspect structure, wording, and tone."
        )
        tool_col1, tool_col2, tool_col3, tool_col4 = st.columns(4)
        tool_col1.metric("Paragraphs", int(extra_tools["paragraph_count"]))
        tool_col2.metric("Unique Words", int(extra_tools["unique_words"]))
        tool_col3.metric("Lexical Diversity", f"{float(extra_tools['lexical_diversity']):.2f}")
        tool_col4.metric("Passive Voice Flags", int(extra_tools["passive_count"]))

        tool_col5, tool_col6, tool_col7 = st.columns(3)
        tool_col5.metric("Hedging Words", int(extra_tools["hedge_count"]))
        tool_col6.metric("Transitions Found", len(extra_tools["transition_hits"]))
        tool_col7.metric("Longest Paragraph", f"{int(extra_tools['longest_paragraph'])} words")

        st.markdown("**What these mean**")
        st.write("`Paragraphs` and `Longest Paragraph` help you see whether the draft is visually broken up or hiding large blocks of text.")
        st.write("`Unique Words` and `Lexical Diversity` give a rough sense of vocabulary variety. Higher is not always better.")
        st.write("`Passive Voice Flags` marks possible passive constructions for review. It is a rough pattern match, not a grammar verdict.")
        st.write("`Hedging Words` counts softeners like `maybe`, `perhaps`, or `likely`, which can make a piece sound cautious or less direct.")
        st.write("`Transitions Found` looks for connectors like `however`, `therefore`, and `for example` to show how explicitly ideas are linked.")

        if extra_tools["transition_hits"]:
            st.markdown("**Transitions detected in this draft**")
            st.write(", ".join(str(item) for item in extra_tools["transition_hits"]))

        if extra_tools["paragraph_lengths"]:
            st.markdown("**Paragraph sizes**")
            st.write(", ".join(f"{count} words" for count in extra_tools["paragraph_lengths"]))

    with tab8:
        st.subheader("Journalistic Angle Analysis")
        st.session_state.setdefault(
            "en_angle_prompt",
            "Analyze the text for a simple journalistic angle. Identify the central news value, the most compelling perspective, who is affected, and one concise angle to pursue. Be specific and grounded only in the supplied text; note important missing information.",
        )
        angle_prompt = st.text_area(
            "Request prompt",
            height=130,
            key="en_angle_prompt",
        )
        if st.button("Generate angle analysis", type="primary", key="en_generate_angle"):
            if not text.strip():
                st.warning("Paste some text before requesting an analysis.")
            else:
                try:
                    with st.spinner("Requesting analysis..."):
                        st.session_state["en_angle_result"] = request_angle_analysis(text, angle_prompt)
                except RuntimeError as exc:
                    st.error(str(exc))
        st.session_state.setdefault("en_angle_result", "")
        st.text_area("Analysis (editable)", height=280, key="en_angle_result")


if __name__ == "__main__":
    main()
