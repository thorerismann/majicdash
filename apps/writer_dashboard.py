from __future__ import annotations

import math
import re
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from textstat.textstat import textstatistics
from langdetect import DetectorFactory, LangDetectException, detect
from modules.pos_highlighting import render_pos_highlighter
from modules.inference import request_angle_analysis
from modules.sentiment_analysis import analyze_sentiment, render_sentiment
from modules.style_analysis import analyze_style, render_style
from modules.text_utils import split_sentences, tokenize_words


DetectorFactory.seed = 0

APP_TITLE = "Writer Dashboard"
ANGLE_MACHINE_PATH = Path(__file__).resolve().parents[1] / "assets" / "code_machine_angle.html"
LANGUAGES = {"en": "English", "fr": "French"}
DEFAULT_TEXT = """Paste a student paragraph, memo, or draft here to analyze clarity, tone, and sentence flow.

English and French texts are supported. Local writing analysis does not require an AI key."""
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
FRENCH_TRANSITIONS = {
    "cependant", "pourtant", "donc", "ainsi", "en outre", "de plus", "néanmoins",
    "parce que", "d'abord", "ensuite", "enfin", "par exemple", "en revanche", "par conséquent",
}
FRENCH_HEDGES = {
    "peut-être", "probablement", "possiblement", "apparemment", "semble", "semblent",
    "plutôt", "éventuellement", "quelque peu",
}


def select_analysis_language(detected: str, choice: str) -> str | None:
    if choice == "Auto-detect":
        return detected if detected in LANGUAGES else None
    return {"English": "en", "French": "fr"}[choice]


def normalize_score(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return max(0.0, min(1.0, (value - low) / (high - low)))


@st.cache_data(show_spinner=False)
def detect_language(text: str) -> str:
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


@st.cache_data(show_spinner=False)
def analyze_basic_metrics(text: str, language_code: str = "en") -> dict[str, float | int | str | None]:
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
def analyze_additional_tools(text: str, language_code: str = "en") -> dict[str, object]:
    words = tokenize_words(text)
    sentences = split_sentences(text)
    paragraphs = [paragraph.strip() for paragraph in re.split(r"\n\s*\n", text) if paragraph.strip()]
    unique_words = len(set(words))
    lexical_diversity = unique_words / len(words) if words else 0.0
    passive_hits = re.findall(r"\b(?:is|are|was|were|be|been|being)\s+\w+(?:ed|en)\b", text.lower())
    hedges = FRENCH_HEDGES if language_code == "fr" else HEDGING_WORDS
    transitions = FRENCH_TRANSITIONS if language_code == "fr" else TRANSITION_WORDS
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


@st.cache_data(show_spinner=False)
def build_summary(text: str, metrics: dict[str, float | int | str | None], sentiment: dict[str, float | str]) -> list[str]:
    words = int(metrics["word_count"])
    sentences = int(metrics["sentence_count"])
    avg_sentence = float(metrics["avg_sentence_length"])
    grade = metrics["fk_grade"]
    sentiment_label = str(sentiment["label"])

    summary = [
        f"This draft is about {words} words across {sentences} sentences, which makes it a compact piece for classroom review.",
        (f"The readability estimate is around grade {grade:.1f}, with an average sentence length of {avg_sentence:.1f} words."
         if grade is not None else
         f"French reading ease is {float(metrics['reading_ease']):.1f}, with an average sentence length of {avg_sentence:.1f} words."),
        f"VADER classifies the document’s emotional wording as {sentiment_label.lower()}; this is not a judgment of writing quality or author intent.",
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
Expected response language: {LANGUAGES.get(language_code, 'English')}

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
    col5.metric("F-K Grade", f"{float(metrics['fk_grade']):.1f}" if metrics['fk_grade'] is not None else "N/A")
    col6.metric("Reading Ease", f"{float(metrics['reading_ease']):.1f}")
    col7.metric("Avg Sentence", f"{float(metrics['avg_sentence_length']):.1f} words")


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

    st.title(APP_TITLE)
    st.caption("Local-first writing feedback for quick classroom review.")

    st.sidebar.header("Settings")
    language_choice = st.sidebar.selectbox("Text language", ["Auto-detect", "English", "French"])
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

    detected_language = detect_language(text)
    language_code = select_analysis_language(detected_language, language_choice)
    if language_code is None:
        st.warning(f"Detected language: `{detected_language}`. Choose English or French under Text language if detection is incorrect.")
        return

    with st.spinner("Running local analysis..."):
        metrics = analyze_basic_metrics(text, language_code)
        try:
            sentiment = analyze_sentiment(text, language_code)
        except RuntimeError as exc:
            st.error(str(exc))
            return
        style = analyze_style(text, language_code)
        extra_tools = analyze_additional_tools(text, language_code)
        summary = build_summary(text, metrics, sentiment)
        score, band = executive_presence_score(metrics, sentiment, style)
        ai_prompt = build_ai_prompt(text, language_code)

    st.success(f"Detected language: `{detected_language}`. Using {LANGUAGES[language_code]} analysis.")

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
            st.write("Flesch-Kincaid grade is shown for English only. French text uses French reading ease; its grade is marked N/A.")
            st.write("Reading ease is higher when prose is easier to scan quickly.")
            st.write("Executive presence here is a heuristic score based on readability, repetition, and sentiment intensity; it is not a validated assessment.")

    with tab2:
        render_style(style, show_examples)

    with tab3:
        render_sentiment(text, language_code)

    with tab4:
        render_pos_highlighter(text, language_code)

    with tab5:
        st.subheader("Use External AI Without Sharing Your Key")
        st.write(
            "For richer critique, copy the prompt below into an AI tool of your choice. The Journalistic Angle tab can also send your text to the configured CSCS API."
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
        tool_col4.metric("Passive Voice Flags", int(extra_tools["passive_count"]) if extra_tools["passive_count"] is not None else "N/A")

        tool_col5, tool_col6, tool_col7 = st.columns(3)
        tool_col5.metric("Hedging Words", int(extra_tools["hedge_count"]))
        tool_col6.metric("Transitions Found", len(extra_tools["transition_hits"]))
        tool_col7.metric("Longest Paragraph", f"{int(extra_tools['longest_paragraph'])} words")

        st.markdown("**What these mean**")
        st.write("`Paragraphs` and `Longest Paragraph` help you see whether the draft is visually broken up or hiding large blocks of text.")
        st.write("`Unique Words` and `Lexical Diversity` give a rough sense of vocabulary variety. Higher is not always better.")
        st.write("`Passive Voice Flags` uses a rough English pattern match. It is unavailable for French and shown as N/A.")
        st.write("`Hedging Words` counts softeners in the selected language, such as `perhaps` or `peut-être`.")
        st.write("`Transitions Found` looks for connectors in the selected language, such as `however` or `cependant`.")

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
