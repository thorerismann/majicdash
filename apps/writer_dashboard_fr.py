from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
import textstat
from langdetect import DetectorFactory, LangDetectException, detect
from vaderSentiment_fr.vaderSentiment import SentimentIntensityAnalyzer
from modules.pos_highlighting import highlight_parts_of_speech, part_of_speech_counts
from modules.inference import request_angle_analysis

# Set textstat to French
textstat.set_lang("fr")

DetectorFactory.seed = 0

APP_TITLE = "Tableau de Bord de l'Écrivain"
ANGLE_MACHINE_PATH = Path(__file__).resolve().parents[1] / "assets" / "code_machine_angle.html"
POS_MODEL = "fr_core_news_sm"
DEFAULT_TEXT = """Collez ici un paragraphe d'élève, un mémo ou un brouillon pour analyser la clarté, le ton et la fluidité des phrases.

Cette première version est axée sur le français et fonctionne localement. Elle ne nécessite aucune clé d'IA."""
STOPWORDS = {
    "le", "la", "les", "un", "une", "des", "du", "de", "à", "et", "ou", "mais", 
    "donc", "car", "ni", "en", "dans", "par", "pour", "sur", "avec", "qui", 
    "que", "quoi", "dont", "où", "je", "tu", "il", "elle", "on", "nous", 
    "vous", "ils", "elles", "me", "te", "se", "y", "ce", "cet", "cette", "ces", 
    "mon", "ton", "son", "notre", "votre", "leur", "ne", "pas", "plus", 
    "est", "sont", "a", "ont", "être", "avoir", "au", "aux", 
    "c", "d", "j", "l", "m", "n", "s", "t", "qu"
}

def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]

def tokenize_words(text: str) -> list[str]:
    # Support French accents
    return re.findall(r"[A-Za-zÀ-ÿ][A-Za-zÀ-ÿ'-]*", text.lower())

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
        return "inconnu"

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
        label = "Positif / encourageant"
    elif compound <= -0.35:
        label = "Négatif / critique"
    else:
        label = "Neutre / mesuré"

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
        strengths.append("La longueur des phrases est généralement dans une fourchette lisible.")
    else:
        improvements.append("La longueur des phrases est inégale ; visez un rythme plus régulier d'un paragraphe à l'autre.")

    if len(long_sentences) == 0:
        strengths.append("Aucune phrase longue et surchargée n'a été détectée.")
    else:
        improvements.append("Quelques phrases sont assez longues pour risquer de perdre le lecteur.")

    if repeated_terms:
        repeated_labels = ", ".join(f"{word} ({count})" for word, count in repeated_terms[:5])
        improvements.append(f"Certains mots sont souvent répétés : {repeated_labels}.")
    else:
        strengths.append("La répétition des mots est limitée, ce qui aide le brouillon à paraître moins redondant.")

    if exclamation_count == 0 and question_count <= 2:
        strengths.append("L'utilisation de la ponctuation semble contrôlée plutôt qu'excessivement dramatique.")
    else:
        improvements.append("La ponctuation ajoute peut-être une intensité excessive ; revoyez les points d'exclamation et les questions rhétoriques.")

    if short_sentences:
        improvements.append("Des phrases très courtes apparaissent dans le texte ; assurez-vous qu'elles sont intentionnelles pour insister.")

    return {
        "strengths": strengths[:3],
        "improvements": improvements[:4],
        "repeated_terms": repeated_terms[:8],
        "long_sentences": long_sentences[:5],
        "short_sentences": short_sentences[:5],
    }

@st.cache_data(show_spinner=False)
def build_summary(text: str, metrics: dict[str, float | int | str], sentiment: dict[str, float | str]) -> list[str]:
    words = int(metrics["word_count"])
    sentences = int(metrics["sentence_count"])
    avg_sentence = float(metrics["avg_sentence_length"])
    grade = float(metrics["fk_grade"])
    sentiment_label = str(sentiment["label"])

    summary = [
        f"Ce brouillon compte environ {words} mots répartis en {sentences} phrases, ce qui en fait un texte compact pour une révision en classe.",
        f"L'estimation de lisibilité (Flesch-Kincaid) est d'environ {grade:.1f}, avec une longueur moyenne de phrase de {avg_sentence:.1f} mots.",
        f"Le ton général est perçu comme {sentiment_label.lower()}, d'après un modèle de sentiment local.",
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
        band = "Fort"
    elif score >= 68:
        band = "En développement"
    else:
        band = "Nécessite une révision"
    return score, band

def build_ai_prompt(text: str, language_code: str) -> str:
    return f"""Vous aidez un étudiant à améliorer un texte.

Langue de l'entrée : {language_code}
Langue de réponse attendue : Français

Veuillez analyser le texte ci-dessous et renvoyer :
1. Un résumé exécutif de 3 phrases.
2. Les 3 principaux points forts.
3. Les 3 principales faiblesses.
4. Des notes sur la clarté, la présence exécutive et le ton.
5. Des suggestions concrètes de révision.

Texte :
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

def render_metric_cards(metrics: dict[str, float | int | str], score: int, band: str) -> None:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Mots", f"{int(metrics['word_count']):,}")
    col2.metric("Phrases", int(metrics["sentence_count"]))
    col3.metric("Temps de lecture", format_reading_time(float(metrics["reading_minutes"])))
    col4.metric("Présence Exécutive", f"{score}/100", band)

    col5, col6, col7 = st.columns(3)
    col5.metric("Niveau F-K", f"{float(metrics['fk_grade']):.1f}")
    col6.metric("Facilité de lecture", f"{float(metrics['reading_ease']):.1f}")
    col7.metric("Phrase Moyenne", f"{float(metrics['avg_sentence_length']):.1f} mots")


def render_pos_highlighter(text: str) -> None:
    st.subheader("Classes grammaticales")
    st.write("Sélectionnez une ou plusieurs classes de mots pour surligner toutes leurs occurrences dans le texte soumis.")
    pos_options = (("Verbes", "VERB"), ("Noms", "NOUN"), ("Adverbes", "ADV"), ("Adjectifs", "ADJ"))
    toggle_columns = st.columns(4)
    selected_pos = {
        pos
        for column, (label, pos) in zip(toggle_columns, pos_options)
        if column.checkbox(label, value=label == "Verbes", key=f"fr_pos_{pos}")
    }

    try:
        counts = part_of_speech_counts(text, POS_MODEL)
        highlighted_html, highlighted_counts = highlight_parts_of_speech(text, POS_MODEL, selected_pos)
    except RuntimeError:
        st.error("spaCy n'est pas installé. Exécutez `pip install -r requirements.txt`, puis redémarrez l'application.")
        return
    except OSError:
        st.error("Le modèle français n'est pas installé. Exécutez `python -m spacy download fr_core_news_sm`, puis redémarrez l'application.")
        return

    highlighted_total = sum(highlighted_counts.values())
    selected_labels = [label.lower() for label, pos in pos_options if pos in selected_pos]
    selection_summary = ", ".join(selected_labels) if selected_labels else "aucune classe de mots"
    st.caption(
        f"{highlighted_total} mots surlignés ({selection_summary}). "
        f"Verbes : {counts['VERB']} · Noms : {counts['NOUN']} · Adverbes : {counts['ADV']} · Adjectifs : {counts['ADJ']}"
    )
    st.markdown(
        f'<div style="line-height: 1.8; padding: 1rem; border: 1px solid #d1d5db; border-radius: 0.5rem; white-space: pre-wrap;">{highlighted_html}</div>',
        unsafe_allow_html=True,
    )

def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon="📝", layout="wide")

    st.title(APP_TITLE)
    st.caption("Retours d'écriture locaux pour une révision rapide en classe.")

    st.sidebar.header("Paramètres")
    french_only = st.sidebar.toggle("Conseils axés sur le français", value=True)
    show_examples = st.sidebar.toggle("Afficher les phrases signalées", value=True)
    st.sidebar.markdown(
        "Cette version fonctionne localement. Aucune clé d'IA n'est requise, et l'application reste utile même sans services externes."
    )

    st.sidebar.subheader("Liens IA Optionnels")
    st.sidebar.markdown(
        "- [ChatGPT](https://chatgpt.com)\n- [Claude](https://claude.ai)\n- [Gemini](https://gemini.google.com)"
    )

    text = st.text_area("Collez le texte à analyser", value=DEFAULT_TEXT, height=320, placeholder="Collez jusqu'à quelques pages de texte ici.")
    analyze = st.button("Analyser le Texte", type="primary", use_container_width=True)

    if analyze:
        st.session_state["fr_analysis_requested"] = True

    if not st.session_state.get("fr_analysis_requested", False):
        st.info("Collez du texte, puis cliquez sur `Analyser le Texte`.")
        return

    if not text.strip():
        st.warning("Veuillez coller du texte avant de lancer l'analyse.")
        return

    with st.spinner("Exécution de l'analyse locale..."):
        language_code = detect_language(text)
        metrics = analyze_basic_metrics(text)
        sentiment = analyze_sentiment(text)
        style = analyze_style(text)
        summary = build_summary(text, metrics, sentiment)
        score, band = executive_presence_score(metrics, sentiment, style)
        ai_prompt = build_ai_prompt(text, language_code)

    if language_code != "fr":
        st.warning(
            f"Langue détectée : `{language_code}`. Cette version est optimisée pour le français, les indicateurs de lisibilité et de sentiment peuvent donc être moins fiables."
        )
    elif french_only:
        st.success("Langue détectée : `fr`. L'évaluation optimisée pour le français est activée.")

    render_metric_cards(metrics, score, band)

    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(
        ["Statistiques de base", "Style", "Sentiment", "Classes grammaticales", "Critique IA", "La Machine à Angler", "Angle journalistique"]
    )

    with tab1:
        st.subheader("Résumé Local")
        for sentence in summary:
            st.write(f"- {sentence}")

        stats_col1, stats_col2 = st.columns(2)
        with stats_col1:
            st.markdown("**Comptes**")
            st.write(f"Caractères : {int(metrics['char_count']):,}")
            st.write(f"Syllabes : {int(metrics['syllable_count']):,}")
            st.write(f"Temps de lecture estimé : {format_reading_time(float(metrics['reading_minutes']))}")
        with stats_col2:
            st.markdown("**Interprétation**")
            st.write("Le niveau Flesch-Kincaid estime la difficulté que peut ressentir un lecteur face au texte.")
            st.write("La facilité de lecture est plus élevée lorsque la prose est facile à parcourir rapidement.")
            st.write("La présence exécutive est ici un score heuristique basé sur la lisibilité, la répétition et la stabilité du ton.")

    with tab2:
        st.subheader("Points Forts")
        if style["strengths"]:
            for item in style["strengths"]:
                st.write(f"- {item}")
        else:
            st.write("- Aucun point fort exceptionnel n'a été identifié par les règles locales.")

        st.subheader("Priorités de Révision")
        if style["improvements"]:
            for item in style["improvements"]:
                st.write(f"- {item}")
        else:
            st.write("- Aucun indicateur de révision majeur n'a été soulevé localement.")

        if style["repeated_terms"]:
            st.subheader("Termes Répétés")
            repeated_text = ", ".join(f"{word} ({count})" for word, count in style["repeated_terms"])
            st.write(repeated_text)

        if show_examples:
            if style["long_sentences"]:
                st.subheader("Phrases Longues à Revoir")
                for sentence in style["long_sentences"]:
                    st.code(sentence, language=None)
            if style["short_sentences"]:
                st.subheader("Phrases Très Courtes")
                for sentence in style["short_sentences"]:
                    st.code(sentence, language=None)

    with tab3:
        st.subheader("Aperçu des Sentiments")
        sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)
        sent_col1.metric("Global", sentiment["label"])
        sent_col2.metric("Positif", f"{float(sentiment['positive']):.2f}")
        sent_col3.metric("Neutre", f"{float(sentiment['neutral']):.2f}")
        sent_col4.metric("Négatif", f"{float(sentiment['negative']):.2f}")

        st.progress((float(sentiment["compound"]) + 1) / 2)
        st.caption(
            f"Score de sentiment composé : {float(sentiment['compound']):.2f}. Les valeurs proches de zéro indiquent généralement un ton plus mesuré."
        )

        st.markdown("**Interprétation**")
        st.write(
            "Ce modèle de sentiment est local et léger. Il est utile pour repérer de larges dérives de ton, et non pour évaluer les nuances ou l'intention."
        )

    with tab4:
        render_pos_highlighter(text)

    with tab5:
        st.subheader("Utiliser une IA externe sans partager votre clé")
        st.write(
            "Pour une critique plus riche, copiez le prompt ci-dessous dans un outil d'IA de votre choix. L'onglet « Angle journalistique » peut aussi envoyer le texte à l'API CSCS configurée."
        )
        st.info(
            "Je n'ai pas intégré de détecteur automatique de paternité par IA. Les meilleures options d'API sont payantes ou limitées à l'anglais, et je ne leur ferais pas assez confiance pour rendre cette partie décisive dans l'évaluation des étudiants."
        )
        st.code(ai_prompt, language="text")
        st.download_button(
            "Télécharger le Prompt IA",
            data=ai_prompt,
            file_name="writer_dashboard_prompt_fr.txt",
            mime="text/plain",
            use_container_width=True,
        )

    with tab6:
        st.subheader("La Machine à Angler")
        st.write(
            "Ceci est intégré depuis l'outil HTML autonome, afin que vous puissiez l'utiliser à l'intérieur du tableau de bord sans réécrire la logique d'interaction."
        )
        angle_html = load_angle_machine_html()
        if angle_html:
            components.html(angle_html, height=980, scrolling=False)
        else:
            st.warning("Le fichier HTML 'Machine à Angler' n'a pas pu être trouvé.")

        st.caption(
            "Conseil : considérez ceci comme un compagnon de planification pour la structuration journalistique/narrative, distinct de l'analyse de qualité d'écriture ci-dessus."
        )

    with tab7:
        st.subheader("Analyse d'angle journalistique")
        st.session_state.setdefault(
            "fr_angle_prompt",
            "Analysez le texte pour en dégager un angle journalistique simple. Identifiez la valeur d'actualité centrale, la perspective la plus intéressante, les personnes concernées et un angle concis à explorer. Soyez précis et appuyez-vous uniquement sur le texte fourni ; signalez les informations importantes qui manquent.",
        )
        angle_prompt = st.text_area(
            "Consigne de la demande",
            height=130,
            key="fr_angle_prompt",
        )
        if st.button("Générer l'analyse d'angle", type="primary", key="fr_generate_angle"):
            if not text.strip():
                st.warning("Collez du texte avant de demander une analyse.")
            else:
                try:
                    with st.spinner("Analyse en cours..."):
                        st.session_state["fr_angle_result"] = request_angle_analysis(text, angle_prompt)
                except RuntimeError as exc:
                    st.error(str(exc))
        st.session_state.setdefault("fr_angle_result", "")
        st.text_area("Analyse (modifiable)", height=280, key="fr_angle_result")

if __name__ == "__main__":
    main()
