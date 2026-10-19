"""Interactive article-level distributions against a compact, aligned corpus."""

from math import ceil, sqrt

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from .corpus_reference import (
    METRIC_KEYS, REFERENCE_PATH, comparison_values, STYLE_METRIC_KEYS, SENTIMENT_METRIC_KEYS,
)
from .pos_highlighting import BREAKDOWN_LABELS


COPY = {
    "en": {
        "title": "Your text in context", "tab": "Compare with JAM",
        "intro": "Explore how your writing compares with articles from the JAM corpus.",
        "body": "Compare body text only: omit the title, standfirst, captions and headings. Separate paragraphs with a blank line.",
        "category": "Explore", "paragraphs": "Paragraphs", "sentences": "Sentences",
        "syllables": "Syllables", "characters": "Characters", "readability": "Readability",
        "measure": "Measure", "filters": "Reference articles", "medium": "Article type",
        "all": "All types", "years": "Publication years", "your": "Your text",
        "median": "Corpus median", "rank": "Percentile", "sample": "Reference articles",
        "count": "Articles", "share": "Share", "from": "From", "to": "To",
        "coverage": "{n:,} {language} articles · {medium} · {start}–{end}. Each article counts once.",
        "language": "English", "small": "Only {n} reference articles match. This sample is too small for a reliable benchmark; fewer than five articles also hides the percentile.",
        "empty": "No reference articles match these filters. Try all article types or a wider year range.",
        "input": "Enter some body text to see where it sits in the corpus.",
        "legend": "Orange: your text · dashed line: corpus median · pale band: middle 50% of articles. Hover for details; scroll to zoom and drag to pan.",
        "meaning": "Percentile uses the share below your value plus half of ties. Higher is not necessarily better; this describes the selected corpus, not a writing-quality score.",
        "missing": "The reference data is unavailable or needs refreshing. Other analysis tools remain available.",
        "method": "About this comparison", "method_text": "The reference was recalculated with the same analysis as this dashboard. Only articles passing the corpus's stricter structural checks are included; duplicates and uncertain extractions are excluded. Languages are compared separately. Coverage varies by year and article type.",
        "download": "Download this comparison", "metric": "Metric",
        "paragraph_count": "Number of paragraphs", "avg_paragraph_words": "Words per paragraph",
        "sentence_count": "Number of sentences", "avg_sentence_length": "Words per sentence",
        "syllable_count": "Total syllables", "syllables_per_word": "Syllables per word",
        "char_count": "Total characters", "reading_ease": "Reading ease", "fk_grade": "Flesch–Kincaid grade",
        "paragraph_help": "Mean paragraph length: total words divided by all paragraphs. Each reference article contributes one average.",
        "sentence_help": "Sentence boundaries and word tokens come from the same spaCy language model used in the dashboard.",
        "syllable_help": "Syllables are estimated with textstat; the per-word measure divides by the dashboard's word count.",
        "character_help": "Characters include spaces, punctuation and line breaks. Total length depends on the amount of text pasted.",
        "readability_help": "Language-specific readability estimate. Higher reading ease usually means easier text; a higher English grade means a higher estimated school level. Scores can fall outside 0–100.",
    },
    "fr": {
        "title": "Votre texte en contexte", "tab": "Comparer avec JAM",
        "intro": "Découvrez comment votre texte se situe parmi les articles du corpus JAM.",
        "body": "Comparez uniquement le corps du texte : sans titre, chapeau, légendes ni intertitres. Séparez les paragraphes par une ligne vide.",
        "category": "Explorer", "paragraphs": "Paragraphes", "sentences": "Phrases",
        "syllables": "Syllabes", "characters": "Caractères", "readability": "Lisibilité",
        "measure": "Mesure", "filters": "Articles de référence", "medium": "Type d’article",
        "all": "Tous les types", "years": "Années de publication", "your": "Votre texte",
        "median": "Médiane du corpus", "rank": "Percentile", "sample": "Articles de référence",
        "count": "Articles", "share": "Part", "from": "De", "to": "À",
        "coverage": "{n:,} articles en {language} · {medium} · {start}–{end}. Chaque article compte une fois.",
        "language": "français", "small": "Seuls {n} articles correspondent. Cet échantillon est trop petit pour une référence fiable ; le percentile est masqué en dessous de cinq articles.",
        "empty": "Aucun article ne correspond à ces filtres. Essayez tous les types ou une période plus large.",
        "input": "Saisissez le corps d’un texte pour le situer dans le corpus.",
        "legend": "Orange : votre texte · pointillés : médiane du corpus · bande claire : moitié centrale des articles. Survolez pour les détails ; faites défiler pour zoomer et glissez pour déplacer.",
        "meaning": "Le percentile compte les valeurs inférieures et la moitié des ex æquo. Une valeur élevée n’est pas forcément meilleure : il s’agit d’une comparaison, pas d’une note de qualité.",
        "missing": "Les données de référence sont indisponibles ou doivent être actualisées. Les autres analyses restent disponibles.",
        "method": "À propos de cette comparaison", "method_text": "La référence a été recalculée avec la même analyse que ce tableau de bord. Seuls les articles satisfaisant les contrôles structurels stricts sont inclus ; les doublons et extractions incertaines sont exclus. Les langues sont comparées séparément. La couverture varie selon l’année et le type d’article.",
        "download": "Télécharger cette comparaison", "metric": "Mesure",
        "paragraph_count": "Nombre de paragraphes", "avg_paragraph_words": "Mots par paragraphe",
        "sentence_count": "Nombre de phrases", "avg_sentence_length": "Mots par phrase",
        "syllable_count": "Total des syllabes", "syllables_per_word": "Syllabes par mot",
        "char_count": "Total des caractères", "reading_ease": "Facilité de lecture", "fk_grade": "Niveau Flesch–Kincaid",
        "paragraph_help": "Longueur moyenne : nombre total de mots divisé par tous les paragraphes. Chaque article de référence fournit une moyenne.",
        "sentence_help": "Les limites de phrases et les mots proviennent du même modèle linguistique spaCy que les autres analyses.",
        "syllable_help": "Les syllabes sont estimées avec textstat ; la mesure par mot utilise le nombre de mots du tableau de bord.",
        "character_help": "Les caractères incluent les espaces, la ponctuation et les retours à la ligne. Le total dépend de la quantité de texte saisie.",
        "readability_help": "Estimation propre à la langue. Une facilité de lecture élevée indique généralement un texte plus accessible. Le score peut sortir de l’intervalle 0–100.",
    },
}

GROUPS = {
    "paragraphs": ("avg_paragraph_words", "paragraph_count"),
    "sentences": ("avg_sentence_length", "sentence_count"),
    "syllables": ("syllables_per_word", "syllable_count"),
    "characters": ("char_count",),
    "readability": ("reading_ease", "fk_grade"),
}
HELP_KEYS = dict(zip(GROUPS, ("paragraph_help", "sentence_help", "syllable_help", "character_help", "readability_help")))
COUNT_METRICS = {"word_count", "paragraph_count", "sentence_count", "syllable_count", "char_count"}

GROUPS.update({
    "vocabulary": ("word_count", "unique_words", "lexical_diversity", "lemma_diversity", "stopword_ratio", "avg_word_length"),
    "style": STYLE_METRIC_KEYS,
    "sentiment": SENTIMENT_METRIC_KEYS,
    "syntax": ("token_count", "entity_count", "entities_per_100_words", "avg_dependency_distance"),
    **{kind: tuple(f"{kind}_{label}_{suffix}" for label in labels for suffix in ("count", "ratio"))
       for kind, labels in BREAKDOWN_LABELS.items()},
})
COUNT_METRICS.update(key for key in METRIC_KEYS if key.endswith("_count"))
COUNT_METRICS.update(("unique_words", "longest_paragraph"))
for language, copy in COPY.items():
    french = language == "fr"
    copy.update({
        "vocabulary": "Vocabulaire" if french else "Vocabulary",
        "style": "Style", "sentiment": "Sentiment VADER", "syntax": "Structure spaCy",
        "pos": "Classes grammaticales" if french else "Parts of speech",
        "dep": "Dépendances" if french else "Dependencies",
        "entity": "Entités nommées" if french else "Named entities", "morph": "Morphologie" if french else "Morphology",
        "dimensions_help": ("Parts grammaticales, syntaxiques et morphologiques : fraction des tokens hors espaces. Entités : fraction des mentions. Les étiquettes absentes du modèle de cette langue sont exclues."
                            if french else "POS, dependency and morphology shares are fractions of non-space tokens. Entity shares are fractions of mentions. Labels unsupported by this language's model are excluded."),
        "sentiment_help": ("VADER : score global de −1 à +1 ; parts de 0 à 1. Les statistiques par phrase utilisent le découpage du volet Sentiment. Les scores français et anglais ne sont pas directement comparables."
                           if french else "VADER: compound −1 to +1; shares 0 to 1. Sentence statistics use the Sentiment tab's segmentation. English and French scores are not directly comparable."),
        "style_help": ("Diversité lexicale : formes uniques / mots ; diversité des lemmes : lemmes uniques / mots. Phrases longues : > 25 mots ; courtes : < 8. Les diagnostics sont des heuristiques."
                       if french else "Lexical diversity: unique surface forms / words; lemma diversity: unique lemmas / words. Long sentences: > 25 words; short: < 8. Style diagnostics are heuristics."),
        "scatter": "Croiser deux dimensions" if french else "Compare any two dimensions",
        "x_axis": "Axe X" if french else "X axis", "y_axis": "Axe Y" if french else "Y axis",
        "all_metrics": "Toutes les dimensions (CSV)" if french else "Download all dimensions (CSV)",
    })
    for key in METRIC_KEYS:
        copy.setdefault(key, key.replace("_", " ").capitalize())
    for kind, labels in BREAKDOWN_LABELS.items():
        for label in labels:
            copy[f"{kind}_{label}_count"] = f"{label} · {'nombre' if french else 'count'}"
            copy[f"{kind}_{label}_ratio"] = f"{label} · {'part' if french else 'share'}"
HELP_KEYS.update({group: "dimensions_help" for group in ("syntax", *BREAKDOWN_LABELS)})
HELP_KEYS.update(vocabulary="style_help", style="style_help", sentiment="sentiment_help")


@st.cache_data(show_spinner=False, max_entries=1)
def _load_reference(csv_path: str, modified_ns: int) -> pd.DataFrame:
    """Read the single reference CSV; rebuilding it invalidates this cache."""
    frame = pd.read_csv(csv_path, dtype={"article_id": str})
    required = {"article_id", "language", "medium", "year", *METRIC_KEYS}
    if not required.issubset(frame.columns) or frame.empty:
        raise ValueError("Invalid reference schema")
    if frame.article_id.isna().any() or frame.article_id.duplicated().any():
        raise ValueError("Reference article IDs must be unique and nonempty")
    frame["year"] = pd.to_numeric(frame["year"], errors="raise")
    if frame.year.isna().any() or not frame.language.isin(["fr", "en"]).all():
        raise ValueError("Invalid reference year or language")
    for key in METRIC_KEYS:
        frame[key] = pd.to_numeric(frame[key], errors="raise")
    return frame


def load_reference() -> pd.DataFrame:
    return _load_reference(str(REFERENCE_PATH), REFERENCE_PATH.stat().st_mtime_ns)


def select_cohort(frame: pd.DataFrame, language: str, medium: str | None = None, years=None) -> pd.DataFrame:
    selected = frame.loc[frame.language.eq(language)]
    if medium is not None:
        selected = selected.loc[selected.medium.eq(medium)]
    if years is not None:
        selected = selected.loc[selected.year.between(*years)]
    return selected


def distribution_summary(values, current: float) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values):
        return {"n": 0, "median": None, "percentile": None}
    ties = np.isclose(values, current, rtol=1e-10, atol=1e-9)
    return {
        "n": len(values), "median": float(np.median(values)),
        "percentile": float(100 * (((values < current) & ~ties).sum() + .5 * ties.sum()) / len(values)),
    }


def build_chart(values, current: float, metric: str, language: str) -> alt.LayerChart:
    """Histogram and reference markers share one axis, including out-of-range input."""
    copy = COPY[language]
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if not len(values) or not np.isfinite(current):
        raise ValueError("A chart needs finite reference values and a current value")
    bins = min(30, max(6, int(sqrt(len(values)))))
    if metric in COUNT_METRICS:
        step = max(1, ceil((values.max() - values.min() + 1) / bins))
        edges = np.arange(values.min() - .5, values.max() + step + .5, step)
        counts, edges = np.histogram(values, bins=edges)
    else:
        counts, edges = np.histogram(values, bins=bins)
    histogram = pd.DataFrame({
        "low": edges[:-1], "high": edges[1:], "count": counts,
        "share": counts / len(values),
    })
    low, high = min(edges[0], current), max(edges[-1], current)
    padding = max((high - low) * .04, .1)
    scale = alt.Scale(domain=[low - padding, high + padding], zero=False)
    x = alt.X("low:Q", title=copy[metric], scale=scale)
    bars = alt.Chart(histogram).mark_bar(color="#319b98", opacity=.85, cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
        x=x, x2="high:Q", y=alt.Y("count:Q", title=copy["count"], axis=alt.Axis(tickMinStep=1)),
        tooltip=[alt.Tooltip("low:Q", title=copy["from"], format=".2f"),
                 alt.Tooltip("high:Q", title=copy["to"], format=".2f"),
                 alt.Tooltip("count:Q", title=copy["count"]),
                 alt.Tooltip("share:Q", title=copy["share"], format=".1%")],
    )
    q25, median, q75 = np.quantile(values, [.25, .5, .75])
    band = alt.Chart(pd.DataFrame({"low": [q25], "high": [q75]})).mark_rect(color="#319b98", opacity=.12).encode(x=x, x2="high:Q")
    reference = alt.Chart(pd.DataFrame({"value": [median]})).mark_rule(color="#64748b", strokeDash=[5, 5], size=2).encode(
        x=alt.X("value:Q", scale=scale), tooltip=[alt.Tooltip("value:Q", title=copy["median"], format=".2f")],
    )
    point = pd.DataFrame({"value": [current], "label": [copy["your"]]})
    marker = alt.Chart(point).mark_rule(color="#f97316", size=3).encode(
        x=alt.X("value:Q", scale=scale), tooltip=[alt.Tooltip("value:Q", title=copy["your"], format=".2f")],
    )
    align_right = current > (low + high) / 2
    label = alt.Chart(point).mark_text(
        color="#f97316", fontWeight="bold", fontSize=13,
        align="right" if align_right else "left", dx=-8 if align_right else 8,
    ).encode(x=alt.X("value:Q", scale=scale), y=alt.value(12), text="label:N")
    return alt.layer(band, bars, reference, marker, label).properties(height=330).interactive(bind_y=False).configure_view(stroke=None)


def render_corpus_comparison(metrics: dict, extra: dict, language: str, sentiment: dict | None = None) -> None:
    copy = COPY[language]
    st.subheader(copy["title"])
    st.write(copy["intro"])
    st.caption(copy["body"])
    if not metrics["word_count"]:
        st.info(copy["input"])
        return
    try:
        frame = load_reference()
    except (OSError, ValueError, KeyError):
        st.warning(copy["missing"])
        return
    current_values = comparison_values(metrics, extra, sentiment)
    available = [key for key in METRIC_KEYS if current_values.get(key) is not None and np.isfinite(current_values[key])]
    groups = [group for group, keys in GROUPS.items() if any(key in available for key in keys)]
    group = st.pills(copy["category"], groups, format_func=copy.get,
                     default="sentences", key=f"corpus_group_{language}") or "sentences"
    options = [key for key in GROUPS[group] if key in available]
    if len(options) > 12:
        metric = st.selectbox(copy["measure"], options, format_func=copy.get, key=f"corpus_metric_{language}_{group}")
    else:
        metric = st.radio(copy["measure"], options, format_func=copy.get, horizontal=True,
                          key=f"corpus_metric_{language}_{group}")
    language_frame = select_cohort(frame, language)
    if language_frame.empty:
        st.info(copy["empty"])
        return
    with st.expander(copy["filters"]):
        media = ["__all__", *sorted(language_frame.medium.dropna().unique())]
        selected_medium = st.selectbox(copy["medium"], media, format_func=lambda value: copy["all"] if value == "__all__" else value,
                              index=media.index("Ecrit") if "Ecrit" in media else 0, key=f"corpus_medium_{language}")
        medium = None if selected_medium == "__all__" else selected_medium
        start, end = int(language_frame.year.min()), int(language_frame.year.max())
        years = st.slider(copy["years"], start, end, (start, end), key=f"corpus_years_{language}") if start < end else (start, end)
    cohort = select_cohort(frame, language, medium, years)
    values = pd.to_numeric(cohort[metric], errors="coerce")
    values = values[np.isfinite(values)]
    current = current_values[metric]
    summary = distribution_summary(values, current)
    if not summary["n"]:
        st.info(copy["empty"])
        return
    st.caption(copy["coverage"].format(n=summary["n"], language=copy["language"],
        medium=medium or copy["all"], start=years[0], end=years[1]))
    columns = st.columns(4)
    value_format = ",.0f" if metric in COUNT_METRICS else ",.2f"
    columns[0].metric(copy["your"], format(current, value_format))
    columns[1].metric(copy["median"], format(summary["median"], ",.2f"))
    columns[2].metric(copy["rank"], f"{summary['percentile']:.0f}" if summary["n"] >= 5 else "—")
    columns[3].metric(copy["sample"], f"{summary['n']:,}")
    if summary["n"] < 20:
        st.warning(copy["small"].format(n=summary["n"]))
    st.altair_chart(build_chart(values, current, metric, language), width="stretch")
    st.caption(copy["legend"])
    st.info(copy[HELP_KEYS[group]])
    with st.expander(copy["method"]):
        st.write(copy["method_text"])
        st.write(copy["meaning"])
    download = pd.DataFrame([{
        "metric": metric, "language": language, "medium": medium or "all",
        "year_from": years[0], "year_to": years[1], "your_value": current,
        "reference_median": summary["median"], "reference_articles": summary["n"],
        "percentile": summary["percentile"] if summary["n"] >= 5 else None,
    }])
    st.download_button(copy["download"], download.to_csv(index=False),
                       file_name="jam_comparison.csv", mime="text/csv", key="corpus_download")
    with st.expander(copy["scatter"]):
        x = st.selectbox(copy["x_axis"], available, format_func=copy.get, key=f"corpus_x_{language}")
        y = st.selectbox(copy["y_axis"], available, index=available.index("sentiment_compound") if "sentiment_compound" in available else 1,
                         format_func=copy.get, key=f"corpus_y_{language}")
        st.altair_chart(build_scatter(cohort, current_values, x, y, language), width="stretch")
    all_rows = []
    for key in available:
        stats = distribution_summary(cohort[key], current_values[key])
        all_rows.append({"metric": key, "language": language, "medium": medium or "all",
                         "year_from": years[0], "year_to": years[1], "your_value": current_values[key],
                         "reference_median": stats["median"], "reference_articles": stats["n"],
                         "percentile": stats["percentile"] if stats["n"] >= 5 else None})
    st.download_button(copy["all_metrics"], pd.DataFrame(all_rows).to_csv(index=False),
                       file_name="jam_all_dimensions.csv", mime="text/csv")


def build_scatter(cohort: pd.DataFrame, current: dict, x: str, y: str, language: str) -> alt.LayerChart:
    """Explicit column aliases support identical axes and punctuation in spaCy labels."""
    copy = COPY[language]
    points = pd.DataFrame({"x": cohort[x], "y": cohort[y], "article_id": cohort.article_id,
                           "medium": cohort.medium, "year": cohort.year})
    points = points.loc[np.isfinite(points.x) & np.isfinite(points.y)]
    reference = alt.Chart(points).mark_circle(opacity=.45, size=40).encode(
        x=alt.X("x:Q", title=copy[x], scale=alt.Scale(zero=False)),
        y=alt.Y("y:Q", title=copy[y], scale=alt.Scale(zero=False)),
        tooltip=["article_id:N", "medium:N", "year:O", "x:Q", "y:Q"], color=alt.value("#319b98"),
    )
    marker = alt.Chart(pd.DataFrame({"x": [current[x]], "y": [current[y]], "label": [copy["your"]]})).mark_point(
        shape="diamond", filled=True, size=180, color="#f97316",
    ).encode(x="x:Q", y="y:Q", tooltip=["label:N", "x:Q", "y:Q"])
    return alt.layer(reference, marker).properties(height=380).interactive()
