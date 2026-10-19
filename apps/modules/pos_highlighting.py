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


POS_MODELS = {"en": "en_core_web_sm", "fr": "fr_core_news_sm"}
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


POS_TAGS = tuple("ADJ ADP ADV AUX CCONJ DET INTJ NOUN NUM PART PRON PROPN PUNCT SCONJ SYM VERB X".split())
DEPENDENCY_LABELS = tuple(sorted(set("""
ROOT acl acl:relcl acomp advcl advmod agent amod appos attr aux auxpass aux:pass
aux:tense case cc ccomp compound conj cop csubj csubjpass dative dep det dobj
expl expl:comp expl:pass expl:subj fixed flat:foreign flat:name intj iobj mark meta
neg nmod npadvmod nsubj nsubjpass nsubj:pass nummod obj obl:agent obl:arg obl:mod
oprd parataxis pcomp pobj poss preconj predet prep prt punct quantmod relcl vocative xcomp
""".split())))
ENTITY_LABELS = tuple("CARDINAL DATE EVENT FAC GPE LANGUAGE LAW LOC MISC MONEY NORP ORDINAL ORG PERCENT PER PERSON PRODUCT QUANTITY TIME WORK_OF_ART".split())
MORPH_FEATURES = tuple("""Tense=Past Tense=Pres Tense=Fut Tense=Imp Mood=Ind Mood=Sub
Mood=Cnd Mood=Imp VerbForm=Fin VerbForm=Inf VerbForm=Part VerbForm=Ger
Number=Sing Number=Plur Person=1 Person=2 Person=3 Voice=Pass Polarity=Neg""".split())
BREAKDOWN_LABELS = {"pos": POS_TAGS, "dep": DEPENDENCY_LABELS, "entity": ENTITY_LABELS, "morph": MORPH_FEATURES}
SPACY_METRIC_KEYS = (
    "token_count", "entity_count", "entities_per_100_words", "stopword_ratio", "lemma_diversity",
    "avg_word_length", "avg_dependency_distance",
    *(f"{kind}_{label}_{suffix}" for kind, labels in BREAKDOWN_LABELS.items()
      for label in labels for suffix in ("count", "ratio")),
)


def describe_doc(doc: Doc, text: str, language_code: str) -> dict:
    """Derive exportable annotations and numeric dimensions from the shared parse.

    POS, dependency and morphology ratios divide by all non-space tokens.
    Entity ratios divide by entity mentions. Unsupported model labels are None.
    """
    nlp = load_pos_model(POS_MODELS[language_code])
    tokens = [token for token in doc if not token.is_space]
    words = [token for token in tokens if any(char.isalpha() for char in token.text)]
    counts = {
        "pos": Counter(token.pos_ for token in tokens),
        "dep": Counter(token.dep_ for token in tokens),
        "entity": Counter(entity.label_ for entity in doc.ents),
        "morph": Counter(feature for token in tokens for feature in str(token.morph).split("|") if feature),
    }
    supported = {"dep": set(nlp.get_pipe("parser").labels), "entity": set(nlp.get_pipe("ner").labels)}
    metrics = {
        "token_count": len(tokens), "entity_count": len(doc.ents),
        "entities_per_100_words": 100 * len(doc.ents) / len(words) if words else 0.0,
        "stopword_ratio": sum(token.is_stop for token in words) / len(words) if words else 0.0,
        "lemma_diversity": len({token.lemma_.lower() for token in words}) / len(words) if words else 0.0,
        "avg_word_length": sum(len(token.text) for token in words) / len(words) if words else 0.0,
        "avg_dependency_distance": sum(abs(token.i - token.head.i) for token in tokens) / len(tokens) if tokens else 0.0,
    }
    for kind, labels in BREAKDOWN_LABELS.items():
        denominator = len(doc.ents) if kind == "entity" else len(tokens)
        for label in labels:
            available = kind not in supported or label in supported[kind]
            count = counts[kind][label] if available else None
            metrics[f"{kind}_{label}_count"] = count
            metrics[f"{kind}_{label}_ratio"] = (count / denominator if denominator else 0.0) if available else None
    return {
        "metrics": metrics,
        "counts": counts,
        "tokens": [{
            "index": token.i, "text": text[token.idx:token.idx + len(token)],
            "lemma": token.lemma_, "pos": token.pos_, "tag": token.tag_,
            "dependency": token.dep_, "head_index": token.head.i,
            "head": text[token.head.idx:token.head.idx + len(token.head)],
            "morphology": str(token.morph), "entity": token.ent_type_,
            "is_stop": token.is_stop, "start": token.idx, "end": token.idx + len(token),
        } for token in tokens],
        "entities": [{"text": text[entity.start_char:entity.end_char], "label": entity.label_,
                      "start": entity.start_char, "end": entity.end_char} for entity in doc.ents],
    }


def render_spacy_breakdown(text: str, language_code: str) -> None:
    import pandas as pd
    import streamlit as st
    from .basic_stats import _analyze_text
    from .language import translator

    t = translator(language_code)
    report = _analyze_text(text, language_code)["spacy"]
    st.subheader(t("spacy_breakdown"))
    st.caption(t("spacy_breakdown_help"))
    kind = st.selectbox(t("breakdown_kind"), list(BREAKDOWN_LABELS),
                        format_func=lambda value: t(f"breakdown_{value}"), key="spacy_breakdown_kind")
    rows = [{"label": label, "count": count, "description": spacy.explain(label) or label}
            for label, count in report["counts"][kind].most_common()]
    if rows:
        frame = pd.DataFrame(rows)
        st.bar_chart(frame, x="label", y="count")
        st.dataframe(frame, hide_index=True, width="stretch")
    else:
        st.info(t("no_annotations"))
    with st.expander(t("token_annotations")):
        tokens = pd.DataFrame(report["tokens"])
        st.dataframe(tokens, hide_index=True, width="stretch")
        st.download_button(t("download_tokens"), tokens.to_csv(index=False),
                           file_name="spacy_tokens.csv", mime="text/csv")
    with st.expander(t("entity_annotations")):
        st.dataframe(pd.DataFrame(report["entities"], columns=["text", "label", "start", "end"]),
                     hide_index=True, width="stretch")
