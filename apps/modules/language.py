"""Shared language selection, JSON dictionaries, and translated wording.

Detect the submitted text language and resolve automatic/manual choices using
stable language codes. French is the default for empty or unsupported input.
get_words loads each locale's labels, prompts, word lists, and angle choices;
translate and translator format labels with named values from those dictionaries.
Callers pass the language explicitly so separate sessions and cached analyses
remain independent. This module has no Streamlit widgets or session-state logic.
"""

from functools import lru_cache, partial
import json
from pathlib import Path
import streamlit as st

from langdetect import DetectorFactory, LangDetectException, detect


DEFAULT_LANGUAGE = "fr"
SUPPORTED_LANGUAGES = ("fr", "en")
LOCALES_PATH = Path(__file__).resolve().parents[1] / "locales"
DetectorFactory.seed = 0


def detect_language(text: str) -> str:
    if not text.strip():
        return "unknown"
    try:
        return detect(text)
    except LangDetectException:
        return "unknown"


def select_analysis_language(detected: str, choice: str = "auto") -> str:
    """Resolve stable option codes; unknown/unsupported detection defaults to French."""
    if choice in SUPPORTED_LANGUAGES:
        return choice
    if choice != "auto":
        st.error("Analysis only valid for text in French and English. For additional languages contact the AJM at UNINE.")
    return detected if detected in SUPPORTED_LANGUAGES else DEFAULT_LANGUAGE


@lru_cache(maxsize=2)
def get_words(language_code: str = DEFAULT_LANGUAGE) -> dict:
    """Load a locale once. Treat the returned dictionary as read-only."""
    if language_code not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported language: {language_code}")
    path = LOCALES_PATH / f"json_words_{language_code}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def translate(key: str, locale: str = DEFAULT_LANGUAGE, **values) -> str:
    return get_words(locale)["labels"][key].format(**values)


def translator(language_code: str = DEFAULT_LANGUAGE):
    """Bind a translator locally without changing any process-wide UI language."""
    return partial(translate, locale=language_code)
