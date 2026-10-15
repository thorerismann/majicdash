"""Shared sentence splitting and Unicode word tokenization for local analysis."""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [sentence.strip() for sentence in sentences if sentence.strip()]


def tokenize_words(text: str) -> list[str]:
    # Keep Unicode letters (including œ) and normalize typographic apostrophes.
    return re.findall(r"[^\W\d_]+(?:['-][^\W\d_]+)*", text.lower().replace("’", "'"))
