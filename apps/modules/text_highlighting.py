"""One escaped, offset-based renderer for local and AI-supported text findings."""

from html import escape
from typing import TypedDict

import streamlit as st


class Highlight(TypedDict):
    start: int
    end: int
    label: str


def highlighted_html(text: str, highlights: list[Highlight]) -> str:
    """Split at span boundaries so overlaps never nest marks or duplicate text."""
    spans = [span for span in highlights
             if type(span["start"]) is int and type(span["end"]) is int
             and 0 <= span["start"] < span["end"] <= len(text)]
    boundaries = sorted({0, len(text), *(span["start"] for span in spans),
                         *(span["end"] for span in spans)})
    pieces = []
    for start, end in zip(boundaries, boundaries[1:]):
        chunk = escape(text[start:end])
        labels = list(dict.fromkeys(span["label"] for span in spans
                                   if span["start"] <= start and span["end"] >= end))
        if labels:
            label = escape(" · ".join(labels), quote=True)
            chunk = f'<mark title="{label}" style="background-color: #ffe08a; color: #1f2937;">{chunk}</mark>'
        pieces.append(chunk)
    return '<div style="white-space: pre-wrap; overflow-wrap: anywhere; line-height: 1.8;">' + "".join(pieces) + '</div>'


def render_highlighted_text(text: str, highlights: list[Highlight]) -> None:
    """Static semantic markup: all source text and labels are escaped; no JavaScript."""
    with st.container(border=True):
        st.html(highlighted_html(text, highlights))


def match_evidence(text: str, sentences: list[dict], evidence: list[dict], label: str) -> tuple[list[Highlight], int]:
    """Resolve exact, unambiguous quotes within trusted sentence offsets only."""
    by_id = {row["sentence"]: row for row in sentences}
    highlights = []
    unmatched = 0
    for item in evidence:
        sentence_id, quote = item.get("sentence"), item.get("quote")
        row = by_id.get(sentence_id) if type(sentence_id) is int else None
        if row is None or not isinstance(quote, str) or not quote.strip():
            unmatched += 1
            continue
        start, end = row["start"], row["end"]
        passage = text[start:end]
        offset = passage.find(quote)
        if (passage != row["text"] or offset < 0
                or passage.find(quote, offset + 1) >= 0):
            unmatched += 1
            continue
        span = {"start": start + offset, "end": start + offset + len(quote), "label": label}
        if span not in highlights:
            highlights.append(span)
    return highlights, unmatched
