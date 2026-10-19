#!/usr/bin/env python3
"""Build apps/data/corpus_reference.csv from assets/cleaned.csv.

Run in the app environment: python scripts/build_dashboard_reference.py
Rerun after changing analysis calculations or editing the cleaned source text.
Uses the dashboard's current analysis, including its French readability formula.
Old metric columns in cleaned.csv are ignored. No API calls are involved.
"""

import argparse
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps"))

from modules.basic_stats import analyze_additional_tools, analyze_basic_metrics, _analyze_text
from modules.corpus_reference import REFERENCE_PATH, comparison_values
from modules.sentiment_analysis import sentiment_dimensions, analyze_sentiment, analyze_sentiment_report


def build(source: Path, output: Path) -> pd.DataFrame:
    if source.resolve() == output.resolve():
        raise ValueError("Output must not overwrite cleaned.csv")
    columns = ["article_id", "body_text", "analysis_language", "medium", "year", "strict_benchmark_eligible"]
    original = pd.read_csv(source, usecols=columns, dtype={"article_id": str}, keep_default_na=False)
    eligible = original.strict_benchmark_eligible.astype(str).str.strip().str.lower().eq("true")
    selected = original.loc[
        eligible & original.analysis_language.isin(["fr", "en"])
        & original.body_text.str.strip().ne("")
    ].copy()
    if selected.empty or selected.article_id.eq("").any() or selected.article_id.duplicated().any():
        raise ValueError("Source must contain unique eligible articles with body text")
    selected["year"] = pd.to_numeric(selected.year, errors="raise").astype(int)
    print(f"Analyzing {len(selected)} reference articles from {len(original)} cleaned rows", flush=True)
    rows = []
    for index, article in enumerate(selected.itertuples(), 1):
        metrics = analyze_basic_metrics(article.body_text, article.analysis_language)
        extra = analyze_additional_tools(article.body_text, article.analysis_language)
        rows.append({
            "article_id": article.article_id, "year": article.year,
            "language": article.analysis_language, "medium": article.medium or "Unknown",
            **comparison_values(metrics, extra, sentiment_dimensions(article.body_text, article.analysis_language)),
        })
        # Bound memory while processing the corpus offline.
        analyze_basic_metrics.clear()
        analyze_additional_tools.clear()
        _analyze_text.clear()
        analyze_sentiment.clear()
        analyze_sentiment_report.clear()
        if index % 100 == 0 or index == len(selected):
            print(f"Analyzed {index}/{len(selected)}", flush=True)
    frame = pd.DataFrame(rows)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False, float_format="%.12g")
    print(f"Wrote {output}: {len(frame)} articles, {output.stat().st_size:,} bytes", flush=True)
    return frame


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "assets/cleaned.csv")
    parser.add_argument("--output", type=Path, default=REFERENCE_PATH)
    args = parser.parse_args()
    build(args.source, args.output)
