"""Streamlit dashboard entry point and coordinator for the analysis modules.

Resolve the text language, collect writing and display preferences, then pass
them to the analysis modules and render the selected view. Input layout belongs in
main_displays; submission validation in handle_inputs.
inference.py owns the API client and angle drafts; ai_summary combines local evidence.
Run with ``python -m streamlit run apps/writer_dashboard.py`` from the project root.
"""

from __future__ import annotations

import streamlit as st

from modules.language import (
    SUPPORTED_LANGUAGES,
    detect_language,
    select_analysis_language,
    translator,
)
from modules.pos_highlighting import render_pos_highlighter, render_spacy_breakdown
from modules.sentiment_analysis import render_sentiment, sentiment_dimensions
from modules.basic_stats import analyze_basic_metrics, analyze_additional_tools, analyze_style
from modules.angle_visual import render_visual_angle
from modules.inference import render_angle_analysis
from modules.ai_summary import render_ai_summary
from modules.main_displays import render_sidebar, render_main, render_text_input, render_basic_stats_tab, render_footer
from modules.handle_inputs import initial_text_analysis
from modules.corpus_comparison import COPY as CORPUS_COPY, render_corpus_comparison


def main() -> None:
    detected_language = detect_language(st.session_state.get("writing_text", ""))
    language_choice = st.session_state.get("language_choice", "auto")
    language_code = select_analysis_language(detected_language, language_choice)
    t = translator(language_code)
    render_main(language_code, t)
    selected_view = render_sidebar(t, {
        "comparison": CORPUS_COPY[language_code]["tab"],
        "ai_summary": t("ai_summary"),
        "basic_stats": t("basic_stats"),
        "sentiment": t("sentiment"),
        "pos": t("pos"),
        "angle_machine": t("angle_machine"),
        "journalistic_angle": t("journalistic_angle"),
    })
    text = render_text_input(t)
    if not initial_text_analysis(t, text, language_choice, detected_language, SUPPORTED_LANGUAGES):
        render_footer(language_code, t)
        return


    if selected_view in ("comparison", "basic_stats"):
        with st.spinner(t("analyzing")):
            metrics = analyze_basic_metrics(text, language_code)
            extra_tools = analyze_additional_tools(text, language_code)

    st.success(t("detected_language", detected=t("unknown") if detected_language == "unknown" else detected_language, language=t(language_code)))


    if selected_view == "comparison":
        render_corpus_comparison(metrics, extra_tools, language_code, sentiment_dimensions(text, language_code))

    elif selected_view == "basic_stats":
        style = analyze_style(text, language_code)
        render_basic_stats_tab(text, metrics, style, extra_tools, language_code)

    elif selected_view == "sentiment":
        render_sentiment(text, language_code)

    elif selected_view == "pos":
        render_pos_highlighter(text, language_code)
        render_spacy_breakdown(text, language_code)

    elif selected_view == "angle_machine":
        st.subheader(t("angle_title"))
        st.write(
            t("angle_intro")
        )
        render_visual_angle(language_code)

    elif selected_view == "journalistic_angle":
        render_angle_analysis(text, language_code)

    elif selected_view == "ai_summary":
        render_ai_summary(text, language_code)

    render_footer(language_code, t)


if __name__ == "__main__":
    main()
