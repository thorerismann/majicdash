"""Streamlit dashboard entry point and coordinator for the analysis modules.

Resolve the text language, collect writing and display preferences, then pass
them to the analysis modules and render the six tabs. Input layout belongs in
main_displays; submission validation in handle_inputs.
inference.py owns the shared API client and editable journalistic-angle drafts.
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
from modules.main_displays import render_sidebar, render_main, render_text_input, render_basic_stats_tab
from modules.handle_inputs import initial_text_analysis
from modules.corpus_comparison import COPY as CORPUS_COPY, render_corpus_comparison


def main() -> None:
    detected_language = detect_language(st.session_state.get("writing_text", ""))
    language_choice = st.session_state.get("language_choice", "auto")
    language_code = select_analysis_language(detected_language, language_choice)
    t = translator(language_code)
    render_main(language_code, t)
    render_sidebar(language_code, t)
    text = render_text_input(t)
    if not initial_text_analysis(t, text, language_choice, detected_language, SUPPORTED_LANGUAGES):
        return


    with st.spinner(t("analyzing")):
        metrics = analyze_basic_metrics(text, language_code)
        style = analyze_style(text, language_code)
        extra_tools = analyze_additional_tools(text, language_code)

    st.success(t("detected_language", detected=t("unknown") if detected_language == "unknown" else detected_language, language=t(language_code)))


    comparison_tab, tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [CORPUS_COPY[language_code]["tab"], t("basic_stats"), t("sentiment"), t("pos"), t("angle_machine"), t("journalistic_angle")]
    )

    with comparison_tab:
        render_corpus_comparison(metrics, extra_tools, language_code, sentiment_dimensions(text, language_code))

    with tab1:
        render_basic_stats_tab(metrics, style, extra_tools, language_code)

    with tab2:
        render_sentiment(text, language_code)

    with tab3:
        render_pos_highlighter(text, language_code)
        render_spacy_breakdown(text, language_code)

    with tab4:
        st.subheader(t("angle_title"))
        st.write(
            t("angle_intro")
        )
        render_visual_angle(language_code)

    with tab5:
        render_angle_analysis(text, language_code)



if __name__ == "__main__":
    main()
