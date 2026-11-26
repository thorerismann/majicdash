import streamlit as st
from .language import get_words, translator
from .handle_inputs import submit_text
from .basic_stats import render_basic_stats, render_diagnostics, render_style, render_style_details
from .inference import MODEL
from .text_highlighting import render_highlighted_text


def render_basic_stats_tab(text: str, metrics: dict, style: dict, extra_tools: dict, language_code: str) -> None:
    """Priorities first, a short overview, then optional supporting details."""
    t = translator(language_code)
    render_style(style, language_code)
    st.subheader(t("highlight_text"))
    selected = st.segmented_control(
        t("highlight_show"), ["all", "long_sentences", "repetition"],
        format_func=lambda value: t(f"highlight_{value}"), default="all", key="style_highlight_kind",
    ) or "all"
    highlights = [span for span in style["highlights"] if selected == "all" or span["kind"] == selected]
    st.caption(t("highlight_style_help"))
    if not highlights:
        st.caption(t("highlight_none"))
    render_highlighted_text(text, highlights)
    render_basic_stats(metrics, language_code)
    with st.expander(t("review_details")):
        render_style_details(style, st.session_state.get("show_examples", True), language_code)
    with st.expander(t("additional_statistics")):
        render_diagnostics(t, extra_tools, metrics)


def render_text_input(t) -> str:
    with st.expander(t("input_panel"), expanded=True, key="input_expanded", on_change="rerun"):
        text = st.text_area(t("input_label"), key="writing_text", height=320,
                            placeholder=t("input_placeholder"))
    st.button(t("analyze"), key="analyze_writing", type="primary",
              width="stretch", on_click=submit_text)
    return text


def render_footer(language_code: str, t) -> None:
    """Keep credits available below the workspace without crowding the sidebar."""
    st.divider()
    with st.expander(t("credits"), expanded=False, type="compact", icon=":material/info:"):
        st.markdown(t("generation_credit", model=MODEL))
        st.caption(t("local_notice"))
        st.markdown(f"**{t('tools')}**")
        st.markdown("\n".join(
            f"- [{tool['name']}]({tool['url']}): {tool['purpose']}"
            for tool in get_words(language_code)["tool_links"]
        ))
        st.markdown(t("development_credit"))


def render_sidebar(t, views: dict[str, str]) -> str:
    st.sidebar.subheader(t("analysis_view"))
    selected_view = st.sidebar.radio(
        t("analysis_view"), list(views), format_func=views.get,
        key="analysis_view", label_visibility="collapsed",
    )
    st.sidebar.markdown(f"**{t('settings')}**")

    st.sidebar.selectbox(t("text_language"), ["auto", "fr", "en"],
                        format_func=t, key="language_choice")
    st.sidebar.toggle(t("sentence_flags"), value=True, key="show_examples")
    return selected_view

def render_main(language_code: str, t) -> None:
    st.set_page_config(page_title=t("app_title"), page_icon=":material/edit_note:", layout="wide")
    st.title(t("app_title"))
    st.caption(t("app_caption"))
