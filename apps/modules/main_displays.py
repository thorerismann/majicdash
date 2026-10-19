import streamlit as st
from .language import get_words, translator
from .handle_inputs import submit_text
from .basic_stats import render_basic_stats, render_diagnostics, render_style, render_style_details


def render_basic_stats_tab(metrics: dict, style: dict, extra_tools: dict, language_code: str) -> None:
    """Priorities first, a short overview, then optional supporting details."""
    t = translator(language_code)
    render_style(style, language_code)
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


def render_tool_links(language_code) -> None:
    for tool in get_words(language_code)["tool_links"]:
        st.markdown(f"- [{tool['name']}]({tool['url']}): {tool['purpose']}")


def render_sidebar(language_code: str, t) -> None:
    st.sidebar.header(t("settings"))

    st.sidebar.selectbox(t("text_language"), ["auto", "fr", "en"],
                        format_func=t, key="language_choice")
    st.sidebar.toggle(t("sentence_flags"), value=True, key="show_examples")
    st.sidebar.markdown(
        t("local_notice")
        )
    st.sidebar.subheader(t("tools"))
    with st.sidebar:
        render_tool_links(language_code)

def render_main(language_code: str, t) -> None:
    st.set_page_config(page_title=t("app_title"), page_icon="📝", layout="wide")
    st.title(t("app_title"))
    st.caption(t("app_caption"))
