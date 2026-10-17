import streamlit as st
from .language import get_words


def render_tool_links(language_code) -> None:
    for tool in get_words(language_code)["tool_links"]:
        st.markdown(f"- [{tool['name']}]({tool['url']}): {tool['purpose']}")


def render_sidebar(language_code: str, t) -> None:
    st.sidebar.header(t("settings"))

    st.sidebar.selectbox(t("text_language"), ["auto", "fr", "en"],
                        format_func=t, key="language_choice")
    show_examples = st.sidebar.toggle(t("sentence_flags"), value=True, key="show_examples")
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