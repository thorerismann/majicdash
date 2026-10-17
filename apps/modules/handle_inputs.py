import streamlit as st

def accept_text(t):
    text = st.text_area(t("input_label"), key="writing_text", height=320, placeholder=t("input_placeholder"))
    return text


def initial_text_analysis(t, text, language_choice, detected_language, SUPPORTED_LANGUAGES):
    analyze = st.button(t("analyze"), key="analyze_writing", type="primary", use_container_width=True)
    if analyze:
        st.session_state["analysis_requested"] = True

    if not st.session_state.get("analysis_requested", False):
        st.info(t("start_hint"))
        return

    if not text.strip():
        st.warning(t("empty_text"))
        return

    if language_choice == "auto" and detected_language not in SUPPORTED_LANGUAGES:
        st.info(t("language_fallback", detected=detected_language))
