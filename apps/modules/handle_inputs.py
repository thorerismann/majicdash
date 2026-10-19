"""Text submission validation and analysis state."""

import streamlit as st


def submit_text() -> None:
    """Run before rendering so the input closes on every valid submission."""
    text = st.session_state.get("writing_text", "")
    valid = bool(text.strip())
    st.session_state["input_expanded"] = not valid
    st.session_state["analysis_requested"] = valid
    st.session_state["submitted_text"] = text if valid else ""
    st.session_state["empty_submission"] = not valid


def initial_text_analysis(t, text, language_choice, detected_language, supported_languages) -> bool:
    """The coordinator must return immediately when this gate is false."""
    if not text.strip():
        if st.session_state.get("empty_submission") or st.session_state.get("analysis_requested"):
            st.warning(t("empty_text"))
        else:
            st.info(t("start_hint"))
        return False
    if not st.session_state.get("analysis_requested") or text != st.session_state.get("submitted_text"):
        st.info(t("start_hint"))
        return False
    if language_choice == "auto" and detected_language not in supported_languages:
        st.info(t("language_fallback", detected=detected_language))
    return True
