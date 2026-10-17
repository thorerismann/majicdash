"""Visual angle-planning tool: embedded wheel and native selection controls.

Load the HTML asset from the project's assets directory and inject the selected
French/English dictionary. ``display_angle`` renders the wheel, linked dropdowns,
and a suggested angle assembled by ``build_angle_output``. Stable option indices
preserve dropdown choices across language switches. This module makes no AI calls;
the separate inference module owns generated journalistic angle analyses.
"""

import json
import streamlit as st
import streamlit.components.v1 as components
from .language import DEFAULT_LANGUAGE, translator, get_words
from pathlib import Path

ANGLE_MACHINE_PATH = Path(__file__).resolve().parents[2] / "assets" / "code_machine_angle.html"


@st.cache_data(show_spinner=False)
def load_angle_machine_html(language_code: str = DEFAULT_LANGUAGE) -> str:
    if not ANGLE_MACHINE_PATH.exists():
        return ""
    # Escape '<' so locale content cannot terminate the embedded script element.
    payload = json.dumps(get_words(language_code), ensure_ascii=False).replace("<", "\\u003c")
    return ANGLE_MACHINE_PATH.read_text(encoding="utf-8").replace("/* LOCALE_DATA */ null", payload)


def build_angle_output(selections: dict[str, str], language_code: str = DEFAULT_LANGUAGE) -> dict[str, str]:
    t = translator(language_code)
    angle_line = (
        f"{selections['time']} | {selections['function_1']} -> {selections['function_2']} | "
        f"{selections['perspective']} | {selections['format']} | {selections['focus']}"
    )
    return {"angle_line": angle_line, "starter": t("angle_starter", **selections)}


def render_visual_angle(language_code: str):
    t = translator(language_code)
    angle_html = load_angle_machine_html(language_code)
    if angle_html:
        components.html(angle_html, height=980, scrolling=False)
    else:
        st.warning(t("angle_missing"))

    st.markdown(t("angle_output"))
    st.write(
        t("angle_companion")
    )

    wheels = get_words(language_code)["angle"]["wheels"]
    angle_columns = st.columns(3)
    selections = {}
    for index, wheel in enumerate(wheels):
        with angle_columns[index // 2]:
            options = list(range(len(wheel["items"])))
            if wheel["id"] == "function_2":
                parent = st.session_state.get("angle_function_1", 0)
                options = [parent * 2, parent * 2 + 1]
            key = "angle_" + wheel["id"]
            if st.session_state.get(key) not in options:
                st.session_state[key] = options[0]
            selected = st.selectbox(wheel["name"], options,
                                    format_func=lambda i, items=wheel["items"]: items[i], key=key)
            selections[wheel["id"]] = wheel["items"][selected]
    angle_output = build_angle_output(selections, language_code)
    st.code(angle_output["angle_line"], language="text")
    st.caption(angle_output["starter"])

    st.caption(
        t("angle_help")
    )
