"""Apertus API access and journalistic angle drafts.

This module owns the journalistic angle tab. ``render_angle_analysis`` receives
the source text and language, displays the editable prompt, sends a request only
when Generate is clicked, and displays the editable response or a localised error.
Prompt/result drafts are kept separately for French and English across reruns.

``request_angle_analysis`` is the independently testable CSCS HTTP client.
``get_api_key`` reads Streamlit secrets, the environment, then the project .env;
credentials are never displayed. UI wording comes from the locale dictionaries.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from .language import DEFAULT_LANGUAGE, translator

API_URL = "https://api.inference.cscs.ch/v1/chat/completions"
MODEL = "swiss-ai/Apertus-v1.5-8B"


def get_api_key() -> str:
    """Read the key from Streamlit secrets, environment, or the ignored project .env."""
    try:
        secret = st.secrets.get("CSCS_INFERENCE_API_KEY", "")
    except Exception:
        secret = ""
    if secret:
        return str(secret).strip()

    key = os.getenv("CSCS_INFERENCE_API_KEY", "").strip()
    if key:
        return key

    env_file = Path(__file__).resolve().parents[2] / ".env"
    try:
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, separator, value = line.partition("=")
            if separator and name.strip() == "CSCS_INFERENCE_API_KEY":
                return value.strip().strip("\"'")
    except OSError:
        pass
    return ""


def _request_analysis(text: str, prompt: str, language_code: str = DEFAULT_LANGUAGE) -> str:
    """Send one explicit request; reject empty input and unusable API responses."""
    t = translator(language_code)
    if not text.strip():
        raise RuntimeError(t("empty_request"))
    if not prompt.strip():
        raise RuntimeError(t("empty_prompt"))
    key = get_api_key()
    if not key:
        raise RuntimeError(t("api_key_missing"))

    payload = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": prompt.strip()},
            {"role": "user", "content": text.strip()},
        ],
        "temperature": 0.2,
    }
    request = Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(t("api_http_error", code=exc.code)) from None
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(t("api_connection_error", reason=exc.reason if isinstance(exc, URLError) else t("api_timeout"))) from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError(t("api_invalid")) from None

    try:
        content = result["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError(t("api_no_analysis")) from None
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError(t("api_no_analysis"))
    return content.strip()


def request_angle_analysis(text: str, prompt: str, language_code: str = DEFAULT_LANGUAGE) -> str:
    return _request_analysis(text, prompt, language_code)


def save_angle_draft(field: str, language_code: str) -> None:
    """Keep edits when Streamlit cleans up widgets hidden by a language switch."""
    st.session_state["angle_drafts"][language_code][field] = st.session_state[f"angle_{field}_{language_code}"]


def render_angle_analysis(text: str, language_code: str = DEFAULT_LANGUAGE) -> None:
    """Render the full inference tab without relying on dashboard-local variables."""
    t = translator(language_code)
    st.subheader(t("angle_analysis"))
    drafts = st.session_state.setdefault("angle_drafts", {})
    draft = drafts.setdefault(language_code, {"prompt": t("angle_prompt"), "result": ""})
    prompt_key = f"angle_prompt_{language_code}"
    result_key = f"angle_result_{language_code}"
    st.session_state.setdefault(prompt_key, draft["prompt"])
    angle_prompt = st.text_area(
        t("request_prompt"),
        height=130,
        key=prompt_key,
        on_change=save_angle_draft,
        args=("prompt", language_code),
    )

    if st.button(t("generate_angle"), type="primary", key="generate_angle"):
        if not text.strip() or not angle_prompt.strip():
            st.warning(t("empty_request" if not text.strip() else "empty_prompt"))
        else:
            try:
                with st.spinner(t("requesting")):
                    draft["result"] = request_angle_analysis(text, angle_prompt, language_code)
                    st.session_state[result_key] = draft["result"]
            except RuntimeError as exc:
                st.error(str(exc))
    st.session_state.setdefault(result_key, draft["result"])
    st.text_area(
        t("editable_analysis"), height=280, key=result_key,
        on_change=save_angle_draft, args=("result", language_code),
    )
