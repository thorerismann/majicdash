"""Small client for the CSCS OpenAI-compatible chat completion endpoint."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

API_URL = "https://api.inference.cscs.ch/v1/chat/completions"
MODEL = "swiss-ai/Apertus-8B-Instruct-2509"


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


def request_angle_analysis(text: str, prompt: str) -> str:
    key = get_api_key()
    if not key:
        raise RuntimeError("CSCS_INFERENCE_API_KEY is not configured. Add it to .env or the environment.")

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
        raise RuntimeError(f"Inference API returned HTTP {exc.code}.") from None
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Could not reach the inference API: {exc.reason if isinstance(exc, URLError) else 'request timed out'}") from None
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise RuntimeError("The inference API returned an invalid response.") from None

    try:
        return result["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, TypeError, AttributeError):
        raise RuntimeError("The inference API response did not contain an analysis.") from None
