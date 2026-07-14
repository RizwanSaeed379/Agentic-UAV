"""
shared/llm_utils.py — Ollama inference utilities for agentic UAV.

Plain requests only — no LangChain, no LangGraph.
"""

from __future__ import annotations

import json
import re
import os

import requests

# ---------------------------------------------------------------------------
# Model constants
# ---------------------------------------------------------------------------

MODEL_REACT    = "llama3"
MODEL_PLANNER  = "qwen2.5:7b"
MODEL_EXECUTOR = "mistral"
MODEL_CRITIC   = "llama3"
MODEL_ACTOR    = "qwen2.5:7b"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------




# keep localhost for local CPU.

_BASE_URL = os.environ.get(
    'OLLAMA_URL',
    'http://localhost:11434/api/generate'
)
_TIMEOUT  = 60    # seconds — GPU inference is fast; 60s is generous headroom




# ---------------------------------------------------------------------------
# Core inference
# ---------------------------------------------------------------------------

def call_ollama(model_name: str, prompt: str, temperature: float = 0.0) -> str:
    """
    Send a prompt to Ollama (local or remote via ngrok) and return the response.

    Args:
        model_name:  Ollama model tag (e.g. "llama3").
        prompt:      The full prompt string.
        temperature: Sampling temperature (default 0.2 for deterministic output).

    Returns:
        The model's response as a plain string, or an error message string.

    NOTE: 'format':'json' forces Ollama to emit valid JSON directly,
    preventing markdown wrapping and preamble text that breaks parsing.
    """
    payload = {
        "model":  model_name,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": temperature,
        },
    }

    try:
        resp = requests.post(
            _BASE_URL,
            json=payload,
            headers=_HEADERS,
            timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        return resp.json().get("response", "")

    except requests.exceptions.ConnectionError:
        msg = "ERROR: Cannot reach Ollama. Check _BASE_URL or ngrok tunnel."
        print(msg)
        return msg

    except requests.exceptions.Timeout:
        msg = f"ERROR: LLM timeout after {_TIMEOUT}s. Model may still be loading."
        print(msg)
        return msg

    except requests.exceptions.HTTPError as exc:
        msg = f"ERROR: Ollama HTTP error: {exc}"
        print(msg)
        return msg

    except Exception as exc:
        msg = f"ERROR: Unexpected error calling Ollama: {exc}"
        print(msg)
        return msg


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------

def parse_json_response(response_str: str) -> dict:
    """
    Safely parse a JSON object from an LLM response.

    With format:'json' set in call_ollama, responses should already be clean
    JSON. This function handles edge cases where models still add surrounding
    text despite the format constraint.

    Strategy (in order):
      1. Strip markdown code fences and try json.loads on the whole text.
      2. Walk the text to find the first balanced '{' ... '}' block and try
         json.loads on that substring — handles any residual prose wrapping.

    Returns an empty dict on any failure.
    """
    if not response_str or response_str.startswith("ERROR:"):
        return {}

    # Strip markdown code fences
    text = response_str.strip()
    text = re.sub(r"```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"```", "", text)
    text = text.strip()

    # Pass 1 — whole text is clean JSON
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except json.JSONDecodeError:
        pass

    # Pass 2 — extract first balanced { … } block from free-form text
    start = text.find("{")
    if start != -1:
        depth = 0
        for idx in range(start, len(text)):
            if text[idx] == "{":
                depth += 1
            elif text[idx] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : idx + 1]
                    try:
                        result = json.loads(candidate)
                        if isinstance(result, dict):
                            return result
                    except json.JSONDecodeError:
                        pass
                    break

    print(f"parse_json_response: no valid JSON object found")
    print(f"  Raw: {text!r}")
    return {}