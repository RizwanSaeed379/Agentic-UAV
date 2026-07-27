"""
shared/agent_loop.py — Bridge between LLM and tool registry.

Handles one complete reasoning cycle per call:
  build prompt → call LLM → parse → execute tools → return flight command.

Shared by all three paradigms (ReAct, Plan-Execute, Reflexion).
No LangChain. No LangGraph. Plain Python only.
"""

from __future__ import annotations

import logging
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.llm_utils import call_ollama, parse_json_response
from shared.tools import execute_tool

# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

_fmt = logging.Formatter(
    "%(asctime)s  %(name)-12s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(_fmt)

log = logging.getLogger("agent_loop")
log.setLevel(logging.DEBUG)
if not log.handlers:
    log.addHandler(_handler)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_MAX_TOOL_CALLS = 5
# fallback=True marks this as a safe-default RTL issued because the LLM never
# produced a valid flight command (parse failure / exhausted retries), NOT a
# decision the agent made. Scoring uses this to distinguish a parse failure from
# a genuine RTL abort.
_SAFE_DEFAULT   = {"type": "flight_command", "command": "RTL", "params": {},
                   "fallback": True}


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

def build_prompt(system_prompt: str, history: list, telemetry: dict) -> str:
    """
    Assemble the full prompt string sent to the LLM.

    Sections:
      {system_prompt}

      === MISSION HISTORY ===
      {each history item on its own line}

      === CURRENT TELEMETRY ===
      {telemetry as key: value lines}

      === YOUR TASK ===
      Based on the above, what is your next action? Output JSON only.
    """
    parts = [system_prompt.strip()]

    parts.append("\n=== MISSION HISTORY ===")
    if history:
        parts.extend(history)
    else:
        parts.append("(no history — this is the first step)")

    parts.append("\n=== CURRENT TELEMETRY ===")
    for key, val in telemetry.items():
        parts.append(f"{key}: {val}")

    parts.append(
        "\n=== YOUR TASK ===\n"
        "Based on the above, what is your next action? Output JSON only."
    )

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Agent step
# ---------------------------------------------------------------------------

def agent_step(
    model:         str,
    system_prompt: str,
    telemetry:     dict,
    history:       list,
    uav_state:     dict,
    step_label:    str = "",
) -> dict:
    """
    One complete agent reasoning step.

    Runs a tool-call inner loop (up to _MAX_TOOL_CALLS times) until the LLM
    produces a flight_command, then returns that command dict.

    Args:
        model:         Ollama model tag (e.g. MODEL_REACT).
        system_prompt: Full system prompt string for this paradigm.
        telemetry:     Current UAV telemetry as a flat dict.
        history:       Mutable list of step strings (modified in-place).
        uav_state:     Live UAV state dict passed through to tool calls.
                       NOTE: pass uav.get_state() at call time so tools have
                       a fresh snapshot independent of the telemetry dict
                       that was already injected into the prompt.
        step_label:    Optional label printed in log lines (e.g. "STEP_3").

    Returns:
        A flight command dict, always. Falls back to RTL on any failure.
    """
    label = f"[{step_label}] " if step_label else ""

    # Build initial prompt
    prompt = build_prompt(system_prompt, history, telemetry)

    last_tool:   str  = ""
    last_args:   dict = {}
    last_result: str  = ""

    for tool_call_n in range(_MAX_TOOL_CALLS + 1):
        # -----------------------------------------------------------------
        # 1. Call LLM
        # -----------------------------------------------------------------
        log.info("%sCalling %s (tool_round=%d) ...", label, model, tool_call_n)
        t0      = time.monotonic()
        raw     = call_ollama(model, prompt)
        elapsed = time.monotonic() - t0
        log.debug("%sLLM response in %.1fs: %r", label, elapsed, raw[:200])

        # -----------------------------------------------------------------
        # 2. Parse
        # -----------------------------------------------------------------
        parsed = parse_json_response(raw)

        if not parsed:
            log.warning("%sFailed to parse LLM response (round %d)", label, tool_call_n)
            if tool_call_n == _MAX_TOOL_CALLS:
                break
            prompt += (
                f"\n\nPREVIOUS OUTPUT WAS NOT VALID JSON: {raw[:300]}\n"
                "Output valid JSON only."
            )
            continue

        action_type = parsed.get("type", "")

        # -----------------------------------------------------------------
        # 3. Flight command — done
        # -----------------------------------------------------------------
        if action_type == "flight_command":
            command = parsed.get("command", "UNKNOWN")
            params  = parsed.get("params", {})
            entry   = f"Command: {command} params={params}"
            history.append(entry)
            log.info("%s%s", label, entry)
            return parsed

        # -----------------------------------------------------------------
        # 4. Mission plan (Plan-Execute output) — treat as done
        # -----------------------------------------------------------------
        if action_type == "mission_plan":
            entry = f"Command: MISSION_PLAN steps={len(parsed.get('steps', []))}"
            history.append(entry)
            log.info("%s%s", label, entry)
            return parsed

        # -----------------------------------------------------------------
        # 5. Tool call
        # -----------------------------------------------------------------
        if action_type == "tool_call":
            if tool_call_n >= _MAX_TOOL_CALLS:
                log.warning(
                    "%sMax tool calls (%d) reached — falling back to RTL",
                    label, _MAX_TOOL_CALLS,
                )
                break

            tool_name = parsed.get("tool_name", "")
            arguments = parsed.get("arguments", {})

            # Deduplicate repeated identical tool calls
            if tool_name == last_tool and arguments == last_args:
                log.info(
                    "%sDuplicate tool call %s — reusing previous result",
                    label, tool_name,
                )
                prompt += f"\n\nTool result (cached): {last_result}\nNext action:"
                continue

            log.info("%sTool call: %s(%s)", label, tool_name, arguments)
            result = execute_tool(tool_name, arguments, uav_state)
            log.info("%sTool result: %s", label, result)

            last_tool   = tool_name
            last_args   = arguments
            last_result = result

            entry = f"Tool: {tool_name} -> {result}"
            history.append(entry)

            # Rebuild prompt with tool result appended
            prompt  = build_prompt(system_prompt, history, telemetry)
            prompt += f"\n\nTOOL RESULT: {result}\nNow output your next action as JSON."
            continue

        # -----------------------------------------------------------------
        # 6. Unknown type — nudge and retry
        # -----------------------------------------------------------------
        log.warning(
            "%sUnrecognised action type %r (round %d) — retrying",
            label, action_type, tool_call_n,
        )
        prompt += (
            f"\n\nYour last output had an unrecognised type: {action_type!r}. "
            "Output JSON with type 'tool_call' or 'flight_command' only."
        )

    # ---------------------------------------------------------------------
    # Fallback: safe default RTL
    # ---------------------------------------------------------------------
    log.warning(
        "%sAll %d rounds exhausted without a flight command — issuing RTL",
        label, _MAX_TOOL_CALLS,
    )
    history.append("Command: RTL params={} [FALLBACK]")
    return _SAFE_DEFAULT
