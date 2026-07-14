"""
main.py — Entry point for the Agentic UAV System.

Presents a paradigm selection menu and runs the chosen agent.
Paradigm modules are imported lazily to avoid slow startup or import errors
blocking the menu.
"""

from __future__ import annotations

import os
import socket
import sys
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
_MEMORY_FILE  = os.path.join(_PROJECT_ROOT, "reflexion_memory.txt")

# ---------------------------------------------------------------------------
# Console encoding — avoid UnicodeEncodeError on Windows cp1252
# ---------------------------------------------------------------------------

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Banner and checklist
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print(
        "\n"
        "╔" + "═" * 58 + "╗\n"
        "║         AGENTIC UAV SYSTEM — Rawalpindi SITL             ║\n"
        "║         Wildfire Boundary Mapping Mission                ║\n"
        "╠" + "═" * 58 + "╣\n"
        "║  A — ReAct Agent        (llama3)                         ║\n"
        "║  B — Plan-and-Execute   (qwen2.5:7b + mistral)           ║\n"
        "║  C — Reflexion Agent    (qwen2.5:7b × 2)                 ║\n"
        "║  Q — Quit                                                ║\n"
        "╚" + "═" * 58 + "╝"
    )


def print_preflight_checklist() -> bool:
    """Print the pre-flight checklist and ask for user confirmation."""
    print(
        "\nPRE-FLIGHT CHECKLIST\n"
        + "─" * 45 + "\n"
        + "[ ] Mission Planner is open\n"
        + "[ ] SITL is running (Simulation -> Plane)\n"
        + "[ ] Home location set to 33.7097, 72.9673\n"
        + "[ ] Plane is visible on map near Rawalpindi\n"
        + "[ ] Ollama is running (http://localhost:11434)\n"
        + "─" * 45
    )
    answer = input("Confirm all checks passed? [y/n]: ").strip().lower()
    if answer != "y":
        print("Complete the checklist first.")
        return False
    return True


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def check_ollama() -> bool:
    """
    GET http://localhost:11434 — returns True if Ollama responds.
    Prints status either way.
    """
    try:
        with urllib.request.urlopen(
            "http://localhost:11434", timeout=3
        ) as resp:
            body = resp.read().decode("utf-8", errors="ignore")
            if resp.status == 200 and "Ollama is running" in body:
                print("  Ollama        [OK] — http://localhost:11434")
                return True
            print(f"  Ollama        [UNEXPECTED RESPONSE] status={resp.status}")
            return False
    except urllib.error.URLError:
        print("  Ollama        [NOT RUNNING] — start with: ollama serve")
        return False
    except Exception as exc:
        print(f"  Ollama        [ERROR] {exc}")
        return False


def check_sitl() -> bool:
    """
    Try TCP connect to 127.0.0.1:5762 then :5760.
    Returns True if either responds. Prints result.
    """
    for port in (5762, 5760):
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=2):
                print(f"  SITL          [OK] — tcp:127.0.0.1:{port}")
                return True
        except (ConnectionRefusedError, socket.timeout, OSError):
            continue
    print("  SITL          [NOT DETECTED] — start Mission Planner SITL first")
    return False


# ---------------------------------------------------------------------------
# Memory file summary (no paradigm import needed)
# ---------------------------------------------------------------------------

def _memory_status() -> str:
    if not os.path.exists(_MEMORY_FILE):
        return "reflexion_memory.txt — not found (attempt 1 when C is run)"
    try:
        with open(_MEMORY_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        count = contents.count("=== ATTEMPT-BLOCK-START")
        return (
            f"reflexion_memory.txt — {count} past attempt(s) recorded "
            f"(next will be #{count + 1})"
        )
    except OSError:
        return "reflexion_memory.txt — unreadable"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ask_scenario() -> str:
    """Ask which scenario this run belongs to. The run number is derived
    automatically from the runs already recorded in uav_test_results.jsonl."""
    while True:
        sc = input("Scenario id [SC1-SC6, Enter for SC1]: ").strip().upper() or "SC1"
        if sc in {"SC1", "SC2", "SC3", "SC4", "SC5", "SC6"}:
            return sc
        print("Invalid scenario. Enter SC1 through SC6.")


def main() -> None:
    print_banner()

    # System info
    print(f"\nSystem info:")
    print(f"  Python        {sys.version.split()[0]}")
    print(f"  Project       {_PROJECT_ROOT}")
    print(f"  Memory        {_memory_status()}")

    # Auto health checks on startup
    print("\nStartup checks:")
    check_ollama()
    check_sitl()

    # Menu loop
    while True:
        choice = input("\nSelect paradigm [A/B/C/Q]: ").strip().upper()

        if choice == "A":
            scenario = ask_scenario()
            if not print_preflight_checklist():
                continue
            from paradigms.react_agent import run_react_mission
            run_react_mission(scenario_id=scenario)

        elif choice == "B":
            scenario = ask_scenario()
            if not print_preflight_checklist():
                continue
            from paradigms.plan_execute import run_plan_execute_mission
            run_plan_execute_mission(scenario_id=scenario)

        elif choice == "C":
            scenario = ask_scenario()
            if not print_preflight_checklist():
                continue
            from paradigms.reflexion_agent import (
                run_reflexion_mission,
                get_attempt_number,
                MEMORY_FILE,
            )
            attempt = get_attempt_number()
            print(f"\nReflexion — Attempt #{attempt}")
            if attempt > 1:
                print(f"Loading {attempt - 1} past reflection(s) from memory")
            run_reflexion_mission(scenario_id=scenario)
            print(f"\nAttempt #{attempt} complete.")
            print("Run again (select C) to attempt with accumulated memory.")

        elif choice == "Q":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Enter A, B, C, or Q.")


if __name__ == "__main__":
    main()
