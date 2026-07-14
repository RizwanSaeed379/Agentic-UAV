"""
paradigms/reflexion_agent.py — Paradigm C: Reflexion (Self-Correction)

TWO roles, different models:
  Actor (qwen2.5:7b) — proposes actions informed by MEMORY of past runs
  Critic (llama3)— evaluates each action and writes a structured reflection

Memory persists in reflexion_memory.txt between runs.
Designed to be run 3 times — the agent improves each attempt.

Mission: Wildfire boundary mapping, Rawalpindi SITL
Route  : Home → WP_ALPHA → MIDPOINT (anomaly triggers) → WP_BRAVO → RTL
"""

from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.mavlink_state import UAVState, telemetry_paused
from shared.llm_utils import call_ollama, parse_json_response, MODEL_PLANNER, MODEL_CRITIC
from shared.tools import (
    HOME_LAT, HOME_LON,
    WP_ALPHA_LAT, WP_ALPHA_LON,
    WP_BRAVO_LAT, WP_BRAVO_LON,
    ANOMALY_LAT, ANOMALY_LON,
    MIDPOINT_LAT, MIDPOINT_LON,
    CRUISE_ALT, GEOFENCE,
    execute_tool, TOOL_REGISTRY,
)
from shared.prompts import REFLEXION_SYSTEM_PROMPT
from shared.logger import log_run
from pymavlink import mavutil
import threading
import time
import logging
import json
import math
import datetime

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MEMORY_FILE = os.path.join(_PROJECT_ROOT, "reflexion_memory.txt")
_LOG_FILE   = os.path.join(_PROJECT_ROOT, "reflexion_mission_log.txt")

# ---------------------------------------------------------------------------
# Logging — console + file
# ---------------------------------------------------------------------------

_fmt = logging.Formatter(
    "%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

def _setup_logger() -> logging.Logger:
    lg = logging.getLogger("reflexion_agent")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_fmt)
    sh.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(_fmt)
    fh.setLevel(logging.DEBUG)
    lg.addHandler(sh)
    lg.addHandler(fh)
    return lg

log = _setup_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_TCP_PORTS       = ["tcp:127.0.0.1:5762", "tcp:127.0.0.1:5760"]
_HEARTBEAT_TO    = 8
_MODE_TO         = 15
_ARM_TO          = 60   # allow EKF/AHRS to re-converge after home relocation
_GPS_MIN_SATS    = 6
_GPS_POLL_S      = 2
_MIDPOINT_RADIUS = 100   # metres — anomaly trigger threshold
_LOOP_INTERVAL   = 6     # seconds between Actor decisions (slowest paradigm)
_MAX_TOOL_CALLS  = 3
_MAX_PARSE_TRIES = 3

PLANE_MODES: dict[int, str] = {
    0:  "MANUAL",      1:  "CIRCLE",    2:  "STABILIZE",  3:  "TRAINING",
    4:  "ACRO",        5:  "FBWA",      6:  "FBWB",        7:  "CRUISE",
    8:  "AUTOTUNE",   10:  "AUTO",     11:  "RTL",         12: "LOITER",
    13: "TAKEOFF",    14:  "AVOID_ADSB", 15: "GUIDED",    16: "INITIALISING",
}
_MODE_REVERSE = {v: k for k, v in PLANE_MODES.items()}

_llm_calls = 0   # total actor + critic LLM calls this run

_SAFE_RTL = {"type": "flight_command", "command": "RTL", "params": {}}

_DEFAULT_CRITIQUE = {
    "outcome":        "FAILURE",
    "failure_reason": "Critic LLM did not respond",
    "lessons":        "Ensure Critic LLM is reachable and prompt is well-formed",
    "severity":       "MEDIUM",
}

# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------

def _haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp     = math.radians(lat2 - lat1)
    dl     = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2.0 * math.asin(math.sqrt(a))


def _haversine(lat1, lon1, lat2, lon2):
    import math
    R = 6_371_000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------

def get_attempt_number() -> int:
    """Count existing ATTEMPT markers and return next attempt index."""
    if not os.path.exists(MEMORY_FILE):
        return 1
    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        return contents.count("=== ATTEMPT") + 1
    except OSError:
        return 1


def read_memory() -> str:
    """
    Read reflexion_memory.txt and return its full contents.
    Prints how many past attempts are recorded.
    """
    if not os.path.exists(MEMORY_FILE):
        msg = "No previous reflections. This is attempt 1."
        log.info("[MEMORY] %s", msg)
        return msg

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            contents = f.read()
        count = contents.count("=== ATTEMPT")
        log.info("[MEMORY] Loaded %d past attempt(s) from %s", count, MEMORY_FILE)
        return contents
    except OSError as exc:
        log.warning("[MEMORY] Could not read memory file: %s", exc)
        return "Memory file unreadable. Treating as attempt 1."


def write_reflection(
    attempt_number: int,
    outcome:        str,
    failure_reason: str,
    lessons:        str,
    commands_issued: list,
) -> None:
    """
    Append one structured reflection block to reflexion_memory.txt.

    Format:
    === ATTEMPT {N} — {timestamp} ===
    Outcome: ...
    Failure reason: ...
    Commands issued: [...]
    Lessons learned: ...
    ==========================================
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    block = (
        f"\n=== ATTEMPT {attempt_number} — {timestamp} ===\n"
        f"Outcome: {outcome}\n"
        f"Failure reason: {failure_reason}\n"
        f"Commands issued: {commands_issued}\n"
        f"Lessons learned: {lessons}\n"
        f"==========================================\n"
    )
    try:
        with open(MEMORY_FILE, "a", encoding="utf-8") as f:
            f.write(block)
        log.info("[MEMORY] Reflection written for attempt %d -> %s", attempt_number, MEMORY_FILE)
    except OSError as exc:
        log.error("[MEMORY] Failed to write reflection: %s", exc)


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _connect() -> mavutil.mavfile:
    log.info("Auto-detecting ArduPlane SITL ...")
    for port in _TCP_PORTS:
        log.info("  Trying %s ...", port)
        try:
            master = mavutil.mavlink_connection(
                port,
                source_system=255,
                source_component=0,
                autoreconnect=True,
            )
            msg = master.wait_heartbeat(timeout=_HEARTBEAT_TO)
            if msg:
                vtype = {1: "Fixed-wing", 2: "Multirotor"}.get(msg.type, f"type={msg.type}")
                log.info("[OK] Connected: %s | sysid=%d | %s",
                         port, master.target_system, vtype)
                return master
            master.close()
            log.info("  No heartbeat on %s", port)
        except Exception as exc:
            log.info("  %s: %s", port, exc)

    log.error(
        "No SITL found on %s\n"
        "  1. Open Mission Planner\n"
        "  2. Simulation -> Plane (wait ~15s)\n"
        "  3. Retry",
        _TCP_PORTS,
    )
    sys.exit(1)


# ---------------------------------------------------------------------------
# Mode change
# ---------------------------------------------------------------------------

def set_mode(master: mavutil.mavfile, mode_name: str) -> None:
    mode_id = _MODE_REVERSE.get(mode_name)
    if mode_id is None:
        try:
            mode_id = master.mode_mapping().get(mode_name)
        except Exception:
            pass
    if mode_id is None:
        log.error("Unknown mode: %s", mode_name)
        return

    log.info("Setting mode -> %s (id=%d) ...", mode_name, mode_id)

    # Exclusive socket access — telemetry thread otherwise eats the HEARTBEAT.
    with telemetry_paused():
        while master.recv_match(blocking=False) is not None:
            pass

        deadline  = time.time() + _MODE_TO
        last_send = 0.0

        while time.time() < deadline:
            if time.time() - last_send >= 2.0:
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                    mode_id, 0, 0, 0, 0, 0,
                )
                last_send = time.time()

            msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg and msg.custom_mode == mode_id:
                log.info("[OK] Mode: %s", mode_name)
                return
            if msg:
                cur = PLANE_MODES.get(msg.custom_mode, f"#{msg.custom_mode}")
                log.debug("  Waiting for %s, currently %s", mode_name, cur)

    log.warning("Mode change to %s timed out after %ds", mode_name, _MODE_TO)


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------

def _arm(master: mavutil.mavfile, uav: UAVState) -> None:
    log.info("Arming ...")
    # NOTE: the arm command must be RE-SENT periodically, not sent once.
    # relocate_home() moves the GPS origin, and the EKF/AHRS then needs
    # ~20-40s to re-converge before pre-arm checks pass ("PreArm: not using
    # AHRS / EKF not ready"). A single arm request fired right after GPS lock
    # is rejected and never retried, so arming times out even though the EKF
    # becomes ready seconds later. Resending every ~3s lets a later request
    # succeed once the EKF has converged.
    deadline  = time.time() + _ARM_TO
    last_send = 0.0
    while time.time() < deadline:
        if uav.get_state().get("armed", False):
            log.info("[OK] Armed")
            return
        if time.time() - last_send >= 3.0:
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM,
                0, 1, 0, 0, 0, 0, 0, 0,
            )
            last_send = time.time()
        time.sleep(0.5)
    log.warning(
        "Arming timed out. In Mission Planner: Full Parameter List → ARSPD_USE=0"
    )


# ---------------------------------------------------------------------------
# GPS wait
# ---------------------------------------------------------------------------

def _wait_for_gps(uav: UAVState) -> None:
    log.info("Waiting for GPS lock (need %d sats) ...", _GPS_MIN_SATS)
    while True:
        sats = uav.get_state().get("sat_count", 0)
        log.debug("  GPS sats: %d", sats)
        if sats >= _GPS_MIN_SATS:
            log.info("[OK] GPS lock — %d satellites", sats)
            return
        time.sleep(_GPS_POLL_S)


# ---------------------------------------------------------------------------
# Mission upload
# ---------------------------------------------------------------------------

def _build_mission_items() -> list[dict]:
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    return [
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
             current=1, autocontinue=1,
             p1=15, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=CRUISE_ALT),
        dict(seq=2, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=20, p3=0, p4=0,
             x=WP_ALPHA_LAT, y=WP_ALPHA_LON, z=CRUISE_ALT),
        dict(seq=3, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=20, p3=0, p4=0,
             x=MIDPOINT_LAT, y=MIDPOINT_LON, z=CRUISE_ALT),
        dict(seq=4, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=20, p3=0, p4=0,
             x=WP_BRAVO_LAT, y=WP_BRAVO_LON, z=CRUISE_ALT),
        dict(seq=5, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=0, y=0, z=0),
    ]


def _send_mission_items(master: mavutil.mavfile, items: list[dict]) -> None:
    count = len(items)
    log.info("Uploading %d mission items ...", count)

    # Exclusive socket access — telemetry thread otherwise consumes the
    # MISSION_REQUEST/MISSION_ACK replies and the upload times out.
    with telemetry_paused():
        # Do NOT send MISSION_CLEAR_ALL here. In flight the vehicle is armed
        # and the mission is running, so AP_Mission::clear() refuses it and
        # replies MISSION_ACK type=1 (MAV_MISSION_ERROR); that stale reject
        # ack then corrupts this upload handshake. MISSION_COUNT already
        # truncates/replaces the existing mission, so the clear is redundant.
        while master.recv_match(blocking=False) is not None:   # drain stale msgs
            pass

        master.mav.mission_count_send(master.target_system, master.target_component, count)

        deadline = time.time() + 30
        while time.time() < deadline:
            msg = master.recv_match(
                type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                blocking=True, timeout=5,
            )
            if msg is None:
                log.debug("  No response — resending MISSION_COUNT")
                master.mav.mission_count_send(
                    master.target_system, master.target_component, count
                )
                continue

            t = msg.get_type()
            if t in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                seq  = msg.seq
                item = items[seq]
                log.debug("  Sending item %d (cmd=%d)", seq, item["command"])
                master.mav.mission_item_int_send(
                    master.target_system, master.target_component,
                    item["seq"], item["frame"], item["command"],
                    item["current"], item["autocontinue"],
                    item["p1"], item["p2"], item["p3"], item["p4"],
                    int(item["x"] * 1e7),
                    int(item["y"] * 1e7),
                    item["z"],
                )
            elif t == "MISSION_ACK":
                if msg.type == 0:
                    log.info("[OK] Mission accepted (%d items)", count)
                else:
                    log.warning("Mission ACK type=%d", msg.type)
                return

    log.warning("Mission upload timed out")


def _upload_loiter_mission(
    master: mavutil.mavfile,
    lat: float, lon: float, alt: float,
    turns: int, radius: int,
) -> None:
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
             current=1, autocontinue=1,
             p1=turns, p2=0, p3=radius, p4=0,
             x=lat, y=lon, z=alt),
        # seq 2 — DO_JUMP back to the loiter so the mission never "completes".
        # Without this, ArduPilot auto-triggers RTL the instant the turns
        # finish (before Python uploads the WP_BRAVO leg), so the plane heads
        # home instead of continuing. Python decides when investigation is done
        # and then overwrites this with the WP_BRAVO mission.
        dict(seq=2, frame=frame,
             command=mavutil.mavlink.MAV_CMD_DO_JUMP,
             current=0, autocontinue=1,
             p1=1, p2=-1, p3=0, p4=0, x=0, y=0, z=0),
    ]
    log.info("Uploading LOITER_TURNS+DO_JUMP: (%.5f, %.5f, %.0fm) turns=%d r=%dm",
             lat, lon, alt, turns, radius)
    _send_mission_items(master, items)


# ---------------------------------------------------------------------------
# Execute flight command
# ---------------------------------------------------------------------------

def execute_flight_command(
    master: mavutil.mavfile,
    command_dict: dict,
    uav: UAVState,
) -> None:
    command = command_dict.get("command", "")
    params  = command_dict.get("params", {})

    log.info("Executing: %s %s", command, params)

    if command == "NAV_WAYPOINT":
        lat = float(params.get("lat", HOME_LAT))
        lon = float(params.get("lon", HOME_LON))
        alt = float(params.get("alt", CRUISE_ALT))
        set_mode(master, "GUIDED")
        master.mav.set_position_target_global_int_send(
            0,
            master.target_system,
            master.target_component,
            mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
            0b0000_1111_1111_1000,
            int(lat * 1e7),
            int(lon * 1e7),
            alt,
            0, 0, 0,
            0, 0, 0,
            0, 0,
        )
        log.info("[CMD] GUIDED → (%.5f, %.5f, %.0fm)", lat, lon, alt)

    elif command == "LOITER_TURNS":
        lat    = float(params.get("lat",    ANOMALY_LAT))
        lon    = float(params.get("lon",    ANOMALY_LON))
        alt    = float(params.get("alt",    CRUISE_ALT))
        turns  = int(params.get("turns",   2))
        radius = int(params.get("radius",  80))
        _upload_loiter_mission(master, lat, lon, alt, turns, radius)
        # Reset the sequencer to seq 1 (the loiter) — a stale current index from
        # the previous mission would otherwise leave AUTO past the end and the
        # plane would loiter in place instead of flying to the anomaly.
        master.mav.mission_set_current_send(
            master.target_system, master.target_component, 1)
        time.sleep(0.3)
        set_mode(master, "AUTO")
        log.info("[CMD] LOITER_TURNS → AUTO at (%.5f, %.5f) turns=%d r=%dm",
                 lat, lon, turns, radius)

    elif command == "RTL":
        set_mode(master, "RTL")
        log.info("[CMD] RTL")

    elif command == "LAND":
        set_mode(master, "LAND")
        log.info("[CMD] LAND")

    else:
        log.warning("[CMD] Unknown command %r — skipped", command)


# ---------------------------------------------------------------------------
# Actor — qwen2.5:7b
# ---------------------------------------------------------------------------

def call_actor(
    telemetry: dict,
    memory:    str,
    history:   list,
    uav_state: dict,
) -> dict:
    """
    Call qwen2.5:7b with REFLEXION_SYSTEM_PROMPT + memory + telemetry.

    Handles tool calls (up to _MAX_TOOL_CALLS rounds) before returning
    a flight command. Falls back to RTL after _MAX_PARSE_TRIES failures.
    """
    global _llm_calls
    telemetry_lines = "\n".join(f"  {k}: {v}" for k, v in telemetry.items())
    history_block   = (
        "\n".join(f"  {h}" for h in history)
        if history
        else "  (no actions taken yet this run)"
    )

    def _build_prompt(extra: str = "") -> str:
        return (
            f"{REFLEXION_SYSTEM_PROMPT}\n\n"
            f"=== MEMORY FROM PAST RUNS ===\n{memory}\n\n"
            f"=== CURRENT TELEMETRY ===\n{telemetry_lines}\n\n"
            f"=== ACTION HISTORY THIS RUN ===\n{history_block}\n\n"
            f"=== YOUR TASK ===\n"
            f"Based on memory and current state, what is your next action?\n"
            f"Output JSON only.{extra}"
        )

    prompt      = _build_prompt()
    parse_fails = 0

    for tool_round in range(_MAX_TOOL_CALLS + 1):
        _llm_calls += 1
        raw    = call_ollama(MODEL_PLANNER, prompt)
        parsed = parse_json_response(raw)

        if not parsed:
            parse_fails += 1
            log.warning("[ACTOR] Parse failure %d/%d", parse_fails, _MAX_PARSE_TRIES)
            if parse_fails >= _MAX_PARSE_TRIES:
                log.warning("[ACTOR] Max parse failures — falling back to RTL")
                return _SAFE_RTL
            prompt += f"\n\nPrevious output was not valid JSON: {raw[:200]}\nOutput JSON only."
            continue

        action_type = parsed.get("type", "")

        if action_type == "flight_command":
            log.info("[ACTOR] Flight command: %s %s",
                     parsed.get("command"), parsed.get("params", {}))
            return parsed

        if action_type == "tool_call":
            if tool_round >= _MAX_TOOL_CALLS:
                log.warning("[ACTOR] Max tool rounds reached — falling back to RTL")
                return _SAFE_RTL

            tool_name = parsed.get("tool_name", "")
            arguments = parsed.get("arguments", {})
            log.info("[ACTOR] Tool call: %s(%s)", tool_name, arguments)

            result = execute_tool(tool_name, arguments, uav_state)
            log.info("[ACTOR] Tool result: %s", result)

            prompt += (
                f"\n\nTOOL RESULT ({tool_name}): {result}\n"
                f"Now output your next action as JSON."
            )
            continue

        # Unrecognised type — nudge and retry
        log.warning("[ACTOR] Unrecognised type %r — nudging", action_type)
        prompt += (
            f"\n\nYour last output had unrecognised type {action_type!r}. "
            "Output JSON with type 'tool_call' or 'flight_command' only."
        )

    log.warning("[ACTOR] All rounds exhausted — falling back to RTL")
    return _SAFE_RTL


# ---------------------------------------------------------------------------
# Critic — qwen2.5:7b
# ---------------------------------------------------------------------------

_CRITIC_SYSTEM = (
    "You are a UAV mission critic. Evaluate this action objectively and return "
    "a structured assessment. Output valid JSON only — no explanation outside JSON."
)


def call_critic(
    action_taken:     dict,
    telemetry_before: dict,
    telemetry_after:  dict,
    anomaly_active:   bool,
) -> dict:
    """
    Call qwen2.5:7b to evaluate one actor decision.

    Returns a critique dict with keys: outcome, failure_reason, lessons, severity.
    Falls back to _DEFAULT_CRITIQUE if parsing fails.
    """
    global _llm_calls

    def _fmt(t: dict) -> str:
        return "  " + "  ".join(f"{k}: {v}\n" for k, v in t.items())

    prompt = (
        f"{_CRITIC_SYSTEM}\n\n"
        f"Action taken:\n  {json.dumps(action_taken)}\n\n"
        f"Telemetry BEFORE action:\n{_fmt(telemetry_before)}\n"
        f"Telemetry AFTER action:\n{_fmt(telemetry_after)}\n"
        f"Anomaly was active: {anomaly_active}\n\n"
        "Assess whether the action improved the mission situation.\n"
        "Consider: altitude change, battery consumption, mode change, "
        "progress toward WP_BRAVO, anomaly response.\n\n"
        "Respond in this exact JSON format:\n"
        "{\n"
        '  "outcome": "SUCCESS" or "PARTIAL" or "FAILURE",\n'
        '  "failure_reason": "what went wrong, or NONE if successful",\n'
        '  "lessons": "specific corrective advice for the next attempt",\n'
        '  "severity": "LOW" or "MEDIUM" or "HIGH"\n'
        "}"
    )

    log.info("[CRITIC] Evaluating action: %s ...", action_taken.get("command"))
    _llm_calls += 1
    raw    = call_ollama(MODEL_CRITIC, prompt)
    parsed = parse_json_response(raw)

    required = {"outcome", "failure_reason", "lessons", "severity"}
    if parsed and required.issubset(parsed.keys()):
        log.info("[CRITIC] outcome=%-8s severity=%-6s lesson=%s",
                 parsed["outcome"], parsed["severity"], parsed["lessons"][:80])
        return parsed

    log.warning("[CRITIC] Could not parse critique — using default")
    return dict(_DEFAULT_CRITIQUE)


# ---------------------------------------------------------------------------
# Scripted navigation helper — no LLM involvement
# ---------------------------------------------------------------------------

def _fly_to(
    master:     mavutil.mavfile,
    uav_state:  UAVState,
    target_lat: float,
    target_lon: float,
    target_alt: float,
    label:      str,
    timeout:    int   = 120,
    threshold:  float = 150.0,
) -> bool:
    """Send a GUIDED waypoint and poll until within threshold metres or timeout.

    Re-sends the position target every 10 s — ArduPlane GUIDED mode requires
    periodic refresh or it may time out and revert to the previous mode.
    """
    log.info("[NAV] Flying to %s (%.5f, %.5f)", label, target_lat, target_lon)

    master.mav.command_long_send(
        master.target_system, master.target_component,
        178, 0, 0, 15.0, -1, 0, 0, 0, 0,
    )
    time.sleep(0.3)

    set_mode(master, "GUIDED")

    def _send_pos_target() -> None:
        master.mav.send(
            mavutil.mavlink.MAVLink_set_position_target_global_int_message(
                0, master.target_system, master.target_component,
                mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT_INT,
                0b0000111111111000,
                int(target_lat * 1e7), int(target_lon * 1e7), target_alt,
                0, 0, 0, 0, 0, 0, 0, 0,
            )
        )

    _send_pos_target()
    last_send = time.time()

    deadline = time.time() + timeout
    while time.time() < deadline:
        if time.time() - last_send >= 10.0:
            _send_pos_target()
            last_send = time.time()

        state = uav_state.get_state()
        cur_lat = state.get("lat", 0.0)
        cur_lon = state.get("lon", 0.0)
        if cur_lat == 0.0 and cur_lon == 0.0:
            time.sleep(2)
            continue
        dist = _haversine(cur_lat, cur_lon, target_lat, target_lon)
        log.info("[NAV] %s — dist: %.0fm | alt: %.0fm | bat: %d%%",
                 label, dist, state.get("alt", 0.0), state.get("battery_pct", -1))
        if dist <= threshold:
            log.info("[NAV] Reached %s (%.0fm)", label, dist)
            return True
        if 0 <= state.get("battery_pct", 100) < 20:
            log.warning("[NAV] Battery critical — aborting navigation to %s", label)
            return False
        time.sleep(5)

    log.warning("[NAV] %s not reached within %ds — continuing", label, timeout)
    return True


def _fly_mission_to(
    master:         mavutil.mavfile,
    uav_state:      UAVState,
    waypoints_list: list,
    label:          str,
    timeout:        int = 180,
) -> bool:
    """Upload a mini NAV_WAYPOINT mission and fly it in AUTO mode.

    Arrival detection uses BOTH wp_seq AND Haversine distance.
    wp_seq alone is unreliable — the previous mission's wp_seq value
    persists in UAVState and can satisfy wp >= final_seq immediately
    before the plane has moved, causing false early returns.
    """
    log.info("[NAV] Uploading mini-mission to %s (%d waypoints)",
             label, len(waypoints_list))

    # Target is the last waypoint in the list
    target_lat, target_lon, _ = waypoints_list[-1]
    reach_dist = 200.0   # metres — arrival threshold

    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        {
            "seq": 0, "frame": 0,
            "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            "current": 0, "autocontinue": 1,
            "p1": 0, "p2": 0, "p3": 0, "p4": 0,
            "x": HOME_LAT, "y": HOME_LON, "z": 0,
        }
    ]

    for i, (lat, lon, alt) in enumerate(waypoints_list):
        items.append({
            "seq": i + 1, "frame": frame,
            "command": mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
            "current": 0, "autocontinue": 1,
            "p1": 0,
            "p2": 80,   # acceptance radius 80m
            "p3": 0, "p4": 0,
            "x": lat, "y": lon, "z": alt,
        })

    _send_mission_items(master, items)
    # ArduPilot does NOT reset the current-waypoint index when a new mission is
    # uploaded in flight — it keeps the previous index (e.g. 2 after the last
    # mini-mission completed). That is past the end of this fresh 2-item mission,
    # so AUTO treats it as already complete and the plane just loiters where it
    # is instead of flying out. Force the sequencer back to seq 1 (first real
    # waypoint; seq 0 is the HOME placeholder).
    master.mav.mission_set_current_send(
        master.target_system, master.target_component, 1)
    time.sleep(0.3)
    set_mode(master, "AUTO")

    final_seq = len(waypoints_list)
    deadline  = time.time() + timeout

    while time.time() < deadline:
        state = uav_state.get_state()
        wp    = state.get("wp_seq", 0)
        bat   = state.get("battery_pct", 100)
        alt   = state.get("alt", 0.0)
        clat  = state.get("lat", 0.0)
        clon  = state.get("lon", 0.0)

        log.info("[NAV] %s — wp_seq=%d/%d | alt=%.0fm | bat=%d%%",
                 label, wp, final_seq, alt, bat)

        # Arrival: wp_seq advanced AND within reach_dist of target
        # Both conditions required — wp_seq alone can be stale from
        # the previous mini-mission and trigger a false early return.
        if clat and wp >= final_seq:
            dist = _haversine(clat, clon, target_lat, target_lon)
            if dist <= reach_dist:
                log.info("[NAV] Reached %s (wp_seq=%d dist=%.0fm)", label, dist, wp)
                return True
            else:
                log.info("[NAV] wp_seq=%d but dist=%.0fm — still flying", wp, dist)

        if 0 <= bat < 20:
            log.warning("[NAV] Battery critical — aborting navigation to %s", label)
            return False

        time.sleep(5)

    log.warning("[NAV] %s timeout after %ds — continuing", label, timeout)
    return True


def run_reflexion_mission(scenario_id: str = "SC1", run_number: int = 1) -> None:
    global _llm_calls
    _llm_calls = 0
    t_start = time.time()
    attempt_number = get_attempt_number()
    log.info("=" * 60)
    log.info("PARADIGM C: Reflexion Agent — Wildfire Boundary Mapping")
    log.info("Actor/Critic: %s / %s", MODEL_PLANNER, MODEL_CRITIC)
    log.info("=== ATTEMPT %d ===", attempt_number)
    log.info("Log    : %s", _LOG_FILE)
    log.info("Memory : %s", MEMORY_FILE)
    log.info("=" * 60)

    memory = read_memory()
    log.info("[MEMORY CONTENTS]\n%s", memory)

    master = _connect()
    from shared.mavlink_state import relocate_home
    relocate_home(master)
    time.sleep(2)
    log.info("Home relocated to Rawalpindi")
    uav = UAVState(master=master, auto_relocate=False)
    uav.start()
    log.info("[OK] Telemetry thread running")
    time.sleep(1.0)
    _wait_for_gps(uav)

    _send_mission_items(master, _build_mission_items())
    time.sleep(1.0)
    set_mode(master, "AUTO")
    _arm(master, uav)

    commands_issued: list[dict] = []
    critiques:       list[dict] = []
    history:         list[str]  = []
    outcome = "PARTIAL — mission incomplete"

    action:      dict = {}
    alpha_ok          = False
    midpoint_ok       = False
    bravo_ok          = False
    interrupted       = False

    try:
        # --- PHASE 1: scripted navigation to WP_ALPHA ---
        log.info("=== PHASE 1: Flying to WP_ALPHA ===")
        alpha_ok = _fly_mission_to(master, uav,
                                   [(WP_ALPHA_LAT, WP_ALPHA_LON, CRUISE_ALT)],
                                   "WP_ALPHA")

        # --- PHASE 2: scripted navigation to MIDPOINT; anomaly pre-triggered ---
        log.info("=== PHASE 2: Flying to MIDPOINT — anomaly pre-triggered ===")
        uav.trigger_anomaly()
        log.info("[ANOMALY] Triggered before MIDPOINT approach")
        midpoint_ok = _fly_mission_to(master, uav,
                                      [(MIDPOINT_LAT, MIDPOINT_LON, CRUISE_ALT)],
                                      "MIDPOINT")

        # --- PHASE 3: ONE LLM decision — how to respond to the anomaly ---
        log.info("=== PHASE 3: LLM DECISION — anomaly response ===")
        telemetry = uav.get_state()
        telemetry["mission_phase"]   = "ANOMALY_DECISION"
        telemetry["memory_summary"]  = (
            f"Attempt {attempt_number}. Past attempts: {attempt_number - 1}"
        )

        log.info("[ACTOR] Calling %s for anomaly decision ...", MODEL_PLANNER)
        action = call_actor(telemetry, memory, history, uav.get_state())
        commands_issued.append(action)
        log.info("[ACTOR] Decision: %s %s",
                 action.get("command"), action.get("params", {}))

        telemetry_before = uav.get_state()
        execute_flight_command(master, action, uav)

        # --- PHASE 4: scripted follow-through based on actor decision ---
        if action.get("command") == "LOITER_TURNS":
            history.append("Actor chose to INVESTIGATE anomaly via LOITER_TURNS")
            # Let the loiter turns actually complete before moving on. The loiter
            # mission loops (DO_JUMP) so the plane holds at the anomaly and never
            # auto-RTLs; we wait the fly-in + turn duration, then upload WP_BRAVO.
            _p      = action.get("params", {})
            _turns  = int(_p.get("turns", 2))
            _radius = int(_p.get("radius", 80))
            invest_s = max(30.0, _turns * (2 * math.pi * _radius) / 15.0)
            log.info("=== PHASE 4a: Investigating anomaly (~%.0fs loiter) then WP_BRAVO ===",
                     invest_s)
            time.sleep(invest_s)
            bravo_ok = _fly_mission_to(master, uav,
                                       [(WP_BRAVO_LAT, WP_BRAVO_LON, CRUISE_ALT)],
                                       "WP_BRAVO")
            outcome = "SUCCESS — anomaly investigated, WP_BRAVO reached"
        elif action.get("command") == "RTL":
            history.append("Actor chose to abort mission via RTL")
            outcome = "PARTIAL — agent aborted at anomaly"
        else:
            history.append(
                f"Actor chose {action.get('command')} — skipped investigation, flying to WP_BRAVO"
            )
            log.info("=== PHASE 4b: Skipped investigation — flying to WP_BRAVO ===")
            bravo_ok = _fly_mission_to(master, uav,
                                       [(WP_BRAVO_LAT, WP_BRAVO_LON, CRUISE_ALT)],
                                       "WP_BRAVO")
            outcome = "PARTIAL — anomaly skipped"

        telemetry_after = uav.get_state()

        # --- Critic evaluates the single key decision ---
        log.info("[CRITIC] Evaluating anomaly decision ...")
        critique = call_critic(action, telemetry_before, telemetry_after, True)
        critiques.append(critique)
        log.info("[CRITIC] outcome=%s severity=%s",
                 critique.get("outcome"), critique.get("severity"))
        log.info("[CRITIC] lesson: %s", critique.get("lessons"))

        # --- PHASE 5: scripted RTL ---
        log.info("=== PHASE 5: RTL ===")
        set_mode(master, "RTL")
        log.info("[CMD] RTL — mission complete")

        write_reflection(
            attempt_number=attempt_number,
            outcome=outcome,
            failure_reason=critique.get("failure_reason", "NONE"),
            lessons=critique.get("lessons", ""),
            commands_issued=[c.get("command", "?") for c in commands_issued],
        )

        log.info("=" * 60)
        log.info("Reflexion attempt %d complete", attempt_number)
        log.info("Outcome: %s", outcome)
        log.info("Reflection written — run again for attempt %d", attempt_number + 1)
        log.info("=" * 60)

    except KeyboardInterrupt:
        interrupted = True
        log.warning("Interrupted — commanding RTL for safety")
        set_mode(master, "RTL")

    finally:
        waypoints = []
        if alpha_ok:    waypoints.append("WP_ALPHA")
        if midpoint_ok: waypoints.append("MIDPOINT")
        if action.get("command") == "LOITER_TURNS": waypoints.append("ANOMALY")
        if bravo_ok:    waypoints.append("WP_BRAVO")

        anomaly_cmd = action.get("command", "")
        if interrupted:                     run_outcome = "ABORTED_RTL"
        elif outcome.startswith("SUCCESS"): run_outcome = "COMPLETED"
        elif anomaly_cmd == "RTL":          run_outcome = "ABORTED_RTL"
        else:                               run_outcome = "FAILED"

        log_run(
            paradigm="Reflexion",
            model_primary=MODEL_PLANNER,
            model_secondary=MODEL_CRITIC,
            scenario_id=scenario_id,
            run_number=run_number,
            outcome=run_outcome,
            failure_type="NONE" if run_outcome == "COMPLETED" else "REASONING",
            waypoints_visited=waypoints,
            anomaly_response=(anomaly_cmd if anomaly_cmd in ("LOITER_TURNS", "RTL") else "NONE"),
            llm_calls=_llm_calls,
            duration_seconds=time.time() - t_start,
            telemetry_final=uav.get_state(),
            notes=f"Attempt {attempt_number} — {outcome}",
        )

        uav.stop()
        master.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== PARADIGM C: Reflexion Agent ===")
    print("Actor: qwen2.5:7b | Critic: qwen2.5:7b")
    print("Mission: Wildfire boundary mapping - Rawalpindi SITL")
    print("Designed to be run 3 times — memory accumulates between runs.")
    attempt = get_attempt_number()
    print(f"This will be attempt #{attempt}")
    if attempt > 1:
        print(f"Loading {attempt - 1} past reflection(s) from memory...")
    print("Ensure Mission Planner SITL is running before starting.")
    input("Press Enter to begin...")
    run_reflexion_mission()