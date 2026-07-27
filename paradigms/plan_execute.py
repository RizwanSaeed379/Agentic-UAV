"""
paradigms/plan_execute.py — Paradigm B: Plan-and-Execute

TWO roles:
  Planner  (qwen2.5:7b) — generates mission plan; replans on anomaly
  Executor (mistral) — validates each step before execution

NAVIGATION ARCHITECTURE (same fixes as ReAct):
  - Segment-by-segment AUTO navigation, single NAV_WAYPOINT per segment
    (DO_JUMP loop-back removed — only needed to hold the plane during slow
     local LLM inference, which is no longer the case)
  - MISSION_CLEAR_ALL → wait ACK → MISSION_COUNT (no re-upload during poll)
  - 200m acceptance radius (80m causes orbiting at 15-20 m/s)
  - Mode drift: re-set AUTO only, never re-upload mid-poll
  - Battery abort at 15% not 25%
  - Loiter via time-based completion (min_wait = turns × 2π × r / speed)
  - NAV_WAYPOINT uses AUTO mode + single NAV_WAYPOINT upload (proven in ReAct)

Route: Home → WP_ALPHA → MIDPOINT (anomaly) → ANOMALY → WP_BRAVO → RTL
"""

from __future__ import annotations
import sys, os, time, logging, json, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.mavlink_state import UAVState, telemetry_paused
from shared.llm_utils import call_ollama, parse_json_response, MODEL_PLANNER, MODEL_EXECUTOR


from shared.tools import (
    HOME_LAT, HOME_LON, WP_ALPHA_LAT, WP_ALPHA_LON,
    WP_BRAVO_LAT, WP_BRAVO_LON, ANOMALY_LAT, ANOMALY_LON,
    MIDPOINT_LAT, MIDPOINT_LON, CRUISE_ALT, GEOFENCE,
    execute_tool, TOOL_REGISTRY,
)
from shared.prompts import (
    plan_execute_planner_prompt, plan_execute_executor_prompt,
)
from shared.logger import log_run
from shared.scenarios import (
    get_scenario, resolve_anomaly, record_arrival, compress, score_run,
)
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE     = os.path.join(_PROJECT_ROOT, "plan_execute_mission_log.txt")
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

def _setup_logger():
    lg = logging.getLogger("plan_execute")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout); sh.setFormatter(_fmt); sh.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8"); fh.setFormatter(_fmt)
    lg.addHandler(sh); lg.addHandler(fh)
    return lg

log = _setup_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TCP_PORTS       = ["tcp:127.0.0.1:5762", "tcp:127.0.0.1:5760"]
_MODE_TO         = 30   # increased from 20 — RTL fires fast after single-item missions
_ARM_TO          = 30
_GPS_MIN_SATS    = 6
_PLANNER_RETRIES = 2
_LOITER_STARTUP  = 8
_NAV_TIMEOUT     = 300
_WP_REACH_DIST   = 200   # 200m acceptance — fixed-wing needs this at 15-20 m/s
_POLL_S          = 2
_STEP_INTERVAL   = 0     # no wait between steps — upload next nav immediately

_VALID_COMMANDS = {
    "NAV_WAYPOINT", "LOITER_TURNS", "RTL", "LAND",
    "CHECK_BATTERY", "CHECK_WEATHER", "CHECK_GEOFENCE", "GET_TELEMETRY",
}

PLANE_MODES = {
    0:"MANUAL",1:"CIRCLE",2:"STABILIZE",3:"TRAINING",4:"ACRO",
    5:"FBWA",6:"FBWB",7:"CRUISE",8:"AUTOTUNE",10:"AUTO",
    11:"RTL",12:"LOITER",13:"TAKEOFF",14:"AVOID_ADSB",15:"GUIDED",16:"INITIALISING",
}
_MODE_REVERSE = {v:k for k,v in PLANE_MODES.items()}

_SAFE_DEFAULT_PLAN = [
    {"command":"NAV_WAYPOINT","params":{"lat":WP_BRAVO_LAT,"lon":WP_BRAVO_LON,"alt":CRUISE_ALT}},
    {"command":"RTL","params":{}},
]

# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def _hav(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    a = (math.sin(math.radians(la2-la1)/2)**2
         + math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo2-lo1)/2)**2)
    return R*2*math.atan2(math.sqrt(a), math.sqrt(1-a))

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def _connect():
    log.info("Auto-detecting ArduPlane SITL ...")
    for port in _TCP_PORTS:
        log.info("  Trying %s ...", port)
        try:
            m = mavutil.mavlink_connection(port, source_system=255,
                                           source_component=0, autoreconnect=True)
            if m.wait_heartbeat(timeout=8):
                vt = {1:"Fixed-wing",2:"Multirotor"}.get(m.mav_type, f"type={m.mav_type}")
                log.info("[OK] Connected: %s | sysid=%d | %s", port, m.target_system, vt)
                return m
            m.close()
        except Exception as e:
            log.info("  %s: %s", port, e)
    log.error("No SITL found."); sys.exit(1)

# ---------------------------------------------------------------------------
# Mode change
# ---------------------------------------------------------------------------
def set_mode(master, mode_name):
    mid = _MODE_REVERSE.get(mode_name)
    if mid is None:
        try: mid = master.mode_mapping().get(mode_name)
        except: pass
    if mid is None: log.error("Unknown mode: %s", mode_name); return False
    log.info("Setting mode -> %s ...", mode_name)
    # Exclusive socket access — telemetry thread otherwise eats the HEARTBEAT.
    with telemetry_paused():
        while master.recv_match(blocking=False): pass
        t0, last = time.time(), 0.0
        while time.time()-t0 < _MODE_TO:
            if time.time()-last >= 2.0:
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid, 0,0,0,0,0)
                last = time.time()
            msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg and msg.custom_mode == mid:
                log.info("[OK] Mode: %s", mode_name); return True
    log.warning("Mode change to %s timed out", mode_name); return False

# ---------------------------------------------------------------------------
# Arm / GPS
# ---------------------------------------------------------------------------
def _arm(master, uav):
    log.info("Arming ...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1,0,0,0,0,0,0)
    t0 = time.time()
    while time.time()-t0 < _ARM_TO:
        if uav.get_state().get("armed"): log.info("[OK] Armed"); return True
        time.sleep(0.5)
    log.warning("Arming timed out"); return False

def _wait_gps(uav):
    log.info("Waiting for GPS lock (%d sats) ...", _GPS_MIN_SATS)
    while True:
        if uav.get_state().get("sat_count", 0) >= _GPS_MIN_SATS:
            log.info("[OK] GPS lock"); return
        time.sleep(2)

# ---------------------------------------------------------------------------
# Mission upload — clear-ACK wait prevents first-attempt timeout
# ---------------------------------------------------------------------------
def _send_items(master, items):
    count = len(items)
    log.info("Uploading %d mission item(s) ...", count)
    for attempt in range(1, 3):
        if attempt == 2: log.info("Retrying..."); time.sleep(3)
        # Exclusive socket access — telemetry thread otherwise consumes the
        # MISSION_REQUEST/MISSION_ACK replies and the upload times out.
        with telemetry_paused():
            # Do NOT send MISSION_CLEAR_ALL here. In flight the vehicle is armed
            # and the mission is running, so AP_Mission::clear() refuses it and
            # replies MISSION_ACK type=1 (MAV_MISSION_ERROR); that stale reject
            # ack then corrupts this upload handshake. MISSION_COUNT already
            # truncates/replaces the existing mission, so the clear is redundant.
            while master.recv_match(blocking=False):   # drain stale/queued msgs
                pass
            master.mav.mission_count_send(master.target_system, master.target_component, count)
            t0 = time.time()
            while time.time()-t0 < 60:
                msg = master.recv_match(
                    type=["MISSION_REQUEST","MISSION_REQUEST_INT","MISSION_ACK"],
                    blocking=True, timeout=5)
                if msg is None:
                    master.mav.mission_count_send(master.target_system, master.target_component, count)
                    continue
                t = msg.get_type()
                if t in ("MISSION_REQUEST","MISSION_REQUEST_INT"):
                    it = items[msg.seq]
                    master.mav.mission_item_int_send(
                        master.target_system, master.target_component,
                        it["seq"], it["frame"], it["command"],
                        it["current"], it["autocontinue"],
                        it["p1"], it["p2"], it["p3"], it["p4"],
                        int(it["x"]*1e7), int(it["y"]*1e7), it["z"])
                elif t == "MISSION_ACK":
                    if msg.type == 0:
                        log.info("[OK] Mission accepted (%d items)", count)
                        return True
                    else:
                        log.warning("Mission ACK error type=%d", msg.type); break
        log.warning("Mission upload timed out (attempt %d/2)", attempt)
    log.error("Mission upload failed"); return False


def _send_items_hold_auto(master, items):
    """
    Upload mission items while keeping the plane in AUTO mode.
    Runs _send_items in a thread and repeatedly sends AUTO commands
    during the upload so the plane doesn't RTL while waiting.
    """
    import threading
    result = [None]

    def upload():
        result[0] = _send_items(master, items)

    t = threading.Thread(target=upload, daemon=True)
    t.start()

    mid = _MODE_REVERSE.get("AUTO", 10)
    while t.is_alive():
        master.mav.command_long_send(
            master.target_system, master.target_component,
            mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
            mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
            mid, 0, 0, 0, 0, 0)
        time.sleep(2)

    t.join()
    return result[0]

# ---------------------------------------------------------------------------
# Segment mission — single NAV_WAYPOINT, distance-polled
# ---------------------------------------------------------------------------
# Fly segment — AUTO + NAV_WAYPOINT upload, distance-polled (proven in ReAct)
# ---------------------------------------------------------------------------
# Rapid AUTO mode — retries every 1s to break the RTL/MANUAL cycle
# ---------------------------------------------------------------------------
def _force_auto(master, attempts=15, interval=1.0):
    """Send AUTO command repeatedly until confirmed. Breaks rapid RTL cycle."""
    mid = _MODE_REVERSE.get("AUTO", 10)
    with telemetry_paused():
        for _ in range(attempts):
            master.mav.command_long_send(
                master.target_system, master.target_component,
                mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
                mid, 0, 0, 0, 0, 0)
            msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=interval)
            if msg and msg.custom_mode == mid:
                log.info("[OK] Mode: AUTO")
                return True
    return False

def _fly_segment(master, uav, lat, lon, alt, label,
                 timeout=_NAV_TIMEOUT, reach=_WP_REACH_DIST):
    """
    Upload a single NAV_WAYPOINT, fly in AUTO, poll distance every 2s.
    On mode drift: send MISSION_SET_CURRENT(0) then call _force_auto()
    which retries AUTO every 1s until confirmed — breaks RTL cycle fast.
    """
    log.info("[NAV] %s -> (%.5f, %.5f, %.0fm)", label, lat, lon, alt)

    s0 = uav.get_state()
    if s0.get("lat", 0) and _hav(s0["lat"], s0["lon"], lat, lon) <= reach:
        log.info("[OK] %s already within reach", label)
        return True

    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        # seq 0 is RESERVED by ArduPilot for HOME — overwritten with the
        # vehicle's home position and never executed. Real mission starts at
        # seq 1, else the target is discarded.
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=1, autocontinue=1,
             p1=0, p2=200, p3=0, p4=0,
             x=lat, y=lon, z=alt),
    ]
    ok = _send_items_hold_auto(master, items)
    if not ok:
        log.warning("[NAV] Upload failed for %s -- attempting anyway", label)

    master.mav.mission_set_current_send(
        master.target_system, master.target_component, 1)
    time.sleep(0.3)
    _force_auto(master)

    t0 = time.time()
    while time.time() - t0 < timeout:
        s    = uav.get_state()
        clat = s.get("lat", 0.0)
        clon = s.get("lon", 0.0)
        bat  = s.get("battery_pct", 100)
        mode = s.get("mode", "")
        if clat == 0.0:
            time.sleep(_POLL_S); continue

        dist = _hav(clat, clon, lat, lon)
        log.info("[NAV] %s -- dist=%.0fm | mode=%s | bat=%d%%",
                 label, dist, mode, bat)

        if dist <= reach:
            log.info("[OK] %s reached (%.0fm)", label, dist)
            return True

        if 0 <= bat < 15:
            log.warning("[NAV] Battery critical -- aborting %s", label)
            return False

        if mode not in ("AUTO", "UNKNOWN", "INITIALISING"):
            log.info("[NAV] Mode=%s -- forcing AUTO", mode)
            master.mav.mission_set_current_send(
                master.target_system, master.target_component, 1)
            time.sleep(0.2)
            _force_auto(master)

        time.sleep(_POLL_S)

    log.warning("[NAV] %s not reached within %ds", label, timeout)
    return False

def _fly_loiter(master, uav, lat, lon, alt, turns, radius):
    """
    Upload single LOITER_TURNS, fly in AUTO, wait by elapsed time.
    Mode-based completion (RTL/MANUAL after turns) with min_wait guard.
    Time-based: turns × circumference / airspeed = min wait.
    """
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        # seq 0 reserved for HOME (see _fly_segment). Real loiter at seq 1.
        dict(seq=0, frame=0, command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame, command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
             current=1, autocontinue=0,   # autocontinue=0 holds the loiter (no DO_JUMP loop)
             p1=turns, p2=0, p3=radius, p4=0,
             x=lat, y=lon, z=alt),
    ]
    log.info("Uploading LOITER_TURNS: (%.5f,%.5f,%.0fm) turns=%d r=%dm",
             lat, lon, alt, turns, radius)
    ok = _send_items(master, items)
    if not ok:
        log.warning("[LOITER] Upload failed — attempting AUTO anyway")
    master.mav.mission_set_current_send(master.target_system,
                                        master.target_component, 1)
    time.sleep(0.5)
    set_mode(master, "AUTO")
    log.info("[CMD] LOITER_TURNS → AUTO turns=%d r=%dm", turns, radius)

    min_wait = turns * (2*math.pi*radius) / 15.0
    log.info("Loiter wait — min=%.0fs | grace=%ds", min_wait, _LOITER_STARTUP)
    time.sleep(_LOITER_STARTUP)

    t0 = time.time()
    while time.time()-t0 < _NAV_TIMEOUT:
        s = uav.get_state()
        bat, mode, wp = s.get("battery_pct",100), s.get("mode",""), s.get("wp_seq",0)
        elapsed = time.time()-t0
        log.info("  [LOITER] elapsed=%.0fs wp=%d mode=%s bat=%d%%",
                 elapsed, wp, mode, bat)
        if 0 <= bat < 15:
            log.warning("[LOITER] Battery critical — aborting"); return False
        if elapsed >= min_wait:
            log.info("[OK] Loiter time complete (%.0fs >= min=%.0fs)", elapsed, min_wait)
            return True
        if mode not in ("AUTO","UNKNOWN","INITIALISING"):
            log.info("[LOITER] Mode=%s — resetting sequencer then AUTO", mode)
            master.mav.mission_set_current_send(master.target_system,
                                                master.target_component, 1)
            time.sleep(0.3)
            set_mode(master, "AUTO")
        time.sleep(3)

    log.warning("[LOITER] Timed out"); return False

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
def call_planner(system_prompt: str, mission_context: str, past_steps: list) -> list:
    completed_summary = (
        "\n".join(f"  {i+1}. {s.get('command','?')} {s.get('params',{})}"
                  for i, s in enumerate(past_steps))
        if past_steps else "  (none — initial plan)"
    )
    # The system_prompt already carries the goal, waypoint coordinates, geofence
    # and cruise altitude (goal-based prompt). We no longer spoon-feed the exact
    # steps — the planner decides the plan that achieves the goal itself.
    prompt = (
        f"{system_prompt}\n\n"
        f"=== SITUATION ===\n{mission_context}\n\n"
        f"=== STEPS ALREADY PLANNED/DONE ===\n{completed_summary}\n\n"
        f"Generate the remaining flight_command steps to ACHIEVE THE GOAL.\n"
        f"NAV_WAYPOINT steps are flown in AUTO mode by Python; use LOITER_TURNS "
        f"only to investigate a disturbance. End by returning home (RTL or LAND).\n"
        f'Output JSON only: {{"type":"mission_plan","steps":[...]}}'
    )
    for attempt in range(1, _PLANNER_RETRIES + 2):
        log.info("PLANNER called (attempt %d/%d) ...", attempt, _PLANNER_RETRIES+1)
        raw    = call_ollama(MODEL_PLANNER, prompt)
        parsed = parse_json_response(raw)
        if parsed.get("type") == "mission_plan":
            steps = parsed.get("steps", [])
            if steps:
                log.info("PLANNER generated %d steps", len(steps))
                for i, s in enumerate(steps):
                    log.info("  Plan[%d]: %s %s", i, s.get("command","?"), s.get("params",{}))
                return steps
        log.warning("PLANNER attempt %d: could not parse", attempt)
        if attempt <= _PLANNER_RETRIES:
            prompt += f'\n\nPrevious output invalid. Output ONLY: {{"type":"mission_plan","steps":[...]}}'
    log.warning("PLANNER failed — using safe default")
    return _SAFE_DEFAULT_PLAN[:]

# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------
# The executor's system prompt now comes from shared/prompts.py
# (plan_execute_executor_prompt) — a validation role, not the old "you CANNOT
# change the command" rubber stamp. Passed into call_executor at call time.

_EXECUTOR_RETRY = (
    "\n\nYour previous response was invalid. Output ONLY a JSON object.\n"
    "The command MUST be the same as the planned step command.\n"
    "Example: {\"type\":\"flight_command\",\"command\":\"NAV_WAYPOINT\","
    "\"params\":{\"lat\":33.712,\"lon\":72.9673,\"alt\":30.0},\"confirmed\":true}"
)

def call_executor(system_prompt: str, step: dict, telemetry: dict) -> dict:
    # Python battery override — safety floor, never trust the LLM for this.
    bat = telemetry.get("battery_pct", 100)
    if 0 <= bat < 15:
        log.warning("[EXECUTOR] Battery critical (%d%%) — RTL override", bat)
        return {"type":"flight_command","command":"RTL","params":{},"abort":True}

    telem_str = "\n".join(f"  {k}: {v}" for k,v in telemetry.items())
    base = (f"{system_prompt}\n\n"
            f"PLANNED STEP:\n{json.dumps(step, indent=2)}\n\n"
            f"TELEMETRY:\n{telem_str}\n\nOutput JSON now.")
    log.info("EXECUTOR validating: %s %s", step.get("command"), step.get("params",{}))
    prompt = base

    for attempt in range(1, 3):
        raw    = call_ollama(MODEL_EXECUTOR, prompt)
        parsed = parse_json_response(raw)
        if parsed.get("type") == "flight_command" and "command" in parsed:
            cmd = parsed.get("command","")
            if cmd not in _VALID_COMMANDS:
                log.warning("EXECUTOR attempt %d: invalid command %r", attempt, cmd)
                prompt = base + _EXECUTOR_RETRY; continue
            if parsed.get("abort"):   log.info("EXECUTOR → RTL (abort)")
            elif parsed.get("modified"): log.info("EXECUTOR → %s (modified)", cmd)
            else: log.info("EXECUTOR confirmed: %s", cmd)
            return parsed
        log.warning("EXECUTOR attempt %d: parse failed", attempt)
        prompt = base + _EXECUTOR_RETRY

    log.warning("EXECUTOR failed — using original step")
    return {**step, "confirmed": True}

# ---------------------------------------------------------------------------
# Execute a validated flight command using segment-by-segment navigation
# ---------------------------------------------------------------------------
TOOL_CMDS = {
    "CHECK_BATTERY":"check_battery","CHECK_WEATHER":"check_weather",
    "CHECK_GEOFENCE":"geofence_check","GET_TELEMETRY":"get_telemetry",
}

def execute_flight_command(master, cmd_dict: dict, uav: UAVState) -> bool:
    command = cmd_dict.get("command","")
    params  = cmd_dict.get("params",{})
    log.info("Executing: %s %s", command, params)

    if command in TOOL_CMDS:
        # BUG FIX: params were discarded here ({} was passed unconditionally),
        # so CHECK_BATTERY/CHECK_GEOFENCE could never receive target coords.
        # Planner emits {} for these steps; execute_tool now falls back to the
        # anomaly location in that case. Return value is unchanged (True) —
        # tool steps remain advisory and do not gate the mission.
        result = execute_tool(TOOL_CMDS[command], params, uav)
        log.info("[TOOL] %s → %s", command, result)
        return True

    if command == "NAV_WAYPOINT":
        lat = float(params.get("lat", HOME_LAT))
        lon = float(params.get("lon", HOME_LON))
        alt = float(params.get("alt", CRUISE_ALT))
        result = _fly_segment(master, uav, lat, lon, alt,
                              label=f"WP({lat:.4f},{lon:.4f})")
        # Hold AUTO immediately after arrival — prevents plane RTLing
        # back toward home before next step mission uploads.
        if result:
            master.mav.mission_set_current_send(
                master.target_system, master.target_component, 1)
            _force_auto(master, attempts=5)
        return result

    elif command == "LOITER_TURNS":
        lat    = float(params.get("lat",    ANOMALY_LAT))
        lon    = float(params.get("lon",    ANOMALY_LON))
        alt    = float(params.get("alt",    CRUISE_ALT))
        turns  = int(params.get("turns",   2))
        radius = int(params.get("radius",  80))
        return _fly_loiter(master, uav, lat, lon, alt, turns, radius)

    elif command == "RTL":
        set_mode(master, "RTL")
        log.info("[CMD] RTL"); return True

    elif command == "LAND":
        set_mode(master, "LAND")
        log.info("[CMD] LAND"); return True

    else:
        log.warning("[CMD] Unknown command %r — skipped", command)
        return False

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run_plan_execute_mission(scenario_id: str = "SC1", run_number: int = 1,
                             anomaly_override=None) -> None:
    _run_start = time.time()
    llm_calls  = 0   # counts planner + executor invocations (paper metric)
    anomaly_response = "NONE"

    # Scenario config drives disturbance + scoring. The ROUTE is the plan the
    # LLM generates — no longer spoon-fed or reverted.
    cfg             = get_scenario(scenario_id)
    anomaly_enabled = resolve_anomaly(cfg, anomaly_override)
    planner_prompt  = plan_execute_planner_prompt(cfg)
    executor_prompt = plan_execute_executor_prompt(cfg)
    step_cap        = cfg["step_cap"]
    score_wps       = cfg["score_waypoints"]
    expected_order  = cfg["expected_order"]
    arrivals        = []
    terminal_cmd    = ""
    timed_out       = False
    safety_note     = ""

    log.info("="*60)
    log.info("PARADIGM B: Plan-and-Execute — Autonomous (LLM plans the route)")
    log.info("Planner : %s", MODEL_PLANNER)
    log.info("Executor: %s", MODEL_EXECUTOR)
    log.info("Scenario: %s — %s", scenario_id, cfg["description"])
    log.info("Goal    : %s", cfg["goal"])
    log.info("Anomaly : %s", "ENABLED" if anomaly_enabled else "DISABLED")
    log.info("Log     : %s", _LOG_FILE)
    log.info("="*60)

    master = _connect()
    from shared.mavlink_state import relocate_home
    relocate_home(master)
    time.sleep(2)

    uav = UAVState(master=master, auto_relocate=False)
    uav.start()
    log.info("[OK] Telemetry thread running")
    time.sleep(1)
    _wait_gps(uav)

    # -----------------------------------------------------------------------
    # STEP 1: Call Planner on the ground — full battery, no time pressure.
    # -----------------------------------------------------------------------
    log.info("="*60)
    log.info("Calling PLANNER before takeoff (ground = free battery) ...")
    log.info("="*60)
    # Situation only — NOT a script. The planner decides the steps that achieve
    # the goal. Python has already scripted takeoff + the climb to WP_ALPHA
    # (below), so the plan should cover the mission from WP_ALPHA onward.
    mission_context = (
        f"The UAV is airborne at cruise altitude and has reached WP_ALPHA "
        f"({WP_ALPHA_LAT},{WP_ALPHA_LON}). Plan the remaining flight to achieve "
        f"the goal from here, and finish by returning home."
    )
    llm_calls += 1
    current_plan = call_planner(planner_prompt, mission_context, past_steps=[])

    # -----------------------------------------------------------------------
    # STEP 2: Executor validates each step on the ground.
    # All LLM calls done before arming — zero battery cost.
    # -----------------------------------------------------------------------
    log.info("="*60)
    log.info("EXECUTOR validating all plan steps on ground ...")
    log.info("="*60)
    validated_plan = []
    for i, step in enumerate(current_plan):
        step_cmd = step.get("command", "")
        if step_cmd in ("RTL", "LAND"):
            validated = {**step, "confirmed": True}
            log.info("  [%d] %s — trivially safe, auto-confirmed", i, step_cmd)
        else:
            # Executor validates against live telemetry. We TRUST its output —
            # the old revert-to-plan forcing (type revert, coord-drift revert,
            # loiter-param revert) is REMOVED so the agent genuinely owns the
            # plan. The only remaining guard is a crash-safety altitude floor,
            # logged as an override, not a silent revert.
            telemetry = uav.get_state()
            llm_calls += 1
            validated = call_executor(executor_prompt, step, telemetry)
            orig_alt = float(step.get("params", {}).get("alt", CRUISE_ALT))
            new_p    = validated.get("params", {})
            new_alt  = float(new_p.get("alt", orig_alt))
            if new_alt <= 5.0 and orig_alt > 5.0:
                log.warning("  [%d] Executor altitude %.0fm unsafe — clamping to %.0fm [SAFETY]",
                            i, new_alt, orig_alt)
                validated = {**validated, "params": {**new_p, "alt": orig_alt}}
            log.info("  [%d] %s -> executor: %s %s", i, step_cmd,
                     validated.get("command"), validated.get("params", {}))
        validated_plan.append(validated)

    # -----------------------------------------------------------------------
    # STEP 3: Build and upload the FULL mission as one ArduPlane mission.
    # Structure:
    #   seq 0: HOME (dummy waypoint, required by ArduPlane)
    #   seq 1: TAKEOFF to CRUISE_ALT
    #   seq 2: NAV_WAYPOINT → WP_ALPHA
    #   seq 3: NAV_WAYPOINT → MIDPOINT
    #   seq 4: LOITER_TURNS → ANOMALY (2 turns, 80m radius)
    #   seq 5: NAV_WAYPOINT → WP_BRAVO
    #   seq 6: RTL
    #
    # With a full multi-item mission, ArduPlane advances through items
    # naturally without ever triggering "mission complete → RTL".
    # -----------------------------------------------------------------------
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    log.info("="*60)
    log.info("Building full mission from validated plan ...")

    full_mission = [
        # seq 0: Home dummy
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        # seq 1: Takeoff
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
             current=1, autocontinue=1,
             p1=15, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=CRUISE_ALT),
        # seq 2: WP_ALPHA
        dict(seq=2, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=200, p3=0, p4=0,
             x=WP_ALPHA_LAT, y=WP_ALPHA_LON, z=CRUISE_ALT),
    ]

    seq = 3
    for step in validated_plan:
        cmd  = step.get("command", "")
        p    = step.get("params", {})
        if cmd == "NAV_WAYPOINT":
            full_mission.append(dict(
                seq=seq, frame=frame,
                command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                current=0, autocontinue=1,
                p1=0, p2=200, p3=0, p4=0,
                x=float(p.get("lat", HOME_LAT)),
                y=float(p.get("lon", HOME_LON)),
                z=float(p.get("alt", CRUISE_ALT))))
            seq += 1
        elif cmd == "LOITER_TURNS":
            full_mission.append(dict(
                seq=seq, frame=frame,
                command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
                current=0, autocontinue=1,
                p1=int(p.get("turns", 2)),
                p2=0, p3=int(p.get("radius", 80)), p4=0,
                x=float(p.get("lat", ANOMALY_LAT)),
                y=float(p.get("lon", ANOMALY_LON)),
                z=float(p.get("alt", CRUISE_ALT))))
            seq += 1
        elif cmd == "RTL":
            full_mission.append(dict(
                seq=seq, frame=frame,
                command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                current=0, autocontinue=1,
                p1=0, p2=0, p3=0, p4=0,
                x=0, y=0, z=0))
            seq += 1
        elif cmd == "LAND":
            full_mission.append(dict(
                seq=seq, frame=frame,
                command=mavutil.mavlink.MAV_CMD_NAV_LAND,
                current=0, autocontinue=1,
                p1=0, p2=0, p3=0, p4=0,
                x=float(p.get("lat", HOME_LAT)),
                y=float(p.get("lon", HOME_LON)),
                z=0))
            seq += 1

    log.info("Full mission: %d items", len(full_mission))
    for item in full_mission:
        log.info("  seq=%d cmd=%d (%.4f,%.4f,%.0fm)",
                 item["seq"], item["command"],
                 item.get("x",0), item.get("y",0), item.get("z",0))

    upload_ok = _send_items(master, full_mission)
    if not upload_ok:
        log.error("[ABORT] Full mission upload failed — not arming.")
        log_run(
            paradigm="PlanExecute",
            model_primary=MODEL_PLANNER,
            model_secondary=MODEL_EXECUTOR,
            scenario_id=scenario_id,
            run_number=run_number,
            outcome="FAILED",
            failure_type="COMMUNICATION",
            waypoints_visited=[],
            anomaly_response="NONE",
            llm_calls=llm_calls,
            duration_seconds=time.time() - _run_start,
            telemetry_final=uav.get_state(),
            notes="mission upload failed before arming",
        )
        uav.stop(); master.close(); return

    log.info("[OK] Full mission uploaded (%d items) — no mid-flight uploads needed", len(full_mission))

    # -----------------------------------------------------------------------
    # STEP 4: Arm and fly. Python only monitors — ArduPlane navigates.
    # -----------------------------------------------------------------------
    set_mode(master, "AUTO")
    _arm(master, uav)
    log.info("[OK] Takeoff started — ArduPlane flying full mission autonomously")

    wp_alpha_ok = False
    midpoint_ok = False
    loiter_ok   = False
    bravo_ok    = False
    anomaly_fired = False
    replan_count  = 0

    # Waypoint sequence numbers in the full mission (used by the anomaly/replan
    # path, which is gated OFF unless the scenario enables the disturbance).
    SEQ_WP_ALPHA  = 2
    SEQ_MIDPOINT  = 3
    SEQ_LOITER    = 4
    SEQ_BRAVO     = 5
    SEQ_RTL       = 6

    # Terminal detection for a VARIABLE-length LLM plan (not the old fixed
    # 4-step mission): the plan's last uploaded item is the terminal.
    terminal_seq  = full_mission[-1]["seq"]
    plan_terminal = next((st["command"] for st in reversed(validated_plan)
                          if st.get("command") in ("RTL", "LAND")), "")

    log.info("="*60)
    log.info("MONITORING flight — ArduPlane handles navigation ...")
    log.info("="*60)

    try:
        t0 = time.time()
        while time.time() - t0 < _NAV_TIMEOUT * 6:
            s    = uav.get_state()
            clat = s.get("lat", 0.0)
            clon = s.get("lon", 0.0)
            bat  = s.get("battery_pct", 100)
            mode = s.get("mode", "")
            wp   = s.get("wp_seq", 0)
            alt  = s.get("alt", 0)

            if clat == 0.0:
                time.sleep(_POLL_S); continue

            # Ground-truth arrival tracking for scoring (observation only)
            record_arrival(clat, clon, score_wps, arrivals)

            # Track waypoint completions by sequence number
            if not wp_alpha_ok and wp > SEQ_WP_ALPHA:
                d = _hav(clat, clon, WP_ALPHA_LAT, WP_ALPHA_LON)
                wp_alpha_ok = True
                log.info("[OK] WP_ALPHA passed (seq=%d dist=%.0fm bat=%d%%)", wp, d, bat)

            if not midpoint_ok and wp > SEQ_MIDPOINT:
                d = _hav(clat, clon, MIDPOINT_LAT, MIDPOINT_LON)
                midpoint_ok = True
                log.info("[OK] MIDPOINT passed (seq=%d dist=%.0fm bat=%d%%)", wp, d, bat)

            # ---------------------------------------------------------------
            # ANOMALY DETECTION + MID-FLIGHT REPLAN
            # When plane reaches the LOITER sequence, trigger anomaly and
            # invoke planner to rewrite the remaining steps mid-flight.
            # This matches the manual: "re-invoke the planner to rewrite
            # the remaining flight plan" when anomaly is detected.
            # ---------------------------------------------------------------
            if anomaly_enabled and not anomaly_fired and wp >= SEQ_LOITER:
                anomaly_fired = True
                uav.trigger_anomaly()
                d = _hav(clat, clon, MIDPOINT_LAT, MIDPOINT_LON)
                log.info("[ANOMALY] Triggered — plane %.0fm from MIDPOINT (seq=%d)", d, wp)

                if replan_count == 0:
                    log.info("="*60)
                    log.info("[REPLAN] Anomaly detected — halting for mid-flight replan ...")
                    log.info("="*60)

                    # Step 1: Hold position safely while planner runs
                    set_mode(master, "LOITER")
                    log.info("[REPLAN] Mode set to LOITER — plane holding position")

                    # Step 2: Build replan context with current telemetry
                    completed = []
                    if wp_alpha_ok: completed.append("WP_ALPHA visited")
                    if midpoint_ok: completed.append("MIDPOINT reached")
                    replan_context = (
                        f"MID-FLIGHT REPLAN. A disturbance was detected at mid-transit. "
                        f"Current position: ({clat:.4f},{clon:.4f}) alt={alt:.0f}m bat={bat}%. "
                        f"Completed so far: {', '.join(completed) if completed else 'none'}. "
                        f"A thermal anomaly is active near ANOMALY({ANOMALY_LAT},{ANOMALY_LON}) "
                        f"and wind has reduced groundspeed ~40%. "
                        f"Re-plan the remaining flight from here to achieve the goal, "
                        f"then return home."
                    )

                    log.info("[REPLAN] Calling Planner with anomaly context ...")
                    # Pass [] for past_steps — completed steps are described
                    # in replan_context string already, not as step dicts.
                    llm_calls += 1
                    new_plan = call_planner(planner_prompt, replan_context, past_steps=[])

                    if new_plan:
                        log.info("[REPLAN] Planner returned %d new steps:", len(new_plan))
                        for j, ns in enumerate(new_plan):
                            log.info("  New[%d]: %s %s", j, ns.get("command"), ns.get("params",{}))

                        # Step 3: Executor validates each new step
                        log.info("[REPLAN] Executor validating new steps ...")
                        validated_new = []
                        for j, step in enumerate(new_plan):
                            step_cmd = step.get("command", "")
                            if step_cmd in ("RTL", "LAND"):
                                vstep = {**step, "confirmed": True}
                                log.info("  [R%d] %s — auto-confirmed", j, step_cmd)
                            else:
                                cur_telem = uav.get_state()
                                llm_calls += 1
                                vstep = call_executor(executor_prompt, step, cur_telem)
                                exec_cmd = vstep.get("command", "")
                                if exec_cmd != step_cmd and exec_cmd != "RTL":
                                    log.warning("  [R%d] Executor changed type %s->%s — reverting", j, step_cmd, exec_cmd)
                                    vstep = {**step, "confirmed": True}
                                orig_p = step.get("params", {})
                                new_p  = vstep.get("params", {})
                                orig_lat2 = float(orig_p.get("lat", 0))
                                orig_lon2 = float(orig_p.get("lon", 0))
                                orig_alt2 = float(orig_p.get("alt", CRUISE_ALT))
                                new_lat2  = float(new_p.get("lat", orig_lat2))
                                new_lon2  = float(new_p.get("lon", orig_lon2))
                                new_alt2  = float(new_p.get("alt", orig_alt2))
                                drift = orig_lat2 and (abs(new_lat2-orig_lat2)>0.002 or abs(new_lon2-orig_lon2)>0.002)
                                adanger = new_alt2 <= 5.0 and orig_alt2 > 5.0
                                lchanged = (step_cmd == "LOITER_TURNS" and new_p != orig_p)
                                if drift or adanger or lchanged:
                                    log.warning("  [R%d] Param rejected — reverting to planner", j)
                                    vstep = {**step, "confirmed": True}
                                else:
                                    log.info("  [R%d] %s confirmed by executor", j, step_cmd)
                            validated_new.append(vstep)

                        # Step 4: Rebuild full mission with completed items + new plan
                        frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
                        new_full = [
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
                                 p1=0, p2=200, p3=0, p4=0,
                                 x=WP_ALPHA_LAT, y=WP_ALPHA_LON, z=CRUISE_ALT),
                            dict(seq=3, frame=frame,
                                 command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                 current=0, autocontinue=1,
                                 p1=0, p2=200, p3=0, p4=0,
                                 x=MIDPOINT_LAT, y=MIDPOINT_LON, z=CRUISE_ALT),
                        ]
                        new_seq = 4
                        for vstep in validated_new:
                            cmd = vstep.get("command", "")
                            p   = vstep.get("params", {})
                            if cmd == "NAV_WAYPOINT":
                                new_full.append(dict(
                                    seq=new_seq, frame=frame,
                                    command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
                                    current=0, autocontinue=1,
                                    p1=0, p2=200, p3=0, p4=0,
                                    x=float(p.get("lat", HOME_LAT)),
                                    y=float(p.get("lon", HOME_LON)),
                                    z=float(p.get("alt", CRUISE_ALT))))
                                new_seq += 1
                            elif cmd == "LOITER_TURNS":
                                new_full.append(dict(
                                    seq=new_seq, frame=frame,
                                    command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
                                    current=0, autocontinue=1,
                                    p1=int(p.get("turns", 2)),
                                    p2=0, p3=int(p.get("radius", 80)), p4=0,
                                    x=float(p.get("lat", ANOMALY_LAT)),
                                    y=float(p.get("lon", ANOMALY_LON)),
                                    z=float(p.get("alt", CRUISE_ALT))))
                                new_seq += 1
                            elif cmd == "RTL":
                                new_full.append(dict(
                                    seq=new_seq, frame=frame,
                                    command=mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH,
                                    current=0, autocontinue=1,
                                    p1=0, p2=0, p3=0, p4=0,
                                    x=0, y=0, z=0))
                                new_seq += 1

                        # Update SEQ constants for new mission
                        SEQ_LOITER = 4
                        # Find WP_BRAVO seq safely
                        SEQ_BRAVO = 5   # default
                        SEQ_RTL   = new_seq - 1
                        for item in new_full:
                            if (item["command"] == mavutil.mavlink.MAV_CMD_NAV_WAYPOINT
                                    and abs(item.get("x", 0) - WP_BRAVO_LAT) < 0.001):
                                SEQ_BRAVO = item["seq"]
                            if item["command"] == mavutil.mavlink.MAV_CMD_NAV_RETURN_TO_LAUNCH:
                                SEQ_RTL = item["seq"]

                        # Step 5: Upload new mission
                        log.info("[REPLAN] Uploading new mission (%d items) ...", len(new_full))
                        replan_ok = _send_items(master, new_full)

                        if replan_ok:
                            replan_count += 1
                            # Resume from LOITER step in the new mission
                            master.mav.mission_set_current_send(
                                master.target_system, master.target_component, SEQ_LOITER)
                            time.sleep(0.5)
                            set_mode(master, "AUTO")
                            log.info("[REPLAN] ✓ New mission uploaded — resuming AUTO from seq=%d", SEQ_LOITER)
                            log.info("[REPLAN] Replans so far: %d", replan_count)
                        else:
                            log.warning("[REPLAN] Upload failed — continuing on original mission")
                            master.mav.mission_set_current_send(
                                master.target_system, master.target_component, 4)
                            time.sleep(0.5)
                            set_mode(master, "AUTO")
                    else:
                        log.warning("[REPLAN] Planner returned empty plan — continuing original")
                        master.mav.mission_set_current_send(
                            master.target_system, master.target_component, 4)
                        time.sleep(0.5)
                        set_mode(master, "AUTO")

            if not loiter_ok and wp > SEQ_LOITER:
                loiter_ok = True
                anomaly_response = "LOITER_TURNS"
                log.info("[OK] LOITER complete (seq=%d bat=%d%%)", wp, bat)

            if not bravo_ok and wp > SEQ_BRAVO:
                d = _hav(clat, clon, WP_BRAVO_LAT, WP_BRAVO_LON)
                bravo_ok = True
                log.info("[OK] WP_BRAVO passed (seq=%d dist=%.0fm bat=%d%%)", wp, d, bat)

            log.info("[MONITOR] seq=%d mode=%s bat=%d%% alt=%.0fm",
                     wp, mode, bat, alt)

            # Battery safety — logged override
            if 0 <= bat < 15:
                log.warning("[SAFETY] Battery critical — forced RTL override")
                set_mode(master, "RTL")
                terminal_cmd = "RTL"; safety_note = "SAFETY_RTL"
                break

            # Mission complete when the plan's terminal item is reached
            if wp >= terminal_seq or mode in ("RTL", "LAND"):
                record_arrival(clat, clon, score_wps, arrivals)
                terminal_cmd = (plan_terminal if plan_terminal in ("RTL", "LAND")
                                else ("LAND" if mode == "LAND" else "RTL"))
                log.info("[OK] Mission complete — terminal=%s reached=%s",
                         terminal_cmd, compress(arrivals))
                break

            time.sleep(_POLL_S)
        else:
            timed_out = True
            log.warning("[TIMEOUT] Monitor loop timed out before completion")

    except KeyboardInterrupt:
        log.warning("Interrupted — RTL")
        set_mode(master, "RTL")

    finally:
        reached = compress(arrivals)
        # Map a monitor-loop timeout onto the shared scorer's TIMEOUT path.
        score_step_n = step_cap + 1 if (timed_out and terminal_cmd == "") else 0
        outcome, failure_type, note = score_run(
            arrivals, expected_order, score_step_n, step_cap, terminal_cmd)
        notes = (f"plan_steps={len(validated_plan)}; replans={replan_count}; "
                 f"terminal={terminal_cmd or 'NONE'}; {note}")
        if safety_note:
            notes += f"; {safety_note}"

        log.info("="*60)
        log.info("MISSION SUMMARY (scenario %s):", scenario_id)
        log.info("  Waypoints reached : %s", reached)
        log.info("  Expected order    : %s", expected_order)
        log.info("  Terminal command  : %s", terminal_cmd or "NONE")
        log.info("  Anomaly enabled   : %s (fired=%s)", anomaly_enabled, anomaly_fired)
        log.info("  LLM calls         : %d", llm_calls)
        log.info("  OUTCOME           : %s (%s)", outcome, failure_type)
        log.info("="*60)

        log_run(
            paradigm="PlanExecute",
            model_primary=MODEL_PLANNER,
            model_secondary=MODEL_EXECUTOR,
            scenario_id=scenario_id,
            run_number=run_number,
            outcome=outcome,
            failure_type=failure_type,
            waypoints_visited=reached,
            anomaly_response=anomaly_response,
            llm_calls=llm_calls,
            duration_seconds=time.time() - _run_start,
            telemetry_final=uav.get_state(),
            notes=notes,
        )

        uav.stop()
        master.close()


if __name__ == "__main__":
    print("=== PARADIGM B: Plan-and-Execute ===")
    print(f"Planner: {MODEL_PLANNER} | Executor: {MODEL_EXECUTOR}")
    input("Press Enter to begin...")
    run_plan_execute_mission()