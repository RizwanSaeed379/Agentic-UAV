"""
paradigms/react_agent.py — Paradigm A: ReAct (Reason + Act)

ARCHITECTURE: Continuous while-True reasoning loop per the manual.
  LLM (llama3) is called on EVERY iteration — approximately every 5 seconds.
  Each call receives current telemetry + mission_phase + history and returns
  exactly one action (tool_call or flight_command).

  "The ReAct framework establishes an immediate execution sequence based on
   an alternating cycle: Thought → Action → Observation. LLM call frequency:
   every loop iteration — continuous." — Manual §3.1

NAVIGATION: AUTO mode segment missions (proven in SITL).
  GUIDED mode causes fixed-wing circling — we use segment missions instead.
  Each segment = [HOME, NAV_WAYPOINT]; the plane holds the waypoint when the
  mission completes. Python detects arrival by Haversine distance and uploads
  the next segment. (DO_JUMP loop-back removed — it existed only to hold the
  plane during slow local LLM inference, which is no longer needed.)

PHASES (injected into telemetry so LLM always knows where it is):
  TRANSIT               — flying toward MIDPOINT
  ANOMALY_INVESTIGATION — anomaly active, LLM decides to investigate
  RESUME                — loiter done, fly to WP_BRAVO
  COMPLETE              — mission done

Route: Home(33.7097,72.9673) → WP_ALPHA(33.7120,72.9673)
       → MIDPOINT(33.7120,72.9812) → ANOMALY(33.7134,72.9812)
       → WP_BRAVO(33.7120,72.9950) → RTL
"""

from __future__ import annotations
import sys, os, time, logging, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.mavlink_state import UAVState, telemetry_paused
from shared.llm_utils import MODEL_REACT
from shared.tools import (
    HOME_LAT, HOME_LON, WP_ALPHA_LAT, WP_ALPHA_LON,
    WP_BRAVO_LAT, WP_BRAVO_LON, ANOMALY_LAT, ANOMALY_LON,
    MIDPOINT_LAT, MIDPOINT_LON, CRUISE_ALT,
)
from shared.prompts import react_system_prompt
from shared.agent_loop import agent_step
from shared.logger import log_run
from shared.scenarios import (
    get_scenario, resolve_anomaly, record_arrival, compress, score_run,
    score_open_ended_run, NAMED_WAYPOINTS,
)
from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_FILE = os.path.join(_PROJECT_ROOT, "react_mission_log.txt")
_fmt = logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")

def _setup_logger():
    lg = logging.getLogger("react_agent")
    if lg.handlers:
        return lg
    lg.setLevel(logging.DEBUG)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(_fmt); sh.setLevel(logging.INFO)
    fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    fh.setFormatter(_fmt)
    lg.addHandler(sh); lg.addHandler(fh)
    return lg

log = _setup_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_TCP_PORTS     = ["tcp:127.0.0.1:5762", "tcp:127.0.0.1:5760"]
_MODE_TO       = 20
_ARM_TO        = 30
_GPS_MIN_SATS  = 6
_NAV_TIMEOUT   = 300
_WP_REACH_DIST = 200
_RTL_ENGAGE_TIMEOUT  = 420   # ceiling waiting for RTL/LAND to actually engage
_RTL_CONFIRM_TIMEOUT = 240   # separate ceiling for actually arriving home once engaged
_HOME_HOLD_S         = 10    # must sit near home this long (not just transit through)
_INSPECT_MIN_DIST_M  = 300   # open-ended only: minimum distance from HOME (and
                             # from any already-recorded spot) to count as a
                             # new distinct inspected location
_POLL_S        = 2
_LOOP_INTERVAL = 5   # seconds between ReAct iterations — LLM called every iteration

PLANE_MODES = {
    0:"MANUAL", 1:"CIRCLE",  2:"STABILIZE",  3:"TRAINING", 4:"ACRO",
    5:"FBWA",   6:"FBWB",    7:"CRUISE",      8:"AUTOTUNE", 10:"AUTO",
    11:"RTL",   12:"LOITER", 13:"TAKEOFF",   14:"AVOID_ADSB",
    15:"GUIDED",16:"INITIALISING",
}
_MODE_REVERSE = {v: k for k, v in PLANE_MODES.items()}

# ---------------------------------------------------------------------------
# Haversine
# ---------------------------------------------------------------------------
def _hav(la1, lo1, la2, lo2):
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    a = (math.sin(math.radians(la2 - la1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _agl(s):
    """Approximate altitude AGL. SITL reports ASL; Rawalpindi ground is ~584m."""
    alt = s.get("alt", 0) or 0
    return alt - 584 if alt > 100 else alt

# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------
def _connect():
    log.info("Auto-detecting ArduPlane SITL ...")
    for port in _TCP_PORTS:
        log.info("  Trying %s ...", port)
        try:
            m = mavutil.mavlink_connection(
                port, source_system=255, source_component=0, autoreconnect=True)
            if m.wait_heartbeat(timeout=8):
                log.info("[OK] Connected: %s | sysid=%d", port, m.target_system)
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
    if mid is None:
        log.error("Unknown mode: %s", mode_name); return False
    log.info("Setting mode -> %s ...", mode_name)
    # Take exclusive socket access — the telemetry thread otherwise eats the
    # confirming HEARTBEAT and this loop times out.
    with telemetry_paused():
        while master.recv_match(blocking=False): pass
        t0, last = time.time(), 0.0
        while time.time() - t0 < _MODE_TO:
            if time.time() - last >= 2.0:
                master.mav.command_long_send(
                    master.target_system, master.target_component,
                    mavutil.mavlink.MAV_CMD_DO_SET_MODE, 0,
                    mavutil.mavlink.MAV_MODE_FLAG_CUSTOM_MODE_ENABLED, mid, 0, 0, 0, 0, 0)
                last = time.time()
            msg = master.recv_match(type="HEARTBEAT", blocking=True, timeout=1)
            if msg and msg.custom_mode == mid:
                log.info("[OK] Mode: %s", mode_name); return True
    log.warning("Mode change to %s timed out", mode_name); return False

# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------
def _arm(master, uav):
    log.info("Arming ...")
    master.mav.command_long_send(
        master.target_system, master.target_component,
        mavutil.mavlink.MAV_CMD_COMPONENT_ARM_DISARM, 0, 1, 0, 0, 0, 0, 0, 0)
    t0 = time.time()
    while time.time() - t0 < _ARM_TO:
        if uav.get_state().get("armed"):
            log.info("[OK] Armed"); return True
        time.sleep(0.5)
    log.warning("Arming timed out"); return False

# ---------------------------------------------------------------------------
# GPS wait
# ---------------------------------------------------------------------------
def _wait_gps(uav):
    log.info("Waiting for GPS lock (%d sats) ...", _GPS_MIN_SATS)
    while True:
        if uav.get_state().get("sat_count", 0) >= _GPS_MIN_SATS:
            log.info("[OK] GPS lock"); return
        time.sleep(2)

# ---------------------------------------------------------------------------
# Mission upload
# ---------------------------------------------------------------------------
def _send_items(master, items):
    count = len(items)
    log.info("Uploading %d mission item(s) ...", count)
    for attempt in range(1, 3):
        if attempt == 2: log.info("Retrying..."); time.sleep(3)
        # Exclusive socket access for the whole count→items→ack handshake.
        # Without this the telemetry thread consumes MISSION_REQUEST/MISSION_ACK
        # and the upload times out.
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
            while time.time() - t0 < 60:
                msg = master.recv_match(
                    type=["MISSION_REQUEST", "MISSION_REQUEST_INT", "MISSION_ACK"],
                    blocking=True, timeout=5)
                if msg is None:
                    master.mav.mission_count_send(
                        master.target_system, master.target_component, count)
                    continue
                t = msg.get_type()
                if t in ("MISSION_REQUEST", "MISSION_REQUEST_INT"):
                    it = items[msg.seq]
                    master.mav.mission_item_int_send(
                        master.target_system, master.target_component,
                        it["seq"], it["frame"], it["command"],
                        it["current"], it["autocontinue"],
                        it["p1"], it["p2"], it["p3"], it["p4"],
                        int(it["x"] * 1e7), int(it["y"] * 1e7), it["z"])
                elif t == "MISSION_ACK":
                    if msg.type == 0:
                        log.info("[OK] Mission accepted (%d items)", count)
                        return True
                    else:
                        log.warning("Mission ACK error type=%d", msg.type); break
        log.warning("Mission upload timed out (attempt %d/2)", attempt)
    log.error("Mission upload failed"); return False

# ---------------------------------------------------------------------------
# Segment mission — single NAV_WAYPOINT
# On mission complete the plane holds the waypoint; Python detects arrival by
# distance and overwrites with the next segment.
# ---------------------------------------------------------------------------
def _build_segment(lat, lon, alt):
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    return [
        # seq 0 is RESERVED by ArduPilot for HOME — it is overwritten with the
        # vehicle's home position and never executed. The real mission must
        # start at seq 1, else the target waypoint is discarded.
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

def _fly_segment(master, uav, lat, lon, alt, label, timeout=_NAV_TIMEOUT, on_poll=None):
    log.info("[NAV] %s → (%.5f, %.5f, %.0fm)", label, lat, lon, alt)
    ok = _send_items(master, _build_segment(lat, lon, alt))
    if not ok:
        log.warning("[NAV] Upload failed for %s — attempting anyway", label)
    set_mode(master, "AUTO")
    t0 = time.time()
    while time.time() - t0 < timeout:
        s    = uav.get_state()
        clat = s.get("lat", 0.0)
        clon = s.get("lon", 0.0)
        bat  = s.get("battery_pct", 100)
        mode = s.get("mode", "")
        if clat == 0.0: time.sleep(_POLL_S); continue
        if on_poll:
            on_poll(s)
        dist = _hav(clat, clon, lat, lon)
        log.info("[NAV] %s — dist=%.0fm | mode=%s | bat=%d%%", label, dist, mode, bat)
        if dist <= _WP_REACH_DIST:
            log.info("[OK] %s reached (%.0fm)", label, dist)
            return True
        if 0 <= bat < 15:
            log.warning("[NAV] Battery critical — aborting %s", label)
            return False
        if mode not in ("AUTO", "UNKNOWN", "INITIALISING"):
            log.info("[NAV] Mode=%s — re-setting AUTO", mode)
            set_mode(master, "AUTO")
        time.sleep(_POLL_S)
    log.warning("[NAV] %s not reached within %ds", label, timeout)
    return False

# ---------------------------------------------------------------------------
# Loiter mission — single LOITER_TURNS
# autocontinue=0 holds the loiter after the turns finish (no mission-complete
# transition), replacing the old DO_JUMP loop. Completion detected by elapsed
# time.
# ---------------------------------------------------------------------------
def _upload_loiter(master, lat, lon, alt, turns, radius):
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    items = [
        # seq 0 reserved for HOME (see _build_segment). Real loiter at seq 1.
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1,
             p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_LOITER_TURNS,
             current=1, autocontinue=0,
             p1=turns, p2=0, p3=radius, p4=0,
             x=lat, y=lon, z=alt),
    ]
    log.info("Uploading LOITER_TURNS: (%.5f,%.5f,%.0fm) turns=%d r=%dm",
             lat, lon, alt, turns, radius)
    _send_items(master, items)

def _wait_loiter(uav, turns, timeout=_NAV_TIMEOUT):
    min_wait = turns * (2 * math.pi * 80) / 15.0
    log.info("Loiter wait — turns=%d min=%.0fs", turns, min_wait)
    time.sleep(8)   # grace period for ArduPlane to start loitering
    t0 = time.time()
    while time.time() - t0 < timeout:
        s = uav.get_state()
        bat = s.get("battery_pct", 100)
        elapsed = time.time() - t0
        log.info("  [LOITER] elapsed=%.0fs mode=%s bat=%d%%",
                 elapsed, s.get("mode", ""), bat)
        if 0 <= bat < 15:
            log.warning("[LOITER] Battery critical — aborting"); return False
        if elapsed >= min_wait:
            log.info("[OK] Loiter time complete (%.0fs >= min=%.0fs)", elapsed, min_wait)
            return True
        time.sleep(3)
    log.warning("[LOITER] Timed out"); return False

# ---------------------------------------------------------------------------
# Post-mission return confirmation — the agent DECIDING to RTL/LAND is not the
# same as the plane actually getting home. execute_flight_command's RTL/LAND
# branches just call set_mode() and return immediately, so without this the
# script declares the mission over the instant the LLM picks that command,
# regardless of where the plane physically is.
# ---------------------------------------------------------------------------
def _confirm_return_home(uav, engage_timeout=_RTL_ENGAGE_TIMEOUT,
                         arrival_timeout=_RTL_CONFIRM_TIMEOUT, reach=_WP_REACH_DIST):
    """
    Two independently timed phases:
      1. Wait for RTL/LAND to actually engage (or the vehicle to disarm).
      2. Once engaged, track distance-to-home on its own fresh clock.
    Confirms return when EITHER the vehicle disarms (landed), or RTL/LAND has
    engaged AND the vehicle is within `reach` of HOME, held there for
    _HOME_HOLD_S so a brief pass-through doesn't count.
    Returns (confirmed: bool, note: str).
    """
    log.info("=" * 60)
    log.info("[RETURN] Confirming plane actually returns home (post-mission) ...")
    log.info("=" * 60)

    # ---- Phase 1: wait for RTL/LAND to actually engage --------------------
    t0 = time.time()
    while True:
        s     = uav.get_state()
        armed = s.get("armed", False)
        clat  = s.get("lat", 0.0)
        mode  = s.get("mode", "")

        if not armed and clat:
            log.info("[RETURN] Disarmed — landed. Confirmed home.")
            return True, "disarmed"

        if mode in ("RTL", "LAND"):
            log.info("[RETURN] %s engaged — now tracking distance to home", mode)
            break

        if time.time() - t0 >= engage_timeout:
            log.warning("[RETURN] RTL/LAND never engaged within %ds (mode=%s)",
                        engage_timeout, mode)
            return False, f"RTL/LAND never engaged within {engage_timeout}s (last mode={mode})"

        log.info("[RETURN] waiting for RTL/LAND to engage (mode=%s) ...", mode)
        time.sleep(_POLL_S)

    # ---- Phase 2: track distance-to-home on its own fresh clock ----------
    t1 = time.time()
    home_hold_start = None

    while time.time() - t1 < arrival_timeout:
        s     = uav.get_state()
        armed = s.get("armed", False)
        clat  = s.get("lat", 0.0)
        clon  = s.get("lon", 0.0)
        bat   = s.get("battery_pct", 100)
        mode  = s.get("mode", "")

        if not armed and clat:
            log.info("[RETURN] Disarmed — landed. Confirmed home.")
            return True, "disarmed"

        if clat:
            d_home = _hav(clat, clon, HOME_LAT, HOME_LON)
            log.info("[RETURN] dist_home=%.0fm mode=%s armed=%s bat=%d%%",
                      d_home, mode, armed, bat)

            if d_home <= reach:
                if home_hold_start is None:
                    home_hold_start = time.time()
                elif time.time() - home_hold_start >= _HOME_HOLD_S:
                    log.info("[RETURN] Held near home %ds — confirmed.", _HOME_HOLD_S)
                    return True, "reached_home_and_held"
            else:
                home_hold_start = None

        if 0 <= bat < 5:
            log.warning("[RETURN] Battery critical during return leg — unresolved")
            return False, f"battery critical at {bat}% before confirmed return"

        time.sleep(_POLL_S)

    log.warning("[RETURN] Timed out (%ds) waiting for confirmed arrival home", arrival_timeout)
    return False, "timed out waiting for confirmed arrival at home (RTL/LAND had engaged)"

# ---------------------------------------------------------------------------
# Execute flight command — called by the ReAct loop after each LLM decision
# ---------------------------------------------------------------------------
def execute_flight_command(master, uav, command_dict, on_poll=None):
    command = command_dict.get("command", "")
    params  = command_dict.get("params", {})
    log.info("Executing: %s %s", command, params)

    if command == "NAV_WAYPOINT":
        lat = float(params.get("lat", HOME_LAT))
        lon = float(params.get("lon", HOME_LON))
        alt = float(params.get("alt", CRUISE_ALT))
        _fly_segment(master, uav, lat, lon, alt, f"WP({lat:.4f},{lon:.4f})", on_poll=on_poll)

    elif command == "LOITER_TURNS":
        lat    = float(params.get("lat",    ANOMALY_LAT))
        lon    = float(params.get("lon",    ANOMALY_LON))
        alt    = float(params.get("alt",    CRUISE_ALT))
        turns  = int(params.get("turns",   2))
        radius = int(params.get("radius",  80))
        # Fly to anomaly site first if not already close
        s = uav.get_state()
        if s.get("lat") and _hav(s["lat"], s["lon"], lat, lon) > _WP_REACH_DIST:
            log.info("[CMD] Flying to anomaly site first ...")
            _fly_segment(master, uav, lat, lon, alt, "ANOMALY_SITE", on_poll=on_poll)
        _upload_loiter(master, lat, lon, alt, turns, radius)
        set_mode(master, "AUTO")
        log.info("[CMD] LOITER_TURNS → AUTO turns=%d r=%dm", turns, radius)
        _wait_loiter(uav, turns)

    elif command == "RTL":
        set_mode(master, "RTL")
        log.info("[CMD] RTL")

    elif command == "LAND":
        set_mode(master, "LAND")
        log.info("[CMD] LAND")

    else:
        log.warning("[CMD] Unknown command %r — no action taken", command)

# ---------------------------------------------------------------------------
# Main ReAct mission — continuous reasoning loop
# ---------------------------------------------------------------------------
def run_react_mission(scenario_id: str = "SC1", run_number: int = 1,
                      anomaly_override=None):
    _run_start = time.time()

    # Scenario config drives the disturbance, step cap, and scoring. The ROUTE is
    # owned by the LLM (goal-based prompt); nothing here flies waypoints for it.
    cfg             = get_scenario(scenario_id)
    anomaly_enabled = resolve_anomaly(cfg, anomaly_override)
    open_ended      = cfg.get("open_ended", False)
    system_prompt   = react_system_prompt(cfg)
    step_cap        = cfg["step_cap"]
    score_wps       = cfg["score_waypoints"]
    expected_order  = cfg["expected_order"]

    log.info("=" * 60)
    log.info("PARADIGM A: ReAct Agent — Autonomous (LLM owns the route)")
    log.info("Model    : %s", MODEL_REACT)
    log.info("Scenario : %s — %s", scenario_id, cfg["description"])
    log.info("Goal     : %s", cfg["goal"])
    log.info("Anomaly  : %s", "ENABLED" if anomaly_enabled else "DISABLED")
    log.info("Step cap : %d", step_cap)
    log.info("Log      : %s", _LOG_FILE)
    log.info("=" * 60)

    master = _connect()
    from shared.mavlink_state import relocate_home
    relocate_home(master)
    time.sleep(2)

    uav = UAVState(master=master, auto_relocate=False)
    uav.start()
    log.info("[OK] Telemetry thread running")
    time.sleep(1)
    _wait_gps(uav)

    # Upload takeoff-ONLY mission — HOME + TAKEOFF to cruise altitude.
    # Getting airborne is vehicle bring-up, not a navigation decision, so it stays
    # scripted. The scripted WP_ALPHA leg was REMOVED: the LLM now owns the route,
    # so it must decide to fly to WP_ALPHA itself once airborne.
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    log.info("Uploading takeoff-only mission (HOME + TAKEOFF to %dm) ...", CRUISE_ALT)
    _send_items(master, [
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1, p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
             current=1, autocontinue=1, p1=15, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=CRUISE_ALT),
    ])
    set_mode(master, "AUTO")
    armed_ok = _arm(master, uav)
    if not armed_ok:
        log.error("[ABORT] Arming failed — cannot proceed with mission")
        uav.stop()
        master.close()
        return
    log.info("[OK] Takeoff started — waiting for airborne, then LLM owns the route")

    # Wait until airborne at ~cruise altitude (vehicle readiness, NOT a route
    # decision). Then LOITER-hold so the plane circles at altitude while the LLM
    # decides its first destination.
    log.info("=" * 60)
    log.info("Waiting for airborne (AGL >= %.0fm) ...", 0.8 * CRUISE_ALT)
    log.info("=" * 60)
    airborne = False
    t0 = time.time()
    while time.time() - t0 < _NAV_TIMEOUT:
        s     = uav.get_state()
        agl   = _agl(s)
        bat   = s.get("battery_pct", 100)
        armed = s.get("armed", False)
        log.info("[TAKEOFF] AGL~%.0fm mode=%s armed=%s bat=%d%%", agl, s.get("mode", ""), armed, bat)
        if armed and agl >= 0.8 * CRUISE_ALT:
            log.info("[OK] Airborne at %.0fm — handing navigation to the LLM", agl)
            airborne = True
            break
        if not armed and time.time() - t0 > 15:
            log.error("[ABORT] Not armed %.0fs after takeoff attempt — vehicle "
                     "is not actually flying (mode=%s). Aborting.", time.time() - t0, s.get("mode", ""))
            uav.stop()
            master.close()
            return
        if 0 <= bat < 15:
            log.warning("[TAKEOFF] Battery critical — RTL")
            set_mode(master, "RTL"); return
        time.sleep(_POLL_S)

    # Hold at cruise altitude while the LLM chooses the first waypoint.
    set_mode(master, "LOITER")

    # -----------------------------------------------------------------------
    # Autonomous ReAct loop — the LLM owns the route.
    # Each step: read telemetry -> record arrivals (observation) -> safety floor
    # -> inject progress -> call LLM -> execute -> re-check arrival -> repeat.
    # The scripted phase machine and the deterministic RESUME leg were REMOVED
    # so the agent, not Python, decides the sequence and when to finish.
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Starting autonomous ReAct loop — LLM decides every action")
    log.info("=" * 60)

    history          = ["Airborne over HOME at cruise altitude. Mission not started."]
    arrivals         = []      # ordered ground-truth waypoint arrivals (scoring)
    inspected_locations = []   # open-ended only: distinct free-form locations
                               # actually reached (arrivals/visited can't work
                               # here since score_waypoints is deliberately [])
    step_n           = 0
    anomaly_fired    = False
    terminal_cmd     = ""
    used_fallback    = False
    safety_note      = ""
    llm_calls        = 0
    anomaly_response = "NONE"
    return_confirmed = False
    return_note      = "not evaluated"
    max_dist_from_home = 0.0
    reasoning_log = []   # captures the model's stated "reasoning" field per
                         # step, when present — needed for scenarios like
                         # SC2.2 that require recording what the agent said
                         # about competing objectives

    def _track_home_dist(state):
        # Called on every internal polling tick during a flight (not just
        # before/after each step) so a mid-flight peak displacement — e.g.
        # right before an unexpected RTL turns the plane back — isn't missed.
        nonlocal max_dist_from_home
        clat_, clon_ = state.get("lat", 0.0), state.get("lon", 0.0)
        if clat_:
            d = _hav(clat_, clon_, HOME_LAT, HOME_LON)
            if d > max_dist_from_home:
                max_dist_from_home = d

    try:
        while True:
            step_n += 1
            if step_n > step_cap:
                log.warning("[TIMEOUT] Step cap %d exceeded", step_cap)
                break

            # a. Telemetry + ground-truth arrival tracking (observation only)
            s    = uav.get_state()
            clat = s.get("lat", 0.0)
            clon = s.get("lon", 0.0)
            bat  = s.get("battery_pct", 100)
            mode = s.get("mode", "")
            record_arrival(clat, clon, score_wps, arrivals)

            # b. Safety floor — logged override, NOT a reasoning action
            if 0 <= bat < 15:
                log.warning("[SAFETY] Battery critical (%d%%) — forced RTL override", bat)
                set_mode(master, "RTL")
                terminal_cmd = "RTL"; safety_note = "SAFETY_RTL"
                break

            # c. Disturbance injection — scenario-gated (OFF for SC1)
            if anomaly_enabled and not anomaly_fired and clat:
                if _hav(clat, clon, MIDPOINT_LAT, MIDPOINT_LON) <= 400 or clon > 72.975:
                    log.info("[ANOMALY] Injecting disturbance (scenario %s)", scenario_id)
                    uav.trigger_anomaly()
                    anomaly_fired = True
                    history.append(
                        "Disturbance active: thermal anomaly at "
                        f"ANOMALY({ANOMALY_LAT},{ANOMALY_LON}) + wind -40% groundspeed.")

            # d. Progress fields so the LLM can sequence the route itself
            visited = compress(arrivals)
            remaining_wps = [
                {"name": w, "lat": NAMED_WAYPOINTS[w][0], "lon": NAMED_WAYPOINTS[w][1]}
                for w in expected_order if w not in visited]
            remaining_names = [w["name"] for w in remaining_wps]

            # Explicit directive, recomputed from ground truth every single step
            # (not dependent on the rolling history window surviving long enough
            # to still contain it). This is intentionally a strong instruction,
            # not a hardcoded action — the LLM can still choose otherwise, but it
            # can no longer plausibly claim it didn't know the state.
            #
            # IMPORTANT: for an open-ended scenario, remaining_names is empty
            # from step 1 because no waypoints were ever defined — NOT because
            # everything is done. That must not be read as "mission complete,
            # RTL now" the way it is for the fixed-route scenarios.
            dist_home = round(_hav(clat, clon, HOME_LAT, HOME_LON), 1) if clat else 0.0
            max_dist_from_home = max(max_dist_from_home, dist_home)

            if open_ended:
                if inspected_locations:
                    progress_text = (
                        f"{len(inspected_locations)} distinct location(s) inspected so far: "
                        f"{inspected_locations}."
                    )
                else:
                    progress_text = "No location away from HOME has been reached yet."
                directive = (
                    f"This is an OPEN-ENDED inspection goal — no predefined "
                    f"waypoints exist. Choose where within the geofence to "
                    f"inspect, then RTL/LAND yourself once you judge the "
                    f"inspection complete. {progress_text}"
                )
                if mode == "RTL":
                    directive += (
                        " NOTE: current mode is RTL. If you did not just command "
                        "RTL yourself, the vehicle's own failsafe engaged it "
                        "unexpectedly — your last NAV_WAYPOINT may not have "
                        "completed as intended."
                    )
            elif remaining_names:
                directive = (
                    f"Already visited (DO NOT fly back to these): {visited or 'none yet'}. "
                    f"Still required, in order: {remaining_names}. "
                    f"Fly to {remaining_names[0]} next."
                )
            else:
                directive = (
                    f"All required waypoints are visited ({visited}). The mission "
                    f"objective is COMPLETE. Do NOT navigate to any waypoint again "
                    f"(including WP_ALPHA). Command RTL now, then LAND once near home."
                )

            telemetry = {
                **s,
                "mission_goal":        cfg["goal"],
                "waypoints_visited":   visited,
                "waypoints_remaining": remaining_wps,
                "directive":           directive,
                "steps_used":          step_n,
                "steps_remaining":     step_cap - step_n,
                "anomaly_active":      anomaly_fired,
                "dist_to_home_m":      dist_home,
            }
            if not open_ended:
                # Don't even hint that Alpha/Bravo exist for an open-ended run.
                telemetry["dist_to_wp_alpha_m"] = round(_hav(clat, clon, WP_ALPHA_LAT, WP_ALPHA_LON), 1) if clat else -1
                telemetry["dist_to_wp_bravo_m"] = round(_hav(clat, clon, WP_BRAVO_LAT, WP_BRAVO_LON), 1) if clat else -1
            if anomaly_enabled:
                telemetry["dist_to_anomaly_m"] = (
                    round(_hav(clat, clon, ANOMALY_LAT, ANOMALY_LON), 1) if clat else -1)

            # Put the same directive at the END of history too — it's the last
            # thing the model reads before the next call, regardless of how the
            # telemetry dict happens to get rendered into the prompt template.
            history.append(f"OBSERVATION (step {step_n}): {directive}")

            log.info("--- Step %d/%d | visited=%s | mode=%s | bat=%d%% ---",
                     step_n, step_cap, visited, mode, bat)

            # e. Call the LLM — it chooses the next action (route is its decision)
            llm_calls += 1
            cmd = agent_step(
                model=MODEL_REACT,
                system_prompt=system_prompt,
                telemetry=telemetry,
                history=history[-6:],
                uav=uav,
                step_label=f"REACT_STEP_{step_n}",
            )
            command = cmd.get("command", "?")
            if cmd.get("fallback"):
                used_fallback = True
            log.info("[LLM] Decision: %s %s%s", command, cmd.get("params", {}),
                     " [FALLBACK]" if cmd.get("fallback") else "")
            if cmd.get("reasoning"):
                log.info("[LLM] Reasoning: %s", cmd["reasoning"])
                reasoning_log.append(f"step{step_n}: {cmd['reasoning']}")

            # Validate the decision against the two invariants the system prompt
            # already states (don't revisit a visited waypoint; finish with
            # RTL/LAND once all required waypoints are done). Small local models
            # don't always follow this reliably across many stateless calls, so
            # we give it ONE explicit chance to reconsider — the same
            # nudge-and-retry pattern agent_loop.py already uses for invalid
            # JSON / unrecognised action types. This does NOT override the
            # LLM's decision; it asks again, and whatever it says next is used.
            violation = None
            if command == "NAV_WAYPOINT" and not open_ended:
                p = cmd.get("params", {})
                if "lat" in p and "lon" in p:
                    for w in visited:
                        w_lat, w_lon = NAMED_WAYPOINTS[w]
                        if _hav(float(p["lat"]), float(p["lon"]), w_lat, w_lon) <= _WP_REACH_DIST:
                            violation = (
                                f"You selected NAV_WAYPOINT to {w}, but {w} is "
                                f"already in waypoints_visited ({visited}). "
                                f"Choose a different action — fly to the next "
                                f"required waypoint, or if none remain, RTL."
                            )
                            break
            # IMPORTANT: no active retry/correction for open-ended scenarios.
            # SC2.1 exists specifically to observe how the LLM handles
            # ambiguity — when to keep exploring, when to consider itself
            # done, whether it recognizes it's repeating itself. Scripting a
            # nudge toward any of those judgment calls doesn't test
            # ambiguity-handling, it manufactures the appearance of it. The
            # agent only gets passive ground-truth information (the
            # `directive`/inspected_locations telemetry above); what it does
            # with that information — including looping, or never finishing
            # — is the actual result being measured, not a bug to correct.
            if (not open_ended and violation is None and not remaining_names
                    and command not in ("RTL", "LAND")):
                violation = (
                    f"All required waypoints are already visited ({visited}) — "
                    f"the mission objective is complete. You selected {command} "
                    f"instead of finishing. Output RTL now (or LAND if already "
                    f"near home)."
                )

            if violation:
                log.warning("[REACT] Step %d decision rejected — re-asking once: %s",
                            step_n, violation)
                history.append(f"CORRECTION (step {step_n}): {violation}")
                llm_calls += 1
                cmd = agent_step(
                    model=MODEL_REACT,
                    system_prompt=system_prompt,
                    telemetry=telemetry,
                    history=history[-6:],
                    uav=uav,
                    step_label=f"REACT_STEP_{step_n}_RETRY",
                )
                command = cmd.get("command", "?")
                if cmd.get("fallback"):
                    used_fallback = True
                log.info("[REACT] Retry decision: %s %s%s", command, cmd.get("params", {}),
                         " [FALLBACK]" if cmd.get("fallback") else "")
                if cmd.get("reasoning"):
                    log.info("[REACT] Reasoning: %s", cmd["reasoning"])
                    reasoning_log.append(f"step{step_n}_retry: {cmd['reasoning']}")

            if command == "LOITER_TURNS":
                if anomaly_fired:
                    anomaly_response = "LOITER_TURNS"
                else:
                    # The agent chose to loiter for some other reason (e.g. its
                    # own judgement on an open-ended goal) — this is NOT a
                    # response to a disturbance that never happened, and must
                    # not be logged as one.
                    log.info("[REACT] LOITER_TURNS issued with no anomaly active "
                            "this run — not counted as anomaly_response")

            # f. Execute (blocking — flies the segment / sets the mode)
            execute_flight_command(master, uav, cmd, on_poll=_track_home_dist)

            # g. Re-check arrival after the flight completes
            s2 = uav.get_state()
            record_arrival(s2.get("lat", 0.0), s2.get("lon", 0.0), score_wps, arrivals)

            if open_ended:
                c2lat, c2lon = s2.get("lat", 0.0), s2.get("lon", 0.0)
                if c2lat:
                    d_from_home = _hav(c2lat, c2lon, HOME_LAT, HOME_LON)
                    max_dist_from_home = max(max_dist_from_home, d_from_home)
                    if d_from_home >= _INSPECT_MIN_DIST_M:
                        is_new = all(
                            _hav(c2lat, c2lon, plat, plon) >= _INSPECT_MIN_DIST_M
                            for plat, plon in inspected_locations)
                        if is_new:
                            inspected_locations.append((round(c2lat, 4), round(c2lon, 4)))
                            log.info("[REACT] New distinct location inspected: "
                                    "(%.4f, %.4f) — %d total",
                                    c2lat, c2lon, len(inspected_locations))

            history.append(
                f"Step {step_n}: {command} {cmd.get('params', {})} "
                f"| visited={compress(arrivals)}")

            # h. Terminal — the AGENT declares the mission over
            if command in ("RTL", "LAND"):
                terminal_cmd = command
                log.info("Terminal command %s — agent ended the mission", command)
                break

        # ---------------------------------------------------------------
        # The agent CHOOSING RTL/LAND (or the safety-floor override doing
        # so) is not the same as the plane actually getting home —
        # execute_flight_command's RTL/LAND branches just set the mode and
        # return immediately. Stay attached and confirm the return leg
        # before declaring the mission summary.
        # ---------------------------------------------------------------
        if terminal_cmd in ("RTL", "LAND"):
            return_confirmed, return_note = _confirm_return_home(uav)
        else:
            return_note = "mission ended without a terminal RTL/LAND (e.g. step cap exceeded); return leg unverified"

    except KeyboardInterrupt:
        log.warning("Interrupted — RTL")
        set_mode(master, "RTL")
        terminal_cmd = "RTL"; safety_note = "INTERRUPT"
        return_note = "run interrupted by user; return leg unverified"

    finally:
        reached = compress(arrivals)
        if open_ended:
            outcome, failure_type, note = score_open_ended_run(
                max_dist_from_home, terminal_cmd, step_n, step_cap,
                fallback=used_fallback)
            note += f"; distinct_locations_inspected={len(inspected_locations)} {inspected_locations}"
        else:
            outcome, failure_type, note = score_run(
                arrivals, expected_order, step_n, step_cap, terminal_cmd,
                fallback=used_fallback)

        # A run can look like SUCCESS purely from the agent picking RTL/LAND,
        # while the actual return-home leg (flown onboard, after this script
        # would otherwise have moved on) still fails. Don't let that go
        # unrecorded.
        if not return_confirmed and outcome == "SUCCESS":
            outcome = "PARTIAL"
            failure_type = failure_type or "RETURN_NOT_CONFIRMED"

        notes = (f"steps={step_n}/{step_cap}; terminal={terminal_cmd or 'NONE'}; {note}; "
                f"return_confirmed={return_confirmed} ({return_note})")
        if safety_note:
            notes += f"; {safety_note}"
        if reasoning_log:
            notes += f"; reasoning=[{' | '.join(reasoning_log)}]"

        log.info("=" * 60)
        log.info("MISSION SUMMARY (scenario %s):", scenario_id)
        log.info("  Waypoints reached : %s", reached)
        log.info("  Expected order    : %s", expected_order)
        log.info("  Terminal command  : %s", terminal_cmd or "NONE")
        log.info("  Steps taken       : %d / %d", step_n, step_cap)
        log.info("  Anomaly enabled   : %s (fired=%s)", anomaly_enabled, anomaly_fired)
        log.info("  Return confirmed  : %s (%s)", return_confirmed, return_note)
        log.info("  Reasoning captured: %d step(s)", len(reasoning_log))
        log.info("  LLM calls         : %d", llm_calls)
        log.info("  OUTCOME           : %s (%s)", outcome, failure_type)
        log.info("=" * 60)

        log_run(
            paradigm="ReAct",
            model_primary=MODEL_REACT,
            model_secondary="",
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
    print("=== PARADIGM A: ReAct Agent ===")
    print(f"Model  : {MODEL_REACT}")
    print("Mission: Wildfire boundary mapping — Rawalpindi SITL")
    print("Loop   : LLM called every 5s — continuous reasoning")
    input("Press Enter to begin...")
    run_react_mission()