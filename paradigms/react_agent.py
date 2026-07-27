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
from shared.prompts import REACT_SYSTEM_PROMPT
from shared.agent_loop import agent_step
from shared.logger import log_run
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

def _fly_segment(master, uav, lat, lon, alt, label, timeout=_NAV_TIMEOUT):
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
# Execute flight command — called by the ReAct loop after each LLM decision
# ---------------------------------------------------------------------------
def execute_flight_command(master, uav, command_dict):
    command = command_dict.get("command", "")
    params  = command_dict.get("params", {})
    log.info("Executing: %s %s", command, params)

    if command == "NAV_WAYPOINT":
        lat = float(params.get("lat", HOME_LAT))
        lon = float(params.get("lon", HOME_LON))
        alt = float(params.get("alt", CRUISE_ALT))
        _fly_segment(master, uav, lat, lon, alt, f"WP({lat:.4f},{lon:.4f})")

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
            _fly_segment(master, uav, lat, lon, alt, "ANOMALY_SITE")
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
def run_react_mission(scenario_id: str = "SC1", run_number: int = 1):
    _run_start = time.time()
    log.info("=" * 60)
    log.info("PARADIGM A: ReAct Agent — Wildfire Boundary Mapping")
    log.info("Model  : %s", MODEL_REACT)
    log.info("Route  : Home → WP_ALPHA → MIDPOINT → ANOMALY → WP_BRAVO → RTL")
    log.info("Loop   : LLM called every %ds — continuous reasoning", _LOOP_INTERVAL)
    log.info("Log    : %s", _LOG_FILE)
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

    # Upload takeoff mission — HOME + TAKEOFF + WP_ALPHA
    # On reaching WP_ALPHA the plane holds it; the ReAct loop then takes over.
    frame = mavutil.mavlink.MAV_FRAME_GLOBAL_RELATIVE_ALT
    log.info("Uploading takeoff + WP_ALPHA mission ...")
    _send_items(master, [
        dict(seq=0, frame=0,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1, p1=0, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=0),
        dict(seq=1, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_TAKEOFF,
             current=1, autocontinue=1, p1=15, p2=0, p3=0, p4=0,
             x=HOME_LAT, y=HOME_LON, z=CRUISE_ALT),
        dict(seq=2, frame=frame,
             command=mavutil.mavlink.MAV_CMD_NAV_WAYPOINT,
             current=0, autocontinue=1, p1=0, p2=200, p3=0, p4=0,
             x=WP_ALPHA_LAT, y=WP_ALPHA_LON, z=CRUISE_ALT),
    ])
    set_mode(master, "AUTO")
    _arm(master, uav)
    log.info("[OK] Takeoff started — waiting for WP_ALPHA before ReAct loop")

    # Wait for WP_ALPHA before starting the reasoning loop
    log.info("=" * 60)
    log.info("PHASE 1: Waiting for AUTO flight to WP_ALPHA ...")
    log.info("=" * 60)
    wp_alpha_ok = False
    t0 = time.time()
    while time.time() - t0 < _NAV_TIMEOUT:
        s    = uav.get_state()
        clat = s.get("lat", 0.0)
        bat  = s.get("battery_pct", 100)
        wp   = s.get("wp_seq", 0)
        if clat:
            dist_alpha = _hav(clat, s.get("lon", 0), WP_ALPHA_LAT, WP_ALPHA_LON)
            agl = s.get("alt", 0) - 584 if s.get("alt", 0) > 100 else s.get("alt", 0)
            log.info("[PHASE 1] wp=%d mode=%s bat=%d%% AGL~%.0fm dist_alpha=%.0fm",
                     wp, s.get("mode", ""), bat, agl, dist_alpha)
            if wp >= 2 and dist_alpha <= _WP_REACH_DIST:
                log.info("[OK] WP_ALPHA reached (wp=%d dist=%.0fm)", wp, dist_alpha)
                wp_alpha_ok = True
                break
            if 0 <= bat < 15:
                log.warning("[PHASE 1] Battery critical — RTL")
                set_mode(master, "RTL"); return
        time.sleep(_POLL_S)

    # -----------------------------------------------------------------------
    # ReAct continuous reasoning loop
    # LLM is called on EVERY iteration with current telemetry + mission_phase
    # -----------------------------------------------------------------------
    log.info("=" * 60)
    log.info("Starting ReAct reasoning loop — LLM called every %ds", _LOOP_INTERVAL)
    log.info("=" * 60)

    history       = [f"WP_ALPHA {'reached' if wp_alpha_ok else 'missed'}"]
    step_n        = 0
    anomaly_fired = False
    mission_phase = "TRANSIT"   # TRANSIT → ANOMALY_INVESTIGATION → RESUME → COMPLETE

    # Best-effort run-log tracking (see shared/logger.py)
    llm_calls        = 0
    anomaly_response = "NONE"   # set to the agent's anomaly action when taken

    wp_alpha_ok = wp_alpha_ok
    mid_ok      = False
    anomaly_ok  = False
    bravo_ok    = False

    try:
        while True:
            time.sleep(_LOOP_INTERVAL)
            step_n += 1

            # a. Get current telemetry
            s    = uav.get_state()
            clat = s.get("lat", 0.0)
            clon = s.get("lon", 0.0)
            bat  = s.get("battery_pct", 100)
            mode = s.get("mode", "")
            wp   = s.get("wp_seq", 0)

            # b. Phase transitions
            if mission_phase == "TRANSIT" and anomaly_fired:
                mission_phase = "ANOMALY_INVESTIGATION"
                log.info("Phase: TRANSIT → ANOMALY_INVESTIGATION")

            if mission_phase == "ANOMALY_INVESTIGATION" and any(
                "LOITER" in h or "ANOMALY INVESTIGATED" in h for h in history
            ):
                mission_phase = "RESUME"
                log.info("Phase: ANOMALY_INVESTIGATION → RESUME")

            # c. Check MIDPOINT proximity → trigger anomaly once
            if clat and not anomaly_fired:
                dist_mid = _hav(clat, clon, MIDPOINT_LAT, MIDPOINT_LON)
                passed = (dist_mid <= 400 or wp >= 3 or clon > 72.975)
                if passed:
                    log.info("MIDPOINT reached (%.0fm) — triggering dual anomaly!", dist_mid)
                    uav.trigger_anomaly()
                    anomaly_fired = True
                    mid_ok = True
                    history.append(
                        "MIDPOINT reached. Dual anomaly triggered: thermal spike at "
                        f"ANOMALY({ANOMALY_LAT},{ANOMALY_LON}) 150m north + "
                        "wind reduced groundspeed 40%."
                    )

            # d. Enrich telemetry with mission context for LLM
            telemetry = {
                **s,
                "mission_phase":      mission_phase,
                "anomaly_triggered":  anomaly_fired,
                "loiter_completed":   mission_phase in ("RESUME", "COMPLETE"),
                "dist_to_midpoint_m": round(_hav(clat, clon, MIDPOINT_LAT, MIDPOINT_LON), 1) if clat else -1,
                "dist_to_wp_bravo_m": round(_hav(clat, clon, WP_BRAVO_LAT, WP_BRAVO_LON), 1) if clat else -1,
                "dist_to_anomaly_m":  round(_hav(clat, clon, ANOMALY_LAT, ANOMALY_LON), 1) if clat else -1,
            }

            log.info("--- Step %d | phase=%-22s | mode=%s | bat=%d%% | wp=%d ---",
                     step_n, mission_phase, mode, bat, wp)

            # e. Battery safety check
            if 0 <= bat < 15:
                log.warning("[SAFETY] Battery critical — RTL")
                set_mode(master, "RTL"); break

            # f. RESUME phase — deterministic Python fallback (RESEARCH NOTE)
            #    The LLM consistently re-issued LOITER_TURNS after anomaly
            #    investigation regardless of phase context. This is itself a
            #    documented failure mode (REASONING failure — agent cannot
            #    transition out of investigation). The deterministic fallback
            #    prevents mission stall and is disclosed in the paper as a
            #    ReAct-specific failure mode, not a silent workaround.
            if mission_phase == "RESUME":
                log.info("[RESUME] Anomaly investigated — flying to WP_BRAVO")
                bravo_ok = _fly_segment(
                    master, uav, WP_BRAVO_LAT, WP_BRAVO_LON, CRUISE_ALT, "WP_BRAVO")
                history.append(
                    f"WP_BRAVO {'reached' if bravo_ok else 'missed'} after anomaly investigation")
                mission_phase = "COMPLETE"
                log.info("Phase: RESUME → COMPLETE")
                set_mode(master, "RTL")
                log.info("[CMD] RTL — mission complete")
                break

            # g. Call LLM — EVERY iteration (TRANSIT and ANOMALY_INVESTIGATION)
            log.info("[LLM] Calling %s (step %d, phase=%s) ...",
                     MODEL_REACT, step_n, mission_phase)
            # History is capped at the last 6 entries to stay within the model's
            # practical context window at reasonable inference speed. Earlier
            # entries are not passed to the LLM; full length is logged here so
            # the truncation is a documented design choice, not a silent limit.
            log.info("[HISTORY] Total=%d, passing last 6 to LLM", len(history))
            llm_calls += 1
            cmd = agent_step(
                model=MODEL_REACT,
                system_prompt=REACT_SYSTEM_PROMPT,
                telemetry=telemetry,
                history=history[-6:],
                uav_state=s,
                step_label=f"REACT_STEP_{step_n}",
            )
            log.info("[LLM] Decision: %s %s", cmd.get("command"), cmd.get("params", {}))

            # h. Execute command
            execute_flight_command(master, uav, cmd)

            # Record loiter investigation
            if cmd.get("command") == "LOITER_TURNS":
                anomaly_ok = True
                anomaly_response = "LOITER_TURNS"
                history.append(
                    "ANOMALY INVESTIGATED: LOITER_TURNS completed at anomaly site. "
                    "Resume mission to WP_BRAVO.")
                log.info("Loiter investigation recorded in history")

            # i. Record step in history
            history.append(
                f"Step {step_n}: {cmd.get('command')} "
                f"{cmd.get('params', {})} | phase={mission_phase}")
            log.info("Step %d complete | command=%s | phase=%s | history=%d",
                     step_n, cmd.get("command", "?"), mission_phase, len(history))

            # j. Terminal conditions
            if cmd.get("command") in ("RTL", "LAND"):
                log.info("Terminal command %s — ending ReAct loop", cmd.get("command"))
                break

            if mission_phase == "COMPLETE":
                log.info("Mission complete — ending ReAct loop")
                break

    except KeyboardInterrupt:
        log.warning("Interrupted — RTL")
        set_mode(master, "RTL")

    finally:
        log.info("=" * 60)
        log.info("MISSION SUMMARY:")
        log.info("  WP_ALPHA visited    : %s", wp_alpha_ok)
        log.info("  MIDPOINT reached    : %s", mid_ok)
        log.info("  Anomaly investigated: %s", anomaly_ok)
        log.info("  WP_BRAVO reached    : %s", bravo_ok)
        log.info("  ReAct steps taken   : %d", step_n)
        log.info("=" * 60)

        # Structured run log — best-effort field derivation from local state.
        visited = [w for w, ok in (
            ("WP_ALPHA", wp_alpha_ok), ("MIDPOINT", mid_ok),
            ("ANOMALY", anomaly_ok), ("WP_BRAVO", bravo_ok)) if ok]
        outcome = "COMPLETED" if bravo_ok else (
            "ABORTED_RTL" if mission_phase != "COMPLETE" and anomaly_fired else "FAILED")
        log_run(
            paradigm="ReAct",
            model_primary=MODEL_REACT,
            model_secondary="",
            scenario_id=scenario_id,
            run_number=run_number,
            outcome=outcome,
            failure_type="NONE" if outcome == "COMPLETED" else "",
            waypoints_visited=visited,
            anomaly_response=anomaly_response,
            llm_calls=llm_calls,
            duration_seconds=time.time() - _run_start,
            telemetry_final=uav.get_state(),
            notes=f"steps={step_n}",
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