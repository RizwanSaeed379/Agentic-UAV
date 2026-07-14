"""
shared/mavlink_state.py — async MAVLink telemetry thread for agentic UAV.

Auto-connects to ArduPlane SITL (tcp:127.0.0.1:5762 then tcp:127.0.0.1:5760),
maintains a live thread-safe state dict, and sends GCS heartbeats every second.
"""

from __future__ import annotations

import copy
import logging
import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from pymavlink import mavutil

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

HOME_LAT, HOME_LON = 33.7097, 72.9673   # Rawalpindi SITL home
HOME_ALT   = 584.0                        # metres ASL — Rawalpindi elevation
CRUISE_ALT = 30                           # metres AGL

TCP_PORTS = [
    "tcp:127.0.0.1:5762",
    "tcp:127.0.0.1:5760",
]

HEARTBEAT_HZ    = 1.0   # GCS heartbeat rate
CONNECT_TIMEOUT = 10.0  # seconds to wait per port
RECV_TIMEOUT    = 0.1   # recv poll interval (seconds)

PLANE_MODES: Dict[int, str] = {
    0: "MANUAL",      1: "CIRCLE",    2: "STABILIZE",   3: "TRAINING",
    4: "ACRO",        5: "FBWA",      6: "FBWB",         7: "CRUISE",
    8: "AUTOTUNE",   10: "AUTO",     11: "RTL",          12: "LOITER",
    13: "TAKEOFF",   14: "AVOID_ADSB", 15: "GUIDED",    16: "INITIALISING",
}

log = logging.getLogger(__name__)

# The most recently started UAVState. Protocol helpers that only receive a bare
# `master` connection (e.g. paradigm set_mode / _send_items) use this to pause
# the background telemetry reader during a handshake.
_ACTIVE_STATE: Optional["UAVState"] = None


@contextmanager
def telemetry_paused(settle: float = 0.25):
    """
    Pause the active UAVState telemetry loop for exclusive socket access.

    Safe to use even if no UAVState is running yet (yields None, does nothing),
    so it can wrap helpers that run both before and after uav.start().
    """
    st = _ACTIVE_STATE
    if st is None:
        yield None
        return
    with st.exclusive(settle) as m:
        yield m


# ---------------------------------------------------------------------------
# UAVState
# ---------------------------------------------------------------------------

class UAVState:
    """
    Thread-safe MAVLink telemetry state for an ArduPlane SITL UAV.

    Auto-detects TCP connection, maintains live state, sends GCS heartbeats.

    Usage:
        uav = UAVState()
        uav.start()                # connects and launches background thread
        state = uav.get_state()   # thread-safe snapshot
        uav.trigger_anomaly()     # flags anomaly + applies wind-speed effect
        uav.stop()
    """

    def __init__(self, master=None, auto_relocate: bool = True) -> None:
        self._external_master = master   # caller-owned connection; we never close it
        self._auto_relocate   = auto_relocate
        self._lock   = threading.Lock()
        self._stop   = threading.Event()
        # Pause primitives — let another thread take EXCLUSIVE ownership of the
        # MAVLink socket (mission upload / mode-set handshakes read protocol
        # replies that would otherwise be stolen by this recv loop).
        self._pause_req = threading.Event()   # caller requests the loop to pause
        self._paused    = threading.Event()   # loop confirms it has paused reads
        self._master: Optional[mavutil.mavfile] = None
        self._thread: Optional[threading.Thread] = None
        self._state: Dict[str, Any] = {
            "lat":               0.0,
            "lon":               0.0,
            "alt":               0.0,
            "alt_asl":           0.0,
            "airspeed":          0.0,
            "groundspeed":       0.0,
            "heading":           0.0,
            "battery_pct":       100,
            "battery_volt":      0.0,
            "sat_count":         0,
            "armed":             False,
            "mode":              "UNKNOWN",
            # BUG FIX: wp_seq was missing from state entirely — all three
            # paradigms read telemetry.get("wp_seq", 0) for phase transitions
            # but it was never populated, so transitions based on wp_seq
            # never fired correctly. Now initialised to 0 and updated from
            # MISSION_CURRENT MAVLink messages in _dispatch.
            "wp_seq":            0,
            "anomaly_triggered": False,
            "wind_speed":        0.0,
        }

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Connect (or reuse existing connection) and launch the telemetry thread."""
        if self._external_master is not None:
            self._master = self._external_master
            print("[UAVState] Using existing MAVLink connection (no new socket opened)")
        else:
            self._master = self._connect()

        if self._auto_relocate:
            self.relocate_to_rawalpindi(self._master)

        thread = threading.Thread(target=self._recv_loop, daemon=True)
        thread.name = "mavlink-telemetry"
        thread.start()
        self._thread = thread

        # Register as the active telemetry instance so protocol helpers
        # (mission upload, mode-set) can pause us via telemetry_paused().
        global _ACTIVE_STATE
        _ACTIVE_STATE = self
        print("[UAVState] Telemetry thread started")

    # ------------------------------------------------------------------
    # Exclusive socket access
    # ------------------------------------------------------------------

    @contextmanager
    def exclusive(self, settle: float = 0.25):
        """
        Temporarily suspend the telemetry recv loop so the calling thread has
        SOLE access to the MAVLink socket.

        pymavlink connections are not safe to read from two threads at once:
        whichever recv_match() fires first consumes the frame. During a mission
        upload or mode-set handshake the background loop would otherwise steal
        the MISSION_REQUEST / MISSION_ACK / HEARTBEAT replies the caller is
        waiting for, causing "upload timed out" / "mode change timed out".

        Usage:
            with uav.exclusive():
                master.mav.mission_count_send(...)
                master.recv_match(type="MISSION_REQUEST", ...)
        """
        # No running loop (e.g. called before start()) → nothing to pause.
        if self._thread is None or not self._thread.is_alive():
            yield self._master
            return

        self._pause_req.set()
        # Wait for the loop to acknowledge it has stopped touching the socket.
        self._paused.wait(timeout=2.0)
        # Let any recv_match already in flight (<= RECV_TIMEOUT) drain out.
        time.sleep(settle)
        try:
            yield self._master
        finally:
            self._pause_req.clear()
            self._paused.clear()

    def stop(self) -> None:
        """Signal the thread to exit and close the connection if we own it."""
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        global _ACTIVE_STATE
        if _ACTIVE_STATE is self:
            _ACTIVE_STATE = None
        # Only close the socket if UAVState opened it — never close a caller-owned master
        if self._master and self._external_master is None:
            try:
                self._master.close()
            except Exception:
                pass

    def get_state(self) -> Dict[str, Any]:
        """Return a shallow copy of the current state (thread-safe)."""
        with self._lock:
            return copy.copy(self._state)

    def relocate_to_rawalpindi(self, master: mavutil.mavfile) -> None:
        """
        Move the SITL home position to Rawalpindi (33.7097, 72.9673, 584m ASL).
        """
        relocate_home(master)

    def trigger_anomaly(self) -> None:
        """
        RESEARCH NOTE — simulation scope:
        This method injects anomaly conditions into the Python telemetry
        state dictionary. ArduPilot SITL does not physically simulate wind
        or thermal events. The LLM agent observes these conditions through
        the prompt (via telemetry fields), not through real sensor data.
        This is explicitly disclosed in the paper's methodology section.
        """
        with self._lock:
            self._state["anomaly_triggered"] = True
            self._state["wind_speed"] = round(self._state["airspeed"] * 0.40, 2)
        print(
            f"[UAVState] Anomaly triggered — "
            f"wind_speed set to {self._state['wind_speed']} m/s "
            f"(40% of airspeed={self._state['airspeed']} m/s)"
        )

    # ------------------------------------------------------------------
    # Connection  (auto-detect across TCP_PORTS)
    # ------------------------------------------------------------------

    def _connect(self) -> mavutil.mavfile:
        print("[UAVState] Auto-detecting ArduPlane SITL connection ...")
        for port in TCP_PORTS:
            print(f"[UAVState]   Trying {port} ...")
            try:
                master = mavutil.mavlink_connection(
                    port,
                    source_system=255,
                    source_component=0,
                    autoreconnect=True,
                )
                msg = master.wait_heartbeat(timeout=CONNECT_TIMEOUT)
                if msg:
                    vtype = {1: "Fixed-wing", 2: "Multirotor"}.get(
                        msg.type, f"type={msg.type}"
                    )
                    print(
                        f"[UAVState] Connected on {port} | "
                        f"sysid={master.target_system} | {vtype}"
                    )
                    return master
                master.close()
                print(f"[UAVState]   No heartbeat on {port} — skipping")
            except Exception as exc:
                print(f"[UAVState]   {port} error: {exc}")

        raise RuntimeError(
            f"No ArduPlane SITL found on {TCP_PORTS}.\n"
            "  1. Open Mission Planner\n"
            "  2. Simulation -> Plane  (wait ~15 s for plane on map)\n"
            "  3. Restart this script"
        )

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _recv_loop(self) -> None:
        last_hb_sent = 0.0

        while not self._stop.is_set():
            # A protocol helper (mission upload / mode-set) has requested
            # exclusive socket access — stop reading AND writing so we don't
            # steal its replies or interleave bytes on the wire.
            if self._pause_req.is_set():
                self._paused.set()
                time.sleep(0.02)
                continue
            self._paused.clear()

            now = time.monotonic()

            # Keep GCS heartbeat alive at 1 Hz
            if now - last_hb_sent >= 1.0 / HEARTBEAT_HZ:
                self._send_heartbeat()
                last_hb_sent = now

            msg = self._master.recv_match(blocking=True, timeout=RECV_TIMEOUT)
            if msg is None or msg.get_type() == "BAD_DATA":
                continue

            self._dispatch(msg)

    def _send_heartbeat(self) -> None:
        try:
            self._master.mav.heartbeat_send(
                mavutil.mavlink.MAV_TYPE_GCS,
                mavutil.mavlink.MAV_AUTOPILOT_INVALID,
                0,   # base_mode
                0,   # custom_mode
                mavutil.mavlink.MAV_STATE_ACTIVE,
            )
        except Exception as exc:
            log.debug("Heartbeat send error: %s", exc)

    # ------------------------------------------------------------------
    # Message dispatch  →  state update
    # ------------------------------------------------------------------

    def _dispatch(self, msg) -> None:
        t = msg.get_type()

        with self._lock:
            s = self._state

            if t == "GLOBAL_POSITION_INT":
                s["lat"] = msg.lat / 1e7
                s["lon"] = msg.lon / 1e7
                s["alt"] = msg.relative_alt / 1000.0   # AGL in metres

            elif t == "VFR_HUD":
                s["airspeed"]    = msg.airspeed
                s["groundspeed"] = msg.groundspeed
                s["heading"]     = msg.heading
                s["alt"]         = msg.alt              # matches Mission Planner HUD

            elif t == "SYS_STATUS":
                if msg.battery_remaining >= 0:
                    s["battery_pct"]  = msg.battery_remaining
                    s["battery_volt"] = msg.voltage_battery / 1000.0

            elif t == "GPS_RAW_INT":
                s["sat_count"] = msg.satellites_visible

            elif t == "HEARTBEAT":
                s["armed"] = bool(
                    msg.base_mode & mavutil.mavlink.MAV_MODE_FLAG_SAFETY_ARMED
                )
                s["mode"] = PLANE_MODES.get(msg.custom_mode, f"#{msg.custom_mode}")

            # BUG FIX: MISSION_CURRENT was never handled — wp_seq was always
            # 0 regardless of mission progress. All three paradigms use wp_seq
            # for phase transitions (e.g. wp_seq >= 3 triggers anomaly in
            # react_agent, wp_seq >= 4 triggers RESUME→COMPLETE transition).
            # Now correctly updated from MISSION_CURRENT messages.
            elif t == "MISSION_CURRENT":
                s["wp_seq"] = msg.seq


# ---------------------------------------------------------------------------
# Module-level relocation helper
# ---------------------------------------------------------------------------

def relocate_home(master: mavutil.mavfile) -> None:
    """
    Move the SITL home position to Rawalpindi (33.7097, 72.9673, 584m ASL).

    Sends three overlapping MAVLink commands so ArduPlane registers the
    relocation at every level, then reads back HOME_POSITION to confirm.
    Can be called from paradigm scripts independently of UAVState.
    """
    lat_1e7 = int(HOME_LAT * 1e7)
    lon_1e7 = int(HOME_LON * 1e7)
    alt_mm  = int(HOME_ALT * 1000)

    # Step 1 — SET_GPS_GLOBAL_ORIGIN
    master.mav.set_gps_global_origin_send(
        master.target_system,
        lat_1e7,
        lon_1e7,
        alt_mm,
    )
    time.sleep(0.5)

    # Step 2 — SET_HOME_POSITION
    master.mav.set_home_position_send(
        master.target_system,
        lat_1e7,
        lon_1e7,
        alt_mm,
        0, 0, 0,
        [1, 0, 0, 0],
        0, 0, 0,
        0,
    )
    time.sleep(0.5)

    # Step 3 — MAV_CMD_DO_SET_HOME (id=179) via command_long
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        179,        # MAV_CMD_DO_SET_HOME
        0,
        0,          # param1: 0 = use specified location
        0, 0, 0,
        HOME_LAT,
        HOME_LON,
        HOME_ALT,
    )
    time.sleep(1.0)

    # Step 4 — Read back HOME_POSITION to confirm
    master.mav.command_long_send(
        master.target_system,
        master.target_component,
        512,    # MAV_CMD_REQUEST_MESSAGE
        0,
        242,    # MESSAGE_ID = HOME_POSITION
        0, 0, 0, 0, 0, 0,
    )
    msg = master.recv_match(type="HOME_POSITION", blocking=True, timeout=5)
    if msg:
        lat = msg.latitude  / 1e7
        lon = msg.longitude / 1e7
        alt = msg.altitude  / 1000.0
        print(
            f"[RELOCATION] Confirmed HOME_POSITION: "
            f"{lat:.4f}, {lon:.4f}, {alt:.1f}m ASL"
        )
    else:
        print(
            "[RELOCATION] WARNING: HOME_POSITION readback timed out — "
            "relocation may still have applied"
        )

    print(
        f"[RELOCATION] Home set to Rawalpindi: "
        f"{HOME_LAT}, {HOME_LON}, {HOME_ALT}m ASL"
    )
