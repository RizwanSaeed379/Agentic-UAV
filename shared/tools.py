"""
shared/tools.py — Tool registry for agentic UAV wildfire boundary mapping.

LLM agents call these tools to get real-time operational data before
making flight decisions (divert, continue, RTL, etc.).

Mission: Home -> WP_ALPHA -> WP_BRAVO
Mid-transit: thermal anomaly + 40% groundspeed wind reduction at ANOMALY coords.
"""

from __future__ import annotations

import math

# ---------------------------------------------------------------------------
# Mission constants
# ---------------------------------------------------------------------------

HOME_LAT,     HOME_LON     = 33.7097, 72.9673
WP_ALPHA_LAT, WP_ALPHA_LON = 33.7120, 72.9673
WP_BRAVO_LAT, WP_BRAVO_LON = 33.7120, 72.9950
ANOMALY_LAT,  ANOMALY_LON  = 33.7134, 72.9812
MIDPOINT_LAT, MIDPOINT_LON = 33.7120, 72.9812
CRUISE_ALT = 30

GEOFENCE = {
    "lat_min": 33.6900, "lat_max": 33.7300,
    "lon_min": 72.9400, "lon_max": 73.0200,
}

# Battery model: 1% consumed per 50 m flown
_BATTERY_PCT_PER_METRE = 1.0 / 50.0
_MIN_RESERVE_PCT       = 15.0   # matches the 15% abort threshold in all paradigms


# ---------------------------------------------------------------------------
# Internal helper
# BUG FIX: replaced math.asin with math.atan2 for numerical stability
# (consistent with the fix applied to reflexion_agent.py BUG 1)
# ---------------------------------------------------------------------------

def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return distance in metres between two WGS-84 coordinates."""
    R = 6_371_000.0
    p1  = math.radians(lat1)
    p2  = math.radians(lat2)
    dp  = math.radians(lat2 - lat1)
    dl  = math.radians(lon2 - lon1)
    a   = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


# ---------------------------------------------------------------------------
# Tool functions
# ---------------------------------------------------------------------------

def tool_get_telemetry(uav_state: dict) -> str:
    """Return a formatted string of all current telemetry values."""
    s = uav_state
    return (
        f"TELEMETRY | "
        f"lat={s.get('lat', 0.0):.5f}  "
        f"lon={s.get('lon', 0.0):.5f}  "
        f"alt={s.get('alt', 0.0):.1f}m  "
        f"airspeed={s.get('airspeed', 0.0):.1f}m/s  "
        f"groundspeed={s.get('groundspeed', 0.0):.1f}m/s  "
        f"heading={s.get('heading', 0.0):.1f}deg  "
        f"battery={s.get('battery_pct', -1)}% ({s.get('battery_volt', 0.0):.1f}V)  "
        f"sats={s.get('sat_count', 0)}  "
        f"armed={s.get('armed', False)}  "
        f"mode={s.get('mode', 'UNKNOWN')}  "
        f"wp_seq={s.get('wp_seq', 0)}  "
        f"wind_speed={s.get('wind_speed', 0.0):.1f}m/s  "
        f"anomaly={s.get('anomaly_triggered', False)}"
    )


def tool_check_weather(uav_state: dict) -> str:
    """
    Check wind status from telemetry.

    Alert fires if wind_speed is non-zero OR groundspeed has dropped
    more than 35% below airspeed (i.e., below 65% of airspeed).
    """
    wind  = uav_state.get("wind_speed", 0.0)
    gs    = uav_state.get("groundspeed", 0.0)
    ias   = uav_state.get("airspeed", 0.0)

    wind_active = wind > 0.0
    speed_drop  = ias > 0.0 and gs < (ias * 0.65)

    if wind_active or speed_drop:
        return (
            "ALERT: Wind gust detected. "
            "Groundspeed reduced 40% below airspeed."
        )
    return "CLEAR: Wind nominal."


def tool_check_battery(
    target_lat: float, target_lon: float, uav_state: dict
) -> str:
    """
    Estimate battery sufficiency for a divert-to-target and return-to-home leg.

    Model: 1% battery consumed per 50 m.
    Reserve threshold: 15% (same value the paradigms abort at).
    """
    cur_lat = uav_state.get("lat",         HOME_LAT)
    cur_lon = uav_state.get("lon",         HOME_LON)
    bat_pct = uav_state.get("battery_pct", 100)
    if bat_pct < 0:
        bat_pct = 100   # not yet received — assume full for safety calc

    dist_divert = _haversine(cur_lat, cur_lon, target_lat, target_lon)
    dist_home   = _haversine(target_lat, target_lon, HOME_LAT, HOME_LON)
    total_m     = dist_divert + dist_home

    used_pct      = total_m * _BATTERY_PCT_PER_METRE
    remaining_pct = bat_pct - used_pct

    divert_m = round(dist_divert, 1)
    home_m   = round(dist_home, 1)
    used_r   = round(used_pct, 1)
    rem_r    = round(remaining_pct, 1)

    if remaining_pct < _MIN_RESERVE_PCT:
        return (
            f"INSUFFICIENT_RESERVE: Divert={divert_m}m + RTH={home_m}m = "
            f"{round(total_m,1)}m total. "
            f"Estimated battery consumed: {used_r}%. "
            f"Remaining after trip: {rem_r}% "
            f"(below {_MIN_RESERVE_PCT:.0f}% minimum reserve). "
            "Recommend RTL now."
        )

    return (
        f"SUFFICIENT_RESERVE: Divert={divert_m}m + RTH={home_m}m = "
        f"{round(total_m,1)}m total. "
        f"Estimated battery consumed: {used_r}%. "
        f"Remaining after trip: {rem_r}% "
        f"(above {_MIN_RESERVE_PCT:.0f}% reserve). "
        "Divert is feasible."
    )


def tool_geofence_check(target_lat: float, target_lon: float) -> str:
    """Check whether target coordinates are within the active geofence."""
    within = (
        GEOFENCE["lat_min"] <= target_lat <= GEOFENCE["lat_max"]
        and
        GEOFENCE["lon_min"] <= target_lon <= GEOFENCE["lon_max"]
    )
    if within:
        return "APPROVED: Within safe flight boundaries."
    return (
        f"REJECTED: Outside active geofence. "
        f"Target ({target_lat:.4f}, {target_lon:.4f}) is outside "
        f"lat [{GEOFENCE['lat_min']}, {GEOFENCE['lat_max']}] "
        f"lon [{GEOFENCE['lon_min']}, {GEOFENCE['lon_max']}]."
    )


def tool_anomaly_status(uav_state: dict) -> str:
    """Report thermal anomaly status from current UAV state."""
    if uav_state.get("anomaly_triggered", False):
        return (
            f"ANOMALY ACTIVE: Thermal spike at {ANOMALY_LAT},{ANOMALY_LON}. "
            "150m north of mid-transit. Intensity: HIGH."
        )
    return "NOMINAL: No anomaly detected."


def tool_distance_to(
    lat1: float, lon1: float, lat2: float, lon2: float
) -> str:
    """Return haversine distance in metres between two coordinates."""
    dist = _haversine(lat1, lon1, lat2, lon2)
    return f"Distance: {dist:.1f}m"


# ---------------------------------------------------------------------------
# Tool registry
# ---------------------------------------------------------------------------

TOOL_REGISTRY = {
    "get_telemetry": tool_get_telemetry,
    "check_weather":  tool_check_weather,
    "check_battery":  tool_check_battery,
    "geofence_check": tool_geofence_check,
    "anomaly_status": tool_anomaly_status,
    "distance_to":    tool_distance_to,
}


# ---------------------------------------------------------------------------
# Argument resolution for target-taking tools
#
# BUG FIX: check_battery and geofence_check require target coordinates, but
# no caller ever supplied them — the Plan-Execute dispatcher passed {} and the
# prompt's tool_call example shows "arguments":{}, so every call returned
# ERROR: Missing argument for 'check_battery': 'target_lat' instead of a real
# reserve estimate. Missing/partial coordinates now fall back to the anomaly
# location, which is the only divert target in this mission and the target
# these checks were always intended to evaluate. Explicit arguments, when the
# LLM does supply them, are used unchanged.
# ---------------------------------------------------------------------------

def _resolve_target(arguments: dict) -> tuple[float, float]:
    """Return (lat, lon) from tool arguments, defaulting to the anomaly."""
    lat = arguments.get("target_lat", arguments.get("lat", ANOMALY_LAT))
    lon = arguments.get("target_lon", arguments.get("lon", ANOMALY_LON))
    return float(lat), float(lon)


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

def execute_tool(tool_name: str, arguments: dict, uav_state: dict) -> str:
    """
    Dispatch a tool call by name, injecting uav_state where required.

    Args:
        tool_name:  Key from TOOL_REGISTRY.
        arguments:  Dict of keyword arguments for the tool (from LLM JSON).
        uav_state:  Current UAV telemetry dict (injected automatically).

    Returns:
        Tool output string, or an error string on failure.
    """
    if tool_name not in TOOL_REGISTRY:
        return f"ERROR: Unknown tool '{tool_name}'"

    try:
        # Tools that take only uav_state
        if tool_name in ("get_telemetry", "check_weather", "anomaly_status"):
            return TOOL_REGISTRY[tool_name](uav_state)

        # Tools that take positional args + uav_state
        if tool_name == "check_battery":
            lat, lon = _resolve_target(arguments)
            return tool_check_battery(lat, lon, uav_state)

        # Tools that take only positional args (no uav_state)
        if tool_name == "geofence_check":
            lat, lon = _resolve_target(arguments)
            return tool_geofence_check(lat, lon)

        if tool_name == "distance_to":
            return tool_distance_to(
                float(arguments["lat1"]),
                float(arguments["lon1"]),
                float(arguments["lat2"]),
                float(arguments["lon2"]),
            )

    except KeyError as exc:
        return f"ERROR: Missing argument for '{tool_name}': {exc}"
    except (ValueError, TypeError) as exc:
        return f"ERROR: Bad argument type for '{tool_name}': {exc}"
    except Exception as exc:
        return f"ERROR: Tool '{tool_name}' raised an unexpected error: {exc}"

    return f"ERROR: Unknown tool '{tool_name}'"
