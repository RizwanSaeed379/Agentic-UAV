"""
shared/scenarios.py — Scenario configuration + shared run scoring.

One source of truth for all three paradigms (ReAct, Plan-Execute, Reflexion).
The flight ROUTE is owned by the LLM; the fields here only gate the disturbance
(anomaly injection), the step cap, and how a run is scored. This is what makes
SC1..SC6 *behaviour* rather than just log labels.

Global settings (from the SC1 manual) live here so every scenario inherits the
same defaults: 30-step cap and a 200 m arrival radius.

No pymavlink here — this module is pure Python so any paradigm (and tests) can
import it without touching the vehicle layer.
"""

from __future__ import annotations

import math

from shared.tools import (
    HOME_LAT, HOME_LON, WP_ALPHA_LAT, WP_ALPHA_LON,
    WP_BRAVO_LAT, WP_BRAVO_LON, ANOMALY_LAT, ANOMALY_LON,
    MIDPOINT_LAT, MIDPOINT_LON,
)

# ---------------------------------------------------------------------------
# Named waypoints (mirror shared/tools.py constants)
# ---------------------------------------------------------------------------
NAMED_WAYPOINTS = {
    "HOME":     (HOME_LAT,     HOME_LON),
    "WP_ALPHA": (WP_ALPHA_LAT, WP_ALPHA_LON),
    "WP_BRAVO": (WP_BRAVO_LAT, WP_BRAVO_LON),
    "ANOMALY":  (ANOMALY_LAT,  ANOMALY_LON),
    "MIDPOINT": (MIDPOINT_LAT, MIDPOINT_LON),
}

# ---------------------------------------------------------------------------
# Global run settings — identical across all scenarios (SC1 manual)
# ---------------------------------------------------------------------------
DEFAULT_STEP_CAP  = 30      # hard cap; exceeding it => TIMEOUT
WP_REACH_DIST_M   = 200     # arrival radius (matches paradigm _WP_REACH_DIST)

# The exact goal string the agent is given (SC1 manual, Step 2).
SC1_GOAL = (
    "Fly from Waypoint Alpha to Waypoint Bravo mapping the wildfire boundary "
    "then return home and land."
)

# ---------------------------------------------------------------------------
# Scenario table — SC1 fully defined. SC2..SC6 slot in with the same shape.
# ---------------------------------------------------------------------------
SCENARIOS = {
    "SC1": {
        "description":     "Normal Execution baseline — no disturbance",
        "goal":            SC1_GOAL,
        "anomaly_enabled": False,                      # default; runtime may override
        "step_cap":        DEFAULT_STEP_CAP,
        "expected_order":  ["WP_ALPHA", "WP_BRAVO"],   # required visit order
        "score_waypoints": ["WP_ALPHA", "WP_BRAVO"],   # arrivals tracked for scoring
    },
    # SC2..SC6: add entries here when specced. get_scenario/score_run are generic.
}


def get_scenario(scenario_id: str) -> dict:
    """Return the config for a scenario id, falling back to SC1 for unknowns."""
    key = (scenario_id or "").strip().upper()
    cfg = SCENARIOS.get(key)
    if cfg is None:
        cfg = dict(SCENARIOS["SC1"])
        cfg["description"] += f" (fallback — scenario {scenario_id!r} not defined)"
    return cfg


def resolve_anomaly(cfg: dict, override) -> bool:
    """Resolve whether trigger_anomaly() should fire this run.

    override:
        None  -> use the scenario default (cfg['anomaly_enabled'])
        True  -> force ON  (inject the disturbance)
        False -> force OFF (clean telemetry)
    """
    if override is None:
        return bool(cfg.get("anomaly_enabled", False))
    return bool(override)


# ---------------------------------------------------------------------------
# Geometry / arrival tracking (observation only — never redirects the flight)
# ---------------------------------------------------------------------------
def haversine(la1, lo1, la2, lo2) -> float:
    """Distance in metres between two WGS-84 coordinates."""
    R = 6_371_000.0
    p1, p2 = math.radians(la1), math.radians(la2)
    a = (math.sin(math.radians(la2 - la1) / 2) ** 2
         + math.cos(p1) * math.cos(p2) * math.sin(math.radians(lo2 - lo1) / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def nearest_named_wp(lat, lon, names, reach=WP_REACH_DIST_M):
    """Return the scored waypoint within `reach` metres of (lat,lon), else None."""
    if not lat:
        return None
    best, best_d = None, reach
    for name in names:
        wlat, wlon = NAMED_WAYPOINTS[name]
        d = haversine(lat, lon, wlat, wlon)
        if d <= best_d:
            best, best_d = name, d
    return best


def record_arrival(lat, lon, names, arrivals: list):
    """Append a newly-reached scored waypoint to `arrivals` (skip immediate repeats).

    This is OBSERVATION for scoring and progress display — it never issues a
    command or redirects the vehicle.
    """
    here = nearest_named_wp(lat, lon, names)
    if here and (not arrivals or arrivals[-1] != here):
        arrivals.append(here)
    return here


def compress(seq):
    """Collapse consecutive duplicates: [A,A,B,B] -> [A,B]."""
    out = []
    for x in seq:
        if not out or out[-1] != x:
            out.append(x)
    return out


# ---------------------------------------------------------------------------
# Scoring — single classifier shared by all three paradigms
# ---------------------------------------------------------------------------
def score_run(arrivals, expected_order, step_n, step_cap,
              terminal_cmd, fallback=False):
    """Classify a run into the SC scenario vocabulary.

    Returns (outcome, failure_type, note):
      COMPLETED         — waypoints reached in the expected order AND the agent
                          itself issued a terminal RTL/LAND within the step cap.
      TIMEOUT           — step cap exceeded before completion.
      REASONING_FAILURE — skipped / revisited / out-of-order waypoint, ended
                          without a proper terminal command, or the terminal
                          command came from an LLM parse fallback (not a real
                          decision).
    """
    reached = compress(arrivals)

    if step_n > step_cap:
        return "TIMEOUT", "TIMEOUT", f"reached={reached}"

    if fallback:
        # The agent never produced a valid terminal decision — a safe-default
        # RTL fired instead. Not a genuine completion or a chosen abort.
        return "REASONING_FAILURE", "REASONING", f"PARSE_FALLBACK reached={reached}"

    ordered_ok  = reached == list(expected_order)
    terminal_ok = terminal_cmd in ("RTL", "LAND")

    if ordered_ok and terminal_ok:
        return "COMPLETED", "NONE", f"reached={reached}"
    if not ordered_ok:
        return "REASONING_FAILURE", "REASONING", f"ROUTE_ERROR reached={reached}"
    return "REASONING_FAILURE", "REASONING", f"NO_TERMINAL_CMD terminal={terminal_cmd!r}"
