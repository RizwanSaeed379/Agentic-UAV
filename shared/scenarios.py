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

# The exact goal string for SC2.2 — dual, competing objectives.
SC2_2_GOAL = (
    "Complete the boundary mapping transit from Waypoint Alpha to Waypoint "
    "Bravo and investigate any thermal anomalies detected en route."
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
    "SC2.1": {
        "description":     "Open-ended inspection — no fixed waypoints given",
        "goal":            "Inspect the area and return when done.",
        "anomaly_enabled": False,                      # no disturbance, same as SC1
        "step_cap":        DEFAULT_STEP_CAP,
        "expected_order":  [],                         # nothing to require/sequence
        "score_waypoints": [],                         # nothing to detect arrival at
        "open_ended":      True,                       # <-- gates the prompt/telemetry
                                                        #     stripping in prompts.py and
                                                        #     all three paradigm files
    },
    "SC2.2": {
        "description":     "Dual/competing objectives — transit + anomaly investigation",
        "goal":            SC2_2_GOAL,
        "anomaly_enabled": True,                       # ON by default — the conflict is the point
        "step_cap":        DEFAULT_STEP_CAP,
        "expected_order":  ["WP_ALPHA", "WP_BRAVO"],   # same fixed route as SC1
        "score_waypoints": ["WP_ALPHA", "WP_BRAVO"],
        # open_ended is absent/False — full context (Alpha/Bravo/Home/geofence/
        # cruise alt) is given exactly as in SC1, no stripping.
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


# ---------------------------------------------------------------------------
# Scoring for OPEN-ENDED scenarios (e.g. SC2.1) — score_run above assumes
# there are named waypoints to match arrivals against, which by design there
# aren't here. This checks the one thing that generalizes: did the vehicle
# actually go somewhere non-trivial before returning, rather than just
# idling near home and then RTLing.
# ---------------------------------------------------------------------------
OPEN_ENDED_MEANINGFUL_DIST_M = 400   # heuristic — tune as needed. Must get at
                                     # least this far from HOME at some point
                                     # to count as having inspected a location
                                     # rather than idling and then RTLing.


def score_open_ended_run(max_dist_from_home_m, terminal_cmd, step_n, step_cap,
                         fallback=False):
    """Classify an open-ended-scenario run (e.g. SC2.1).

    Args:
        max_dist_from_home_m: the largest distance from HOME observed at any
            point during the run (paradigm must track this as a running max
            each step — it is not derived from named-waypoint arrivals, since
            there are none in an open-ended scenario).
        terminal_cmd, step_n, step_cap, fallback: same meaning as score_run.

    Returns (outcome, failure_type, note) using the same vocabulary as
    score_run: COMPLETED / TIMEOUT / REASONING_FAILURE, NONE / REASONING.

    The notes field this returns is intentionally minimal (just the max
    distance reached) — a human reviewing the log should still add what the
    agent actually did (e.g. "agent invented waypoints at random coordinates",
    "agent issued only LOITER_TURNS at HOME for 30 steps"), since that's a
    qualitative read of the log this function can't make on its own.
    """
    if step_n > step_cap:
        return "TIMEOUT", "TIMEOUT", f"max_dist_from_home={max_dist_from_home_m:.0f}m"

    if fallback:
        return ("REASONING_FAILURE", "REASONING",
                f"PARSE_FALLBACK max_dist_from_home={max_dist_from_home_m:.0f}m")

    if terminal_cmd not in ("RTL", "LAND"):
        return ("REASONING_FAILURE", "REASONING",
                f"NO_TERMINAL_CMD terminal={terminal_cmd!r} "
                f"max_dist_from_home={max_dist_from_home_m:.0f}m")

    if max_dist_from_home_m < OPEN_ENDED_MEANINGFUL_DIST_M:
        return ("REASONING_FAILURE", "REASONING",
                f"NO_MEANINGFUL_MOVEMENT max_dist_from_home={max_dist_from_home_m:.0f}m "
                f"(threshold={OPEN_ENDED_MEANINGFUL_DIST_M}m)")

    return "COMPLETED", "NONE", f"max_dist_from_home={max_dist_from_home_m:.0f}m"