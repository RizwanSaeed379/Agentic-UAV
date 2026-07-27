# Plan — Autonomous Decision-Making Refactor

> **STATUS: IMPLEMENTED** (2026-07-27). All 6 work items done: `shared/scenarios.py`
> (new), `shared/prompts.py` (goal-based), ReAct / Plan-Execute / Reflexion
> de-scaffolded, `main.py` anomaly-override prompt, `agent_loop.py` fallback tag.
> Verified: all files compile + import; scoring unit-checked. NOT yet run against
> live SITL + Ollama (requires the hardware/endpoint — see §11).


**Goal:** Turn the three paradigms (ReAct, Plan-Execute, Reflexion) from scripted
pipelines into genuinely autonomous LLM agents. The LLM must own the *mission*
(which destination next, how to handle disturbances, when the mission is done).
Python keeps only the *vehicle* (bring-up, MAVLink I/O, arrival sensing, a logged
safety floor). Prompts become **goal-based**, with mission context, geofence
bounds, and cruise altitude supplied through `prompts.py`. Anomaly injection is
**toggleable** (scenario default + per-run override).

Scope confirmed with user:
- **All three paradigms.**
- **Anomaly toggle:** scenario config sets the default; a launch-time prompt can override per run.
- **Safety floor kept**, but logged explicitly as overrides so it never pollutes the reasoning score.
- **SC1 fully built now**; the scenario mechanism is structured so SC2–SC6 slot in later (no placeholder guesses).

---

## 0. The one principle behind every change

> **Python may own the vehicle. The LLM must own the mission. Nothing in between.**

- **Python (keep):** connect, GPS wait, arm, mode changes, mission-item upload
  handshake, takeoff, segment flying, arrival detection, battery/step safety floor.
- **LLM (hand back):** the *sequence* of waypoints, disturbance response, and the
  decision that the mission is complete (issue RTL/LAND).
- **Removed everywhere:** anything that forces, reverts, or overrides the LLM's
  *decision* — phase machines that fly legs for it, "output exactly these steps",
  executor "you CANNOT change the command" reverts, deterministic RESUME legs,
  scripted navigation phases.

---

## 1. HARD CONSTRAINT — code that must NOT change

Per the user, all fundamental pymavlink plumbing stays **byte-for-byte identical**
so flight behavior in SITL is unaffected:

- **Connection:** `_connect()` (all three).
- **Mode changes:** `set_mode()` incl. the `telemetry_paused()` exclusive-socket logic.
- **Arming:** `_arm()` and its arm-confirm wait.
- **GPS wait:** `_wait_gps()` / `_wait_for_gps()`.
- **Mission upload handshake:** `_send_items()` / `_send_mission_items()` — the
  MISSION_COUNT → MISSION_REQUEST → MISSION_ITEM_INT → MISSION_ACK loop, the
  "no MISSION_CLEAR_ALL in flight" rule, retries, `telemetry_paused()` usage.
- **Mission item structure:** `_build_segment()`, takeoff item layout, seq-0-is-HOME
  reservation, `MAV_FRAME_GLOBAL_RELATIVE_ALT`, acceptance radius (200m).
- **Segment/loiter flight:** `_fly_segment()`, `_fly_loiter()`/`_wait_loiter()`,
  `_force_auto()`, `mission_set_current_send` re-sequencing after arrival.

We change **control flow and prompts**, never the MAVLink transport layer.

---

## 2. New shared scenario mechanism — `shared/scenarios.py` (new file)

Single source of truth all three paradigms read. Route stays LLM-owned; these
fields gate the *disturbance*, the *step cap*, and *scoring*.

```python
_NAMED_WAYPOINTS = { "HOME":(...), "WP_ALPHA":(...), "WP_BRAVO":(...),
                     "ANOMALY":(...), "MIDPOINT":(...) }   # from shared.tools

SCENARIOS = {
  "SC1": {
    "description":     "Normal Execution baseline — no disturbance",
    "anomaly_enabled": False,          # default; runtime prompt may override
    "step_cap":        30,
    "expected_order":  ["WP_ALPHA", "WP_BRAVO"],   # required visit order
    "score_waypoints": ["WP_ALPHA", "WP_BRAVO"],   # arrivals tracked for scoring
  },
  # SC2–SC6 added when specced. Mechanism below is generic.
}

def get_scenario(scenario_id) -> dict         # SC1 fallback + warning if unknown
def resolve_anomaly(cfg, override) -> bool    # override in {None,True,False}
```

**Shared scoring helpers** (used by all three so outcomes are comparable):

```python
def nearest_named_wp(lat, lon, names, reach=200) -> str|None
def record_arrival(lat, lon, names, arrivals) -> None   # observation, not control
def compress(seq) -> list                                # [A,A,B]->[A,B]
def score_run(arrivals, expected_order, step_n, step_cap, terminal_cmd)
      -> (outcome, failure_type)
```

`score_run` classification (matches the SC1 spec vocabulary exactly):
- **COMPLETED** — reached == expected_order **and** agent issued terminal RTL/LAND within cap.
- **TIMEOUT** — `step_n > step_cap` before completion.
- **REASONING_FAILURE** — skipped / revisited / out-of-order waypoint, or ended
  without a proper terminal command.

Failure-type field: `NONE` / `TIMEOUT` / `REASONING`. Safety overrides are recorded
in `notes` (e.g. `SAFETY_RTL`) and only turn a run into a failure if the route was
actually incomplete — a dead-battery RTL is never scored as reasoning.

---

## 3. Goal-based prompts + context — `shared/prompts.py` (rewrite)

Replace the phase/step-forcing prompts with **goal-based role prompts**, and inject
context (mission objective, waypoint coords, geofence bounds, cruise altitude)
through a builder so it lives in `prompts.py`, not scattered in the paradigm files.

```python
def build_mission_context(cfg) -> str:
    """Objective, waypoint coords, GEOFENCE bounds, CRUISE_ALT, and a note that
       live telemetry (position/battery/distances/visited) arrives each step."""
```

Per-paradigm prompt rewrites:

- **REACT_SYSTEM_PROMPT** → goal-driven. State the goal ("visit WP_ALPHA then
  WP_BRAVO in order, then return home and land"), list coords + geofence +
  cruise alt, explain the progress fields it will see (`waypoints_visited`,
  `waypoints_remaining`, distances, `steps_remaining`, `anomaly_active`), and a
  short decision *policy* (safety first; visit remaining in order; don't re-visit;
  RTL only when done; investigate an anomaly only if one is active). **Remove**
  the `if phase == ANOMALY_INVESTIGATION` rulebook.

- **PLAN_EXECUTE_SYSTEM_PROMPT (planner)** → "produce a complete plan that
  achieves the mission goal." **Remove** the run-time injected line *"Output
  exactly these 4 steps and nothing else"* (`plan_execute.py` ~583-588) and the
  spoon-fed numbered list. Give context + goal; the plan is genuinely the LLM's.

- **Executor prompt (`_EXECUTOR_SYSTEM`)** → reframe from "you CANNOT change the
  command / NEVER change the type" to "validate this step against live telemetry;
  confirm, adjust, or abort for safety." It keeps a real validation role but is no
  longer a rubber stamp forced back to the plan (the Python reverts are removed in §4B).

- **REFLEXION_SYSTEM_PROMPT (actor)** → goal-driven step-wise actor that reads
  memory and current state and picks the next action across the whole mission
  (not one canned decision). Memory format + critic reflection retained.

Coordinates remain in the prompt (they're fixed constants) — we are testing
*sequencing / disturbance-handling / completion reasoning*, not coordinate discovery.
(Flag: confirm this framing is acceptable for the paper.)

---

## 4. Per-paradigm de-scaffolding

### 4A. ReAct — `paradigms/react_agent.py`

| Action | Location (today) | Replacement |
|---|---|---|
| Drop scripted Home→WP_ALPHA leg | takeoff mission `381-397` (seq-2 WP_ALPHA item) | Upload **HOME + TAKEOFF only**; climb to cruise |
| Remove "wait for WP_ALPHA" phase | `399-422` | Replace with **airborne-readiness** wait (`_agl >= 0.8*cruise`) then `LOITER` hold while LLM thinks |
| Remove phase state machine | `460-468` | Phases gone; loop is telemetry→LLM→execute |
| **Remove deterministic RESUME leg** (flies to Bravo + RTL for the LLM) | `511-521` | Deleted — the LLM must choose Bravo and choose to finish |
| Gate anomaly firing | `471-476` | Fire `trigger_anomaly()` **only if** resolved anomaly flag is True |
| Replace outcome logic | `588-604` | `score_run(...)` → COMPLETED / REASONING_FAILURE / TIMEOUT |

New ReAct loop: each step read telemetry, `record_arrival` (observation), enforce
safety floor + step cap, inject progress into telemetry, call `agent_step`, execute
the chosen command, re-check arrival, break on terminal RTL/LAND. **Keep:** connect,
relocate_home, UAVState, GPS wait, takeoff upload, arm, `set_mode`, `_fly_segment`,
`execute_flight_command`, `_send_items`.

### 4B. Plan-Execute — `paradigms/plan_execute.py`

| Action | Location (today) | Replacement |
|---|---|---|
| Remove "output exactly these 4 steps" spoon-feeding | run ~`583-588` | Goal + context only; planner produces the plan |
| **Remove executor revert logic** | type-revert `613-616`; coord/alt/loiter revert `626-635` | Trust executor output; keep only a genuine **safety** check (alt≤5 crash-guard, battery→RTL) as a *logged override*, not a silent revert-to-plan |
| Reframe executor forcing rules | `_EXECUTOR_SYSTEM` `431-449` | Validation role, not "you CANNOT change" |
| Gate anomaly firing | `797` | Only if resolved flag True |
| Keep genuine mid-flight replan | `795-808…` | Planner truly re-plans (goal-based), not re-forced |
| Replace outcome logic | ~`1011` | `score_run(...)` |

**Keep:** `call_planner`/`call_executor` structure, battery override in
`call_executor` (`459-463`, it's safety), full-mission build+upload (`655-716`),
wp_seq monitoring, `_fly_segment`/`_fly_loiter`, `_force_auto`. Note: in
Plan-Execute the LLM's "route ownership" = the *plan it generates* is genuinely its
own and is no longer reverted toward a canned answer.

### 4C. Reflexion — `paradigms/reflexion_agent.py`  (DECIDED — full actor loop; largest behavioral change)

| Action | Location (today) | Replacement |
|---|---|---|
| Remove scripted Phase-1/2 navigation | `862-874` | LLM chooses waypoints |
| Move + gate anomaly (was pre-triggered unconditionally) | `870` | Proximity-based, only if resolved flag True |
| **Replace single-decision + scripted follow-through** | `876-934` | Real **actor loop**: read memory + telemetry → next action → execute → observe → repeat, until terminal or step cap |
| Remove scripted RTL | `947-950` | LLM issues terminal RTL/LAND |
| Replace outcome mapping | `972-994` | `score_run(...)` |

**Keep:** memory read/write, `get_attempt_number`, connection, mission item build +
upload, arm, `set_mode`, and the **Critic** call + reflection write (cross-attempt
learning is the point of Reflexion). **Risk flagged:** single-decision existed
because a full multi-step actor stalled on small local models. Making it a real loop
may surface genuine instability — which is now *data* (a Reflexion failure mode),
not a bug, but expect more REASONING_FAILUREs on weak models.

---

## 5. Safety floor (kept, logged — not reasoning)

Applied uniformly, always logged with a `[SAFETY]` tag and recorded in `notes`:
- `battery_pct < 15` → forced RTL override.
- `step_n > step_cap` → TIMEOUT.
- (Optional crash-guard) reject NAV with `alt <= 5` when the plan had a real altitude.

**Geofence (DECIDED — advisory only):** bounds are provided to the LLM as *context*
in the prompt, and the `geofence_check` tool stays available for the agent to call.
It is **advisory** — not a hard flight-side block — so geofence handling is part of
what the agent's reasoning is judged on (a breach shows up as REASONING_FAILURE).

---

## 6. Runtime wiring — `main.py`

- Keep banner, preflight checklist, health checks, scenario select.
- After scenario select, add: **"Inject anomaly this run? [Enter=scenario default / y / n]"** → resolves to `override ∈ {None, True, False}`.
- Pass `scenario_id` **and** the resolved anomaly flag into each
  `run_*_mission(scenario_id, run_number, anomaly_override=...)`.

---

## 7. Honesty fix — `shared/agent_loop.py`  (DECIDED — include now)

Today a parse failure / max-rounds silently returns `RTL` (`_SAFE_DEFAULT`, `44`,
`143-151`, `229-234`). Under real scoring that RTL is indistinguishable from a
*chosen* abort. **We tag the fallback** (`"fallback": True`) so scoring can separate
*malformed-output failure* from *bad-decision failure*. Scoring treats a
`fallback`-tagged terminal command as a REASONING_FAILURE (with `notes=PARSE_FALLBACK`),
never as a clean COMPLETED. Touches shared code used by all three paradigms.

---

## 8. Files touched

| File | Change |
|---|---|
| `shared/scenarios.py` | **new** — SCENARIOS config, resolver, shared scoring |
| `shared/prompts.py` | rewrite — goal-based prompts + `build_mission_context` |
| `paradigms/react_agent.py` | de-scaffold route (§4A) |
| `paradigms/plan_execute.py` | remove forcing/reverts (§4B) |
| `paradigms/reflexion_agent.py` | actor loop (§4C) |
| `main.py` | anomaly-override prompt + pass-through |
| `shared/agent_loop.py` | optional fallback tagging (§7) |
| `shared/logger.py` | **unchanged** (vocab already supports the outcomes) |
| MAVLink plumbing (all) | **unchanged** (§1) |

---

## 9. How SC1 runs after this

Take off over HOME → LLM sees `visited: []` → NAV to WP_ALPHA → arrival recorded →
`visited: [WP_ALPHA]` → NAV to WP_BRAVO → `visited: [WP_ALPHA, WP_BRAVO]` → RTL.
Score = **COMPLETED**. If the model loops on Alpha, jumps to Bravo first, or RTLs
early → **REASONING_FAILURE**. If it never finishes in 30 steps → **TIMEOUT**.
Anomaly stays off (SC1 default), so `trigger_anomaly()` is never called.

Run: start SITL + Ollama (fix `_BASE_URL` first — see §11), `python main.py` → A/B/C
→ checklist → SC1 → anomaly prompt (Enter for default = off).

---

## 10. Risks / things to accept going in

1. **Weak local models may genuinely fail SC1** once scaffolding is gone. That
   failure is the finding, not a bug — but if `llama3`/`qwen2.5`/`mistral` fail
   constantly, it's an argument for a stronger model.
2. **Reflexion multi-step instability** (§4C) — biggest unknown.
3. **Fixed-wing holding between decisions** — handled by LOITER between commands and
   the proven segment-hold behavior; no plumbing change.
4. **Prompt still contains coords** — sequencing is tested, not coordinate discovery.

---

## 11. Pre-existing issue to flag (not in scope unless you want it)

`shared/llm_utils.py:38` — `_BASE_URL` points at a Cloudflare tunnel, not
`localhost:11434`, despite the "local running code" commit. The preflight Ollama
check in `main.py` therefore doesn't reflect where requests actually go. Decide
which endpoint SC1 runs against before testing.

---

## 12. Decisions locked (from review)

1. **Scope** — all three paradigms. ✅
2. **Anomaly toggle** — scenario default + per-run runtime override. ✅
3. **Safety floor** — kept, logged as overrides. ✅
4. **Scenarios** — SC1 built now; mechanism ready for SC2–SC6. ✅
5. **Reflexion** — convert to a genuine multi-step actor loop (§4C). ✅
6. **Geofence** — advisory only (§5). ✅
7. **agent_loop fallback tagging** — include now (§7). ✅

### Still assumed (veto during review if wrong)

- **Coords stay in the prompt.** We test sequencing / disturbance-handling /
  completion reasoning, not coordinate discovery. If you'd rather the agent derive
  or confirm coordinates via tools, say so and I'll adjust §3.
- **`_BASE_URL` endpoint (§11)** is yours to set before running; not changed by this refactor.
