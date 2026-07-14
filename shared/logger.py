import json, os
from datetime import datetime, timezone
 
LOG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    'uav_test_results.jsonl'
)
 
def next_run_number(paradigm: str, scenario_id: str) -> int:
    """Return 1 + the number of runs already logged for this paradigm+scenario."""
    if not os.path.exists(LOG_FILE):
        return 1
    count = 0
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (entry.get('paradigm') == paradigm
                    and entry.get('scenario_id') == scenario_id):
                count += 1
    return count + 1


def log_run(
    paradigm:        str,   # 'ReAct' | 'PlanExecute' | 'Reflexion'
    model_primary:   str,   # primary model name
    model_secondary: str,   # executor/critic model, or '' if none
    scenario_id:     str,   # 'SC1' through 'SC6'
    run_number:      int,   # 1, 2, or 3
    outcome:         str,   # 'COMPLETED' | 'FAILED' | 'ABORTED_RTL' | 'TIMEOUT'
    failure_type:    str,   # 'NONE' | 'REASONING' | 'REPLANNING' |
                           # 'CONSTRAINT_VIOLATION' | 'MISINTERPRETATION' |
                           # 'COMMUNICATION'
    waypoints_visited: list,  # e.g. ['WP_ALPHA', 'WP_BRAVO']
    anomaly_response:  str,   # 'LOITER_TURNS' | 'RTL' | 'CONTINUE' | 'NONE'
    llm_calls:         int,   # total LLM calls made during run
    duration_seconds:  float, # wall-clock time for the run
    telemetry_final:   dict,  # last telemetry snapshot
    notes:             str = '',
) -> None:
    entry = {
        'timestamp':        datetime.now(timezone.utc).isoformat(),
        'paradigm':         paradigm,
        'model_primary':    model_primary,
        'model_secondary':  model_secondary,
        'scenario_id':      scenario_id,
        'run_number':       run_number,
        'outcome':          outcome,
        'failure_type':     failure_type,
        'waypoints_visited':waypoints_visited,
        'anomaly_response': anomaly_response,
        'llm_calls':        llm_calls,
        'duration_seconds': round(duration_seconds, 2),
        'telemetry_final':  telemetry_final,
        'notes':            notes,
    }
    with open(LOG_FILE, 'a', encoding='utf-8') as f:
        f.write(json.dumps(entry) + '\n')
    print(f'[LOG] {paradigm} | {scenario_id} | Run {run_number} | {outcome}')