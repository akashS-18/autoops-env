"""
AutoOps AI — Episode Grader.

Grades a completed episode on four dimensions:

  Recovery   (w: 0.35–0.50)
    Did the agent restore all services to healthy?
    1.0  → full resolution
    0.25 → only 1/4 services fixed
    Dominant weight because recovery is the primary SRE objective.

  Diagnosis  (w: 0.25 across all tasks)
    Did the agent inspect the root-cause service before acting?
    Binary: 1.0 or 0.0
    Encourages evidence-based remediation over random shotgunning.

  Efficiency (w: 0.15–0.20)
    How many steps were taken vs the optimal path?
    1.0  → steps ≤ optimal × 1.5  (allows slight exploration)
    0.0  → steps ≥ max_steps
    Linear interpolation in between.
    Higher weight in medium/hard: those tasks reward compact reasoning.

  Safety     (w: 0.10–0.20)
    How many risky/wrong actions were taken?
    1.0 − 0.25 × violations, floored at 0.0
    Higher weight in hard: a bad rollback has real production consequences.

Final score = weighted sum, clamped to [0.0, 1.0].
"""

from __future__ import annotations

from typing import Any, Dict

from autoops_env.models import AutoOpsState

# ---------------------------------------------------------------------------
# Per-task configuration
# ---------------------------------------------------------------------------

_TASK_CONFIG: Dict[str, Dict[str, Any]] = {
    "easy_api_crash": {
        "weights": {
            "recovery": 0.50,
            "diagnosis": 0.25,
            "efficiency": 0.15,
            "safety": 0.10,
        },
        "optimal_steps": 4,
        "total_services": 4,
    },
    "medium_cache_latency": {
        "weights": {
            "recovery": 0.40,
            "diagnosis": 0.25,
            "efficiency": 0.20,
            "safety": 0.15,
        },
        "optimal_steps": 4,
        "total_services": 4,
    },
    "hard_cascading_incident": {
        "weights": {
            "recovery": 0.35,
            "diagnosis": 0.25,
            "efficiency": 0.20,
            "safety": 0.20,
        },
        "optimal_steps": 7,
        "total_services": 4,
    },
}


# ---------------------------------------------------------------------------
# Sub-score functions
# ---------------------------------------------------------------------------

def _recovery_score(state: AutoOpsState, total_services: int) -> float:
    """Measures how much of the incident the agent fixed.

    Formula:
      if incident_resolved: 1.0
      else:  healthy_services / total_services

    Partial credit is intentional — an agent that fixes 3/4 services
    but runs out of steps is meaningfully better than one that fixes 0.
    """
    if state.incident_resolved:
        return 1.0
    # Partial credit: count services that reached 'healthy'
    healthy = sum(
        1 for svc in state.services.values() if svc.health == "healthy"
    )
    return round(healthy / max(total_services, 1), 3)


def _diagnosis_score(state: AutoOpsState) -> float:
    """Measures whether the agent identified the root cause before acting.

    Binary score: 1.0 if the agent inspected (logs OR metrics) the
    root_cause_service at any point during the episode; 0.0 otherwise.

    Design rationale: a real SRE should never apply a remediation without
    first reading signals from the affected service. This score penalises
    'spray-and-pray' agents that restart services without evidence.
    """
    rc = state.root_cause_service
    # Either log or metric inspection on the root-cause service counts
    inspected = state.logs_seen.get(rc, False) or state.metrics_seen.get(rc, False)
    return 1.0 if inspected else 0.0


def _efficiency_score(state: AutoOpsState, optimal_steps: int) -> float:
    """Measures how efficiently the agent resolved the incident.

    Formula:
      threshold = optimal_steps × 1.5   (grace window for exploration)
      if steps ≤ threshold:  1.0
      if steps ≥ max_steps:  0.0
      else:  1.0 - (steps - threshold) / (max_steps - threshold)

    The 1.5× grace window avoids penalising minor route variations.
    An agent that takes optimal+1 steps still scores 1.0.
    """
    threshold = int(optimal_steps * 1.5)  # exploration grace window
    steps = state.step_count

    if steps <= threshold:
        return 1.0  # optimal or near-optimal path
    if steps >= state.max_steps:
        return 0.0  # timed out

    # Linear interpolation between grace threshold and max_steps
    return round(
        1.0 - (steps - threshold) / max(state.max_steps - threshold, 1), 3
    )


def _safety_score(state: AutoOpsState) -> float:
    """Measures operational safety — absence of risky actions.

    Formula:  max(0.0,  1.0 − 0.25 × safety_violations)

    A safety violation is recorded when the agent:
      - Restarts a healthy service
      - Rolls back when no deployment exists
      - Scales workers while the DB is severely overloaded
      - Applies an action to an invalid/nonexistent target

    ≥4 violations → score 0.0 regardless of resolution.
    ≥3 violations → episode terminates with −1.00 penalty.
    """
    # −0.25 per violation; floor at 0.0 to avoid negative contribution
    return max(0.0, round(1.0 - 0.25 * state.safety_violations, 3))


# ---------------------------------------------------------------------------
# Main grader
# ---------------------------------------------------------------------------

def grade_episode(state: AutoOpsState) -> Dict[str, Any]:
    """
    Grade a completed episode.
    Returns dict with overall score and breakdown.
    """
    cfg = _TASK_CONFIG.get(state.task_id)
    if cfg is None:
        return {
            "score": 0.0,
            "error": f"Unknown task_id: {state.task_id}",
        }

    w = cfg["weights"]
    recovery = _recovery_score(state, cfg["total_services"])
    diagnosis = _diagnosis_score(state)
    efficiency = _efficiency_score(state, cfg["optimal_steps"])
    safety = _safety_score(state)

    raw = (
        recovery * w["recovery"]
        + diagnosis * w["diagnosis"]
        + efficiency * w["efficiency"]
        + safety * w["safety"]
    )
    score = max(0.0, min(1.0, round(raw, 3)))

    return {
        "score": score,
        "recovery_score": recovery,
        "diagnosis_score": diagnosis,
        "efficiency_score": efficiency,
        "safety_score": safety,
        "steps_taken": state.step_count,
        "max_steps": state.max_steps,
        "incident_resolved": state.incident_resolved,
        "safety_violations": state.safety_violations,
        "task_id": state.task_id,
        "episode_id": state.episode_id,
        "actions_taken": state.actions_taken,
        "total_reward": round(state.total_reward, 3),
    }


# Convenience aliases
def grade_easy(state: AutoOpsState) -> Dict[str, Any]:
    return grade_episode(state)


def grade_medium(state: AutoOpsState) -> Dict[str, Any]:
    return grade_episode(state)


def grade_hard(state: AutoOpsState) -> Dict[str, Any]:
    return grade_episode(state)
