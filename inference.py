"""
AutoOps AI — Deterministic Baseline Agent.

Hard-coded if/then logic for all 3 tasks.
Expected scores:  Easy ≈ 0.78 | Medium ≈ 0.65 | Hard ≈ 0.52

This exists to prove the grader works and give reproducible benchmark numbers.
"""

from __future__ import annotations

from typing import Any, Dict, List

from models import AutoOpsAction, AutoOpsObservation
from graders.grader import grade_episode


def _act(env, action_type: str, target: str | None = None) -> AutoOpsObservation:
    """Shortcut: build an action and step the environment."""
    action = AutoOpsAction(action_type=action_type, target=target)
    return env.step(action)


# ---------------------------------------------------------------------------
# Per-task baselines
# ---------------------------------------------------------------------------

def _run_easy(env) -> Dict[str, Any]:
    """Easy: API crash loop."""
    obs = env.reset("easy_api_crash")
    episode_id = env.state.episode_id

    # Step 1: inspect logs
    obs = _act(env, "inspect_logs", "api")

    # Step 2: if crash_loop info → restart
    obs = _act(env, "restart_service", "api")

    # Step 3-4: wait for recovery
    obs = _act(env, "wait")
    if not obs.done:
        obs = _act(env, "wait")

    return grade_episode(env.state)


def _run_medium(env) -> Dict[str, Any]:
    """Medium: cache miss storm."""
    obs = env.reset("medium_cache_latency")
    episode_id = env.state.episode_id

    # Step 1: inspect API metrics
    obs = _act(env, "inspect_metrics", "api")

    # Step 2: inspect cache metrics (root cause)
    obs = _act(env, "inspect_metrics", "cache")

    # Step 3: clear cache
    obs = _act(env, "clear_cache", "cache")

    # Step 4: wait for recovery propagation
    obs = _act(env, "wait")

    # Step 5: extra wait if not resolved
    if not obs.done:
        obs = _act(env, "wait")

    return grade_episode(env.state)


def _run_hard(env) -> Dict[str, Any]:
    """Hard: cascading deployment failure."""
    obs = env.reset("hard_cascading_incident")
    episode_id = env.state.episode_id

    # Step 1: inspect API metrics
    obs = _act(env, "inspect_metrics", "api")

    # Step 2: inspect API logs (root cause)
    obs = _act(env, "inspect_logs", "api")

    # Step 3: rollback deployment
    obs = _act(env, "rollback_deployment", "api")

    # Step 4: wait for 2-step propagation
    obs = _act(env, "wait")

    # Step 5: wait for propagation to finish
    obs = _act(env, "wait")

    # Step 6: inspect worker metrics
    obs = _act(env, "inspect_metrics", "worker")

    # Step 7: scale workers if queue is high
    obs = _act(env, "scale_worker", "worker")

    # Step 8: final wait
    if not obs.done:
        obs = _act(env, "wait")

    return grade_episode(env.state)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline_all(env=None) -> List[Dict[str, Any]]:
    """
    Run baseline on all 3 tasks using the provided environment.
    Returns list of grader results.
    """
    if env is None:
        from server.environment import DevOpsEnvironment
        env = DevOpsEnvironment()

    results = []
    for name, runner in [
        ("easy_api_crash", _run_easy),
        ("medium_cache_latency", _run_medium),
        ("hard_cascading_incident", _run_hard),
    ]:
        result = runner(env)
        results.append(result)

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from server.environment import DevOpsEnvironment

    env = DevOpsEnvironment()
    results = run_baseline_all(env)

    print("\n" + "=" * 60)
    print("  AutoOps AI — Deterministic Baseline Results")
    print("=" * 60)

    for r in results:
        print(f"\n  Task: {r['task_id']}")
        print(f"    Score:      {r['score']:.3f}")
        print(f"    Recovery:   {r['recovery_score']:.3f}")
        print(f"    Diagnosis:  {r['diagnosis_score']:.3f}")
        print(f"    Efficiency: {r['efficiency_score']:.3f}")
        print(f"    Safety:     {r['safety_score']:.3f}")
        print(f"    Steps:      {r['steps_taken']}")
        print(f"    Resolved:   {r['incident_resolved']}")
        print(f"    Actions:    {r['actions_taken']}")

    print("\n" + "=" * 60)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average Score: {avg:.3f}")
    print("=" * 60 + "\n")
