"""
AutoOps AI — Deterministic Baseline Agent.

Hard-coded if/then logic for all 3 tasks.
Expected scores:  Easy ≈ 0.78 | Medium ≈ 0.65 | Hard ≈ 0.52

Prints structured output to stdout in the OpenEnv-required format:
  [START] task=<task_id>
  [STEP]  step=<n> reward=<r>
  [END]   task=<task_id> score=<s> steps=<n>
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict, List
from openai import OpenAI

# Required Environment Variables for Dashboard Submission Checklist
API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")
HF_TOKEN = os.getenv("HF_TOKEN")

from models import AutoOpsAction, AutoOpsObservation
from graders.grader import grade_episode


# ---------------------------------------------------------------------------
# Structured-output helpers (validator requirement)
# ---------------------------------------------------------------------------

def _log_start(task_id: str) -> None:
    print(f"[START] task={task_id}", flush=True)


def _log_step(step: int, reward: float) -> None:
    print(f"[STEP] step={step} reward={round(reward, 4)}", flush=True)


def _log_end(task_id: str, score: float, steps: int) -> None:
    print(f"[END] task={task_id} score={round(score, 4)} steps={steps}", flush=True)


# ---------------------------------------------------------------------------
# Environment interaction helper
# ---------------------------------------------------------------------------

def _act(env, action_type: str, target: str | None = None) -> AutoOpsObservation:
    """Shortcut: build an action and step the environment."""
    action = AutoOpsAction(action_type=action_type, target=target)
    return env.step(action)


# ---------------------------------------------------------------------------
# Per-task baselines
# ---------------------------------------------------------------------------

def _run_easy(env) -> Dict[str, Any]:
    """Easy: API crash loop."""
    task_id = "easy_api_crash"
    _log_start(task_id)

    obs = env.reset(task_id)
    step = 0

    # Step 1: inspect logs
    obs = _act(env, "inspect_logs", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 2: restart service
    obs = _act(env, "restart_service", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 3: wait for recovery
    obs = _act(env, "wait")
    step += 1
    _log_step(step, obs.reward)

    # Step 4: extra wait if not done
    if not obs.done:
        obs = _act(env, "wait")
        step += 1
        _log_step(step, obs.reward)

    result = grade_episode(env.state)
    _log_end(task_id, result["score"], result["steps_taken"])
    return result


def _run_medium(env) -> Dict[str, Any]:
    """Medium: cache miss storm."""
    task_id = "medium_cache_latency"
    _log_start(task_id)

    obs = env.reset(task_id)
    step = 0

    # Step 1: inspect API metrics
    obs = _act(env, "inspect_metrics", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 2: inspect cache metrics (root cause)
    obs = _act(env, "inspect_metrics", "cache")
    step += 1
    _log_step(step, obs.reward)

    # Step 3: clear cache
    obs = _act(env, "clear_cache", "cache")
    step += 1
    _log_step(step, obs.reward)

    # Step 4: wait for recovery propagation
    obs = _act(env, "wait")
    step += 1
    _log_step(step, obs.reward)

    # Step 5: extra wait if not resolved
    if not obs.done:
        obs = _act(env, "wait")
        step += 1
        _log_step(step, obs.reward)

    result = grade_episode(env.state)
    _log_end(task_id, result["score"], result["steps_taken"])
    return result


def _run_hard(env) -> Dict[str, Any]:
    """Hard: cascading deployment failure."""
    task_id = "hard_cascading_incident"
    _log_start(task_id)

    obs = env.reset(task_id)
    step = 0

    # Step 1: inspect API metrics
    obs = _act(env, "inspect_metrics", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 2: inspect API logs (root cause)
    obs = _act(env, "inspect_logs", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 3: rollback deployment
    obs = _act(env, "rollback_deployment", "api")
    step += 1
    _log_step(step, obs.reward)

    # Step 4: wait for 2-step propagation
    obs = _act(env, "wait")
    step += 1
    _log_step(step, obs.reward)

    # Step 5: wait for propagation to finish
    if not obs.done:
        obs = _act(env, "wait")
        step += 1
        _log_step(step, obs.reward)

    # Step 6: inspect worker metrics
    if not obs.done:
        obs = _act(env, "inspect_metrics", "worker")
        step += 1
        _log_step(step, obs.reward)

    # Step 7: scale workers if queue is high
    if not obs.done:
        obs = _act(env, "scale_worker", "worker")
        step += 1
        _log_step(step, obs.reward)

    # Step 8: final wait
    if not obs.done:
        obs = _act(env, "wait")
        step += 1
        _log_step(step, obs.reward)

    result = grade_episode(env.state)
    _log_end(task_id, result["score"], result["steps_taken"])
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_baseline_all(env=None) -> List[Dict[str, Any]]:
    """
    Run baseline on all 3 tasks using the provided environment.
    Prints structured output for each task and returns list of grader results.
    """
    if env is None:
        from server.environment import DevOpsEnvironment
        env = DevOpsEnvironment()

    results = []
    for runner in [_run_easy, _run_medium, _run_hard]:
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

    # Human-readable summary (does NOT interfere with structured output above)
    print("", flush=True)
    print("=" * 60, flush=True)
    print("  AutoOps AI — Deterministic Baseline Results", flush=True)
    print("=" * 60, flush=True)

    for r in results:
        print(f"\n  Task:      {r['task_id']}", flush=True)
        print(f"    Score:      {r['score']:.3f}", flush=True)
        print(f"    Recovery:   {r['recovery_score']:.3f}", flush=True)
        print(f"    Diagnosis:  {r['diagnosis_score']:.3f}", flush=True)
        print(f"    Efficiency: {r['efficiency_score']:.3f}", flush=True)
        print(f"    Safety:     {r['safety_score']:.3f}", flush=True)
        print(f"    Steps:      {r['steps_taken']}", flush=True)
        print(f"    Resolved:   {r['incident_resolved']}", flush=True)
        print(f"    Actions:    {r['actions_taken']}", flush=True)

    print("", flush=True)
    print("=" * 60, flush=True)
    avg = sum(r["score"] for r in results) / len(results)
    print(f"  Average Score: {avg:.3f}", flush=True)
    print("=" * 60, flush=True)
