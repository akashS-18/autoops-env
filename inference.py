"""
AutoOps AI — Deterministic Baseline Agent.

Prints structured output to stdout in the OpenEnv-required format:
  [START] task=<task_id>
  [STEP]  step=<n> reward=<r>
  [END]   task=<task_id> score=<s> steps=<n>
"""

from __future__ import annotations

import os
import sys
from typing import Any, Dict
from openai import OpenAI

# Required Environment Variables for Dashboard Submission Checklist
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/Meta-Llama-3-8B-Instruct")
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
# Easy Task
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


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from server.environment import DevOpsEnvironment

    env = DevOpsEnvironment()
    _run_easy(env)
