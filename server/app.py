"""
AutoOps AI — FastAPI application.

Endpoints:
  Standard:   GET /health, POST /reset, POST /step, GET /state, GET /docs
  Required:   GET /tasks, GET /grader, GET /baseline
"""

from __future__ import annotations

import copy
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from autoops_env.models import AutoOpsAction, AutoOpsObservation
from autoops_env.server.environment import DevOpsEnvironment
from autoops_env.graders.grader import grade_episode
from autoops_env.tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AutoOps AI",
    description=(
        "OpenEnv environment for autonomous DevOps incident response. "
        "An AI agent inspects production signals, chooses remediation actions, "
        "and restores service across three escalating incident scenarios."
    ),
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# single shared environment instance
env = DevOpsEnvironment()


# ---------------------------------------------------------------------------
# Request / response helpers
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    task_id: str = "easy_api_crash"


class TaskInfo(BaseModel):
    task_id: str
    description: str
    difficulty: str
    max_steps: int
    optimal_steps: int


# ---------------------------------------------------------------------------
# Standard endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/reset", response_model=AutoOpsObservation)
def reset(req: ResetRequest):
    """Start a new episode for the given task_id."""
    try:
        obs = env.reset(req.task_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.post("/step", response_model=AutoOpsObservation)
def step(action: AutoOpsAction):
    """Submit an action and receive the next observation."""
    try:
        obs = env.step(action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return obs


@app.get("/state")
def get_state():
    """Return the full internal AutoOpsState (debug / grader use)."""
    if env.state is None:
        raise HTTPException(status_code=400, detail="No active episode. Call /reset first.")
    state_dict = env.state.model_dump()
    state_dict["time_remaining"] = env.state.time_remaining
    state_dict["business_impact_score"] = env.state.business_impact_score
    return state_dict


# ---------------------------------------------------------------------------
# Required custom endpoints
# ---------------------------------------------------------------------------

TASK_DESCRIPTIONS = {
    "easy_api_crash": TaskInfo(
        task_id="easy_api_crash",
        description=(
            "API Crash Loop — the API service is stuck in a crash loop due to OOM. "
            "The agent must diagnose the crash, restart the service, and confirm recovery."
        ),
        difficulty="easy",
        max_steps=15,
        optimal_steps=4,
    ),
    "medium_cache_latency": TaskInfo(
        task_id="medium_cache_latency",
        description=(
            "Cache Miss Storm — the cache hit rate has dropped to 12%, causing API latency "
            "to spike to 4200ms. The agent must identify the cache as root cause, clear it, "
            "and wait for recovery to propagate."
        ),
        difficulty="medium",
        max_steps=20,
        optimal_steps=4,
    ),
    "hard_cascading_incident": TaskInfo(
        task_id="hard_cascading_incident",
        description=(
            "Cascading Deployment Failure — a bad API deployment 3 hours ago has caused "
            "cascading degradation across db and worker services. The agent must rollback "
            "the deployment, wait for propagation, then scale workers to drain the queue."
        ),
        difficulty="hard",
        max_steps=25,
        optimal_steps=7,
    ),
}


@app.get("/tasks")
def list_tasks():
    """Return all available tasks with descriptions and difficulty."""
    return {"tasks": list(TASK_DESCRIPTIONS.values())}


@app.get("/grader")
def grader(episode_id: str = Query(..., description="Episode ID to grade")):
    """
    Grade a completed episode.
    Returns score breakdown: recovery, diagnosis, efficiency, safety.
    """
    state = env.get_episode(episode_id)
    if state is None:
        raise HTTPException(
            status_code=404,
            detail=f"Episode '{episode_id}' not found. Run /reset + /step first.",
        )
    result = grade_episode(state)
    return result


@app.get("/baseline")
def run_baseline():
    """Run the deterministic baseline agent on all 3 tasks and return scores."""
    from autoops_env.baseline import run_baseline_all
    results = run_baseline_all(env)
    return {"baseline_results": results}
