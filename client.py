"""
AutoOps AI — Typed API Client.

Mirrors the OpenEnv guide pattern for a clean, professional client.
Works against both local and deployed endpoints.

Usage:
    from client import AutoOpsClient

    client = AutoOpsClient("http://localhost:8000")
    obs = client.reset("easy_api_crash")
    obs = client.step("inspect_logs", "api")
    print(obs.summary)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx

from models import AutoOpsAction, AutoOpsObservation


class AutoOpsClient:
    """Typed HTTP client for the AutoOps environment API."""

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    # -- standard endpoints -------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """GET /health"""
        r = self._client.get("/health")
        r.raise_for_status()
        return r.json()

    def reset(self, task_id: str = "easy_api_crash") -> AutoOpsObservation:
        """POST /reset — start a new episode."""
        r = self._client.post("/reset", json={"task_id": task_id})
        r.raise_for_status()
        return AutoOpsObservation(**r.json())

    def step(
        self,
        action_type: str,
        target: Optional[str] = None,
    ) -> AutoOpsObservation:
        """POST /step — submit an action."""
        action = AutoOpsAction(action_type=action_type, target=target)
        r = self._client.post("/step", json=action.model_dump())
        r.raise_for_status()
        return AutoOpsObservation(**r.json())

    def state(self) -> Dict[str, Any]:
        """GET /state — full internal state (debugging)."""
        r = self._client.get("/state")
        r.raise_for_status()
        return r.json()

    # -- custom endpoints ---------------------------------------------------

    def tasks(self) -> List[Dict[str, Any]]:
        """GET /tasks — list all available tasks."""
        r = self._client.get("/tasks")
        r.raise_for_status()
        return r.json()["tasks"]

    def grader(self, episode_id: str) -> Dict[str, Any]:
        """GET /grader — grade a completed episode."""
        r = self._client.get("/grader", params={"episode_id": episode_id})
        r.raise_for_status()
        return r.json()

    def baseline(self) -> List[Dict[str, Any]]:
        """GET /baseline — run baseline agent on all tasks."""
        r = self._client.get("/baseline")
        r.raise_for_status()
        return r.json()["baseline_results"]

    # -- lifecycle ----------------------------------------------------------

    def close(self) -> None:
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
