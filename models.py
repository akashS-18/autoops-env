"""
AutoOps AI — Pydantic models for the DevOps incident-response environment.

Three core models:
  • AutoOpsAction       — what the agent sends each step
  • AutoOpsObservation  — what the agent receives back
  • AutoOpsState        — full internal state (never exposed to the agent directly)

Action cost reference (informational; does not reduce reward directly):
  Inspection actions   → free   (observe only)
  restart_service      → low    (safe if justified; 10% failure risk)
  clear_cache          → low    (idempotent, non-destructive)
  scale_worker         → medium (resource cost; risky if DB is overloaded)
  rollback_deployment  → high   (irreversible; 2-step propagation delay)
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# Action cost table — relative operational cost/risk of each action type.
# Used for transparency, displayed via /tasks metadata.
# ---------------------------------------------------------------------------

ACTION_COSTS: Dict[str, str] = {
    "inspect_logs": "free",
    "inspect_metrics": "free",
    "wait": "free",
    "ack_alert": "free",
    "restart_service": "low",
    "clear_cache": "low",
    "scale_worker": "medium",
    "rollback_deployment": "high",
}


# ---------------------------------------------------------------------------
# Action — agent → environment
# ---------------------------------------------------------------------------

class AutoOpsAction(BaseModel):
    """Action submitted by the agent at each step."""

    action_type: str = Field(
        ...,
        description=(
            "One of: inspect_logs, inspect_metrics, restart_service, "
            "clear_cache, scale_worker, rollback_deployment, ack_alert, wait"
        ),
    )
    target: Optional[str] = Field(
        None,
        description="Service name, alert name, or None (for 'wait').",
    )


# ---------------------------------------------------------------------------
# Observation — environment → agent
# ---------------------------------------------------------------------------

class AutoOpsObservation(BaseModel):
    """The slice of state the agent is allowed to see."""

    summary: str = Field(
        ...,
        description="Plain-English description of the current system status.",
    )
    visible_services: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Service name → {health, latency_ms, error_rate, status}.",
    )
    recent_alerts: List[str] = Field(
        default_factory=list,
        description="Active alert names.",
    )
    available_actions: List[str] = Field(
        default_factory=list,
        description="Valid action_type strings for this step.",
    )
    reward: float = Field(
        0.0,
        description="Reward from the *previous* action (0.0 on first observation).",
    )
    done: bool = Field(
        False,
        description="True when incident is resolved or step limit reached.",
    )


# ---------------------------------------------------------------------------
# Service snapshot (embedded inside AutoOpsState.services)
# ---------------------------------------------------------------------------

class ServiceState(BaseModel):
    """Runtime state of a single service."""

    health: str = "healthy"            # healthy | degraded | down | recovering
    latency_ms: int = 50
    error_rate: float = 0.0            # 0.0 – 100.0 (percentage)
    status: str = "running"            # free-form label, e.g. crash_loop, miss_storm
    # optional extended fields (used per scenario)
    hit_rate: Optional[float] = None   # cache only
    query_rate: Optional[int] = None   # db only
    query_latency: Optional[int] = None  # db only
    queue_depth: Optional[int] = None  # worker only
    worker_count: Optional[int] = None  # worker only
    recent_deploy: Optional[str] = None  # timestamp string if a deploy happened


# ---------------------------------------------------------------------------
# Full internal state
# ---------------------------------------------------------------------------

class AutoOpsState(BaseModel):
    """Complete environment state — kept server-side, never sent to the agent."""

    # identifiers
    episode_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    task_id: str = "easy_api_crash"

    # step tracking
    step_count: int = 0
    max_steps: int = 15

    # resolution
    incident_resolved: bool = False
    root_cause: str = ""
    root_cause_service: str = ""       # service that holds the root cause

    # time pressure
    @property
    def time_remaining(self) -> int:
        return max(0, self.max_steps - self.step_count)

    @property
    def business_impact_score(self) -> float:
        """Simulates real SRE cost-of-downtime.

        Starts at 1.0 (no impact). Each step while the incident is unresolved
        costs business_impact_per_step. Resolving quickly keeps this high;
        slow or failed resolution drives it toward 0.

        Not used in grading — exposed in observation for agent awareness.
        A real-world corollary: every minute of downtime costs money/SLA points.
        """
        if self.incident_resolved:
            return round(max(0.0, 1.0 - self.step_count * 0.03), 2)
        return round(max(0.0, 1.0 - self.step_count * 0.05), 2)

    # services
    services: Dict[str, ServiceState] = Field(default_factory=dict)

    # alerts & tracking
    recent_alerts: List[str] = Field(default_factory=list)
    # alert_severity maps each alert name to its severity level:
    #   "warning"  — degraded performance, no user impact yet
    #   "critical" — user-visible degradation, SLA at risk
    #   "fatal"    — complete service failure, revenue impacted
    alert_severity: Dict[str, str] = Field(default_factory=dict)
    logs_seen: Dict[str, bool] = Field(default_factory=dict)
    metrics_seen: Dict[str, bool] = Field(default_factory=dict)
    actions_taken: List[str] = Field(default_factory=list)
    safety_violations: int = 0

    # reward accumulator (for grader convenience)
    total_reward: float = 0.0

    # rollback propagation tracker
    rollback_pending: Dict[str, int] = Field(default_factory=dict)

    # log / metric text returned on last inspect
    last_inspection_result: str = ""

    class Config:
        # allow the @property to be serialized via model_dump / dict
        json_schema_extra = {}
