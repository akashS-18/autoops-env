"""
Task 1 — Easy: API Crash Loop  (task_id: easy_api_crash)

Starting state:
  • api  : health=down, error_rate=100%, crash_loop
  • db, cache, worker: all healthy

Optimal path (4 steps):
  1. inspect_logs(api) → reveals crash loop (+0.20 + 0.30)
  2. restart_service(api) → api recovering → healthy (+0.50)
  3. wait → propagation
  4. incident_resolved → +1.00 + speed bonus
"""

from autoops_env.models import AutoOpsState, ServiceState


def get_easy_state() -> AutoOpsState:
    """Return the initial AutoOpsState for the easy scenario."""
    return AutoOpsState(
        task_id="easy_api_crash",
        max_steps=15,
        root_cause="api_crash_loop_oom",
        root_cause_service="api",
        services={
            "api": ServiceState(
                health="down",
                latency_ms=0,
                error_rate=100.0,
                status="crash_loop",
            ),
            "db": ServiceState(
                health="healthy",
                latency_ms=45,
                error_rate=0.2,
                status="running",
            ),
            "cache": ServiceState(
                health="healthy",
                latency_ms=8,
                error_rate=0.0,
                status="running",
                hit_rate=94.0,
            ),
            "worker": ServiceState(
                health="healthy",
                latency_ms=30,
                error_rate=0.1,
                status="running",
                queue_depth=120,
                worker_count=4,
            ),
        },
        recent_alerts=["api.crash_loop", "api.health_critical"],
        alert_severity={
            "api.crash_loop": "fatal",        # container dead, user traffic failing
            "api.health_critical": "fatal",   # health checks failing consistently
        },
        logs_seen={"api": False, "db": False, "cache": False, "worker": False},
        metrics_seen={"api": False, "db": False, "cache": False, "worker": False},
    )
