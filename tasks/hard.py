"""
Task 3 — Hard: Cascading Deployment Failure  (task_id: hard_cascading_incident)

Starting state:
  • api   : degraded, error_rate 35%, latency 2800ms — bad deploy 3 h ago
  • db    : degraded, query_latency 1800ms — connections backing up
  • worker: degraded, queue_depth 8200 — tasks piling up
  • cache : healthy — not involved in root cause

Optimal path (5-7 steps):
  1. inspect_metrics(api)       → error rate + deploy timestamp (+0.20)
  2. inspect_logs(api)          → deploy trace, bad version (+0.20 + 0.30)
  3. rollback_deployment(api)   → triggers 2-step propagation (+0.50)
  4. wait                       → rollback propagating
  5. inspect_metrics(worker)    → queue still critical (+0.20)
  6. scale_worker(worker)       → queue drains (+0.50)
  7. wait                       → all healthy, resolved (+1.00)
"""

from autoops_env.models import AutoOpsState, ServiceState


def get_hard_state() -> AutoOpsState:
    """Return the initial AutoOpsState for the hard scenario."""
    return AutoOpsState(
        task_id="hard_cascading_incident",
        max_steps=25,
        root_cause="bad_api_deployment",
        root_cause_service="api",
        services={
            "api": ServiceState(
                health="degraded",
                latency_ms=2800,
                error_rate=35.0,
                status="error_spike",
                recent_deploy="2026-03-26T12:30:00Z",
            ),
            "db": ServiceState(
                health="degraded",
                latency_ms=450,
                error_rate=2.5,
                status="connection_backlog",
                query_rate=3200,
                query_latency=1800,
            ),
            "cache": ServiceState(
                health="healthy",
                latency_ms=10,
                error_rate=0.0,
                status="running",
                hit_rate=91.0,
            ),
            "worker": ServiceState(
                health="degraded",
                latency_ms=120,
                error_rate=4.0,
                status="queue_backlog",
                queue_depth=8200,
                worker_count=4,
            ),
        },
        recent_alerts=[
            "api.error_rate_high",
            "db.latency_spike",
            "worker.queue_depth_critical",
            "deploy.api.failed",
        ],
        alert_severity={
            "api.error_rate_high": "fatal",         # 35% error rate, most requests failing
            "db.latency_spike": "critical",          # connection pool near exhaustion
            "worker.queue_depth_critical": "fatal",  # 8200 tasks backed up, growing fast
            "deploy.api.failed": "critical",         # recent deploy correlated with spike
        },
        logs_seen={"api": False, "db": False, "cache": False, "worker": False},
        metrics_seen={"api": False, "db": False, "cache": False, "worker": False},
    )
