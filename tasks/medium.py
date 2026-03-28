"""
Task 2 — Medium: Cache Miss Storm  (task_id: medium_cache_latency)

Starting state:
  • api  : degraded, latency 4200ms, error_rate 8%
  • cache: degraded, hit_rate 12%, miss_storm
  • db   : healthy but query_rate elevated
  • worker: healthy, slight queue elevation

Optimal path (4 steps):
  1. inspect_metrics(api)  → high latency visible (+0.20)
  2. inspect_metrics(cache) → hit_rate=12% (+0.20 + 0.30 root cause)
  3. clear_cache(cache)    → hit_rate resets to 95% (+0.50)
  4. wait                  → api latency drops, resolved (+1.00 + speed)
"""

from models import AutoOpsState, ServiceState


def get_medium_state() -> AutoOpsState:
    """Return the initial AutoOpsState for the medium scenario."""
    return AutoOpsState(
        task_id="medium_cache_latency",
        max_steps=20,
        root_cause="cache_miss_storm",
        root_cause_service="cache",
        services={
            "api": ServiceState(
                health="degraded",
                latency_ms=4200,
                error_rate=8.0,
                status="slow_responses",
            ),
            "cache": ServiceState(
                health="degraded",
                latency_ms=320,
                error_rate=0.5,
                status="miss_storm",
                hit_rate=12.0,
            ),
            "db": ServiceState(
                health="healthy",
                latency_ms=180,
                error_rate=0.3,
                status="running",
                query_rate=2400,
                query_latency=180,
            ),
            "worker": ServiceState(
                health="healthy",
                latency_ms=55,
                error_rate=0.2,
                status="running",
                queue_depth=380,
                worker_count=4,
            ),
        },
        recent_alerts=["api.latency_high", "cache.hit_rate_low"],
        alert_severity={
            "api.latency_high": "critical",   # p99 latency >10× threshold, SLA breach
            "cache.hit_rate_low": "critical",  # hit rate at 12% triggering db fallthrough
        },
        logs_seen={"api": False, "db": False, "cache": False, "worker": False},
        metrics_seen={"api": False, "db": False, "cache": False, "worker": False},
    )
