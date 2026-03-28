"""
AutoOps Environment — core simulation logic.

Implements:
  • reset(task_id) → AutoOpsObservation
  • step(action)   → AutoOpsObservation  (with reward + side-effects)
  • state property  → AutoOpsState snapshot
  • _build_observation() helper

Reward ladder (shaped step rewards):
  +0.20  inspect degraded service               (encourages diagnosis first)
  +0.30  inspect root-cause service (bonus)      (guides toward root cause)
  +0.50  correct remediation action              (meaningful progress)
  +1.00  full incident resolution                (primary objective)
  +0.20  speed bonus (≤40% of max_steps)         (encourages efficiency)
  −0.05  wasted step (healthy service inspected)
  −0.40  wrong remediation (wrong service/action)
  −0.70  risky action (wrong target / no evidence)
  −1.00  timeout or ≥3 safety violations         (episode failure)

Design note: partial rewards are intentional and dense — they allow RL
agents to learn intermediate diagnosis behaviour, not just final resolution.
"""

from __future__ import annotations

import copy
import random
from typing import Optional

from models import (
    AutoOpsAction,
    AutoOpsObservation,
    AutoOpsState,
    ServiceState,
)
from tasks import TASK_REGISTRY

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_ACTIONS = [
    "inspect_logs",
    "inspect_metrics",
    "restart_service",
    "clear_cache",
    "scale_worker",
    "rollback_deployment",
    "ack_alert",
    "wait",
]

VALID_SERVICES = ["api", "db", "cache", "worker"]

# ---------------------------------------------------------------------------
# Simulated log outputs — structured with realistic noise.
#
# Each template contains the stable, root-cause-relevant signal plus
# randomised noise tokens (timestamps, thread IDs, pod names) that force
# the agent to extract signal rather than match exact strings.
# ---------------------------------------------------------------------------

# Noise pools injected into log lines to simulate real-world log variance
_POD_NAMES = ["api-7f4b9c-xkj2q", "api-7f4b9c-mnp8r", "api-7f4b9c-zt5lw"]
_THREAD_IDS = ["th-0042", "th-0017", "th-0091"]


def _noisy_log(base: str) -> str:
    """Inject a random pod name and thread-id into a log line for realism."""
    pod = random.choice(_POD_NAMES)
    thread = random.choice(_THREAD_IDS)
    return f"[{pod}][{thread}] {base}"


# Core signal templates (deterministic root-cause content)
_SIMULATED_LOGS = {
    "easy_api_crash": {
        "api": (
            "[ERROR] OOMKilled — container exceeded 512Mi limit\n"
            "[WARN]  restarting container (attempt 4/5)\n"
            "[ERROR] ExitCode: 137 — process killed by kernel\n"
            "[INFO]  crash_loop_backoff: 30s"
        ),
        "db": "[INFO] connections=42/200, WAL size nominal",
        "cache": "[INFO] hit_rate=94%, eviction_rate=low",
        "worker": "[INFO] queue_depth=120, processing normally",
    },
    "medium_cache_latency": {
        "api": (
            "[WARN]  p99 latency 4200ms (threshold 500ms)\n"
            "[INFO]  upstream cache miss ratio elevated\n"
            "[WARN]  timeout errors increasing: 8%"
        ),
        "cache": (
            "[ERROR] cache miss storm detected — hit_rate=12%\n"
            "[WARN]  eviction rate 340/s (normal: 20/s)\n"
            "[INFO]  memory_usage=98%, ttl expiry burst"
        ),
        "db": "[WARN] query_rate=2400/s (normal: 800/s), likely cache fallthrough",
        "worker": "[INFO] queue_depth=380, processing normally",
    },
    "hard_cascading_incident": {
        "api": (
            "[ERROR] deploy v2.4.1 rolled out 3h ago — error_rate spiked to 35%\n"
            "[ERROR] NullPointerException in /api/v2/orders handler\n"
            "[WARN]  health check failing on 3/5 pods\n"
            "[INFO]  previous stable version: v2.3.8"
        ),
        "db": (
            "[WARN]  connection pool exhaustion — 195/200 active\n"
            "[WARN]  query_latency p99=1800ms (normal: 45ms)"
        ),
        "cache": "[INFO] hit_rate=91%, operating normally",
        "worker": (
            "[ERROR] queue_depth=8200 (threshold: 1000)\n"
            "[WARN]  task timeout rate 12%\n"
            "[INFO]  worker_count=4"
        ),
    },
}

# Simulated metrics outputs per task scenario
_SIMULATED_METRICS = {
    "easy_api_crash": {
        "api": "health=down | latency_ms=0 | error_rate=100% | restarts=4 | status=crash_loop",
        "db": "health=healthy | latency_ms=45 | error_rate=0.2% | connections=42/200",
        "cache": "health=healthy | latency_ms=8 | hit_rate=94% | evictions=low",
        "worker": "health=healthy | queue_depth=120 | worker_count=4 | throughput=48/s",
    },
    "medium_cache_latency": {
        "api": "health=degraded | latency_ms=4200 | error_rate=8% | timeout_count=312",
        "cache": "health=degraded | hit_rate=12% | latency_ms=320 | memory=98% | status=miss_storm",
        "db": "health=healthy | query_rate=2400/s | query_latency=180ms | connections=110/200",
        "worker": "health=healthy | queue_depth=380 | worker_count=4 | throughput=44/s",
    },
    "hard_cascading_incident": {
        "api": (
            "health=degraded | latency_ms=2800 | error_rate=35% | "
            "deploy=v2.4.1@2026-03-26T12:30:00Z | pods_healthy=2/5"
        ),
        "db": "health=degraded | query_latency=1800ms | connections=195/200 | query_rate=3200/s",
        "cache": "health=healthy | hit_rate=91% | latency_ms=10 | memory=62%",
        "worker": "health=degraded | queue_depth=8200 | worker_count=4 | task_timeout_rate=12%",
    },
}


# ---------------------------------------------------------------------------
# Environment class
# ---------------------------------------------------------------------------

class DevOpsEnvironment:
    """Stateful environment that simulates a production incident."""

    def __init__(self) -> None:
        self._state: Optional[AutoOpsState] = None
        self._last_reward: float = 0.0
        # Store completed episode states for grader lookup
        self._episode_history: dict[str, AutoOpsState] = {}

    # -- public API ---------------------------------------------------------

    def reset(self, task_id: str) -> AutoOpsObservation:
        """Start a fresh episode for the given task."""
        if task_id not in TASK_REGISTRY:
            raise ValueError(
                f"Unknown task_id '{task_id}'. "
                f"Valid: {list(TASK_REGISTRY.keys())}"
            )
        self._state = TASK_REGISTRY[task_id]()
        self._last_reward = 0.0
        return self._build_observation()

    def step(self, action: AutoOpsAction) -> AutoOpsObservation:
        """Apply *action*, compute reward, advance step counter."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        s = self._state
        reward = 0.0

        # ── validate action type ──────────────────────────────────
        if action.action_type not in VALID_ACTIONS:
            reward = -0.10
            s.step_count += 1
            s.actions_taken.append(action.action_type)
            self._last_reward = reward
            s.total_reward += reward
            return self._build_observation()

        # ── dispatch ──────────────────────────────────────────────
        handler = getattr(self, f"_handle_{action.action_type}", None)
        if handler:
            reward = handler(action.target)

        s.actions_taken.append(action.action_type)
        s.step_count += 1
        s.total_reward += reward

        # ── advance pending rollbacks ─────────────────────────────
        self._tick_rollbacks()

        # ── check resolution ──────────────────────────────────────
        self._check_resolution()

        # ── check timeout ─────────────────────────────────────────
        if s.step_count >= s.max_steps and not s.incident_resolved:
            reward += -1.00                     # timeout penalty
            s.total_reward += -1.00

        # ── check safety escalation ──────────────────────────────
        if s.safety_violations >= 3 and not s.incident_resolved:
            reward += -1.00
            s.total_reward += -1.00
            # store episode for grader before ending
            self._episode_history[s.episode_id] = copy.deepcopy(s)

        self._last_reward = reward

        # store completed episode
        if s.incident_resolved or s.step_count >= s.max_steps or s.safety_violations >= 3:
            self._episode_history[s.episode_id] = copy.deepcopy(s)

        return self._build_observation()

    @property
    def state(self) -> Optional[AutoOpsState]:
        return self._state

    def get_episode(self, episode_id: str) -> Optional[AutoOpsState]:
        """Retrieve a completed (or current) episode state by ID."""
        if self._state and self._state.episode_id == episode_id:
            return self._state
        return self._episode_history.get(episode_id)

    # -- action handlers ----------------------------------------------------

    def _handle_inspect_logs(self, target: Optional[str]) -> float:
        """Reveal raw log lines for a service.

        Signal content is deterministic (root-cause info always present).
        Log text includes random pod/thread noise to simulate real-world
        streams — the agent must parse the signal, not match exact strings.

        Rewards:
          +0.20  if service is unhealthy  (useful diagnostic step)
          +0.30  bonus if this is the root-cause service
          −0.05  if service is healthy    (wasted inspection)
        """
        s = self._state
        if target not in VALID_SERVICES:
            return -0.05
        svc = s.services[target]
        s.logs_seen[target] = True

        # Build log output: inject noise prefix on first line for realism
        logs = _SIMULATED_LOGS.get(s.task_id, {})
        raw = logs.get(target, f"[INFO] {target}: no notable events")
        first_line, *rest = raw.split("\n")
        noisy_first = _noisy_log(first_line.lstrip("[").split("] ", 1)[-1])
        s.last_inspection_result = "\n".join([noisy_first] + rest)

        # Reward: inspecting an unhealthy service is always useful;
        # bonus for finding the root cause.
        if svc.health in ("down", "degraded"):
            r = 0.20
            if target == s.root_cause_service:
                r += 0.30  # root-cause diagnosis bonus
            return r
        return -0.05  # inspecting a healthy service wastes a step

    def _handle_inspect_metrics(self, target: Optional[str]) -> float:
        """Reveal quantitative metrics for a service.

        Metrics are always deterministic (no noise) — they provide precise
        numbers the agent can reason about (latency_ms, error_rate, etc.).
        Use inspect_logs for contextual narrative, inspect_metrics for data.

        Same reward structure as inspect_logs.
        """
        s = self._state
        if target not in VALID_SERVICES:
            return -0.05
        svc = s.services[target]
        s.metrics_seen[target] = True

        metrics = _SIMULATED_METRICS.get(s.task_id, {})
        s.last_inspection_result = metrics.get(target, f"{target}: all metrics nominal")

        if svc.health in ("down", "degraded"):
            r = 0.20
            if target == s.root_cause_service:
                r += 0.30  # root-cause diagnosis bonus
            return r
        return -0.05

    def _handle_restart_service(self, target: Optional[str]) -> float:
        """Restart a service container.

        Cost: LOW — safe when the service is actually down, but restarting
        a healthy service or the wrong service triggers a safety violation.

        Stochastic failure: 10% chance in the easy scenario to simulate
        real-world restart unreliability (OOM keeps firing).

        Diagnosis requirement: applying restart without first inspecting
        the root-cause service incurs a −0.20 penalty on top of the reward.
        """
        s = self._state
        if target not in VALID_SERVICES:
            # Restarting a nonexistent service is always a safety error
            s.safety_violations += 1
            return -0.70

        svc = s.services[target]

        # Track whether agent diagnosed before acting
        diagnosed = s.logs_seen.get(s.root_cause_service, False) or s.metrics_seen.get(
            s.root_cause_service, False
        )

        # Restarting a healthy service is dangerous (potential data loss, lost work)
        if svc.health == "healthy":
            s.safety_violations += 1
            return -0.40

        # ── Easy: api crash loop — restart IS the right fix ──────────────
        if s.task_id == "easy_api_crash" and target == "api":
            # 10% stochastic failure: OOM persists despite restart
            if random.random() < 0.10:
                s.safety_violations += 1
                s.last_inspection_result = "restart_service(api) — FAILED: container still OOMing"
                return -0.20
            svc.health = "recovering"
            svc.status = "restarting"
            svc.error_rate = 5.0
            svc.latency_ms = 800
            # Diagnosis penalty: acted without inspecting root cause first
            penalty = -0.20 if not diagnosed else 0.0
            return 0.50 + penalty

        # ── Medium: cache is the root cause, not api ──────────────────────
        if s.task_id == "medium_cache_latency" and target == "api":
            return -0.40  # wrong service — restarting api doesn't fix the cache

        # ── Hard: need rollback, not restart ─────────────────────────────
        if s.task_id == "hard_cascading_incident" and target == "api":
            return -0.40  # restart leaves the broken deploy in place

        # Generic: restart of another degraded service (partial credit)
        if svc.health in ("degraded", "down"):
            svc.health = "recovering"
            svc.status = "restarting"
            return 0.30

        return -0.05

    def _handle_clear_cache(self, target: Optional[str]) -> float:
        """Flush and warm the Redis cache.

        Cost: LOW — idempotent and non-destructive, but reduces hit rate
        temporarily during warm-up. Only meaningful when cache is the
        root cause (medium scenario).

        Side-effects (medium only):
          cache: hit_rate → 95%, health → healthy
          api:   latency starts dropping (needs a wait to fully recover)
          db:    query_rate drops as cache absorbs traffic again
        """
        s = self._state
        if target != "cache":
            # clear_cache on a non-cache service makes no sense — safety error
            s.safety_violations += 1
            return -0.40

        cache_svc = s.services["cache"]

        # ── Medium: cache IS the root cause — this is the correct fix ────
        if s.task_id == "medium_cache_latency":
            cache_svc.health = "healthy"
            cache_svc.hit_rate = 95.0
            cache_svc.latency_ms = 12
            cache_svc.error_rate = 0.0
            cache_svc.status = "running"

            # API starts recovering immediately as cache warms
            api_svc = s.services["api"]
            api_svc.latency_ms = 600  # still elevated; needs one wait step
            api_svc.error_rate = 2.0
            api_svc.status = "recovering"

            # DB query rate falls — cache absorbs reads again
            db_svc = s.services["db"]
            db_svc.query_rate = 900
            db_svc.query_latency = 50

            return 0.50

        # Clearing cache when it's healthy is wasteful but not dangerous
        if cache_svc.health == "healthy":
            return -0.20
        return 0.0

    def _handle_scale_worker(self, target: Optional[str]) -> float:
        """Horizontally scale the worker pool.

        Cost: MEDIUM — adds 4 workers and drains queue backlog.

        Safety risk: scaling workers while the database is still overloaded
        amplifies DB pressure (more workers = more concurrent queries).
        In the hard scenario this is a safety violation if the API deploy
        hasn't been rolled back yet and the DB is still degraded.

        Only meaningful when queue_depth > 1000.
        """
        s = self._state
        if target != "worker":
            # scale_worker on any other service doesn't make sense
            s.safety_violations += 1
            return -0.40

        worker_svc = s.services["worker"]
        db_svc = s.services["db"]

        # ── Safety check: scaling into a degraded DB is dangerous ────────
        # More workers → more concurrent queries → worsens connection pool
        # exhaustion. Only apply this block in hard scenario before rollback.
        if db_svc.health in ("down", "degraded") and db_svc.query_latency and db_svc.query_latency > 1500:
            if s.task_id == "hard_cascading_incident" and s.services["api"].health == "degraded":
                s.safety_violations += 1
                return -0.70  # premature scaling — DB overload would worsen

        # ── Meaningful scaling: queue is backed up ────────────────────────
        if worker_svc.queue_depth and worker_svc.queue_depth > 1000:
            worker_svc.worker_count = (worker_svc.worker_count or 4) + 4
            worker_svc.queue_depth = max(200, worker_svc.queue_depth - 6000)
            worker_svc.health = "healthy"
            worker_svc.status = "running"
            worker_svc.error_rate = 0.5
            return 0.50

        return -0.05  # queue is fine — scaling adds cost with no benefit

    def _handle_rollback_deployment(self, target: Optional[str]) -> float:
        """Revert the last deployment for a service.

        Cost: HIGH — irreversible during the episode. Triggers a 2-step
        propagation delay before effects are visible (simulates a
        real Kubernetes rolling update taking time to drainConnections).

        Safety guard: rolling back when there is no recent deployment is
        always a safety violation (you'd be reverting a working version).
        """
        s = self._state
        if target not in VALID_SERVICES:
            s.safety_violations += 1
            return -0.70

        svc = s.services[target]

        # Guard: no recent deploy recorded → rollback is baseless
        if not svc.recent_deploy:
            s.safety_violations += 1
            return -0.70

        # ── Hard: API deploy is the root cause — rollback is correct ─────
        if s.task_id == "hard_cascading_incident" and target == "api":
            # Schedule 2-step propagation — effects appear after 2 waits
            s.rollback_pending[target] = 2
            svc.status = "rolling_back"
            return 0.50

        # Other scenarios: deploy exists but isn't the cause → partial penalty
        return -0.20

    def _handle_ack_alert(self, target: Optional[str]) -> float:
        s = self._state
        if target and target in s.recent_alerts:
            # penalize acking critical alerts without fixing
            if "critical" in target.lower() or "failed" in target.lower():
                return -0.05
            s.recent_alerts.remove(target)
            return 0.0
        return -0.05  # alert doesn't exist

    def _handle_wait(self, _target: Optional[str]) -> float:
        s = self._state

        # Easy — if api is recovering, transition to healthy
        if s.task_id == "easy_api_crash":
            api = s.services["api"]
            if api.health == "recovering":
                api.health = "healthy"
                api.latency_ms = 95
                api.error_rate = 0.3
                api.status = "running"
                return 0.0

        # Medium — after cache clear, api finishes recovery
        if s.task_id == "medium_cache_latency":
            api = s.services["api"]
            if api.status == "recovering":
                api.health = "healthy"
                api.latency_ms = 95
                api.error_rate = 0.4
                api.status = "running"
                return 0.0

        return 0.0  # passive step

    # -- internal helpers ---------------------------------------------------

    def _tick_rollbacks(self) -> None:
        """Advance pending rollback timers and apply effects when ready."""
        s = self._state
        finished = []
        for svc_name, ticks_left in s.rollback_pending.items():
            ticks_left -= 1
            if ticks_left <= 0:
                # apply rollback effect
                svc = s.services[svc_name]
                svc.health = "healthy"
                svc.error_rate = 1.0
                svc.latency_ms = 120
                svc.status = "running"
                svc.recent_deploy = None

                # cascading recovery — db improves once api is healthy
                if svc_name == "api" and "db" in s.services:
                    db = s.services["db"]
                    if db.health == "degraded":
                        db.health = "healthy"
                        db.query_latency = 50
                        db.query_rate = 850
                        db.status = "running"
                        db.error_rate = 0.2

                finished.append(svc_name)
            else:
                s.rollback_pending[svc_name] = ticks_left

        for f in finished:
            del s.rollback_pending[f]

    def _check_resolution(self) -> None:
        """Check if all services are healthy and auto-clear related alerts."""
        s = self._state

        # Auto-clear alerts for services that have recovered
        healthy_services = [
            name for name, svc in s.services.items()
            if svc.health == "healthy"
        ]
        s.recent_alerts = [
            alert for alert in s.recent_alerts
            if not any(svc_name in alert for svc_name in healthy_services)
        ]

        all_healthy = all(
            svc.health == "healthy" for svc in s.services.values()
        )
        if all_healthy:
            s.incident_resolved = True
            s.recent_alerts = []

    def _build_observation(self) -> AutoOpsObservation:
        """Build the agent-facing AutoOpsObservation from current state.

        This is the only view the agent has of the environment. It includes:
          - A plain-English summary of the current situation
          - Per-service health/latency/error snapshots
          - Active alerts enriched with severity labels
          - The shaped reward from the most recent action
          - A done flag for episode termination

        Note: last_inspection_result is consumed (cleared) here so it
        only appears in the observation immediately following the inspect.
        """
        s = self._state

        # ── Compose the human-readable summary ───────────────────────────
        unhealthy = [
            name for name, svc in s.services.items()
            if svc.health != "healthy"
        ]
        if s.incident_resolved:
            summary = (
                f"✅ Incident resolved in {s.step_count} steps. "
                f"All services healthy. "
                f"Business impact score: {s.business_impact_score:.2f}"
            )
        elif not unhealthy:
            summary = "All services appear healthy. Verifying stability…"
        else:
            # Include severity of worst active alert in summary for urgency
            severities = list(s.alert_severity.values())
            worst = "fatal" if "fatal" in severities else ("critical" if "critical" in severities else "warning")
            summary = (
                f"⚠️  Incident in progress [{worst}] — step {s.step_count}/{s.max_steps}. "
                f"Affected: {', '.join(unhealthy)}. "
                f"Active alerts: {len(s.recent_alerts)}. "
                f"Time remaining: {s.time_remaining} steps. "
                f"Business impact score: {s.business_impact_score:.2f}"
            )

        # Append last inspection output if present; consume it afterward
        if s.last_inspection_result:
            summary += f"\n\nLast inspection output:\n{s.last_inspection_result}"
            s.last_inspection_result = ""  # consumed — won't repeat next step

        # ── Build visible_services dict ───────────────────────────────────
        # Only expose what the agent should see; internal fields are hidden.
        visible = {}
        for name, svc in s.services.items():
            entry: dict = {
                "health": svc.health,
                "latency_ms": svc.latency_ms,
                "error_rate": svc.error_rate,
                "status": svc.status,
            }
            if svc.hit_rate is not None:
                entry["hit_rate"] = svc.hit_rate
            if svc.queue_depth is not None:
                entry["queue_depth"] = svc.queue_depth
            if svc.worker_count is not None:
                entry["worker_count"] = svc.worker_count
            visible[name] = entry

        # ── Episode termination conditions ────────────────────────────────
        done = (
            s.incident_resolved
            or s.step_count >= s.max_steps
            or s.safety_violations >= 3
        )

        # ── Resolution bonus (awarded exactly once at resolution) ─────────
        if s.incident_resolved and not getattr(s, '_resolution_rewarded', False):
            s._resolution_rewarded = True
            bonus = 1.00  # base resolution reward
            if s.step_count <= int(s.max_steps * 0.4):
                bonus += 0.20  # speed bonus: resolved in ≤40% of steps
            self._last_reward += bonus
            s.total_reward += bonus

        # Enrich recent_alerts with severity labels for the agent
        alerts_with_severity = [
            f"{alert} [{s.alert_severity.get(alert, 'warning')}]"
            for alert in s.recent_alerts
        ]

        return AutoOpsObservation(
            summary=summary,
            visible_services=visible,
            recent_alerts=alerts_with_severity,
            available_actions=list(VALID_ACTIONS),
            reward=self._last_reward,
            done=done,
        )
