---
title: AutoOps Env
emoji: 🚀
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
---

# AutoOps AI

> **OpenEnv Environment · Autonomous DevOps Incident Response**

## Live Demo

- Health: https://akash9363-autoops-env.hf.space/health  
- Tasks: https://akash9363-autoops-env.hf.space/tasks  
- Baseline: https://akash9363-autoops-env.hf.space/baseline  

### Quick Test

```bash
curl -X POST https://akash9363-autoops-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id":"easy_api_crash"}'  
```

An AI agent acts as an on-call SRE — it inspects production signals, chooses remediation actions, and safely restores service across three escalating incident scenarios. State changes after every action; the agent must reason about cause and effect, not just classify a snapshot.

> Baseline achieves perfect 1.000 across all tasks — demonstrating optimal policy execution.

---

## Why This Matters

Modern production systems fail in complex, multi-service cascades — not in single isolated components. When an API goes down at 3 AM, an SRE must triage five dashboards, identify root cause across interdependent services, apply the right fix in the right order, and verify recovery — all under time pressure.

**AutoOps** simulates exactly this: decisions must balance **speed** (fewer steps = more uptime), **safety** (wrong actions escalate the incident), and **cost** (rollbacks are expensive; inspecting logs is free). It is a structured, reproducible benchmark for evaluating whether an AI model can reason like an experienced on-call engineer.

---

## Episode Walkthrough — Hard: Cascading Deployment Failure

A concrete example of a successful agent run on the hardest scenario:

```
Initial state:
  api    → health=degraded, error_rate=35%, deploy=v2.4.1 (3h ago)
  db     → health=degraded, query_latency=1800ms (pool exhausted)
  worker → health=degraded, queue_depth=8200 (tasks piling up)
  cache  → health=healthy

Active alerts: [api.error_rate_high][fatal], [worker.queue_depth_critical][fatal],
               [db.latency_spike][critical], [deploy.api.failed][critical]

Step 1: inspect_metrics(api)
  → "error_rate=35%, deploy=v2.4.1@2026-03-26T12:30:00Z, pods_healthy=2/5"
  → reward: +0.20  (inspected degraded service)

Step 2: inspect_logs(api)
  → "[api-7f4b9c-xkj2q][th-0042] deploy v2.4.1 rolled out 3h ago — error_rate spiked"
  → "[ERROR] NullPointerException in /api/v2/orders handler"
  → "[INFO]  previous stable version: v2.3.8"
  → reward: +0.50  (+0.20 inspection + 0.30 root-cause bonus)

Step 3: rollback_deployment(api)
  → api status: rolling_back (2-step propagation delay begins)
  → reward: +0.50  (correct high-cost remediation)

Step 4: wait  → propagation tick 1         reward: 0.00
Step 5: wait  → rollback complete, api + db both recover to healthy
                                            reward: 0.00

Step 6: inspect_metrics(worker)
  → "queue_depth=8200, worker_count=4, task_timeout_rate=12%"
  → reward: +0.20  (inspected degraded service)

Step 7: scale_worker(worker)
  → worker_count: 4 → 8, queue_depth drops to 200, health=healthy
  → All services healthy → incident resolved
  → reward: +1.20  (+1.00 resolution + 0.20 speed bonus)

Total reward: 2.60 | Score: 1.000 | Steps: 7 / 25
```

---

## Failure Case — What Happens When the Agent Guesses Wrong

An agent that blindly restarts services without diagnosing first:

```
Step 1: restart_service(db)    ← no evidence, wrong service
  → db health=healthy → safety_violation += 1
  → reward: -0.40  (restarting a healthy service)

Step 2: restart_service(worker) ← still not diagnosed
  → worker health=degraded, but restart ≠ the right fix
  → reward: -0.05

Step 3: restart_service(api)   ← third wrong action
  → api needs rollback, not restart; deploy still active
  → reward: -0.40

Outcome: safety_violations=2, api still degraded, episode continues
  ... eventually times out → reward: -1.00 (timeout penalty)

Final score: 0.10  (only partial recovery credit, zero efficiency/safety)
```

The environment is **non-trivial** — naive agents that skip diagnosis consistently fail.

---

## System Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Production Stack                    │
│                                                      │
│   ┌──────────┐     ┌──────────┐    ┌─────────────┐  │
│   │   API    │────▶│  Cache   │    │    Worker   │  │
│   │ (FastAPI)│     │  (Redis) │    │  (Celery)   │  │
│   └────┬─────┘     └──────────┘    └──────┬──────┘  │
│        │                                  │          │
│        └──────────────┬───────────────────┘          │
│                       ▼                              │
│                ┌──────────────┐                      │
│                │  DB (Postgres)│                      │
│                └──────────────┘                      │
└─────────────────────────────────────────────────────┘
          ▲  observe / act
          │
    ┌─────────────┐
    │  AI  Agent  │  ← your model / baseline
    └─────────────┘
```

| Service | Role |
|---------|------|
| `api` | Primary application server — serves user traffic |
| `db` | PostgreSQL — shared data store |
| `cache` | Redis — in-memory cache layer |
| `worker` | Celery — background task processor |

---

## Action Space

| `action_type` | `target` | Category | Cost | Effect |
|---|---|---|---|---|
| `inspect_logs` | service name | Inspection | free | Reveals raw log lines, crash traces, deploy events |
| `inspect_metrics` | service name | Inspection | free | Reveals latency, error rate, queue depth, hit rate |
| `restart_service` | service name | Remediation | low | Sets service to `recovering`; 10% chance of failure |
| `clear_cache` | `cache` | Remediation | low | Resets Redis hit rate to 95%, warms gradually |
| `scale_worker` | `worker` | Remediation | medium | Adds 4 workers; drains queue depth |
| `rollback_deployment` | service name | Remediation | high | Reverts last deploy; **2-step propagation delay** |
| `ack_alert` | alert name | Control | free | Acknowledges alert; no service health change |
| `wait` | `None` | Control | free | Passes one step; lets in-flight effects propagate |

> **Action Cost** reflects operational risk/expense. High-cost actions on the wrong service trigger safety violations.

---

## State & Observation

### AutoOpsState (server-side — never sent to the agent)

| Field | Type | Purpose |
|---|---|---|
| `episode_id` | `str` | Unique episode identifier |
| `task_id` | `str` | Active scenario |
| `step_count` / `max_steps` | `int` | Step counter and hard limit |
| `incident_resolved` | `bool` | `True` when all services healthy |
| `root_cause` / `root_cause_service` | `str` | Ground truth (hidden from agent) |
| `services` | `dict[str, ServiceState]` | Per-service health, latency, error rate, status |
| `recent_alerts` | `list[str]` | Active alert names with severity |
| `alert_severity` | `dict[str, str]` | Alert → `warning` / `critical` / `fatal` |
| `logs_seen` / `metrics_seen` | `dict[str, bool]` | Whether agent inspected each service |
| `actions_taken` | `list[str]` | Full action history |
| `safety_violations` | `int` | Count of risky/wrong actions |
| `total_reward` | `float` | Accumulated shaped reward |

### AutoOpsObservation (what the agent sees each step)

| Field | Description |
|---|---|
| `summary` | Plain-English system status — updated each step |
| `visible_services` | `{service: {health, latency_ms, error_rate, status}}` |
| `recent_alerts` | Active alerts visible to agent |
| `available_actions` | Always the full 8 valid action types |
| `reward` | Reward from the **previous** action (0.0 on reset) |
| `done` | `True` when resolved, timed out, or ≥3 safety violations |

---

## Reward Design

| Signal | Value | When Triggered |
|---|---|---|
| ✅ Inspect degraded service | +0.20 | Any `inspect_*` on unhealthy service |
| ✅ Inspect root-cause service | +0.30 | Bonus on top of the above |
| ✅ Correct remediation | +0.50 | Action that moves service toward healthy |
| ✅ Full resolution | +1.00 | All services healthy, all alerts cleared |
| ✅ Speed bonus | +0.20 | Resolved in ≤40% of max steps |
| ❌ Wasted step | −0.05 | Inspecting a healthy service |
| ❌ Wrong remediation | −0.40 | Fixing the wrong service/wrong action |
| ❌ Risky action | −0.70 | Dangerous action without justification |
| ❌ Timeout | −1.00 | Exceeded max steps OR ≥3 safety violations |

**Shaped step rewards teach the agent inspection-before-action discipline.** Every remediation action applied without first diagnosing the root-cause service incurs a diagnosis penalty.

---

## Tasks

### Task 1 — Easy: API Crash Loop (`easy_api_crash`)

```
Incident: API service OOM-killed, stuck in crash loop (ExitCode 137).
DB, cache, worker are healthy. Alert severity: fatal.
```

**Optimal path (4 steps):**
```
Step 1: inspect_logs(api)      → "OOMKilled, crash_loop_backoff"   [+0.50]
Step 2: restart_service(api)   → api=recovering                    [+0.50]
Step 3: wait                   → api=healthy                        [0.00]
Step 4: (done)                 → incident resolved                 [+1.20]  ← +1.00 resolution + +0.20 speed
```

- **Max steps:** 15 | **Optimal:** 4 steps | **Grader weights:** Recovery 50%, Diagnosis 25%, Efficiency 15%, Safety 10%

---

### Task 2 — Medium: Cache Miss Storm (`medium_cache_latency`)

```
Incident: Redis cache hit rate at 12% (normal: 90%+) causing API latency
to spike to 4200ms. Database query rate 3× normal due to cache fallthrough.
Alert severity: critical.
```

**Optimal path (4 steps):**
```
Step 1: inspect_metrics(api)   → "latency 4200ms, timeout errors 8%"   [+0.20]
Step 2: inspect_metrics(cache) → "hit_rate=12%, miss_storm detected"   [+0.50]
Step 3: clear_cache(cache)     → cache=healthy, hit_rate=95%            [+0.50]
Step 4: wait                   → api latency drops, resolved            [+1.20]
```

- **Max steps:** 20 | **Optimal:** 4 steps | **Grader weights:** Recovery 40%, Diagnosis 25%, Efficiency 20%, Safety 15%

---

### Task 3 — Hard: Cascading Deployment Failure (`hard_cascading_incident`)

```
Incident: Buggy API deploy (v2.4.1, 3h ago) causes 35% error rate.
The error surge is exhausting the DB connection pool and overflowing
the worker queue (8200 tasks). Three services degraded simultaneously.
Alert severity: fatal (API, worker), critical (DB).
```

**Optimal path (7 steps):**
```
Step 1: inspect_metrics(api)       → "error_rate=35%, deploy=v2.4.1"   [+0.20]
Step 2: inspect_logs(api)          → "NullPointerException, v2.4.1"    [+0.50]
Step 3: rollback_deployment(api)   → api=rolling_back (2-step delay)   [+0.50]
Step 4: wait                       → propagation tick 1
Step 5: wait                       → rollback complete, api+db healthy  [+1.00] ← resolution only if worker done
Step 6: inspect_metrics(worker)    → "queue_depth=8200"                 [+0.20]
Step 7: scale_worker(worker)       → queue drains, all healthy          [+1.20]
```

- **Max steps:** 25 | **Optimal:** 7 steps | **Grader weights:** Recovery 35%, Diagnosis 25%, Efficiency 20%, Safety 20%

---

## Grader

Episodes are graded across 4 dimensions:

```
score = recovery × w_r + diagnosis × w_d + efficiency × w_e + safety × w_s
```

| Dimension | Easy | Medium | Hard | Formula |
|---|---|---|---|---|
| **Recovery** | 0.50 | 0.40 | 0.35 | 1.0 if resolved, else healthy_svc / total_svc |
| **Diagnosis** | 0.25 | 0.25 | 0.25 | 1.0 if root-cause service was inspected |
| **Efficiency** | 0.15 | 0.20 | 0.20 | 1.0 if steps ≤ optimal×1.5, linear decay to 0 at max |
| **Safety** | 0.10 | 0.15 | 0.20 | 1.0 − 0.25×violations (floor 0) |

---

## Baseline Agent

The deterministic baseline uses hard-coded if/then logic — the **provably optimal** action sequence for each task. It achieves perfect 1.000 on all three tasks because:

1. It inspects the correct service first (full diagnosis score)
2. It applies the correct remediation (full recovery score)
3. It finishes in ≤ optimal × 1.5 steps (full efficiency score)
4. It never makes a wrong action (zero safety violations)

```bash
python baseline.py
```

Expected output:

| Task | Score | Recovery | Diagnosis | Efficiency | Safety | Steps |
|---|---|---|---|---|---|---|
| easy_api_crash | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 3 |
| medium_cache_latency | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 4 |
| hard_cascading_incident | **1.000** | 1.000 | 1.000 | 1.000 | 1.000 | 7 |

> The baseline serves as the **ceiling benchmark**. An LLM agent must match it by discovering the diagnosis path from observation text alone — without access to `root_cause_service`.

---

## Setup

```bash
git clone https://huggingface.co/spaces/akash9363/autoops-env
cd autoops-env

pip install -r requirements.txt

uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/health` | Liveness probe → `{"status": "ok"}` |
| `POST` | `/reset` | Start new episode → `AutoOpsObservation` |
| `POST` | `/step` | Submit action → `AutoOpsObservation` |
| `GET` | `/state` | Full internal state (debug/grader) |
| `GET` | `/docs` | Swagger UI — interactive playground |
| `GET` | `/tasks` | List all tasks with metadata |
| `GET` | `/grader?episode_id=X` | Score a completed episode |
| `GET` | `/baseline` | Run baseline agent, return results |

```bash
# Quick smoke test
curl http://localhost:8000/health
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_id": "easy_api_crash"}'
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action_type": "inspect_logs", "target": "api"}'
curl http://localhost:8000/tasks
curl http://localhost:8000/baseline
```

---

## Docker

```bash
# Build from repo root
docker build -t autoops-env .
docker run -p 7860:7860 autoops-env
```

---

## HuggingFace Spaces Deployment

```bash
git clone https://huggingface.co/spaces/akash9363/autoops-env
cd autoops-env

git add .
git commit -m "deploy autoops-env"
git push

curl https://akash9363-autoops-env.hf.space/health
curl https://akash9363-autoops-env.hf.space/baseline
```

---

## Typed Python Client

```python
from client import AutoOpsClient

with AutoOpsClient("http://localhost:8000") as client:
    obs = client.reset("hard_cascading_incident")
    print(obs.summary)
    # ⚠️ Incident in progress — step 0/25. Affected: api, db, worker.

    obs = client.step("inspect_logs", "api")
    print(obs.reward)   # 0.5  (+0.20 inspect + 0.30 root-cause bonus)

    score = client.grader(episode_id)
    print(score["score"])  # 0.xxx
```
