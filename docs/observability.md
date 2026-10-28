# Model accounting, monitoring and budgets

Every production chat, structured-sentiment and Jev request now reserves budget
in `outputs/usage.sqlite` before dispatch. SDK retries are disabled; explicit
retries and provider fallback each create a separate request row. This adds no
model calls. Existing `usage` and `llm_calls` summary fields retain their original
agent-only meaning; `model_usage` covers the complete model request ledger.

## Defaults and configuration

Existing agent-turn and time limits remain enabled. Additional request, token
and estimated-USD limits default to `null` (observe only). Chat responses have an
8192-token output limit, configurable with `llm.max_output_tokens`; this includes
tool arguments and generated code. Jev's typed-response API has no equivalent
output-token cap, so its admission reservation uses an estimated response size.
DeepSeek requests send `max_tokens` explicitly, matching its
[request schema](https://api-docs.deepseek.com/api/create-chat-completion/).
MiMo uses [`max_completion_tokens`](https://mimo.mi.com/docs/en-US/api/chat/openai-api),
as does ChatOpenAI's default OpenAI mapping. A provider configuration can override
this with `output_token_parameter` for other compatible gateways.

For example, to bound all model attempts per run while observing tokens/cost:

```yaml
budget:
  max_model_requests: 40
  max_total_tokens: null
  max_estimated_cost_usd: null
  prices: {}
```

Price entries use exact `provider:model` keys (including `jev:<configured-model>`)
and these fields:

| Field | Meaning |
|---|---|
| `version` | Required price source/date or other reproducible version |
| `input_per_million` | USD per million ordinary input tokens |
| `output_per_million` | USD per million output tokens |
| `cached_input_per_million` | Optional; defaults to ordinary input rate |
| `per_request` | Optional fixed USD charge per request; defaults to zero |

No current provider prices are assumed. Populate verified prices before enabling
`max_estimated_cost_usd`. An unpriced model, or an earlier unpriced request in
the run, blocks further dispatch when the cost limit is enabled. Prices are
snapshotted per request; editing the configuration never reprices history.
This simple price table supports token rates and fixed request charges; billing
by questions, tiers or other provider rules requires a matching pricing adapter.

## Admission, recovery and limits

- Request-count limits are atomic across concurrent threads and worker processes.
  Every admitted attempt counts, including errors, retries and fallback.
- Token and USD admission reserve estimated input plus maximum configured chat
  output. Input estimation is a padded character heuristic, not provider-exact
  tokenization. These are **estimated admission limits, not guaranteed billing
  ceilings**. Actual returned usage settles the reservation and may exceed the
  estimate. A subsequent call then sees the larger settled usage.
- Missing/partial usage stays unknown, including unsuccessful requests that may
  have been billed. Its reservation is retained; totals never silently treat an
  unpriced or unknown request as free. `known_estimated_cost_usd` is only a known
  subtotal; `estimated_cost_usd` is `null` if any request cost is unknown.
- Pending requests survive crashes without releasing their reserved budget.
  Resuming the same run ID retains both requests and the original start time.
  Runs predating this ledger adopt the first `run_started` trace timestamp when
  available. No historical token/cost values are invented for old requests.
  Adopted legacy runs mark prior usage unknown; enabling any extra budget blocks
  new model dispatch for those runs because earlier consumption cannot be proven.
- Every model attempt receives a timeout bounded by the remaining run time;
  retry waits cannot cross that time. The deadline remains cooperative:
  synchronous HTTP operation timeouts are not a hard process-kill deadline.
  Public market-data HTTP and sandbox operations retain their existing limits.
- Budget exhaustion bypasses model retries, provider fallback and rule fallback
  in semantic checking. The graph produces its existing deterministic degraded
  report and records `budget_exceeded: <dimension>`.

## Inspection

- `summary.json` contains `model_usage`, budget settings and total wall time
  including recovery downtime (`total_elapsed_s`). Original `elapsed_s` remains
  the duration of the current runner invocation.
- `GET /api/runs/{id}/usage?limit=100&offset=0` returns ledger totals and paginated
  requests: stage, provider/model, attempt, fallback, latency, HTTP error status,
  usage completeness, pricing snapshot, costs and retained reservations.
- `GET /api/metrics?limit=100` aggregates the latest ledger runs in that output
  directory (maximum 1000): completion/degradation/failure rates, p50/p95/max
  latency, guard rejections, budget stops, model errors/timeouts/429/retries,
  usage completeness, cost and queue depth/oldest queued age.
- Cost per qualified report includes **all finished runs' costs**, including
  failures, divided by `agent_completed` reports. Missing cost makes this metric
  unknown. `task_completed` separately counts successful actual SMTP sends;
  inbox delivery is not inferred. Costs for unfinished runs are visible in the
  overall subtotal but excluded from the finished-run ratio.

The API's existing bearer authentication protects these endpoints. Ledger rows
contain no prompt bodies or credentials. Existing full trace files retain their
existing access policy. Monitoring is computed on demand, without a polling
worker or new external monitoring dependency; no automatic alerts are scheduled.
Different output directories isolate production and evaluation statistics.

New synthetic live-evaluation reports add all-model `model_usage` and USD cost
completeness. Legacy `total_tokens` still means agent chat only. Old evidence
files are unchanged; they cannot establish post-change live performance or cost.

## Performance and verification

One running context reuses a serialized SQLite connection across tool threads.
Transactions keep SQLite FULL durability. There are two small synchronous
transactions per request (reserve/settle), and no per-token database writes.
Connections close when the run ends, including error/interruption paths.

```powershell
uv run pytest -q tests/test_usage_budget.py
uv run python scripts/benchmark_observability.py --calls 50 --delay-ms 20 --output outputs/observability-overhead.json
```

The benchmark alternates baseline SDK requests and instrumented requests against
the same local fixed-response transport. It excludes constructors and warm-up;
startup time is reported separately. It measures local accounting overhead,
**not live-model latency or end-to-end report completion**. Both initial and
connection-reuse measurements are retained locally in the ignored evaluation output directories.
No real model requests or email are sent by this benchmark.
