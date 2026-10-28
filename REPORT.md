# NASDAQ Top-Gainer Agent: design report

## 1. Framework and architecture

A LangGraph `StateGraph` drives one model node and seven typed LangChain tools, using any OpenAI-compatible model (MiMo primary, DeepSeek fallback).

```text
START → agent ⇄ tools ──(email delivered)──→ END
          ├─ send_email → guard ─pass→ tools │ reject→ agent
          └─ deadline / budget / LLM outage / no email sent → fallback → END
```

The model chooses each next tool and its arguments; code routes, enforces prerequisites and controls the only outward action (email). LangGraph gives typed state, conditional routing and SQLite checkpoints for resume. One agent suffices: the real risks are data quality and unsupported claims, which more agents would not reduce.

## 2. Planning and execution

The prompt states the goal, constraints and analysis contract, not a step order. The model starts with `record_plan` and `get_top_nasdaq_gainers` and revises the plan, with a reason, when results change its approach.

| Tool | Role |
|---|---|
| `record_plan` | Record or revise the plan (audited, not enforced) |
| `get_top_nasdaq_gainers` | Yahoo screener, Nasdaq.com fallback; dated, flagged candidates, the raw #1 and a ranking status |
| `get_price_history` | Validated stock and index closes, returned as dataset IDs |
| `execute_python` | Runs model-written code in a Docker sandbox and verifies it |
| `get_news` / `analyze_sentiment` | ID-tagged headlines; sentiment as a labelled model estimate |
| `send_email` | Takes evidence IDs and 2–4 ordered key findings; code writes and sends the report (dry-run by default) |

Tools return an `{ok, data, warnings, error}` envelope and write validated facts into checkpointed state. The model passes identifiers such as the symbol and `ds_<SYMBOL>`, so price tables never enter the prompt. Dependent or duplicate calls in one batch are rejected; `send_email` must run alone.

By default the agent ranks common stocks priced at least $1 with 100,000+ shares traded, narrower than the assignment's unqualified ranking; the user can uncheck both filters and the report names the unfiltered #1.

## 3. Dynamic code generation and execution

The prompt defines the result fields, not the code: daily, average and cumulative return, trend, volatility, drawdown, and benchmark and excess return. The model writes a full script using `load()`, `save_result()` and `save_chart()`. Each `execute_python` call then passes four stages:

1. **Policy.** An AST check whitelists imports and blocks I/O, `eval`/`exec` and dunders.
2. **Sandbox.** `docker run --network none --read-only --cap-drop ALL`, non-root, with resource and time limits; only result JSON and PNG files are exported. Nothing runs locally without Docker.
3. **Verification.** An independent implementation recomputes every number; mismatches and attempts to override the symbol or window are retryable errors.
4. **Self-repair.** The model gets the traceback and a hint and retries, up to three failed attempts; code then renders a 150–250 word email from the verified result.


## 4. Challenges and solutions

- **Defining "the top gainer".** The raw #1 is often a warrant or an illiquid sub-$1 stock. *Solution:* user-owned common-stock filters and disclosure of the raw #1.
- **Stale rows and an over-confident ranking.** On 23 October, 5 of Yahoo's top 20 rows were older quotes; Nasdaq.com's screener lagged a session. *Solution:* per-row quote dates, daily-bar checks for untimestamped rows and near-threshold rows, and a ranking that is "verified" only when nothing that could outrank the pick is unresolved.
- **Wrong but plausible code.** Recorded runs returned the wrong number of daily returns. *Solution:* independent recomputation and a hint naming the window; the model repairs its code.
- **Unsupported claims in free prose.** A semantic guard still passed unsupported causes and quantifiers. *Solution:* `evidence-v1`: the model chooses the evidence and the key findings that lead the report; code writes every sentence.

## 5. Error handling and reliability

- **Network.** Timeouts, per-host rate limits, jittered backoff honouring `Retry-After`, caching, a fallback source for every data type, and circuit breakers.
- **Models and tools.** Provider failover under optional budgets; tool exceptions become structured errors with a `retryable` flag and a hint.
- **Fallback.** The deadline (420 s), turn limit, repeated guard rejections or model outages hand over to a deterministic node that sends a degraded report from validated data; a missing sandbox or the last failed analysis attempt goes there directly (26 s instead of 189 s without Docker).
- **Delivery.** A pre-send guard rechecks snapshot hashes and provenance; a SQLite ledger prevents duplicate sends on retry or resume and blocks automatic resend after an ambiguous SMTP result. Exit codes: 0, 2 (degraded), 1.
- **Responsible access.** Identifying User-Agent, no bot-challenge bypass; headlines are untrusted data.

**Validation.** Offline regressions include recorded 23 October data; Docker tests are opt-in. In 8 live synthetic scenarios (MiMo, DeepSeek) every report used the agent's own key findings; two first choices were rejected and corrected once. Test code and fixtures are versioned; generated evaluation results and historical archives remain local-only. Open: historical ranking accuracy, the uncalibrated 10% boundary band, human review.
