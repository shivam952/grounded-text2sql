# GroundedSQL

A production-grade ReAct SQL agent with **hallucination/grounding detection** and a formal eval harness, benchmarked on [BIRD](https://bird-bench.github.io/).

Built to demonstrate that text-to-SQL agents need more than just good SQL generation — they need *grounding verification* to be trustworthy in production.

**[Try the Live Demo](https://grounded-text2sql.fly.dev/)** · **[GitHub Repository](https://github.com/shivam952/grounded-text2sql)**

---

## What this is

Most text-to-SQL demos generate SQL and print whatever the model says. This agent does something different:

1. **ReAct loop**: Iteratively probes the database — checks the schema, confirms column values before filtering, then writes the main query. Self-corrects on SQL errors using the same feedback mechanism.

2. **Grounding check**: Before accepting the final answer, every numeric claim is verified against the data that was *actually returned* by `run_sql()`. If the model states a number that isn't in the rows, the answer is rejected and the loop continues — forcing the agent to re-examine the data. This catches hallucinated statistics, misread aggregations, and confident-but-wrong summaries.

3. **Formal eval**: Runs on BIRD mini-dev questions, computes Execution Accuracy (the BIRD benchmark metric), and tracks grounding intervention rate — the proof the mechanism works, not just that it exists.

---

## Architecture

```
groundedsql/
  config.py          # pydantic-settings: OpenRouter keys, model slugs, tuning params
  models.py          # shared dataclasses: QueryResult, GroundingResult, ReActTrace
  db.py              # read-only SQLite layer (mode=ro URI + progress-handler timeout)
  sql_guard.py       # UX-layer SQL validation (defence in depth; real boundary is mode=ro)
  schema.py          # dynamic schema introspection via PRAGMA (no hand-authored descriptions)
  prompts.py         # system prompt template (domain-agnostic, modular blocks)
  grounding.py       # hallucination check: numeric extraction + optional LLM-as-judge
  agent.py           # ReAct loop (chat.completions / OpenRouter-compatible)
  observability.py   # Langfuse Tracer wrapper (optional; no-op if keys absent)
  api.py             # FastAPI demo server + web UI + rate limiting + daily budget guard
  cli.py             # `groundedsql ask`, `eval`, and `serve` commands
eval/
  download_bird.py   # downloads BIRD mini-dev to data/
  runner.py          # eval harness: EX accuracy, iterations, grounding stats
tests/
  test_agent.py      # 5 tests (golden ReAct loop, retries, self-correction)
  test_sql_guard.py  # 19 tests
  test_schema.py     # 12 tests (real SQLite fixture)
  test_grounding.py  # 18 tests
  test_api.py        # 6 tests (endpoints, rate limits, daily budget cap)
```

### Security: two-layer defence-in-depth

| Layer | What it does |
|---|---|
| `sql_guard.validate_select_sql()` | Regex check — catches accidental non-SELECT statements from the LLM and returns a clear error message the agent can reason about. Best-effort UX guard. |
| `connect_readonly()` — SQLite `mode=ro` URI | OS-level refusal of all writes. This is the real security boundary. Works even if the guard is somehow bypassed. |

### Dynamic schema introspection

Instead of hand-writing table descriptions, `schema.py` auto-generates context from:
- `PRAGMA table_info` — column names, types, nullability, PKs
- `PRAGMA foreign_key_list` — FK relationships (auto-inferred join paths)
- `SELECT * … LIMIT 3` — sample rows (values truncated for token efficiency)

This means the agent works on **any** SQLite database dropped in — not just one it was hand-tuned for.

---

## Quick start

```bash
git clone <repo>
cd grounded-text2sql
cp .env.example .env
# Fill in OPENROUTER_API_KEY at https://openrouter.ai
uv sync
```

### Ask a single question

```bash
# Download BIRD first
python eval/download_bird.py

groundedsql ask "How many schools are in Alameda county?" \
  --db data/bird_mini_dev/databases/california_schools/california_schools.sqlite
```

Output:

```
GroundedSQL Result
──────────────────
There are 52 schools in Alameda county according to the database.

  Confidence   87%
  Iterations   3
  Tool calls   4
  Grounding    ✓ (0 interventions)
  Elapsed      6.2s

SELECT COUNT(*) FROM schools WHERE County = 'Alameda'
```

### Run the eval harness

```bash
groundedsql eval \
  --db-dir data/bird_mini_dev/databases \
  --questions data/bird_mini_dev/mini_dev_sqlite.json \
  --n 50
```

### Launch the local demo server & UI

```bash
groundedsql serve --port 8080
# Open http://localhost:8080 in your browser
```

---

## Eval results

Evaluated against 50 questions from BIRD mini-dev across complex domain databases (`debit_card_specializing`, `student_club`) using **`deepseek/deepseek-chat-v3-0324`** as a cost-effective zero-shot baseline.

| Metric | GroundedSQL (`deepseek-chat-v3`) | Description / Context |
|---|---|---|
| **Execution Accuracy (EX)** | **38.0%** (19/50) | Zero-shot ReAct without domain-specific few-shot examples or schema linking hints |
| **Avg iterations / question** | **3.8** | Multi-step probing (schema inspection, distinct value checking) before final query |
| **Avg tool calls / question** | **3.8** | Concise tool execution budget |
| **Grounding interventions** | **40 total** (0.80 / q across 24 questions) | Real-time rejection of hallucinated / miscalculated numeric claims |
| **Grounding recovery rate** | **37.5%** (9 / 24 questions) | Percentage of questions where grounding intervention steered the agent to a correct final answer |

### Breakdown by Database

- **`student_club`**: **50.0%** EX (10/20) — 5 grounding interventions
- **`debit_card_specializing`**: **30.0%** EX (9/30) — 35 grounding interventions (heavy date manipulation & multi-table financial aggregates)

---

## Architectural Insight: Consistency vs. Correctness

Our evaluation surfaced a critical property of runtime grounding checks:

> **Grounding is a *Consistency Check*, not a *Correctness Oracle*.**

- **What Grounding Guarantees**: Any number or statistic presented in the final natural language answer is strictly verified against the records returned by `run_sql()`. If an agent hallucinates a figure or misreads an aggregation table, the check rejects the answer and forces self-correction.
- **What Grounding Cannot Guarantee**: Grounding cannot verify whether the executed SQL queried the *intended* business logic. On complex financial datasets like `debit_card_specializing` (35 interventions across 30 questions, yet only 30% EX), an agent can write a syntactically valid query joining the wrong table or filtering the wrong date range, receive real SQLite rows, and formulate an answer that is 100% grounded in those rows — but still semantically wrong with respect to the user's intent.

Foregrounding this boundary is essential: runtime grounding stops **hallucination of data**, while schema introspection, domain few-shots, and prompt engineering address **correctness of intent**.

---

## Path to Production

This project implements core reliability and cost guardrails for public demonstration, while maintaining a clear roadmap for enterprise deployment:

### Implemented Guardrails & Reliability
- **Retry with Exponential Backoff**: LLM calls in `agent.py` automatically retry transient errors (`429` rate limits, `503` timeouts) with jitter.
- **Safety Alerting**: Grounding judge failures log explicit `ALERT [GroundingSafety]` messages rather than failing silently.
- **Cost Protection**: Per-IP rate limiting (`slowapi`), forced lightweight eval model, lowered iteration budget (8), and a daily request circuit breaker (`503` once cap reached).
- **CI/CD & Golden Testing**: Automated GitHub Actions CI workflow running 60 unit, integration, and golden multi-step ReAct simulation tests.

### Enterprise Scale Roadmap
- **Schema Linking / Vector RAG**: For databases with 100+ tables, introspecting full schemas exceeds context limits. Enterprise scale requires embedding-based schema retrieval (BM25 / vector search) to inject only top-k relevant tables.
- **Distributed State**: Moving the daily request cap and rate limiter from single-node in-memory storage to a shared Redis/Upstash cluster.
- **Tenant Authentication & Security**: API key bearer token authentication (`Authorization: Bearer <key>`) with per-tenant usage quotas and prompt-injection guardrails.
- **Prompt & Eval Regression CI**: Running automated eval runs against golden benchmark slices on PRs to catch accuracy regressions before deploying prompt modifications.

---

## Configuration

All settings via environment variables or `.env`:

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *required* | Your OpenRouter key |
| `GROUNDEDSQL_MODEL` | `anthropic/claude-sonnet-4-5` | Main agent model |
| `GROUNDEDSQL_EVAL_MODEL` | `google/gemini-2.0-flash-001` | Bulk eval model (~10× cheaper) |
| `GROUNDEDSQL_GROUNDING_MODEL` | `google/gemini-2.0-flash-001` | LLM-as-judge for grounding |
| `LANGFUSE_PUBLIC_KEY` | *(optional)* | Langfuse observability |
| `LANGFUSE_SECRET_KEY` | *(optional)* | Langfuse observability |
| `LANGFUSE_BASE_URL` | *(optional)* | Langfuse host (cloud or self-hosted) |

---

## Running tests

```bash
uv run pytest tests/ -v
# 60 tests, all passing, no network calls required
```

---

## Observability

If Langfuse keys are set, every `ask` call creates a trace with:
- Full thought → tool call → observation chain reconstructed
- Per-iteration generation spans with token usage
- Grounding check outcome + flagged values
- `trace_url` returned in the result and shown in the CLI

The agent runs identically without Langfuse — tracing degrades silently to no-ops.

---

## License

MIT
