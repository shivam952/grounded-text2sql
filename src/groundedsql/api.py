"""FastAPI wrapper around ReActSqlAgent — exposes /ask, /health, and / (demo UI).

Design constraints (per DEPLOY.md):
- Only allowlisted bundled DBs — no arbitrary path from caller.
- Public demo always uses the cheap eval model, not the frontier model.
- Rate limiting per IP via slowapi.
- Hard daily request budget circuit breaker (in-memory, resets on restart).
- max_iterations capped at demo-safe value via env var.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from groundedsql.agent import ReActSqlAgent
from groundedsql.config import settings

logger = logging.getLogger(__name__)

# ── Demo DB allowlist ──────────────────────────────────────────────────────────
_DEMO_DATA = Path(__file__).parent / "demo_data"
DB_ALLOWLIST: dict[str, Path] = {
    "student_club": _DEMO_DATA / "student_club.sqlite",
    "superhero": _DEMO_DATA / "superhero.sqlite",
}
DB_DESCRIPTIONS: dict[str, str] = {
    "student_club": "University student club management — members, events, budgets",
    "superhero": "Comic book superheroes — powers, attributes, publishers",
}

# ── Cost / abuse protection ────────────────────────────────────────────────────
# Forced cheap model for the public demo regardless of GROUNDEDSQL_MODEL env var.
DEMO_MODEL = os.getenv("DEMO_MODEL", settings.groundedsql_eval_model)

# Max iterations for public demo — keep tight, 8 is plenty to see the mechanism.
DEMO_MAX_ITERATIONS = int(os.getenv("DEMO_MAX_ITERATIONS", "8"))

# Hard daily budget circuit breaker — in-memory, resets on process restart.
DAILY_REQUEST_CAP = int(os.getenv("DAILY_REQUEST_CAP", "100"))
_budget_lock = threading.Lock()
_budget_state: dict[str, int | str] = {"count": 0, "date": ""}


def _check_and_increment_budget() -> None:
    """Increment the daily request counter; raise 503 if cap is exceeded."""
    today = datetime.date.today().isoformat()
    with _budget_lock:
        if _budget_state["date"] != today:
            _budget_state["count"] = 0
            _budget_state["date"] = today
        if _budget_state["count"] >= DAILY_REQUEST_CAP:
            raise HTTPException(
                status_code=503,
                detail=f"Daily request cap of {DAILY_REQUEST_CAP} reached. Try again tomorrow.",
            )
        _budget_state["count"] += 1  # type: ignore[operator]


# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="GroundedSQL Demo", version="0.1.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

RATE_LIMIT = os.getenv("RATE_LIMIT", "5/minute")


# ── Request / response models ──────────────────────────────────────────────────
class AskRequest(BaseModel):
    question: str
    db: str  # must be a key in DB_ALLOWLIST


class AskResponse(BaseModel):
    answer: str
    sql_used: str
    confidence: float
    iterations: int
    tool_calls: int
    grounding_interventions: int
    grounding_passed: bool
    elapsed_s: float


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
@limiter.limit(RATE_LIMIT)
async def ask(request: Request, body: AskRequest) -> AskResponse:
    _check_and_increment_budget()

    if body.db not in DB_ALLOWLIST:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown db '{body.db}'. Allowed: {list(DB_ALLOWLIST)}",
        )
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    db_path = DB_ALLOWLIST[body.db]
    agent = ReActSqlAgent(
        db_path=db_path,
        model=DEMO_MODEL,
        max_iterations=DEMO_MAX_ITERATIONS,
    )

    import time
    t0 = time.perf_counter()
    try:
        trace = agent.answer(body.question)
    except Exception as exc:
        logger.error("Agent error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail="Something went wrong processing this question.") from exc
    elapsed = round(time.perf_counter() - t0, 2)

    logger.info(
        "ask question=%r db=%s model=%s iters=%d grounding_interventions=%d elapsed=%.1fs",
        body.question[:80],
        body.db,
        DEMO_MODEL,
        trace.iterations_used,
        trace.grounding_interventions,
        elapsed,
    )

    return AskResponse(
        answer=trace.answer,
        sql_used=trace.sql_used,
        confidence=trace.confidence,
        iterations=trace.iterations_used,
        tool_calls=trace.total_tool_calls,
        grounding_interventions=trace.grounding_interventions,
        grounding_passed=trace.grounding.passed,
        elapsed_s=elapsed,
    )


@app.post("/ask/stream")
@limiter.limit(RATE_LIMIT)
async def ask_stream(request: Request, body: AskRequest) -> StreamingResponse:
    """Same as /ask, but emits each ReAct loop step as an SSE event as it happens.

    The agent's LLM calls are synchronous (blocking), so the agent runs on a
    background thread; its on_step callback pushes one-line progress updates
    onto a queue that this generator drains and forwards as SSE frames. This
    is a demo-scale pattern (one thread per request) — fine given the existing
    rate limit and daily cap already bound concurrency, not meant to scale to
    high traffic without a proper async task queue.
    """
    _check_and_increment_budget()

    if body.db not in DB_ALLOWLIST:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown db '{body.db}'. Allowed: {list(DB_ALLOWLIST)}",
        )
    if not body.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    db_path = DB_ALLOWLIST[body.db]
    question = body.question
    db_key = body.db

    def generate():
        agent = ReActSqlAgent(
            db_path=db_path,
            model=DEMO_MODEL,
            max_iterations=DEMO_MAX_ITERATIONS,
        )
        q: queue.Queue = queue.Queue()
        outcome: dict = {}
        t0 = time.perf_counter()

        def on_step(trace, step_text: str) -> None:
            q.put({"type": "step", "text": step_text, "iteration": trace.iterations_used})

        def run() -> None:
            try:
                outcome["trace"] = agent.answer(question, on_step=on_step)
            except Exception as exc:
                logger.error("Agent error (stream): %s", exc, exc_info=True)
                outcome["error"] = "Something went wrong processing this question."
            finally:
                q.put(None)  # sentinel: done

        worker = threading.Thread(target=run, daemon=True)
        worker.start()

        while True:
            item = q.get()
            if item is None:
                break
            yield f"data: {json.dumps(item)}\n\n"

        worker.join()
        elapsed = round(time.perf_counter() - t0, 2)

        if "error" in outcome:
            yield f"data: {json.dumps({'type': 'error', 'detail': outcome['error']})}\n\n"
            return

        trace = outcome["trace"]
        logger.info(
            "ask/stream question=%r db=%s model=%s iters=%d grounding_interventions=%d elapsed=%.1fs",
            question[:80],
            db_key,
            DEMO_MODEL,
            trace.iterations_used,
            trace.grounding_interventions,
            elapsed,
        )
        final = {
            "type": "final",
            "answer": trace.answer,
            "sql_used": trace.sql_used,
            "confidence": trace.confidence,
            "iterations": trace.iterations_used,
            "tool_calls": trace.total_tool_calls,
            "grounding_interventions": trace.grounding_interventions,
            "grounding_passed": trace.grounding.passed,
            "elapsed_s": elapsed,
        }
        yield f"data: {json.dumps(final)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ── Demo UI ────────────────────────────────────────────────────────────────────
_DB_OPTIONS = "\n".join(
    f'<option value="{k}">{k} — {v}</option>'
    for k, v in DB_DESCRIPTIONS.items()
)

_HTML = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GroundedSQL — Live Demo</title>
<style>
  :root {{
    --bg: #0d1117; --surface: #161b22; --border: #30363d;
    --accent: #58a6ff; --text: #e6edf3; --sub: #8b949e;
    --green: #3fb950; --red: #f85149;
  }}
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ background: var(--bg); color: var(--text); font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; min-height: 100vh; display: flex; flex-direction: column; align-items: center; padding: 2rem 1rem; }}
  h1 {{ font-size: 1.6rem; font-weight: 700; margin-bottom: .25rem; }}
  .sub {{ color: var(--sub); font-size: .9rem; margin-bottom: 2rem; }}
  .card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1.5rem; width: 100%; max-width: 680px; }}
  label {{ display: block; font-size: .8rem; color: var(--sub); margin-bottom: .4rem; text-transform: uppercase; letter-spacing: .05em; }}
  select, textarea {{ width: 100%; background: var(--bg); border: 1px solid var(--border); border-radius: 6px; color: var(--text); padding: .6rem .8rem; font-size: .95rem; margin-bottom: 1rem; }}
  textarea {{ resize: vertical; min-height: 80px; font-family: inherit; }}
  button {{ background: var(--accent); color: #0d1117; font-weight: 700; border: none; border-radius: 6px; padding: .65rem 1.5rem; font-size: .95rem; cursor: pointer; width: 100%; }}
  button:disabled {{ opacity: .5; cursor: default; }}
  #result {{ margin-top: 1.5rem; display: none; }}
  .answer {{ font-size: 1.05rem; line-height: 1.6; margin-bottom: 1rem; padding-bottom: 1rem; border-bottom: 1px solid var(--border); }}
  .sql {{ background: var(--bg); border: 1px solid var(--border); border-radius: 6px; padding: .8rem; font-family: monospace; font-size: .85rem; white-space: pre-wrap; word-break: break-all; margin-bottom: 1rem; }}
  .meta {{ display: flex; flex-wrap: wrap; gap: .5rem; }}
  .pill {{ background: var(--bg); border: 1px solid var(--border); border-radius: 20px; padding: .25rem .75rem; font-size: .8rem; color: var(--sub); }}
  .pill.pass {{ border-color: var(--green); color: var(--green); }}
  .pill.fail {{ border-color: var(--red); color: var(--red); }}
  .error {{ color: var(--red); padding: .6rem; background: rgba(248,81,73,.1); border-radius: 6px; border: 1px solid var(--red); }}
  .spinner {{ display: inline-block; width: 14px; height: 14px; border: 2px solid currentColor; border-top-color: transparent; border-radius: 50%; animation: spin .7s linear infinite; vertical-align: middle; margin-right: .4rem; }}
  @keyframes spin {{ to {{ transform: rotate(360deg); }} }}
  #thinking {{ display: none; margin-top: 1.5rem; padding-bottom: .5rem; }}
  #thinking .label {{ font-size: .8rem; color: var(--sub); text-transform: uppercase; letter-spacing: .05em; margin-bottom: .5rem; }}
  .think-step {{ display: flex; align-items: baseline; gap: .5rem; font-size: .85rem; color: var(--sub); padding: .25rem 0; font-family: monospace; animation: fadeIn .3s ease; }}
  .think-step .n {{ color: var(--accent); flex-shrink: 0; }}
  .think-step.latest {{ color: var(--text); }}
  @keyframes fadeIn {{ from {{ opacity: 0; transform: translateY(-2px); }} to {{ opacity: 1; transform: translateY(0); }} }}
  footer {{ color: var(--sub); font-size: .8rem; margin-top: 2.5rem; text-align: center; }}
  footer a {{ color: var(--accent); text-decoration: none; }}
</style>
</head>
<body>
<h1>GroundedSQL</h1>
<p class="sub">ReAct text-to-SQL with runtime hallucination detection · <a href="https://github.com/shivam952/grounded-text2sql" target="_blank">GitHub</a></p>
<div class="card">
  <label for="db">Database</label>
  <select id="db">
    {_DB_OPTIONS}
  </select>
  <label for="question">Question</label>
  <textarea id="question" placeholder="e.g. How many members does the Anime club have?"></textarea>
  <button id="btn" onclick="ask()">Ask</button>
  <div id="thinking"><div class="label">Agent thinking</div><div id="thinkingSteps"></div></div>
  <div id="result"></div>
</div>
<footer>Powered by <a href="https://openrouter.ai" target="_blank">OpenRouter</a> · Zero-shot ReAct · Grounding check on every answer</footer>
<script>
async function ask() {{
  const btn = document.getElementById('btn');
  const q = document.getElementById('question').value.trim();
  const db = document.getElementById('db').value;
  const out = document.getElementById('result');
  const thinking = document.getElementById('thinking');
  const thinkingSteps = document.getElementById('thinkingSteps');
  if (!q) return;

  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Thinking…';
  out.style.display = 'none';
  thinkingSteps.innerHTML = '';
  thinking.style.display = 'block';

  try {{
    const r = await fetch('/ask/stream', {{
      method: 'POST',
      headers: {{'Content-Type': 'application/json'}},
      body: JSON.stringify({{question: q, db}})
    }});
    if (!r.ok) {{
      const data = await r.json().catch(() => ({{}}));
      out.innerHTML = `<div class="error">${{data.detail || r.statusText}}</div>`;
      out.style.display = 'block';
      return;
    }}

    const reader = r.body.getReader();
    const decoder = new TextDecoder();
    let buf = '';
    let stepNum = 0;

    while (true) {{
      const {{ done, value }} = await reader.read();
      if (done) break;
      buf += decoder.decode(value, {{ stream: true }});
      const frames = buf.split('\\n\\n');
      buf = frames.pop();  // last (possibly incomplete) frame stays in buf
      for (const frame of frames) {{
        const line = frame.split('\\n').find(l => l.startsWith('data: '));
        if (!line) continue;
        const evt = JSON.parse(line.slice(6));
        if (evt.type === 'step') {{
          stepNum++;
          document.querySelectorAll('.think-step.latest').forEach(el => el.classList.remove('latest'));
          const div = document.createElement('div');
          div.className = 'think-step latest';
          div.innerHTML = `<span class="n">${{stepNum}}</span><span>${{escHtml(evt.text)}}</span>`;
          thinkingSteps.appendChild(div);
        }} else if (evt.type === 'error') {{
          out.innerHTML = `<div class="error">${{escHtml(evt.detail)}}</div>`;
          out.style.display = 'block';
        }} else if (evt.type === 'final') {{
          const gp = evt.grounding_passed;
          const gi = evt.grounding_interventions;
          out.innerHTML = `
            <div class="answer">${{escHtml(evt.answer)}}</div>
            <div class="sql">${{escHtml(evt.sql_used)}}</div>
            <div class="meta">
              <span class="pill">${{evt.iterations}} iter</span>
              <span class="pill">${{evt.elapsed_s}}s</span>
              <span class="pill">${{Math.round(evt.confidence * 100)}}% confidence</span>
              <span class="pill ${{gp ? 'pass' : 'fail'}}">${{gp ? '✓' : '✗'}} grounding</span>
              ${{gi > 0 ? `<span class="pill">${{gi}} intervention${{gi > 1 ? 's' : ''}}</span>` : ''}}
            </div>`;
          out.style.display = 'block';
        }}
      }}
    }}
  }} catch (e) {{
    out.innerHTML = `<div class="error">Network error: ${{e.message}}</div>`;
    out.style.display = 'block';
  }} finally {{
    btn.disabled = false;
    btn.textContent = 'Ask';
  }}
}}
function escHtml(s) {{
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;');
}}
document.getElementById('question').addEventListener('keydown', e => {{
  if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) ask();
}});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index() -> HTMLResponse:
    return HTMLResponse(_HTML)
