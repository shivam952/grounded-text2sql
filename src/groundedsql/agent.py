"""ReAct SQL agent — iterative tool-calling loop using chat.completions.

Architecture:

  The agent runs an iterative loop:
    1. Send messages to the LLM (including prior tool results).
    2. LLM either calls a tool or calls submit_result() to finish.
    3. If run_sql(): execute, append result, loop.
    4. If submit_result(): run grounding check.
       - If grounding passes → done.
       - If grounding fails → inject error as a tool result, loop continues
         (same self-correction mechanism as SQL errors).
    5. If max_iterations reached → force submit_result().

Key implementation notes:
  - Uses chat.completions (OpenRouter-compatible) instead of the OpenAI
    Responses API.  Tool result message format:
      {"role": "tool", "tool_call_id": ..., "content": ...}
  - Tool set: run_sql + submit_result only.
  - submit_result fields: answer, sql_used, confidence, grounding_summary.
  - Grounding check wired into the loop before submit_result is accepted.
"""
from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from openai import OpenAI

from groundedsql.config import settings
from groundedsql.db import run_readonly_sql
from groundedsql.grounding import check_grounding
from groundedsql.models import GroundingResult, QueryResult, ReActStep, ReActTrace
from groundedsql.observability import Tracer
from groundedsql.prompts import SYSTEM_PROMPT_TEMPLATE
from groundedsql.schema import schema_context

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions (OpenAI function-calling schema)
# ---------------------------------------------------------------------------

RUN_SQL_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "run_sql",
        "description": (
            "Execute a read-only SQLite SELECT query. "
            "Returns a JSON object with 'rows' (list of dicts), 'row_count' (int), "
            "and 'error' (null or string). "
            "Call it as many times as needed — probe first, then query for the answer. "
            "Errors are information: read them, fix the SQL, and retry."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sql": {
                    "type": "string",
                    "description": "A valid SQLite SELECT or WITH (CTE) statement.",
                }
            },
            "required": ["sql"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

SUBMIT_RESULT_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "submit_result",
        "description": (
            "Submit your final answer. This MUST be your very last tool call. "
            "Do not call run_sql after this."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": (
                        "2–4 plain sentences answering the question directly. "
                        "No raw SQL, no column names, no backticks."
                    ),
                },
                "sql_used": {
                    "type": "string",
                    "description": "The final SQL query that produced the answer.",
                },
                "confidence": {
                    "type": "number",
                    "description": (
                        "How confident you are the answer is correct and grounded "
                        "in the data. 0.0–1.0. Use 0.9+ only for unambiguous results."
                    ),
                },
                "grounding_summary": {
                    "type": "string",
                    "description": (
                        "One sentence: what data supports this answer. "
                        "E.g. 'Based on 45 rows from schools table filtered to county=Alameda.'"
                    ),
                },
            },
            "required": ["answer", "sql_used", "confidence", "grounding_summary"],
            "additionalProperties": False,
        },
        "strict": True,
    },
}

TOOLS = [RUN_SQL_TOOL, SUBMIT_RESULT_TOOL]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_args(arguments: str) -> dict[str, Any]:
    try:
        return json.loads(arguments or "{}")
    except json.JSONDecodeError:
        return {}


def _extract_step_annotation(content: str) -> str:
    """Pull the STEP: line written by the model before a tool call."""
    m = re.search(r"^STEP:\s*(.+)$", content or "", re.MULTILINE)
    return m.group(1).strip() if m else ""


def _is_reasoning_model(model: str) -> bool:
    """Reasoning models don't accept temperature=0."""
    m = model.lower()
    # o-series, claude-3-5 thinking variants — extend as needed
    return m.startswith("o1") or m.startswith("o3") or "thinking" in m


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class ReActSqlAgent:
    """Iterative ReAct loop: LLM discovers what it needs by querying live data.

    The agent operates entirely through two tools:
      - run_sql(): probe schema, run queries, observe results
      - submit_result(): structured final answer (gated by grounding check)

    Error feedback (SQL errors AND grounding failures) is injected as tool
    results, letting the LLM's own reasoning handle self-correction without
    any special retry logic in the host code.
    """

    def __init__(
        self,
        db_path: Path,
        model: str | None = None,
        client: OpenAI | None = None,
        tracer: Tracer | None = None,
        max_iterations: int | None = None,
    ) -> None:
        self.db_path = db_path
        self.model = model or settings.groundedsql_model
        self.client = client or settings.openai_client()
        self.tracer = tracer or Tracer.from_env()
        self.max_iterations = max_iterations if max_iterations is not None else settings.max_iterations

    def answer(
        self,
        question: str,
        on_step: Callable[[ReActTrace], None] | None = None,
    ) -> ReActTrace:
        """Run the agent loop and return a fully populated ReActTrace."""
        ctx = schema_context(self.db_path)
        system_content = SYSTEM_PROMPT_TEMPLATE.format(schema_context=ctx)

        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": question},
        ]

        # Accumulate all rows returned this turn for the grounding check
        all_rows_this_turn: list[dict[str, Any]] = []

        result = ReActTrace(question=question, answer="")
        total_usage: dict[str, int] = {"input_tokens": 0, "output_tokens": 0}

        with self.tracer.trace(
            input_payload={"question": question, "db": str(self.db_path.name)},
            metadata={"model": self.model},
        ) as root_obs:
            result.trace_id = self.tracer.trace_id

            for iteration in range(1, self.max_iterations + 1):
                result.iterations_used = iteration

                # Build request kwargs
                req: dict[str, Any] = {
                    "model": self.model,
                    "messages": messages,
                    "tools": TOOLS,
                    "tool_choice": (
                        {"type": "function", "function": {"name": "submit_result"}}
                        if iteration == self.max_iterations
                        else "auto"
                    ),
                    # Cap output tokens — some OpenRouter providers (e.g. Novita
                    # for Kimi K2) reject requests where max_tokens equals the
                    # full context window length rather than just the output budget.
                    # 4096 is plenty for a tool call + reasoning trace per iteration.
                    "max_tokens": 4096,
                }
                if not _is_reasoning_model(self.model):
                    req["temperature"] = 0

                with self.tracer.generation(
                    f"llm_iteration_{iteration}",
                    model=self.model,
                    input_payload=messages[-3:],  # last few msgs for token efficiency
                    metadata={"iteration": iteration},
                ) as gen_obs:
                    response = self.client.chat.completions.create(**req)

                    usage = response.usage
                    if usage:
                        total_usage["input_tokens"] += usage.prompt_tokens or 0
                        total_usage["output_tokens"] += usage.completion_tokens or 0

                    self.tracer.update_generation(
                        gen_obs,
                        usage_details={
                            "input": total_usage["input_tokens"],
                            "output": total_usage["output_tokens"],
                        },
                    )

                msg = response.choices[0].message
                step_annotation = _extract_step_annotation(msg.content or "")

                # Append assistant message (with tool_calls if any)
                messages.append(msg.model_dump(exclude_none=True))

                tool_calls = msg.tool_calls or []

                if not tool_calls:
                    # Model returned plain text without calling submit_result.
                    # Force a submit_result call to always get structured output.
                    logger.warning("Iteration %d: no tool call — forcing submit_result", iteration)
                    forced_req = {
                        "model": self.model,
                        "messages": messages,
                        "tools": [SUBMIT_RESULT_TOOL],
                        "tool_choice": {"type": "function", "function": {"name": "submit_result"}},
                    }
                    if not _is_reasoning_model(self.model):
                        forced_req["temperature"] = 0
                    forced_resp = self.client.chat.completions.create(**forced_req)
                    forced_msg = forced_resp.choices[0].message
                    if forced_msg.tool_calls:
                        tc = forced_msg.tool_calls[0]
                        args = _parse_args(tc.function.arguments)
                        draft_answer = str(args.get("answer") or msg.content or "")
                        # ── Grounding check — same gate as the normal path ──────
                        # This branch is where the model is behaving least reliably
                        # (it skipped tool-calling entirely).  Skipping the grounding
                        # check here would mean the one path where hallucination is
                        # most likely is also the one path without the guard.
                        grounding = check_grounding(
                            answer_text=draft_answer,
                            all_rows=all_rows_this_turn,
                            client=self.client,
                            grounding_model=settings.groundedsql_grounding_model,
                        )
                        result.answer = draft_answer
                        result.sql_used = str(args.get("sql_used") or "")
                        result.confidence = float(args.get("confidence") or 0.0)
                        result.grounding = grounding
                        if not grounding.passed:
                            result.grounding_interventions += 1
                            logger.warning(
                                "Grounding check failed on forced submit (iter %d) — "
                                "accepting with failed flag (model already deviated from tool protocol)",
                                iteration,
                            )
                    else:
                        result.answer = msg.content or ""
                    break

                submitted = False
                for tc in tool_calls:
                    result.total_tool_calls += 1
                    args = _parse_args(tc.function.arguments)

                    if submitted:
                        # Drain extra tool calls after submit_result in the same batch
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": '{"skipped": "submit_result already called"}',
                        })
                        continue

                    if tc.function.name == "submit_result":
                        draft_answer = str(args.get("answer") or "")
                        sql_used = str(args.get("sql_used") or "")
                        confidence = float(args.get("confidence") or 0.0)
                        grounding_summary = str(args.get("grounding_summary") or "")

                        # ── Grounding check ────────────────────────────────────
                        grounding = check_grounding(
                            answer_text=draft_answer,
                            all_rows=all_rows_this_turn,
                            client=self.client,
                            grounding_model=settings.groundedsql_grounding_model,
                        )

                        if not grounding.passed and iteration < self.max_iterations - 1:
                            # Reject and continue the loop — same pattern as SQL errors.
                            # The iteration guard (< max_iterations - 1) is intentional:
                            # on the final iteration or two, accepting the best available
                            # answer (with grounding.passed=False recorded on the trace)
                            # is preferable to exhausting the budget and returning nothing.
                            # The failed flag is visible in Langfuse and in ReActTrace.
                            result.grounding_interventions += 1
                            logger.info(
                                "Grounding check failed on iter %d — injecting rejection and continuing",
                                iteration,
                            )
                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps({"error": grounding.rejection_reason}),
                            })
                            with self.tracer.span(
                                f"grounding_rejection_{result.grounding_interventions}",
                                metadata={"flagged": grounding.flagged_claims, "iteration": iteration},
                            ):
                                pass
                            submitted = False
                            continue
                        # ── Accept ─────────────────────────────────────────────
                        result.answer = draft_answer
                        result.sql_used = sql_used
                        result.confidence = confidence
                        result.grounding = grounding

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": '{"ok": true}',
                        })
                        with self.tracer.span(
                            f"submit_result_{result.total_tool_calls}",
                            input_payload={
                                "answer": draft_answer[:120],
                                "confidence": confidence,
                                "grounding_passed": grounding.passed,
                            },
                            metadata={"iteration": iteration},
                        ) as span_obs:
                            self.tracer.update(span_obs, output={"answer": draft_answer[:120]})

                        if on_step:
                            on_step(result)
                        submitted = True
                        continue

                    # ── run_sql ────────────────────────────────────────────────
                    if tc.function.name == "run_sql":
                        sql = args.get("sql", "")
                        qr = run_readonly_sql(
                            self.db_path,
                            sql,
                            row_limit=settings.tool_row_limit,
                            timeout_ms=settings.query_timeout_ms,
                        )
                        all_rows_this_turn.extend(qr.rows)

                        result.steps.append(ReActStep(
                            iteration=iteration,
                            sql=sql,
                            row_count=len(qr.rows),
                            rows_sample=qr.rows[:5],
                            error=qr.error,
                            summary=step_annotation,
                        ))

                        tool_output = json.dumps(
                            {
                                "rows": qr.rows,
                                "row_count": len(qr.rows),
                                "truncated": qr.truncated,
                                "elapsed_ms": round(qr.elapsed_ms, 1),
                                "error": qr.error or None,
                            },
                            ensure_ascii=False,
                        )
                        with self.tracer.span(
                            f"run_sql_{result.total_tool_calls}",
                            input_payload={"sql": sql},
                            metadata={"iteration": iteration, "row_count": len(qr.rows)},
                        ) as span_obs:
                            self.tracer.update(span_obs, output={
                                "row_count": len(qr.rows),
                                "error": qr.error or None,
                                "elapsed_ms": round(qr.elapsed_ms, 1),
                            })

                        messages.append({
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": tool_output,
                        })
                        if on_step:
                            on_step(result)
                        continue

                    # Unknown tool
                    logger.warning("Unknown tool call: %r", tc.function.name)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps({"error": f"unknown tool '{tc.function.name}'"}),
                    })

                if submitted:
                    break

            else:
                # Max iterations exhausted
                result.answer = (
                    f"[max iterations {self.max_iterations} reached] "
                    "The agent could not find a complete answer within the iteration limit."
                )
                if on_step:
                    on_step(result)

            self.tracer.update(root_obs, output={
                "answer": result.answer[:200],
                "iterations": result.iterations_used,
                "tool_calls": result.total_tool_calls,
                "grounding_interventions": result.grounding_interventions,
                "total_tokens": sum(total_usage.values()),
            })
            result.trace_url = self.tracer.trace_url()
            self.tracer.flush()

        return result
