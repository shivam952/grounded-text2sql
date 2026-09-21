"""Eval harness — runs the agent on BIRD mini-dev questions and computes metrics.

Metrics reported:
  - Execution Accuracy (EX): agent SQL result set matches gold SQL result set
    (after sorting both, as per BIRD standard).  This is directly comparable
    to published BIRD leaderboard numbers.
  - Average iterations per question
  - Average tool calls per question
  - Grounding check intervention rate (interventions / total)
  - Grounding check fix rate (interventions that led to correct EX / total interventions)

Output: JSONL to eval/results/run_<timestamp>.jsonl + summary table to stdout.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).parent / "results"


# ---------------------------------------------------------------------------
# Execution accuracy
# ---------------------------------------------------------------------------

def _run_gold_sql(db_path: Path, gold_sql: str) -> list[tuple] | None:
    """Execute the gold SQL and return sorted result rows, or None on error."""
    try:
        con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        rows = con.execute(gold_sql).fetchall()
        con.close()
        return sorted(rows)
    except Exception as exc:
        logger.warning("Gold SQL failed for %s: %s", db_path.name, exc)
        return None


def _run_agent_sql(db_path: Path, sql: str) -> list[tuple] | None:
    """Execute the agent's SQL and return sorted result rows, or None on error."""
    if not sql.strip():
        return None
    try:
        con = sqlite3.connect(f"file:{db_path.resolve().as_posix()}?mode=ro", uri=True)
        rows = con.execute(sql).fetchall()
        con.close()
        return sorted(rows)
    except Exception as exc:
        logger.warning("Agent SQL failed: %s | sql=%s", exc, sql[:80])
        return None


def execution_accuracy(
    db_path: Path,
    agent_sql: str,
    gold_sql: str,
) -> bool:
    """Return True if agent SQL and gold SQL produce identical sorted result sets."""
    gold_rows = _run_gold_sql(db_path, gold_sql)
    if gold_rows is None:
        return False
    agent_rows = _run_agent_sql(db_path, agent_sql)
    if agent_rows is None:
        return False
    return agent_rows == gold_rows


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def run_eval(
    db_dir: Path,
    questions_path: Path,
    n: int = 50,
    model: str | None = None,
    out_path: Path | None = None,
) -> None:
    from rich.console import Console
    from rich.table import Table
    from rich import box
    from groundedsql.agent import ReActSqlAgent
    from groundedsql.config import settings

    console = Console()

    # Load questions
    with open(questions_path) as f:
        questions: list[dict[str, Any]] = json.load(f)

    questions = questions[:n]
    console.print(f"\n[bold]GroundedSQL Eval Harness[/bold] — {len(questions)} questions")
    console.print(f"Model: [cyan]{model or settings.groundedsql_eval_model}[/cyan]\n")

    # Output file
    if out_path is None:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        out_path = RESULTS_DIR / f"run_{ts}.jsonl"

    eval_model = model or settings.groundedsql_eval_model

    # Counters
    correct = 0
    total_iterations = 0
    total_tool_calls = 0
    total_grounding_interventions = 0
    questions_with_interventions = 0
    questions_with_interventions_correct = 0  # questions with >=1 intervention that achieved correct EX
    errors = 0

    with open(out_path, "w") as out_f:
        for q in tqdm(questions, desc="Evaluating"):
            db_id = q.get("db_id", "")
            question_text = q.get("question", "")
            gold_sql = q.get("SQL", "")
            evidence = q.get("evidence", "")

            # Locate the database
            db_path = db_dir / db_id / f"{db_id}.sqlite"
            if not db_path.exists():
                logger.warning("DB not found: %s", db_path)
                errors += 1
                continue

            # Augment question with BIRD evidence field if present
            augmented_question = question_text
            if evidence.strip():
                augmented_question = f"{question_text}\n\nHint: {evidence}"

            agent = ReActSqlAgent(db_path=db_path, model=eval_model)

            try:
                t_start = time.perf_counter()
                trace = agent.answer(augmented_question)
                elapsed = time.perf_counter() - t_start
            except Exception as exc:
                logger.error("Agent error on %r: %s", question_text[:60], exc)
                errors += 1
                continue

            # Compute execution accuracy
            ex = execution_accuracy(db_path, trace.sql_used, gold_sql)

            if ex:
                correct += 1
            if trace.grounding_interventions > 0:
                questions_with_interventions += 1
                if ex:
                    questions_with_interventions_correct += 1

            total_iterations += trace.iterations_used
            total_tool_calls += trace.total_tool_calls
            total_grounding_interventions += trace.grounding_interventions

            record = {
                "db_id": db_id,
                "question": question_text,
                "gold_sql": gold_sql,
                "agent_sql": trace.sql_used,
                "agent_answer": trace.answer,
                "confidence": trace.confidence,
                "execution_accuracy": ex,
                "iterations": trace.iterations_used,
                "tool_calls": trace.total_tool_calls,
                "grounding_interventions": trace.grounding_interventions,
                "grounding_passed": trace.grounding.passed,
                "grounding_flagged": trace.grounding.flagged_claims,
                "elapsed_s": round(elapsed, 2),
            }
            out_f.write(json.dumps(record) + "\n")
            out_f.flush()

    # Summary
    total_evaluated = len(questions) - errors
    ex_pct = correct / total_evaluated if total_evaluated else 0
    avg_iters = total_iterations / total_evaluated if total_evaluated else 0
    avg_tools = total_tool_calls / total_evaluated if total_evaluated else 0
    intervention_rate = total_grounding_interventions / total_evaluated if total_evaluated else 0
    fix_rate = (
        questions_with_interventions_correct / questions_with_interventions
        if questions_with_interventions else 0
    )

    console.print()
    t = Table(title="[bold]Eval Results[/bold]", box=box.SIMPLE_HEAVY)
    t.add_column("Metric", style="bold")
    t.add_column("Value", justify="right")
    t.add_row("Questions evaluated", str(total_evaluated))
    t.add_row("Errors / skipped", str(errors))
    t.add_row("Execution Accuracy (EX)", f"[bold green]{ex_pct:.1%}[/bold green]")
    t.add_row("Avg iterations / question", f"{avg_iters:.1f}")
    t.add_row("Avg tool calls / question", f"{avg_tools:.1f}")
    t.add_row("Grounding interventions / q", f"{intervention_rate:.2f}")
    t.add_row("Grounding recovery rate", f"{fix_rate:.1%}" if questions_with_interventions else "n/a")
    t.add_row("Results file", str(out_path))
    console.print(t)

    # Benchmark note
    console.print(
        "\n[dim]Note: Zero-shot execution without few-shot examples or domain linking on BIRD mini-dev.[/dim]\n"
        "[dim]Compare against published benchmarks: https://bird-bench.github.io/[/dim]\n"
    )
