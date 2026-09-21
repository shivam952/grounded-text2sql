"""CLI entry point for GroundedSQL.

Usage:
    groundedsql ask "How many schools are in the database?" --db ./data/bird_mini_dev/databases/california_schools/california_schools.sqlite
    groundedsql eval --db-dir ./data/bird_mini_dev/databases --questions ./data/bird_mini_dev/mini_dev_sqlite.json --n 50
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import click
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.rule import Rule
from rich.syntax import Syntax
from rich.table import Table
from rich import box

from groundedsql.agent import ReActSqlAgent
from groundedsql.config import settings
from groundedsql.models import ReActTrace

console = Console()
logging.basicConfig(level=logging.WARNING)


def _render_trace(trace: ReActTrace) -> Panel:
    """Render a rich Panel summarising the current trace state."""
    lines: list[str] = []
    for step in trace.steps:
        icon = "✓" if not step.error else "✗"
        summary = f"  [dim]STEP:[/dim] {step.summary}" if step.summary else ""
        lines.append(f"[green]{icon}[/green] [bold]iter {step.iteration}[/bold]{summary}")
        if step.error:
            lines.append(f"   [red]error:[/red] {step.error[:120]}")
        else:
            lines.append(f"   [dim]{step.row_count} rows returned[/dim]")
    if trace.grounding_interventions:
        lines.append(f"[yellow]⚠ grounding check fired {trace.grounding_interventions}×[/yellow]")
    if trace.answer:
        lines.append("")
        lines.append(f"[bold green]Answer:[/bold green] {trace.answer}")
    return Panel(
        "\n".join(lines) or "[dim]Starting…[/dim]",
        title=f"[bold]GroundedSQL[/bold] [dim]— {trace.question[:60]}[/dim]",
        border_style="blue",
    )


@click.group()
def main() -> None:
    """GroundedSQL — ReAct SQL agent with grounding verification."""


@main.command()
@click.argument("question")
@click.option("--db", required=True, type=click.Path(exists=True, path_type=Path), help="Path to SQLite database.")
@click.option("--model", default=None, help="OpenRouter model slug (overrides GROUNDEDSQL_MODEL).")
@click.option("--no-trace", is_flag=True, help="Disable Langfuse tracing even if keys are set.")
@click.option("--json-out", is_flag=True, help="Output result as JSON.")
def ask(question: str, db: Path, model: str | None, no_trace: bool, json_out: bool) -> None:
    """Ask a natural-language question about a SQLite database."""
    if not settings.openrouter_api_key or settings.openrouter_api_key == "sk-or-...":
        console.print("[bold red]Error:[/bold red] OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in your key.")
        sys.exit(1)

    from groundedsql.observability import Tracer
    tracer = Tracer() if no_trace else Tracer.from_env()

    agent = ReActSqlAgent(db_path=db, model=model, tracer=tracer)
    trace: ReActTrace = None  # type: ignore[assignment]

    with Live(console=console, refresh_per_second=4) as live:
        def on_step(t: ReActTrace, step_text: str) -> None:
            live.update(_render_trace(t))

        start = time.perf_counter()
        trace = agent.answer(question, on_step=on_step)
        elapsed = time.perf_counter() - start

    if json_out:
        out = {
            "question": trace.question,
            "answer": trace.answer,
            "sql_used": trace.sql_used,
            "confidence": trace.confidence,
            "iterations": trace.iterations_used,
            "tool_calls": trace.total_tool_calls,
            "grounding_interventions": trace.grounding_interventions,
            "grounding_passed": trace.grounding.passed,
            "trace_url": trace.trace_url,
            "elapsed_s": round(elapsed, 2),
        }
        click.echo(json.dumps(out, indent=2))
        return

    # Rich formatted output
    console.print()
    console.print(Rule("[bold blue]GroundedSQL Result[/bold blue]"))
    console.print(f"\n[bold]{trace.answer}[/bold]\n")

    meta = Table(box=box.SIMPLE, show_header=False, padding=(0, 1))
    meta.add_row("[dim]Confidence[/dim]", f"{trace.confidence:.0%}")
    meta.add_row("[dim]Iterations[/dim]", str(trace.iterations_used))
    meta.add_row("[dim]Tool calls[/dim]", str(trace.total_tool_calls))
    meta.add_row("[dim]Grounding checks[/dim]", f"{trace.grounding_interventions} intervention(s)")
    meta.add_row("[dim]Grounding passed[/dim]", "✓" if trace.grounding.passed else "✗")
    meta.add_row("[dim]Elapsed[/dim]", f"{elapsed:.1f}s")
    if trace.trace_url:
        meta.add_row("[dim]Trace[/dim]", trace.trace_url)
    console.print(meta)

    if trace.sql_used:
        console.print()
        console.print(Syntax(trace.sql_used, "sql", theme="monokai", line_numbers=False))

    if trace.grounding.flagged_claims:
        console.print(f"\n[yellow]Grounding flagged values:[/yellow] {trace.grounding.flagged_claims}")


@main.command()
@click.option("--db-dir", required=True, type=click.Path(exists=True, path_type=Path), help="Directory containing BIRD database folders.")
@click.option("--questions", required=True, type=click.Path(exists=True, path_type=Path), help="BIRD mini_dev JSON questions file.")
@click.option("--n", default=50, show_default=True, help="Number of questions to evaluate.")
@click.option("--model", default=None, help="Model slug override (default: GROUNDEDSQL_EVAL_MODEL).")
@click.option("--out", default=None, help="Output JSONL file path (default: eval/results/run_<timestamp>.jsonl).")
def eval(db_dir: Path, questions: Path, n: int, model: str | None, out: str | None) -> None:
    """Run the eval harness on BIRD mini-dev questions."""
    import importlib.util, pathlib
    _runner_path = pathlib.Path(__file__).parent.parent.parent / "eval" / "runner.py"
    _spec = importlib.util.spec_from_file_location("eval.runner", _runner_path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    run_eval = _mod.run_eval
    run_eval(db_dir=db_dir, questions_path=questions, n=n, model=model, out_path=Path(out) if out else None)


@main.command()
@click.option("--host", default="0.0.0.0", show_default=True, help="Host to bind to.")
@click.option("--port", default=8080, show_default=True, help="Port to bind to.")
@click.option("--reload", is_flag=True, help="Enable auto-reload for development.")
def serve(host: str, port: int, reload: bool) -> None:
    """Run the FastAPI demo server."""
    import uvicorn
    console.print(f"[bold blue]Starting GroundedSQL API server on http://{host}:{port}[/bold blue]")
    uvicorn.run("groundedsql.api:app", host=host, port=port, reload=reload)


if __name__ == "__main__":
    main()
