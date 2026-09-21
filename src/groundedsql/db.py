"""Read-only SQLite query layer.

Security design (two-layer defence-in-depth):
  1. sql_guard.validate_select_sql() — regex UX check that catches accidental
     non-SELECT statements from the LLM before they reach the database.
     This is a best-effort guard, not the real security boundary.
  2. connect_readonly() opens SQLite with `mode=ro` in the URI, which causes
     the OS to refuse all write operations regardless of the SQL text sent.
     This is the actual security boundary.

Both layers are kept because they serve different purposes: the guard gives
the agent a clear error message it can reason about, while the URI mode
ensures writes are impossible even if the guard is somehow bypassed.
"""
from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any

from groundedsql.models import QueryResult
from groundedsql.sql_guard import enforce_limit, validate_select_sql


def connect_readonly(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite database in read-only mode.

    The `mode=ro` URI parameter is the real security boundary — it prevents
    all writes at the OS/VFS level regardless of what SQL text is sent.
    """
    uri = f"file:{db_path.resolve().as_posix()}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=2.0)
    con.row_factory = sqlite3.Row
    return con


def run_readonly_sql(
    db_path: Path,
    sql: str,
    row_limit: int = 200,
    timeout_ms: int = 10_000,
) -> QueryResult:
    """Validate, execute, and return results for a read-only SQL query.

    Args:
        db_path:    Path to the SQLite database file.
        sql:        SQL text (SELECT or CTE only — validated by sql_guard).
        row_limit:  Maximum rows returned; additional rows are dropped and
                    QueryResult.truncated is set to True.
        timeout_ms: Hard query timeout; interrupted queries set .timed_out=True.

    Returns:
        QueryResult with rows, timing, truncation flag, and any error message.
        Errors are returned as structured data rather than raised so the agent
        loop can feed them back into the LLM as tool output.
    """
    validation = validate_select_sql(sql)
    if not validation.ok:
        return QueryResult(sql=sql, rows=[], elapsed_ms=0.0, truncated=False, error=validation.error)

    limited_sql = enforce_limit(validation.sql, row_limit + 1)
    started = time.perf_counter()
    con = connect_readonly(db_path)
    deadline = started + (timeout_ms / 1000.0)

    def progress_handler() -> int:
        # Return non-zero to interrupt the query when the deadline passes.
        return 1 if time.perf_counter() > deadline else 0

    try:
        con.set_progress_handler(progress_handler, 1000)
        rows = [dict(row) for row in con.execute(limited_sql).fetchall()]
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        truncated = len(rows) > row_limit
        return QueryResult(
            sql=limited_sql,
            rows=rows[:row_limit],
            elapsed_ms=elapsed_ms,
            truncated=truncated,
        )
    except sqlite3.OperationalError as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        timed_out = "interrupted" in str(exc).lower()
        error_msg = "query timed out" if timed_out else str(exc)
        return QueryResult(
            sql=limited_sql,
            rows=[],
            elapsed_ms=elapsed_ms,
            truncated=False,
            error=error_msg,
            timed_out=timed_out,
        )
    except sqlite3.Error as exc:
        elapsed_ms = (time.perf_counter() - started) * 1000.0
        return QueryResult(sql=limited_sql, rows=[], elapsed_ms=elapsed_ms, truncated=False, error=str(exc))
    finally:
        con.close()
