"""SQL validation guard — first layer of defence-in-depth.

Security note: the checks here are a best-effort UX guard that catches
accidental non-SELECT statements from the LLM and returns a clear error
message the agent can reason about.  The actual security boundary is the
SQLite `mode=ro` URI used in connect_readonly() in db.py, which prevents
all writes at the OS level regardless of what SQL text gets through here.
Both layers are kept because they serve different purposes.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


FORBIDDEN_SQL_RE = re.compile(
    r"\b("
    r"insert|update|delete|drop|alter|create|truncate|"
    r"attach|detach|pragma|vacuum|reindex|begin|commit|rollback"
    r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SqlValidation:
    ok: bool
    sql: str
    error: str = ""


def strip_sql_fence(sql: str) -> str:
    """Remove markdown code fences that the LLM sometimes wraps SQL in."""
    text = (sql or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("sql\n"):
        text = text[4:].strip()
    return text


_QUOTED_OR_COMMENT_RE = re.compile(
    r"'(?:[^']|'')*'"
    r'|"(?:[^"]|"")*"'
    r"|`(?:[^`]|``)*`"
    r"|\[[^\]]*\]"
    r"|--[^\n]*"
    r"|/\*.*?\*/",
    re.DOTALL,
)


def strip_quoted_and_comments(sql: str) -> str:
    """Replace string literals and comments with spaces for safe keyword scanning."""
    return _QUOTED_OR_COMMENT_RE.sub(" ", sql)


def validate_select_sql(sql: str) -> SqlValidation:
    """Validate that sql is a single read-only SELECT or CTE.

    Returns SqlValidation with ok=False and a descriptive error message if
    the query fails any check.  These error messages are fed back into the
    agent loop so the LLM can correct its SQL on the next iteration.
    """
    cleaned = strip_sql_fence(sql).strip()
    if not cleaned:
        return SqlValidation(False, cleaned, "SQL is empty.")

    without_trailing = cleaned[:-1].strip() if cleaned.endswith(";") else cleaned
    scan = strip_quoted_and_comments(without_trailing)

    if ";" in scan:
        return SqlValidation(False, cleaned, "Only one SQL statement is allowed.")

    lowered = scan.lstrip().lower()
    if not (lowered.startswith("select") or lowered.startswith("with")):
        return SqlValidation(False, cleaned, "Only read-only SELECT queries are allowed.")

    if FORBIDDEN_SQL_RE.search(scan):
        return SqlValidation(False, cleaned, "SQL contains a forbidden command.")

    return SqlValidation(True, without_trailing)


def enforce_limit(sql: str, limit: int = 200) -> str:
    """Append a LIMIT clause to the query if one is not already present.

    Handles both plain SELECT and WITH (CTE) queries correctly.
    SQLite does not allow a WITH clause inside a subquery, so CTE queries
    get the LIMIT appended directly instead of being wrapped.
    """
    validation = validate_select_sql(sql)
    if not validation.ok:
        raise ValueError(validation.error)
    cleaned = validation.sql
    if re.search(r"\blimit\s+\d+\b", cleaned, re.IGNORECASE):
        return cleaned
    if cleaned.lstrip().upper().startswith("WITH"):
        return f"{cleaned}\nLIMIT {int(limit)}"
    return f"SELECT * FROM (\n{cleaned}\n) AS _agent_query LIMIT {int(limit)}"
