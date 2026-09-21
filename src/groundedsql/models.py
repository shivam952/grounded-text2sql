"""Shared dataclasses / result types used across the agent pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ── DB layer ───────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class QueryResult:
    sql: str
    rows: list[dict[str, Any]]
    elapsed_ms: float
    truncated: bool
    error: str = ""
    timed_out: bool = False


# ── Grounding ──────────────────────────────────────────────────────────────────

@dataclass
class GroundingResult:
    passed: bool
    flagged_claims: list[str] = field(default_factory=list)
    verified_by_llm: bool = False
    rejection_reason: str = ""


# ── Agent trace ────────────────────────────────────────────────────────────────

@dataclass
class ReActStep:
    """One SQL tool call within the agent loop."""
    iteration: int
    sql: str
    row_count: int
    rows_sample: list[dict[str, Any]]
    error: str
    summary: str = ""  # STEP: annotation written by the model before the tool call


@dataclass
class ReActTrace:
    """Full execution record for a single question."""
    question: str
    answer: str
    sql_used: str = ""
    confidence: float = 0.0
    grounding: GroundingResult = field(default_factory=lambda: GroundingResult(passed=True))
    trace_id: str = ""
    trace_url: str = ""
    steps: list[ReActStep] = field(default_factory=list)
    total_tool_calls: int = 0
    iterations_used: int = 0
    grounding_interventions: int = 0  # how many times grounding check fired and retried
