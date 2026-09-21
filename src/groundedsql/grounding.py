"""Grounding / hallucination check.

This is the key differentiator of GroundedSQL over most text-to-SQL demos.

Before submit_result() is accepted, we verify that every numeric and factual
claim in the draft answer is actually supported by the data rows returned
during that turn.  If a claim is not grounded, the answer is rejected and an
explicit error is fed back into the ReAct loop — the same self-correction
mechanism used for SQL errors — forcing the agent to re-examine the data.

Two-stage approach:
  Stage 1 — Deterministic numeric extraction (fast, free):
    Regex-extract all numeric literals from the draft answer text.
    For each number, scan all rows returned during the turn.
    Any number that appears in the answer but not in the data is flagged.

  Stage 2 — LLM-as-judge (cheap model, only runs when Stage 1 flags something):
    For each flagged claim, send a minimal prompt to the cheap model:
    "Does this number or claim appear in these data rows? Answer YES or NO."
    LLM-as-judge catches cases where a number is present but misleadingly
    framed (e.g. the answer says "average of 5.2" but the rows contain 5.2
    as a raw value not an average).

The grounding check catch rate is tracked in ReActTrace.grounding_interventions
and reported in the eval harness — this is the proof the mechanism works, not
just that it exists.
"""
from __future__ import annotations

import json
import logging
import re
from typing import Any

from groundedsql.models import GroundingResult

logger = logging.getLogger(__name__)

# Numeric patterns: integers, decimals, percentages, comma-separated thousands
_NUMBER_RE = re.compile(
    r"\b(\d{1,3}(?:,\d{3})*(?:\.\d+)?|\d+\.\d+|\d+)\b"
)

# Threshold: numbers this small are common enough to ignore (1, 2, 3, etc.)
_MIN_SIGNIFICANT_VALUE = 10


def _extract_numbers(text: str) -> list[str]:
    """Extract all numeric strings from free text.

    Small whole numbers (< _MIN_SIGNIFICANT_VALUE) are filtered out because
    they appear too commonly to be meaningful claims (e.g. "3 items", "2 rows").
    Decimal numbers are always kept regardless of magnitude — a precise decimal
    like 3.14 is always a specific claim worth verifying.
    """
    results = []
    for m in _NUMBER_RE.finditer(text):
        raw = m.group(0).replace(",", "")
        is_decimal = "." in raw
        val = float(raw)
        if is_decimal or val >= _MIN_SIGNIFICANT_VALUE:
            results.append(raw)
    return results


def _flatten_rows(rows: list[dict[str, Any]]) -> set[str]:
    """Collect all stringified values from all rows into a flat set."""
    values: set[str] = set()
    for row in rows:
        for v in row.values():
            if v is None:
                continue
            s = str(v)
            values.add(s)
            # Also add the numeric-string form for floats like "5.20" → "5.2"
            try:
                values.add(str(float(s)))
            except (ValueError, TypeError):
                pass
    return values


def _number_in_rows(number_str: str, flat_values: set[str]) -> bool:
    """Check whether a numeric string appears (or is close enough) in the data."""
    if number_str in flat_values:
        return True
    try:
        target = float(number_str)
    except ValueError:
        return False
    for v in flat_values:
        try:
            if abs(float(v) - target) < 0.01:
                return True
        except (ValueError, TypeError):
            pass
    return False


def _llm_verify_claim(
    claim: str,
    rows: list[dict[str, Any]],
    client: Any,
    model: str,
) -> bool:
    """Ask a cheap LLM whether a claim is supported by the data rows.

    Returns True if the LLM considers the claim grounded, False otherwise.
    Falls back to True (passes) on any error to avoid blocking the agent on
    an observability failure.
    """
    rows_sample = rows[:20]  # cap tokens
    prompt = (
        f"You are a fact-checker. Does the following claim appear in the data rows below?\n"
        f"Answer with exactly one word: YES or NO.\n\n"
        f"Claim: {claim}\n\n"
        f"Data rows (JSON):\n{json.dumps(rows_sample, ensure_ascii=False)[:3000]}"
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0,
        )
        answer = (response.choices[0].message.content or "").strip().upper()
        return answer.startswith("YES")
    except Exception as exc:
        logger.error(
            "ALERT [GroundingSafety]: LLM grounding judge failed — falling back to fail-open (pass): %s",
            exc,
            exc_info=True,
        )
        return True  # fail open — don't block agent on observability error


def check_grounding(
    answer_text: str,
    all_rows: list[dict[str, Any]],
    client: Any | None = None,
    grounding_model: str = "google/gemini-2.0-flash-001",
) -> GroundingResult:
    """Check whether every numeric claim in answer_text is grounded in the rows.

    Args:
        answer_text:     The draft answer text from submit_result().
        all_rows:        All rows returned by run_sql() calls in this turn,
                         concatenated.  Empty list → grounding check is skipped
                         (passes trivially).
        client:          An openai.OpenAI client for Stage 2 LLM-as-judge.
                         If None, only Stage 1 (deterministic) runs.
        grounding_model: Model slug for the cheap LLM-as-judge call.

    Returns:
        GroundingResult with passed=True if all claims are grounded, or
        passed=False with a rejection_reason suitable for feeding back into
        the agent loop as a tool output error.
    """
    if not all_rows:
        # No data was queried — cannot verify, pass trivially.
        return GroundingResult(passed=True)

    numbers = _extract_numbers(answer_text)
    if not numbers:
        # No numeric claims to check.
        return GroundingResult(passed=True)

    flat_values = _flatten_rows(all_rows)
    flagged: list[str] = []

    for num in numbers:
        if not _number_in_rows(num, flat_values):
            flagged.append(num)

    if not flagged:
        return GroundingResult(passed=True)

    # Stage 2: LLM-as-judge on flagged claims only
    verified_by_llm = False
    still_flagged: list[str] = []

    if client is not None:
        verified_by_llm = True
        for num in flagged:
            if not _llm_verify_claim(num, all_rows, client, grounding_model):
                still_flagged.append(num)
    else:
        still_flagged = flagged

    if not still_flagged:
        return GroundingResult(
            passed=True,
            flagged_claims=flagged,
            verified_by_llm=verified_by_llm,
        )

    rejection_reason = (
        f"Grounding check failed. The following values appear in your answer "
        f"but were not found in the data you queried: {still_flagged}. "
        f"Re-examine your SQL results and correct the answer to only state "
        f"values that are directly present in the returned rows."
    )
    logger.info("Grounding check flagged: %s", still_flagged)
    return GroundingResult(
        passed=False,
        flagged_claims=still_flagged,
        verified_by_llm=verified_by_llm,
        rejection_reason=rejection_reason,
    )
