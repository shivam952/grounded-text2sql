"""Tests for grounding.py — numeric extraction, row matching, GroundingResult."""
from __future__ import annotations

import pytest

from groundedsql.grounding import (
    GroundingResult,
    _extract_numbers,
    _flatten_rows,
    _number_in_rows,
    check_grounding,
)


class TestExtractNumbers:
    def test_integer(self):
        assert "42" in _extract_numbers("There are 42 schools.")

    def test_decimal(self):
        assert "3.14" in _extract_numbers("The value is 3.14.")

    def test_small_numbers_ignored(self):
        # Numbers below _MIN_SIGNIFICANT_VALUE (10) are filtered out
        nums = _extract_numbers("There are 5 categories and 3 items.")
        assert "5" not in nums
        assert "3" not in nums

    def test_thousands_separator(self):
        nums = _extract_numbers("Total budget: $1,234,567")
        assert "1234567" in nums

    def test_percentage(self):
        nums = _extract_numbers("Accuracy is 85.3%")
        assert "85.3" in nums

    def test_no_numbers(self):
        assert _extract_numbers("No numbers here.") == []


class TestFlattenRows:
    def test_basic(self):
        rows = [{"count": 42, "name": "Alice"}, {"count": 100}]
        flat = _flatten_rows(rows)
        assert "42" in flat
        assert "100" in flat

    def test_float_normalisation(self):
        rows = [{"val": "5.20"}]
        flat = _flatten_rows(rows)
        assert "5.2" in flat  # normalised float form

    def test_none_ignored(self):
        rows = [{"val": None}]
        flat = _flatten_rows(rows)
        assert "None" not in flat


class TestNumberInRows:
    def test_exact_match(self):
        flat = {"42", "100"}
        assert _number_in_rows("42", flat) is True

    def test_float_tolerance(self):
        flat = {"5.199"}
        assert _number_in_rows("5.2", flat) is True

    def test_not_present(self):
        flat = {"10", "20", "30"}
        assert _number_in_rows("99", flat) is False


class TestCheckGrounding:
    def test_passes_with_no_rows(self):
        result = check_grounding("There are 42 schools.", all_rows=[])
        assert result.passed is True

    def test_passes_when_number_in_rows(self):
        rows = [{"count": 42}]
        result = check_grounding("There are 42 schools.", all_rows=rows)
        assert result.passed is True

    def test_fails_when_number_not_in_rows(self):
        rows = [{"count": 99}]
        # 42 is not in rows, 99 is not in the answer — grounding should fail
        result = check_grounding("There are 42 schools.", all_rows=rows, client=None)
        assert result.passed is False
        assert "42" in result.flagged_claims
        assert result.rejection_reason  # non-empty error message

    def test_passes_no_numeric_claims(self):
        rows = [{"name": "Alice"}]
        result = check_grounding("The top school is Alice Academy.", all_rows=rows)
        assert result.passed is True

    def test_rejection_reason_is_usable(self):
        rows = [{"val": 10}]
        result = check_grounding("The answer is 99.", all_rows=rows, client=None)
        assert not result.passed
        # Rejection reason should be a complete sentence the agent can act on
        assert len(result.rejection_reason) > 20
        assert "99" in result.rejection_reason
