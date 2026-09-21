"""Tests for sql_guard — validate_select_sql, enforce_limit, strip_sql_fence."""
from __future__ import annotations

import pytest

from groundedsql.sql_guard import (
    SqlValidation,
    enforce_limit,
    strip_sql_fence,
    validate_select_sql,
)


# ---------------------------------------------------------------------------
# strip_sql_fence
# ---------------------------------------------------------------------------

class TestStripSqlFence:
    def test_no_fence(self):
        assert strip_sql_fence("SELECT 1") == "SELECT 1"

    def test_basic_fence(self):
        sql = "```\nSELECT 1\n```"
        assert strip_sql_fence(sql) == "SELECT 1"

    def test_language_fence(self):
        sql = "```sql\nSELECT * FROM t\n```"
        assert strip_sql_fence(sql) == "SELECT * FROM t"

    def test_sql_prefix(self):
        sql = "sql\nSELECT 1"
        assert strip_sql_fence(sql) == "SELECT 1"

    def test_empty(self):
        assert strip_sql_fence("") == ""

    def test_whitespace_only(self):
        assert strip_sql_fence("   ") == ""


# ---------------------------------------------------------------------------
# validate_select_sql
# ---------------------------------------------------------------------------

class TestValidateSelectSql:
    def test_valid_select(self):
        r = validate_select_sql("SELECT * FROM schools")
        assert r.ok is True

    def test_valid_with_cte(self):
        r = validate_select_sql("WITH cte AS (SELECT 1) SELECT * FROM cte")
        assert r.ok is True

    def test_empty(self):
        r = validate_select_sql("")
        assert r.ok is False
        assert "empty" in r.error.lower()

    def test_multiple_statements(self):
        r = validate_select_sql("SELECT 1; SELECT 2")
        assert r.ok is False
        assert "one" in r.error.lower()

    def test_forbidden_insert(self):
        r = validate_select_sql("INSERT INTO t VALUES (1)")
        assert r.ok is False

    def test_forbidden_drop(self):
        r = validate_select_sql("DROP TABLE schools")
        assert r.ok is False

    def test_forbidden_pragma(self):
        r = validate_select_sql("PRAGMA journal_mode=WAL")
        assert r.ok is False

    def test_strips_trailing_semicolon(self):
        r = validate_select_sql("SELECT 1;")
        assert r.ok is True
        assert not r.sql.endswith(";")

    def test_insert_in_string_literal_allowed(self):
        # "INSERT" appears inside a string value — should not be flagged
        r = validate_select_sql("SELECT 'INSERT is a keyword' AS note")
        assert r.ok is True

    def test_update_in_column_name_allowed(self):
        r = validate_select_sql('SELECT "last_update" FROM t')
        assert r.ok is True

    def test_case_insensitive_forbidden(self):
        r = validate_select_sql("delete from schools")
        assert r.ok is False


# ---------------------------------------------------------------------------
# enforce_limit
# ---------------------------------------------------------------------------

class TestEnforceLimit:
    def test_adds_limit_to_select(self):
        sql = enforce_limit("SELECT * FROM t", 100)
        assert "LIMIT 100" in sql.upper()

    def test_does_not_double_limit(self):
        sql = enforce_limit("SELECT * FROM t LIMIT 10", 100)
        assert sql.upper().count("LIMIT") == 1
        assert "10" in sql

    def test_handles_cte(self):
        sql = enforce_limit("WITH cte AS (SELECT 1) SELECT * FROM cte", 50)
        # CTE gets LIMIT appended directly, not wrapped in subquery
        assert sql.upper().endswith("LIMIT 50")

    def test_raises_on_invalid_sql(self):
        with pytest.raises(ValueError):
            enforce_limit("DELETE FROM t", 100)
