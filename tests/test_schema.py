"""Tests for schema.py — dynamic introspection against a real in-memory SQLite DB."""
from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path

import pytest

from groundedsql.schema import inspect_database, schema_context


@pytest.fixture()
def sample_db(tmp_path: Path) -> Path:
    """Create a small SQLite database with two tables and FK relationships."""
    db_path = tmp_path / "test.sqlite"
    con = sqlite3.connect(str(db_path))
    con.executescript("""
        CREATE TABLE departments (
            id   INTEGER PRIMARY KEY,
            name TEXT NOT NULL
        );
        CREATE TABLE employees (
            id            INTEGER PRIMARY KEY,
            name          TEXT NOT NULL,
            department_id INTEGER NOT NULL REFERENCES departments(id),
            salary        REAL
        );
        INSERT INTO departments VALUES (1, 'Engineering'), (2, 'Marketing');
        INSERT INTO employees VALUES
            (1, 'Alice', 1, 90000.0),
            (2, 'Bob',   1, 85000.0),
            (3, 'Carol', 2, 75000.0);
    """)
    con.commit()
    con.close()
    return db_path


class TestInspectDatabase:
    def test_finds_both_tables(self, sample_db: Path):
        infos = inspect_database(db_path=sample_db)
        names = [i.name for i in infos]
        assert "departments" in names
        assert "employees" in names

    def test_columns_detected(self, sample_db: Path):
        infos = {i.name: i for i in inspect_database(db_path=sample_db)}
        emp_cols = [c.name for c in infos["employees"].columns]
        assert "id" in emp_cols
        assert "name" in emp_cols
        assert "salary" in emp_cols

    def test_primary_key_flagged(self, sample_db: Path):
        infos = {i.name: i for i in inspect_database(db_path=sample_db)}
        pk_cols = [c.name for c in infos["employees"].columns if c.primary_key]
        assert pk_cols == ["id"]

    def test_foreign_keys_detected(self, sample_db: Path):
        infos = {i.name: i for i in inspect_database(db_path=sample_db)}
        fks = infos["employees"].foreign_keys
        assert len(fks) == 1
        assert fks[0].from_col == "department_id"
        assert fks[0].to_table == "departments"

    def test_sample_rows_populated(self, sample_db: Path):
        infos = {i.name: i for i in inspect_database(db_path=sample_db)}
        assert len(infos["departments"].sample_rows) > 0
        assert "name" in infos["departments"].sample_rows[0]

    def test_no_system_tables(self, sample_db: Path):
        infos = inspect_database(db_path=sample_db)
        names = [i.name for i in infos]
        assert not any(n.startswith("sqlite_") for n in names)


class TestSchemaContext:
    def test_returns_string(self, sample_db: Path):
        ctx = schema_context(sample_db)
        assert isinstance(ctx, str)
        assert len(ctx) > 0

    def test_contains_table_names(self, sample_db: Path):
        ctx = schema_context(sample_db)
        assert "departments" in ctx
        assert "employees" in ctx

    def test_contains_column_names(self, sample_db: Path):
        ctx = schema_context(sample_db)
        assert "salary" in ctx

    def test_contains_fk_info(self, sample_db: Path):
        ctx = schema_context(sample_db)
        # FK should be surfaced as "department_id → departments.id" or similar
        assert "departments" in ctx and "department_id" in ctx

    def test_contains_sample_data(self, sample_db: Path):
        ctx = schema_context(sample_db)
        assert "Engineering" in ctx or "Alice" in ctx
