"""Fully dynamic SQLite schema introspection.

Design decision:
    Rather than using hand-written table descriptions that encode domain
    knowledge about a specific database, this module infers everything
    automatically from PRAGMA table_info, PRAGMA foreign_key_list, and sample
    rows — no hand-authored schema knowledge at all.

    The practical consequence is that this agent works on ANY SQLite database
    dropped in, not just one it was hand-tuned for.  That makes it genuinely
    general-purpose, which is worth calling out in the README as a deliberate
    design choice over hardcoding.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from groundedsql.db import connect_readonly


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    notnull: bool
    default_value: Any
    primary_key: bool


@dataclass(frozen=True)
class ForeignKey:
    from_col: str
    to_table: str
    to_col: str


@dataclass(frozen=True)
class RelationInfo:
    name: str
    kind: str                   # "table" or "view"
    columns: list[ColumnInfo]
    foreign_keys: list[ForeignKey]
    sample_rows: list[dict[str, Any]]


def _list_relations(con: sqlite3.Connection) -> list[tuple[str, str]]:
    """Return (name, type) pairs for all non-system tables and views."""
    rows = con.execute(
        """
        SELECT name, type
        FROM sqlite_master
        WHERE type IN ('table', 'view')
          AND name NOT LIKE 'sqlite_%'
        ORDER BY type DESC, name
        """
    ).fetchall()
    return [(row[0], row[1]) for row in rows]


def _quote(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _compact_value(v: Any, max_chars: int = 120) -> Any:
    if isinstance(v, str) and len(v) > max_chars:
        return v[:max_chars] + "…"
    return v


def inspect_database(
    db_path: Path | None = None,
    *,
    con: sqlite3.Connection | None = None,
    sample_rows: int = 3,
) -> list[RelationInfo]:
    """Introspect every table and view in the database.

    Pass an existing connection via `con` to avoid opening a second one.
    Either `db_path` or `con` must be provided.
    """
    close_after = con is None
    if con is None:
        if db_path is None:
            raise ValueError("Either db_path or con must be provided")
        con = connect_readonly(db_path)

    try:
        relations = _list_relations(con)
        infos: list[RelationInfo] = []

        for name, kind in relations:
            # Column metadata
            col_rows = con.execute(f"PRAGMA table_info({_quote(name)})").fetchall()
            columns = [
                ColumnInfo(
                    name=r[1],
                    type=r[2] or "ANY",
                    notnull=bool(r[3]),
                    default_value=r[4],
                    primary_key=bool(r[5]),
                )
                for r in col_rows
            ]

            # Foreign key relationships
            fk_rows = con.execute(f"PRAGMA foreign_key_list({_quote(name)})").fetchall()
            foreign_keys = [
                ForeignKey(from_col=r[3], to_table=r[2], to_col=r[4])
                for r in fk_rows
            ]

            # Sample rows (truncated per value)
            try:
                raw_samples = con.execute(
                    f"SELECT * FROM {_quote(name)} LIMIT ?", (sample_rows,)
                ).fetchall()
                samples = [
                    {k: _compact_value(v) for k, v in dict(zip([c[0] for c in con.execute(f"PRAGMA table_info({_quote(name)})").description or []], row)).items()}
                    for row in raw_samples
                ]
                # Simpler: fetchall returns Row objects with dict() available
                samples = [
                    {k: _compact_value(row[k]) for k in row.keys()}
                    for row in con.execute(f"SELECT * FROM {_quote(name)} LIMIT ?", (sample_rows,)).fetchall()
                ]
            except sqlite3.Error:
                samples = []

            infos.append(RelationInfo(
                name=name,
                kind=kind,
                columns=columns,
                foreign_keys=foreign_keys,
                sample_rows=samples,
            ))

        return infos
    finally:
        if close_after:
            con.close()


def schema_context(db_path: Path, max_relations: int = 30) -> str:
    """Build the schema context string injected into the agent system prompt.

    Includes for each table/view:
    - Column names and types
    - Foreign key relationships (auto-inferred)
    - Up to 3 sample rows (values truncated for token efficiency)

    No hand-authored descriptions — everything comes from the database itself.
    """
    con = connect_readonly(db_path)
    try:
        infos = inspect_database(con=con)
    finally:
        con.close()

    chunks: list[str] = [
        f"Database: {db_path.name}",
        f"Tables/views: {len(infos)} relations",
    ]

    for info in infos[:max_relations]:
        col_str = ", ".join(
            f"{c.name} {c.type}"
            + (" PK" if c.primary_key else "")
            + (" NOT NULL" if c.notnull else "")
            for c in info.columns
        )
        fk_str = ""
        if info.foreign_keys:
            fk_parts = [f"{fk.from_col} → {fk.to_table}.{fk.to_col}" for fk in info.foreign_keys]
            fk_str = f"\n  Foreign keys: {', '.join(fk_parts)}"

        sample_str = ""
        if info.sample_rows:
            sample_str = f"\n  Sample rows: {json.dumps(info.sample_rows[:2], ensure_ascii=False)}"

        chunks.append(
            f"{info.kind.upper()} {info.name}:\n"
            f"  Columns: {col_str}"
            f"{fk_str}"
            f"{sample_str}"
        )

    return "\n\n".join(chunks)
