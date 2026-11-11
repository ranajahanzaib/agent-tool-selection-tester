"""Append-only local SQLite storage for experiment runs and case traces."""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_DB = Path(os.environ.get("AGENT_TOOL_SELECTION_TESTER_DB", Path.home() / ".agent_tool_selection_tester" / "results.sqlite3"))


class Store:
    def __init__(self, path: Path = DEFAULT_DB) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(path)
        self.db.row_factory = sqlite3.Row
        self.db.executescript("""
        PRAGMA journal_mode=WAL;
        CREATE TABLE IF NOT EXISTS runs(id TEXT PRIMARY KEY, name TEXT NOT NULL, started_at TEXT NOT NULL, manifest TEXT NOT NULL, config TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS cases(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL REFERENCES runs(id), variant_id TEXT NOT NULL, task_id TEXT NOT NULL, repetition INTEGER NOT NULL, result TEXT NOT NULL);
        CREATE INDEX IF NOT EXISTS cases_run_variant ON cases(run_id, variant_id);
        """)

    def create_run(
        self, run_id: str, name: str, manifest: dict[str, Any], config: dict[str, Any]
    ) -> None:
        self.db.execute(
            "INSERT INTO runs VALUES(?,?,?,?,?)",
            (run_id, name, datetime.now(UTC).isoformat(), json.dumps(manifest), json.dumps(config)),
        )
        self.db.commit()

    def add_case(self, run_id: str, result: dict[str, Any]) -> None:
        self.db.execute(
            "INSERT INTO cases(run_id,variant_id,task_id,repetition,result) VALUES(?,?,?,?,?)",
            (
                run_id,
                result["variant_id"],
                result["task_id"],
                result["repetition"],
                json.dumps(result),
            ),
        )
        self.db.commit()

    def runs(self) -> list[dict[str, Any]]:
        return [dict(x) for x in self.db.execute("SELECT * FROM runs ORDER BY started_at DESC")]

    def get_run(self, run_id: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        row = self.db.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        if row is None:
            raise KeyError(f"No run found with ID {run_id}")
        cases = [
            json.loads(x[0])
            for x in self.db.execute(
                "SELECT result FROM cases WHERE run_id=? ORDER BY id", (run_id,)
            )
        ]
        return dict(row), cases

    def close(self) -> None:
        self.db.close()
