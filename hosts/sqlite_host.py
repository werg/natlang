"""Optional scoped SQLite engine; SQL text and bound values stay separate."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from natlang.execution import CrispRequest, ExecutionError, portable
from natlang.nodes import MISSING


class SQLiteExecutor:
    name = "sqlite-v1"

    def __init__(self, path: str | Path = ":memory:"):
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.closed = False
        self.events = []
        self.in_transaction = False

    def prepare_schema(self, schema: str):
        """Host-only setup, before the model executes statements."""
        self.connection.executescript(schema)

    def begin(self):
        self.connection.execute("BEGIN")
        self.in_transaction = True
        self.events.append({"operation": "sql.begin"})

    def commit(self):
        self.connection.commit()
        self.in_transaction = False
        self.events.append({"operation": "sql.commit"})

    def rollback(self):
        self.connection.rollback()
        self.in_transaction = False
        self.events.append({"operation": "sql.rollback"})

    def run(self, request: CrispRequest, effect):
        if self.closed:
            raise ExecutionError("SQL environment is disposed")
        if request.effectful:
            raise ExecutionError("Python effects are unavailable in sqlite-v1")
        args = request.scope.get("args") or {}
        if not isinstance(args, dict):
            raise ExecutionError("SQL bindings must be a record")
        bindings = {k: portable(v, f"args/{k}") for k, v in args.items() if v is not MISSING}
        if any(isinstance(v, (dict, list)) for v in bindings.values()):
            raise ExecutionError("SQL bindings must be scalar values")
        try:
            cursor = self.connection.execute(request.code, bindings)
            if cursor.description:
                rows = [dict(row) for row in cursor.fetchall()]
                result = portable(rows)
            else:
                result = {"rows_affected": cursor.rowcount, "last_insert_id": cursor.lastrowid}
            if not self.in_transaction and self.connection.in_transaction:
                self.connection.commit()
            self.events.append({"operation": "sql.execute", "statement": request.code,
                                "bindings": sorted(bindings), "rows": len(result) if isinstance(result, list) else result["rows_affected"]})
            return result
        except (sqlite3.Error, ExecutionError) as exc:
            self.events.append({"operation": "sql.error", "message": str(exc)})
            raise ExecutionError(str(exc)) from exc

    def drain_events(self):
        events, self.events = self.events, []
        return events

    def close(self):
        if not self.closed:
            if self.in_transaction:
                self.rollback()
            self.connection.close()
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
