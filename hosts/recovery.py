"""Optional local checkpoint and external-operation receipt ledger.

An operation marked dispatched may or may not have reached the remote system.
Recovery looks up its idempotency key; it never infers rollback from a crash.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from natlang.execution import portable


class RecoveryStore:
    def __init__(self, path: Path):
        self.connection = sqlite3.connect(str(path))
        self.connection.row_factory = sqlite3.Row
        self.connection.executescript("""
            CREATE TABLE IF NOT EXISTS checkpoints(
                workflow TEXT PRIMARY KEY, revision INTEGER NOT NULL, position INTEGER NOT NULL,
                state_json TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS operations(
                workflow TEXT NOT NULL, operation_id TEXT NOT NULL, request_json TEXT NOT NULL,
                status TEXT NOT NULL, receipt_json TEXT,
                PRIMARY KEY(workflow, operation_id));
        """)

    def checkpoint(self, workflow: str, revision: int, position: int, state):
        if revision < 0 or position < 0:
            raise ValueError("revision and consumed source position must be nonnegative")
        encoded = json.dumps(portable(state), sort_keys=True)
        with self.connection:
            current = self.connection.execute("SELECT revision, position FROM checkpoints WHERE workflow=?",
                                              (workflow,)).fetchone()
            if current and (revision <= current["revision"] or position < current["position"]):
                raise ValueError("checkpoint revision or source position moved backward")
            self.connection.execute("INSERT INTO checkpoints VALUES(?,?,?,?) "
                                    "ON CONFLICT(workflow) DO UPDATE SET revision=excluded.revision, "
                                    "position=excluded.position, state_json=excluded.state_json",
                                    (workflow, revision, position, encoded))

    def load(self, workflow: str) -> dict:
        row = self.connection.execute("SELECT * FROM checkpoints WHERE workflow=?", (workflow,)).fetchone()
        operations = self.connection.execute("SELECT * FROM operations WHERE workflow=? ORDER BY operation_id",
                                             (workflow,)).fetchall()
        return {"checkpoint": None if row is None else {
                    "revision": row["revision"], "position": row["position"],
                    "state": json.loads(row["state_json"])},
                "operations": [{"id": op["operation_id"], "request": json.loads(op["request_json"]),
                                "status": op["status"],
                                "receipt": json.loads(op["receipt_json"]) if op["receipt_json"] else None}
                               for op in operations]}

    def plan(self, workflow: str, operation_id: str, request):
        encoded = json.dumps(portable(request), sort_keys=True)
        with self.connection:
            row = self.connection.execute("SELECT request_json FROM operations WHERE workflow=? AND operation_id=?",
                                          (workflow, operation_id)).fetchone()
            if row is not None and row["request_json"] != encoded:
                raise ValueError("operation id was reused for a different request")
            self.connection.execute("INSERT OR IGNORE INTO operations VALUES(?,?,?,?,NULL)",
                                    (workflow, operation_id, encoded, "planned"))

    def _status(self, workflow, operation_id, status, receipt=None):
        with self.connection:
            self.connection.execute("UPDATE operations SET status=?, receipt_json=? "
                                    "WHERE workflow=? AND operation_id=?",
                                    (status, json.dumps(portable(receipt)) if receipt is not None else None,
                                     workflow, operation_id))

    def perform(self, workflow: str, operation_id: str, request, adapter, crash_hook=None):
        """Dispatch once locally; the adapter owns idempotency and remote lookup."""
        self.plan(workflow, operation_id, request)
        existing = next(op for op in self.load(workflow)["operations"] if op["id"] == operation_id)
        if existing["status"] == "confirmed":
            return existing["receipt"]
        if existing["status"] == "dispatched":
            return self.reconcile(workflow, operation_id, adapter)
        if crash_hook:
            crash_hook("before_dispatch")
        self._status(workflow, operation_id, "dispatched")
        key = f"{workflow}:{operation_id}"
        receipt = adapter.dispatch(key, request)
        if crash_hook:
            crash_hook("after_external_commit_before_ack")
        self._status(workflow, operation_id, "confirmed", receipt)
        return receipt

    def reconcile(self, workflow: str, operation_id: str, adapter):
        key = f"{workflow}:{operation_id}"
        receipt = adapter.lookup(key)
        if receipt is None:
            return None  # unresolved; do not dispatch again without adapter guarantees
        self._status(workflow, operation_id, "confirmed", receipt)
        return receipt

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
