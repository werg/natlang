"""Natlang-planned import with exact accounting and transactional SQLite apply."""
from __future__ import annotations

import copy
import decimal
import json
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from applications.experiment_lab import digest
from natlang.host import load
from natlang.invocation import RunOptions, SeedPolicy
from natlang.runtime import Runtime
from natlang.trace import TraceRecorder
from natlang.values import dump


ROOT = Path(__file__).resolve().parent.parent
MAP = ROOT / "codebases/data_migration/map.nl"
DECIDE = ROOT / "codebases/data_migration/decide.nl"
MAPPING_FIELDS = ("customer_id", "email", "name", "order_id", "order_customer", "amount", "unit")


@dataclass(frozen=True)
class Export:
    source: str
    customers: tuple[dict, ...]
    orders: tuple[dict, ...]


def _money(value, unit: str) -> int:
    try:
        raw = decimal.Decimal(str(value))
    except decimal.InvalidOperation as exc:
        raise ValueError("invalid decimal amount") from exc
    cents = raw if unit == "cents" else raw * 100 if unit == "dollars" else None
    if cents is None or not cents.is_finite() or cents != cents.to_integral_value():
        raise ValueError("unsupported unit or fractional cents")
    return int(cents)


class MigrationStudio:
    def __init__(self, path: Path, *, agent_factory: Callable, model_id: str, root_seed: int,
                 trace_dir: Path | None = None):
        self.path, self.agent_factory, self.model_id = Path(path), agent_factory, model_id
        self.root_seed, self.trace_dir = root_seed, trace_dir
        self._previews: set[str] = set()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value INTEGER NOT NULL);
                INSERT OR IGNORE INTO meta VALUES ('revision', 0);
                CREATE TABLE IF NOT EXISTS customers (id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS source_customers (source_key TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id));
                CREATE TABLE IF NOT EXISTS orders (source_key TEXT PRIMARY KEY, customer_id TEXT NOT NULL REFERENCES customers(id), cents INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS lineage (source_key TEXT NOT NULL, field TEXT NOT NULL, target_key TEXT NOT NULL,
                                                     source_column TEXT NOT NULL, value_hash TEXT NOT NULL,
                                                     PRIMARY KEY (source_key, field));
            """)

    def _connect(self):
        db = sqlite3.connect(self.path)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        return db

    def _natlang(self, path: Path, inputs: dict, label: str):
        options = RunOptions(seed=SeedPolicy("derived", self.root_seed))
        trace_path = None
        if self.trace_dir:
            self.trace_dir.mkdir(parents=True, exist_ok=True)
            trace_path = self.trace_dir / f"{label}-{options.run_id}.jsonl"
        recorder = TraceRecorder({"run_id": options.run_id, "source_sha256": digest(path.read_text()),
                                  "model": self.model_id, "seed_policy": vars(options.seed),
                                  "phase": label}, trace_path)
        try:
            outcome, value = Runtime(self.agent_factory, options=options,
                                     trace_sink=recorder).run_root(load(path, copy.deepcopy(inputs)))
        finally:
            recorder.close()
        if outcome.kind != "done":
            raise ValueError(f"natlang {label} did not complete: {outcome.detail}")
        return dump(value), {"sha256": digest(recorder.events), "path": str(trace_path) if trace_path else None,
                             "events": recorder.events if trace_path is None else None}

    def _snapshot(self):
        with closing(self._connect()) as db:
            revision, customers, source_map, orders = self._snapshot_from_db(db)
        return revision, customers, source_map, orders

    @staticmethod
    def _snapshot_from_db(db):
        revision = db.execute("SELECT value FROM meta WHERE key='revision'").fetchone()[0]
        customers = [dict(row) for row in db.execute("SELECT id,email,name FROM customers ORDER BY id")]
        source_map = {row["source_key"]: row["customer_id"] for row in
                      db.execute("SELECT source_key,customer_id FROM source_customers")}
        orders = {row["source_key"]: (row["customer_id"], row["cents"]) for row in
                  db.execute("SELECT source_key,customer_id,cents FROM orders")}
        return revision, customers, source_map, orders

    def preview(self, exports: list[Export]) -> dict:
        if (not exports or len({e.source for e in exports}) != len(exports) or
                any(not e.source or ":" in e.source for e in exports)):
            raise ValueError("exports need unique source names")
        frozen = copy.deepcopy(exports)
        source_revision = digest([{"source": e.source, "customers": e.customers, "orders": e.orders}
                                  for e in frozen])
        target_revision, existing, source_map, old_orders = self._snapshot()
        normalized_customers, normalized_orders, mappings, traces = [], [], {}, {}
        forced_review, early_review = set(), []
        total_orders = 0
        for export in frozen:
            customer_columns = sorted({key for row in export.customers for key in row})
            order_columns = sorted({key for row in export.orders for key in row})
            mapping, trace = self._natlang(MAP, {"source": export.source,
                                               "customer_columns": customer_columns,
                                               "order_columns": order_columns}, f"map-{export.source}")
            for field in MAPPING_FIELDS:
                columns = customer_columns if field in ("customer_id", "email", "name") else order_columns
                if mapping[field] not in columns:
                    raise ValueError(f"mapping {field} is not a supplied column")
            mappings[export.source], traces[f"map-{export.source}"] = mapping, trace
            for index, row in enumerate(export.customers):
                raw_value = row[mapping["customer_id"]]
                raw_id = "" if raw_value is None else str(raw_value).strip()
                if not raw_id:
                    raw_id = f"missing@{index}"
                    forced_review.add(f"{export.source}:customer:{raw_id}")
                email_value = row[mapping["email"]]
                normalized_customers.append({"source_key": f"{export.source}:customer:{raw_id}",
                                             "email": "" if email_value is None else str(email_value).strip().lower(),
                                             "name": str(row[mapping["name"]]).strip()})
            for index, row in enumerate(export.orders):
                total_orders += 1
                raw_value = row[mapping["order_id"]]
                raw_id = "" if raw_value is None else str(raw_value).strip()
                if not raw_id:
                    early_review.append({"source_key": f"{export.source}:order:missing@{index}",
                                         "kind": "order", "reason": "missing stable source ID"})
                    continue
                normalized_orders.append({"source_key": f"{export.source}:order:{raw_id}",
                                          "customer_key": f"{export.source}:customer:{str(row[mapping['order_customer']]).strip()}",
                                          "cents": _money(row[mapping["amount"]],
                                                          str(row[mapping["unit"]]).strip().lower())})
        customer_keys = [x["source_key"] for x in normalized_customers]
        order_keys = [x["source_key"] for x in normalized_orders]
        if len(customer_keys) != len(set(customer_keys)) or len(order_keys) != len(set(order_keys)):
            raise ValueError("duplicate source identities")
        decisions, traces["decide"] = self._natlang(DECIDE,
            {"customers": normalized_customers, "existing": existing}, "decide")
        if len(decisions) != len(normalized_customers) or set(d["source_key"] for d in decisions) != set(customer_keys):
            raise ValueError("natlang decisions must account for every customer exactly once")
        by_key = {x["source_key"]: x for x in normalized_customers}
        known_email = {x["email"]: x["id"] for x in existing}
        known_names = {x["email"]: x["name"] for x in existing}
        planned_customers, customer_ids, customer_facts, review = [], {}, {}, list(early_review)
        for decision in decisions:
            row = by_key[decision["source_key"]]
            email = row["email"]
            if row["source_key"] in forced_review or not email or "@" not in email:
                if decision["action"] != "review":
                    raise ValueError("missing stable ID or email requires review")
            if decision["action"] == "review":
                review.append({"source_key": row["source_key"], "kind": "customer", "reason": decision["reason"]})
                continue
            if email in known_names and row["name"] != known_names[email]:
                raise ValueError("conflicting customer name requires review")
            if decision["target_email"] != email:
                raise ValueError("merge target lacks exact email identity evidence")
            match = known_email.get(email)
            if (decision["action"] == "new" and match is not None) or (decision["action"] == "merge" and match is None):
                raise ValueError("new/merge decision contradicts snapshot identity")
            customer_id = match or "c-" + digest(email)[:20]
            if match is None:
                known_email[email] = customer_id
                known_names[email] = row["name"]
                planned_customers.append({"id": customer_id, "email": email, "name": row["name"]})
            if row["source_key"] in source_map and source_map[row["source_key"]] != customer_id:
                raise ValueError("source customer identity changed")
            customer_ids[row["source_key"]] = customer_id
            customer_facts[row["source_key"]] = {"email": row["email"], "name": row["name"]}
        planned_orders = []
        unchanged_orders = 0
        reviewed_customers = {r["source_key"] for r in review if r["kind"] == "customer"}
        for row in normalized_orders:
            customer_id = (None if row["customer_key"] in reviewed_customers else
                           customer_ids.get(row["customer_key"]) or source_map.get(row["customer_key"]))
            if customer_id is None:
                review.append({"source_key": row["source_key"], "kind": "order", "reason": "unresolved customer"})
                continue
            if row["source_key"] in old_orders:
                if old_orders[row["source_key"]] != (customer_id, row["cents"]):
                    raise ValueError("reimport changes a committed order")
                unchanged_orders += 1
                continue
            planned_orders.append({**row, "customer_id": customer_id})
        patch = {"customers": planned_customers, "customer_sources": customer_ids,
                 "customer_facts": customer_facts,
                 "orders": planned_orders, "review": review}
        review_customers = sum(r["kind"] == "customer" for r in review)
        review_orders = sum(r["kind"] == "order" for r in review)
        if (len(customer_ids) + review_customers != len(normalized_customers) or
                len(planned_orders) + unchanged_orders + review_orders != total_orders):
            raise ValueError("source row accounting failed")
        preview = {"schema": "migration-preview/v1", "source_revision": source_revision,
                "target_revision": target_revision,
                "target_snapshot_sha256": digest([target_revision, existing, source_map, old_orders]),
                "patch": patch, "patch_sha256": digest(patch),
                "mappings": mappings, "decisions": decisions, "traces": traces,
                "counts": {"source_customers": len(normalized_customers),
                           "mapped_customers": len(customer_ids),
                           "source_orders": total_orders,
                           "new_orders": len(planned_orders),
                           "unchanged_orders": unchanged_orders,
                           "review": len(review)}}
        self._previews.add(digest(preview))
        return preview

    def apply(self, preview: dict, exports: list[Export], *, fail_after: int | None = None) -> dict:
        current_source = digest([{"source": e.source, "customers": e.customers, "orders": e.orders}
                                 for e in exports])
        if current_source != preview["source_revision"] or digest(preview["patch"]) != preview["patch_sha256"]:
            raise ValueError("preview source or patch changed")
        if digest(preview) not in self._previews:
            raise ValueError("preview was not issued by this studio")
        patch = preview["patch"]
        db = self._connect()
        writes = 0
        try:
            db.execute("BEGIN IMMEDIATE")
            snapshot = self._snapshot_from_db(db)
            revision = snapshot[0]
            if revision != preview["target_revision"]:
                raise ValueError("target revision changed since preview")
            if digest(list(snapshot)) != preview["target_snapshot_sha256"]:
                raise ValueError("target snapshot changed outside migration protocol")
            def execute(sql, args):
                nonlocal writes
                db.execute(sql, args)
                writes += 1
                if fail_after is not None and writes >= fail_after:
                    raise RuntimeError("injected transaction failure")
            for row in patch["customers"]:
                execute("INSERT INTO customers(id,email,name) VALUES (?,?,?)",
                        (row["id"], row["email"], row["name"]))
            for source_key, customer_id in patch["customer_sources"].items():
                execute("INSERT OR IGNORE INTO source_customers VALUES (?,?)", (source_key, customer_id))
                source = source_key.rsplit(":customer:", 1)[0]
                for field in ("email", "name"):
                    execute("INSERT OR IGNORE INTO lineage VALUES (?,?,?,?,?)",
                            (source_key, field, customer_id,
                             preview["mappings"][source][field],
                             digest(patch["customer_facts"][source_key][field])))
            for row in patch["orders"]:
                execute("INSERT INTO orders VALUES (?,?,?)",
                        (row["source_key"], row["customer_id"], row["cents"]))
                source = row["source_key"].rsplit(":order:", 1)[0]
                execute("INSERT INTO lineage VALUES (?,?,?,?,?)",
                        (row["source_key"], "cents", row["source_key"],
                         preview["mappings"][source]["amount"] + "+" + preview["mappings"][source]["unit"],
                         digest(row["cents"])))
            execute("UPDATE meta SET value=value+1 WHERE key='revision'", ())
            db.commit()
            return {"revision": revision + 1, "writes": writes, "review": patch["review"]}
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()
