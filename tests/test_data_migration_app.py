import copy
import sqlite3

import pytest

from applications.data_migration import Export, MigrationStudio
from natlang.surface import ToolSurface
from natlang.types import format_type


SURFACE = ToolSurface()


class Analyst:
    def __init__(self, lam):
        self.lam = lam

    def run(self, session):
        if self.lam.fn_name == "map":
            if self.lam.in_["source"] == "legacy":
                value = {"customer_id": "cid", "email": "mail", "name": "person",
                         "order_id": "oid", "order_customer": "buyer", "amount": "price",
                         "unit": "unit", "reason": "Legacy export fields."}
            else:
                value = {"customer_id": "id", "email": "email", "name": "name",
                         "order_id": "id", "order_customer": "customer", "amount": "amount",
                         "unit": "unit", "reason": "Direct fields."}
        else:
            known = {row["email"]: row["name"] for row in self.lam.in_["existing"]}
            value = []
            for row in self.lam.in_["customers"]:
                email = row["email"]
                action = ("review" if not email or "missing@" in row["source_key"] or
                          (email in known and known[email] != row["name"]) else
                          "merge" if email in known else "new")
                value.append({"source_key": row["source_key"], "action": action,
                              "target_email": email if action != "review" else "",
                              "reason": "Exact email evidence."})
                if action == "new":
                    known[email] = row["name"]
        result = SURFACE.apply(session, "write", {"path": "return",
                                                 "type": format_type(self.lam.type.returns), "value": value})
        assert result.kind not in ("rejected", "refused", "error"), result.text
        assert session.finish()


def exports():
    return [Export("direct", (
        {"id": "c1", "email": "Ada@Example.org", "name": "Ada"},), (
        {"id": "o1", "customer": "c1", "amount": "1234", "unit": "cents"},)),
        Export("legacy", (
            {"cid": "c7", "mail": "ada@example.org", "person": "Ada"},
            {"cid": "c8", "mail": "bea@example.org", "person": "Ada"}), (
            {"oid": "o7", "buyer": "c7", "price": "12.34", "unit": "dollars"},
            {"oid": "o8", "buyer": "c8", "price": "2.50", "unit": "dollars"}))]


def studio(tmp_path):
    return MigrationStudio(tmp_path / "target.sqlite", agent_factory=lambda lam: Analyst(lam),
                           model_id="scripted", root_seed=14, trace_dir=tmp_path / "traces")


def test_preview_apply_lineage_units_false_friends_and_idempotence(tmp_path):
    s = studio(tmp_path)
    source = exports()
    preview = s.preview(source)
    assert preview["counts"] == {"source_customers": 3, "mapped_customers": 3,
                                   "source_orders": 3, "new_orders": 3,
                                   "unchanged_orders": 0, "review": 0}
    assert len(preview["patch"]["customers"]) == 2  # same email merges; same name does not
    assert sorted(o["cents"] for o in preview["patch"]["orders"]) == [250, 1234, 1234]
    with pytest.raises(RuntimeError, match="injected"):
        s.apply(preview, source, fail_after=3)
    with sqlite3.connect(s.path) as db:
        assert db.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 0
        assert db.execute("SELECT value FROM meta").fetchone()[0] == 0
    assert s.apply(preview, source)["revision"] == 1
    with sqlite3.connect(s.path) as db:
        assert db.execute("SELECT COUNT(*) FROM customers").fetchone()[0] == 2
        assert db.execute("SELECT SUM(cents) FROM orders").fetchone()[0] == 2718
        assert db.execute("SELECT COUNT(*) FROM lineage").fetchone()[0] == 9
        assert db.execute("SELECT source_column FROM lineage WHERE source_key='legacy:order:o7' AND field='cents'").fetchone()[0] == "price+unit"
    again = s.preview(source)
    assert again["patch"]["customers"] == [] and again["patch"]["orders"] == []
    assert again["counts"]["unchanged_orders"] == 3
    assert len(again["patch"]["customer_sources"]) == 3


def test_stale_and_tampered_previews_are_rejected(tmp_path):
    s = studio(tmp_path)
    source = exports()
    preview = s.preview(source)
    changed = copy.deepcopy(source)
    changed[0] = Export("direct", changed[0].customers,
                        ({"id": "o1", "customer": "c1", "amount": "9999", "unit": "cents"},))
    with pytest.raises(ValueError, match="source or patch changed"):
        s.apply(preview, changed)
    forged = copy.deepcopy(preview)
    forged["patch"]["orders"][0]["cents"] = 9999
    with pytest.raises(ValueError, match="source or patch changed"):
        s.apply(forged, source)
    s.apply(preview, source)
    with pytest.raises(ValueError, match="target revision changed"):
        s.apply(preview, source)


def test_out_of_protocol_target_edit_and_false_merge_are_rejected(tmp_path):
    s = studio(tmp_path)
    source = exports()
    preview = s.preview(source)
    with sqlite3.connect(s.path) as db:
        db.execute("INSERT INTO customers VALUES ('external','external@example.org','External')")
    with pytest.raises(ValueError, match="outside migration protocol"):
        s.apply(preview, source)

    class WrongMerge(Analyst):
        def run(self, session):
            if self.lam.fn_name != "decide":
                return super().run(session)
            value = [{"source_key": row["source_key"], "action": "merge",
                      "target_email": "unrelated@example.org", "reason": "same display name"}
                     for row in self.lam.in_["customers"]]
            result = SURFACE.apply(session, "write", {"path": "return",
                                                     "type": format_type(self.lam.type.returns),
                                                     "value": value})
            assert result.kind not in ("rejected", "refused", "error"), result.text
            assert session.finish()

    wrong = MigrationStudio(tmp_path / "wrong.sqlite", agent_factory=lambda lam: WrongMerge(lam),
                            model_id="scripted-wrong", root_seed=1)
    with pytest.raises(ValueError, match="exact email identity"):
        wrong.preview(source)


def test_invalid_money_blocks_preview(tmp_path):
    s = studio(tmp_path)
    source = exports()
    bad = Export("direct", source[0].customers,
                 ({"id": "o1", "customer": "c1", "amount": "1.005", "unit": "dollars"},))
    with pytest.raises(ValueError, match="fractional cents"):
        s.preview([bad])


def test_missing_stable_ids_are_explicit_review_items(tmp_path):
    s = studio(tmp_path)
    source = [Export("direct", ({"id": None, "email": "x@example.org", "name": "X"},),
                     ({"id": None, "customer": "", "amount": "1", "unit": "cents"},))]
    preview = s.preview(source)
    assert preview["counts"]["source_customers"] == 1
    assert preview["counts"]["source_orders"] == 1
    assert preview["counts"]["review"] == 2
    assert preview["patch"]["customers"] == [] and preview["patch"]["orders"] == []


def test_same_email_different_name_stays_in_review(tmp_path):
    s = studio(tmp_path)
    source = exports()
    conflict = Export("legacy", (
        {"cid": "c7", "mail": "ada@example.org", "person": "Different Ada"},
        source[1].customers[1]), source[1].orders)
    preview = s.preview([source[0], conflict])
    assert {r["source_key"] for r in preview["patch"]["review"]} == {
        "legacy:customer:c7", "legacy:order:o7"}
    assert len(preview["patch"]["orders"]) == 2


def test_valid_batch_identity_decisions_do_not_depend_on_reply_order(tmp_path):
    class ReversedAnalyst(Analyst):
        def run(self, session):
            if self.lam.fn_name == "decide":
                self.lam.in_["customers"] = list(reversed(self.lam.in_["customers"]))
            return super().run(session)

    s = MigrationStudio(tmp_path / "target.sqlite", agent_factory=lambda lam: ReversedAnalyst(lam),
                        model_id="scripted-reversed", root_seed=14)
    preview = s.preview(exports())
    assert preview["counts"]["mapped_customers"] == 3
    assert len(preview["patch"]["customers"]) == 2
