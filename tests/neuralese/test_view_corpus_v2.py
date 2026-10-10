"""v2 of the view/ask corpus builder: new sources, licence provenance, cross-corpus closure (fixtures only)."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from neuralese_data import cross_corpus as cc  # noqa: E402
from neuralese_data import view_corpus as vc  # noqa: E402
from neuralese_data import view_extract as vx  # noqa: E402
from neuralese_data import view_sources as vs  # noqa: E402
from neuralese_data.records import group_key, validate  # noqa: E402

BASH = ("collected 3 items\n\nTraceback (most recent call last):\n  File \"/workspace/repo/src/mod.py\", line 12, in run\n"
        "    raise ValueError('bad input')\nValueError: bad input\n"
        "FAILED tests/test_a.py::test_two - ValueError: bad input\n"
        "========================= 1 failed, 2 passed in 0.12s =========================\n" + "log line\n" * 120)


def _ctx(raw: Path, **caps) -> vc.Ctx:
    return vc.Ctx(raw, caps, 32_000, 50, 0)


def _all(gen) -> list[dict]:
    return [r for doc in gen for r in doc.records]


def test_license_detection_and_classes():
    assert vx.detect_license("GNU LESSER GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007") == "LGPL-3.0"
    assert vx.detect_license("GNU GENERAL PUBLIC LICENSE\n                       Version 2, June 1991") == "GPL-2.0"
    assert vx.detect_license("MIT License\n\nPermission is hereby granted, free of charge, to any person") == "MIT"
    assert vx.detect_license("All rights reserved.") is None
    assert [vx.license_class(s) for s in ("MIT", "CC-BY-4.0", "CC-BY-SA-4.0", "GPL-3.0", "CC-BY-NC-4.0", None)] == \
        ["permissive", "attribution", "share-alike", "copyleft", "noncommercial", "unverified"]
    p = vc.provenance("CC-BY-4.0", "card", content_source="web pages", content_unverified=True)
    assert p["class"] == "unverified"
    assert vc.provenance("Apache-2.0", "card", content_spdx="CC-BY-SA-4.0")["class"] == "share-alike"


def test_restricted_unpickler_refuses_classes():
    import pickle

    assert vs._NoClasses.load(pickle.dumps({"a/b": [("a/b/LICENSE", "MIT")]})) == {"a/b": [("a/b/LICENSE", "MIT")]}
    with pytest.raises(pickle.UnpicklingError):
        vs._NoClasses.load(pickle.dumps(Path("/tmp")))


def test_sql_answers_execute_gold_queries(tmp_path, monkeypatch):
    db_root = tmp_path / "database"
    (db_root / "shop").mkdir(parents=True)
    conn = sqlite3.connect(db_root / "shop/shop.sqlite")
    conn.execute("CREATE TABLE items (id INTEGER, name TEXT, price REAL)")
    conn.executemany("INSERT INTO items VALUES (?, ?, ?)", [(i, f"item{i}", 1.5 * i) for i in range(1, 13)])
    conn.execute("CREATE TABLE big (x INTEGER)")
    conn.executemany("INSERT INTO big VALUES (?)", [(i,) for i in range(vc.SQL_MAX_ROWS + 5)])
    conn.commit()
    conn.close()
    questions = [{"db_id": "shop", "question": "What is the price of item3?", "query": "SELECT price FROM items WHERE name = 'item3'"},
                 {"db_id": "shop", "question": "How many items cost more than 10?", "query": "SELECT count(*) FROM items WHERE price > 10"},
                 {"db_id": "shop", "question": "How many rows in big?", "query": "SELECT count(*) FROM big"}]
    monkeypatch.setattr(vc, "_sql_items", lambda ctx: iter([("spider", "train", questions, db_root)]))
    records = _all(vc.spider_bird(_ctx(tmp_path, spider_bird=4)))
    qa = {r["lineage"]["notes"]["question"]: r for r in records if r["family"] == "view_table_qa"}
    assert qa["What is the price of item3?"]["target"]["value"] == "4.5"
    assert qa["How many items cost more than 10?"]["target"]["value"] == "6"
    assert "How many rows in big?" not in qa  # the table exceeds SQL_MAX_ROWS
    r = qa["What is the price of item3?"]
    assert r["split_groups"] == [group_key("sql-db", "spider:shop")]
    assert r["outcome"]["checked"] == "code-computed:sqlite-gold-sql"
    assert r["sources"][0]["text"].startswith("# table: items\nid,name,price\n1,item1,1.5")
    assert any(x["task"] == "reconstruct" for x in records)
    assert all(validate(vc.seal(json.loads(json.dumps(x)))) == [] for x in records)


def test_tabfact_and_toolace(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    root = raw / "tabfact"
    (root / "collected_data").mkdir(parents=True)
    (root / "data/all_csv").mkdir(parents=True)
    table = "year#team#wins\n" + "".join(f"{1990 + i}#t{i}#{i * 2}\n" for i in range(8))
    (root / "data/all_csv/1-1-1.html.csv").write_text(table)
    (root / "collected_data/r1_training_all.json").write_text(json.dumps({"1-1-1.html.csv": [["t3 won 6 games", "t3 won 9 games"], [1, 0], "wins"]}))
    (root / "collected_data/r2_training_all.json").write_text("{}")
    (root / "data/train_id.json").write_text(json.dumps(["1-1-1.html.csv"]))
    (root / "data/val_id.json").write_text("[]")
    (root / "data/test_id.json").write_text("[]")
    (root / "data/table_to_page.json").write_text(json.dumps({"1-1-1.html.csv": ["Some League", "https://en.wikipedia.org/wiki/Some_League"]}))
    (raw / "sources.json").write_text(json.dumps({"files": [{"dataset": "tabfact", "revision": "r"}]}))
    records = _all(vc.tabfact(_ctx(raw, tabfact=2)))
    qa = sorted((r["target"]["value"], r["consumer"]["context"][0]["content"]) for r in records if r["family"] == "view_table_qa")
    assert [a for a, _ in qa] == ["no", "yes"]
    assert group_key("wiki", "Some League") in records[0]["split_groups"]
    assert records[0]["lineage"]["notes"]["license_provenance"]["class"] == "share-alike"

    tools = [{"name": "getWeather", "description": "Weather", "parameters": {"type": "dict", "properties": {"city": {"type": "string"}}}},
             {"name": "getTime", "description": "Time", "parameters": {"type": "dict", "properties": {"tz": {"type": "string"}}}}]
    data = [{"system": "You are an expert.\n" + vc.TOOLACE_MARK + "\n" + json.dumps(tools) + "\nShould you decide...",
             "conversations": [{"from": "user", "value": "Weather in Oslo?"}, {"from": "assistant", "value": "[getWeather(city='Oslo')]"},
                               {"from": "tool", "value": json.dumps([{"name": "getWeather", "results": {"temp": 3, "unit": "C", "city": "Oslo", "wind": [1, 2, 3]}}])}]}]
    path = tmp_path / "toolace.json"
    path.write_text(json.dumps(data))
    monkeypatch.setattr(vc, "TOOLACE", path)
    records = _all(vc.toolace(_ctx(tmp_path, toolace=4)))
    assert {r["sources"][0]["role"] for r in records} == {"schema", "tool_output"}
    assert all(r["lineage"]["notes"]["target_origin"] in ("code-computed", "identity") for r in records)
    assert all("getWeather(city" not in json.dumps(r["target"]) for r in records)  # never the generated call
    assert records[0]["split_groups"] == [group_key("tool-schema", "getTime+getWeather")]


def test_s1_tool_outputs_keep_s1_split_and_groups(tmp_path, monkeypatch):
    calls = "[{'type': 'function', 'function': {'name': 'execute_bash', 'arguments': '{\"command\": \"pytest -q\"}'}, 'id': 'c1'}]"
    rec = {"id": "s1:x:1", "split": "validation", "split_groups": ["repo:octo/demo", "swe-instance:octo__demo-1"],
           "license": {"spdx": "CC-BY-4.0", "noncommercial": False, "notes": "Card: CC BY 4.0; repository octo/demo under MIT."},
           "lineage": {"upstream": "nvidia/Open-SWE-Traces:v1.0", "upstream_id": "t1", "upstream_revision": None},
           "sources": [{"role": "trajectory", "messages": [{"role": "assistant", "content": "", "tool_calls": calls},
                                                           {"role": "tool", "content": BASH}]}]}
    nebius = dict(rec, id="s1:x:2", lineage={"upstream": "nebius/SWE-rebench-openhands-trajectories"})
    s1 = tmp_path / "s1"
    s1.mkdir()
    (s1 / "trajectory_continuation_swe.port-records.jsonl").write_text("\n".join(json.dumps(r) for r in [rec, nebius] * 40) + "\n")
    monkeypatch.setattr(vc, "S1", s1)
    records = _all(vc.s1_tool_outputs(_ctx(tmp_path, s1_tool_outputs=3)))
    assert records and all(r["split"] == "validation" and r["split_groups"] == rec["split_groups"] for r in records)
    assert all(r["lineage"]["upstream"].startswith("nvidia/") for r in records)
    assert len({r["sources"][0]["text"] for r in records}) == 1  # the same output is taken once
    extract = {r["lineage"]["notes"]["extractor"]: r["target"]["value"] for r in records if r["family"] == "view_log_extract"}
    assert extract.get("log-pytest-summary") == "1 failed, 2 passed"
    assert records[0]["lineage"]["notes"]["license_provenance"]["content"]["spdx"] == "MIT"


def _rec(rid, split, groups, text, request=None, target="a", question=True):
    request = request or f"question {rid}?"
    notes = {"artifact": "prose", "target_origin": "dataset-qa"}
    if question:
        notes["question"] = request
    return {"id": rid, "family": "view_prose_qa", "task": "consume", "split": split, "split_groups": groups,
            "sources": [{"role": "document", "text": text}], "consumer": {"context": [{"role": "user", "content": request}]},
            "target": {"kind": "text", "value": target, "alternatives": []}, "lineage": {"store": "d", "notes": notes}}


def test_cross_corpus_index_and_apply(tmp_path):
    words = " ".join(f"w{i}" for i in range(300))
    published = [
        _rec("p1", "test", ["wiki:alpha"], "Alpha article. " + words, "Who?", "A"),
        _rec("p2", "train", ["wiki:beta"], "Beta article text " * 30, "When?", "1999"),
        _rec("p3", "train", ["repo:o/r"], "Gamma text " * 40, "Why?", "because"),
    ]
    src = tmp_path / "pub.port-records.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in published))
    cc.index_corpus([src], tmp_path / "idx", corpus_id="pub", workers=1, log=lambda *_: None)
    ix = cc.Index(tmp_path / "idx")
    ours = [
        _rec("a", "train", ["wiki:alpha"], "Different alpha paragraph " * 20),       # group -> test
        _rec("b", "train", ["wiki:x"], "Alpha article. " + words + " tail"),         # near duplicate of p1 -> test
        _rec("c", "validation", ["repo:o/r"], "Something else " * 20),               # own validation vs published train
        _rec("d", "train", ["wiki:beta"], "Beta article text " * 30, "When?", "1999"),  # exact example: dropped
        _rec("e", "train", ["wiki:new"], "Fresh text " * 30),
        _rec("f", "train", ["wiki:new2"], "Other fresh " * 30, "What exit code?", "1", question=False),
        _rec("g", "train", ["wiki:new3"], "Third fresh " * 30, "What exit code?", "1", question=False),
    ]
    kept, report = cc.apply(ours, [ix], log=lambda *_: None)
    by = {r["id"]: r["split"] for r in kept}
    assert by == {"a": "test", "b": "test", "e": "train", "f": "train", "g": "train"}
    assert any("own validation vs published train" in k for k in report["dropped"])
    assert any("exact example already published" in k for k in report["dropped"])
    assert cc.check(kept, [ix]) == []
