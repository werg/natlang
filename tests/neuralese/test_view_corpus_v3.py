"""v3 of the view/ask corpus builder: the sources added when licences stopped being a reason to omit data
(owner rule 2026-10-10), multi-source records, DOM reduction and WARC reading (fixtures only, no downloads)."""
from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "training" / "neuralese"))

from neuralese_data import view_corpus as vc  # noqa: E402
from neuralese_data import view_extract as vx  # noqa: E402
from neuralese_data.records import seal, validate  # noqa: E402
from natlang_neuralese.data import view_records as vr  # noqa: E402


def _ctx(raw: Path, **caps) -> vc.Ctx:
    return vc.Ctx(raw, caps, 32_000, 50, 0)


def _all(gen) -> list[dict]:
    return [r for doc in gen for r in doc.records]


def _sources_json(raw: Path, *datasets: str) -> None:
    raw.mkdir(parents=True, exist_ok=True)
    (raw / "sources.json").write_text(json.dumps({"files": [{"dataset": d, "revision": "r"} for d in datasets]}))


def test_reduce_dom_keeps_the_target_within_the_cap():
    filler = "".join(f"<div class='f'><p>filler paragraph {i} " + "x" * 300 + "</p></div>" for i in range(60))
    html = (f"<html><head><title>Shop</title></head><body><nav>{filler}</nav><main><ul>"
            + "".join(f"<li backend_node_id='{i}'>item {i}</li>" for i in range(30))
            + f"<button backend_node_id='77'>Search</button></ul></main><footer>{filler}</footer></body></html>")
    out = vc.reduce_dom(html, "backend_node_id", "77", 4000, drop_attrs=("backend_node_id",))
    assert out is not None and len(out) <= 4000
    assert "<button>Search</button>" in out and "item 29" in out  # the target and its nearest neighbours
    assert "backend_node_id" not in out and out.startswith("<html>")
    assert vc.reduce_dom(html, "backend_node_id", "999", 4000) is None


def test_warc_prefix_reader_skips_the_cut_member(tmp_path):
    def member(uri, body):
        http = b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n\r\n" + body
        head = (f"WARC/1.0\r\nWARC-Type: response\r\nWARC-Target-URI: {uri}\r\n"
                f"Content-Length: {len(http)}\r\n\r\n").encode()
        return gzip.compress(head + http + b"\r\n\r\n")
    data = member("https://a.example/x", b"<html>a</html>") + member("https://b.example/y", b"<html>b</html>")
    path = tmp_path / "x.warc.gz"
    path.write_bytes(data + member("https://c.example/z", b"<html>c</html>")[:30])
    got = [h["warc-target-uri"] for h, _ in vc._warc_records(path)]
    assert got == ["https://a.example/x", "https://b.example/y"]


def test_pmc_sections_remove_the_abstract():
    text = ("---\ntitle: A study\npmcid: PMC1\nlicense: CC BY 4.0\n---\n\n# A study\n\n## Abstract\n\n### Simple Summary\n\n"
            "Plain words.\n\n### Abstract\n\nWe measured things.\n\n## Introduction\n\nBody text.\n")
    header, rest, abstract, simple = vc.pmc_sections(text)
    assert header["pmcid"] == "PMC1" and abstract == "We measured things." and simple == "Plain words."
    assert "We measured" not in rest and "Body text." in rest and "# A study" in rest
    assert vc.pmc_sections("# No abstract\n\n## Methods\n\nx") is None


def test_multi_news_records_are_multi_source_and_convert_to_one_value(tmp_path, monkeypatch):
    root = tmp_path / "mn"
    (root / "data").mkdir(parents=True)
    a1 = "First article about the storm. " * 12
    a2 = "Second article, with other details. " * 12
    for name in ("train", "val", "test"):
        (root / f"data/{name}.src.cleaned").write_text(f"{a1} NEWLINE_CHAR NEWLINE_CHAR more ||||| {a2}\n")
        (root / f"data/{name}.tgt").write_text("– A storm hit the coast on Monday, the articles report, and many of their details differ widely.\n")
    monkeypatch.setattr(vc, "MULTI_NEWS", root)
    records = _all(vc.multi_news(_ctx(tmp_path, multi_news=20)))
    assert records and all(len(r["sources"]) == 2 and r["task"] == "consume" for r in records)
    assert all(r["lineage"]["notes"]["view_call"] == "view(xs, instructions)" for r in records)
    assert not any(r["task"] == "reconstruct" for r in records)
    for r in records:
        assert validate(seal(r)) == []
    port = next(r for r in records if r["split"] == "train")
    stage = vr.stage_record(port, corpus="c")
    part = stage["messages"][0]["content"][0]
    assert part["source"].startswith("Document 1 of 2:\n") and "\n\nDocument 2 of 2:\n" in part["source"]


def test_loghub_windows_use_the_labelled_templates(tmp_path, monkeypatch):
    raw = tmp_path / "raw"
    _sources_json(raw, "loghub")
    monkeypatch.setattr(vc, "view_sources_loghub_systems", lambda: ["Demo", "B", "C", "D"])
    sysdir = raw / "loghub/Demo"
    sysdir.mkdir(parents=True)
    lines, rows = [], ["LineId,Content,EventId,EventTemplate"]
    for i in range(1, 41):
        if i % 3:
            lines.append(f"2024-01-01 10:00:{i:02d} INFO conn opened from 10.0.0.{i}")
            rows.append(f"{i},conn opened from 10.0.0.{i},E1,conn opened from <*>")
        else:
            lines.append(f"2024-01-01 10:00:{i:02d} WARN retry {i} failed")
            rows.append(f"{i},retry {i} failed,E2,retry <*> failed")
    (sysdir / "Demo_2k.log").write_text("\n".join(lines) + "\n")
    (sysdir / "Demo_2k.log_structured.csv").write_text("\n".join(rows) + "\n")
    records = _all(vc.loghub(_ctx(raw, loghub=5)))
    qa = [r for r in records if r["family"] == "view_log_qa"]
    assert qa and {r["lineage"]["notes"]["target_origin"] for r in qa} == {"dataset-labels"}
    templates = next(r for r in qa if r["consumer"]["context"][0]["content"].startswith("List the distinct"))
    assert templates["target"]["value"] == "conn opened from <*>\nretry <*> failed"
    count = next(r for r in qa if r["consumer"]["context"][0]["content"].startswith("How many lines"))
    assert count["target"]["value"] in ("27", "13")
    assert {r["split"] for r in records} == {vc._loghub_split("Demo")}


def test_repository_configs_keep_any_licence():
    assert vc.classify_license("GNU GENERAL PUBLIC LICENSE\n Version 3, 29 June 2007") == "GPL-3.0"
    assert vc.classify_license("All rights reserved.") is None
    prov = vc.provenance("LicenseRef-repository-content", "x", content_spdx=None, content_unverified=True,
                         facts=["no known licence text"])
    assert prov["class"] == "unverified" and prov["facts"] == ["no known licence text"] and "concerns" not in prov


def test_v3_caps_extend_v2():
    assert set(vc.V2_CAPS) < set(vc.V3_CAPS) and all(vc.V3_CAPS[k] == v for k, v in vc.V2_CAPS.items())
    assert set(vc.V3_CAPS) <= set(vc.ADAPTERS) | {f"codesearchnet_{l}" for l in vc.CSN_LANGS}
    assert not hasattr(vc, "filter_by_license")  # no licence cut of a corpus (owner rule 2026-10-10)
    assert vx.license_class("CC-BY-NC-4.0") == "noncommercial"


def test_pmc_structured_abstracts_are_whole_sections():
    text = "# T\n\n## Abstract\n\n### Background\n\nWhy.\n\n### Methods\n\nHow.\n\n## Body\n\nText.\n"
    _, rest, abstract, simple = vc.pmc_sections(text)
    assert abstract == "### Background\n\nWhy.\n\n### Methods\n\nHow." and simple is None and "Why." not in rest


def test_registered_view_v3_is_gated_only_on_the_stage_and_pinned_by_the_recipe():
    entries = {e["id"]: e for e in json.loads((ROOT / "training/neuralese_corpora.json").read_text())["corpora"]}
    v3 = entries["view-ask-20261010-v3"]
    assert v3["training_admission"] is False and v3["supersedes"] == "view-ask-20261010-v2"
    for entry in (v3, entries["view-ask-20261010-v2"], entries["view-ask-slice-20261010-v1"]):
        held = " ".join(str(entry.get(k, "")) for k in ("admission", "training_admission_reason")).lower()
        assert "licen" not in held or "facts only" in held
    for ix in v3["build"]["cross_corpus_indexes"]:
        assert "path" not in ix and entries[ix["registry_id"]]["kind"] == "cross-corpus-index"
    recipe = json.loads((ROOT / "training/neuralese/recipes/raw-recurrence-v4.json").read_text())["overrides"]
    binding = recipe["input_bindings"]["view-stage-records"]
    assert binding["path"] == v3["path"] + "/view-stage/records.jsonl"
    assert binding["sha256"] in v3["build"]["view_stage_conversion"]
    assert recipe["view_operator"]["corpus"]["id"] == v3["id"] and recipe["view_operator"]["admitted"] is False
    assert "licence" not in recipe["view_operator"]["admission"].split("(")[0]
