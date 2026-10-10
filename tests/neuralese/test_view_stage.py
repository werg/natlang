"""View-stage records (data/view_records.py) and the view gate's selection and verdict (eval/view_gate.py); no model."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "neuralese"))

from natlang_neuralese.data import view_records as vr  # noqa: E402
from natlang_neuralese.eval import view_gate as vg  # noqa: E402
from natlang_neuralese.view import INSTRUCTIONS  # noqa: E402


def _port(rid, task, split="train", instructions="Keep what is needed to answer this question: who?", pairs=(),
          artifact="prose", source="The value text. " * 20, target="Ann"):
    return {"id": rid, "family": f"view_{artifact}_qa", "task": task, "split": split, "split_groups": ["g:1"],
            "sources": [{"role": "document", "text": source}],
            "writer": {"instructions": instructions, "instructions_general": "Keep what later questions need.",
                       "result_type": "Neuralese<string>", "context": []},
            "consumer": {"context": [{"role": "user", "content": "Who?"}], "withheld": ["sources"]},
            "target": {"kind": "text", "value": target, "alternatives": []},
            "contrasts": {"purpose_pairs": list(pairs), "distractors": []},
            "outcome": {"label": "gold", "checked": "reference-answer"},
            "lineage": {"sha256": "0" * 64, "notes": {"artifact": artifact, "target_origin": "dataset-qa",
                                                      "license_provenance": {"class": "share-alike"}}},
            "license": {"spdx": "CC-BY-SA-4.0", "noncommercial": False}}


def test_stage_records_render_views_for_the_trajectory_trainer(tmp_path):
    a = _port("a", "compare", pairs=["b"])
    b = _port("b", "compare", instructions="Keep what is needed to answer this question: when?", pairs=["a"], target="1999")
    rec = _port("r", "reconstruct", instructions="Keep the whole document, so that it can be reproduced exactly.",
                target="The value text. " * 20)
    val = _port("v", "consume", split="validation")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "x.port-records.jsonl").write_text("".join(json.dumps(r) + "\n" for r in (a, b, rec, val)))
    summary = vr.convert(corpus, tmp_path / "out")
    assert summary["records"] == 3 and summary["not_converted"] == {"validation": 1}
    rows = {r["id"]: r for r in map(json.loads, (tmp_path / "out/records.jsonl").read_text().splitlines())}
    pieces = [json.loads(l) for l in (tmp_path / "out/pieces.jsonl").read_text().splitlines()]
    assert pieces == [{"name": "prompt:view", "kind": "system-prompt", "text": INSTRUCTIONS}]
    qa = rows["view-stage:a"]
    part = qa["messages"][0]["content"][0]
    assert part["type"] == "view" and part["instructions"] == a["writer"]["instructions"]
    assert part["source"] == part["preview"] == a["sources"][0]["text"]  # crisp = the full-text reader
    assert qa["messages"][0]["content"][1]["text"].endswith("Who?")
    assert qa["target"] == {"role": "assistant", "content": "Ann"}
    assert qa["view_stage"]["contrast_instructions"] == [b["writer"]["instructions"]]
    assert qa["cohort"] == "view" and qa["training_admission"]["approved"] is False
    faithful = rows["view-stage:r"]["messages"][0]["content"][0]
    assert faithful["faithful"] is True and "instructions" not in faithful


def test_consumer_messages_and_gate_verdict():
    port = _port("a", "consume")
    assert vr.consumer_messages(port, {"block": "nz1"})[0]["content"][0] == {"type": "neuralese", "id": "nz1"}
    assert vr.consumer_messages(port, {"text": "full"})[0]["content"] == "full\n\nWho?"
    assert vr.consumer_messages(port, {})[0]["content"] == "Who?"
    good = {a: {"recovery": 0.95} for a in vg.ARTIFACTS}
    report = {"reconstruction": good, "qa": good, "shuffle": {a: {"margin": 0.2} for a in vg.ARTIFACTS},
              "purpose": {"prose": {"margin": 0.1}}, "harness": {"records": 4, "delta_views_minus_preview": 0.0}}
    assert vg.verdict(report, vg.DEFAULTS)["view_gate_passed"] is True
    report["purpose"] = {"prose": {"margin": -0.1}}
    out = vg.verdict(report, vg.DEFAULTS)
    assert out["view_gate_passed"] is False and out["checks"]["purpose"] is False
    report["purpose"] = {"prose": {"margin": 0.1}}
    del report["qa"]["log"]
    assert vg.verdict(report, vg.DEFAULTS)["missing_artifacts"] == ["log"]


def test_gate_selection_pairs_purposes_over_one_value():
    a = _port("a", "compare", split="test", pairs=["b"])
    b = _port("b", "compare", split="test", instructions="Keep what is needed to answer this question: when?", pairs=["a"])
    r = _port("r", "reconstruct", split="test")
    chosen = vg.select([a, b, r], per_artifact=4, seed=0)
    assert [x["id"] for x in chosen["reconstruct"]["prose"]] == ["r"]
    pairs = {(p["id"], q["id"]) for p, q in chosen["compare"]["prose"]}
    assert pairs == {("a", "b"), ("b", "a")}
