"""Learned updater v0 data plumbing: recorded soft-gold deltas and the support view (no model needed)."""
import json
from types import SimpleNamespace

import torch

from natlang_neuralese.train import learned_updater


def test_examples_are_in_place_soft_gold_deltas(tmp_path, monkeypatch):
    base, after, other = torch.zeros(3, 4), torch.ones(3, 4), torch.ones(5, 4)
    exports = {name: SimpleNamespace(block_id=bid, payload=value, dialect="nd:natlang@1")
               for name, bid, value in (("a", "b0", base), ("b", "b1", after), ("c", "b2", other))}
    monkeypatch.setattr(learned_updater, "read_nz", lambda path: ({}, exports))
    steps = [
        {"id": "s1", "episode": {"family": "decision:x"}, "operator": {"kind": "method-arm:soft-gold"},
         "before": [{"kind": "soft-skill", "id": "b0"}], "after": [{"kind": "soft-skill", "id": "b1"}], "outcome": {}},
        {"id": "s2", "episode": {"family": "decision:y"}, "operator": {"kind": "method-arm:soft-teacher"},
         "before": [{"kind": "soft-skill", "id": "b0"}], "after": [{"kind": "soft-skill", "id": "b1"}]},
        {"id": "s3", "episode": {"family": "decision:z"}, "operator": {"kind": "method-arm:soft-gold"},
         "before": [{"kind": "soft-skill", "id": "b0"}], "after": [{"kind": "soft-skill", "id": "b2"}]},  # shape change
    ]
    (tmp_path / "improvement-steps.jsonl").write_text("".join(json.dumps(s) + "\n" for s in steps))
    (tmp_path / "artifacts").mkdir()
    (tmp_path / "artifacts" / "decision_x.nz").write_bytes(b"")
    dialect, examples = learned_updater.load_examples(tmp_path)
    assert dialect == "nd:natlang@1" and [e["step"] for e in examples] == ["s1"]
    assert torch.equal(examples[0]["delta"], after - base) and examples[0]["family"] == "decision:x"


def test_support_view_interleaves_answer_strata_and_only_uses_train_cases(tmp_path):
    rows = [{"id": f"c{i}", "family": "decision:x", "role": "train", "question": "Spam?", "state": f"text {i}",
             "answer": "True" if i < 6 else "False"} for i in range(8)]
    rows.append({"id": "h0", "family": "decision:x", "role": "heldout", "question": "Spam?", "state": "secret", "answer": "True"})
    path = tmp_path / "cases.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    view = learned_updater.support_view(path, "decision:x", count=4)
    assert view.startswith("Question: Spam?") and "secret" not in view
    answers = [line.split(": ", 1)[1] for line in view.splitlines() if line.startswith("Answer:")]
    assert answers.count("True") == 2 and answers.count("False") == 2
