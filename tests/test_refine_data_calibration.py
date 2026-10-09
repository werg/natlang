"""refine-calibration: queue, review tool, verdict file and the calibration metrics."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from refine_data import calibration, exact  # noqa: E402


def _rows():
    rows, _ = exact.generate(per_predicate=6)
    return rows


def test_queue_is_balanced_per_predicate_and_hides_model_probabilities():
    rows = _rows()
    queue = calibration.build_queue(rows, per_predicate=4)
    assert len(queue) >= 4 * len({row["predicate_id"] for row in rows}) - 4
    per = {}
    for item in queue:
        per.setdefault(item["predicate_id"], []).append(item["stratum"])
        assert set(item) == {"schema", "id", "predicate", "predicate_id", "family", "value", "value_sha256", "source", "stratum"}
    assert all(len(strata) <= 4 and abs(strata.count("positive") - strata.count("negative")) <= 1 for strata in per.values())
    assert sum(len(strata) == 4 for strata in per.values()) >= len(per) - 2
    assert calibration.build_queue(rows, per_predicate=4) == queue, "deterministic"
    assert isinstance(queue[0]["value"], str)


def test_review_appends_verdicts_and_resumes(tmp_path):
    queue = calibration.build_queue(_rows(), per_predicate=1)[:4]
    path = tmp_path / "verdicts.jsonl"
    answers = iter(["t", "f", "u", "needs outside knowledge", "s", "q"])
    shown = []
    stats = calibration.review(queue, path, "tester", ask=lambda prompt: next(answers), show=shown.append)
    assert stats == {"pending_at_start": 4, "recorded": 3, "skipped": 1}
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["verdict"] for r in records] == [True, False, "unclear"]
    assert records[2]["note"] == "needs outside knowledge" and records[0]["reviewer"] == "tester" and records[0]["schema"] == calibration.VERDICT_SCHEMA
    assert any("Property: the value is" in text for text in shown)
    # A second session sees only what is still pending (the skipped item and the one never shown).
    again = calibration.review(queue, path, "tester", ask=lambda prompt: "q", show=lambda text: None)
    assert again["pending_at_start"] == 1
    assert calibration.status(queue, records)["reviewed"] == 3


def test_latest_verdict_wins_and_unclear_is_excluded_from_rows_and_metrics():
    queue = calibration.build_queue(_rows(), per_predicate=1)[:3]
    first = [calibration.make_verdict(queue[0], True, "a"), calibration.make_verdict(queue[1], "unclear", "a"), calibration.make_verdict(queue[2], False, "a")]
    revised = first + [calibration.make_verdict(queue[0], False, "b", review_round=2)]
    rows = calibration.to_rows(revised)
    assert [row["gold"][0] for row in rows] == sorted([0.0, 0.0], key=float) and len(rows) == 2
    assert all(row["role"] == "heldout" and row["split"] == "calibration" and row["family"] == calibration.FAMILY for row in rows)
    assert rows[0]["messages"][1]["content"].count(queue[0]["predicate"]) >= 0
    report = calibration.calibration_report(revised, {q["id"]: p for q, p in zip(queue, (0.9, 0.5, 0.2))})
    assert report["all"]["n"] == 2
    assert abs(report["all"]["brier"] - ((0.9 - 0) ** 2 + (0.2 - 0) ** 2) / 2) < 1e-9


def test_metrics_on_known_cases():
    perfect = [(1.0, True), (0.0, False)]
    assert calibration.brier(perfect) == 0 and calibration.ece(perfect) == 0
    overconfident = [(0.9, True)] * 5 + [(0.9, False)] * 5
    assert abs(calibration.ece(overconfident) - 0.4) < 1e-9
    assert abs(calibration.brier(overconfident) - (0.01 + 0.81) / 2) < 1e-9
    assert calibration.make_verdict(calibration.build_queue(_rows(), 1)[0], "unclear", "r")["verdict"] == "unclear"
    try:
        calibration.make_verdict(calibration.build_queue(_rows(), 1)[0], "maybe", "r")
    except ValueError:
        pass
    else:
        raise AssertionError("an unknown verdict is rejected")
