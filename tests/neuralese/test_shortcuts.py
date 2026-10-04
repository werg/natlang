"""Shortcut checks for supervised stop decisions."""

from natlang_neuralese.data.shortcuts import audit_lengths, count_baseline, count_hazard, length_profile


def test_a_single_span_length_is_flagged_and_counting_places_every_end():
    report = audit_lengths([16] * 200, "phase A")
    assert report["profile"]["distinct"] == 1 and report["profile"]["most_common_share"] == 1.0
    assert report["count_baseline"]["stop_recall"] == 1.0 and report["count_baseline"]["bce"] < 1e-4
    assert len(report["flags"]) == 2


def test_varied_lengths_leave_the_head_something_to_learn():
    lengths = [n for n in range(4, 33) for _ in range(10)]
    report = audit_lengths(lengths, "phase D")
    assert report["flags"] == []
    baseline = report["count_baseline"]
    assert baseline["stop_recall"] < 0.5 and baseline["bce"] > 0.05
    assert length_profile(lengths)["distinct"] == 29


def test_hazard_is_the_conditional_stop_probability():
    hazard = count_hazard([1, 2, 2, 4])
    assert hazard == {1: 0.25, 2: 2 / 3, 3: 0.0, 4: 1.0}
    # A fixed hazard scores other lengths: a length the hazard never stops at costs its final decision.
    assert count_baseline([3], hazard)["bce"] > 1
