"""CPU checks of the GRPO actor's overlapped scheduler switch and the engine-mismatch telemetry that guards it."""
import argparse
import collections
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402

PARSE_ARGS = argparse.ArgumentParser.parse_args


def parsed(monkeypatch, *extra):
    seen = {}

    def capture(self, args=None, namespace=None):
        seen['args'] = PARSE_ARGS(self, args, namespace)
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', capture)
    monkeypatch.setattr(sys, 'argv', ['grpo.py', 'train', '--work', 'w', '--checkpoint', 'c', '--checkpoint-sha256', 'x',
                                      *extra])
    with pytest.raises(SystemExit):
        grpo.main()
    return seen['args']


def test_overlap_is_on_unless_held_evals_measure_on_the_training_actor(monkeypatch):
    assert parsed(monkeypatch).actor_overlap == 'auto'
    # held evals on Pop: the training actor serves only collection
    assert grpo.actor_overlap(parsed(monkeypatch, '--held-remote', 'pop'))
    # the default measures held on the training actor, whose pinned eval engine serves without overlap
    assert not grpo.actor_overlap(parsed(monkeypatch))
    assert not grpo.actor_overlap(parsed(monkeypatch, '--held-every-hours', '0'))   # the start-policy measurement
    assert grpo.actor_overlap(parsed(monkeypatch, '--held-every-hours', '0', '--skip-base-held'))
    assert not grpo.actor_overlap(parsed(monkeypatch, '--held-every-hours', '0', '--skip-base-held', '--held-extra', 'a'))
    assert not grpo.actor_overlap(parsed(monkeypatch, '--held-remote', 'pop', '--actor-overlap', 'off'))
    assert grpo.actor_overlap(parsed(monkeypatch, '--actor-overlap', 'on'))


def test_mismatch_counts_tokens_far_from_the_actors_law_apart_from_consumer_calls():
    log_rho = torch.tensor([0.01, -23.8, 6.0, -5.5, -4.9, 0.0])
    consumer = torch.tensor([False, False, False, True, False, False])
    out = grpo.actor_mismatch_counts(log_rho, consumer)
    assert out == {'below': 2, 'above': 1, 'below_ordinary': 1, 'min': pytest.approx(-23.8)}
    assert grpo.actor_mismatch_counts(log_rho)['below_ordinary'] == 2
    assert grpo.actor_mismatch_counts(torch.zeros(0))['min'] == 0.


def test_mismatch_flag_needs_a_rate_and_a_few_tokens():
    sums = collections.Counter(mismatch_below=4, mismatch_above=1, mismatch_below_ordinary=3)
    sums['mismatch_min'] = -24.
    summary, flag = grpo.actor_mismatch_summary(sums, 200_000)   # 15 per million: the 2026-11-03 defect's rate
    assert flag and summary['below_ordinary_per_m'] == 15. and summary['min_log_rho'] == -24.
    assert not grpo.actor_mismatch_summary(sums, 2_000_000)[1]   # 1.5 per million
    assert not grpo.actor_mismatch_summary(collections.Counter(mismatch_below_ordinary=2), 1000)[1]   # too few tokens
    assert grpo.actor_mismatch_summary(collections.Counter(), 0) == (
        {'below': 0, 'above': 0, 'below_ordinary': 0, 'below_ordinary_per_m': 0., 'min_log_rho': 0., 'nats': 5.}, False)
