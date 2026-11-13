"""CPU checks of grpo.py's task supply arguments: errata exclusion by default, one episode budget with held eval."""
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
from natlang_neuralese.rl import held_eval  # noqa: E402

PARSE_ARGS = argparse.ArgumentParser.parse_args


def parsed(monkeypatch, *extra):
    """grpo.py train's parsed arguments (main stops before any work)."""
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


def test_errata_files_default_to_the_errata_directory_with_an_opt_out(tmp_path, monkeypatch):
    (tmp_path / 'b.jsonl').write_text('{"id": "y"}\n')
    (tmp_path / 'a.jsonl').write_text('{"id": "x"}\n')
    (tmp_path / 'notes.txt').write_text('not errata')
    monkeypatch.setattr(grpo, 'ERRATA', str(tmp_path))
    args = parsed(monkeypatch)
    assert args.exclude_ids == [] and not args.no_default_errata
    assert grpo.errata_files(args) == [str(tmp_path / 'a.jsonl'), str(tmp_path / 'b.jsonl')]
    assert grpo.errata_files(parsed(monkeypatch, '--no-default-errata')) == []
    assert grpo.errata_files(parsed(monkeypatch, '--exclude-ids', 'own.jsonl')) == ['own.jsonl']
    monkeypatch.setattr(grpo, 'ERRATA', str(tmp_path / 'missing'))
    assert grpo.errata_files(SimpleNamespace(exclude_ids=[], no_default_errata=False)) == []


def test_training_and_held_eval_share_one_episode_budget(monkeypatch):
    assert held_eval.EPISODE_MODEL_CALLS == 40
    assert parsed(monkeypatch).max_model_requests == held_eval.EPISODE_MODEL_CALLS
    assert parsed(monkeypatch, '--max-model-requests', '64').max_model_requests == 64
    monkeypatch.setattr(argparse.ArgumentParser, 'parse_args', PARSE_ARGS)
    seen = {}
    monkeypatch.setattr(held_eval, 'measure', lambda *a, **k: seen.update(k))
    held_eval.main(['measure', '--manifest', 'm.json', '--out', 'o', '--endpoint', 'http://x=1', '--policy-id', 'p'])
    assert seen['max_model_requests'] == held_eval.EPISODE_MODEL_CALLS


def test_reward_defaults_to_graded_with_a_binary_opt_out(monkeypatch):
    assert parsed(monkeypatch).reward == 'graded'
    assert parsed(monkeypatch, '--reward', 'binary').reward == 'binary'
