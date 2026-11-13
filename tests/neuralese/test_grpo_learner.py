"""CPU checks of the GRPO learner's host-side pieces (scripts/neuralese_data/grpo.py)."""
import collections
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/neuralese_data"))
import grpo  # noqa: E402
from lower_rollout_turns import episode_turns  # noqa: E402


def _turn(samples, temperature=1, guidance=False):
    return {'invocation_id': 'i', 'sampling': {
        'prompt': [{'kind': 'tokens', 'token_ids': [1, 2, 3]}], 'policy': {'checkpoint_sha256': 'p'},
        'sampling_law': {'temperature': temperature, 'guidance': guidance, 'probabilities': 'post-processing-normalized'},
        'samples': [{'token_id': t, 'logp': lp, 'input_role': 'ordinary'} for t, lp in samples]}}


def test_only_calls_sampled_from_the_learners_law_train():
    """A call without per-token logprobs (None -> NaN) would make rho, the loss and every master NaN; a call sampled at
    another temperature or with guidance is another law than the learner's T=1 softmax."""
    row = {'turns': [_turn([(5, -0.1), (6, -0.2)]), _turn([(5, None)]), _turn([(5, -0.1)], temperature=0.7),
                     _turn([(5, -0.1)], guidance=True)]}
    stats = collections.Counter()
    rows = episode_turns(row, stats)
    assert [r['temperature'] for r in rows] == [1, 1, 0.7, 1]
    kept = grpo.trainable_rows(rows, stats)
    assert [r['index'] for r in kept] == [0]
    assert stats['skip_nonfinite_actor_logp'] == 1 and stats['skip_other_sampling_law'] == 2


def test_group_advantages():
    assert grpo.group_advantages([1., 1., 1.]) is None
    a = grpo.group_advantages([1., 0., 0., 0.])
    assert a[0] > 0 > a[1] and abs(sum(a)) < 1e-9


def test_decision_tokens_count_the_prompt_once_per_option():
    row = {'prompt': np.zeros(100, np.int32), 'options': [np.zeros(3, np.int32), np.zeros(5, np.int32)]}
    assert grpo.decision_tokens(row) == 208


def _sync(tmp_path):
    torch = pytest.importorskip('torch')
    sync = object.__new__(grpo.WeightSync)
    sync.header, sync.buffer = b'h', torch.zeros(8, dtype=torch.uint8)
    sync.weights_dir, sync.sglang_url, sync.front_url = tmp_path, None, None
    sync.thread, sync.last = None, {'version': 0, 'ok': True}
    return sync


def test_a_checkpoint_write_failure_keeps_the_actor_update(tmp_path):
    sync, done = _sync(tmp_path), []

    def fail():
        raise OSError('disk full')
    sync.publish(3, lambda version, ok: done.append((version, ok)), also=fail)
    sync.wait()
    assert done == [(3, True)] and sync.last['ok'] and 'disk full' in sync.last['checkpoint_error']


def test_a_publish_failure_is_this_versions_failure(tmp_path):
    """An exception in the publish thread used to leave the previous version's ok in ``last`` (the loop went on)."""
    sync, done = _sync(tmp_path), []
    sync.weights_dir = tmp_path / 'missing'          # the export write fails
    sync.publish(4, lambda version, ok: done.append((version, ok)))
    sync.wait()
    assert done == [(4, False)] and sync.last['version'] == 4 and not sync.last['ok']
    assert 'publish failed' in sync.last['answer']
