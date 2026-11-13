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


def test_pruned_groups_keep_their_rollouts_compressed(tmp_path):
    import types
    import zstandard
    directory = tmp_path / 'groups' / 'g000007'
    (directory / 'workers').mkdir(parents=True)
    rollouts = b''.join(b'{"task_id": "t", "turns": []}\n' for _ in range(100))
    (directory / 'rollouts.jsonl').write_bytes(rollouts)
    for name in ('score.json', 'rewards.jsonl', 'versions.json', 'cases.jsonl', 'journal.jsonl'):
        (directory / name).write_text('{}\n')
    collector = types.SimpleNamespace(groups={7: {'dir': directory}})
    stats = collections.Counter()
    grpo.prune_group(collector, 7, keep=False, archive=tmp_path / 'rollouts', stats=stats)
    archived = tmp_path / 'rollouts' / 'g000007.rollouts.jsonl.zst'
    assert zstandard.ZstdDecompressor().stream_reader(archived.open('rb')).read() == rollouts
    assert stats['rollouts_archived'] == 1 and stats['rollouts_archived_bytes'] == archived.stat().st_size
    assert sorted(p.name for p in directory.iterdir()) == ['cases.jsonl', 'rewards.jsonl', 'score.json', 'versions.json']


def test_a_failed_archive_keeps_the_group_in_place(tmp_path):
    import types
    directory = tmp_path / 'g000008'
    directory.mkdir()
    (directory / 'rollouts.jsonl').write_text('{}\n')
    (tmp_path / 'blocked').write_text('a file, not a directory')
    stats = collections.Counter()
    grpo.prune_group(types.SimpleNamespace(groups={8: {'dir': directory}}), 8, keep=False,
                     archive=tmp_path / 'blocked', stats=stats)
    assert (directory / 'rollouts.jsonl').exists() and stats['rollout_archive_failed'] == 1


def test_decision_diagnostics_separate_numerics_from_drift():
    import math
    row = {'prompt': np.arange(50, dtype=np.int32), 'options': [np.array([1, 2]), np.array([3, 4, 5])], 'chosen': 1,
           'temperature': 1.0, 'policy': 'p', 'index': 3, 'invocation': 'i', 'imports': []}
    actor = np.array([-1.0, -2.0])
    row['behavior_logq'] = -2.0 - math.log(math.exp(-1.0) + math.exp(-2.0))
    item = {'row': row, 'advantage': 0.5, 'group': 4, 'task_id': 't', 'stale': 1, 'actor_version': 2}
    learner = actor + 3.0                       # a shared offset: the categorical is unchanged
    reference = actor + np.array([0.0, 0.4])    # the start policy prefers the chosen option more
    record = grpo.decision_record(item, learner, actor, reference, 0.0)
    assert record['actor_logq_recomputed'] == pytest.approx(row['behavior_logq'], abs=1e-5)
    assert record['learner_logpi'] == pytest.approx(row['behavior_logq'], abs=1e-5)
    assert record['dscore_abs_mean'] == pytest.approx(3.0) and record['dscore_centered_abs_max'] == pytest.approx(0.)
    assert record['ref_log_rho'] > 0 and record['option_ids'] == [[1, 2], [3, 4, 5]]
    other = dict(record, log_rho=-0.9, stale=0)
    summary = grpo.decision_summary([record, other])
    assert summary['n'] == 2 and summary['abs_log_rho_max'] == pytest.approx(0.9)
    assert summary['abs_log_rho_mean_by_stale'] == {'0': 0.9, '1': 0.0} and summary['worst']['stale'] == 0
    assert summary['option_tokens_max'] == 3 and summary['prompt_tokens_max'] == 50
    assert grpo.decision_summary([]) == {}
