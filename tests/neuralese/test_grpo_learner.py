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
    assert sorted(p.name for p in directory.iterdir()) == ['cases.jsonl', 'retired', 'rewards.jsonl', 'score.json', 'versions.json']


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


def test_episode_loss_scales_weigh_every_episode_equally_and_a_decision_like_a_call():
    import torch
    # episode 0: 6 tokens in 2 calls and 1 decision; episode 1: 30 tokens in 3 calls; episode 2: nothing trained
    scales = grpo.loss_scales({0: [6, 2, 1], 1: [30, 3, 0], 2: [0, 0, 0]}, "episode", "call")
    assert scales[2] == (0., 0.)
    assert grpo.decision_unit(6, 2, "call") == 3. and grpo.decision_unit(6, 2, "token") == 1.
    assert grpo.decision_unit(0, 0, "call") == 1.
    # per episode: tokens * token factor + decisions * decision factor = 1 / trained episodes
    assert 6 * scales[0][0] + scales[0][1] == pytest.approx(.5)
    assert 30 * scales[1][0] == pytest.approx(.5)
    assert scales[0][1] == pytest.approx(3 * scales[0][0])       # the decision weighs like a 3-token call
    # accumulating chain by chain equals A_i * mean over the episode's units, averaged over episodes
    logp = torch.linspace(-2., -.1, 36, requires_grad=True)
    decision = torch.tensor(-.7, requires_grad=True)
    advantages = {0: 1.5, 1: -.5}
    chains = [(0, slice(0, 4)), (1, slice(6, 36)), (0, slice(4, 6))]   # episode 0 in two chains
    for episode, where in chains:
        (-(advantages[episode] * logp[where]).sum() * scales[episode][0]).backward()
    (-(advantages[0] * decision) * scales[0][1]).backward()
    expected = torch.cat([torch.full((6,), -1.5 / 9 / 2), torch.full((30,), .5 / 30 / 2)])
    assert torch.allclose(logp.grad, expected)
    assert decision.grad.item() == pytest.approx(-1.5 * 3 / 9 / 2)


def test_token_loss_scales_are_the_old_sum_over_all_units():
    scales = grpo.loss_scales({0: [6, 2, 1], 1: [30, 3, 0]}, "token", "token")
    assert scales[0] == pytest.approx((1 / 37, 1 / 37)) and scales[1] == pytest.approx((1 / 37, 1 / 37))
    scales = grpo.loss_scales({0: [6, 2, 1], 1: [30, 3, 0]}, "token", "call")
    assert scales[0] == pytest.approx((1 / 39, 3 / 39))


def test_tail_guard_masks_only_tail_tokens_of_positive_episodes():
    import torch
    rho = torch.tensor([1., 3., .5, 1.])
    actor = torch.tensor([-.1, -7.5, -9., -6.9])
    weight, tail = grpo.policy_weight(rho, actor, 1.2, 2., -7.)
    assert weight.tolist() == [1., 0., 0., 1.] and tail.tolist() == [False, True, True, False]
    weight, tail = grpo.policy_weight(rho, actor, -1.2, 2., -7.)       # a failed episode still pushes its tail down
    assert weight.tolist() == [1., 2., .5, 1.] and not tail.any()
    for off in (0., None):
        weight, tail = grpo.policy_weight(rho, actor, 1.2, 2., off)
        assert weight.tolist() == [1., 2., .5, 1.] and not tail.any()
    weight, tail = grpo.policy_weight(torch.tensor(1.), torch.tensor(-8.), 1., 2., -7.)   # a decision (0-dim)
    assert weight.item() == 0. and int(tail) == 1


def test_think_mask_and_moment_cosine():
    import torch
    assert grpo.think_mask(np.array([5, 6, 9, 7, 9, 8]), 9).tolist() == [True, True, True, False, False, False]
    assert grpo.think_mask(np.array([5, 6]), 9).all() and grpo.think_mask(np.array([5, 6]), None).all()
    a, b = torch.tensor([1., 0.]), torch.tensor([[0., 2.]])
    a.grad, b.grad = torch.tensor([3., 0.]), torch.tensor([[0., 4.]])
    import types
    optimizer = types.SimpleNamespace(state={})
    assert grpo.moment_cosine([a, b], optimizer) is None
    optimizer.state = {a: {"exp_avg": torch.tensor([1., 0.])}, b: {"exp_avg": torch.tensor([[0., -1.]])}}
    assert grpo.moment_cosine([a, b], optimizer) == pytest.approx((3 - 4) / (5 * 2 ** .5))


def test_the_pop_loop_reads_held_complete_from_the_module():
    """A local named like the module helper in train() made it a closure variable of the Pop loop (NameError at start)."""
    import symtable
    table = symtable.symtable(Path(grpo.__file__).read_text(), 'grpo.py', 'exec')
    scopes = [table]
    while scopes:
        scope = scopes.pop()
        scopes += scope.get_children()
        names = scope.get_identifiers()
        if 'held_complete' in names and scope.get_type() == 'function':
            assert scope.lookup('held_complete').is_global(), scope.get_name()
