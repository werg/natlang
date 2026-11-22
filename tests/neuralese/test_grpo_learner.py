"""CPU checks of the GRPO learner's host-side pieces (scripts/neuralese_data/grpo.py)."""
import collections
import json
import math
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
    """A WeightSync without a learner or mapped slots: what release/restore/publish do around the actor, publishing the
    start export (``publish_start``)."""
    pytest.importorskip('torch')
    sync = object.__new__(grpo.WeightSync)
    sync.export_dir, sync.weights_dir, sync.sglang_url, sync.front_url = tmp_path, tmp_path, None, None
    sync.thread, sync.last, sync.released, sync.pending, sync.publishing = None, {'version': 0, 'ok': True}, None, None, None
    sync.sentinel_ids, sync.slots, sync.staging = [], [], 0
    sync.start_digests = {'weights_sha256': '1' * 64, 'tensor_identity': '2' * 64}
    sync.last_good = {'version': 0, 'dir': str(tmp_path), 'policy_id': 'a' * 64, 'slot': None, **sync.start_digests}
    sync.pending = {'kind': 'start', 'policy_id': 'a' * 64}
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
    sync.pending = None                       # nothing to publish: the thread raises
    sync.publish(4, lambda version, ok: done.append((version, ok)))
    sync.wait()
    assert done == [(4, False)] and sync.last['version'] == 4 and not sync.last['ok']
    assert 'publish failed' in sync.last['answer']


def _recording_post(monkeypatch, answers=None):
    calls = []

    def post(url, body=None, timeout=600):
        path = url.rsplit('/', 1)[1]
        calls.append((path, body))
        return (answers or {}).get(path, json.dumps({'success': True}))
    monkeypatch.setattr(grpo, 'post', post)
    return calls


def test_a_released_actor_resumes_before_its_weight_update(tmp_path, monkeypatch):
    """--actor-release: retract, release the tags for the learner phase; the step's publish resumes the memory before
    reloading the weights (released weights are undefined) and only then continues generation."""
    calls = _recording_post(monkeypatch)
    sync, done = _sync(tmp_path), []
    sync.sglang_url, sync.released = 'http://sglang', None
    sync.release(['kv_cache', 'weights', 'cuda_graph'])
    assert [c[0] for c in calls] == ['pause_generation', 'release_memory_occupation']
    assert calls[0][1] == {'mode': 'retract'} and calls[1][1] == {'tags': ['kv_cache', 'weights', 'cuda_graph']}
    calls.clear()
    sync.publish(5, lambda version, ok: done.append((version, ok)))
    sync.wait()
    assert [c[0] for c in calls] == ['resume_memory_occupation', 'pause_generation', 'update_weights_from_disk',
                                     'continue_generation']
    assert calls[0][1] == {'tags': ['kv_cache', 'weights', 'cuda_graph']}
    assert done == [(5, True)] and sync.released is None and 'actor_resume_s' in sync.last
    calls.clear()
    sync.pending = {'kind': 'start', 'policy_id': 'a' * 64}
    sync.publish(6, lambda version, ok: None)   # not released: no resume
    sync.wait()
    assert [c[0] for c in calls] == ['pause_generation', 'update_weights_from_disk', 'continue_generation']


def test_a_skipped_update_restores_the_released_actor_on_its_served_weights(tmp_path, monkeypatch):
    calls = _recording_post(monkeypatch)
    sync = _sync(tmp_path)
    sync.sglang_url, sync.released = 'http://sglang', None
    sync.restore(3)                     # nothing released: no call
    assert calls == []
    sync.release(['kv_cache'])
    calls.clear()
    sync.restore(3)
    assert [c[0] for c in calls] == ['resume_memory_occupation', 'update_weights_from_disk', 'continue_generation']
    assert calls[1][1]['model_path'] == str(tmp_path) and calls[1][1]['weight_version'] == '0'   # the last good version
    assert sync.released is None


def test_a_failed_release_continues_generation(tmp_path, monkeypatch):
    calls = []

    def post(url, body=None, timeout=600):
        calls.append(url.rsplit('/', 1)[1])
        if calls[-1] == 'release_memory_occupation':
            raise OSError('not idle')
        return '{}'
    monkeypatch.setattr(grpo, 'post', post)
    sync = _sync(tmp_path)
    sync.sglang_url, sync.released = 'http://sglang', None
    with pytest.raises(OSError):
        sync.release(['kv_cache'])
    assert calls == ['pause_generation', 'release_memory_occupation', 'continue_generation'] and sync.released is None


def test_no_actor_sync_releases_nothing(tmp_path, monkeypatch):
    calls = _recording_post(monkeypatch)
    sync = _sync(tmp_path)
    sync.released = None
    assert sync.release(['kv_cache']) == 0. and calls == [] and sync.released is None


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


def test_k2_kl_gradient_stays_linear_on_tail_tokens():
    import torch
    logp = torch.tensor([-.5, -11., -1.], requires_grad=True)
    ref = torch.tensor([-.5, -.5, -1.2])
    grpo.kl_term(ref - logp, 'k2').sum().backward()
    assert logp.grad.tolist() == pytest.approx([0., -10.5, .2])
    logp.grad = None
    grpo.kl_term(ref - logp, 'k3').sum().backward()
    assert logp.grad[1].item() < -3e4                      # 1 - e^10.5: why k3 is not the loss
    sums = collections.Counter()
    grpo.kl_stats(sums, ref - logp)
    grpo.kl_stats(sums, torch.tensor(.1))                 # a decision (0-dim)
    assert sums['ref_dlogp_over_5'] == 1 and sums['ref_dlogp_max'] == pytest.approx(10.5)
    assert sums['kl'] == pytest.approx(float(sum(math.exp(x) - x - 1 for x in (0., 10.5, -.2, .1))), rel=1e-5)
    assert sums['kl_k2'] == pytest.approx(sum(.5 * x * x for x in (0., 10.5, -.2, .1)), rel=1e-5)
    assert sums['kl_k3_max'] == pytest.approx(math.exp(10.5) - 11.5, rel=1e-5)   # one token is ~all of the k3 sum
    assert sums['kl_k3_max'] / sums['kl'] > .99 and sums['ref_dlogp_pos_max'] == pytest.approx(10.5)


def test_close_push_reads_the_close_probability_from_the_target_logp_and_is_zero_on_policy():
    import torch
    torch.manual_seed(0)
    vocab, close = 11, 3
    h = torch.randn(40, 8)
    weight = torch.randn(vocab, 8)
    positions = torch.arange(5, 35)
    logits = (h[positions] @ weight.T).log_softmax(-1)
    targets = torch.multinomial(logits.exp(), 1, generator=torch.Generator().manual_seed(1))[:, 0]
    target_logp = logits.gather(1, targets[:, None])[:, 0]
    p_close = grpo.close_probability(h, positions, weight, targets, target_logp, close, chunk=7)
    assert p_close.tolist() == pytest.approx(logits[:, close].exp().tolist(), rel=1e-4)
    mask = torch.ones(len(targets), dtype=torch.bool)
    # a reinforced close raises the close logit; a pushed-down continuation raises it too (by its p_close)
    one = torch.tensor([.1, .2, .3])
    assert grpo.close_push(torch.ones(1), torch.tensor([close]), torch.tensor([.3]), close, torch.tensor([True])) \
        == pytest.approx(.7)
    assert grpo.close_push(-torch.ones(1), torch.tensor([0]), torch.tensor([.3]), close, torch.tensor([True])) \
        == pytest.approx(.3)
    assert grpo.close_push(torch.ones(3), torch.tensor([0, 1, 2]), one, close, torch.tensor([True, False, True])) \
        == pytest.approx(-.4)
    # on policy, a coefficient that does not depend on the draw pushes nothing in expectation
    draws = torch.multinomial(logits.exp(), 4000, replacement=True, generator=torch.Generator().manual_seed(2))
    pushes = [grpo.close_push(torch.ones(len(targets)), draws[:, j], logits[:, close].exp(), close, mask)
              for j in range(draws.shape[1])]
    assert abs(np.mean(pushes)) < 4 * np.std(pushes) / math.sqrt(len(pushes))


def test_close_probability_and_twin_kl_read_the_trained_head_rows():
    """The learner's logits use the trainable Neuralese rows at their ids (tied head); the frozen head's rows there
    are stale from step 1. Both diagnostics must see the effective head."""
    import torch
    torch.manual_seed(3)
    vocab, close = 11, 3
    h = torch.randn(20, 8)
    weight = torch.randn(vocab, 8)
    rows = torch.randn(2, 8, requires_grad=True)
    ids = torch.tensor([close, 7])               # </think> itself is one of the trained ids here
    effective = weight.clone()
    effective[ids] = rows.detach()
    positions = torch.arange(20)
    logits = (h @ effective.T).log_softmax(-1)
    targets = torch.tensor([7, 1] * 10)
    target_logp = logits.gather(1, targets[:, None])[:, 0]
    p_close = grpo.close_probability(h, positions, weight, targets, target_logp, close, trainable=(ids, rows))
    assert p_close.tolist() == pytest.approx(logits[:, close].exp().tolist(), rel=1e-4)
    stale = grpo.close_probability(h, positions, weight, targets, target_logp, close)
    assert not torch.allclose(stale, p_close)
    top = torch.tensor([[close, 7, 0]] * 20)
    teacher = torch.log_softmax(torch.randn(20, 3), -1) - 0.1
    fkl = grpo.twin_forward_kl(h, positions, weight, targets, target_logp, top, teacher, trainable=(ids, rows))
    p_t = teacher.exp()
    tail_t = 1 - p_t.sum(-1)
    p_s = logits.gather(1, top).exp()
    exact = (p_t * (teacher - p_s.log())).sum(-1) + tail_t * (tail_t.log() - (1 - p_s.sum(-1)).log())
    assert fkl.tolist() == pytest.approx(exact.tolist(), rel=1e-4, abs=1e-6)
    fkl.sum().backward()
    assert rows.grad is not None and rows.grad.abs().sum() > 0       # the twin term reaches the trained rows


def test_interface_with_rows_writes_the_trained_marker_and_override_rows():
    import torch
    interface = {'input_rows': torch.zeros(2, 4, dtype=torch.bfloat16), 'output_rows': None,
                 'extra_input_rows': torch.zeros(1, 4, dtype=torch.bfloat16), 'extra_output_rows': None}
    rows = torch.arange(12, dtype=torch.float32).reshape(3, 4)
    out = grpo.interface_with_rows(interface, torch.tensor([9, 5, 6]), rows, (5, 6), (9,))
    assert torch.equal(out['input_rows'], rows[1:].to(torch.bfloat16))
    assert torch.equal(out['extra_input_rows'], rows[:1].to(torch.bfloat16))
    assert out['output_rows'] is None and out['extra_output_rows'] is None and interface['input_rows'].sum() == 0


def test_a_resume_refuses_another_base_and_a_full_ft_state(tmp_path):
    state = tmp_path / 'state'
    learner = state / 'learner-state'
    learner.mkdir(parents=True)
    (learner / 'learner_manifest.json').write_text(json.dumps({'base_weights_sha256': 'b' * 64}))
    grpo.refuse_legacy_resume(learner, state, 'b' * 64)
    with pytest.raises(SystemExit, match='trained on base'):
        grpo.refuse_legacy_resume(learner, state, 'c' * 64)
    (learner / 'learner_manifest.json').write_text(json.dumps({'base_weights_sha256': ''}))   # unrecorded base
    with pytest.raises(SystemExit, match='unrecorded'):
        grpo.refuse_legacy_resume(learner, state, 'b' * 64)
    old = tmp_path / 'old'
    old.mkdir()
    (old / 'training-state.safetensors').write_bytes(b'x')
    with pytest.raises(SystemExit, match='full-FT training state'):
        grpo.refuse_legacy_resume(old / 'learner-state', old, 'b' * 64)
    grpo.refuse_legacy_resume(tmp_path / 'fresh' / 'learner-state', tmp_path / 'fresh', 'b' * 64)   # nothing: fresh


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


def test_prefix_tree_over_a_subset_maps_back_to_the_steps_chains():
    from natlang_neuralese.train import lfm3_fast as lfm3_fast_train
    system = np.arange(10)
    chains = [np.concatenate([system, [50 + i] * 4]) for i in range(3)] + [np.arange(30, 40)]
    keep = [0, 2, 3]                      # chain 1 reads imported state: it stays whole
    nodes = lfm3_fast_train.remap_tree(lfm3_fast_train.prefix_tree([chains[i] for i in keep], [11, 11, 3], 4), keep)
    leaves = []

    def walk(node):
        if 'index' in node:
            leaves.append((node['index'], node['start']))
        else:
            assert node['first'] in keep
            for child in node['children']:
                walk(child)
    for node in nodes:
        walk(node)
    assert sorted(leaves) == [(0, 10), (2, 10), (3, 0)]


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


def test_reasoning_style_sentinel_flags_a_halved_reasoning_length(tmp_path):
    close = 9
    chain = {'segs': [(0, 6, {'sample': [1, 2, 3, close, 4, 5]}), (6, 3, {'sample': [1, close, 4]})]}
    style = grpo.reasoning_style([chain], close)
    assert style == {'reasoning_tokens_per_call': 3.0, 'reply_tokens_per_call': 1.5}
    assert grpo.style_shift(tmp_path, style) == {}                       # no first step logged yet
    (tmp_path / 'train.jsonl').write_text('{"event": "x"}\n{"step": 1, "reasoning_tokens_per_call": 6.0}\n')
    assert grpo.style_shift(tmp_path, style) == {}                       # launch-order steps are no baseline
    (tmp_path / 'train.jsonl').write_text('{"step": 1, "reasoning_tokens_per_call": 1.0}\n'
                                          '{"step": 4, "reasoning_tokens_per_call": 6.0}\n')
    assert grpo.style_shift(tmp_path, style) == {'reasoning_vs_first_step': 0.5, 'reasoning_shift': True}
    assert grpo.style_shift(tmp_path, {'reasoning_tokens_per_call': 5.0})['reasoning_shift'] is False


def test_grad_spike_sentinel_flags_five_times_the_median(tmp_path):
    assert grpo.grad_spike(tmp_path, 6.0) == {}                           # no steps logged yet
    (tmp_path / 'train.jsonl').write_text(''.join(f'{{"step": {i}, "grad_norm": {g}}}\n'
                                                  for i, g in enumerate([0.58, 0.43, 0.76, 0.5], 1)))
    assert grpo.grad_spike(tmp_path, 6.09) == {'grad_norm_vs_median': 11.28, 'grad_spike': True}
    assert grpo.grad_spike(tmp_path, 0.66)['grad_spike'] is False


def test_advantage_controls_zero_and_shuffle_keep_the_episodes():
    import random
    items = [{'episode': 0, 'advantage': 1.}, {'episode': 0, 'advantage': 1.}, {'episode': 1, 'advantage': -1.},
             {'episode': 2, 'advantage': 0.5}]
    advantages = {0: 1., 1: -1., 2: 0.5}
    zero = [dict(i) for i in items]
    assert grpo.control_advantages(zero, advantages, 'zero', random.Random(0)) == {0: 0., 1: 0., 2: 0.}
    assert all(i['advantage'] == 0. for i in zero)
    shuffled = [dict(i) for i in items]
    new = grpo.control_advantages(shuffled, advantages, 'shuffle', random.Random(3))
    assert sorted(new.values()) == sorted(advantages.values())          # same magnitudes, permuted
    assert shuffled[0]['advantage'] == shuffled[1]['advantage'] == new[0]  # one episode, one advantage


def test_replay_collector_offers_archived_groups_in_order(tmp_path):
    import zstandard
    from types import SimpleNamespace
    archive = tmp_path / 'rollouts'
    archive.mkdir()
    for g in (3, 1, 2):
        (archive / f'g{g:06d}.rollouts.jsonl.zst').write_bytes(
            zstandard.ZstdCompressor().compress(json.dumps({'task_id': f't{g}'}).encode() + b'\n'))
    work = tmp_path / 'work'
    collector = grpo.ReplayCollector(SimpleNamespace(replay=str(archive), actor_store=str(tmp_path / 'store')), work)
    stop = threading.Event()
    assert collector.take_groups(2, 2, stop) == [1, 2]
    assert collector.groups[2]['task_id'] == 't2' and (work / 'groups' / 'g000002' / 'rollouts.jsonl').exists()
    assert collector.take_groups(2, 2, stop) == [3] and stop.is_set()    # used up: no short step afterwards
    (tmp_path / 'schedule.json').write_text('[[3, 1], [2]]')
    collector = grpo.ReplayCollector(SimpleNamespace(replay=str(archive), actor_store=str(tmp_path / 'store'),
                                                     replay_schedule=str(tmp_path / 'schedule.json')), tmp_path / 'w2')
    stop = threading.Event()
    assert collector.take_groups(24, 24, stop) == [3, 1] and not stop.is_set()    # the old run's own batches
    assert collector.take_groups(24, 24, stop) == [2] and not stop.is_set()
    assert collector.take_groups(24, 24, stop) == [] and stop.is_set()


def test_episode_normalization_rewards_short_reasoning_the_task_never_selected():
    """Exact expectation of the policy term on a reasoning-length knob the reward ignores: each call independently reasons
    1 or 2 tokens (logit theta of 1), the answer is right with p 0.5 independently, and a wrong answer costs a second call
    (grpo-v2-run1: failures 18.1 calls vs 8.9 for successes, reasoning per call 1,021 vs 1,106 tokens). The true gradient
    on theta is 0. Per-episode normalization (`episode`: A_i times the mean over its own T_i tokens) is the
    score-function estimator of E[A / T], whose T depends on every call's sampled length: short successes outweigh long
    failures and it pushes every call toward shorter reasoning. One constant normalization (`token`) is unbiased."""
    import itertools
    pi = {1: 0.5, 2: 0.5}
    score = {1: 1 - pi[1], 2: -pi[1]}       # d log pi(l) / d theta
    outcomes = []                           # (lengths of the calls, correct, probability)
    for c in (0, 1):
        for lengths in itertools.product((1, 2), repeat=1 if c else 2):
            outcomes.append((lengths, c, 0.5 * math.prod(pi[l] for l in lengths)))
    episodes = {k: (sum(lengths), len(lengths), 0) for k, (lengths, _, _) in enumerate(outcomes)}
    for norm in ('token', 'episode'):
        scales = grpo.loss_scales(episodes, norm, 'token')
        if norm == 'token':      # one constant for the whole step (the expectation, not this enumeration's total)
            scales = {k: (1., 0.) for k in scales}
        grad = sum(p * scales[k][0] * (c - 0.5) * sum(score[l] for l in lengths)
                   for k, (lengths, c, p) in enumerate(outcomes))
        if norm == 'token':
            assert abs(grad) < 1e-12
        else:
            assert grad > 0.002  # toward shorter reasoning in every call


def test_a_decision_counts_at_most_the_cap_in_token_units():
    assert grpo.decision_unit(6000, 6, "call") == 8.            # a 1,000-token average call: capped
    assert grpo.decision_unit(6000, 6, "call", cap=0) == 1000.  # uncapped
    assert grpo.decision_unit(6, 2, "call") == 3.               # under the cap: the episode's mean call


def test_lr_scales_with_the_groups_that_carry_signal():
    assert grpo.lr_data_scale(2, 12) == pytest.approx(1 / 6)    # grpo-v2-run1 step 3: 2 groups with spread
    assert grpo.lr_data_scale(18, 12) == 1. and grpo.lr_data_scale(2, 0) == 1.


def test_call_subsampling_keeps_dropped_calls_as_context():
    import random
    rows = []
    prompt = np.array([1, 2, 3])
    for i in range(6):
        sample = np.array([10 + i, 20 + i])
        rows.append({'invocation': 'a', 'index': i, 'prompt': prompt, 'sample': sample, 'imports': []})
        prompt = np.concatenate([prompt, sample, [99]])
    stats = collections.Counter()
    kept = grpo.subsample_calls({'rows': rows, 'task_id': 't', 'round': 0}, 0.5, random.Random(3), stats)
    assert 0 < len(kept['rows']) < 6 and stats['calls_subsampled_out'] == 6 - len(kept['rows'])
    import exp0_sft
    chains = exp0_sft.chains_of([kept])
    assert len(chains) == 1                                      # chained across the dropped calls
    last = kept['rows'][-1]
    assert len(chains[0]['ids']) == len(last['prompt']) + len(last['sample'])   # ends at the last kept call
    assert grpo.subsample_calls({'rows': rows}, 0.0, random.Random(0))['rows']   # at least one call


def test_behaviour_summary_and_shift():
    import behaviour_probe
    start = {'sampled_logp': [-0.65, -0.65], 'entropy': [0.76, 0.76], 'close_per_1k': [0.2, 0.2], 'close_logp': [-0.3, -0.3]}
    same = behaviour_probe.summary(start, start)
    assert same['logp_change'] == 0 and same['close_ratio'] == 1 and not behaviour_probe.shifted(same)
    step2 = {'sampled_logp': [-0.642, -0.642], 'entropy': [0.753, 0.753], 'close_per_1k': [0.22, 0.22], 'close_logp': [-0.2, -0.2]}
    assert not behaviour_probe.shifted(behaviour_probe.summary(step2, start))
    step8 = {'sampled_logp': [-0.57, -0.57], 'entropy': [0.75, 0.75], 'close_per_1k': [1.8, 1.8], 'close_logp': [-0.1, -0.1]}
    moved = behaviour_probe.summary(step8, start)
    assert moved['logp_change'] == pytest.approx(0.08) and moved['close_ratio'] == 9 and behaviour_probe.shifted(moved)
    step12 = {'sampled_logp': [-0.61, -0.61], 'entropy': [0.835, 0.835], 'close_per_1k': [0.25, 0.25], 'close_logp': [-0.3, -0.3]}
    assert behaviour_probe.shifted(behaviour_probe.summary(step12, start))   # entropy +0.075
    assert not behaviour_probe.shifted(behaviour_probe.summary(step8))   # no start measurement: no flag


def test_behaviour_end_of_turn_after_a_short_answer():
    """Probe v2's end-of-turn rows: p(<|im_end|>) where the start policy ended a short answer, against the start's own
    values (its measurement, else the probe file's ``end_start``); a mean drop beyond 1 nat flags behaviour_shift
    (grpo-v4-run2 step 12: -0.07 to -13.6 nats on gap-v1's short-answer rows, DECISIONS 2026-11-19)."""
    import math
    import behaviour_probe
    base = {'sampled_logp': [-0.65], 'entropy': [0.76], 'close_per_1k': [0.2], 'close_logp': [-0.3]}
    start = {**base, 'end_logp': [-0.1, -2.5], 'reclose_logp': [-5.0, -3.0]}
    same = behaviour_probe.summary(start, start)
    assert same['end_logp'] == pytest.approx(-1.3) and same['end_logp_change'] == 0 and not behaviour_probe.shifted(same)
    assert same['reclose_p'] == pytest.approx((math.exp(-5) + math.exp(-3)) / 2, abs=1e-5) and 'reclose_logp' not in same
    drift = {**base, 'end_logp': [-0.07, -13.6], 'reclose_logp': [-2.7, -0.04]}
    moved = behaviour_probe.summary(drift, start)
    assert moved['end_logp_change'] == pytest.approx(-5.535) and moved['end_logp_min'] == -13.6
    assert behaviour_probe.shifted(moved) and moved['logp_change'] == 0
    # A start measured before the probe had these rows: the probe's own start values.
    probe = {'end_start': {'end_logp': [-0.1, -2.5], 'reclose_logp': [-5.0, -3.0]}}
    assert behaviour_probe.summary(drift, base, probe)['end_logp_change'] == pytest.approx(-5.535)
    assert 'end_logp_change' not in behaviour_probe.summary(drift, base)   # nothing to compare with: no change, no flag


def test_replay_bench_keeps_short_groups_only(tmp_path):
    import zstandard
    from types import SimpleNamespace
    archive = tmp_path / 'rollouts'
    archive.mkdir()
    for g, tokens in ((1, 500), (2, 50_000), (3, 900)):
        row = {'task_id': f't{g}', 'turns': [{'usage': {'completion_tokens': tokens}}]}
        (archive / f'g{g:06d}.rollouts.jsonl.zst').write_bytes(zstandard.ZstdCompressor().compress(json.dumps(row).encode() + b'\n'))
    collector = grpo.ReplayCollector(SimpleNamespace(replay=str(archive), actor_store=str(tmp_path / 'store'),
                                                     replay_max_group_tokens=1000), tmp_path / 'work')
    stop = threading.Event()
    assert collector.take_groups(2, 2, stop) == [1, 3] and not (tmp_path / 'work' / 'groups' / 'g000002').exists()
    branch = grpo.ReplayCollector(SimpleNamespace(replay=str(archive), actor_store=str(tmp_path / 'store'),
                                                  replay_max_group_tokens=0), tmp_path / 'branch', first_group=2)
    assert branch.take_groups(2, 2, stop) == [2, 3]           # a resumed replay continues after its state's groups


def test_free_cuda_cache_returns_what_the_garbage_collection_frees(monkeypatch):
    """A tensor kept alive only by a reference cycle is freed by ``gc.collect`` into PyTorch's cache: an ``empty_cache``
    that ran first left it reserved (the post-optimizer read of grpo-v4-run2: 31-43 GB reserved, 24 GB live)."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('needs CUDA')
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    gc.disable()
    try:
        cycle = [torch.empty(256 << 20, dtype=torch.uint8, device='cuda')]
        cycle.append(cycle)
        del cycle
        legacy = grpo.free_cuda_cache(collect=False)           # the cycle is still alive: nothing to return
        assert legacy['gc_freed_gb'] == 0 and legacy['reserved_gb'] >= 0.25
        fixed = grpo.free_cuda_cache(collect=True)
        assert fixed['gc_freed_gb'] >= 0.24 and fixed['reserved_gb'] < 0.1
    finally:
        gc.enable()


def test_a_second_think_close_and_what_follows_up_to_the_action_are_never_reinforced():
    import exp0_sft
    import torch
    CLOSE, CALL, END = 124902, 125019, 124900
    reply = [5, 6, CLOSE, 7, 8, CLOSE, 9, CALL, 3, END]            # reasoning </think> prose </think> ws <tool_call> ... end
    mask = exp0_sft.second_close_mask(reply, CLOSE, {CALL, END})
    assert mask.tolist() == [1, 1, 1, 1, 1, 0, 0, 1, 1, 1]
    assert exp0_sft.second_close_mask([5, CLOSE, CALL, END], CLOSE, {CALL, END}) is None
    assert exp0_sft.second_close_mask([CLOSE, 10, 11, 10, CLOSE], CLOSE, {CALL, END}).tolist() == [1, 1, 1, 1, 0]
    chain = {'segs': [(3, 4, {'sample': np.array([5, CLOSE, CALL, END])}), (9, 10, {'sample': np.array(reply)})]}
    bad = grpo.malformed_close_targets(chain, 0.8, CLOSE, {CALL, END})
    assert bad.tolist() == [False] * 4 + [False] * 5 + [True, True] + [False] * 3
    assert not grpo.malformed_close_targets(chain, -0.8, CLOSE, {CALL, END}).any()   # a failure still pushes it down
    per_token = torch.tensor([0.5] * 4 + [-1.] * 10)                                  # reasoning RL rows carry their own
    assert not grpo.malformed_close_targets(chain, per_token, CLOSE, {CALL, END}).any()
    assert grpo.malformed_close_targets({'segs': chain['segs'][:1]}, 0.8, CLOSE, {CALL, END}) is None


def test_mask_second_close_is_a_load_time_context_mask():
    import exp0_sft

    class Tok:
        def convert_tokens_to_ids(self, token):
            return {'</think>': 124902}[token]

        def get_vocab(self):
            return {'</think>': 124902, '<tool_call>': 125019, '<|im_end|>': 124900}
    row = {'sample': np.array([5, 124902, 7, 124902, 125019, 124900])}
    chains = [{'segs': [(2, 6, row)]}]
    assert exp0_sft.mask_second_close(chains, Tok()) == 1 and row['weight'].tolist() == [1, 1, 1, 0, 1, 1]
    assert exp0_sft.mask_second_close(chains, Tok()) == 0                       # idempotent


def test_decision_token_kl_anchors_the_chosen_options_shared_tokens():
    import torch
    token = torch.tensor([-0.1, -0.2, -11.0], requires_grad=True)     # </think> "book" <|im_end|> of the chosen option
    ref = torch.tensor([-0.1, -0.3, -0.05])
    grpo.decision_token_kl(token, ref, 0.05, 'k2', 0.5).backward()
    grad = token.grad.tolist()
    assert grad[0] == pytest.approx(0.) and grad[2] == pytest.approx(-0.05 * 10.95 * 0.5)   # descent raises <|im_end|>
    assert grad[1] == pytest.approx(0.05 * 0.1 * 0.5)
