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
    sync.buffer, sync.front_url = None, 'http://127.0.0.1:9'    # the served weights' digest fails
    sync.publish(4, lambda version, ok: done.append((version, ok)))
    sync.wait()
    assert done == [(4, False)] and sync.last['version'] == 4 and not sync.last['ok']
    assert 'publish failed' in sync.last['answer']


def _export(tmp_path, model, extra):
    torch = pytest.importorskip('torch')
    from safetensors.torch import save_file
    export = tmp_path / 'export'
    export.mkdir()
    tensors = {name: torch.zeros_like(p) for name, p in model.named_parameters()}
    tensors.update(extra)
    save_file(tensors, str(export / 'model.safetensors'), metadata={'format': 'pt'})
    (export / 'config.json').write_text('{}')
    return export


def _tiny_model(torch):
    model = torch.nn.Module()
    model.model = torch.nn.Module()
    model.model.embed_tokens = torch.nn.Embedding(6, 4).to(torch.bfloat16)
    model.layer = torch.nn.Linear(4, 3, bias=False).to(torch.bfloat16)
    return model


def test_weight_sync_serves_the_snapshot_from_the_mapped_export_file(tmp_path):
    """One host copy: the snapshot lands in the actor's export file itself (no pinned buffer written out per publish),
    the producer table keeps its exported values, and the native checkpoint's views leave the served weights alone."""
    torch = pytest.importorskip('torch')
    from safetensors import safe_open
    model = _tiny_model(torch)
    producer = torch.arange(10, dtype=torch.float32).reshape(2, 5).to(torch.bfloat16)
    export = _export(tmp_path, model, {'producer.weight': producer})
    sync = grpo.WeightSync(model, export, None, tmp_path / 'weights')
    assert (tmp_path / 'weights' / 'config.json').exists()
    embed, layer = 'model.embed_tokens.weight', 'layer.weight'
    live = dict(model.named_parameters())
    with torch.no_grad():
        live[layer].add_(1)
    sync.snapshot()
    with safe_open(str(tmp_path / 'weights' / 'model.safetensors'), 'pt') as handle:
        assert torch.equal(handle.get_tensor(layer), live[layer].detach())
        assert torch.equal(handle.get_tensor(embed), live[embed].detach())
        assert torch.equal(handle.get_tensor('producer.weight'), producer)
    backbone = sync.backbone()
    assert backbone[layer].untyped_storage().nbytes() == live[layer].numel() * 2      # its own storage, as a clone's
    served = backbone[embed].clone()
    path = tmp_path / 'checkpoint.pt'
    native_rows = {'embedding_rows': torch.full((1, 4), 7.).to(torch.bfloat16), 'tied_head': True, 'names': [embed, layer]}
    state = {'schema': 's', 'model_config': {}, 'interface_state': {}, 'step': 1}
    grpo.write_pruned_checkpoint(state, native_rows, [2], path, 'sha', backbone)
    assert torch.equal(sync.backbone()[embed], served)            # the served weights untouched
    saved = torch.load(path, weights_only=False)['backbone_state']
    assert torch.equal(saved[embed][2], native_rows['embedding_rows'][0]) and torch.equal(saved[layer], live[layer].detach())
    assert saved['lm_head.weight'] is saved[embed]
    done = []
    sync.publish(1, lambda version, ok: done.append((version, ok)))
    sync.wait()
    assert done == [(1, True)]
    sync.close()     # a registration left behind makes a later test's cudaHostRegister of the reused range fail


def test_weight_sync_publishes_the_producer_table_from_the_trained_embedding(tmp_path):
    """The producer table is the embedding (no producer adapter): kept at the export's start values it served stale rows
    to every Neuralese text write after step 1. It follows the embedding, stays out of the native checkpoint's backbone,
    and ``producer_from_embedding=False`` keeps the exported table."""
    torch = pytest.importorskip('torch')
    from safetensors import safe_open
    model = _tiny_model(torch)
    start = torch.full((6, 4), 3.).to(torch.bfloat16)
    export = _export(tmp_path, model, {grpo.PRODUCER_TABLE: start})
    live = dict(model.named_parameters())
    with torch.no_grad():
        live[grpo.EMBEDDING].add_(1)
    for follow, directory in ((True, 'follow'), (False, 'frozen')):
        sync = grpo.WeightSync(model, export, None, tmp_path / directory, producer_from_embedding=follow)
        sync.snapshot()
        with safe_open(str(tmp_path / directory / 'model.safetensors'), 'pt') as handle:
            table = handle.get_tensor(grpo.PRODUCER_TABLE)
            assert torch.equal(handle.get_tensor(grpo.EMBEDDING), live[grpo.EMBEDDING].detach())
        assert torch.equal(table, live[grpo.EMBEDDING].detach() if follow else start)
        assert grpo.PRODUCER_TABLE not in sync.backbone() and grpo.EMBEDDING in sync.backbone()
        sync.close()
    assert grpo.producer_adapter_keys({'input_rows': 0, 'adapter.up.weight': 0}) == ['adapter.up.weight']


def test_reference_weights_load_into_the_policy_and_restore_it_exactly(tmp_path):
    torch = pytest.importorskip('torch')
    model = _tiny_model(torch)
    live = dict(model.named_parameters())
    start = {name: p.detach().clone() for name, p in live.items()}
    path = tmp_path / 'reference.safetensors'
    reference = grpo.ReferenceWeights(path, live, {'source_sha256': 'a'})
    written = path.stat().st_mtime_ns
    grpo.ReferenceWeights(path, live, {'source_sha256': 'a'})          # the same start policy: reused as it is
    assert path.stat().st_mtime_ns == written
    masters = {name: (p.float() + torch.randn(p.shape) * 0.3) for name, p in live.items()}   # the policy moved on
    with torch.no_grad():
        for name, p in live.items():
            p.copy_(masters[name])
    policy = {name: p.detach().clone() for name, p in live.items()}
    reference.load_into(live)
    assert all(torch.equal(live[name], start[name]) for name in live)
    grpo.ReferenceWeights.restore(live, masters)
    assert all(torch.equal(live[name], policy[name]) for name in live)
    assert dict(reference.tensors()).keys() == live.keys()
    grpo.ReferenceWeights(path, live, {'source_sha256': 'b'})          # another start policy: rewritten
    assert torch.equal(dict(reference.tensors())['layer.weight'], policy['layer.weight'])


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
    import lfm3_fast_train
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


def test_reference_tensors_hold_no_mapping_of_the_file(tmp_path):
    """Each reference tensor is its own buffer: a kept tensor must not keep a mapping of the whole file (a private
    mapping the GPU copy made anonymous held 3.4 GB in grpo-v3-run1's learner between steps)."""
    torch = pytest.importorskip('torch')
    live = dict(_tiny_model(torch).named_parameters())
    path = tmp_path / 'reference.safetensors'
    reference = grpo.ReferenceWeights(path, live, {'source_sha256': 'a'})
    kept = list(reference.tensors())
    assert [name for name, _ in kept] == list(live)
    for name, tensor in kept:
        assert tensor.dtype == torch.bfloat16 and torch.equal(tensor, live[name].detach())
        assert tensor.untyped_storage().nbytes() == tensor.numel() * 2      # not a view of a whole-file storage
    assert str(path) not in Path('/proc/self/maps').read_text()
    path.write_bytes(path.read_bytes()[:-2])                                # a truncated file is refused
    with pytest.raises(ValueError, match='ends inside tensor'):
        list(reference.tensors())


def test_moment_offload_round_trips_bit_exactly_and_frees_the_moments(tmp_path):
    """MomentOffload: after ``put`` the moments hold no storage; ``get`` restores the same tensor objects bit for bit, after
    an in-flight saver that reads them has finished."""
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('MomentOffload stages through pinned memory and a CUDA stream')
    generator = torch.Generator(device='cuda').manual_seed(0)
    tensors = [torch.randn(shape, device='cuda', generator=generator) for shape in ((300, 7), (5,), (1 << 20,))]
    tensors[1][0] = float('nan')                        # bit-exact means bit-exact
    expected = [t.clone() for t in tensors]
    ids = [id(t) for t in tensors]
    order = []

    class Saver:                                         # a background state save that reads the moments
        def wait(self):
            order.append(('saver', all(t.untyped_storage().nbytes() > 0 for t in tensors)))

    offload = grpo.MomentOffload(tmp_path / 'moments.offload', staging_mb=1)   # several staging chunks per tensor
    offload.put(tensors, after=Saver())
    offload.wait()
    assert order == [('saver', True)]
    assert all(t.untyped_storage().nbytes() == 0 for t in tensors)
    assert (tmp_path / 'moments.offload').stat().st_size == sum(t.numel() * 4 for t in expected)
    assert offload.get() >= 0 and offload.get() < 1 and not offload.out   # a second get is a no-op
    assert [id(t) for t in tensors] == ids
    for t, e in zip(tensors, expected):
        assert torch.equal(t.view(torch.int32), e.view(torch.int32))


def test_a_failed_moment_offload_write_keeps_the_moments_resident(tmp_path):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('MomentOffload stages through pinned memory and a CUDA stream')
    (tmp_path / 'file').write_text('')
    tensors = [torch.arange(10., device='cuda')]
    offload = grpo.MomentOffload(tmp_path / 'file' / 'moments.offload', staging_mb=1)   # a parent that is a file
    offload.put(tensors)
    assert offload.get() < 1 and not offload.out and "put_error" in offload.last
    assert torch.equal(tensors[0], torch.arange(10., device='cuda'))


def test_state_saver_writes_a_safetensors_file_that_loads_back_exactly(tmp_path, monkeypatch):
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available():
        pytest.skip('StateSaver stages through pinned memory and a CUDA stream')
    from safetensors.torch import load_file
    monkeypatch.setattr(grpo, 'CACHE_DROP_BYTES', 1 << 20)   # several flush-and-drop rounds
    tensors = {'master/a': torch.randn(700, 1000, device='cuda'), 'exp_avg/b': torch.randn(3, device='cuda')}
    saver = grpo.StateSaver(staging_mb=1)
    saver.save(tmp_path / 'state.safetensors', tensors, {'step': '3'})
    saver.wait()
    loaded = load_file(str(tmp_path / 'state.safetensors'))
    assert set(loaded) == set(tensors) and all(torch.equal(loaded[k], tensors[k].cpu()) for k in tensors)
