"""Pop pre-collects the next round during the learner (rl.precollect): eligibility, sealing, drain, import, staleness."""
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.jsonio import read_jsonl
from natlang_neuralese.rl import precollect, production
from natlang_neuralese.rl.actors import actor_declaration
from natlang_neuralese.rl.collection_generations import Generations
from test_collection_generations import Harness, POLICY

NEW = 'b' * 64


def config(**extra):
    return {'remote_endpoints': ['http://pop'], 'actor_command': ['x'], 'update_objective': 'policy', **extra}


def state(**extra):
    return {'round': 3, 'trained_rounds': 3, 'interleave': {'rl_rounds_at_last_window': 2, 'sft_windows': 1}, **extra}


LOSS = {'schema': 'jolt-plus/1', 'outcome': {'importance': 'tis'}}


def test_eligibility_bounds_staleness_to_one_update():
    window = {'parameters': {'rl_rounds_per_window': 2}}
    ok, _ = precollect.eligible(config(), state(), None, LOSS)
    assert ok
    ok, reason = precollect.eligible(config(), state(), window, LOSS)   # round 4 - 2 >= 2: a window runs first
    assert not ok and 'SFT window' in reason
    assert precollect.eligible(config(), state(interleave={'rl_rounds_at_last_window': 3, 'sft_windows': 1}), window, LOSS)[0]
    assert not precollect.eligible(config(precollect=False), state(), None, LOSS)[0]
    assert not precollect.eligible({**config(), 'remote_endpoints': []}, state(), None, LOSS)[0]
    assert not precollect.eligible(config(), state(), None, {'schema': 'jolt-plus/1', 'outcome': {'importance': 'none'}})[0]
    assert not precollect.eligible(config(memetic_round=1), state(), None, LOSS)[0]


def test_next_selection_is_sealed_as_the_next_iteration_would_choose_it(tmp_path):
    cases = [{'id': f'c{i}', 'split': 'train'} for i in range(6)]
    cases_path = tmp_path / 'cases.jsonl'
    cases_path.write_text(''.join(json.dumps(c) + '\n' for c in cases))
    cfg = {'rows_per_round': 2, 'group': 4, 'cases': str(cases_path)}
    this_round = [{'row': 1, 'samples': 4}, {'row': 2, 'samples': 4}]
    number, chosen, selected = precollect.seal_next_selection(cfg, state(), cases, this_round, tmp_path, production.select_tasks)
    expected = production.select_tasks(cases, [], {'rows': 2, 'group': 4, 'offset': 3})
    assert number == 4 and chosen == expected and [c['id'] for c in selected] == ['c3', 'c4']
    sealed = json.loads((tmp_path / 'round-000004-selection.json').read_text())
    assert sealed['selection'] == chosen and sealed['budget']['offset'] == 3
    # a restart finds the sealed file and never reselects
    again = precollect.seal_next_selection(cfg, state(), cases, this_round, tmp_path,
                                           lambda *a, **k: pytest.fail('selection is already sealed'))
    assert again[1] == chosen


def test_a_pre_collected_state_block_of_the_previous_policy_joins_the_new_policys_bundle(tmp_path):
    """The learner reads one bundle: blocks whose native identity is policy N sit in a bundle owned by N+1."""
    from natlang_neuralese.model.shared_state import SharedStateModel
    from natlang_neuralese.model.state_block import NativeStateBlock
    from natlang_neuralese.rl.graph_replay import NativeBundleStore, save_bundle, verified_bundle_bytes
    from natlang_neuralese.serve.store import encode_native_state, native_state_id
    from test_shared_state import tiny
    model = SharedStateModel(tiny('lfm'), marker_ids=(58, 59, 60, 61), rank=4)

    def block(checkpoint, token):
        identity = {'checkpoint_sha256': checkpoint, 'tokenizer_fingerprint': 'f' * 64,
                    'embedding_weight_sha256': '2' * 64, 'architecture': 'lfm2'}
        layers = []
        for index, layer in enumerate(model.port.layers):
            if model.port.is_attention(index):
                attention = layer.self_attn
                heads = attention.k_proj.out_features // attention.head_dim
                layers.append({'kind': 'attention', 'raw_k': torch.randn(2, heads, attention.head_dim),
                               'v': torch.randn(2, heads, attention.head_dim)})
            else:
                layers.append({'kind': 'finite_convolution', 'gated_inputs': torch.randn(2, layer.conv.conv.weight.shape[0])})
        return NativeStateBlock(text='s', token_ids=torch.tensor([token, token + 1]), positions=torch.tensor([0, 1]),
                                embeddings=torch.randn(2, model.port.embedding_weight.shape[1]),
                                layers=tuple(layers), **identity).validate()
    old, new = block('1' * 64, 9), block(NEW, 20)
    blobs = {native_state_id(b): encode_native_state(b) for b in (old, new)}
    rollouts = [{'ids': list(blobs)}]
    manifest = save_bundle(rollouts, tmp_path / 'blocks', policy_id=NEW, fetch=lambda ident: blobs[ident])
    assert set(manifest['blocks']) == set(blobs) and manifest['policy_id'] == NEW
    store = NativeBundleStore(tmp_path / 'blocks/manifest.json', policy_id=NEW)
    for ident in blobs:
        store.state(ident)   # derivation and identity receipts verify for the previous policy's block too
        assert verified_bundle_bytes(tmp_path / 'blocks/manifest.json', ident, policy_id=NEW) == blobs[ident]


def make_pre(tmp_path, *, finished=('s7-task-000-00', 's7-task-000-01'), settled=True, start=None):
    pre = precollect.pre_directory(tmp_path / 'round-000004')
    (pre / 'workers').mkdir(parents=True)
    queue = []
    for key in ('s7-task-000-00', 's7-task-000-01', 's7-task-001-00'):
        group, sample = int(key.split('-')[2]), int(key.split('-')[3])
        case = pre / f'row-{group:03d}' / f'sample-{sample:02d}'
        case.mkdir(parents=True, exist_ok=True)
        (case / 'results.jsonl').write_text('{}\n')
        queue.append({'key': key, 'group': group, 'sample': sample, 'output': str(case / 'results.jsonl'),
                      'jobs': str(case / 'jobs'), 'log': str(case / 'collector.log')})
    (pre / 'queue.jsonl').write_text(''.join(json.dumps(e) + '\n' for e in queue))
    rows = [row for key in finished for row in ({'event': 'start', 'key': key}, {'event': 'finish', 'key': key})]
    (pre / 'workers/journal-g1-0.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))
    (pre / 'workers/queue-g1-0.jsonl').write_text(''.join(json.dumps({'key': k, 'rl_endpoint': 'http://pop'}) + '\n' for k in finished))
    export = {'kind': 'exact', 'format': 'native-pop-rtx4060-bf16', 'sha256': 'c' * 64}
    (pre / 'score.json').write_text(json.dumps({'collection_complete': False, 'actor_receipts': {'http://pop': {'actor_export': export}}}))
    (pre / 'blocks').mkdir()
    (pre / 'blocks/manifest.json').write_text(json.dumps({'schema': 'natlang.rl-block-bundle/1', 'policy_id': POLICY, 'blocks': {}}))
    (pre / precollect.START).write_text(json.dumps(start or {'round': 4, 'behavior_policy_id': POLICY, 'trained_rounds': 3,
                                                             'sft_windows': 1, 'update_round': 3, 'kwargs': {}}))
    if settled:
        (pre / precollect.RESULT).write_text(json.dumps({'settled': True}))
    return pre, export


def test_import_moves_only_finished_episodes_and_keeps_their_behaviour_policy(tmp_path):
    pre, export = make_pre(tmp_path)
    main = tmp_path / 'round-000004'
    after = {'round': 4, 'trained_rounds': 4, 'interleave': {'sft_windows': 1}}
    receipt = precollect.import_into_round(pre, main, after, learner_policy_id=NEW)
    assert receipt['keys'] == ['s7-task-000-00', 's7-task-000-01'] and receipt['behavior_policy_id'] == POLICY
    assert receipt['staleness_updates'] == 1 and receipt['actor_export'] == export
    assert (main / 'row-000/sample-00/results.jsonl').exists() and not (pre / 'row-000/sample-00').exists()
    assert (pre / 'row-001/sample-00/results.jsonl').exists()            # unfinished: left for a fresh collection
    assert (main / 'pre-blocks/manifest.json').exists() and receipt['blocks_manifest'] == 'pre-blocks/manifest.json'
    generations = Generations(main)
    assert generations.attempted() == {'s7-task-000-00', 's7-task-000-01'}          # never dispatched again
    assert generations.finished_endpoints() == {'s7-task-000-00': 'http://pop', 's7-task-000-01': 'http://pop'}
    assert precollect.import_into_round(pre, main, after, learner_policy_id=NEW) == receipt   # idempotent
    declaration = actor_declaration(precollect.actors_declaration(receipt, NEW),
                                    learner_identity={'checkpoint_sha256': NEW})
    assert declaration['actors'][POLICY]['export'] == export and declaration['actors'][POLICY]['lineage_step'] == 3
    rollouts = tmp_path / 'rollouts.jsonl'
    rollouts.write_text(''.join(json.dumps(r) + '\n' for r in [
        {'group': 0, 'sample': 0, 'policy': {'checkpoint_sha256': NEW, 'endpoint': 'http://pop'}},
        {'group': 0, 'sample': 2, 'policy': {'checkpoint_sha256': NEW}}]))
    assert precollect.stamp_rollouts(rollouts, receipt) == 1
    stamped, fresh = read_jsonl(rollouts)
    assert stamped['policy']['checkpoint_sha256'] == POLICY and stamped['policy']['staleness_updates'] == 1
    assert fresh['policy']['checkpoint_sha256'] == NEW


@pytest.mark.parametrize('after,reason', [
    ({'round': 5, 'trained_rounds': 5, 'interleave': {'sft_windows': 1}}, 'more than one update'),
    ({'round': 4, 'trained_rounds': 4, 'interleave': {'sft_windows': 2}}, 'more than one update'),
    ({'round': 4, 'trained_rounds': 3, 'interleave': {'sft_windows': 1}}, 'more than one update')])
def test_an_older_behaviour_policy_is_not_admitted_and_the_reason_is_recorded(tmp_path, after, reason):
    pre, _ = make_pre(tmp_path)
    result = precollect.import_into_round(pre, tmp_path / 'round-000004', after, learner_policy_id=NEW)
    assert reason in result['rejected']
    assert (pre / 'pre-collection-rejected.json').exists() and (pre / 'row-000/sample-00').exists()
    assert not (tmp_path / 'round-000004' / precollect.RECEIPT).exists()


def test_an_unsettled_pre_collection_is_not_imported(tmp_path):
    pre, _ = make_pre(tmp_path, settled=False)
    after = {'round': 4, 'trained_rounds': 4, 'interleave': {'sft_windows': 1}}
    assert 'did not settle' in precollect.import_into_round(pre, tmp_path / 'round-000004', after, learner_policy_id=NEW)['rejected']


def test_drain_takes_no_new_entry_lets_in_flight_ones_finish_and_the_round_imports_them(tmp_path, monkeypatch):
    import threading
    drain = threading.Event()

    class Draining(Harness):
        def collector(self, command, log, control, custody_receipt=None, detach_path=None):
            code = super().collector(command, log, control, custody_receipt, detach_path)
            if len(self.ran) >= 2:
                drain.set()   # the learner finished: the entry just completed was the last one taken
            return code
    (tmp_path / 'a').mkdir()
    pre = Draining(tmp_path / 'a', monkeypatch, entries=6)
    pre.budget = 99
    report = pre.collect(workers=1, drain=drain, slot_poll_seconds=0.02)
    assert report['collection_complete'] is False and report['collection_keys']['finished'] == 2
    assert len(Generations(pre.out).attempted()) == 2 and (pre.out / 'score.json').exists()
    # the round imports the two finished episodes; its own collection runs exactly the other four
    (pre.out / precollect.START).write_text(json.dumps({'round': 4, 'behavior_policy_id': POLICY, 'trained_rounds': 3,
                                                        'sft_windows': 1, 'update_round': 3, 'kwargs': {}}))
    (pre.out / precollect.RESULT).write_text(json.dumps({'settled': True}))
    (tmp_path / 'b').mkdir()
    main = Harness(tmp_path / 'b', monkeypatch, entries=6)
    main.budget = 99
    after = {'round': 4, 'trained_rounds': 4, 'interleave': {'sft_windows': 1}}
    receipt = precollect.import_into_round(pre.out, main.out, after, learner_policy_id=NEW)
    assert len(receipt['keys']) == 2
    report = main.collect(slot_poll_seconds=0.02)
    assert sorted(key for key, *_ in main.ran) == sorted(set(f's7-task-{i:03d}' for i in range(6)) - set(receipt['keys']))
    assert report['collection_complete'] is True and report['collection_keys']['finished'] == 6
    assert report['pre_collected'] == {'episodes': 0, 'behavior_policy_id': POLICY, 'staleness_updates': 1}


def _controller(tmp_path, monkeypatch, *, begin):
    """production.run for one JOLT+ round with fakes: collect, remote hooks, learner command and ``precollect.begin``."""
    from natlang_neuralese.rl import native_raw_policy
    from natlang_neuralese.rl.policy_math import default_policy_loss
    monkeypatch.setattr(native_raw_policy, 'prepare_raw_policy', lambda *a, **kw: {'tokenizer': 'bound-parent'})
    source = tmp_path / 'cases.jsonl'
    source.write_text(''.join(json.dumps({'id': f't{i}', 'split': 'train'}) + '\n' for i in range(6)))
    checkpoint = tmp_path / 'parent.pt'
    checkpoint.write_bytes(b'full state')
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({
        'out': str(tmp_path / 'production'), 'checkpoint': str(checkpoint), 'cases': str(source),
        'endpoints': ['http://managed'], 'remote_endpoints': ['http://pop'], 'remote_workers': {'http://pop': {'workers': 2, 'weight': 1}},
        'actor_command': ['serve'], 'after_collection': ['stop-pop'], 'rows_per_round': 2, 'group': 2,
        'learner_command': ['learn', '--rl-policy-loss', '{rl_policy_loss}', '--jolt-cases', '{cases}'], 'trainer_parameters': {'learning_rate': 1e-6},
        'after_update': ['publish'], 'native_state': True, 'base': 'unused', 'update_objective': 'policy',
        'policy_loss': default_policy_loss(), 'jolt_teacher_attempts': 0}))
    seen = {'kept': [], 'commands': [], 'begun': []}

    def collect_with_actor(config, values, keep_remote_actor=False, **kwargs):
        seen['kept'].append(keep_remote_actor)
        kwargs['out'].mkdir(parents=True, exist_ok=True)
        (kwargs['out'] / 'rollouts.jsonl').write_text(json.dumps({'task_id': 't0', 'split': 'train', 'reward': {'quality': 1}, 'turns': [{}]}) + '\n')
        return {'groups_with_spread': 1, 'rollouts': 1, 'scored': 1, 'successes': 1, 'groups': [], 'collection_complete': True}
    monkeypatch.setattr(production, 'collect_with_actor', collect_with_actor)

    class Stop(Exception):
        pass

    def command(argv, values):
        seen['commands'].append(list(argv))
        if argv[0] == 'learn':
            raise Stop
    monkeypatch.setattr(production, 'run_command', command)
    monkeypatch.setattr(precollect, 'begin', begin(seen))
    return config, seen, Stop


def test_the_controller_starts_pre_collection_at_the_learner_and_drains_it_when_the_learner_ends(tmp_path, monkeypatch):
    class Handle:
        directory = tmp_path / 'pre'
        def finish(self_):
            seen['events'].append('drained')
            return None

    def begin(seen):
        def start(**kwargs):
            seen['begun'].append((kwargs['number'], [item['row'] for item in kwargs['selection']]))
            seen['events'] = seen.get('events', []) + ['begun']
            return Handle()
        return start
    config, seen, Stop = _controller(tmp_path, monkeypatch, begin=begin)
    seen['events'] = []
    with pytest.raises(Stop):
        production.run(config, rounds=1)
    assert seen['kept'] == [True]                                    # Pop keeps serving policy N past the collection
    assert seen['begun'] and seen['begun'][0][0] == 1                 # round N+1's selection is sealed and started
    assert seen['events'] == ['begun', 'drained']                     # drained when the learner ended (here: failed)
    assert ['stop-pop'] not in seen['commands']                       # the hook is not run while Pop pre-collects
    assert (tmp_path / 'production/round-000001-selection.json').exists()


def test_the_kept_remote_actor_is_stopped_when_no_pre_collection_starts(tmp_path, monkeypatch):
    config, seen, Stop = _controller(tmp_path, monkeypatch, begin=lambda seen: (lambda **kw: None))
    with pytest.raises(Stop):
        production.run(config, rounds=1)
    assert seen['kept'] == [True] and seen['commands'][-1] == ['stop-pop']
