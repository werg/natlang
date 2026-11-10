"""The policy's own verified successful RL episodes queue as distillation packages for the next SFT window."""
import copy
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.rl import interleave, memetic_admission as admission, rollout_distillation as distill
from natlang_neuralese.rl.native_replay import prepare_objective
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_logical_families import load_map
from test_memetic_admission import FP, Guard, UNCAPPED, graph, lineage, window_harness
from test_native_rl_replay import actor_rollout
from test_rl_episode_graph import attach_execution_graph


def population():
    _, roles, identity, _, block, turns = actor_rollout('lfm')
    raw = [{'messages': [], 'target': {}, 'invocation_id': turn['graph_turn'], 'sampling': turn['sampling'],
            'replay': {'raw_calls_retained': True}} for turn in turns]
    policy = identity['checkpoint_sha256']

    def row(task, sample, quality=1.0, **fields):
        turns = copy.deepcopy(raw)
        for index, turn in enumerate(turns):  # each sample's sampled outputs are its own
            turn['target'] = {'content': f'{task}:{sample}:{index}'}
        value = {'task_id': task, 'sample': sample, 'split': 'train', 'group': task,
                 'reward': {'quality': quality},
                 'policy': {'checkpoint_sha256': policy}, 'turns': turns}
        value.update(fields)
        return value

    ordinary_turns = [copy.deepcopy(raw[0])]  # a consumer alone is not a closed graph (old external state)
    rows = [row('a', 0), row('a', 1, reward={'quality': 1.0, 'neuralese_bonus': .1}),
            row('b', 0), row('c', 0, quality=0.0), row('d', 0, jolt_role='teacher'), row('e', 0, split='test'),
            row('f', 0, policy={'checkpoint_sha256': '9' * 64}),
            row('g', 0, reward={'quality': 1.0, 'memetic': {'gain': .5}}),
            row('h', 0, turns=ordinary_turns)]
    attach_execution_graph(rows)
    return rows, identity, roles, block


def lower(identity, roles, block):
    def prepare(selected, binding, checkpoint, bundle):
        clean = [{k: v for k, v in item.items() if k != '_neuralese_exchanges'} for item in selected]
        return prepare_objective(clean, identity=identity, roles=roles, fetch_block=lambda _: block, mode='rft'), []
    return prepare


def test_selection_is_every_verified_student_success_of_the_current_policy_neuralese_first():
    rows, identity, _, _ = population()
    selected, skipped = distill.select(rows, identity['checkpoint_sha256'])
    # Expert iteration: both successes of task a, Neuralese-using first and by return; h is plain.
    assert [(row['task_id'], row['sample']) for row in selected] == [('a', 1), ('a', 0), ('b', 0), ('h', 0)]
    assert not {'c', 'd', 'e', 'f', 'g'} & {row['task_id'] for row in selected}
    assert skipped == {'teacher episode': 1, 'not a train task': 1, 'sampled by another policy': 1,
                       'memetic improvement episode': 1}
    assert [row['_neuralese_exchanges'] for row in selected] == [1, 1, 1, 0]
    # A Neuralese-using success outranks a plain one of the same task even with a smaller return.
    plain = copy.deepcopy(rows[1])
    plain['sample'] = 7
    plain['reward'] = {'quality': 1.0, 'neuralese_bonus': 0.}
    plain['turns'] = [{**turn, 'target': {'content': 'plain'}} for turn in rows[8]['turns']]
    plain['execution_graph'] = rows[8]['execution_graph']
    chosen, _ = distill.select([plain, rows[0]], identity['checkpoint_sha256'])
    assert [(row['task_id'], row['sample']) for row in chosen] == [('a', 0), ('a', 7)]


def test_exact_duplicates_and_already_queued_episodes_are_never_repeated_and_a_task_can_be_bounded():
    rows, identity, _, _ = population()
    policy = identity['checkpoint_sha256']
    twin = copy.deepcopy(rows[2])  # task b, another sample with the same sampled outputs
    twin['sample'] = 3
    selected, skipped = distill.select([*rows, twin], policy)
    assert [(row['task_id'], row['sample']) for row in selected if row['task_id'] == 'b'] == [('b', 0)]
    assert skipped['duplicate sampled outputs'] == 1
    chosen, skipped = distill.select(rows, policy, exclude={('a', 1)})
    assert [(row['task_id'], row['sample']) for row in chosen][:2] == [('a', 0), ('b', 0)]
    assert skipped['already queued'] == 1
    bounded, skipped = distill.select(rows, policy, max_per_task=1)
    assert [(row['task_id'], row['sample']) for row in bounded] == [('a', 1), ('b', 0), ('h', 0)]
    assert skipped['over the per-task bound'] == 1
    # Several policies may be allowed (an earlier round of the lineage); each row keeps its own sampling policy.
    older = {**copy.deepcopy(rows[2]), 'task_id': 'z', 'policy': {'checkpoint_sha256': '8' * 64}}
    pair, _ = distill.select([older, rows[2]], {policy, '8' * 64})
    assert {row['task_id'] for row in pair} == {'b', 'z'}


def test_round_package_holds_ranked_closed_whole_graphs_with_actor_provenance(tmp_path):
    rows, identity, roles, block = population()
    config = {'out': str(tmp_path / 'controller')}
    state = {'round': 4, 'checkpoint': 'unused'}
    round_dir = tmp_path / 'controller' / 'round-000004'
    round_dir.mkdir(parents=True)
    geometry = lambda tokenizer: (identity['tokenizer_fingerprint'], [roles.produce_open, roles.produce_close,
                                                                     roles.consume_open, roles.consume_close])
    record = distill.package_round(config, state, round_dir, rows, policies=identity['checkpoint_sha256'],
                                   binding={'rl_tokenizer': 'tok'}, bundle='', protected=Guard(),
                                   prepare=lower(identity, roles, block), geometry=geometry, log=lambda line: None)
    # a's two successes (Neuralese-using, weighted by their return) and b; h imports old external state: skipped.
    assert record['rows'] == 3 and record['selected_episodes'] == 4 and record['selected_tasks'] == 3
    assert record['neuralese_using_rows'] == 3
    assert record['not_closed'] == [{'task_id': 'h', 'reason': 'imports old external actor state'}]
    [item] = [json.loads(line) for line in admission.queue_path(config['out']).read_text().splitlines()]
    assert item['manifests'] == [str(round_dir / distill.RECORD)] and item['family'] == 'rollout-distillation'
    receipt = json.loads(Path(item['package']).read_text())
    assert receipt['family'] == admission.ROLLOUT_FAMILY and receipt['training_admission'] is False
    assert receipt['episode_actors'] == [identity['checkpoint_sha256']] and receipt['round'] == 4
    source = Path(receipt['source']['path'])
    graphs = [json.loads(line) for line in source.read_text().splitlines()]
    assert [(row['rollout_distillation']['task_id'], row['rollout_distillation']['sample']) for row in graphs] == [
        ('a', 1), ('a', 0), ('b', 0)]
    assert {row['rollout_distillation']['actor_checkpoint_sha256'] for row in graphs} == {identity['checkpoint_sha256']}
    assert all(row['id'].startswith('rollout:') and not row['external_states'] and row['split'] == 'train'
               for row in graphs)
    assert graphs[0]['rollout_distillation']['sampled_token_weight'] == pytest.approx(1.1)
    assert [row['rollout_distillation']['sampled_token_weight'] for row in graphs[1:]] == [1., 1.]
    weights = lambda row: sorted({w for node in row['nodes'] for w in node['target_weights']})
    assert weights(graphs[0]) == pytest.approx([0., 1.1]) and weights(graphs[1]) == [0., 1.]
    # Only sampled own tokens carry loss; the indexed source is the exact receipt.
    indexed = NativeGraphSource(source, identity['tokenizer_fingerprint'], required_split='train',
                                role_ids=receipt['role_ids'])
    assert [entry['id'] for entry in indexed.entries] == [row['id'] for row in graphs]
    # Idempotent: the same round is never queued twice.
    again = distill.package_round(config, state, round_dir, rows, policies=identity['checkpoint_sha256'],
                                  binding={'rl_tokenizer': 'tok'}, bundle='', protected=Guard(),
                                  prepare=lower(identity, roles, block), geometry=geometry, log=lambda line: None)
    assert again == record and len(admission.queue_path(config['out']).read_text().splitlines()) == 1


def test_protected_and_unsuccessful_rounds_queue_nothing(tmp_path):
    rows, identity, roles, block = population()
    config = {'out': str(tmp_path)}
    round_dir = tmp_path / 'round'
    round_dir.mkdir()
    none = distill.package_round(config, {'round': 1}, round_dir, [rows[3]], policies=identity['checkpoint_sha256'],
                                 binding={}, bundle='', protected=Guard(), log=lambda line: None)
    assert none['selected_episodes'] == 0 and none['package'] is None and not admission.queue_path(tmp_path).exists()


def test_transition_is_recorded_once_and_a_round_in_flight_keeps_its_behaviour(tmp_path):
    state, path = {'round': 5}, tmp_path / 'state.json'
    config = {'out': 'x'}
    adopted = distill.adopt_defaults(config, state, path, interleaved=True)
    assert adopted['rollout_distillation_from_round'] == 5 and distill.due(adopted, {'round': 5})
    assert distill.adopt_defaults(config, state, path, interleaved=True) == adopted  # recorded once
    assert len(state['rollout_distillation_transitions']) == 1
    sealed = distill.adopt_defaults(config, {'round': 8}, tmp_path / 'other.json', interleaved=True, sealed_selection=True)
    assert sealed['rollout_distillation_from_round'] == 9 and not distill.due(sealed, {'round': 8})
    assert distill.adopt_defaults({**config, 'rollout_distillation': False}, {'round': 1}, path, interleaved=True) \
        == {**config, 'rollout_distillation': False}
    assert distill.adopt_defaults(config, {'round': 1}, path, interleaved=False) == config
    assert not distill.due({**adopted, 'rollout_distillation': False}, {'round': 9})


def test_hook_failure_is_recorded_not_raised(tmp_path):
    logged = []
    config = {'out': str(tmp_path), 'rollout_distillation_from_round': 0}
    state = {'round': 1, 'checkpoint': str(tmp_path / 'missing.pt')}
    assert distill.after_scoring(config, state, tmp_path, {}, [], 'x' * 64, log=logged.append) is None
    assert 'rollout_distillation_failed' in logged[0]
    assert distill.after_scoring({'out': str(tmp_path)}, state, tmp_path, {}, [], 'x' * 64, log=logged.append) is None
    assert len(logged) == 1


def test_window_boundary_admits_rollout_rows_next_to_memetic_rows_with_their_own_family(tmp_path, monkeypatch):
    config, declared, state, out, train = lineage(tmp_path, **UNCAPPED)
    actor = resident.sha(config['checkpoint'])
    memetic = out / 'memetic'
    admission.queue_package(memetic, [graph('rollout:a', 6), graph('rollout:b', 8)], family=admission.ROLLOUT_FAMILY,
                            tokenizer=FP, roles=[], protected=Guard(), manifests=[tmp_path / 'record.json'],
                            actors=[actor], scope='test', log=lambda line: None)
    admission.queue_package(memetic, [graph('memetic:a', 9)], family=admission.FAMILY, tokenizer=FP, roles=[],
                            protected=Guard(), manifests=[tmp_path / 'm.json'], actors=[actor], scope='test',
                            log=lambda line: None)
    seen = []
    window_harness(monkeypatch, seen)
    state['round'] = 2
    assert interleave.window_due(declared, state)
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
    recipe = load_recipe(seen[0][0])
    inputs = resolve_stage_inputs(recipe, seen[0][1], {})
    successor = NativeGraphSource(inputs['native_graphs']['path'], FP, required_split='train')
    assert [e['id'] for e in successor.entries] == ['old', 'rollout:a', 'rollout:b', 'memetic:a']
    balance = json.loads(Path(inputs['ce_supervision_policy']['path']).read_text())['objective']['family_balance']
    assert balance['scales'] == {'ordinary': 1., 'rollout-distillation': .2, 'memetic-guidance': .2}
    assert load_map(balance['map'], successor) == ['ordinary', 'rollout-distillation', 'rollout-distillation',
                                                   'memetic-guidance']
    ledger = state['interleave']['memetic_distillation']
    assert {value['disposition'] for value in ledger['rows'].values()} == {'admitted'}
    # A completed stage reopens for new packages only at the declared window cadence.
    state['interleave']['sft_complete'] = True
    admission.queue_package(memetic, [graph('rollout:c', 10)], family=admission.ROLLOUT_FAMILY, tokenizer=FP, roles=[],
                            protected=Guard(), manifests=[tmp_path / 'r2.json'], actors=[actor], scope='test',
                            log=lambda line: None)
    state['interleave']['rl_rounds_at_last_window'] = state['round']
    assert not interleave.window_due(declared, state)
    state['round'] += 1
    assert interleave.window_due(declared, state)


def test_earlier_rounds_are_topped_up_once_with_each_rows_own_sampling_policy(tmp_path, monkeypatch):
    rows, identity, roles, block = population()
    policy = identity['checkpoint_sha256']
    out = tmp_path / 'controller'
    config = {'out': str(out)}
    kwargs = dict(binding={'rl_tokenizer': 'tok'}, protected=Guard(), prepare=lower(identity, roles, block),
                  geometry=lambda tokenizer: (identity['tokenizer_fingerprint'],
                                              [roles.produce_open, roles.produce_close, roles.consume_open,
                                               roles.consume_close]), log=lambda line: None)
    bundle = tmp_path / 'manifest.json'
    bundle.write_text('{}')
    for number, subset in ((0, rows[:3]), (1, rows[:1]), (2, rows[:3])):
        directory = out / f'round-{number:06d}'
        directory.mkdir(parents=True)
        subset = [{**row, 'round': number} for row in subset]
        (directory / 'rollouts.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in subset))
        (directory / 'score.json').write_text(json.dumps({'collection_complete': True, 'block_bundle': str(bundle)}))
    (out / 'round-000003').mkdir()  # a round without scored rollouts is left alone
    monkeypatch.setattr(admission, 'lineage_checkpoints', lambda config, state: {policy})
    monkeypatch.setattr(admission, 'lineage_checkpoint_files', lambda config, state: {policy: 'checkpoint.pt'})
    # Round 2 was queued by the first form of the hook (one episode per task): that episode is not queued again.
    legacy = {**graph_for_queue('rollout:legacy'), 'tokenizer_fingerprint': identity['tokenizer_fingerprint'],
              'rollout_distillation': {'round': 2, 'task_id': 'a', 'sample': 1}}
    admission.queue_package(out / 'memetic', [legacy], family=admission.ROLLOUT_FAMILY,
                            tokenizer=identity['tokenizer_fingerprint'], roles=[], protected=Guard(),
                            manifests=[out / 'round-000002' / distill.LEGACY_RECORD], actors=[policy], scope='test',
                            extra={'round': 2}, log=lambda line: None)
    state = {'round': 3, 'checkpoint': 'unused'}
    done = distill.backfill(config, state, current_round=3, **kwargs)
    assert [item['round'] for item in done] == [0, 1, 2]
    assert [item['rows'] for item in done] == [3, 1, 2]  # round 2: a(1) was already queued; a(0), b remain
    assert [item['skipped'].get('already queued', 0) for item in done] == [0, 0, 1]
    queue = [json.loads(line) for line in admission.queue_path(out).read_text().splitlines()]
    assert [item['family'] for item in queue] == ['rollout-distillation'] * 4
    # Idempotent: nothing is queued twice, and the round in progress is never backfilled.
    assert distill.backfill(config, state, current_round=3, **kwargs) == []
    assert len(admission.queue_path(out).read_text().splitlines()) == 4
    assert distill.backfill(config, state, current_round=0, **kwargs) == []
    # A row sampled by a policy outside the lineage is skipped with its reason.
    monkeypatch.setattr(admission, 'lineage_checkpoints', lambda config, state: {'7' * 64})
    other = out / 'round-000004'
    other.mkdir()
    (other / 'rollouts.jsonl').write_text(json.dumps(rows[0]) + '\n')
    (other / 'score.json').write_text(json.dumps({'collection_complete': True, 'block_bundle': str(bundle)}))
    [skipped] = distill.backfill(config, state, current_round=5, **kwargs)
    assert skipped['package'] is None and skipped['skipped'] == {'sampled by another policy': 1}
    # A round whose collector bundle was pruned is recorded as lost without reading its rollouts, once.
    lost = out / 'round-000005'
    lost.mkdir()
    (lost / 'rollouts.jsonl').write_text('not read\n')
    (lost / 'score.json').write_text(json.dumps({'collection_complete': True, 'block_bundle': str(tmp_path / 'gone.json')}))
    [record] = distill.backfill(config, state, current_round=6, **kwargs)
    assert record['package'] is None and 'no longer on disk' in record['skipped']
    assert distill.backfill(config, state, current_round=6, **kwargs) == []


def graph_for_queue(row_id):
    from test_memetic_admission import graph as plain
    return plain(row_id)


def test_a_policy_whose_checkpoint_is_gone_is_recorded_not_lowered(tmp_path):
    rows, identity, roles, block = population()
    config = {'out': str(tmp_path)}
    directory = tmp_path / 'round-000000'
    directory.mkdir()
    record = distill.package_round(config, {'round': 1}, directory, rows, policies=identity['checkpoint_sha256'],
                                   binding={}, bundle='', protected=Guard(), checkpoints={},
                                   prepare=lower(identity, roles, block), log=lambda line: None)
    assert record['package'] is None and len(record['lowering_failed']) == 4
    assert all('no longer on disk' in item['error'] for item in record['lowering_failed'])
