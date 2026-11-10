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
        value = {'task_id': task, 'sample': sample, 'split': 'train', 'group': task,
                 'reward': {'quality': quality},
                 'policy': {'checkpoint_sha256': policy}, 'turns': copy.deepcopy(raw)}
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


def test_selection_is_student_current_policy_one_per_task_neuralese_first():
    rows, identity, _, _ = population()
    selected, skipped = distill.select(rows, identity['checkpoint_sha256'])
    assert [(row['task_id'], row['sample']) for row in selected if row['task_id'] in 'ab'] == [('a', 1), ('b', 0)]
    assert [row['task_id'] for row in selected] == ['a', 'b', 'h']
    assert not {'c', 'd', 'e', 'f', 'g'} & {row['task_id'] for row in selected}
    assert skipped == {'teacher episode': 1, 'not a train task': 1, 'sampled by another policy': 1,
                       'memetic improvement episode': 1}
    assert all(row['_neuralese_exchanges'] for row in selected if row['task_id'] in 'ab')
    # The Neuralese-using success of a task beats an earlier plain one even with a smaller return.
    plain = copy.deepcopy(rows[1])
    plain['sample'] = 7
    plain['reward'] = {'quality': 1.0, 'neuralese_bonus': 0.}
    plain['turns'] = rows[8]['turns']
    plain['execution_graph'] = rows[8]['execution_graph']
    chosen, _ = distill.select([plain, rows[0]], identity['checkpoint_sha256'])
    assert [(row['task_id'], row['sample']) for row in chosen] == [('a', 0)]


def test_round_package_holds_ranked_closed_whole_graphs_with_actor_provenance(tmp_path):
    rows, identity, roles, block = population()
    config = {'out': str(tmp_path / 'controller')}
    state = {'round': 4, 'checkpoint': 'unused'}
    round_dir = tmp_path / 'controller' / 'round-000004'
    round_dir.mkdir(parents=True)
    geometry = lambda tokenizer: (identity['tokenizer_fingerprint'], [roles.produce_open, roles.produce_close,
                                                                     roles.consume_open, roles.consume_close])
    record = distill.package_round(config, state, round_dir, rows, policy_sha256=identity['checkpoint_sha256'],
                                   binding={'rl_tokenizer': 'tok'}, bundle='', protected=Guard(),
                                   prepare=lower(identity, roles, block), geometry=geometry, log=lambda line: None)
    # a (Neuralese-using, weighted by its return), b; h imports old external state and is skipped.
    assert record['rows'] == 2 and record['selected_tasks'] == 3 and record['neuralese_using_rows'] == 2
    assert record['not_closed'] == [{'task_id': 'h', 'reason': 'imports old external actor state'}]
    [item] = [json.loads(line) for line in admission.queue_path(config['out']).read_text().splitlines()]
    assert item['manifests'] == [str(round_dir / distill.RECORD)] and item['family'] == 'rollout-distillation'
    receipt = json.loads(Path(item['package']).read_text())
    assert receipt['family'] == admission.ROLLOUT_FAMILY and receipt['training_admission'] is False
    assert receipt['episode_actors'] == [identity['checkpoint_sha256']] and receipt['round'] == 4
    source = Path(receipt['source']['path'])
    graphs = [json.loads(line) for line in source.read_text().splitlines()]
    assert [row['rollout_distillation']['task_id'] for row in graphs] == ['a', 'b']
    assert all(row['id'].startswith('rollout:') and not row['external_states'] and row['split'] == 'train'
               for row in graphs)
    assert graphs[0]['rollout_distillation']['sampled_token_weight'] == pytest.approx(1.1)
    assert graphs[1]['rollout_distillation']['sampled_token_weight'] == 1.
    weights = lambda row: sorted({w for node in row['nodes'] for w in node['target_weights']})
    assert weights(graphs[0]) == pytest.approx([0., 1.1]) and weights(graphs[1]) == [0., 1.]
    # Only sampled own tokens carry loss; the indexed source is the exact receipt.
    indexed = NativeGraphSource(source, identity['tokenizer_fingerprint'], required_split='train',
                                role_ids=receipt['role_ids'])
    assert [entry['id'] for entry in indexed.entries] == [row['id'] for row in graphs]
    # Idempotent: the same round is never queued twice.
    again = distill.package_round(config, state, round_dir, rows, policy_sha256=identity['checkpoint_sha256'],
                                  binding={'rl_tokenizer': 'tok'}, bundle='', protected=Guard(),
                                  prepare=lower(identity, roles, block), geometry=geometry, log=lambda line: None)
    assert again == record and len(admission.queue_path(config['out']).read_text().splitlines()) == 1


def test_protected_and_unsuccessful_rounds_queue_nothing(tmp_path):
    rows, identity, roles, block = population()
    config = {'out': str(tmp_path)}
    round_dir = tmp_path / 'round'
    round_dir.mkdir()
    none = distill.package_round(config, {'round': 1}, round_dir, [rows[3]], policy_sha256=identity['checkpoint_sha256'],
                                 binding={}, bundle='', protected=Guard(), log=lambda line: None)
    assert none['selected_tasks'] == 0 and none['package'] is None and not admission.queue_path(tmp_path).exists()


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
