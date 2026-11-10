"""Memetic distillation packages enter the joint SFT source at an interleaved window boundary, exactly once."""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from natlang_neuralese.common.jsonio import write_json_durable as write_json
from natlang_neuralese.data.native_sources import NativeGraphSource
from natlang_neuralese.rl import interleave, memetic_admission as admission, memetic_stage, production
from natlang_neuralese.rl.curriculum import DEFAULT as CURRICULUM
from natlang_neuralese.train import native_resident as resident
from natlang_neuralese.train.native_logical_families import load_map
from natlang_neuralese.train.native_resident_ce import extension_policy
from natlang_neuralese.train.own_output_supervision import identity, policy_objective
from scripts.neuralese_data.build_native_ce_supervision_policy import complete_source_contract
from test_native_supervision_mass import ordinary, declared
from test_own_output_supervision import objective

FP = 'tokenizer-sha'


def graph(row_id, token=5, split='train'):
    row = ordinary()
    row.update(id=row_id, tokenizer_fingerprint=FP, split=split, source_groups=[row_id])
    row['nodes'][0]['token_ids'][-1] = token
    return row


class Guard:
    digest = 'protection-digest'

    def __init__(self, protected=()):
        self.protected = set(protected)

    def related(self, row):
        return row['id'] in self.protected


UNCAPPED = {'distillation_max_share': .25, 'distillation_family_scale': .2}


def lineage(tmp_path, **parameters):
    """A controller whose recipe has a declared-scale logical-family joint SFT stage and an interleaved RL stage."""
    paths = {}
    for name, split in (('old', 'train'), ('held', 'test')):
        paths[name] = tmp_path / f'{name}.jsonl'
        paths[name].write_text(json.dumps(graph(name, split=split)) + '\n')
    train = tmp_path / 'train.json'
    write_json(train, {'schema': 'natlang.native-graph-sources/1',
                       'sources': [{'kind': 'native-graphs', **resident.binding(paths['old'])}]})
    source = NativeGraphSource(train, FP, required_split='train')
    families = tmp_path / 'families.json'
    write_json(families, {'schema': 'natlang.logical-family-map/1', 'source_sha256': source.file_sha256,
                          'rows': [{'id': e['id'], 'binding': e['binding'], 'family': 'ordinary'}
                                   for e in source.entries]})
    ce = complete_source_contract(declared(digest=resident.sha(paths['old'])), [resident.binding(paths['old'])],
                                  context_weight=.05, inventory=[resident.binding(train)])
    ce['held_sources'] = [dict(sha256=resident.sha(paths['held']), encoding='native-row-objectives-v1',
                               context_weight=.05)]
    ce['objective'] = objective(mass='token', family_balance={
        'key': 'logical-family', 'max_scale': 8., 'normalization': 'declared/1',
        'map': resident.binding(families), 'scales': {'ordinary': 1.}})
    ce_path = tmp_path / 'ce.json'
    write_json(ce_path, ce)
    cases = tmp_path / 'cases.jsonl'
    cases.write_text(json.dumps({'id': 'task-0', 'split': 'train'}) + '\n')
    recipe = {'schema': 'natlang.neuralese-training-recipe/1', 'id': 'memetic-admission',
              'identity_qualification': 'native_state_runtime_qualified',
              'input_bindings': {'train': resident.binding(train), 'held': resident.binding(paths['held']),
                                 'ce': resident.binding(ce_path), 'rl.tasks': resident.binding(cases)},
              'stages': [{'id': 'sft', 'kind': 'native_state_sft', 'requires': [],
                          'parameters': {'refresh_after_updates': 64, 'epochs': 2.},
                          'inputs': {'native_graphs': 'train', 'foundation_native_graphs': 'train',
                                     'held_native_graphs': 'held', 'ce_supervision_policy': 'ce'}},
                         {'id': 'interleaved_rl', 'kind': 'native_interleaved_rl', 'requires': ['sft'],
                          'inputs': {'rl_task_source': 'rl.tasks'},
                          'parameters': {'sft_stage': 'sft', 'rl_rounds_per_window': 1, 'rows_per_round': 16,
                                         'group': 8, 'task_curriculum': CURRICULUM, **parameters}}]}
    recipe_path = tmp_path / 'recipe.json'
    write_json(recipe_path, recipe)
    parent = tmp_path / 'parent.pt'
    parent.write_bytes(b'parent')
    out = tmp_path / 'controller'
    out.mkdir()
    config = {'production_recipe': str(recipe_path), 'production_stage': 'sft',
              'interleaved_rl_stage': 'interleaved_rl', 'out': str(out), 'checkpoint': str(parent)}
    state = {'round': 1, 'checkpoint': str(parent)}
    config, declared_stage = interleave.bind_stage(config, state, out / 'state.json')
    return config, declared_stage, state, out, train


def package(out, rows, actor):
    """A real producer package (memetic_stage.package_distillation) of ``rows`` run by ``actor``."""
    distill = out / 'memetic' / 'distill'
    graphs = distill / f'source-{actor[:8]}-{rows[0]["id"]}.jsonl'
    graphs.parent.mkdir(parents=True, exist_ok=True)
    graphs.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    manifest = distill / f'{actor[:16]}-{rows[0]["id"]}' / 'manifest.json'
    manifest.parent.mkdir(parents=True)
    manifest.write_text(json.dumps({'parent_sha256': actor, 'supervisor_episodes': [{'policy': actor}],
                                    'guidance_graphs': [{'path': str(graphs),
                                                         'sha256': hashlib.sha256(graphs.read_bytes()).hexdigest()}]}))
    return memetic_stage.package_distillation(out / 'memetic', tokenizer=FP, roles=[], protected=Guard(),
                                              log=lambda line: None)


def window_harness(monkeypatch, seen):
    from natlang_neuralese.rl import native_supervised

    def continuation(source, stage, *, checkpoint, out):
        seen.append((source['production_recipe'], stage))
        return {'checkpoint': checkpoint, 'out': out, 'production_recipe': source['production_recipe']}, {}

    def execute(window, window_state, window_state_path, control):
        learner = Path(window['out']) / 'learner'
        learner.mkdir(parents=True, exist_ok=True)
        (learner / 'checkpoint.pt').write_bytes(b'trained-' + str(len(seen)).encode())
        (learner / 'report.json').write_text(json.dumps({'continuation_boundary': {'new_optimizer_updates': 8}}))
        window_state.update(phase='awaiting_semantic_refresh', pending_native_checkpoint=str(learner / 'checkpoint.pt'))
        return window_state

    monkeypatch.setattr(native_supervised, 'continuation_config', continuation)
    monkeypatch.setattr(native_supervised, 'execute', execute)
    monkeypatch.setattr(production, 'initial_controller_state', lambda c: {'checkpoint': c['checkpoint']})
    monkeypatch.setattr('natlang_neuralese.common.custody.ProtectedWorlds.load',
                        classmethod(lambda cls, *a, **k: Guard({'memetic:protected'})))


def test_window_boundary_appends_admitted_rows_once_with_their_own_family(tmp_path, monkeypatch):
    config, declared, state, out, train = lineage(tmp_path, **UNCAPPED)
    actor = resident.sha(config['checkpoint'])
    package(out, [graph('memetic:a', 6), graph('memetic:protected', 7), graph('memetic:b', 8)], actor)
    seen = []
    window_harness(monkeypatch, seen)
    state_path = out / 'state.json'
    assert interleave.run_sft_window(config, declared, state, state_path, None) == 'adopted'
    recipe_path, stage = seen[0]
    assert stage['inputs']['continued_native_graphs'].startswith('memetic_predecessor_')
    from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
    old_inputs = resolve_stage_inputs(load_recipe(config['production_recipe']),
                                      load_recipe(config['production_recipe'])['stages'][0], {})
    recipe = load_recipe(recipe_path)
    inputs = resolve_stage_inputs(recipe, stage, {})
    assert inputs['continued_native_graphs']['sha256'] == resident.sha(train)
    assert inputs['foundation_native_graphs'] == inputs['native_graphs']
    successor = NativeGraphSource(inputs['native_graphs']['path'], FP, required_split='train')
    prior = NativeGraphSource(train, FP, required_split='train')
    # Exact predecessor prefix and order, then only the unprotected package rows, prioritized.
    assert [e['id'] for e in successor.entries] == ['old', 'memetic:a', 'memetic:b']
    assert successor.entries[0]['binding'] == prior.entries[0]['binding']
    assert [e['training_priority'] for e in successor.entries] == [0, 1, 1]
    # The CE family extension is the trainer's accepted resident form; the map binds the exact successor.
    policy = extension_policy(old_inputs, inputs, prior_train=prior, candidate_train=successor)
    balance = policy['objective']['family_balance']
    assert balance['scales'] == {'ordinary': 1., 'memetic-guidance': .2}
    assert load_map(balance['map'], successor) == ['ordinary', 'memetic-guidance', 'memetic-guidance']
    ledger = state['interleave']['memetic_distillation']
    [record_entry] = ledger['admissions']
    record = json.loads(Path(record_entry['record']['path']).read_text())
    assert record['actor_provenance']['kind'] == 'student-self-distillation' and actor in \
        record['actor_provenance']['lineage_checkpoints']
    assert record['packages'][0]['excluded'] == {
        'memetic:protected': 'related to a protected world in the current protection union'}
    assert ledger['rows']['memetic:a']['disposition'] == 'admitted'
    assert list(ledger['packages'].values())[0]['status'] == 'admitted'
    window_config = json.loads((out / 'sft-window-000000' / 'config.json').read_text())
    assert window_config['interleaved_window']['memetic_admission'] == record_entry['record']
    # The next window continues on the appended source without re-appending.
    state['round'] = 3
    assert interleave.run_sft_window(config, declared, state, state_path, None) == 'adopted'
    later_recipe, later_stage = seen[1]
    assert later_recipe == ledger['recipe']['path'] and 'continued_native_graphs' not in later_stage['inputs']
    later = resolve_stage_inputs(load_recipe(later_recipe), later_stage, {})
    assert later['native_graphs']['sha256'] == inputs['native_graphs']['sha256']
    assert len(state['interleave']['memetic_distillation']['admissions']) == 1


def test_retried_boundary_reuses_its_committed_admission(tmp_path, monkeypatch):
    config, declared, state, out, _ = lineage(tmp_path, **UNCAPPED)
    package(out, [graph('memetic:a', 6)], resident.sha(config['checkpoint']))
    window_harness(monkeypatch, [])
    directory = out / 'sft-window-000000'
    first = admission.prepare_window(config, declared, state, out / 'state.json', directory, log=lambda l: None)
    # The controller died before writing the window config: the same window gets the same committed append.
    saved = json.loads((out / 'state.json').read_text())
    second = admission.prepare_window(config, declared, saved, out / 'state.json', directory, log=lambda l: None)
    assert first[0] == second[0] and first[2] == second[2]
    assert len(saved['interleave']['memetic_distillation']['admissions']) == 1


def test_foreign_actor_is_refused_and_the_boundary_is_unchanged(tmp_path, monkeypatch):
    config, declared, state, out, _ = lineage(tmp_path)
    package(out, [graph('memetic:a', 6)], 'e' * 64)
    seen = []
    window_harness(monkeypatch, seen)
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    assert seen[0][0] == config['production_recipe'] and seen[0][1] == declared['sft_stage']
    [entry] = state['interleave']['memetic_distillation']['packages'].values()
    assert entry['status'] == 'refused' and 'not checkpoints of this lineage' in entry['reason']
    assert not state['interleave']['memetic_distillation']['admissions']


def test_share_cap_defers_rows_and_new_packages_reopen_a_completed_stage(tmp_path, monkeypatch):
    # One train row: n x 0.4 <= 0.2 x (1 + n) admits exactly one memetic row.
    config, declared, state, out, _ = lineage(tmp_path, distillation_max_share=.2, distillation_family_scale=.4)
    package(out, [graph('memetic:a', 6), graph('memetic:b', 8)], resident.sha(config['checkpoint']))
    seen = []
    window_harness(monkeypatch, seen)
    state['interleave']['sft_complete'] = True
    assert not interleave.window_due(declared, state)  # the declared window cadence still applies
    state['round'] = 2
    assert interleave.window_due(declared, state)  # an unseen package reopens the joint SFT stage
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    ledger = state['interleave']['memetic_distillation']
    assert ledger['rows']['memetic:a']['disposition'] == 'admitted'
    assert ledger['rows']['memetic:b'] == {**ledger['rows']['memetic:b'], 'disposition': 'deferred',
                                            'reason': 'family share cap'}
    assert list(ledger['packages'].values())[0]['status'] == 'partial'
    record = json.loads(Path(ledger['admissions'][0]['record']['path']).read_text())
    assert record['share_cap']['budget_rows'] == 1 and record['share_cap']['bound_scale'] == .4
    with pytest.raises(ValueError, match='distillation_max_share'):
        admission.validate_parameters({'distillation_max_share': .5})


def test_no_package_leaves_the_boundary_unchanged(tmp_path, monkeypatch):
    config, declared, state, out, _ = lineage(tmp_path)
    seen = []
    window_harness(monkeypatch, seen)
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    assert seen == [(config['production_recipe'], declared['sft_stage'])]
    assert 'memetic_distillation' not in state['interleave']


STATIC = {'origin': 'unit-test source', 'verification': 'unit-test verification receipt', 'selection': 'all rows'}


def test_prepared_static_source_is_admitted_after_the_lineages_own_rows_without_an_actor(tmp_path, monkeypatch):
    # The static package is queued first, but the share cap (budget 2 rows) goes to the lineage's own rows first.
    config, declared, state, out, _ = lineage(tmp_path, distillation_max_share=.2, distillation_family_scale=.3)
    memetic = out / 'memetic'
    admission.queue_package(memetic, [graph('static:a', 5), graph('static:b', 6)], family=admission.STATIC_FAMILY,
                            tokenizer=FP, roles=[], protected=Guard(), manifests=[tmp_path / 's.json'], actors=[],
                            scope='test', static_source=STATIC, log=lambda line: None)
    admission.queue_package(memetic, [graph('rollout:a', 7), graph('rollout:b', 8)], family=admission.ROLLOUT_FAMILY,
                            tokenizer=FP, roles=[], protected=Guard(), manifests=[tmp_path / 'r.json'],
                            actors=[resident.sha(config['checkpoint'])], scope='test', log=lambda line: None)
    seen = []
    window_harness(monkeypatch, seen)
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    ledger = state['interleave']['memetic_distillation']
    assert [ledger['rows'][i]['disposition'] for i in ('rollout:a', 'rollout:b')] == ['admitted', 'admitted']
    assert [ledger['rows'][i]['disposition'] for i in ('static:a', 'static:b')] == ['deferred', 'deferred']
    record = json.loads(Path(ledger['admissions'][0]['record']['path']).read_text())
    static = next(item for item in record['packages'] if item['family'] == admission.STATIC_FAMILY)
    assert static['status'] == 'partial' and static['episode_actors'] == [] and static['source_provenance'] == STATIC
    assert not record['refused']


def test_prepared_static_source_rows_take_their_own_family_and_pass_the_protection_recheck(tmp_path, monkeypatch):
    config, declared, state, out, _ = lineage(tmp_path, **UNCAPPED)
    admission.queue_package(out / 'memetic', [graph('static:a', 5), graph('static:protected', 6)],
                            family=admission.STATIC_FAMILY, tokenizer=FP, roles=[], protected=Guard(),
                            manifests=[tmp_path / 's.json'], actors=[], scope='test', static_source=STATIC,
                            log=lambda line: None)
    seen = []
    window_harness(monkeypatch, seen)
    from natlang_neuralese.common import custody
    monkeypatch.setattr(custody.ProtectedWorlds, 'load', classmethod(lambda cls: Guard({'static:protected'})))
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    ledger = state['interleave']['memetic_distillation']
    assert ledger['rows']['static:a']['disposition'] == 'admitted'
    assert ledger['rows']['static:protected']['disposition'] == 'excluded'
    from natlang_neuralese.train.recipe import load_recipe, resolve_stage_inputs
    recipe = load_recipe(seen[0][0])
    inputs = resolve_stage_inputs(recipe, seen[0][1], {})
    balance = json.loads(Path(inputs['ce_supervision_policy']['path']).read_text())['objective']['family_balance']
    assert balance['scales'] == {'ordinary': 1., 'prepared-source': .2}


def test_static_and_lineage_packages_validate_their_provenance(tmp_path):
    out = tmp_path / 'memetic'
    common = dict(tokenizer=FP, roles=[], protected=Guard(), manifests=[tmp_path / 'm.json'], scope='test',
                  log=lambda line: None)
    with pytest.raises(ValueError, match='names no actor'):
        admission.queue_package(out, [graph('s:a')], family=admission.STATIC_FAMILY, actors=['a' * 64],
                                static_source=STATIC, **common)
    with pytest.raises(ValueError, match='names no actor'):
        admission.queue_package(out, [graph('s:a')], family=admission.STATIC_FAMILY, actors=[],
                                static_source={'origin': 'x'}, **common)
    with pytest.raises(ValueError, match='names its actors'):
        admission.queue_package(out, [graph('s:a')], family=admission.ROLLOUT_FAMILY, actors=[], **common)
    with pytest.raises(ValueError, match='names its actors'):
        admission.queue_package(out, [graph('s:a')], family=admission.ROLLOUT_FAMILY, actors=['a' * 64],
                                static_source=STATIC, **common)
    assert not (out / 'distill').exists()


def test_an_append_that_cannot_be_prepared_is_recorded_and_the_window_runs_on_the_existing_source(tmp_path, monkeypatch):
    config, declared, state, out, _ = lineage(tmp_path, **UNCAPPED)
    package(out, [graph('memetic:a', 6)], resident.sha(config['checkpoint']))
    seen = []
    window_harness(monkeypatch, seen)

    def refuse(*args, **kwargs):
        raise ValueError('resident CE historical observation is not the authenticated target-mask replacement')

    monkeypatch.setattr(admission, '_ce_update', refuse)
    assert interleave.run_sft_window(config, declared, state, out / 'state.json', None) == 'adopted'
    assert seen[0][0] == config['production_recipe'] and seen[0][1] == declared['sft_stage']
    [failure] = state['interleave']['memetic_distillation_failures']
    assert failure['window'] == 0 and 'target-mask replacement' in failure['error']
    assert 'memetic_distillation' not in state['interleave']  # nothing committed: the package stays pending
    assert admission.unseen_packages(config['out'], state)
