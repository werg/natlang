"""One representation from tuner to learned optimizer: a fit on the bank basis is exact bank coordinates, its build
becomes free-tune and in-bank targets through the same plan, and the optimizer revises the tuned bundle it releases."""
import dataclasses
import json
from pathlib import Path

import torch

from test_learned_optimizer_bank import _component
from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from natlang_neuralese.model.native_adapters import target_parameters
from natlang_neuralese.serve.store import TensorStore
from natlang_neuralese.tune.build import main as build, persist_artifacts
from natlang_neuralese.tune.learned.mixture import MixtureFormat
from natlang_neuralese.tune.learned.__main__ import main as learned
from natlang_neuralese.tune.learned.apply import Context, release_fragment, service
from natlang_neuralese.tune.learned.bank import load_bank, new_bank, save_bank, weight_svd_basis
from natlang_neuralese.tune.learned.bundle import verify_bundle
from natlang_neuralese.tune.learned.targets import project_adapter
from natlang_neuralese.tune.learned.trainer import OuterRecipe, OuterTrainer
from natlang_neuralese.tune.targets import module_map


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def operation(tmp_path):
    """A bank over every tuner target matrix, a tuner fit on its basis with boundaries, and an untrained optimizer."""
    engine = engine_fixture(tmp_path, 'lfm')
    store = tmp_path / 'store'
    engine.store.close()
    engine.store = TensorStore(store)
    names, _ = module_map(sorted(target_parameters(engine.shared_model)))
    bank = new_bank(engine, weight_svd_basis(engine, sorted(names), 4), MixtureFormat())
    bank = bank.with_components([dataclasses.replace(_component(bank, seed), key=engine.native_write_text(f't{40 + seed} t{50 + seed}')['id'])
                                 for seed in (1, 2, 3)], reason='planted')
    write(tmp_path / 'bank.json', {'bank': save_bank(engine, bank)})
    persist_artifacts(engine, [bank.basis_artifact], tmp_path / 'bank-artifacts')
    trainer = OuterTrainer(engine, bank, OuterRecipe(writer={'kind': 'free-lora', 'rank': 2, 'alpha': 2.}), [])
    write(tmp_path / 'state.json', trainer.state)
    manifest = manifest_for(engine)
    rows = rows_for(engine, 'bx', 3) + rows_for(engine, 'by', 2, offset=2)
    engine.store.close()
    (tmp_path / 'rows.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
    plan = write(tmp_path / 'plan.json', {
        'schema': 'natlang.tune-plan/v1', 'manifest': manifest, 'rows': 'rows.jsonl', 'artifacts': ['bank-artifacts'],
        'student': {'name': 'lfm', 'checkpoint': str(tmp_path / 'native.pt')}, 'softStates': False, 'exports': [],
        'emitBoundaries': True, 'label': 'fit',
        'recipe': {'steps': 2, 'batch_rows': 5, 'prompt_weight': 0, 'basis': bank.basis_artifact,
                   'ranks': {'tree': 4, 'folder': 2, 'lambda': 2}},
        'learned': {'bank': str(tmp_path / 'bank.json'), 'state': str(tmp_path / 'state.json'), 'store': str(store),
                    'scopes': {'tree': 3, 'folder': 3, 'lambda': 3}, 'steps': 2, 'patience': 2, 'seed': 1}})
    assert build(['--plan', plan, '--out', str(tmp_path / 'fit')]) == 0
    return plan, store, bank


def open_engine(tmp_path, store):
    from natlang_neuralese.serve import load_native_engine
    engine = load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=torch.float32)
    engine.store.close()
    engine.store = TensorStore(store)
    return engine


def test_tuner_on_the_bank_basis_yields_exact_bank_coordinates_and_targets(tmp_path):
    plan, store, bank = operation(tmp_path)
    state = json.loads((tmp_path / 'fit' / 'state.json').read_text())
    engine = open_engine(tmp_path, store)
    try:
        from natlang_neuralese.tune.build import import_archive
        import_archive(engine, [tmp_path / 'fit' / 'artifacts'], [tmp_path / 'fit' / 'states'])
        bank = load_bank(engine, json.loads((tmp_path / 'bank.json').read_text())['bank'])
        for node, adapter in state['node_artifacts'].items():
            record = engine.store.get_native_tensor_artifact(adapter).artifact
            assert record.provenance['adapter_spec']['bases'] == 'bank-basis-slice-v1'
            assert record.source_artifact_id == bank.basis_artifact
            projected = project_adapter(bank, engine, adapter)
            # The tuned node's update lies inside the bank basis: captured exactly, and its R x R cores embed it.
            if projected['basisCapture'] is not None:
                assert abs(projected['basisCapture'] - 1) < 1e-4
            cores = bank.adapter_cores(engine, adapter)
            for name, core in cores.items():
                assert torch.allclose(core, projected['cores'][name], atol=1e-5)
    finally:
        engine.store.close()
    spec = write(tmp_path / 'targets-spec.json', {
        'checkpoint': str(tmp_path / 'native.pt'), 'store': str(store), 'bank': str(tmp_path / 'bank.json'),
        'out': str(tmp_path / 'targets.jsonl'), 'work': str(tmp_path / 'work'), 'fit': {'steps': 2, 'refine_steps': 1},
        'programs': [{'plan': plan, 'build': str(tmp_path / 'fit'), 'subsets': 2}]})
    assert learned(['targets', '--spec', spec]) == 0
    targets = [json.loads(line) for line in (tmp_path / 'targets.jsonl').read_text().splitlines()]
    free = [item for item in targets if item['tier'] == 'free-tune']
    kinds = [item['lineage']['episode']['kind'] for item in free]
    assert kinds == ['final-residual', 'incremental']
    final, incremental = free
    assert final['incoming_adapters'] == {} and final['target_bundle']['adapters'] == state['node_artifacts']
    assert final['target_factors'] == state['node_artifacts']
    # The incremental revision starts from the partly tuned bundle of step 1 and targets step 2.
    assert set(incremental['incoming_adapters']) == set(state['node_artifacts'])
    assert incremental['target_bundle']['adapters'] == state['node_artifacts']
    starts = {item['lineage']['episode']['start'] for item in targets if item['tier'] == 'in-bank-fit'}
    assert starts == {'base', 'tuned'}
    train = write(tmp_path / 'train-spec.json', {
        'checkpoint': str(tmp_path / 'native.pt'), 'store': str(store), 'bank': str(tmp_path / 'bank.json'),
        'targets': str(tmp_path / 'targets.jsonl'), 'state': str(tmp_path / 'outer.json'), 'receipt': str(tmp_path / 'receipt.json'),
        'steps': 2,
        'recipe': {'writer': {'kind': 'free-lora', 'rank': 2, 'alpha': 2.}, 'batch_targets': 2}})
    assert learned(['train', '--spec', train]) == 0
    assert json.loads((tmp_path / 'receipt.json').read_text())['step'] == 2


def test_optimizer_revises_the_tuned_bundle_it_releases(tmp_path):
    plan, store, bank = operation(tmp_path)
    settings = json.loads((tmp_path / 'plan.json').read_text())
    settings['learned']['incoming'] = str(tmp_path / 'fit')
    plan = write(tmp_path / 'plan.json', settings)
    state = json.loads((tmp_path / 'fit' / 'state.json').read_text())
    context = Context(plan, tmp_path / 'apply')
    try:
        assert context.incoming_adapters == state['node_artifacts']
        # Nothing accepted: the release is the tuned incoming bundle itself, with each node's actual rank.
        artifacts, _ = release_fragment(context, None)
        assert {item['node']: item['artifact'] for item in artifacts} == state['node_artifacts']
        assert {item['rank'] for item in artifacts if item['node'] == 'tree:'} == {4}
        tuned = context.evaluate(None)
        bundle, _, _ = context.propose(None, sample=False, seed=0)
        for scope, mixture in bundle['loraMixtures'].items():
            assert mixture['incoming'] == state['node_artifacts'][scope]
        verify_bundle(context.engine, bundle, context.bank)
        # A proposal whose mixture is all null executes exactly the tuned bundle.
        if all(not any(mixture['distribution'][1:]) for mixture in bundle['loraMixtures'].values()):
            assert abs(context.evaluate(bundle) - tuned) < 1e-5
    finally:
        context.close()
    refined = service({'plan': plan, 'out': str(tmp_path / 'svc'), 'op': 'refine', 'parent': None, 'steps': 2})
    engine = open_engine(tmp_path, store)
    try:
        verify_bundle(engine, refined['bundle'], load_bank(engine, json.loads((tmp_path / 'bank.json').read_text())['bank']))
    finally:
        engine.store.close()


def test_planted_controls_compare_the_optimizer_with_the_in_bank_reference(tmp_path):
    plan, store, _ = operation(tmp_path)
    spec = write(tmp_path / 'controls-spec.json', {
        'checkpoint': str(tmp_path / 'native.pt'), 'store': str(store), 'bank': str(tmp_path / 'bank.json'),
        'state': str(tmp_path / 'state.json'), 'plan': plan, 'work': str(tmp_path / 'controls'), 'out': str(tmp_path / 'controls.json'),
        'trials': 2, 'prompts': 2, 'prompt_tokens': 3, 'fit': {'steps': 2, 'refine_steps': 1}})
    assert learned(['controls', '--spec', spec]) == 0
    report = json.loads((tmp_path / 'controls.json').read_text())
    assert len(report['trials']) == 2
    for trial in report['trials']:
        assert sum(1 for value in trial['planted'] if value) == 2 and abs(sum(trial['planted']) - 1) < 1e-9
        assert len(trial['optimizer']) == len(trial['planted']) == len(trial['inBankFit'])
        assert 0 <= trial['optimizerRecovery']['supportRecall'] <= 1


def test_corpus_rebinds_an_earlier_students_plan_and_labels_targets(tmp_path):
    _, store, _ = operation(tmp_path)
    source = json.loads((tmp_path / 'plan.json').read_text())
    earlier = {**source['manifest']['base'], 'checkpoint_sha256': '0' * 64}
    source['manifest'] = {**source['manifest'], 'base': earlier}
    for key in ('learned', 'emitBoundaries'):
        source.pop(key)
    old = write(tmp_path / 'earlier-plan.json', source)
    spec = write(tmp_path / 'corpus-spec.json', {
        'checkpoint': str(tmp_path / 'native.pt'), 'store': str(store), 'bank': str(tmp_path / 'bank.json'),
        'work': str(tmp_path / 'corpus'), 'out': str(tmp_path / 'corpus' / 'targets.jsonl'), 'rebase': True,
        'recipe': {'steps': 2, 'ranks': {'tree': 4, 'folder': 2, 'lambda': 2}}, 'fit': {'steps': 1, 'refine_steps': 1}, 'subsets': 1,
        'programs': [{'name': 'p', 'plan': old}]})
    assert learned(['corpus', '--spec', spec]) == 0
    plan = json.loads((tmp_path / 'corpus' / 'p' / 'plan.json').read_text())
    assert plan['rebasedFrom'] == earlier and plan['manifest']['base']['checkpoint_sha256'] != '0' * 64
    assert plan['recipe']['basis'] and plan['emitBoundaries'] is True
    tiers = {json.loads(line)['tier'] for line in (tmp_path / 'corpus' / 'targets.jsonl').read_text().splitlines()}
    assert tiers == {'free-tune', 'in-bank-fit'}


def test_meta_examples_label_the_optimizers_own_stages_exactly(tmp_path):
    from natlang_neuralese.tune.learned.meta import stop_labels
    assert stop_labels([3., 2., 2., 1.]) == [False, False, False, True]
    assert stop_labels([3., 3.]) == [True, True]
    plan, _, _ = operation(tmp_path)
    spec = write(tmp_path / 'meta-spec.json', {'programs': [{'plan': plan}], 'work': str(tmp_path / 'meta'),
                                               'out': str(tmp_path / 'meta' / 'tune.examples.jsonl'), 'steps': 2,
                                               'refine_steps': 1, 'fit': {'steps': 1, 'refine_steps': 1}})
    assert learned(['meta-examples', '--spec', spec]) == 0
    rows = [json.loads(line) for line in (tmp_path / 'meta' / 'tune.examples.jsonl').read_text().splitlines()]
    functions = [row['function'] for row in rows]
    assert functions == ['adapt/chooseFocus.nl', 'adapt/chooseRevision.nl', 'adapt/chooseOperator.nl', 'adapt/chooseOperator.nl',
                         'adapt/shouldStop.nl', 'adapt/shouldStop.nl', 'adapt.nl']
    assert {row['group'] for row in rows} == {'meta:p'}
    focus, revision, operators, root = rows[0], rows[1], rows[2:4], rows[-1]
    assert focus['expected'] == sorted(focus['expected']) and set(focus['expected']) <= {'bx', 'by'}
    receipts = [json.loads(line) for line in (tmp_path / 'meta' / 'tune.examples.jsonl.receipts.jsonl').read_text().splitlines()]
    # The revision label: per focused binding, the single operation that lowers the support loss most, else "keep".
    assert set(revision['expected']['operations']) == {'bx', 'by'} and revision['input']['focus'] == focus['expected']
    for binding, chosen in revision['expected']['operations'].items():
        tried = revision['reference']['losses'].get(binding, {})
        if binding not in focus['expected']:
            assert chosen == 'keep' and not tried
            continue
        best = min(tried, key=tried.get)
        assert chosen == (best if tried[best] < revision['reference']['incomingLoss'] else 'keep')
    for row, step in zip(operators, receipts[0]['steps']):
        losses = step['losses']
        best = min(losses, key=losses.get) if losses else None
        assert row['expected'] == (best if best and losses[best] < step['incumbentLoss'] else 'keep')
        assert row['input']['plan']['operations'] == revision['expected']['operations']
        assert set(losses) == (set() if set(revision['expected']['operations'].values()) == {'keep'}
                               else {'learned-proposal', 'local-refinement', 'retrieval'})
    assert rows[-2]['expected'] is True and rows[-2]['input']['state']['remainingSteps'] == 0
    # Graded rewards (applications/learned-optimizer/tune.reward.mjs) read these exact measurements.
    assert focus['reference'] == {'focus': focus['expected']}
    assert rows[-2]['reference'] == {'stop': True, 'remainingShare': 0.}
    # The whole episode is a root too, run against the live services of this program's context.
    assert root['input'] == {'maxSteps': 2} and 'expected' not in root
    assert root['reference']['plan'] == str(Path(plan).resolve()) and Path(root['reference']['out']).is_dir()
    assert root['reference']['bestLoss'] == receipts[0]['bestLosses'][-1] <= root['reference']['incomingLoss']
    assert learned(['meta-examples', '--spec', spec]) == 0  # finished programs are skipped
    assert len((tmp_path / 'meta' / 'tune.examples.jsonl').read_text().splitlines()) == len(rows)
