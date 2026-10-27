"""The learned optimizer's resumable entrypoint end to end: bank-init, targets, train, propose, ladder."""
import json

from test_native_serving_adapters import engine_fixture
from test_tune_trainer import manifest_for, rows_for
from test_learned_optimizer_trainer import soft_rows
from test_learned_optimizer_writer import GRID
from natlang_neuralese.tune.learned.__main__ import main
from natlang_neuralese.tune.soft import soft_key


def write(path, value):
    path.write_text(json.dumps(value))
    return str(path)


def jsonl(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def test_entrypoint_runs_the_whole_operation(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm')
    checkpoint = str(tmp_path / 'native.pt')
    from natlang_neuralese.serve.store import TensorStore
    store = str(tmp_path / 'store')
    engine.store.close()
    engine.store = TensorStore(store)
    common = {'checkpoint': checkpoint, 'store': store, 'grid_tokens': {**GRID.__dict__, 'digits': list(GRID.digits)}}
    sources = [{'name': 'bx', 'kind': 'binding', 'rows': jsonl(tmp_path / 'bx.jsonl', rows_for(engine, 'bx', 2)),
                'description': 't30 t31'},
               {'name': 'by', 'kind': 'binding', 'rows': jsonl(tmp_path / 'by.jsonl', rows_for(engine, 'by', 2, offset=2)),
                'description': 't32 t33'}]
    bank_spec = write(tmp_path / 'bank-spec.json', {**common, 'out': str(tmp_path / 'bank.json'), 'rank': 4, 'per_source_rank': 2,
                                                    'component_rank': 2, 'component_steps': 2, 'step': .08, 'sources': sources})
    incoming = engine.native_write_text('t8 t9')['id']
    key = soft_key('bx', 'instructions', None)
    rows = soft_rows(engine, incoming, 2)
    manifest = manifest_for(engine)
    engine.store.close()  # every command opens the operation's durable store itself
    assert main(['bank-init', '--spec', bank_spec]) == 0
    program = {'manifest': write(tmp_path / 'manifest.json', manifest),
               'rows': jsonl(tmp_path / 'rows.jsonl', rows), 'contract': 'route tickets',
               'soft_texts': {key: 't8 t9'}, 'incoming_states': {key: incoming},
               'scopes': {'lambda:x': {'level': 'lambda', 'size': 2}}, 'episodes': 1}
    targets_spec = write(tmp_path / 'targets-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'out': str(tmp_path / 'targets.jsonl'),
                                                          'programs': [program], 'fit': {'steps': 2, 'refine_steps': 1}})
    assert main(['targets', '--spec', targets_spec]) == 0
    train_spec = write(tmp_path / 'train-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'targets': str(tmp_path / 'targets.jsonl'),
                                                      'state': str(tmp_path / 'state.json'), 'receipt': str(tmp_path / 'receipt.json'), 'steps': 1,
                                                      'recipe': {'writer': {'kind': 'free-lora', 'rank': 2, 'alpha': 2.}, 'batch_targets': 1}})
    assert main(['train', '--spec', train_spec]) == 0
    assert json.loads((tmp_path / 'state.json').read_text())['step'] == 1
    assert main(['train', '--spec', train_spec]) == 0  # resumes
    assert json.loads((tmp_path / 'receipt.json').read_text())['step'] == 2
    context = json.loads((tmp_path / 'targets.jsonl').read_text().splitlines()[0])
    context['program'] = {'id': 'p', 'buildHash': 'b'}
    propose_spec = write(tmp_path / 'propose-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'state': str(tmp_path / 'state.json'),
                                                          'context': write(tmp_path / 'context.json', context), 'out': str(tmp_path / 'proposal.json')})
    assert main(['propose', '--spec', propose_spec]) == 0
    proposal = json.loads((tmp_path / 'proposal.json').read_text())
    assert proposal['bundle']['instructions']['bx'].startswith('nzs1_') and proposal['bundle']['adapters']['lambda:x'].startswith('nnt1_')
    ladder_spec = write(tmp_path / 'ladder-spec.json', {'out': str(tmp_path / 'ladder.json'), 'problems': [
        {'role': 'development', 'measurements': {'incoming': .1, 'learned_episode': .3, 'in_bank_support': .5, 'free_tune': .6, 'in_bank_query': .7}}]})
    assert main(['ladder', '--spec', ladder_spec]) == 0
    report = json.loads((tmp_path / 'ladder.json').read_text())
    assert report['problems'][0]['largest'] == 'headroom' and 'in_bank_query' not in report['problems'][0]['rungs']
