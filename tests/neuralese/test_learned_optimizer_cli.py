"""The learned optimizer's resumable entrypoint end to end: bank-init, targets, train, proposals and outcomes, collect,
propose, ladder."""
import json
from pathlib import Path

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
    # A tuner plan with already-lowered rows importing the program's initial instruction state (`soft-init.json`).
    plan = write(tmp_path / 'plan.json', {'schema': 'natlang.tune-plan/v1', 'manifest': manifest, 'rows': 'rows.jsonl',
                                          'student': {'checkpoint': checkpoint}, 'softStates': True})
    jsonl(tmp_path / 'rows.jsonl', rows)
    (tmp_path / 'work' / 'program-0').mkdir(parents=True)
    write(tmp_path / 'work' / 'program-0' / 'soft-init.json',
          {key: {'state': incoming, 'binding': 'bx', 'kind': 'instructions', 'target': None, 'textSha256': 'x', 'tokens': 2}})
    targets_spec = write(tmp_path / 'targets-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'out': str(tmp_path / 'targets.jsonl'),
                                                          'work': str(tmp_path / 'work'), 'fit': {'steps': 2, 'refine_steps': 1},
                                                          'programs': [{'plan': plan, 'root': str(tmp_path / 'program'),
                                                                        'scopes': {'lambda': 2}, 'subsets': 2}]})
    assert main(['targets', '--spec', targets_spec]) == 0
    labeled = [json.loads(line) for line in (tmp_path / 'targets.jsonl').read_text().splitlines()]
    assert {item['tier'] for item in labeled} == {'in-bank-fit'} and all(item['id'].startswith('njt1:') for item in labeled)
    assert main(['targets', '--spec', targets_spec]) == 0  # finished episodes are skipped on rerun
    assert len((tmp_path / 'targets.jsonl').read_text().splitlines()) == len(labeled)
    train_spec = write(tmp_path / 'train-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'targets': str(tmp_path / 'targets.jsonl'),
                                                      'state': str(tmp_path / 'state.json'), 'receipt': str(tmp_path / 'receipt.json'), 'steps': 1,
                                                      'recipe': {'writer': {'kind': 'free-lora', 'rank': 2, 'alpha': 2.}, 'batch_targets': 1}})
    assert main(['train', '--spec', train_spec]) == 0
    assert json.loads((tmp_path / 'state.json').read_text())['step'] == 1
    assert main(['train', '--spec', train_spec]) == 0  # resumes
    assert json.loads((tmp_path / 'receipt.json').read_text())['step'] == 2
    collect_spec = write(tmp_path / 'collect-spec.json', {**common, 'bank': str(tmp_path / 'bank.json'), 'state': str(tmp_path / 'state.json'),
                                                          'targets': str(tmp_path / 'targets.jsonl'), 'perContext': 2,
                                                          'out': str(tmp_path / 'records.jsonl'), 'summary': str(tmp_path / 'collect.json')})
    assert main(['collect', '--spec', collect_spec]) == 0
    records = [json.loads(line) for line in (tmp_path / 'records.jsonl').read_text().splitlines()]
    assert len(records) == 2 * len(labeled) and all(item['reward']['task'] is None and item['reward']['shaping'] is not None for item in records)
    # Real task outcomes: proposals become release fragments with one scoring request per program root; `collect`
    # then reads the proposals back with the outcomes `natlang tune --score-proposals` writes.
    proposals_spec = write(tmp_path / 'proposals-spec.json', {**json.loads(Path(collect_spec).read_text()), 'out': str(tmp_path / 'proposals.jsonl'),
                                                              'releases': str(tmp_path / 'releases')})
    assert main(['proposals', '--spec', proposals_spec]) == 0
    request_path = tmp_path / 'releases' / 'requests' / 'program-0.json'
    request = json.loads(request_path.read_text())
    assert request['root'] == str((tmp_path / 'program').resolve()) and len(request['targets']) == len(labeled)
    for entry in request['targets']:
        assert entry['support'] and len(entry['proposals']) == 2
        for release in [entry['incoming'], *(item['release'] for item in entry['proposals'])]:
            result = json.loads((Path(release) / 'result.json').read_text())
            assert result['schema'] == 'natlang.tune.build-result/1' and result['artifacts']
            assert all((Path(release) / 'artifacts' / f"{item['artifact']}.nnt").exists() for item in result['artifacts'])
            assert all((Path(release) / result['softStateFiles'][item['state']]).exists() for item in result['softStates'])
    sampled = [json.loads(line) for line in (tmp_path / 'proposals.jsonl').read_text().splitlines()]
    outcomes = {item['bundle']: {'before': .25, 'after': .75} for item in sampled[:2]}
    scored_spec = write(tmp_path / 'scored-spec.json', {'proposals': str(tmp_path / 'proposals.jsonl'), 'outcomes': outcomes,
                                                        'out': str(tmp_path / 'scored.jsonl')})
    assert main(['collect', '--spec', scored_spec]) == 0
    scored = [json.loads(line) for line in (tmp_path / 'scored.jsonl').read_text().splitlines()]
    assert len(scored) == len(sampled) and [item['bundle'] for item in scored] == [item['bundle'] for item in sampled]
    assert [item['reward']['task'] for item in scored[:2]] == [.5, .5] and all(item['reward']['task'] is None for item in scored[2:])
    policy_spec = write(tmp_path / 'policy-spec.json', {**json.loads((tmp_path / 'train-spec.json').read_text()), 'steps': 0,
                                                        'policy_records': [str(tmp_path / 'records.jsonl')]})
    assert main(['train', '--spec', policy_spec]) == 0
    assert json.loads((tmp_path / 'receipt.json').read_text())['step'] == 3
    context = json.loads((tmp_path / 'targets.jsonl').read_text().splitlines()[0])
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
