import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks'))
import authored_novelty  # noqa: E402
import authoring_rotation  # noqa: E402


def case(ident, family, program, inputs):
    return {'id': ident, 'split': 'train', 'family': family, 'generation': {'generator': 'natlang.rl-authored-tasks/1'},
            'curriculum': {'mode': 'single_call', 'rl': {'level': 1, 'family': family}},
            'semantics': {'root': 'f.nl', 'files': {'f.nl': '---\nargs: {}\nreturns: "string"\n---\n' + program},
                          'inputs': inputs, 'expected': 'x', 'oracle': 'exact'}}


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def test_same_program_on_different_worlds_is_not_a_duplicate_but_literal_swaps_are(tmp_path):
    program = 'Reconcile the ledger from messages, then return the stock record.'
    pool = write(tmp_path / 'pool.jsonl', [case(f'p{i}', 'ledger', 'pool program ' + 'abcdefghijklmnopqrstu'[i] * (i + 1), {'m': i})
                                           for i in range(20)])
    batch = write(tmp_path / 'batch.jsonl', [
        case('a', 'ledger', program, {'messages': ['Delivery of 12 oat milk cases arrived crushed and was refused at the door']}),
        case('b', 'ledger', program, {'messages': ['The bakery returned two bags of flour that were the wrong grind']}),
        # Same text as 'a' up to digits: a literal swap, counted as an exact duplicate.
        case('c', 'ledger', program, {'messages': ['Delivery of 15 oat milk cases arrived crushed and was refused at the door']}),
    ])
    out = tmp_path / 'novelty.json'
    assert authored_novelty.main([batch, '--pool', pool, '-o', str(out)]) == 1
    report = json.loads(out.read_text())
    assert report['exact_duplicates'] == ['c']
    assert report['families']['ledger']['batch_tasks'] == 3 and report['families']['ledger']['batch_new_programs'] == 1
    batch = write(tmp_path / 'batch2.jsonl', [
        case('a', 'ledger', program, {'messages': ['Delivery of oat milk arrived crushed and was refused at the door']}),
        case('b', 'ledger', program, {'messages': ['The bakery returned two bags of flour that were the wrong grind']})])
    assert authored_novelty.main([batch, '--pool', pool, '-o', str(out)]) == 0


def test_claims_reserve_distinct_cells_record_finalizes_and_claims_expire(tmp_path, capsys):
    ledger = tmp_path / 'rotation.jsonl'
    lists = {}
    for name, values in (('datasets', ['ds1', 'ds2']), ('constructs', ['relay']), ('problems', ['p1', 'p2'])):
        lists[name] = tmp_path / f'{name}.txt'
        lists[name].write_text('\n'.join(values) + '\n')
    base = ['--ledger', str(ledger), '--datasets', str(lists['datasets']), '--constructs', str(lists['constructs']),
            '--problems', str(lists['problems'])]
    # Make dataset mode the next mode: one recorded invented batch.
    ledger.write_text(json.dumps({'unix': 0, 'status': 'recorded', 'dataset': 'invented:apiary', 'construct': 'relay',
                                  'problem': 'p1', 'worker': 'w0', 'batch': 'b0', 'admitted': 3}) + '\n')
    picks = []
    for worker, batch in (('w1', 'b1'), ('w2', 'b2')):
        capsys.readouterr()
        authoring_rotation.main(['next', *base, '--worker', worker, '--claim', '--batch', batch])
        picks.append(json.loads(capsys.readouterr().out))
    assert {(p['dataset'], p['problem']) for p in picks} == {(picks[0]['dataset'], picks[0]['problem']),
                                                             (picks[1]['dataset'], picks[1]['problem'])}
    assert (picks[0]['dataset'], picks[0]['problem']) != (picks[1]['dataset'], picks[1]['problem'])
    authoring_rotation.main(['record', '--ledger', str(ledger), '--dataset', picks[0]['dataset'], '--construct', 'relay',
                             '--problem', picks[0]['problem'], '--worker', 'w1', '--batch', 'b1', '--admitted', '12'])
    capsys.readouterr()
    authoring_rotation.main(['next', *base, '--worker', 'w3'])
    after = json.loads(capsys.readouterr().out)
    assert after['admitted_by_construct'] == {'relay': 15} and after['active_claims'] == 1
    capsys.readouterr()
    authoring_rotation.main(['next', *base, '--worker', 'w3', '--claim-hours', '0'])
    assert json.loads(capsys.readouterr().out)['active_claims'] == 0
