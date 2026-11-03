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


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks' / 'sources'))
import authored  # noqa: E402


def folder_task(**changes):
    task = {'schema': 'natlang.rl-authored-task/1', 'id': 'returns-0001', 'author': {'model': 'm', 'worker': 'w'},
            'family': 'returns_sorting', 'world': 'returns-w44', 'level': 1,
            'constructs': ['folder_reducer', 'decision_readout', 'effect_state'], 'oracle_depth': 'effect-state',
            'root': 'sort_returns.nl',
            'files': {'sort_returns.nl': '---\nkind: directory-reducer\nargs: {}\nreturns: "number"\n---\nSort returns/.\n',
                      'sort_returns/assess.nl': '---\nreadout: decision\nargs:\n  note: "string"\nreturns: "\\"a\\" | \\"b\\""\n---\nDecide.\n'},
            'inputs': {}, 'folders': {'returns/r1.md': 'one\n', 'returns/r2.md': 'two\n'},
            'expected_folders': {'a/r1.md': 'one\n', 'b/r2.md': 'two\n'}, 'expected': 2, 'oracle': 'exact',
            'wrong_answers': [1, 3], 'alias_answers': [],
            'reference': {'root': [['eval', {'code': 'return 2;', 'finish': True}]],
                          'children': [{'match': ['one'], 'value': 'a'}, {'match': ['two'], 'value': 'b'}]},
            'wrong_code': ['return 2;'],
            'smoke': {'code': 'return 2;', 'children': {'assess': {'$match': [{'match': 'one', 'value': 'a'}]}}}}
    task.update(changes)
    return task


def test_authored_folders_become_the_reducer_world_and_its_effect_oracle():
    case = authored.convert(folder_task())
    semantics = case['semantics']
    assert semantics['folder_files'] == {'returns/r1.md': 'one\n', 'returns/r2.md': 'two\n'}
    assert semantics['expected_files'] == {'a/r1.md': 'one\n', 'b/r2.md': 'two\n'}
    assert semantics['files_oracle'] == {'compare': 'content', 'threshold': 1}
    assert case['curriculum']['reference']['children'][1] == {'match': ['two'], 'value': 'b'}
    assert case['curriculum']['external']['checks']['wrong_code'] == ['return 2;']
    plain = authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[],
                                         constructs=['decision_readout'], oracle_depth='answer',
                                         reference=None))
    assert 'folder_files' not in plain['semantics'] and 'wrong_code' not in plain['curriculum']['external']['checks']
    assert plain['curriculum']['reference'] == {'root': [['return_result', {'status': 'success', 'value': 2}]]}


def test_authored_folders_are_rejected_without_a_directory_reducer_root_or_with_unsafe_paths():
    import pytest
    bad = [folder_task(constructs=['decision_readout', 'effect_state']),
           folder_task(files={**folder_task()['files'],
                              'sort_returns.nl': '---\nargs: {}\nreturns: "number"\n---\nSort.\n'}),
           folder_task(folders={'../escape.md': 'x'}),
           folder_task(folders={'/abs.md': 'x'}),
           folder_task(folders=None),
           folder_task(oracle_depth='answer', constructs=['folder_reducer'])]
    for task in bad:
        with pytest.raises(ValueError):
            authored.convert(task)
