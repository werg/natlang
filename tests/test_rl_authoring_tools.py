import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks'))
import authored_novelty  # noqa: E402
import authoring_rotation  # noqa: E402
import blind_solve  # noqa: E402


def case(ident, family, program, inputs):
    return {'id': ident, 'split': 'train', 'family': family, 'generation': {'generator': 'natlang.rl-authored-tasks/1'},
            'curriculum': {'mode': 'single_call', 'rl': {'level': 1, 'family': family}},
            'semantics': {'root': 'f.nl', 'files': {'f.nl': '---\nargs: {}\nreturns: "string"\n---\n' + program},
                          'inputs': inputs, 'expected': 'x', 'oracle': 'exact'}}


def write(path, rows):
    path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    return str(path)


def test_same_program_on_different_worlds_is_not_a_duplicate_but_literal_swaps_are(tmp_path):
    program = 'Reconcile the ledger from messages, then return the stock record with `return applyMoves(stock, moves)`.'
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


def test_prose_apostrophes_are_not_string_literals():
    from measure_repetition import normalize
    one = normalize("Check the lane's limit. Keep Bob's notes and the driver's log apart.")
    two = normalize("Check the lane's limit. Merge Bob's notes into the driver's log.")
    assert one != two and "Keep Bob's notes" in one
    assert normalize("Route 'east' traffic") == normalize("Route 'west' traffic")


def functional_program(name, helper='tally', text='For each item call `await tally(item)` and sum the results.',
                       root_name='count', returns='number'):
    files = {f'{root_name}.nl': '---\nargs:\n  items: "string[]"\nreturns: "' + returns + '"\n---\n' + text + '\n',
             f'{root_name}/{helper}.nl': '---\nargs:\n  item: "string"\nreturns: "number"\n---\nScore the item.\n'}
    row = case(name, 'tally', '', {'items': [name]})
    row['semantics'].update(root=f'{root_name}.nl', files=files)
    return row


def test_renamed_roots_are_one_functional_program_and_new_helpers_are_new_programs(tmp_path):
    pool = write(tmp_path / 'pool.jsonl', [functional_program('p0')])
    out = tmp_path / 'novelty.json'
    renamed = write(tmp_path / 'renamed.jsonl', [functional_program(f'q{i}', root_name=f'count_{name}')
                                                 for i, name in enumerate(['crates', 'bins', 'hives', 'rows'])])
    assert authored_novelty.main([renamed, '--pool', pool, '-o', str(out)]) == 1
    report = json.loads(out.read_text())
    assert report['functional_repeats'] == ['q0', 'q1', 'q2', 'q3']
    assert report['families']['tally']['batch_new_functional'] == 0
    real = write(tmp_path / 'real.jsonl', [functional_program(f'r{i}', helper) for i, helper in
                                           enumerate(['weigh', 'rank', 'price', 'grade', 'audit'])])
    assert authored_novelty.main([real, '--pool', pool, '-o', str(out)]) == 0
    assert json.loads(out.read_text())['families']['tally']['batch_new_functional'] == 5


def test_functional_programs_see_types_and_rule_sentences_but_not_literals():
    from measure_repetition import FunctionalPrograms, functional_signature, functional_words
    base = functional_program('a')
    other_type = functional_program('b')
    other_type['semantics']['files']['count/tally.nl'] = other_type['semantics']['files']['count/tally.nl'].replace(
        '"number"', '"{ points: number }"')
    assert functional_signature(base) != functional_signature(other_type)
    root_type = functional_program('c', returns='{ total: number }')  # the root's answer shape is prose-carried
    assert functional_signature(base) == functional_signature(root_type)
    rule = 'Score every item. ' * 6 + 'An item named "x" counts double. ' + 'Sum the scores. ' * 6
    other_rule = rule.replace('counts double', 'counts nothing and is skipped entirely')
    literal = rule.replace('"x"', '"y"')
    programs = FunctionalPrograms()
    for name, text in (('a', rule), ('b', other_rule), ('c', literal)):
        row = functional_program(name, text=text)
        assert programs.add('tally', functional_signature(row), functional_words(row)) == (name != 'c')
    assert programs.counts['tally'] == 2


# The auditors' distinct-program counts for the 12 audited round-1 batches (plans/neuralese/RL_AUTHORING_AUDIT_ROUND1_20261103.md).
AUDITED_PROGRAMS = {'w01/b0001': 10, 'w01/b0002': 6, 'w02/b0001': 12, 'w02/b0002': 7, 'w03/b0001': 4, 'w03/b0002': 3,
                    'w04/b0001': 8, 'w04/b0002': 5, 'w05/b0001': 14, 'w05/b0002': 1, 'w06/b0001': 11, 'w06/b0002': 8}


def test_functional_program_counts_match_the_audited_batches():
    import glob
    import pytest
    from measure_repetition import FunctionalPrograms, program_features
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese'))
    from natlang_neuralese.common.paths import resolve
    audited = resolve('data_hdd', 'natlang-rl-authoring', 'batches')
    if not audited.is_dir():
        pytest.skip('audited authoring batches are not on this machine')
    within, total = 0, 0
    for batch, truth in AUDITED_PROGRAMS.items():
        found = sorted(glob.glob(f'{audited}/{batch}/audited/qualified*/cases.jsonl')) or \
            sorted(glob.glob(f'{audited}/{batch}/qualified*/cases.jsonl'))
        if not found:
            pytest.skip('audited authoring batches are incomplete on this machine')
        programs = FunctionalPrograms()
        counted = sum(programs.add(f['family'], f['functional'], f['functional_words'])
                      for f in (program_features(json.loads(line)) for line in open(found[-1]) if line.strip()))
        within += 0.7 <= counted / truth <= 1.3
        total += counted
    assert within >= 10 and abs(total - sum(AUDITED_PROGRAMS.values())) <= 0.3 * sum(AUDITED_PROGRAMS.values())


def test_outputs_are_written_atomically_and_failures_are_loud(tmp_path):
    import pytest
    from measure_repetition import write_text_atomic
    target = tmp_path / 'out.json'
    write_text_atomic(target, '{}\n')
    assert target.read_text() == '{}\n' and not list(tmp_path.glob('*.tmp*'))
    with pytest.raises(OSError):
        write_text_atomic(tmp_path / 'missing-dir' / 'out.json', '{}\n')
    pool = write(tmp_path / 'pool.jsonl', [functional_program('p0')])
    batch = write(tmp_path / 'batch.jsonl', [functional_program('q0')])
    with pytest.raises(OSError):
        authored_novelty.main([batch, '--pool', pool, '-o', str(tmp_path / 'missing-dir' / 'novelty.json')])


def test_rotation_creates_a_missing_ledger(tmp_path, capsys):
    lists = {}
    for name in ('datasets', 'constructs', 'problems'):
        lists[name] = tmp_path / f'{name}.txt'
        lists[name].write_text('x\n')
    ledger = tmp_path / 'new' / 'rotation.jsonl'
    assert authoring_rotation.main(['next', '--ledger', str(ledger), *[f'--{k}={v}' for k, v in lists.items()],
                                    '--worker', 'w', '--claim', '--batch', 'b']) in (0, None)
    assert ledger.exists() and json.loads(ledger.read_text().splitlines()[0])['status'] == 'claimed'


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


def test_rotation_credits_every_genuine_construct(tmp_path, capsys):
    ledger = tmp_path / 'rotation.jsonl'
    lists = {}
    for name, values in (('datasets', ['ds1']), ('constructs', ['call_depth_3', 'iterate', 'skills']), ('problems', ['p1'])):
        lists[name] = tmp_path / f'{name}.txt'
        lists[name].write_text('\n'.join(values) + '\n')
    cases = tmp_path / 'admitted.jsonl'
    cases.write_text(''.join(json.dumps({'curriculum': {'external': {'constructs': constructs}}}) + '\n'
                             for constructs in (['call_depth_3', 'iterate'], ['call_depth_3'], ['call_depth_3', 'iterate'])))
    authoring_rotation.main(['record', '--ledger', str(ledger), '--dataset', 'ds1', '--construct', 'call_depth_3',
                             '--problem', 'p1', '--worker', 'w', '--batch', 'b', '--credit-from', str(cases)])
    entry = json.loads(capsys.readouterr().out)
    assert entry['admitted'] == 3 and entry['credits'] == {'call_depth_3': 3, 'iterate': 2}
    authoring_rotation.main(['next', '--ledger', str(ledger), *[f'--{k}={v}' for k, v in lists.items()], '--worker', 'w'])
    assert json.loads(capsys.readouterr().out)['admitted_by_construct'] == {'call_depth_3': 3, 'iterate': 2}
    authoring_rotation.main(['record', '--ledger', str(ledger), '--dataset', 'ds1', '--construct', 'skills',
                             '--problem', 'p1', '--worker', 'w', '--batch', 'b2', '--admitted', '4'])
    capsys.readouterr()
    authoring_rotation.main(['next', '--ledger', str(ledger), *[f'--{k}={v}' for k, v in lists.items()], '--worker', 'w'])
    assert json.loads(capsys.readouterr().out)['admitted_by_construct']['skills'] == 4


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts' / 'rl_tasks'))
import authored_checks  # noqa: E402


def nl(args='', returns='number', body='Do it.', readout=False, kind=None):
    head = ('readout: decision\n' if readout else '') + (f'kind: {kind}\n' if kind else '')
    argtext = 'args: {}\n' if not args else 'args:\n' + ''.join(f'  {a}: "string"\n' for a in args.split(','))
    return f'---\n{head}{argtext}returns: "{returns}"\n---\n{body}\n'


def task(**changes):
    value = {'id': 'desk-0001', 'root': 'desk.nl', 'world': 'w', 'constructs': [],
             'files': {'desk.nl': nl('items', body='Total the items.')}}
    value.update(changes)
    return value


def findings(*tasks):
    return authored_checks.lint_batch(list(tasks))


def test_folder_reducer_needs_the_tasks_own_folders():
    bare = task(constructs=['folder_reducer'], files={'desk.nl': nl(kind='directory-reducer')})
    assert any('needs `folders`' in e for e in findings(bare)['desk-0001']['errors'])
    ok = dict(bare, folders={'inbox/a.md': 'x'})
    assert findings(ok)['desk-0001']['errors'] == []
    assert any('directory-reducer' in e for e in findings(task(files={'desk.nl': nl(kind='directory-reducer')}))['desk-0001']['errors'])


def test_decision_readouts_are_leaves_that_call_nothing():
    files = {'desk.nl': nl('items', body='Judge with `await decide(judge, items)`.'),
             'desk/judge.nl': nl('items', '\\"a\\" | \\"b\\"', 'Pick; use `await lookup(items)` if unsure.', readout=True),
             'desk/lookup.nl': nl('items', body='Look it up.')}
    errors = findings(task(files=files, constructs=['decision_readout']))['desk-0001']['errors']
    assert any('judge' in e and 'calls lookup' in e for e in errors)
    files = {**files, 'desk/judge.nl': nl('items', '\\"a\\" | \\"b\\"', 'Pick.', readout=True),
             'desk/judge/deeper.nl': nl('items')}
    assert any('owns a callable folder' in e for e in findings(task(files=files))['desk-0001']['errors'])
    del files['desk/judge/deeper.nl']
    assert findings(task(files=files))['desk-0001']['errors'] == []


def test_skills_must_sit_where_the_runtime_binds_them():
    skill = '---\nname: rules\ndescription: d\n---\nRule.\n'
    files = {'desk.nl': nl('items', body='Call `await grade(items)`.'),
             'desk/grade.nl': nl('items', body='Read `await read_code({ name: "skills.rules" })` and grade.'),
             'desk/skills/rules/SKILL.md': skill}
    errors = findings(task(files=files, constructs=['skills']))['desk-0001']['errors']
    assert any('rules is bound to desk.nl' in e and 'desk/grade.nl' in e for e in errors)
    files = {**files, 'desk/grade/skills/rules/SKILL.md': skill}
    del files['desk/skills/rules/SKILL.md']
    assert findings(task(files=files, constructs=['skills']))['desk-0001']['errors'] == []
    orphan = {**files, 'desk/nobody/skills/rules/SKILL.md': skill}
    assert any('no function desk/nobody.nl' in e for e in findings(task(files=orphan))['desk-0001']['errors'])


def test_multi_operation_worlds_need_several_roots_over_shared_helpers():
    def operation(ident, root, helper_text='Count.'):
        return task(id=ident, root=f'{root}.nl', world='shop', constructs=['multi_operation_codebase'],
                    files={f'{root}.nl': nl('items', body=f'Use `await count(items)`.'),
                           f'{root}/count.nl': nl('items', body=helper_text)})
    one = findings(operation('t1', 'close_day'), operation('t2', 'close_day'))
    assert all(any('one root operation' in e for e in v['errors']) for v in one.values())
    two = findings(operation('t1', 'close_day'), operation('t2', 'trial_balance'))
    assert all(not v['errors'] and not v['warnings'] for v in two.values())
    unshared = findings(operation('t1', 'close_day'), operation('t2', 'trial_balance', 'Other.'))
    assert all(any('no helper file' in e and 'identical' in e for e in v['errors']) for v in unshared.values())


def test_iterate_needs_an_explicit_stop_and_a_real_iterateOn():
    loop = 'Go through items with iterateOn and add each.'
    errors = findings(task(files={'desk.nl': nl('items', body=loop)}, constructs=['iterate']))['desk-0001']['errors']
    assert any('without an explicit stopping condition' in e for e in errors)
    stopped = findings(task(files={'desk.nl': nl('items', body=loop + ' Stop when every item is added.')},
                            constructs=['iterate']))
    assert stopped['desk-0001']['errors'] == []
    assert any('no function uses iterateOn' in e for e in findings(task(constructs=['iterate']))['desk-0001']['errors'])


def test_call_depth_needs_a_chain_and_warns_on_pass_through_middles():
    flat = task(constructs=['call_depth_3'], files={'desk.nl': nl('items', body='Call `await mid(items)`.'),
                                                    'desk/mid.nl': nl('items', body='Total.')})
    assert any('needs a root -> helper -> sub-helper chain' in e for e in findings(flat)['desk-0001']['errors'])
    work = ' '.join(f'Step {i} of the middle helper does real work on the items.' for i in range(6))
    chain = lambda mid_body, mid_returns='number': task(constructs=['call_depth_3'], files={  # noqa: E731
        'desk.nl': nl('items', body='Call `await mid(items)`.'), 'desk/mid.nl': nl('items', mid_returns, mid_body),
        'desk/mid/leaf.nl': nl('items', body='Score the items.')})
    assert findings(chain(work + ' Then `await leaf(items)` and combine.', '{ a: number }'))['desk-0001'] == \
        {'errors': [], 'warnings': []}
    pass_through = findings(chain('Call `await leaf(items)` and return it.'))['desk-0001']
    assert pass_through['errors'] == [] and all(w.startswith('heuristic:') for w in pass_through['warnings'])
    assert len(pass_through['warnings']) == 1 and 'pass-through' in pass_through['warnings'][0]
    never = findings(chain('Total the items by hand.'))['desk-0001']['errors']
    assert any('never calls its sub-helper leaf' in e for e in never)


def test_helper_calls_in_instructions_match_the_declared_argument_count():
    files = {'desk.nl': nl('items', body='Call `const r = await grade(items, items, items);` then `await decide(pick, items)`.'),
             'desk/grade.nl': nl('items,policy', body='Grade.'),
             'desk/pick.nl': nl('items', '\\"a\\" | \\"b\\"', 'Pick.', readout=True)}
    errors = findings(task(files=files))['desk-0001']['errors']
    assert errors == ['desk.nl: grade is called with 3 argument(s) but declares 2']
    assert authored_checks.arity_problems('await grade(a, b)', {'grade': (2, 2)}) == []
    assert authored_checks.arity_problems('await decide(pick, a, b)', {'pick': (1, 1)}) != []
    assert authored_checks.arity_problems('await grade({ a: [1, 2], b: "x, y" })', {'grade': (1, 1)}) == []


def test_smoke_and_reference_scripts_get_the_same_argument_count_check():
    import shutil
    import subprocess
    node = sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True) or [shutil.which('node')]
    package = Path(__file__).resolve().parents[1] / 'ts-host' / 'package.json'
    if not node[0] or not (package.parent / 'node_modules' / 'typescript').exists():
        import pytest
        pytest.skip('node or typescript is not available')
    script = """
import { createRequire } from 'node:module';
import { helperArities, arityProblems } from '%s/scripts/rl_tasks/script_arity.mjs';
const ts = createRequire('%s')('typescript');
const readNatlangFrontmatter = () => ({ args: { items: 'string', 'policy?': 'string' } });
const row = { semantics: { root: 'desk.nl', files: { 'desk.nl': '', 'desk/grade.nl': '---\\n---', 'desk/grade/deep.nl': '---\\n---' } } };
const arities = helperArities(row, { readNatlangFrontmatter });
const tools = { ts };
console.log(JSON.stringify([arities, arityProblems('const a = await grade(x, y, z);', arities, tools),
  arityProblems('const a = await grade(x); const b = await decide(grade, x, y);', arities, tools),
  arityProblems('const a = await grade();', arities, tools)]));
""" % (Path(__file__).resolve().parents[1], package)
    out = subprocess.run([str(node[0]), '--input-type=module', '-e', script], capture_output=True, text=True, check=True)
    arities, three, ok, none = json.loads(out.stdout)
    assert arities == {'grade': [1, 2]}
    assert three == ['grade is called with 3 argument(s) but declares 1 to 2']
    assert ok == [] and none == ['grade is called with 0 argument(s) but declares 1 to 2']


def test_authored_tasks_may_claim_no_construct_and_aliases_are_accepted_by_the_oracle():
    import pytest
    plain = authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[], constructs=[],
                                         oracle_depth='answer', reference=None, alias_answers=[2.0]))
    assert plain['curriculum']['external']['constructs'] == []
    assert plain['semantics']['oracle'] == {'level': 'exact', 'alternates': [2.0]}
    unaliased = authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[], constructs=[],
                                             oracle_depth='answer', reference=None))
    assert unaliased['semantics']['oracle'] == 'exact'
    with pytest.raises(ValueError, match='alias'):
        authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[], constructs=[],
                                     oracle_depth='answer', reference=None, alias_answers=[3]))
    with pytest.raises(ValueError):
        authored.convert(folder_task(constructs='folder_reducer'))


def test_blind_solve_packets_hide_answers_and_compare_uses_the_oracle(tmp_path):

    task = {'id': 't', 'root': 'f.nl', 'files': {'f.nl': '---\nargs:\n  x: "number"\nreturns: "{ n: number; s: string }"\n---\nbody'},
            'inputs': {'x': 1}, 'expected': {'n': 2, 's': 'Ok'}, 'oracle': 'normalized', 'alias_answers': [{'n': 2, 's': 'fine'}],
            'wrong_answers': [{'n': 3, 's': 'Ok'}], 'groups': ['g'], 'smoke': {'code': 'x'}}
    row = blind_solve.packet_row(task)
    assert row['returns'] == '{ n: number; s: string }' and set(row) == {'id', 'root', 'files', 'inputs', 'returns', 'answer_format'}
    ok = blind_solve.compare([task], {'t': {'answer': {'s': ' ok', 'n': 2.0}}})
    assert not ok['disagreements'] and ok['agree'] == 1
    assert blind_solve.compare([task], {'t': {'answer': {'n': 2, 's': 'fine'}}})['agree'] == 1
    assert blind_solve.compare([task], {'t': {'answer': {'n': 3, 's': 'Ok'}}})['disagreements'][0]['differs'] == ['answer']
    assert blind_solve.compare([task], {})['missing'] == ['t']
    exact = dict(task, oracle='exact', alias_answers=[])
    assert blind_solve.compare([exact], {'t': {'answer': {'n': 2, 's': 'ok'}}})['disagreements']
    folder = dict(exact, folders={'a/x.md': 'T'}, expected_folders={'b/x.md': 'T'})
    assert blind_solve.compare([folder], {'t': {'answer': {'n': 2, 's': 'Ok'}, 'folders': {'a/x.md': 'T'}}})['disagreements'][0]['differs'] == ['folders']


def test_admission_appends_atomically_and_refuses_clashes_failures_and_unapproved_batches(tmp_path):
    import admit_authored
    batch, pool = tmp_path / 'b', tmp_path / 'pool'
    (batch / 'q' / 'verified').mkdir(parents=True)
    rows = [{'id': 'c1', 'split': 'train'}, {'id': 'c2', 'split': 'dev'}]
    write(batch / 'q' / 'verified' / 'cases.jsonl', rows)
    status = {'c1': {'task': 't1', 'problems': []}, 'c2': {'task': 't2', 'problems': []}}
    (batch / 'q' / 'report.json').write_text(json.dumps({'result': 'PASS', 'tasks_status': status}))
    (batch / 'nov.json').write_text(json.dumps({'result': 'PASS'}))
    out = admit_authored.admit(batch, 'q', 'nov.json', ['t2'], [], 'tester', '', pool)
    assert out['admitted'] == 1 and out['held'] == 1
    assert [json.loads(line)['id'] for line in (pool / 'cases.jsonl').read_text().splitlines()] == ['c1']
    log = json.loads((pool / 'admissions.jsonl').read_text().splitlines()[0])
    assert log['admitted_case_ids'] == ['c1'] and log['dropped_tasks'] == ['t2'] and log['reviewer'] == 'tester'
    try:
        admit_authored.admit(batch, 'q', 'nov.json', ['t2'], [], 'tester', '', pool)
        raise AssertionError('duplicate admission accepted')
    except SystemExit as error:
        assert 'already admitted' in str(error)
    assert admit_authored.admit(batch, 'q', 'nov.json', ['t2'], ['c1'], 'tester', '', pool)['superseded'] == 1
    (batch / 'nov.json').write_text(json.dumps({'result': 'FAIL'}))
    try:
        admit_authored.admit(batch, 'q', 'nov.json', [], [], 'tester', '', pool)
        raise AssertionError('failed novelty admitted')
    except SystemExit as error:
        assert 'novelty' in str(error)


def test_folder_worlds_are_part_of_a_tasks_identity(tmp_path):
    base = case('a', 'reducers', 'Sort the files in the folder with `file.moveTo("done/")`.', {})
    other = json.loads(json.dumps(base))
    other['id'] = 'b'
    base['semantics']['folder_files'] = {'inbox/x.md': 'Kettle with an unopened box'}
    other['semantics']['folder_files'] = {'inbox/x.md': 'Toaster with a burnt element'}
    pool = write(tmp_path / 'pool.jsonl', [case(f'p{i}', 'reducers', 'pool ' + 'abcdefghijklmnopqrstu'[i] * (i + 2), {}) for i in range(20)])
    out = tmp_path / 'n.json'
    authored_novelty.main([write(tmp_path / 'b.jsonl', [base, other]), '--pool', pool, '-o', str(out)])
    assert json.loads(out.read_text())['exact_duplicates'] == []
    same = json.loads(json.dumps(base))
    same['id'] = 'c'
    authored_novelty.main([write(tmp_path / 'c.jsonl', [base, same]), '--pool', pool, '-o', str(out)])
    assert json.loads(out.read_text())['exact_duplicates'] == ['c']


def test_neuralese_tasks_need_a_full_smoke_replay_judged_by_the_oracle():
    import pytest
    bare = task(constructs=['neuralese_relay'])
    assert any('smoke.replay' in e for e in findings(bare)['desk-0001']['errors'])
    replayed = dict(bare, smoke={'code': 'const n = await begin(x);\nreturn n.length;', 'children': {}, 'replay': True})
    assert findings(replayed)['desk-0001']['errors'] == []
    literal = dict(bare, smoke={'code': 'const n = await begin(x);\nreturn 3;', 'children': {}, 'replay': True})
    assert any('bare literal' in w for w in findings(literal)['desk-0001']['warnings'])
    case = authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[], constructs=[],
                                        oracle_depth='answer', reference=None,
                                        smoke={'code': 'return 2;', 'children': {}, 'replay': True}))
    smoke = case['curriculum']['external']['structure']['smoke']
    assert smoke['replay'] is True and smoke['root'] == [['eval', {'code': 'return 2;', 'finish': True}]]
    with pytest.raises(ValueError, match='replay'):
        authored.convert(folder_task(folders=None, expected_folders=None, wrong_code=[], constructs=[],
                                     oracle_depth='answer', reference=None,
                                     smoke={'code': 'const a = 1;', 'children': {}, 'replay': True}))


def test_authored_child_notes_only_from_replay_verified_input_specific_notes():
    import authored_child_notes as notes

    def row(children, code, *, replay=True, reference=None):
        files = {'root.nl': '---\nargs: {}\nreturns: "string"\n---\nx',
                 'root/read.nl': '---\nargs:\n  s: "string"\nreturns: "Neuralese<string>"\n---\nWrite a note.'}
        return {'id': 'r', 'split': 'train', 'semantics': {'root': 'root.nl', 'files': files},
                'curriculum': {'reference': reference or {'root': []}, 'external': {'structure': {'smoke': {
                    'replay': replay, 'root': [['eval', {'code': code}]], 'children': children}}}}}
    note_a = 'Okafor holds plot P1 and is first in the queue for P2'
    per_call = {'read': {'$match': [{'match': 'Okafor', 'value': note_a},
                                    {'match': 'Ruiz', 'value': 'Ruiz asks to leave the queue for plot P5 on Friday'}]}}
    got, _ = notes.derive(row(per_call, 'for (const s of items) await read(s);'))
    assert [c['match'][1] for c in got] == ['Okafor', 'Ruiz'] and got[0]['provenance'] == 'checked'
    assert got[0]['match'][0] == 'You are inside this call: read('
    assert notes.derive(row({'read': {'$match': [{'match': 'a', 'value': note_a}, {'match': 'b', 'value': note_a}]}}, ''))[0] == []
    assert notes.derive(row({'read': 'open'}, 'await read(x);'))[0] == []  # a constant is never a note
    once = 'Okafor holds plot P1 and is first in the queue'
    assert len(notes.derive(row({'read': once}, 'const n = await read(x);'))[0]) == 1
    assert notes.derive(row({'read': once}, 'for (const x of xs) await read(x);'))[0] == []  # one value, many calls
    new, why = notes.with_notes(row({'read': once}, 'await read(x);', replay=False))
    assert new['curriculum']['reference'].get('children') is None and why == {'smoke_not_replay_verified': 1}


def test_folder_only_variants_with_numbered_files_are_not_exact_duplicates(tmp_path):
    base = case('a', 'reducers', 'Sort the files in the folder with `file.moveTo("done/")`.', {})
    variants = []
    for ident, files in (('b', {'inbox/01.md': 'Entry 12 kettle sealed', 'inbox/02.md': 'Entry 13 toaster burnt'}),
                         ('c', {'inbox/01.md': 'Entry 14 kettle sealed', 'inbox/02.md': 'Entry 15 toaster burnt'}),
                         ('d', {'inbox/01.md': 'Entry 12 kettle sealed', 'inbox/02.md': 'Entry 13 toaster burnt',
                                'inbox/03.md': 'Entry 16 fan'})):
        row = json.loads(json.dumps(base))
        row['id'] = ident
        row['semantics']['folder_files'] = files
        variants.append(row)
    pool = write(tmp_path / 'pool.jsonl', [case(f'p{i}', 'reducers', 'pool ' + 'abcdefghijklmnopqrstu'[i] * (i + 2), {}) for i in range(20)])
    out = tmp_path / 'n.json'
    authored_novelty.main([write(tmp_path / 'b.jsonl', variants), '--pool', pool, '-o', str(out)])
    # Digit-normalized names and contents used to collapse b, c and d into one key.
    assert json.loads(out.read_text())['exact_duplicates'] == []
    twin = json.loads(json.dumps(variants[0]))
    twin['id'] = 'e'
    authored_novelty.main([write(tmp_path / 'c.jsonl', [variants[0], twin]), '--pool', pool, '-o', str(out)])
    assert json.loads(out.read_text())['exact_duplicates'] == ['e']
