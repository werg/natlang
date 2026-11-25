"""Crisp checks and packaging of invented programs (scripts/rl_tasks/sources/invented.py) and atlas retrieval
(scripts/rl_tasks/program_atlas.py). No runtime: the replay checks (verify.mjs, ablate.mjs) run in qualification."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'scripts' / 'rl_tasks'))

from scripts.rl_tasks.sources import invented  # noqa: E402
import program_atlas  # noqa: E402

ROOT = '''---
args:
  notes: "Untrusted<string>[]"
returns: "Ruling"
---
Call `rule(notes)` and return its result.
'''
TYPES = '''/** The ruling over the notes. */
export type Ruling = {
  /** Ids of the notes that stand. */
  standing: string[];
};
'''
JUDGE = '''---
readout: decision
args:
  note: "Untrusted<string>"
returns: "boolean"
---
Decide whether note withdraws an earlier note.
'''
HELPER = '\n'.join(['import withdraws from "./withdraws.nl";', 'import supersedes from "./supersedes.nl";',
                    'import sameItem from "./same_item.nl";', 'import { semanticFilter } from "@natlang/node/std";',
                    'export async function rule(notes: string[]): Promise<{ standing: string[] }> {',
                    '  const kept = await semanticFilter(notes, withdraws);'] +
                   [f'  const step{i} = kept.length + {i};' for i in range(100)] +
                   ['  return { standing: kept.map(n => n.slice(0, 4)) };', '}', 'export function order(a: string[]): string[] { return [...a].sort(); }'])


PROSE = ('the tenant wrote again about the leaking roof over the back bedroom and said the landlord had promised a '
         'repair before the winter but nobody came and the ceiling stain kept spreading after every storm so she asked '
         'whether the earlier complaint still stood or whether this letter took its place')


def instance(name):
    return {'name': name, 'level': 3, 'inputs': {'notes': [f'n{i} {PROSE}' for i in range(16)]}, 'oracle': 'exact',
            'expected': {'standing': ['n1']},
            'reference': {'root': [['eval', {'code': 'const r = await rule(notes); return r;', 'finish': True}]],
                          'children': [{'match': ['withdraws an earlier'], 'value': False}]},
            'ablations': [{'judgment': j, 'children': [{'match': ['x'], 'value': True}]}
                          for j in ('withdraws', 'supersedes', 'same_item')],
            'wrong_answers': [{'children': [{'match': ['withdraws an earlier', w], 'value': True}], 'mistake': 'm'}
                              for w in ('n2', 'n3', 'n4')],
            'keyword_baseline': 'return { standing: notes.filter(n => n.includes("1")) };'}


def write(tmp, extra=None):
    sub = tmp / 'sub'
    files = {'rule_notes.nl': ROOT, 'types.ts': TYPES, 'rule_notes/withdraws.nl': JUDGE,
             'rule_notes/supersedes.nl': JUDGE.replace('withdraws an earlier', 'replaces'),
             'rule_notes/same_item.nl': JUDGE.replace('withdraws an earlier note', 'names the same item'),
             'rule_notes/rule.ts': HELPER, 'natlang.json': '{}', 'rule_notes.d.nl.ts': 'export {};'}
    for path, text in files.items():
        (sub / 'program' / path).parent.mkdir(parents=True, exist_ok=True)
        (sub / 'program' / path).write_text(text)
    data = {'schema': invented.SCHEMA, 'slug': 'withdrawal-ruling', 'author': {'inventor': 'a', 'builder': 'b'},
            'root': 'rule_notes.nl', 'constructs': ['decide', 'std-collection', 'refinement'],
            'judgments': [{'name': n, 'file': f'rule_notes/{n}.nl', 'trap': 't'} for n in ('withdraws', 'supersedes', 'same_item')],
            'instances': [instance('a'), instance('b')], **(extra or {})}
    (sub / 'submission.json').write_text(json.dumps(data))
    (sub / 'idea.json').write_text(json.dumps({'slug': 'withdrawal-ruling', 'inputs': 'sixteen notes of a dozen words'}))
    (sub / 'NOTES.md').write_text('withdraws: the trap.\n\n## Hardest items\n\n- n1: reads as a withdrawal.\n'
                                  '- n2: replaces rather than withdraws.\n- n3: names the item only by a nickname.\n')
    return sub


def check(sub):
    data, files = invented.load(sub)
    m = invented.metrics(data, files)
    return invented.static_problems(sub, data, files, m), m


def test_complete_submission_passes_static_checks(tmp_path):
    problems, m = check(write(tmp_path))
    assert problems == [], problems
    assert m['call_depth'] == 3 and m['functions'] >= 5
    assert {'decide', 'std-combinator', 'refinement'} <= set(m['constructs'])


def test_generator_script_and_missing_controls_fail(tmp_path):
    sub = write(tmp_path)
    (sub / 'gen.py').write_text('print(1)')
    data = json.loads((sub / 'submission.json').read_text())
    data['instances'][0]['wrong_answers'] = data['instances'][0]['wrong_answers'][:2]
    data['instances'][0]['ablations'] = data['instances'][0]['ablations'][:2]
    data['instances'][1]['keyword_baseline'] = 'return await nl`pick`(notes);'
    data['instances'] = data['instances'] * 2
    (sub / 'submission.json').write_text(json.dumps(data))
    problems, _ = check(sub)
    text = '\n'.join(problems)
    for code in ('N2: gen.py', 'F2: two or three', 'F4: instance a', 'S2: instance a', 'S3: instance b'):
        assert code in text, (code, problems)


def test_the_bar_size_card_size_notes_code_depth_and_untrusted_fields(tmp_path):
    sub = write(tmp_path)
    data = json.loads((sub / 'submission.json').read_text())
    data['instances'][0]['inputs'] = {'notes': ['n1', 'n2']}
    data['instances'][0]['wrong_answers'][0] = {'value': {'standing': []}, 'mistake': 'a value with no mistake to run'}
    (sub / 'submission.json').write_text(json.dumps(data))
    (sub / 'idea.json').write_text(json.dumps({'slug': 'withdrawal-ruling', 'inputs': 'notes, about 1,300 words in all'}))
    (sub / 'NOTES.md').write_text('withdraws: the trap.\n')
    types = TYPES + '''/** A note as filed. */
export type Note = {
  /** The note's id. */
  id: string;
  /** Who wrote it (the clerk's nickname). */
  by: string;
  sourceKey: string;
  text: Untrusted<string>;
};
'''
    (sub / 'program' / 'types.ts').write_text(types)
    (sub / 'program' / 'rule_notes' / 'same_item.nl').write_text(
        JUDGE.replace('note: "Untrusted<string>"', 'note: Note\n  label: string\n  noteId: string'))
    helper = sub / 'program' / 'rule_notes' / 'rule.ts'
    helper.write_text(helper.read_text().replace('\n'.join(f'  const step{i} = kept.length + {i};' for i in range(50, 100)) + '\n', ''))
    problems, _ = check(sub)
    text = '\n'.join(problems)
    assert 'C2: instance a: 0 words of prose in the inputs' in text, problems
    assert 'C2: instance b: 800 words of prose, but the idea card declares 1300' in text, problems
    assert 'C4: NOTES.md lists 0 hardest items' in text and 'D1: ' in text, problems
    assert 'F4: instance a: wrong answer 1 declares no mistake to run' in text, problems
    flagged = [p.split(' is a plain string')[0] for p in problems if p.startswith('S6')]
    assert flagged == ['S6: program/rule_notes/same_item.nl: note (Note).by', 'S6: program/rule_notes/same_item.nl: label'], problems


def test_package_one_case_per_instance_sharing_a_world(tmp_path):
    data, files = invented.load(write(tmp_path))
    cases = invented.package(data, files)
    assert len(cases) == 2
    assert {c['curriculum']['split_group'] for c in cases} == {'invented:withdrawal-ruling'}
    assert len({c['split'] for c in cases}) == 1
    case = cases[0]
    assert case['family'] == 'inv_withdrawal_ruling'
    assert case['curriculum']['external']['checks']['keyword_baseline_code']
    assert len(case['curriculum']['external']['checks']['ablations']) == 3
    assert 'natlang.json' not in case['semantics']['files'] and 'rule_notes/rule.ts' in case['semantics']['files']


def test_near_ranks_the_overlapping_entry_first():
    entries = [{'id': 'a', 'line': 'tide table reconciliation of harbour pilot bookings'},
               {'id': 'b', 'line': 'merge conflicting wiki edits by intent'},
               {'id': 'c', 'line': 'grade transcripts against degree requirements'}]
    idea = {'slug': 'pilot-booking-tides', 'pitch': 'reconcile harbour pilot bookings with the tide table'}
    assert program_atlas.near(entries, idea, 2)[0]['id'] == 'a'


def trace_for(cases):
    """A reference trace as scripts/rl_tasks/reference_trace.mjs writes it: two leaf judgment calls per case (one twice),
    one call with a function argument and one call of a judgment that has a callable folder of its own."""
    results = []
    for case in cases:
        children = [{'path': 'rule_notes/withdraws.nl', 'args': {'note': 'n1 text'}, 'value': False, 'status': 'success',
                     'returned': True, 'functional': False},
                    {'path': 'rule_notes/withdraws.nl', 'args': {'note': 'n1 text'}, 'value': False, 'status': 'success',
                     'returned': True, 'functional': False},
                    {'path': 'rule_notes/supersedes.nl', 'args': {'note': 'n2 text'}, 'value': True, 'status': 'success',
                     'returned': True, 'functional': False},
                    {'path': 'rule_notes/same_item.nl', 'args': {'note': {'$lambda': {}}}, 'value': True,
                     'status': 'success', 'returned': True, 'functional': True}]
        results.append({'id': case['id'], 'calls': 7, 'children': children})
    return {'results': results}


def test_cases_declare_twice_their_reference_calls(tmp_path):
    data, files = invented.load(write(tmp_path))
    cases = invented.with_reference_calls(invented.package(data, files), trace_for(invented.package(data, files)))
    assert [c['semantics']['episode_model_calls'] for c in cases] == [14, 14]


def test_stage_tasks_one_per_distinct_leaf_judgment_call(tmp_path):
    data, files = invented.load(write(tmp_path))
    cases = invented.package(data, files)
    stages = invented.stage_cases(cases, trace_for(cases))
    # per case: withdraws once (the repeated call is one task), supersedes once; the functional call is left out
    assert len(stages) == 4
    first = stages[0]
    assert first['semantics']['root'] == 'rule_notes/withdraws.nl'
    assert first['semantics']['inputs'] == {'note': 'n1 text'} and first['semantics']['expected'] is False
    assert first['curriculum']['reference']['root'] == [['return_result', {'status': 'success', 'value': False}]]
    assert first['split'] == cases[0]['split'] and first['curriculum']['split_group'] == 'invented:withdrawal-ruling'
    assert first['curriculum']['external']['stage']['of'] == cases[0]['id']
    assert len({s['id'] for s in stages}) == 4
    # a judgment with a callable folder of its own is no leaf: no stage task
    files['rule_notes/supersedes/inner.nl'] = JUDGE
    cases = invented.package(data, files)
    assert {s['semantics']['root'] for s in invented.stage_cases(cases, trace_for(cases))} == {'rule_notes/withdraws.nl'}


def card(slug, pitch, judgments):
    return {'slug': slug, 'pitch': pitch, 'judgments': [{'name': j, 'question': j} for j in judgments],
            'shape': 'notes to rulings', 'inputs': 'notes', 'output': 'record'}


def test_bank_cards_go_one_per_builder_and_near_duplicates_are_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(program_atlas, 'base_entries', lambda root: [
        {'id': 'application:sem-merge', 'kind': 'application', 'name': 'sem-merge', 'line': 'merge wiki edits by intent'}])
    bank = tmp_path / 'bank' / 'ideas'
    bank.mkdir(parents=True)
    tide = card('tide-pilot-bookings', 'Reconcile harbour pilot bookings against the tide table windows.',
                ['withinWindow', 'samePilot', 'cancels'])
    (bank / '01.json').write_text(json.dumps(tide))
    (bank / '02.json').write_text(json.dumps({**tide, 'slug': 'tide-pilot-bookings-2'}))
    (bank / '03.json').write_text(json.dumps(card('stitch-count-errata', 'Verify knitting pattern stitch counts after '
                                                  'errata across sizes.', ['appliesToSize', 'errataOverrides', 'repeatSpan'])))
    root = tmp_path / 'root'
    check = program_atlas.bank_check(root, bank)
    assert [r['slug'] for r in check['refused']] == ['tide-pilot-bookings-2']
    first = program_atlas.take(root, bank, 'builder-a')
    assert first['ok'] and first['slug'] == 'tide-pilot-bookings'
    second = program_atlas.take(root, bank, 'builder-b')
    assert second['ok'] and second['slug'] == 'stitch-count-errata', second
    states = program_atlas.claim_states(root)
    assert states['tide-pilot-bookings-2']['status'] == 'duplicate'
    assert program_atlas.take(root, bank, 'builder-c') == {'ok': False, 'reason': 'no free card left in the bank'}
    program_atlas.release(root, 'stitch-count-errata', 'failed', reason='build failed')
    assert program_atlas.claim_states(root)['stitch-count-errata']['status'] == 'failed'
    # a claimed idea is in the atlas an inventor reads, marked as being built
    assert any(e['id'] == 'invented:tide-pilot-bookings' and 'being built' in e['line']
               for e in program_atlas.campaign_entries(root))


def test_admit_needs_a_review_of_this_version_then_stages_by_split(tmp_path, monkeypatch):
    sub = write(tmp_path)
    data, files = invented.load(sub)
    out = sub / 'qualification'
    out.mkdir()
    cases = invented.package(data, files)
    invented.save_rows(out / 'cases.jsonl', cases)
    invented.save_rows(out / 'stages.jsonl', invented.stage_cases(cases, trace_for(cases)))
    (out / 'report.json').write_text(json.dumps({'ok': True, 'problems': [], 'bar': {'version': 'old'}, 'digest': invented.digest(sub)}))
    monkeypatch.setattr(invented, 'qualify', lambda submission, runtime: {'ok': False, 'problems': ['F4: requalified']})
    stale = invented.admit(sub, review_path=None, idea_path=None, root=tmp_path / 'root', runtime=tmp_path, skip_qualify=True)
    assert stale == {'admitted': False, 'stage': 'qualify', 'problems': ['F4: requalified']}, 'a report of an older bar is redone'
    (out / 'report.json').write_text(json.dumps({'ok': True, 'problems': [], 'bar': invented.bar(), 'digest': invented.digest(sub)}))
    (sub / 'idea.json').write_text(json.dumps(card('withdrawal-ruling', 'Rule on notes.', ['withdraws', 'supersedes', 'same_item'])))
    root = tmp_path / 'root'
    kwargs = dict(review_path=None, idea_path=None, root=root, runtime=tmp_path, skip_qualify=True)
    review = {'schema': invented.REVIEW_SCHEMA, 'reviewer': 'claude-opus-5-5', 'digest': invented.digest(sub),
              'requirements': {r: {'met': True, 'findings': []} for r in invented.JUDGED}}
    review['requirements']['S4'] = {'met': False, 'findings': ['note text reaches withdraws as plain string']}
    review['wrong_expected'] = ['n2: the note withdraws n1, so the expected "stands" is wrong']
    (sub / 'review.json').write_text(json.dumps(review))
    result = invented.admit(sub, **kwargs)
    assert not result['admitted'] and 'wrong expected' in result['problems'][0], 'a wrong expected answer blocks'
    monkeypatch.setenv('NATLANG_INVENTED_STRICT_REVIEW', '1')
    del review['wrong_expected']
    (sub / 'review.json').write_text(json.dumps(review))
    assert 'S4' in invented.admit(sub, **kwargs)['problems'][0], 'strict mode: an unmet requirement blocks'
    monkeypatch.delenv('NATLANG_INVENTED_STRICT_REVIEW')
    review['wrong_expected'] = ['stale finding']
    review['digest'] = 'stale'
    (sub / 'review.json').write_text(json.dumps(review))
    assert invented.review_problems(review, sub) == [], 'a review of another version neither blocks nor stands'
    (sub / 'review.json').unlink()
    assert invented.static_problems(sub, data, files, invented.metrics(data, files)) == []   # review/idea are allowed
    result = invented.admit(sub, **kwargs)
    assert result['admitted'] and result['split'] == invented.split_of(data) and result['reviewed'] is False, 'qualification alone admits'
    assert result['cases'] == 2 and result['stages'] == 4
    assert (root / 'admitted' / 'withdrawal-ruling' / 'stages.jsonl').exists()
    ledger = invented.rows_of(root / 'accepted.jsonl')
    assert [row['idea']['slug'] for row in ledger] == ['withdrawal-ruling']
    invented.admit(sub, **kwargs)
    assert len(invented.rows_of(root / 'accepted.jsonl')) == 1, 'admitting again does not repeat the ledger line'
    assert invented.publish(root, dry_run=True)['fresh'] == ['withdrawal-ruling']


def test_reference_report_reads_the_trace(tmp_path):
    sub = write(tmp_path)
    data, files = invented.load(sub)
    cases = invented.package(data, files)
    case = cases[0]
    case['curriculum']['reference']['children'] = [
        {'match': ['withdraws an earlier', 'n1 the'], 'value': 'replaces'},
        {'match': ['withdraws an earlier'], 'value': 'stands'},
        {'match': ['never used'], 'value': 'stands'}]
    note = lambda i, extra='': f'n{i} {extra}{PROSE}'  # noqa: E731
    children = [{'name': 'withdraws', 'path': 'rule_notes/withdraws.nl', 'args': {'note': note(1, 'This replaces the earlier one. ')}, 'value': 'replaces'},
                {'name': 'withdraws', 'path': 'rule_notes/withdraws.nl', 'args': {'note': note(2)}, 'value': 'stands'},
                {'name': 'withdraws', 'path': 'rule_notes/withdraws.nl', 'args': {'note': note(3)}, 'value': 'stands'},
                {'name': 'same_item', 'path': 'rule_notes/same_item.nl', 'args': {'a': note(4), 'b': note(4)}, 'value': True}]
    answers = [{'name': 'withdraws', 'entry': 0, 'shown': 'note: ' + note(1, 'This replaces the earlier one. ')},
               {'name': 'withdraws', 'entry': 1, 'shown': 'note: ' + note(2)},
               {'name': 'withdraws', 'entry': 1, 'shown': 'note: ' + note(3)}]
    trace = {'results': [{'id': case['id'], 'status': 'done', 'calls': 5, 'value': {'standing': []},
                          'children': children, 'answers': answers},
                         {'id': cases[1]['id'], 'error': "TypeError: Cannot read properties of undefined (reading 'slice')"}]}
    problems, _, text = invented.reference_report(data, cases, trace, '## Hardest items\n\n- n1: the replacement.\n- x9: absent.\n', None)
    joined = '\n'.join(problems)
    assert 'F3: instance b: the reference trace did not run to an answer' in joined, problems
    assert 'F3: instance a: the reference run took' not in joined, problems
    case['curriculum']['reference']['children'] += [{'match': [f'unused {i}'], 'value': 'x'} for i in range(4)]
    assert 'F3: instance a: the reference run took 2 of its 7 children entries' in '\n'.join(
        invented.reference_report(data, cases, trace, '', None)[0])
    assert 'S7: instance a: withdraws: the one call whose text says "replaces" answers replaces' in joined, problems
    assert 'S8: instance a: same_item is asked about a and b' in joined, problems
    assert 'C2: instance a: ' in joined and 'C4: hardest item 2' in joined and 'C4: hardest item 1' not in joined, problems
    assert 'entry 1 ["withdraws an earlier"] -> "stands": 2 call(s)' in text, text


def test_notes_and_mistakes_quote_the_text_as_it_stands(tmp_path):
    sub = write(tmp_path)
    (sub / 'NOTES.md').write_text((sub / 'NOTES.md').read_text() + '\nn1 says "the ceiling stain kept spreading" and "a sentence that was removed".\n')
    data = json.loads((sub / 'submission.json').read_text())
    data['instances'][0]['wrong_answers'][0]['mistake'] = 'reads "nobody came to look" as a repair'
    (sub / 'submission.json').write_text(json.dumps(data))
    text = '\n'.join(check(sub)[0])
    assert 'C4: NOTES.md quotes "a sentence that was removed"' in text and 'ceiling stain' not in text, text
    assert 'F4: instance a: a wrong answer\'s mistake quotes "nobody came to look"' in text, text


def test_reports_name_the_bar_they_were_qualified_against():
    assert invented.bar()['version'] == invented.BAR_VERSION and len(invented.bar()['requirements_sha256']) == 16


def test_a_dist_older_than_its_sources_is_stale(tmp_path):
    runtime = tmp_path / 'ts-host'
    for rel in ('dist/cli/main.js', 'dist/teacher/curriculum.js', 'src/a.ts', 'package.json'):
        (runtime / rel).parent.mkdir(parents=True, exist_ok=True)
        (runtime / rel).write_text('x')
    import os
    os.utime(runtime / 'src/a.ts', (2_000_000_000, 2_000_000_000))
    assert invented.stale_runtime(runtime)
    for rel in ('dist/cli/main.js', 'dist/teacher/curriculum.js'):
        os.utime(runtime / rel, (2_000_000_100, 2_000_000_100))
    assert not invented.stale_runtime(runtime)
    assert invented.runtime_problems(runtime) == []


def test_verify_counts_mutant_wrong_answers_in_its_summary(tmp_path):
    import shutil
    import subprocess
    runtime = tmp_path / 'runtime'
    (runtime / 'dist/native').mkdir(parents=True)
    (runtime / 'dist/teacher').mkdir()
    (runtime / 'package.json').write_text('{"type":"module"}')
    (runtime / 'dist/native/prompt.js').write_text('export const TOOLS_PROMPT = "p";')
    (runtime / 'dist/native/external.js').write_text('export function externalModule() { throw Error("unused"); }')
    (runtime / 'dist/teacher/curriculum.js').write_text('''
      export async function verifyCases(rows) { return rows.map(() => ({ ok: true, problems: [] })); }
      export async function replayReference(row) {
        const wrong = row.curriculum.reference.children?.[0]?.value === 'misread';
        return { run: { outcome: { status: 'done', accepted: !wrong, value: wrong ? 'w' : 'r' } } };
      }''')
    row = {'id': 'm', 'family': 'f', 'split': 'train', 'semantics': {'expected': 'r'},
           'curriculum': {'reference': {'root': [['eval', {'code': 'const r = await f(); return r;', 'finish': True}]],
                                        'children': [{'match': ['x'], 'value': 'read'}]},
                          'external': {'checks': {'wrong_answers': [], 'keyword_baseline_code': [],
                                                  'wrong_mutants': [{'mistake': 'm', 'children': [{'match': ['x'], 'value': 'misread'}], 'value': 'w'}]}}}}
    source = tmp_path / 'cases.jsonl'
    source.write_text(json.dumps(row) + '\n')
    node = next(iter(sorted(Path.home().glob('.local/share/natlang-dev/node-v24*/bin/node'), reverse=True)), None) or shutil.which('node')
    subprocess.run([str(node), str(REPO / 'scripts/rl_tasks/verify.mjs'), str(runtime), str(source), str(tmp_path / 'out'), '1'],
                   check=True, capture_output=True, text=True, timeout=60)
    summary = json.loads((tmp_path / 'out' / 'verification.json').read_text())['summary']
    assert (summary['kept'], summary['wrong_controls'], summary['mutants_rejected']) == (1, 1, 1), summary


def test_c2_counts_only_calls_whose_own_entry_the_result_depends_on():
    shared = 'the datasheet says the enzyme works from thirty seven to fifty degrees in this buffer'
    item = lambda i: f'step {i} incubates the reaction at the stated temperature for ten minutes then cools it'  # noqa: E731
    row = {'id': 'c', 'status': 'done', 'calls': 4,
           'children': [{'name': 'j', 'args': {'sheet': shared, 'step': item(i)}, 'value': 'ok'} for i in range(3)],
           'answers': [{'name': 'j', 'entry': 0, 'shown': shared + ' ' + item(0)},
                       {'name': 'j', 'entry': 1, 'shown': shared + ' ' + item(1)},
                       {'name': 'j', 'entry': 1, 'shown': shared + ' ' + item(2)}]}
    counted = invented.calls_of_own_entries(row, set(invented.own_entries(row)))
    assert [c['args']['step'] for c in counted] == [item(0)], 'a default-answered call sharing the datasheet does not count'
    assert invented.calls_of_own_entries(row, set()) == [], 'an entry no answer depends on does not count'
