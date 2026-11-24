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
  note: "string"
returns: "boolean"
---
Decide whether note withdraws an earlier note.
'''
HELPER = '\n'.join(['import withdraws from "./withdraws.nl";', 'import supersedes from "./supersedes.nl";',
                    'import sameItem from "./same_item.nl";', 'import { semanticFilter } from "@natlang/node/std";',
                    'export async function rule(notes: string[]): Promise<{ standing: string[] }> {',
                    '  const kept = await semanticFilter(notes, withdraws);'] +
                   [f'  const step{i} = kept.length + {i};' for i in range(40)] +
                   ['  return { standing: kept.map(n => n.slice(0, 4)) };', '}', 'export function order(a: string[]): string[] { return [...a].sort(); }'])


def instance(name):
    return {'name': name, 'level': 3, 'inputs': {'notes': [f'n{i} text' for i in range(9)]}, 'oracle': 'exact',
            'expected': {'standing': ['n1']},
            'reference': {'root': [['eval', {'code': 'const r = await rule(notes); return r;', 'finish': True}]],
                          'children': [{'match': ['withdraws an earlier'], 'value': False}]},
            'ablations': [{'judgment': j, 'children': [{'match': ['x'], 'value': True}]}
                          for j in ('withdraws', 'supersedes', 'same_item')],
            'wrong_answers': [{'value': {'standing': [w]}, 'mistake': 'm'} for w in ('n2', 'n3', 'n4')],
            'keyword_baseline': 'return { standing: notes.filter(n => n.includes("1")) };'}


def write(tmp, extra=None):
    sub = tmp / 'sub'
    files = {'rule_notes.nl': ROOT, 'types.ts': TYPES, 'rule_notes/withdraws.nl': JUDGE,
             'rule_notes/supersedes.nl': JUDGE.replace('withdraws an earlier', 'replaces'),
             'rule_notes/same_item.nl': JUDGE.replace('withdraws an earlier note', 'names the same item'),
             'rule_notes/rule.ts': HELPER, 'natlang.json': '{}'}
    for path, text in files.items():
        (sub / 'program' / path).parent.mkdir(parents=True, exist_ok=True)
        (sub / 'program' / path).write_text(text)
    data = {'schema': invented.SCHEMA, 'slug': 'withdrawal-ruling', 'author': {'inventor': 'a', 'builder': 'b'},
            'root': 'rule_notes.nl', 'constructs': ['decide', 'std-collection', 'refinement'],
            'judgments': [{'name': n, 'file': f'rule_notes/{n}.nl', 'trap': 't'} for n in ('withdraws', 'supersedes', 'same_item')],
            'instances': [instance('a'), instance('b')], **(extra or {})}
    (sub / 'submission.json').write_text(json.dumps(data))
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
