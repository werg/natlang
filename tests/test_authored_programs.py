"""Authored programs (scripts/rl_tasks/sources/authored_programs.py): trajectory entries -> reference children,
packaging, the consistency check over a replay row, and the statistics. No runtime: rows are written by hand in the
shape scripts/rl_tasks/authored_replay.mjs emits."""
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.rl_tasks.sources import authored_programs as ap  # noqa: E402

ROOT = '''---
description: Digest a thread.
args:
  thread: Message[]
returns: Digest
---
Read every message with readMessage and summarize each topic with summarizeTopic.
'''
FILES = {
    'digest.nl': ROOT,
    'types.ts': '/** A message. */\nexport type Message = { messageId: string; body: Untrusted<string> };\n',
    'digest/readMessage.nl': '---\nargs:\n  message: Message\nreturns: Gist\n---\nRead message.\n',
    'digest/summarizeTopic.nl': '---\nargs:\n  topic: string\nreturns: Summary\n---\nSummarize topic.\n',
    'digest/summarizeTopic/isSettled.nl': '---\nargs:\n  topic: string\nreturns: boolean\nreadout: decision\n---\nDecide.\n',
    'digest/count.ts': 'export function count(xs: string[]): number {\n  return xs.length;\n}\n',
}


def submission(**instance):
    inst = {'name': 'a', 'inputs': {'thread': []},
            'trajectory': {'root': [['eval', {'code': 'const g = await readMessage(thread[0]);'}],
                                    ['eval', {'code': 'return { g };', 'finish': True}]],
                           'calls': [{'fn': 'readMessage', 'value': {'topic': 'x'}},
                                     {'fn': 'readMessage', 'match': ['messageId: "m1"'], 'value': {'topic': 'y'}},
                                     {'match': ['an inline literal'], 'value': True}]}}
    inst.update(instance)
    return {'schema': ap.SCHEMA, 'slug': 'thread-digest', 'author': {'builder': 'test'}, 'root': 'digest.nl',
            'instances': [inst]}


def test_entries_become_children_most_specific_first():
    data = submission()
    children = ap.children(data['instances'][0]['trajectory'])
    assert children[0]['match'] == [ap.ANCHOR.format('readMessage'), 'messageId: "m1"']
    assert children[1]['match'] == [ap.ANCHOR.format('readMessage')]
    assert children[2] == {'match': ['an inline literal'], 'value': True}
    assert ap.entry_order(data['instances'][0]['trajectory']) == [1, 0, 2]
    turns = ap.children({'calls': [{'fn': 'f', 'turns': [['eval', {'code': 'x'}]]}]})
    assert turns[0]['calls'] == [['eval', {'code': 'x'}]] and 'value' not in turns[0]


def test_package_takes_expected_from_the_replay_unless_an_oracle_is_declared():
    data = submission()
    case = ap.package(data, FILES, {'a': {'g': 1}})[0]
    assert case['semantics']['expected'] == {'g': 1} and case['semantics']['oracle'] == 'exact'
    assert case['curriculum']['reference']['root'] == data['instances'][0]['trajectory']['root']
    assert case['curriculum']['external']['rl_task'] is False
    assert case['curriculum']['rl']['basis'] == 'authored-trajectory-only'
    rl = submission(rl={'oracle': 'exact', 'expected': {'g': 2}}, folder_files={'a.md': 'x'})
    case = ap.package(rl, FILES, {'a': {'g': 1}})[0]
    assert case['semantics']['expected'] == {'g': 2} and case['curriculum']['external']['rl_task'] is True
    assert case['semantics']['folder_files'] == {'a.md': 'x'}


def test_static_problems_name_shape_errors(tmp_path):
    (tmp_path / 'program').mkdir()
    (tmp_path / 'scratch.py').write_text('x')
    data = submission()
    data['instances'][0]['trajectory']['calls'].append({'fn': 'f', 'value': 1, 'turns': []})
    data['instances'][0]['trajectory']['calls'].append({'value': 1})
    problems = ap.static_problems(tmp_path, data, FILES)
    assert any('scratch.py' in p for p in problems)
    assert any('calls[4] needs exactly one of value, turns, call' in p for p in problems)
    assert any('calls[5] names no call' in p for p in problems)
    assert ap.static_problems(tmp_path, {**data, 'root': 'missing.nl'}, FILES)[-1].startswith('root must')


def tree_row():
    root_id = 'curriculum-reference:replay:x'
    return {'status': 'done', 'accepted': False, 'value': {'g': 1}, 'model_calls': 6, 'failed_actions': [],
            'answers': [{'name': 'readMessage', 'entry': 0, 'shown': 'message m1'},
                        {'name': 'readMessage', 'entry': 1, 'shown': 'message m2'},
                        {'name': 'readMessage', 'entry': 1, 'shown': 'message m3'}],
            'tree': [
                {'id': 't/1', 'parent': root_id, 'definition': 'nl:digest/readMessage.nl', 'name': 'readMessage',
                 'outcome': 'done', 'actions': [{'tool': 'return_result'}], 'decision': None, 'functional': False,
                 'value': {'topic': 'venue', 'gist': 'Ana proposes the lake lodge again for its large room.'}},
                {'id': 't/2', 'parent': root_id, 'definition': 'nl:digest/summarizeTopic.nl', 'name': 'summarizeTopic',
                 'outcome': 'done', 'decision': None, 'functional': False, 'value': {'topic': 'venue', 'settled': True},
                 'actions': [{'tool': 'eval', 'code': 'const v = await decide(isSettled, topic, gists);\n'
                                                      'const t = nl<(g: Gist) => boolean>`g bears on ${topic}.`;\n'
                                                      'const n = count(gists.map(g => g.topic));'}]},
                {'id': 't/3', 'parent': 't/2', 'definition': 'nl:digest/summarizeTopic/isSettled.nl', 'name': 'isSettled',
                 'outcome': 'done', 'decision': {'options': ['true', 'false'], 'chosen': 0}, 'functional': False,
                 'actions': [], 'value': True},
                {'id': 't/4', 'parent': 't/2', 'definition': 'nl:0123456789abcdef', 'name': 'nl@eval:1',
                 'outcome': 'done', 'decision': None, 'functional': False, 'actions': [], 'value': False},
                {'id': 't/5', 'parent': root_id, 'definition': 'iterateOn:site', 'name': 'iterateOn site',
                 'outcome': 'done', 'detail': '2 steps', 'iteration': {'predicate': 'nl:fedcba9876543210', 'steps': 2},
                 'actions': [], 'decision': None, 'functional': False, 'value': None},
            ]}


def test_consistency_reports_unfinished_runs_failed_actions_and_loose_entries():
    data = submission()
    row = tree_row()
    problems, warnings = ap.consistency(data, [row])
    assert problems == []
    # entry 1 (author index 0, fn only) answered two different calls with one value but has one fragment: no warning;
    # the inline entry answered nothing
    assert any('calls[3]' in w and 'answered no call' in w for w in warnings)
    row = {**row, 'answers': [{'name': 'readMessage', 'entry': 0, 'shown': 'a'}, {'name': 'readMessage', 'entry': 0, 'shown': 'b'}]}
    _, warnings = ap.consistency(data, [row])
    assert any('answers 2 different calls with one value' in w for w in warnings)
    bad = {**tree_row(), 'status': 'quiesced', 'detail': 'NO ENTRY', 'failed_actions': [{'tool': 'eval', 'result': 'boom'}]}
    problems, _ = ap.consistency(data, [bad])
    assert any('does not finish the root call' in p for p in problems)
    assert any('1 actions failed where the trajectory declares 0' in p for p in problems)
    problems, _ = ap.consistency(submission(expected_failures=1), [{**tree_row(), 'failed_actions': [{'tool': 'eval', 'result': 'x'}]}])
    assert problems == []
    problems, _ = ap.consistency(submission(rl={'oracle': 'exact', 'expected': 1}), [tree_row()])
    assert any('declared oracle rejects' in p for p in problems)


def test_statistics_report_the_semantic_fp_shape_and_nl_literals():
    data = submission()
    stats = ap.program_stats(data, FILES, [tree_row()])
    inst = stats['instances'][0]
    # root (added) -> summarizeTopic -> isSettled / the inline literal: depth 3; root and summarizeTopic are interior
    assert inst['nl_depth'] == 3 and inst['interior_functions'] == 2
    assert inst['nl_calls'] == 5            # root, readMessage, summarizeTopic, isSettled, the inline call
    for kind in ('decision', 'decide', 'iterate-nl-until', 'inline-nl', 'nl-literal-from-data', 'untrusted'):
        assert kind in inst['construct_kinds'], kind
    assert inst['nl_literals'] == {'created': 1, 'built_from_data': 1, 'calls': 1, 'call_tree_share': 0.2}
    assert inst['crisp_helper_calls'] == 1 and inst['nl_call_share'] == round(5 / 6, 3)
    assert inst['outputs'] == {'open': 1, 'closed': 3, 'structured': 1}
    assert stats['source']['ts_logic_lines'] == 2 and stats['source']['ts_unjustified'] == ['digest/count.ts']
    assert stats['source']['declared']['decision'] == 1
    short = ap.shortfalls(stats)
    assert any(s.startswith('NL depth 3') for s in short)


def test_nl_literal_texts():
    code = ('const a = nl`Summarize x.`(x);\nconst b = nl<(g: G) => boolean>`g is about ${topic}.`;\n'
            'const c = nl.with({ rubric })`Grade essay against rubric.`(essay);\nconst d = "nl" + `not a literal`;')
    assert ap.nl_literals(code) == ['Summarize x.', 'g is about ${topic}.', 'Grade essay against rubric.']


def test_wave_report_writes_report_and_emphasis(tmp_path, monkeypatch):
    sub = tmp_path / 'sub' / 'thread-digest'
    (sub / 'program').mkdir(parents=True)
    for rel, text in FILES.items():
        (sub / 'program' / rel).parent.mkdir(parents=True, exist_ok=True)
        (sub / 'program' / rel).write_text(text)
    (sub / 'submission.json').write_text(json.dumps(submission()))
    monkeypatch.setattr(ap, 'replay', lambda cases, runtime: {'results': [tree_row()]})
    monkeypatch.setattr(ap.invented, 'runtime_problems', lambda runtime: [])
    report = ap.wave_report(tmp_path, tmp_path)
    assert report['wave']['programs'] == 1
    assert (tmp_path / 'REPORT.md').read_text().count('thread-digest') == 1
    emphasis = (tmp_path / 'EMPHASIS.draft.md').read_text()
    assert 'Grow the tree downwards' in emphasis and 'Programs of the last wave closest' in emphasis
