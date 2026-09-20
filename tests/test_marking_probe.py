"""Check the probe against actual runtime range semantics, not a second parser."""
import json
import random
from types import SimpleNamespace

from natlang.runtime import Runtime, Session
from natlang.types import TypeEnv
from natlang.values import load_program
from scripts.marking_probe import assess, cases, summarize
from natlang.gen.codebases import with_marks, webserver, nlprolog, highlighter
from natlang.gen.policy import ReferenceAgent
from natlang.gen.synth import guarded_call, MARK_STYLES
from scripts.generate import run_program


def test_false_branch_range_reproduces_teacher_failure_and_separate_marks_fix_it():
    _, doc, _, expected = list(cases())[1]
    root = load_program(doc)
    session = Session(Runtime(None), root, TypeEnv())
    args = {'path': 'return', 'type': 'Num', 'value': 0, 'done': [4, 7]}
    assert session.apply('write', args).kind == 'ok'
    assert root.marks[5] == 'done'  # The actual runtime agrees with the probe.
    session.apply('mark_done', {'start': 3})
    log = [{'action': 'write ' + json.dumps(args)}]
    bad = assess(root.marks, expected, log, 'en_passant')
    assert bad['mark_errors'] == [{'line': 5, 'expected': 'skipped', 'actual': 'done'}]
    assert bad['untaken_lines_marked_done'] == [5]
    session.apply('mark_done', {'start': 5, 'skipped': True})
    assert assess(root.marks, expected, log, 'en_passant')['all_marks_correct']


def test_style_violation_is_separate_from_correct_marks():
    row = assess({3: 'done'}, {3: 'done'}, [{'action': 'write {"done": 3}'}], 'grouped', [])
    assert row['all_marks_correct'] and row['style_compliant'] is False
    assert len(row['style_violations']) == 2
    assert summarize([dict(row, correct=True)])['correct_marking_episodes'] == 1
    assert summarize([dict(row, correct=True)])['style_compliant_episodes'] == 0


def test_last_mark_feedback_tells_agent_when_result_is_complete():
    root = load_program({'$lambda': {'type': 'Lambda<{}, Bool>',
                                     'instructions': 'return true'}})
    session = Session(Runtime(None), root, TypeEnv())
    assert session.apply('write', {'path': 'return', 'type': 'Bool', 'value': True}).kind == 'ok'
    result = session.apply('mark_done', {'start': 1})
    assert 'Reply normally to finish' in result.text


def test_extended_cases_cover_false_items_and_early_return():
    examples = {name: (doc, value, marks) for name, doc, value, marks in cases(True)}
    assert examples['all_false'][1] == 0
    assert examples['all_false'][2][5] == 'skipped'
    assert examples['early_return_True'][2][5] == 'skipped'
    assert examples['early_return_False'][2][5] == 'done'


def test_reference_does_not_credit_an_uncalled_occurrence_of_the_same_function():
    lam = SimpleNamespace(original_body='', body='first = f()\nsecond = f()', codebase={'f': object()})

    def script(lam):
        yield [('call', {'function': 'f', 'to': 'let/first'})]

    gen = with_marks(script)(lam)
    assert next(gen)[0][0] == 'call'
    assert gen.send(SimpleNamespace(kind='done')) == [
        ('mark_done', {'start': 1}), ('mark_done', {'start': 2, 'skipped': True})]


def test_reference_marks_real_branch_occurrences_and_constructed_returns(monkeypatch):
    captured = []
    original = ReferenceAgent.run

    def record(self, session):
        result = original(self, session)
        captured.append((session.lam.original_body or session.lam.body, dict(session.lam.marks)))
        return result

    monkeypatch.setattr(ReferenceAgent, 'run', record)
    for make in (webserver, nlprolog, highlighter):
        for seed in range(4):
            run_program(make(random.Random(seed)))
    page_branches = 0
    for body, marks in captured:
        lines = body.splitlines()
        wraps = [i for i, text in enumerate(lines, 1) if 'response = wrap_page(' in text]
        if len(wraps) == 2:
            statuses = [marks.get(i) for i in wraps]
            assert statuses.count('done') <= 1
            page_branches += statuses.count('done')
        for i, line in enumerate(lines, 1):
            if 'return { verdict, derived:' in line or 'return { path: file.path,' in line:
                assert marks[i] == 'done'
    assert page_branches > 0


def test_guarded_training_marks_both_paths_in_all_styles(monkeypatch):
    captured = []
    original = ReferenceAgent.run

    def record(self, session):
        result = original(self, session)
        captured.append(dict(session.lam.marks))
        return result

    monkeypatch.setattr(ReferenceAgent, 'run', record)
    paths = set()
    for style in MARK_STYLES:
        monkeypatch.setenv('NATLANG_MARK_STYLE', style)
        for seed in range(8):
            prog = guarded_call(random.Random(seed))
            run_program(prog)
            early = prog.expected < 0
            paths.add(early)
            assert captured[-1] == {3: 'done', 4: 'done' if early else 'skipped',
                                    5: 'skipped' if early else 'done', 6: 'skipped' if early else 'done'}
    assert paths == {True, False}
