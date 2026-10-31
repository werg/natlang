import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts' / 'neuralese_data'))
from idiom_scores import score_calls, unhandled_nl  # noqa: E402

HANDLED = [
    'const assess = nl<boolean>`Is ${x} ok?`;\nconst r = await assess({x});',
    'const assess = nl<boolean>`Is it ok?`;\nconst rs = await Promise.all(items.map(i => assess(i)));',
    'const r = nl`summarize ${t}`(t);\nreturn r;',
    'const r = nl`summarize ${t}`(t);\nconst v = await r;',
    'const v = await nl`x`();',
    'const out = await Promise.all(items.map(async (i) => nl<string>`label ${i}`(i)));',
    'const f = nl`x`;\nreturn await Promise.all(items.map(f));',
    'for (const i of items) { out.push(await nl`classify ${i}`(i)); }',
    'const settled = await Promise.allSettled(xs.map(x => nl`check ${x}`(x)));',
]
UNHANDLED = [
    'nl`do ${a}`(a);\nreturn 1;',
    'const f = nl`x`;\nf(1);',
    'const f = nl`x`;\nconst a = items.map(f);\nreturn a;',
    'const r = nl`summarize`(t);\nreturn r.length;',
]


def test_handled_nl_promises_are_not_flagged():
    for code in HANDLED:
        assert unhandled_nl(code) == 0, code


def test_unhandled_nl_promises_are_flagged():
    for code in UNHANDLED:
        assert unhandled_nl(code) > 0, code


def test_score_marks_only_unhandled_evals():
    calls = [{'function': {'name': 'eval', 'arguments': {'code': HANDLED[0]}}},
             {'function': {'name': 'eval', 'arguments': {'code': UNHANDLED[0]}}}]
    assert score_calls(calls, [], child=False)['unawaited_nl'] == 1


def test_typed_named_templates_are_functions():
    code = ('const check: Neuralese<(x: {a: string}) => Promise<boolean>> = nl<boolean>`Is ${x} ok?`;\n'
            'const v = await decide(check, {a: input.a}); return v.value;')
    assert unhandled_nl(code) == 0
    assert unhandled_nl('const check: Neuralese<(x: string) => Promise<boolean>> = nl<boolean>`ok?`;\ncheck(t);') == 1


def test_object_typed_annotation_with_semicolons():
    code = ('const judge: Neuralese<(args: {a: string; b: unknown}) => Promise<boolean>> = nl<boolean>`Decide.`; '
            'const verdict = await decide(judge, input); return verdict.value;')
    assert unhandled_nl(code) == 0


def test_crisp_checker_labels_count_as_verified():
    from idiom_scores import turn_scores

    class NoHolds:
        def __call__(self, program, groups):
            return None
    row = {'id': 'refine-judge:exact:x:0:0', 'split': 'train', 'source_groups': ['g'],
           'messages': [{'role': 'user', 'content': 'judge'}], 'target': {'role': 'assistant', 'content': 'false'},
           'training_admission': {'approved': True, 'kind': 'crisp-checker-label (refine-judge-exact-20261009-v1)'}}
    assert turn_scores(row, 'refine', NoHolds())['verified'] is True
    row['training_admission'] = {'approved': True, 'kind': 'teacher'}
    assert turn_scores(row, 'refine', NoHolds())['verified'] is None
