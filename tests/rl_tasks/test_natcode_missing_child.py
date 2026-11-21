"""Missing-child repair tasks: parameter names are free, positions and types are the contract (H2, 2026-11-03)."""
import random

from scripts.rl_tasks.sources import natcode


def test_parameter_helpers_rename_and_split_only_top_level_names():
    args = '{ request: Req, prior?: A|null, rows: { a: string, b: number }[], f: (x: A) => B }'
    assert natcode._split_parameters(args) == ['request: Req', 'prior?: A|null', 'rows: { a: string, b: number }[]', 'f: (x: A) => B']
    assert natcode._rename_parameters(args) == (
        '{ requestValue: Req, priorValue?: A|null, rowsValue: { a: string, b: number }[], fValue: (x: A) => B }')


def test_missing_child_task_states_that_names_are_free():
    assert 'parameter names are yours to choose' in natcode.HINTS['missing-child']
    files = {
        'natlang.json': '{}\n',
        'op.nl': '---\nargs: { request: Req, pages: Page[] }\nreturns: Out\nuses: [op/child]\n---\nDo it.\n',
        'op/child.nl': '---\nargs: { request: Req, approvals: Page[] }\nreturns: Guard\n---\nBind approvals.\n',
        'types.ts': 'export type Req = {}; export type Page = {}; export type Out = {}; export type Guard = {};\n',
    }
    kinds = {kind: (extras, contract) for kind, path, broken, extras, wrong, contract in natcode._repairs(files, random.Random(3), 100)}
    extras, contract = kinds['missing-child']
    assert extras['signatures'] == [{'path': 'op/child.nl', 'args': '{ request: Req, approvals: Page[] }', 'returns': 'Guard'}]
    assert 'approvals: Page[]' in contract


def test_type_aliases_keep_the_last_member_before_the_closing_brace():
    # held-v2 row 246: HistoricPlan's oracle lacked evidenceRef, the member written just before `}` without a `;`.
    files = {'types.ts': 'export type HistoricPlan = { drawingId: ID; revision: string; areaId: ID; feature: string; '
                         'sourceDate: string; evidenceRef: Ref };\n'
                         'export type Note = {\n  text: string;\n  tags?: string[];\n};\n'}
    aliases = natcode._type_aliases(files)
    assert sorted(aliases['HistoricPlan']) == ['areaId', 'drawingId', 'evidenceRef', 'feature', 'revision', 'sourceDate']
    assert aliases['HistoricPlan']['evidenceRef'] == (False, 'Ref')
    assert aliases['Note'] == {'text': (False, 'string'), 'tags': (True, 'string[]')}


def test_explore_questions_stay_inside_the_scope_the_task_root_exposes():
    # gap-v1 audit 2026-11-21: the root `uses` lists only top-level operations, so a namespace folder (qualification/)
    # is unreachable, and golds naming its functions or types had 0 teacher successes in 94 episodes.
    files = {
        'natlang.json': '{}\n',
        'op.nl': '---\nargs: { request: Req }\nreturns: Out\n---\nRead request.marker and request.target.\n',
        'op/child.nl': '---\nargs: { request: Req }\nreturns: Out\n---\nCheck request.marker and request.target.\n',
        'types.ts': 'export type Req = { marker: string; target: string; extra: string }; export type Out = { a: string; b: string; c: string };\n',
        'qualification/check.nl': '---\nargs: { request: Req, gate: Gate }\nreturns: Gate\n---\nUse request.marker and gate.target.\n',
        'qualification/types.ts': 'export type Gate = { target: string; level: string; owner: string };\n',
    }
    assert set(natcode._exposed(files)) == {'op.nl', 'op/child.nl', 'types.ts'}
    questions = natcode._explore_questions(files, random.Random(1))
    assert questions
    for kind, x, answer in questions:
        assert 'Gate' not in str(x) and not any(str(item).startswith('qualification') for item in (answer if isinstance(answer, list) else [answer])), (kind, x, answer)
    assert ('mentions', 'marker', ['op', 'op.child']) in questions
