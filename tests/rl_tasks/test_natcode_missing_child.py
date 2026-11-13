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
