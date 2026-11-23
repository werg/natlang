"""Several held sets (plans/semantic-fp/09 B1): held-v3 stays the gate (``grpo.default_held``), every ``held-eval*``
registry role (held-v4, its reservations) is excluded from training by every path (``held_eval.training_exclusion_ids``,
``sft_sources.held_closure``)."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'scripts/neuralese_data'), str(REPO / 'training/neuralese'), str(REPO)]

from natlang_neuralese.common import data_registry  # noqa: E402
from natlang_neuralese.rl import held_eval  # noqa: E402


def row(i, group, source_groups=(), source_ids=()):
    return {'id': i, 'split': 'train', 'family': 'fam', 'source_groups': list(source_groups),
            'source_ids': list(source_ids), 'curriculum': {'split_group': group, 'rl': {'level': 0, 'family': 'fam'}}}


def pool():
    return [row('v3-held', 'g3'), row('v3-sib', 'g3'),
            row('v4-held', 'g4'), row('v4-sib', 'g4'), row('v4-src', 'gx', source_groups=['g4-source']),
            row('brief', 'ext-unit:x', source_groups=['design-brief-app-004']),
            row('brief-code', 'ext-unit:y', source_ids=['natlang:codebases:codebases/design_briefs/004-org-workbench:repair']),
            row('free', 'g9'), row('free2', 'g10', source_ids=['design_briefs/014-other'])]


def manifest(path, tasks, groups, cases, **extra):
    path.mkdir(parents=True, exist_ok=True)
    cases_path = path / 'cases.jsonl'
    cases_path.write_text(''.join(json.dumps(r) + '\n' for r in cases))
    body = {'schema': held_eval.MANIFEST_SCHEMA, 'tasks': [{'id': t} for t in tasks], 'excluded_split_groups': groups,
            'pool': {'path': str(cases_path), 'sha256': held_eval._sha256(cases_path)},
            'cases': {'path': 'cases.jsonl', 'sha256': held_eval._sha256(cases_path)}, **extra}
    (path / 'manifest.json').write_text(json.dumps(body))
    return path / 'manifest.json'


def reservations(path, **fields):
    path.write_text(json.dumps({'schema': held_eval.RESERVATIONS_SCHEMA, **fields}))
    return path


@pytest.fixture
def held(tmp_path, monkeypatch):
    rows = pool()
    v3 = manifest(tmp_path / 'held-v3', ['v3-held'], ['g3'], [rows[0]])
    res = reservations(tmp_path / 'reservations.json', groups=['design-brief-app-004'],
                       patterns=[r'(?:^|[/:])004-org-workbench(?:$|[/.:])'],
                       share_rules=[{'pattern': '^flagship:', 'share': 0.2, 'seed': 'held-v4-flagship'}])
    v4 = manifest(tmp_path / 'held-v4', ['v4-held'], ['g4'], [{**rows[2], 'source_groups': ['g4-source']}],
                  also_excludes=[{'path': str(v3), 'sha256': held_eval._sha256(v3)},
                                 {'path': str(res), 'sha256': held_eval._sha256(res)}])
    roles = [('held-eval', str(v3)), ('held-eval-semantic-fp', str(v4)), ('held-eval-semantic-fp-reservations', str(res))]
    monkeypatch.setattr(data_registry, 'role_prefix_paths', lambda prefix, repo=None: [r for r in roles if r[0].startswith(prefix)])
    monkeypatch.setattr(data_registry, 'role_path', lambda name, repo=None: dict(roles)[name])
    return SimpleNamespace(v3=v3, v4=v4, res=res, rows=rows)


def test_the_gate_stays_held_v3_and_every_held_role_is_excluded(held):
    import grpo
    assert grpo.default_held() == str(held.v3)
    assert grpo.held_role_paths() == [str(held.v3), str(held.v4), str(held.res)]
    excluded = held_eval.training_exclusion_ids(held.rows, grpo.default_held())
    # held-v3 and its split group, held-v4 and its split group, a row sharing a source group with a held-v4 case, and
    # the reserved design brief rows (by group and by path)
    assert excluded == {'v3-held', 'v3-sib', 'v4-held', 'v4-sib', 'v4-src', 'brief', 'brief-code'}
    # held-v4's own exclusion carries held-v3 and its reservations (also_excludes), with or without the registry
    assert held_eval.exclusion_ids(held.v4, held.rows) == excluded - {'v4-src'}


def test_a_reservation_needs_the_rows_and_reserves_a_fixed_share_of_flagship_groups(held):
    with pytest.raises(ValueError, match='pass the rows'):
        held_eval.exclusion_ids(held.res)
    rule = {'pattern': '^flagship:', 'share': 0.2, 'seed': 'held-v4-flagship'}
    groups = [f'flagship:pr-shepherd:case-{i}' for i in range(2000)]
    share = sum(held_eval.reservation_share_held(rule, g) for g in groups) / len(groups)
    assert 0.17 < share < 0.23
    spec = json.loads(held.res.read_text())
    flagged = [g for g in groups if held_eval.reservation_share_held(rule, g)]
    assert held_eval.reserved(spec, row('f', flagged[0]))
    assert not held_eval.reserved(spec, row('f', next(g for g in groups if g not in flagged)))
    assert not held_eval.reserved(spec, row('f', 'not-a-flagship:' + flagged[0]))


def test_grpo_selection_never_takes_a_held_v4_sibling_or_a_reserved_row(held):
    import grpo
    args = SimpleNamespace(pool=[str(write(held.v3.parent.parent / 'pool.jsonl', held.rows))], held_manifest=str(held.v3),
                           exclude_manifest=[], task_ids=None, exclude_family_regex=None, exclude_ids=[],
                           no_default_errata=True, seed=1, pass_rates=None, pass_band=(0.1, 0.9), pass_min_scored=1,
                           explore=None, max_model_requests=None, case_seconds=None)
    eligible, excluded = grpo.load_tasks(args)
    # v4-src shares a source group with a held-v4 row: the source-group sibling rule covers the new set too
    assert sorted(r['id'] for r in eligible) == ['free', 'free2'] and excluded == 7


def write(path, rows):
    path.write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return path


def test_sft_held_closure_covers_every_held_role_and_reservations(held):
    import sft_sources
    closure = sft_sources.held_closure(held.v3)
    assert {'v3-held', 'v4-held'} <= closure[0] and 'g4-source' in closure[1]
    assert sft_sources.held_hit('x', ['codebases/design_briefs/004-org-workbench/a.ts'], ['g9'], closure)
    assert sft_sources.held_hit('x', [], ['design-brief-app-004'], closure)
    assert not sft_sources.held_hit('x', ['design_briefs/014-other'], ['g9'], closure)
    tasks = {r['id']: r for r in held.rows}
    assert sft_sources.held_excluded_ids(held.v3, tasks, closure) >= {'v4-held', 'v4-sib', 'v4-src', 'brief', 'brief-code'}


def test_the_live_registry_keeps_held_v3_as_the_gate():
    roles = dict(data_registry.role_prefix_paths('held-eval'))
    assert roles['held-eval'] == data_registry.role_path('held-eval')
    assert roles['held-eval'].endswith('held-v3/manifest.json')
    assert 'held-eval-semantic-fp-reservations' in roles
    assert all(name.startswith('held-eval') for name in roles)


def test_an_authored_held_set_is_pinned_whole_and_refuses_an_exposed_row(tmp_path):
    rows = [{**row(f'h{i}', 'shared' if i < 3 else f'g{i}'), 'split': 'dev' if i % 2 else 'train'} for i in range(6)]
    cases = write(tmp_path / 'cases.jsonl', rows)
    manifest, held = held_eval.build_manifest(cases, avoid=set(), all_rows=True, reject={'h5'})
    # every row but the rejected one, whatever its split or split-group siblings
    assert [t['id'] for t in manifest['tasks']] == ['h0', 'h1', 'h2', 'h3', 'h4'] and len(held) == 5
    with pytest.raises(ValueError, match='exposed'):
        held_eval.build_manifest(cases, avoid=set(), avoid_groups={'shared'}, all_rows=True)
