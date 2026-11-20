import gzip
import json

import pytest

from scripts.vibestudio.audit_open_goal_references import audit, corpus_floor, fingerprint, inventory_coverage


def test_registrations_only_gain_credit_for_bound_model_free_results(tmp_path):
    scenario = 'open-goal-review-shift'
    goal = 'Find out what the gallery team still needs to finish.'
    catalog = tmp_path / 'catalog.jsonl'
    catalog.write_text(json.dumps({
        'interaction': 'open-user-goal',
        'scenario': scenario,
        'family': 'mission',
        'split': 'train',
        'sourceGroups': ['gallery'],
        'goalSha256': fingerprint(goal),
    }) + '\n')
    proof = {
        'caseId': 'review-shift',
        'contextId': 'ctx-original',
        'goal': goal,
        'passed': True,
        'checks': {'readback': True},
    }
    diagnostics = {'natlangTeacher': None, 'openGoalVerification': proof}
    trace = {
        'test': {'name': 'open-goal-oracle-review-shift'},
        'result': {'passed': True, 'details': proof['checks']},
        'execution': {'provenance': {'contextId': 'ctx-original'}, 'diagnostics': diagnostics},
    }
    output = tmp_path / 'trajectory-open-goal-oracle-review-shift-full.json.gz'

    def verified_count():
        with gzip.open(output, 'wt', encoding='utf-8') as stream:
            json.dump(trace, stream)
        return audit(catalog, [tmp_path], {'mission'})['verified']

    assert verified_count() == 1
    result = audit(catalog, [tmp_path], {'mission'})
    assert result['families']['mission'] == {
        'train': 1, 'held': 0, 'verified_train': 1, 'verified_held': 0,
    }
    assert result['families_without_verified_holdout'] == ['mission']
    catalog.write_text(catalog.read_text().replace('"split": "train"', '"split": "test"'))
    assert audit(catalog, [tmp_path], {'mission'})['families_without_verified_holdout'] == []
    proof['goal'] = 'Review a different shift.'
    assert verified_count() == 0
    proof['goal'], proof['contextId'] = goal, 'ctx-replacement'
    assert verified_count() == 0
    proof['contextId'] = 'ctx-original'
    diagnostics['natlangTeacher'] = {'calls': []}
    assert verified_count() == 0


def test_corpus_floor_comes_from_the_pinned_baseline(tmp_path):
    baseline = tmp_path / 'baseline.jsonl'
    snapshot = {
        'schema': 'vibestudio.open-goal-inventory/1',
        'agentic_baseline': 12,
        'minimum_distinct_open_goals': 24,
    }
    baseline.write_text(json.dumps(snapshot) + '\n')
    assert corpus_floor(baseline) == 24
    baseline.write_text(json.dumps({**snapshot, 'minimum_distinct_open_goals': 12}) + '\n')
    with pytest.raises(ValueError, match='Invalid pinned baseline'):
        corpus_floor(baseline)


def test_inventory_links_count_only_after_a_reference_pass(tmp_path):
    baseline = tmp_path / 'baseline.jsonl'
    baseline.write_text('\n'.join(map(json.dumps, [
        {'kind': 'snapshot', 'workspace_sources': {'base': ['base'], 'one': ['base', 'one'], 'two': ['base', 'two']},
         'pins': {'base': 'pinned'}},
        {'kind': 'system-test', 'capability': 'retained-session'},
        {'kind': 'system-test', 'capability': 'file-version'},
        {'kind': 'skill', 'id': 'source:session'},
    ])))
    path = tmp_path / 'crosswalk.json'
    mapping = {
        'schema': 'vibestudio.open-goal-coverage/1',
        'families': {'recovery': {'capabilities': ['retained-session'], 'skills': ['source:session']}},
    }
    path.write_text(json.dumps(mapping))
    family = {'verified_train': 0, 'verified_held': 0}
    report = {'families': {'workspace-open-user-goal-recovery': family}}
    features = inventory_coverage(report, baseline, path)['capabilities']
    assert features['unmapped'] == ['file-version']
    assert features['verified_reference_links'] == []
    family['verified_held'] = 1
    assert inventory_coverage(report, baseline, path)['capabilities']['verified_reference_links'] == [
        'retained-session',
    ]
    live = tmp_path / 'installed.json'
    live.write_text(json.dumps({
        'schema': 'vibestudio.installed-skill-inventory/1',
        'baselinePins': {'base': 'pinned'},
        'mechanism': 'workspace.listSkills+workspace.readSkill',
        'instantiableWorkspaces': ['one', 'two'],
        'workspaces': [{'workspace': 'one', 'matchesPinnedSources': True,
                       'observed': {'skills': [{'expectedSource': 'source:session'}, {'expectedSource': 'source:new'}]}}],
    }))
    installed = inventory_coverage(report, baseline, path, live)['installed_skills']
    assert installed['missing_workspaces'] == ['two']
    assert installed['workspaces']['one'] == {'observed': 2, 'without_verified_reference_links': ['source:new']}
    mapping['families']['recovery']['skills'] = ['unknown']
    path.write_text(json.dumps(mapping))
    with pytest.raises(ValueError, match='Unknown skills'):
        inventory_coverage(report, baseline, path)
