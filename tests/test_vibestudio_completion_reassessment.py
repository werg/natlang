import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts/neuralese_data'))
import reassess_campaign_completion as audit


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return {'path': path.name, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def archived_case(root, name, *, semantic_completed, verification_status='passed'):
    capture = root/'capture'/name
    goal = f'Update {name}'
    world = {'instance': 'owned', 'context_id': name + '-context', 'case_id': name}
    proof = {'goal': goal, 'passed': semantic_completed, 'contextId': name + '-context', 'caseId': name,
             'checks': {'actualUserGoal': semantic_completed},
             'outcomeDimensions': {'workspaceEffect': {'required': True, 'passed': semantic_completed}}}
    evidence = write(capture/'workspace.json', {'proof': proof})
    verification = write(capture/'verification.json', {'task_id': name, 'world': world,
        'checks': {'capturedEvidence': True}, 'root_call_ids': ['actual-root'],
        'status': verification_status, 'evidence': [{**evidence, 'kind': 'workspace'}]})
    write(capture/'manifest.json', {'id': name, 'world': world,
         'interaction': {'kind': 'open-user-goal', 'goal': goal}, 'verification': verification})
    return {'status': 'verified', 'manifest': f'{name}/manifest.json'}


def test_reassessment_reopens_only_actual_incomplete_goals_and_preserves_sources(tmp_path):
    wave = tmp_path/'0001'
    rows = {name: archived_case(wave, name, semantic_completed=complete, verification_status='passed' if complete else 'partial')
            for name, complete in [('complete', True), ('partial', False)]}
    write(wave/'receipt.json', rows)
    before = {p: p.read_bytes() for p in tmp_path.rglob('*.json')}
    result = audit.reassess(tmp_path)
    assert result['summary']['before_completed_goals'] == 2
    assert result['summary']['after_completed_goals'] == 1
    assert result['summary']['reopened_goals'] == ['partial']
    assert result['summary']['unresolved'] == 0
    assert {p: p.read_bytes() for p in before} == before
    assert all(row['manifest']['sha256'] and row['verification']['sha256'] for row in result['receipts']['0001-result.json']['cases'].values())


def test_partial_scope_cannot_overrule_pending_whole_goal_review(tmp_path):
    wave = tmp_path/'0002'
    write(wave/'receipt.json', {'goal': archived_case(wave, 'goal', semantic_completed=True, verification_status='partial')})
    result = audit.reassess(tmp_path)
    assert result['summary']['after_completed_goals'] == 0
    assert result['summary']['reopened_goals'] == ['goal']


def test_tampered_bound_proof_is_unresolved_not_a_reopened_goal(tmp_path):
    wave = tmp_path/'0003'
    write(wave/'receipt.json', {'goal': archived_case(wave, 'goal', semantic_completed=True)})
    (wave/'capture/goal/workspace.json').write_text('{}')
    result = audit.reassess(tmp_path)
    assert result['summary']['unresolved'] == 1
    assert result['summary']['reopened_goals'] == []
