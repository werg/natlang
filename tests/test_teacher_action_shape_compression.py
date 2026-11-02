"""Quality witnesses must survive a lossless change of storage representation."""
import json
from pathlib import Path
import subprocess
import sys
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_compressed_quality_review_preserves_holds_and_collateral(tmp_path):
    def row(parent, family, group, code):
        return {
            'training_admission': {'approved': True}, 'split': 'train',
            'teacher_trajectory_id': parent, 'task_family': family,
            'source_groups': [group],
            'target': {'tool_calls': [{'function': {
                'arguments': json.dumps({'code': code})}}]},
        }

    rows = [
        row('missing-iteration', 'curriculum_composed_process', 'shared-é', 'return 1'),
        row('sound-sibling', 'curriculum_actor_greenhouse', 'shared-é', 'await step.iterateOn(state)'),
        row('independent', 'curriculum_folder_criteria_reducer', 'independent', 'await nl<boolean>`Judge`(item)'),
    ]
    plain = tmp_path / 'turns.jsonl'
    plain.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    compressed = plain.with_suffix('.jsonl.zst')
    subprocess.run(['zstd', '-q', '-3', str(plain), '-o', str(compressed)], check=True)
    results = []
    for index, source in enumerate((plain, compressed)):
        witness, review = tmp_path / f'witness-{index}.jsonl', tmp_path / f'review-{index}.json'
        subprocess.run([sys.executable, str(ROOT / 'scripts/neuralese_data/audit_teacher_action_shape.py'),
                        '--turns', str(source), '--out', str(witness), '--review', str(review)], check=True)
        results.append((witness.read_bytes(), json.loads(review.read_text())))
    assert results[0][0] == results[1][0]
    for _, review in results:
        assert review['parents'] == 3
        assert review['held_parent_ids'] == {'missing-iteration': 'required_iterateOn_missing'}
        assert review['collateral_parent_ids'] == ['sound-sibling']
        assert review['source_groups'] == 1


def test_child_ticket_keyword_bypass_holds_only_its_parent(tmp_path):
    def row(parent, code, child):
        return {
            'training_admission': {'approved': True}, 'split': 'train',
            'teacher_trajectory_id': parent, 'task_family': 'curriculum_folder_criteria_reducer',
            'source_groups': [parent],
            'source_ref': {'parent_invocation_id': 'root'} if child else {},
            'target': {'tool_calls': [{'function': {'arguments': json.dumps({'code': code})}}]},
        }

    cases = [
        row('regex-child', 'await nl<boolean>`Judge the ticket`(ticket)', False),
        row('regex-child', 'return /problem.*resolved/i.test(ticket) && /customer.*confirmed/i.test(ticket)', True),
        row('keyword-child', 'await nl<boolean>`Judge the ticket`(ticket)', False),
        row('keyword-child', 'return ticket.includes("resolved") && ticket.includes("customer")', True),
        row('sound-child', 'await nl<boolean>`Judge the ticket`(ticket)', False),
        row('sound-child', 'return true', True),
    ]
    source = tmp_path / 'turns.jsonl'
    source.write_text(''.join(json.dumps(case) + '\n' for case in cases))
    review = tmp_path / 'review.json'
    subprocess.run([sys.executable, str(ROOT / 'scripts/neuralese_data/audit_teacher_action_shape.py'),
                    '--turns', str(source), '--out', str(tmp_path / 'holds.jsonl'),
                    '--review', str(review)], check=True)
    result = json.loads(review.read_text())
    import hashlib
    assert len(result['reviewer_dependencies']) == 3
    for reference in result['reviewer_dependencies']:
        assert hashlib.sha256(Path(reference['path']).read_bytes()).hexdigest() == reference['sha256']
    assert result['held_parent_ids'] == {
        'regex-child': 'semantic_ticket_judgment_crisp_bypass',
        'keyword-child': 'semantic_ticket_judgment_crisp_bypass',
    }


@pytest.mark.parametrize('family', ['curriculum_workbench_tickets', 'workbench_tickets'])
def test_churn_child_cannot_substitute_keywords_for_semantic_judgment(family):
    from scripts.neuralese_data.audit_teacher_action_shape import semantic_ticket_bypass
    turn = {'task_family': family, 'source_ref': {'parent_invocation_id': 'root'},
            'target': {'tool_calls': [{'function': {'arguments': json.dumps({
                'code': 'return /cancel|leave|switch(?:ing)? to another/i.test(text);'})}}]}}
    assert semantic_ticket_bypass(turn)
    # A typed child answer and crisp computation at the root remain eligible.
    turn['source_ref'] = {}
    assert not semantic_ticket_bypass(turn)


def test_semantic_screen_ignores_slashes_inside_nl_templates_paths_and_comments():
    from scripts.neuralese_data.audit_teacher_action_shape import semantic_ticket_bypass
    turn = {'task_family': 'curriculum_workbench_tickets', 'source_ref': {'parent_invocation_id': 'root'},
            'target': {'tool_calls': [{'function': {'arguments': json.dumps({'code':
                'const path = "tickets/fix.md"; /* /cancel|leave/i.test(text) */ '
                'return await nl<boolean>`Does this suggest leave/switch providers or cancellation/non-renewal?`(text);'})}}]}}
    assert not semantic_ticket_bypass(turn)
    turn['source_ref'] = {'parent_invocation_id': 'root'}
    turn['target']['tool_calls'][0]['function']['arguments'] = json.dumps({'code': 'return true;'})
    assert not semantic_ticket_bypass(turn)
