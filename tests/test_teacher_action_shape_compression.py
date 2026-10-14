"""Quality witnesses must survive a lossless change of storage representation."""
import json
from pathlib import Path
import subprocess
import sys


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
