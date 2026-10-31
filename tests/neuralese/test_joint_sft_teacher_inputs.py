import json

from scripts.neuralese_data import build_lfm3_joint_sft as build


def test_joint_recipe_discovers_direct_and_twin_teacher_receipts(tmp_path, monkeypatch):
    receipts = {
        'direct': {'schema': 'natlang.verified-teacher-native-graph-build/1',
                   'model_key': 'lfm3', 'status': 'passed'},
        'twin': {'schema': 'natlang.verified-teacher-native-graph-student-twin/1',
                 'model_key': 'lfm3', 'verification': {'passed': True}},
        'failed': {'schema': 'natlang.verified-teacher-native-graph-build/1',
                   'model_key': 'lfm3', 'status': 'failed'},
        'wrong-model': {'schema': 'natlang.verified-teacher-native-graph-build/1',
                        'model_key': 'lfm25', 'status': 'passed'},
        'unrecognized': {'schema': 'unreviewed/1', 'model_key': 'lfm3',
                         'verification': {'passed': True}},
    }
    for name, receipt in receipts.items():
        path = tmp_path / name / 'lfm3' / 'receipt.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(receipt))
    monkeypatch.setattr(build, 'TEACHERS', tmp_path)
    assert {path.parent.name for path, _ in build.teacher_twins()} == {'direct', 'twin'}
