import json
from pathlib import Path
import sys

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'scripts'),
                str(Path(__file__).resolve().parents[1] / 'training' / 'neuralese')]

from neuralese_data import acquired_agents
from neuralese_data.common import Reject
from neuralese_data.trajectory import Trajectory
from scripts.neuralese_data import export_d12_feature_status as exporter


def test_d12_status_export_binds_live_measurement_and_preserves_adapter_omissions(tmp_path, monkeypatch):
    source_root = tmp_path / 'source'
    source_dir = source_root / 'D12'
    source_dir.mkdir(parents=True)
    source_bytes = b'seed'
    (source_dir / 'sample.tar.gz').write_bytes(source_bytes)
    import hashlib
    entry = {'path': 'sample.tar.gz', 'bytes': len(source_bytes),
             'sha256': hashlib.sha256(source_bytes).hexdigest()}
    revision = 'pinned-revision'
    acquisition = {'status': 'acquired', 'revision': revision, 'files': [entry]}
    (source_dir / 'acquisition.json').write_text(json.dumps(acquisition))
    monkeypatch.setattr(acquired_agents, 'validate_acquisition', lambda *a, **k: acquisition)
    measurement = tmp_path / 'measurement.json'
    measurement.write_text(json.dumps({'schema': 'natlang.d12-live48-measurement-summary/1',
        'rows': 48, 'exact_matches': 48, 'extra_json_claims': 0,
        'packet_sha256': 'b' * 64}))
    trajectory = Trajectory(id='source:status-view', family='source_reported_feature_status',
        messages=[{'role': 'user', 'content': 'full source context'},
                  {'role': 'assistant', 'content': '{"both_passed":false}'}], head=1,
        noun='source-reported feature status', lineage={'notes': {
            'source_reported_feature_status': {'review_kind': 'source-curated-reported-status',
                'failed_actions_positive_targets': False, 'causal_claims': False,
                'native_execution_verified': False},
            'original_cooperation_context': {'archive_context': {'member': {
                'path': 'nested/member.json', 'sha256': 'c' * 64}}}}},
        license={'spdx': 'MIT'}, split='train', split_groups=['source-group'],
        outcome={'label': 'teacher', 'details': {}})
    monkeypatch.setattr(acquired_agents, 'ACQUIRED_FILES', None)
    monkeypatch.setattr(exporter, 'd12_feature_status_diagnoses', lambda: iter([
        ('sample.tar.gz:0:final:d12-feature-status:view', trajectory),
        ('sample.tar.gz:1', Reject('no structured reported status'))]))

    output = tmp_path / 'exports'
    result = exporter.export(source_root, output, measurement)
    assert result['training_admission'] is False
    assert result['live48_summary']['packet_sha256'] == 'b' * 64
    assert len(result['files']) == 1
    record = json.loads((output / result['files'][0]['receipt']).read_text())
    assert record['complete_sessions'] == 1
    assert record['explicit_adapter_omissions'] == 1
    assert record['contract']['source'] == entry
    assert record['contract']['live48_summary_sha256'] == exporter.sha256_file_hex(measurement)
