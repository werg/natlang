import json
from pathlib import Path

import pytest

from scripts.neuralese_data.audit_acquired_conversion_coverage import inventory
from scripts.neuralese_data.convert_acquired_sessions_student import output_dir


def conversion(tmp_path):
    export = tmp_path / 'source' / 'D32' / 'session' / 'export.json'
    export.parent.mkdir(parents=True)
    export.write_text('{}')
    manifest = tmp_path / 'exports.json'
    manifest.write_text(json.dumps({'exports': [str(export)]}))
    root = tmp_path / 'converted'
    directory = output_dir(root, str(export))
    output = directory / 'train' / 'native-graphs.jsonl.zst'
    output.parent.mkdir(parents=True)
    output.write_bytes(b'committed output fixture')
    receipt = {'training_admission': False, 'export_receipt': {'path': str(export), 'sha256': 'source-hash'},
               'outputs': [{'path': str(output), 'bytes': output.stat().st_size, 'rows': 2, 'sha256': 'output-hash'}],
               'counts': {'graphs:train': 2, 'positive_sessions': 2}, 'source_record_count': 2,
               'builder_sha256': 'builder-hash'}
    receipt_path = directory / 'receipt.json'
    receipt_path.write_text(json.dumps(receipt))
    return manifest, root, receipt_path, receipt, output


def test_complete_and_pending_exports_are_separate(tmp_path):
    manifest, root, _, _, _ = conversion(tmp_path)
    document = json.loads(manifest.read_text())
    document['exports'].append(str(tmp_path / 'source/D15/pending/export.json'))
    manifest.write_text(json.dumps(document))
    result = inventory(manifest, root)
    assert result['complete_receipts'] == 1
    assert result['pending'] == 1
    assert result['per_source']['D32']['reason:graphs:train'] == 2
    assert not result['errors']
    assert 'not output hash verification' in result['verification_scope']


@pytest.mark.parametrize('defect', ['admission', 'foreign_export', 'output_size', 'row_count'])
def test_invalid_commits_are_not_counted_complete(tmp_path, defect):
    manifest, root, receipt_path, receipt, output = conversion(tmp_path)
    if defect == 'admission':
        receipt['training_admission'] = True
    elif defect == 'foreign_export':
        receipt['export_receipt']['path'] = str(tmp_path / 'foreign/export.json')
    elif defect == 'output_size':
        output.write_bytes(b'truncated')
    else:
        receipt['counts']['graphs:train'] = 3
    receipt_path.write_text(json.dumps(receipt))
    result = inventory(manifest, root)
    assert result['complete_receipts'] == 0
    assert len(result['errors']) == 1


def test_duplicate_exports_rejected(tmp_path):
    manifest, root, _, _, _ = conversion(tmp_path)
    document = json.loads(manifest.read_text())
    document['exports'] *= 2
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError, match='duplicate exports'):
        inventory(manifest, root)


def test_filtered_inventory_does_not_claim_unselected_source_coverage(tmp_path):
    manifest, root, _, _, _ = conversion(tmp_path)
    result = inventory(manifest, root, sources={'D15'})
    assert result['manifest_exports'] == 1
    assert result['expected_exports'] == 0
    assert result['source_filter'] == ['D15']
    assert result['complete_receipts'] == 0
    assert not result['per_source']
