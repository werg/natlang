import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.intake_natlang_codebase import intake
from tests.test_natlang_codebase_intake import _package


def test_intake_carries_namespace_scope_and_report_gate_contracts(tmp_path):
    package, root = _package(tmp_path)
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    metadata['service_aliases'] = {
        'telecom': {'source': 'telecom', 'methods': ['readAccount', 'writeAccount']},
        'device': {'source': 'telecom', 'methods': ['inspectDevice']},
    }
    metadata['service_scopes'] = {
        'telecom': ['triage/inspectEvidence.nl', 'triage/chooseAction.nl'],
        'device': ['triage/chooseAction.nl'],
    }
    metadata['report_completeness'] = {
        'schema': 'natlang.report-completeness/1', 'mutating_methods': ['writeAccount'],
        'report_effects_path': 'effects', 'checkpoint_effects_path': 'checkpoint.receipts',
        'report_diagnosis_path': 'diagnosis', 'diagnosis_function': 'triage/inspectEvidence.nl',
    }
    metadata['external_services'] = {'telecom': {
        'source_path': 'service.ts', 'source_sha256': '0' * 64,
    }}
    source = root / 'service.ts'
    source.write_text('export function readAccount(): number { return 1; }\n'
                      'export function writeAccount(): number { return 2; }\n'
                      'export function inspectDevice(): number { return 3; }')
    import hashlib
    metadata['external_services']['telecom']['source_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
    metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / 'qualified.jsonl'
    intake(package, output=output)
    row = json.loads((tmp_path / 'qualified.judged.jsonl').read_text())
    assert row['semantics']['service_aliases'] == metadata['service_aliases']
    assert row['semantics']['service_scopes'] == metadata['service_scopes']
    assert row['semantics']['report_completeness'] == metadata['report_completeness']


def test_intake_rejects_scope_contracts_that_name_missing_helpers_or_exports(tmp_path):
    package, root = _package(tmp_path)
    source = root / 'service.ts'
    source.write_text('export function readAccount(): number { return 1; }')
    metadata_path = root / 'metadata.json'
    metadata = json.loads(metadata_path.read_text())
    import hashlib
    metadata['external_services'] = {'service': {'source_path': source.name,
        'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest()}}
    metadata['service_aliases'] = {'account': {'source': 'service', 'methods': ['readAccount']}}
    metadata['service_scopes'] = {'account': ['triage.missing.nl']}
    metadata_path.write_text(json.dumps(metadata))
    import pytest
    with pytest.raises(ValueError, match='missing Natlang function'):
        intake(package, output=tmp_path / 'bad-scope.jsonl')
    metadata['service_scopes'] = {'account': ['triage/inspectEvidence.nl']}
    metadata['service_aliases']['account']['methods'] = ['notExported']
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match='not an exported host function'):
        intake(package, output=tmp_path / 'bad-export.jsonl')
