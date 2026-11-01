import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.rl_tasks.root_input_binding_frontend import validate_root_input_bindings


def _record(inputs):
    return {
        'id': 'nllb-stale-target-rebase-v2',
        'semantics': {
            'root': 'recoverStaleRevision.nl',
            'files': {
                'recoverStaleRevision.nl': '''---\nargs:\n  caseId: string\nreturns: string\n---\nReturn caseId.\n''',
            },
            'inputs': inputs,
            # Collector service fixtures live beside the root args and are
            # deliberately not positional/model-visible root parameters.
            'services': {'corpusLedger': 'export async function readCase() {}'},
        },
    }


def test_root_binding_accepts_declared_arguments_and_external_service_metadata():
    receipt = validate_root_input_bindings([_record({'caseId': 'case-1'})])
    assert receipt['loaded_by'] == 'teacher.prepareProgramNode'
    assert receipt['checked'] == [{
        'id': 'nllb-stale-target-rebase-v2',
        'root': 'recoverStaleRevision.nl',
        'input_names': ['caseId'],
        'binding': 'teacher.prepareProgramNode/collector-root-argument-binding',
    }]


def test_root_binding_rejects_service_fixture_metadata_leaked_as_unknown_argument():
    with pytest.raises(ValueError, match=r'serviceFixtureId is not a parameter of recoverStaleRevision'):
        validate_root_input_bindings([_record({
            'caseId': 'case-1', 'serviceFixtureId': 'case-1',
        })])
