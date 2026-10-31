import copy
import hashlib
import json

import pytest

from scripts.neuralese_data.provider_campaign_source import ledger_dispatch_identity
from natlang_neuralese.common.hashing import canonical_json_sha256_hex
from scripts.neuralese_data.provider_judged_regrade import validate_projection


def sample():
    raw = {
        'id': 'run-1',
        'version': 'natlang.teacher_trajectory.native/1',
        'task': {'program_ir': {'id': 'program-1', 'split': 'train'}},
        'provenance': {'model': 'gpt-6-luna', 'provider': 'openai-codex'},
        'execution_identity': {'execution_run_id': 'run-1'},
        'trajectory': [{'phase': 'action', 'invocation_id': 'call-1'}],
        'outcome': {'status': 'done', 'accepted': False, 'checks': {'files': False},
                    'rejection_reasons': ['files'], 'effects': {}},
    }
    effective = {
        'status': 'done', 'accepted': True, 'checks': {'files': True},
        'rejection_reasons': [], 'effects': {}, 'files_check': {'accepted': True, 'score': 1},
    }
    projection = {
        'schema': 'natlang.provider-judged-outcome-regrade/1',
        'raw_result_sha256': 'a' * 64,
        'raw_outcome_sha256': canonical_json_sha256_hex(raw['outcome']),
        'effective_outcome': effective,
        'effective_outcome_sha256': canonical_json_sha256_hex(effective),
        'task_sha256': canonical_json_sha256_hex(raw['task']['program_ir']),
    }
    derived = copy.deepcopy(raw)
    derived['outcome'] = effective
    derived['source_regrade'] = {'projection': projection}
    return raw, derived, projection


def test_judged_regrade_is_an_outcome_only_overlay():
    raw, derived, projection = sample()
    validate_projection(raw, derived, projection)


@pytest.mark.parametrize('mutation', ['task', 'actor', 'trajectory', 'execution', 'oracle'])
def test_judged_regrade_rejects_changes_outside_the_outcome_projection(mutation):
    raw, derived, projection = sample()
    changed = copy.deepcopy(derived)
    if mutation == 'task':
        changed['task']['program_ir']['id'] = 'different'
    elif mutation == 'actor':
        changed['provenance']['model'] = 'different-model'
    elif mutation == 'trajectory':
        changed['trajectory'][0]['invocation_id'] = 'different-call'
    elif mutation == 'execution':
        changed['execution_identity']['execution_run_id'] = 'different-run'
    else:
        changed['outcome']['oracle'] = {'accepted': True, 'level': 'exact'}
    with pytest.raises(ValueError):
        validate_projection(raw, changed, projection)


def test_judged_regrade_rejects_modified_projection_or_raw_outcome():
    raw, derived, projection = sample()
    changed = copy.deepcopy(projection)
    changed['effective_outcome']['accepted'] = False
    with pytest.raises(ValueError, match='effective outcome hash'):
        validate_projection(raw, derived, changed)
    changed = copy.deepcopy(raw)
    changed['outcome']['rejection_reasons'] = []
    with pytest.raises(ValueError, match='original outcome'):
        validate_projection(changed, derived, projection)


def test_historical_dispatch_identity_reconstructs_the_claim_header():
    original = {'campaign_id': 'campaign-1', 'plan_sha256': 'b' * 64}
    original['identity_sha256'] = hashlib.sha256(
        json.dumps(original, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    attested = {**original, 'dispatch_plan_history_attestation': '/outside/attestation.json'}
    attested['identity_sha256'] = hashlib.sha256(
        json.dumps(attested, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    assert ledger_dispatch_identity(attested) == original
