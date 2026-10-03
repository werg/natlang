import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from scripts.start_reviewed_luna_slots import validate_immutable_artifact_pins


def test_two_slot_plan_pinning_authority_is_rejected_before_claim(tmp_path):
    authority = tmp_path / 'authority.json'
    state = tmp_path / 'luna-state.json'
    authority.write_text('{}\n')
    state.write_text('{}\n')
    plan = {
        'root_approved': True, 'provider': 'openai-codex', 'model_id': 'gpt-6-luna',
        'model_concurrency': 1,
        'workers': [{'number': 1}, {'number': 2}],
        'predecessors': [{}, {}],
        'authority': str(authority), 'luna_state': str(state),
        'launch_record': str(tmp_path / 'launch-record.json'),
        'artifact_hashes': {str(authority): hashlib.sha256(authority.read_bytes()).hexdigest()},
    }
    plan_path = tmp_path / 'plan.json'
    plan_path.write_text(json.dumps(plan))
    digest = hashlib.sha256(plan_path.read_bytes()).hexdigest()
    result = subprocess.run(
        [sys.executable, 'scripts/start_reviewed_luna_slots.py', str(plan_path), '--sha256', digest],
        text=True, capture_output=True)
    assert result.returncode != 0
    assert 'pins mutable plan authority' in result.stderr
    assert not Path(plan['launch_record']).exists()
    assert not Path(plan['launch_record']).with_suffix('.lock').exists()


def test_normalized_symlink_to_mutable_state_is_rejected(tmp_path):
    authority = tmp_path / 'authority.json'
    state = tmp_path / 'state.json'
    alias = tmp_path / 'state-alias.json'
    authority.write_text('{}')
    state.write_text('{}')
    alias.symlink_to(state)
    plan = {
        'authority': str(authority), 'luna_state': str(state),
        'artifact_hashes': {str(alias): '0' * 64},
    }
    with pytest.raises(ValueError, match='pins mutable plan luna_state'):
        validate_immutable_artifact_pins(plan)


def test_immutable_artifact_pins_are_accepted(tmp_path):
    authority = tmp_path / 'authority.json'
    state = tmp_path / 'state.json'
    queue = tmp_path / 'queue.jsonl'
    authority.write_text('{}')
    state.write_text('{}')
    queue.write_text('{}\n')
    plan = {
        'authority': str(authority), 'luna_state': str(state),
        'artifact_hashes': {str(queue): hashlib.sha256(queue.read_bytes()).hexdigest()},
    }
    normalized = validate_immutable_artifact_pins(plan)
    assert normalized == {'authority': str(authority.resolve()), 'luna_state': str(state.resolve())}
