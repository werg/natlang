import importlib.util
import hashlib
import json
from pathlib import Path
import pytest


def test_revised_receipts_match_the_real_loader_after_staged_aliases(tmp_path):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'training/neuralese'))
    path = Path(__file__).resolve().parents[1] / 'scripts/neuralese_data/finalize_source_correction_quality_hold.py'
    spec = importlib.util.spec_from_file_location('finalize_correction_identity_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    train, held = tmp_path / 'train.json', tmp_path / 'held.json'
    train.write_text('{"source":"train"}')
    held.write_text('{"source":"held"}')
    logical_train, logical_held = tmp_path / 'original/train.json', tmp_path / 'original/held.json'
    (tmp_path / '.natlang-source-map.json').write_text(json.dumps({'schema': 'natlang.staged-source-map/1', 'paths': {
        str(logical_train): str(train), str(logical_held): str(held)}}))
    proof = {'current_source': {'path': str(train), 'sha256': hashlib.sha256(train.read_bytes()).hexdigest()},
             'protected_held_correction': {'current_source': {'path': str(held), 'sha256': hashlib.sha256(held.read_bytes()).hexdigest()}}}
    module.bind_current_receipt_identities(proof, train, held)
    assert proof['current_source']['path'] == str(logical_train)
    assert proof['protected_held_correction']['current_source']['path'] == str(logical_held)
    held.write_text('changed')
    with pytest.raises(ValueError, match='cannot change source bytes'):
        module.bind_current_receipt_identities(proof, train, held)
