"""Union admission must bind current reviews and exact held decision IDs."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest


@pytest.fixture
def union_fixture(tmp_path, monkeypatch):
    script = Path(__file__).parents[1] / 'scripts/neuralese_data/register_native_turn_render.py'
    spec = importlib.util.spec_from_file_location('registration_protection_test', script)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    reviews = tmp_path / 'training/source-reviews'; reviews.mkdir(parents=True)
    (reviews / 'holds.jsonl').write_text('{}\n')
    review_sha = hashlib.sha256((reviews / 'holds.jsonl').read_bytes()).hexdigest()
    directory = tmp_path / 'training/decision-holds'; directory.mkdir()
    bindings = []
    for name in ('semantic-ticket-crisp-bypass-v3.jsonl', 'native-non-success-v1.jsonl'):
        path = directory / name
        path.write_text(json.dumps({'id': name, 'reason': 'reviewed hold'}) + '\n')
        bindings.append({'path': str(path), 'kind': 'decision-hold',
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    render = tmp_path / 'union'; render.mkdir()
    receipt = {'source_review': {'holds_sha256': review_sha}, 'excluded_source_references': bindings,
               'held_native_decision_ids': 2, 'renderings': {}}
    files = {}
    for mode in ('full', 'program-header', 'whole-trajectory'):
        receipt['renderings'][mode] = {'by_tokenizer': {}}
        for tok in ('lfm25', 'mellum21'):
            path = render / mode / tok / 'text-data.jsonl'; path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'id': mode + tok}) + '\n')
            receipt['renderings'][mode]['by_tokenizer'][tok] = {}
            files[mode + ':' + tok] = {'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
    proof = {'schema': 'natlang.protected-union-independent-audit/2', 'status': 'passed',
             'current_source_review_sha256': review_sha, 'current_exact_hold_files': bindings,
             'twin_identities_equal': True, 'pending_source_review_intersections': 0,
             'exact_decision_hold_intersections': 0, 'train_held_group_text_intersections': 0,
             'quality_quarantine_intersections': 0, 'files': files}
    return module, render, receipt, proof


def test_current_build_requires_both_reviewed_decision_hold_bindings(union_fixture):
    module, render, receipt, _ = union_fixture
    assert module.verify_union_protection(render, receipt)['current_build_receipt'] is True
    receipt['excluded_source_references'] = receipt['excluded_source_references'][:1]
    with pytest.raises(ValueError, match='requires current'):
        module.verify_union_protection(render, receipt)


def test_stale_build_needs_complete_current_audit_bound_to_every_view(union_fixture):
    module, render, receipt, proof = union_fixture
    receipt['source_review']['holds_sha256'] = '0' * 64
    with pytest.raises(ValueError, match='requires current'):
        module.verify_union_protection(render, receipt)
    path = render / 'independent-protection-audit.json'
    path.write_text(json.dumps(proof))
    assert module.verify_union_protection(render, receipt)['independent_audit'] is not None
    # Changing one view after the audit must invalidate its admission.
    (render / 'whole-trajectory/mellum21/text-data.jsonl').write_text('{"changed":true}\n')
    with pytest.raises(ValueError, match='output changed'):
        module.verify_union_protection(render, receipt)


@pytest.mark.parametrize('defect', ['old_schema', 'missing_exact_holds', 'held_intersection', 'missing_view'])
def test_incomplete_outside_audit_cannot_launder_stale_build(union_fixture, defect):
    module, render, receipt, proof = union_fixture
    receipt['source_review']['holds_sha256'] = '0' * 64
    if defect == 'old_schema':
        proof['schema'] = 'natlang.protected-union-independent-audit/1'
    elif defect == 'missing_exact_holds':
        proof.pop('current_exact_hold_files')
    elif defect == 'held_intersection':
        proof['exact_decision_hold_intersections'] = 1
    else:
        proof['files'].pop('program-header:lfm25')
    (render / 'independent-protection-audit.json').write_text(json.dumps(proof))
    with pytest.raises(ValueError):
        module.verify_union_protection(render, receipt)
