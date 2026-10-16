"""Inherited memory evidence must not become a fictitious new qualification."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

from natlang_neuralese.train.native_resident import binding, code_inventory, geometry_capacity
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest

spec = importlib.util.spec_from_file_location('native_capacity_continuation',
    Path(__file__).resolve().parents[2]/'scripts/neuralese_data/continue_native_capacity.py')
mechanism = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mechanism)


def fixture(tmp_path):
    old, new = tmp_path/'old', tmp_path/'new'
    for root in (old, new):
        (root/'train').mkdir(parents=True)
        (root/'model').mkdir()
        (root/'model/forward.py').write_text('unchanged forward\n')
        (root/'train/optim.py').write_text('old routing\n')
    (new/'train/optim.py').write_text('reviewed corrected routing\n')
    codes = [code_inventory(root) for root in (old, new)]
    identity = {'checkpoint_input_offload': False, 'runtime': {'total_memory_bytes': 20 << 30}}
    cases = [{'id': 'actual maximum', 'status': 'passed', 'forward_tokens': 1234, 'kv_slots': 2345,
        'cuda_peak_allocated_bytes': 8 << 30, 'host_peak': {'cgroup_current_bytes': 4 << 30}}]
    proof = tmp_path/'proof.json'
    proof.write_text(json.dumps({'schema': 'natlang.native-memory-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(codes[0]),
        'parent_checkpoint_sha256': 'a'*64, 'optimizer_updates': 0,
        'numerical_reference': {'passed': True}, 'capacity': {'passed': True, 'cases': cases}}))
    cert = tmp_path/'source-capacity.json'
    cert.write_text(json.dumps({'schema': 'natlang.native-resident-geometry-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(codes[0]),
        'limits': {'forward_tokens': 1234, 'kv_slots': 2345}, 'evidence': binding(proof)}))
    review = {'schema': 'natlang.native-capacity-continuation-review/1', 'approved': True,
        'old_code': codes[0], 'new_code': codes[1], 'changes': {
            'train/optim.py': {'old': codes[0]['train/optim.py'], 'new': codes[1]['train/optim.py']}},
        'identity': identity, 'source_certificate': binding(cert),
        'old_parent_checkpoint_sha256': 'a'*64, 'new_parent_checkpoint_sha256': 'b'*64,
        'parameter_layout_unchanged': True, 'optimizer_policy': mechanism.POLICY,
        'ordinary_parameters': [{'name': 'learned role rows', 'shape': [4, 2304], 'dtype': 'torch.float32'}],
        'device_memory_bytes': 20 << 30, 'host_memory_limit_bytes': 16 << 30, 'reserve_bytes': 1 << 30}
    return old, new, review, proof


def invoke(tmp_path, old, new, review):
    path = tmp_path/'review.json'
    path.write_text(json.dumps(review))
    return mechanism.continue_capacity(binding(path), old_package=old, new_package=new, out=tmp_path/'out')


def test_mechanical_continuation_retains_measured_parent_and_bounds_extra_memory(tmp_path):
    old, new, review, _ = fixture(tmp_path)
    result = invoke(tmp_path, old, new, review)
    cert = json.loads(result.read_text())
    evidence = json.loads(Path(cert['evidence']['path']).read_text())
    assert evidence['new_parent_measured'] is False
    assert evidence['measured_parent_checkpoint_sha256'] == 'a'*64
    assert evidence['continued_parent_checkpoint_sha256'] == 'b'*64
    assert evidence['extra_bytes_per_memory_domain'] == 64*4*2304
    assert geometry_capacity(binding(result), identity=review['identity'], code=code_inventory(new)) == cert['limits']
    assert 'new parent not measured' in cert['scope']
    assert invoke(tmp_path, old, new, review) == result


@pytest.mark.parametrize('mutation', ['unapproved', 'offload', 'unreviewed', 'forward',
    'device', 'host', 'reserve', 'layout', 'parameters', 'source-parent', 'source-mutated', 'numerics', 'nonzero-update'])
def test_changed_execution_unbound_proof_or_memory_shortfall_cannot_inherit_capacity(tmp_path, mutation):
    old, new, review, proof = fixture(tmp_path)
    if mutation == 'unapproved': review['approved'] = False
    elif mutation == 'offload': review['identity'] = {**review['identity'], 'checkpoint_input_offload': True}
    elif mutation == 'unreviewed': (new/'train/optim.py').write_text('unreviewed edit\n')
    elif mutation == 'forward':
        (new/'model/forward.py').write_text('changed forward\n')
        review['new_code'] = code_inventory(new)
        review['changes']['model/forward.py'] = {'old': review['old_code']['model/forward.py'],
            'new': review['new_code']['model/forward.py']}
    elif mutation == 'device': review['device_memory_bytes'] = 8 << 30
    elif mutation == 'host': review['host_memory_limit_bytes'] = 4 << 30
    elif mutation == 'reserve': review['reserve_bytes'] = 0
    elif mutation == 'layout': review['parameter_layout_unchanged'] = False
    elif mutation == 'parameters': review['ordinary_parameters'] *= 2
    elif mutation == 'source-parent': review['old_parent_checkpoint_sha256'] = 'c'*64
    elif mutation == 'source-mutated': proof.write_text('{}')
    else:
        value = json.loads(proof.read_text())
        if mutation == 'numerics': value['numerical_reference']['passed'] = False
        else: value['optimizer_updates'] = 1
        proof.write_text(json.dumps(value))
        cert_path = Path(review['source_certificate']['path'])
        cert = json.loads(cert_path.read_text()); cert['evidence'] = binding(proof)
        cert_path.write_text(json.dumps(cert)); review['source_certificate'] = binding(cert_path)
    with pytest.raises(ValueError):
        invoke(tmp_path, old, new, review)
    assert not (tmp_path/'out').exists()


def test_combined_evidence_requires_exact_transitive_actual_case_union(tmp_path):
    old, new, review, proof = fixture(tmp_path)
    value = json.loads(proof.read_text())
    combined = tmp_path/'combined.json'
    combined.write_text(json.dumps({**value, 'schema': 'natlang.native-memory-combined-capacity/1',
        'proofs': [binding(proof)], 'capacity': {'cases': copy.deepcopy(value['capacity']['cases'])}}))
    cert_path = Path(review['source_certificate']['path'])
    cert = json.loads(cert_path.read_text()); cert['evidence'] = binding(combined)
    cert_path.write_text(json.dumps(cert)); review['source_certificate'] = binding(cert_path)
    assert invoke(tmp_path, old, new, review).is_file()
    value = json.loads(combined.read_text()); value['capacity']['cases'][0]['kv_slots'] += 1
    combined.write_text(json.dumps(value))
    cert['evidence'] = binding(combined); cert_path.write_text(json.dumps(cert))
    review['source_certificate'] = binding(cert_path)
    with pytest.raises(ValueError, match='actual bound proofs'):
        invoke(tmp_path, old, new, review)
