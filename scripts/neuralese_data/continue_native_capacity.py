#!/usr/bin/env python3
"""Verify an explicitly reviewed optimizer or CPU-metadata capacity continuation.

This does not run a model and never claims a new parent was memory-qualified.
The independently approved review is policy; this tool verifies its exact bounds.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]/'training/neuralese'))
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import write_json_durable
from natlang_neuralese.train.native_resident import binding, bound, code_inventory

ALLOWED_CHANGES = {
    'train/native_optimizer_migration.py', 'train/native_semantic_stage.py',
    'train/native_state_sft.py', 'train/optim.py', 'train/optim_restore.py',
    'train/recipe.py',
}
POLICY = 'native-ordinary-to-adamw-new-history-v1'
METADATA_POLICY = 'native-scheduling-pending-obligations-v1'
METADATA_CHANGES = {
    'train/native_state_sft.py', 'train/recipe.py', 'train/native_source_streams.py',
    'train/native_pending_sources.py', 'train/native_continuation.py',
    'train/native_readiness.py', 'train/native_resident.py',
    'rl/native_supervised.py', 'rl/production.py',
}


def _memory_allowance(review):
    parameters = review.get('ordinary_parameters') or []
    if not parameters or len({p['name'] for p in parameters}) != len(parameters):
        raise ValueError('capacity continuation requires distinct actual ordinary parameters')
    for parameter in parameters:
        if (parameter.get('dtype') != 'torch.float32' or not parameter.get('shape')
                or any(type(n) is not int or n < 1 for n in parameter['shape'])):
            raise ValueError('capacity continuation requires the unchanged FP32 ordinary layout')
    if review['optimizer_policy'] == METADATA_POLICY:
        extra = review.get('metadata_extra_bytes')
        if (review.get('optimizer_state_unchanged') is not True
                or type(extra) is not int or extra < (64 << 20)):
            raise ValueError('metadata continuation requires unchanged optimizer state and >=64MiB explicit working-memory allowance')
        return extra
    return 64 * sum(math.prod(p['shape']) for p in parameters)


def _proof_cases(reference, *, identity, code_sha, parent, seen=()):
    value = json.loads(bound(reference).read_text())
    if reference['sha256'] in seen:
        raise ValueError('capacity continuation evidence contains a cycle')
    seen = (*seen, reference['sha256'])
    if value.get('schema') == 'natlang.native-capacity-continuation/1':
        review = json.loads(bound(value['review']).read_text())
        certificate = json.loads(bound(value['source_certificate']).read_text())
        changes = {name: {'old': review['old_code'].get(name), 'new': review['new_code'].get(name)}
            for name in review['old_code'].keys() | review['new_code'].keys()
            if review['old_code'].get(name) != review['new_code'].get(name)}
        policy = review.get('optimizer_policy')
        allowed = ALLOWED_CHANGES if policy == POLICY else METADATA_CHANGES if policy == METADATA_POLICY else set()
        if (value.get('passed') is not True or value.get('identity') != identity
                or value.get('runtime_code_sha256') != code_sha or value.get('optimizer_updates') != 0
                or value.get('new_parent_measured') is not False
                or value.get('continued_parent_checkpoint_sha256') != parent
                or review.get('schema') != 'natlang.native-capacity-continuation-review/1'
                or review.get('approved') is not True or review.get('parameter_layout_unchanged') is not True
                or value.get('continuation_policy', policy) != policy
                or not changes or set(changes) - allowed or changes != review.get('changes')
                or digest(review['new_code']) != code_sha
                or digest(review['old_code']) != value.get('old_runtime_code_sha256')
                or review.get('identity') != identity
                or review.get('source_certificate') != value['source_certificate']
                or review.get('new_parent_checkpoint_sha256') != parent
                or review.get('old_parent_checkpoint_sha256') != value.get('source_parent_checkpoint_sha256', value.get('measured_parent_checkpoint_sha256'))
                or certificate.get('passed') is not True or certificate.get('identity') != identity
                or certificate.get('schema') != 'natlang.native-resident-geometry-qualification/1'
                or certificate.get('runtime_code_sha256') != value['old_runtime_code_sha256']
                or certificate.get('evidence') != value.get('source_evidence')
                or certificate.get('limits') != value.get('limits')):
            raise ValueError('inherited capacity continuation differs from its approved evidence chain')
        cases = _proof_cases(value['source_evidence'], identity=identity,
            code_sha=value['old_runtime_code_sha256'],
            parent=value.get('source_parent_checkpoint_sha256', value['measured_parent_checkpoint_sha256']), seen=seen)
        origin = json.loads(bound(value['source_evidence']).read_text()).get(
            'measured_parent_checkpoint_sha256', review['old_parent_checkpoint_sha256'])
        if value['measured_parent_checkpoint_sha256'] != origin:
            raise ValueError('inherited capacity changed its actual measured origin')
        if {key: max(case[key] for case in cases) for key in ('forward_tokens', 'kv_slots')} != value['limits']:
            raise ValueError('inherited capacity limits differ from actual source cases')
        extra = value.get('extra_bytes_per_memory_domain')
        if type(extra) is not int or extra != _memory_allowance(review):
            raise ValueError('inherited capacity lost its additional memory allowance')
        return [{**case, 'inherited_extra_bytes': case.get('inherited_extra_bytes', 0) + extra} for case in cases]
    if (value.get('passed') is not True or value.get('identity') != identity
            or value.get('runtime_code_sha256') != code_sha
            or value.get('parent_checkpoint_sha256') != parent
            or value.get('optimizer_updates') != 0):
        raise ValueError('capacity source proof differs from its measured parent/runtime')
    if value.get('schema') == 'natlang.native-memory-combined-capacity/1':
        children = value.get('proofs') or []
        cases = [case for child in children for case in _proof_cases(child,
            identity=identity, code_sha=code_sha, parent=parent, seen=seen)]
        if not cases or cases != (value.get('capacity') or {}).get('cases'):
            raise ValueError('combined source cases differ from their actual bound proofs')
        return cases
    if (value.get('schema') != 'natlang.native-memory-qualification/1'
            or (value.get('numerical_reference') or {}).get('passed') is not True
            or (value.get('capacity') or {}).get('passed') is not True):
        raise ValueError('continuation requires actual numerical and capacity evidence')
    cases = value['capacity'].get('cases') or []
    if not cases or any(case.get('status') != 'passed' for case in cases):
        raise ValueError('continuation requires successful actual whole-graph cases')
    if any('inherited_extra_bytes' in case for case in cases):
        raise ValueError('raw measured cases cannot declare inherited memory allowances')
    return cases


def continue_capacity(review_binding, *, old_package, new_package, out):
    review = json.loads(bound(review_binding).read_text())
    if (review.get('schema') != 'natlang.native-capacity-continuation-review/1'
            or review.get('approved') is not True
            or review.get('parameter_layout_unchanged') is not True
            or review.get('optimizer_policy') not in {POLICY, METADATA_POLICY}):
        raise ValueError('capacity continuation requires an approved exact optimizer review')
    old, new = code_inventory(old_package), code_inventory(new_package)
    if old != review.get('old_code') or new != review.get('new_code'):
        raise ValueError('capacity continuation runtime inventory differs from reviewed bytes')
    changes = {name: {'old': old.get(name), 'new': new.get(name)}
        for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    policy = review['optimizer_policy']
    allowed = ALLOWED_CHANGES if policy == POLICY else METADATA_CHANGES
    if not changes or set(changes) - allowed or changes != review.get('changes'):
        raise ValueError('capacity continuation contains unreviewed forward/runtime changes')
    identity = review['identity']
    source = json.loads(bound(review['source_certificate']).read_text())
    if (source.get('schema') != 'natlang.native-resident-geometry-qualification/1'
            or source.get('passed') is not True or source.get('identity') != identity
            or source.get('runtime_code_sha256') != digest(old)):
        raise ValueError('capacity identity changed: model/packing/offload/backend must match')
    cases = _proof_cases(source['evidence'], identity=identity, code_sha=digest(old),
        parent=review['old_parent_checkpoint_sha256'])
    limits = {key: max(case[key] for case in cases) for key in ('forward_tokens', 'kv_slots')}
    if (source.get('limits') != limits or any(type(v) is not int or v < 1 for v in limits.values())):
        raise ValueError('capacity limits differ from actual source cases')
    # Adam first/second moments, retained old buffers, telemetry snapshots and
    # per-parameter update intermediates. No backbone-sized copy is authorized.
    extra = _memory_allowance(review)
    reserve = review.get('reserve_bytes')
    device_limit, host_limit = review.get('device_memory_bytes'), review.get('host_memory_limit_bytes')
    if (type(reserve) is not int or reserve < (1 << 30)
            or any(type(n) is not int or n <= 0 for n in (device_limit, host_limit))
            or identity.get('runtime', {}).get('total_memory_bytes') != device_limit):
        raise ValueError('capacity continuation requires actual hardware limits and >=1GiB reserve')
    peaks = []
    for case in cases:
        device_peak = case.get('cuda_peak_allocated_bytes')
        host_peak = (case.get('host_peak') or {}).get('cgroup_current_bytes')
        if any(type(n) is not int or n <= 0 for n in (device_peak, host_peak)):
            raise ValueError('capacity continuation requires measured device and cgroup peaks')
        inherited_extra = case.get('inherited_extra_bytes', 0)
        if device_peak + inherited_extra + extra + reserve >= device_limit or host_peak + inherited_extra + extra + reserve >= host_limit:
            raise ValueError('capacity continuation exceeds measured memory headroom')
        peaks.append({'id': case['id'], 'device_bytes': device_peak, 'host_bytes': host_peak,
                      **({'inherited_extra_bytes': inherited_extra} if inherited_extra else {})})
    result = {'schema': 'natlang.native-capacity-continuation/1', 'passed': True,
        'review': review_binding, 'source_certificate': review['source_certificate'],
        'source_evidence': source['evidence'], 'identity': identity,
        'runtime_code_sha256': digest(new), 'old_runtime_code_sha256': digest(old),
        'measured_parent_checkpoint_sha256': json.loads(bound(source['evidence']).read_text()).get(
            'measured_parent_checkpoint_sha256', review['old_parent_checkpoint_sha256']),
        'source_parent_checkpoint_sha256': review['old_parent_checkpoint_sha256'],
        'continued_parent_checkpoint_sha256': review['new_parent_checkpoint_sha256'],
        'new_parent_measured': False, 'extra_bytes_per_memory_domain': extra,
        'continuation_policy': policy,
        'reserve_bytes': reserve, 'measured_peaks': peaks, 'optimizer_updates': 0,
        'task_quality_qualified': False, 'limits': limits}
    directory = Path(out)/digest(result)
    directory.mkdir(parents=True, exist_ok=True)
    evidence, certificate = directory/'continuation.json', directory/'capacity.json'
    def immutable(path, value):
        if path.exists():
            if json.loads(path.read_text()) != value:
                raise ValueError('immutable capacity continuation changed')
        else:
            write_json_durable(path, value)
    immutable(evidence, result)
    immutable(certificate, {'schema': 'natlang.native-resident-geometry-qualification/1',
        'passed': True, 'identity': identity, 'runtime_code_sha256': digest(new),
        'limits': limits, 'evidence': binding(evidence),
        'scope': 'inherited actual capacity through explicitly reviewed ' +
            ('optimizer/telemetry-only' if policy == POLICY else 'CPU scheduling/pending-obligation-only') +
            ' continuation; new parent not measured'})
    return certificate


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, help='JSON exact path/SHA binding to independently approved review')
    parser.add_argument('--old-package', type=Path, required=True)
    parser.add_argument('--new-package', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(binding(continue_capacity(json.loads(args.review),
        old_package=args.old_package, new_package=args.new_package, out=args.out))))
