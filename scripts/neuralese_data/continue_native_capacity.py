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
    'train/native_source_preparation.py', 'data/native_source_tree.py',
    'rl/native_supervised.py', 'rl/production.py',
}
LOADER_IO_POLICY = 'native-resident-mmap-advice-v1'
LOADER_IO_CHANGES = METADATA_CHANGES | {
    'train/checkpoint_policy.py', 'train/checkpoint_readahead.py',
}
LINEAGE_METADATA_POLICY = 'native-saved-recipe-lineage-metadata-v1'
LINEAGE_METADATA_FILE = 'train/native_state_sft.py'
LINEAGE_METADATA_COMPOSITION = (
    'ca5acf17bbcdfd0eaeff2712e9af07a990960d8235c6ab5bb89ae2625d04321e',
    '14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8',
)
POLICY_GATE_SFT_POLICY = 'native-frozen-policy-gate-ordinary-sft-v1'
POLICY_GATE_SFT_COMPOSITION = {
    'eval/native_memory.py': ('69885bd62fdfd4d2bdc73e5d4ac2f3c41aa4a006ecc4a72bd8125b1b2735d392', '05b23b60f54795f48261515022ee4fc6b01eedd5e87bbfcdfec98f3510e7dbeb'),
    'rl/native_replay.py': ('976e066cbe4103fd67a75bf97547a9a2b5f12398223a28e9686330e27522b0e2', '99340bac53db80afdc8fcc3d8f2d9beba73c42ad0a2813857ef80cd2abbd78dd'),
    'rl/native_stage.py': ('6dfa075691aae40bb97a7f5ed495c7ea1f29b257f05072fa1e217224840c4ac8', '9e8559a9e96c4ed94194a6b3bf0c9803b308d475380e0485f9b9c6e2c3997f8f'),
    'rl/production.py': ('207e44e05ccd970819695fddec66851fea8ee7c7d46acb11f7f5f3051c18e7f4', '6a02e01c1b0d623001ed19c34baed867ec456b74a89dcdf4f0e952cca99288bc'),
    'train/native_state_sft.py': ('14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8', 'c2b1688724eed94117f2ea1019bd7354579f5aeaeec0390719dac051e0fc461f'),
}


ORDINARY_WRITER_POLICY = 'native-frozen-ordinary-sft-report-writer-v1'
ORDINARY_WRITER_COMPOSITION = {
    'train/native_state_sft.py': (
        '14e903b5baf160a35f7d33b75eafa83bc5a79a466aef4737a956c8d244abf9e8',
        'dd38e191c06500435e3cff556c09e3bfe7aa88b2460c0232f5b4221dc283cceb'),
    'train/native_operation_report.py': (None,
        '9243caf33e1a2e015c4cfca99c1600c418eaf339fa3900b3c19e4e756908a633'),
}


def _ordinary_writer_contract(review):
    if review['optimizer_policy'] != ORDINARY_WRITER_POLICY:
        return
    if (review.get('identity', {}).get('objective') is not None
            or review.get('fresh_policy_capacity_required') is not True):
        raise ValueError('report-writer continuation is ordinary SFT only; actual policy capacity remains required')
    changes = {name for name in review['old_code'].keys() | review['new_code'].keys()
               if review['old_code'].get(name) != review['new_code'].get(name)}
    if (changes != set(ORDINARY_WRITER_COMPOSITION)
            or any((review['old_code'].get(name), review['new_code'].get(name)) != pair
                   for name, pair in ORDINARY_WRITER_COMPOSITION.items())):
        raise ValueError('report-writer continuation requires the exact two-file frozen ordinary SFT composition')


def _policy_gate_sft_contract(review):
    if review['optimizer_policy'] != POLICY_GATE_SFT_POLICY:
        return
    if (review.get('identity', {}).get('objective') is not None
            or review.get('fresh_policy_capacity_required') is not True):
        raise ValueError('policy-gate continuation is ordinary SFT only; actual policy capacity remains required')
    changes = {name for name in review['old_code'].keys() | review['new_code'].keys()
               if review['old_code'].get(name) != review['new_code'].get(name)}
    if (changes != set(POLICY_GATE_SFT_COMPOSITION)
            or any((review['old_code'].get(name), review['new_code'].get(name)) != pair
                   for name, pair in POLICY_GATE_SFT_COMPOSITION.items())):
        raise ValueError('ordinary SFT continuation requires the exact five-file frozen policy-gate composition')


ATTENTION_INDEX_POLICY = 'native-graph-attention-index-reuse-v1'
ATTENTION_INDEX_REVIEW_SHA256 = 'e100c0814676e340b6c2bf3341dee101d02958a249757304ca3fb2df16a0b7a2'
ATTENTION_INDEX_COMPOSITIONS = {
    'lfm': ('14ca1758cacf99a4dc8f237fea7b8005d2123c40',
            '21d878c9c8e7b1345d813b0827b05513cf8c9b55',
            'cac5df15b82939c4d618e70cdf1b82db00dd2fbf66a73e0ee5a3985eb6ee369d',
            '7cb9ab4fb9ea7e4c882f9021c0dc81507e33c4ecfde1e85ff4024f58c3644693'),
    'mellum': ('b50e8823aaa1f92d2d3be853afc1955948a8fd1f',
               '92c7b4ae03d13baab1d88f83df20c0b185f9439e',
               '0dd5044fe1a3d232c4160edbe9da92240c18f3341178d0f252503043bae5078a',
               'd663389095b4fe6fa4d6ec991f2a9c6f47697995f4d887978d0e57fb13b5f96d'),
}


def allowed_changes(policy):
    return {POLICY: ALLOWED_CHANGES, METADATA_POLICY: METADATA_CHANGES,
            LOADER_IO_POLICY: LOADER_IO_CHANGES,
            LINEAGE_METADATA_POLICY: {LINEAGE_METADATA_FILE},
            POLICY_GATE_SFT_POLICY: set(POLICY_GATE_SFT_COMPOSITION),
            ORDINARY_WRITER_POLICY: set(ORDINARY_WRITER_COMPOSITION),
            ATTENTION_INDEX_POLICY: {'model/shared_state.py'}}.get(policy, set())


def _lineage_metadata_contract(review):
    if review['optimizer_policy'] != LINEAGE_METADATA_POLICY:
        return
    actual = (review['old_code'].get(LINEAGE_METADATA_FILE),
              review['new_code'].get(LINEAGE_METADATA_FILE))
    if actual != LINEAGE_METADATA_COMPOSITION:
        raise ValueError('saved recipe metadata continuation requires the exact independently reviewed trainer repair')


def _attention_index_contract(review, limits):
    if review['optimizer_policy'] != ATTENTION_INDEX_POLICY:
        return
    basis = review.get('attention_index_basis') or {}
    composition = ATTENTION_INDEX_COMPOSITIONS.get(basis.get('model'))
    reference = basis.get('review') or {}
    if composition is None or reference.get('sha256') != ATTENTION_INDEX_REVIEW_SHA256:
        raise ValueError('attention index continuation requires its exact independently approved frozen review')
    approved = json.loads(bound(reference).read_text())
    parent, candidate, old_sha, new_sha = composition
    report = next((row for row in approved.get('reports', []) if row.get('model') == basis['model']), {})
    if (basis.get('parent_commit') != parent or basis.get('candidate_commit') != candidate
            or report.get('parent') != parent or report.get('candidate') != candidate
            or report.get('shared_state_sha256') != new_sha
            or review['old_code'].get('model/shared_state.py') != old_sha
            or review['new_code'].get('model/shared_state.py') != new_sha):
        raise ValueError('attention index continuation changed the exact reviewed composition')
    regressions = json.loads(bound(basis.get('cuda_regressions')).read_text())
    expected = {(values[1], port) for values in ATTENTION_INDEX_COMPOSITIONS.values()
                for port in ('lfm', 'mellum')}
    cases = regressions.get('cases') or []
    if (regressions.get('schema') != 'natlang.attention-index-cuda-regressions/1'
            or len(cases) != 4 or {(row.get('candidate_commit'), row.get('port')) for row in cases} != expected):
        raise ValueError('attention index continuation requires all four frozen both-port CUDA cases')
    for case in cases:
        if any(case.get(key) is not True for key in ('passed', 'cuda', 'outputs_bitwise',
                'all_named_adjoints_bitwise', 'checkpoint_recompute', 'distinct_graph_lifetime')):
            raise ValueError('attention index CUDA regression is not strict bitwise qualification')
        bound(case.get('log'))
    paired = json.loads(bound(basis.get('normal_batch_paired')).read_text())
    normal = paired.get('batch') or {}
    if (paired.get('schema') != 'natlang.attention-index-production-paired/1'
            or paired.get('passed') is not True or paired.get('identity') != review['identity']
            or paired.get('old_runtime_code_sha256') != digest(review['old_code'])
            or paired.get('new_runtime_code_sha256') != digest(review['new_code'])
            or paired.get('parent_checkpoint_sha256') != basis.get('normal_probe_parent_checkpoint_sha256')
            or not isinstance(paired.get('parent_checkpoint_sha256'), str)
            or len(paired['parent_checkpoint_sha256']) != 64
            or paired.get('optimizer_updates') != 0
            or any(paired.get(key) is not True for key in ('cuda_synchronized', 'outputs_bitwise',
                'loss_bitwise', 'all_named_adjoints_bitwise', 'optimizer_state_unchanged',
                'rng_restored', 'source_credits_unchanged'))):
        raise ValueError('attention index continuation lacks an exact unchanged-state production paired result')
    for key in ('forward_tokens', 'kv_slots'):
        if type(normal.get(key)) is not int or not 0 < normal[key] <= limits[key]:
            raise ValueError('attention index paired batch differs from the inherited geometry')
    if normal != basis.get('normal_batch_geometry'):
        raise ValueError('attention index paired batch differs from the independently reviewed normal batch')
    bound(paired.get('source'))
    for key in ('old_seconds', 'new_seconds'):
        value = paired.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError('attention index paired timings require actual synchronized positive durations')
    memory = basis.get('index_memory') or {}
    old_count = memory.get('old_attention_index_allocations')
    index_bytes = limits['forward_tokens'] * 8
    extra = review.get('metadata_extra_bytes')
    if (type(old_count) is not int or old_count < 1
            or memory.get('new_attention_index_allocations') != 1
            or memory.get('index_dtype') != 'torch.int64'
            or memory.get('max_index_elements') != limits['forward_tokens']
            or memory.get('new_index_bytes') != index_bytes
            or memory.get('old_index_bytes') != old_count * index_bytes
            or type(memory.get('host_workspace_extra_bytes_upper_bound')) is not int
            or type(extra) is not int or extra < (1 << 30)
            or not 0 <= memory['host_workspace_extra_bytes_upper_bound'] <= extra):
        raise ValueError('attention index continuation requires conservative explicit index/workspace memory bounds')


def _memory_allowance(review):
    parameters = review.get('ordinary_parameters') or []
    if not parameters or len({p['name'] for p in parameters}) != len(parameters):
        raise ValueError('capacity continuation requires distinct actual ordinary parameters')
    for parameter in parameters:
        if (parameter.get('dtype') != 'torch.float32' or not parameter.get('shape')
                or any(type(n) is not int or n < 1 for n in parameter['shape'])):
            raise ValueError('capacity continuation requires the unchanged FP32 ordinary layout')
    if review['optimizer_policy'] in {METADATA_POLICY, LOADER_IO_POLICY, ATTENTION_INDEX_POLICY, LINEAGE_METADATA_POLICY, POLICY_GATE_SFT_POLICY, ORDINARY_WRITER_POLICY}:
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
        allowed = allowed_changes(policy)
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
        _attention_index_contract(review, value['limits'])
        _lineage_metadata_contract(review)
        _policy_gate_sft_contract(review)
        _ordinary_writer_contract(review)
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
            or review.get('optimizer_policy') not in {POLICY, METADATA_POLICY, LOADER_IO_POLICY, ATTENTION_INDEX_POLICY, LINEAGE_METADATA_POLICY, POLICY_GATE_SFT_POLICY, ORDINARY_WRITER_POLICY}):
        raise ValueError('capacity continuation requires an approved exact optimizer review')
    old, new = code_inventory(old_package), code_inventory(new_package)
    if old != review.get('old_code') or new != review.get('new_code'):
        raise ValueError('capacity continuation runtime inventory differs from reviewed bytes')
    changes = {name: {'old': old.get(name), 'new': new.get(name)}
        for name in old.keys() | new.keys() if old.get(name) != new.get(name)}
    policy = review['optimizer_policy']
    allowed = allowed_changes(policy)
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
    _attention_index_contract(review, limits)
    _lineage_metadata_contract(review)
    _policy_gate_sft_contract(review)
    _ordinary_writer_contract(review)
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
            ('optimizer/telemetry-only' if policy == POLICY else
             'exact frozen attention-index-only' if policy == ATTENTION_INDEX_POLICY else
             'exact frozen saved-recipe-metadata-only' if policy == LINEAGE_METADATA_POLICY else
             'exact frozen policy-gate ordinary-SFT-only (fresh policy measurement required)' if policy == POLICY_GATE_SFT_POLICY else
             'exact frozen ordinary-SFT report-writer-only (fresh policy measurement required)' if policy == ORDINARY_WRITER_POLICY else
             'CPU scheduling/pending-obligation-only') +
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
