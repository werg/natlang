"""Only the two independently approved index-only compositions can inherit."""
import json

import pytest

from test_native_capacity_continuation import fixture, invoke, mechanism, binding, code_inventory, digest


def attention_fixture(tmp_path, monkeypatch):
    old, new, review, proof = fixture(tmp_path)
    (new/'train/optim.py').write_bytes((old/'train/optim.py').read_bytes())
    for package, text in [(old, 'old index allocation\n'), (new, 'reviewed graph-local index\n')]:
        (package/'model/shared_state.py').write_text(text)
    review['old_code'], review['new_code'] = code_inventory(old), code_inventory(new)
    review['changes'] = {'model/shared_state.py': {
        'old': review['old_code']['model/shared_state.py'], 'new': review['new_code']['model/shared_state.py']}}
    raw = json.loads(proof.read_text()); raw['runtime_code_sha256'] = digest(review['old_code'])
    proof.write_text(json.dumps(raw))
    cert = tmp_path/'source-capacity.json'; raw = json.loads(cert.read_text())
    raw.update(runtime_code_sha256=digest(review['old_code']), evidence=binding(proof))
    cert.write_text(json.dumps(raw)); review['source_certificate'] = binding(cert)
    composition = ('a'*40, 'b'*40, review['old_code']['model/shared_state.py'],
                   review['new_code']['model/shared_state.py'])
    monkeypatch.setattr(mechanism, 'ATTENTION_INDEX_COMPOSITIONS', {
        'lfm': composition, 'mellum': ('c'*40, 'd'*40, 'e'*64, 'f'*64)})
    independent = tmp_path/'independent.json'
    independent.write_text(json.dumps({'reports': [{'model': 'lfm', 'parent': composition[0],
        'candidate': composition[1], 'shared_state_sha256': composition[3]}]}))
    monkeypatch.setattr(mechanism, 'ATTENTION_INDEX_REVIEW_SHA256', binding(independent)['sha256'])
    log = tmp_path/'cuda.log'; log.write_text('actual strict CUDA regression log fixture\n')
    regressions = tmp_path/'regressions.json'
    regressions.write_text(json.dumps({'schema': 'natlang.attention-index-cuda-regressions/1', 'cases': [
        {'candidate_commit': values[1], 'port': port, 'passed': True, 'cuda': True,
         'outputs_bitwise': True, 'all_named_adjoints_bitwise': True,
         'checkpoint_recompute': True, 'distinct_graph_lifetime': True, 'log': binding(log)}
        for values in mechanism.ATTENTION_INDEX_COMPOSITIONS.values() for port in ('lfm', 'mellum')]}))
    source = tmp_path/'source.jsonl'; source.write_text('source-bound normal batch fixture\n')
    normal = {'forward_tokens': 1024, 'kv_slots': 2000}
    paired = tmp_path/'paired.json'
    paired.write_text(json.dumps({'schema': 'natlang.attention-index-production-paired/1',
        'passed': True, 'identity': review['identity'], 'old_runtime_code_sha256': digest(review['old_code']),
        'new_runtime_code_sha256': digest(review['new_code']), 'parent_checkpoint_sha256': 'b'*64,
        'optimizer_updates': 0, 'cuda_synchronized': True, 'outputs_bitwise': True,
        'loss_bitwise': True, 'all_named_adjoints_bitwise': True, 'optimizer_state_unchanged': True,
        'rng_restored': True, 'source_credits_unchanged': True, 'batch': normal,
        'source': binding(source), 'old_seconds': 4.0, 'new_seconds': 3.9}))
    review.update(optimizer_policy=mechanism.ATTENTION_INDEX_POLICY, optimizer_state_unchanged=True,
                  metadata_extra_bytes=1 << 30, attention_index_basis={
        'model': 'lfm', 'parent_commit': composition[0], 'candidate_commit': composition[1],
        'review': binding(independent), 'cuda_regressions': binding(regressions),
        'normal_batch_paired': binding(paired), 'normal_probe_parent_checkpoint_sha256': 'b'*64,
        'normal_batch_geometry': normal, 'index_memory': {'old_attention_index_allocations': 4,
            'new_attention_index_allocations': 1, 'index_dtype': 'torch.int64',
            'max_index_elements': 1234, 'new_index_bytes': 1234*8, 'old_index_bytes': 4*1234*8,
            'host_workspace_extra_bytes_upper_bound': 64 << 20}})
    return old, new, review, paired, regressions


def test_exact_index_continuation_retains_original_max_and_latest_parent(tmp_path, monkeypatch):
    old, new, review, _, _ = attention_fixture(tmp_path, monkeypatch)
    result = json.loads(invoke(tmp_path, old, new, review).read_text())
    assert result['limits'] == {'forward_tokens': 1234, 'kv_slots': 2345}
    evidence = json.loads(mechanism.bound(result['evidence']).read_text())
    assert evidence['new_parent_measured'] is False
    assert evidence['measured_parent_checkpoint_sha256'] == 'a'*64
    assert evidence['continued_parent_checkpoint_sha256'] == 'b'*64
    assert evidence['extra_bytes_per_memory_domain'] == 1 << 30
    assert 'attention-index-only' in result['scope']


@pytest.mark.parametrize('mutation', ['review-hash', 'candidate', 'kernel-file', 'four-cases',
    'cpu-test', 'loose-gradients', 'missing-recompute', 'log-bytes', 'normal-runtime',
    'normal-parent', 'nonzero-update', 'changed-state', 'geometry', 'timing', 'index-growth',
    'workspace-bound', 'reserve'])
def test_index_contract_rejects_unproven_numerics_state_or_memory(tmp_path, monkeypatch, mutation):
    old, new, review, paired_path, regressions_path = attention_fixture(tmp_path, monkeypatch)
    basis = review['attention_index_basis']
    paired = json.loads(paired_path.read_text()); regressions = json.loads(regressions_path.read_text())
    if mutation == 'review-hash': basis['review']['sha256'] = '0'*64
    elif mutation == 'candidate': basis['candidate_commit'] = '0'*40
    elif mutation == 'kernel-file':
        (new/'model/forward.py').write_text('different kernel\n')
        review['new_code'] = code_inventory(new)
        review['changes']['model/forward.py'] = {'old': review['old_code']['model/forward.py'],
                                               'new': review['new_code']['model/forward.py']}
    elif mutation == 'four-cases': regressions['cases'].pop()
    elif mutation == 'cpu-test': regressions['cases'][0]['cuda'] = False
    elif mutation == 'loose-gradients': regressions['cases'][0]['all_named_adjoints_bitwise'] = False
    elif mutation == 'missing-recompute': regressions['cases'][0]['checkpoint_recompute'] = False
    elif mutation == 'log-bytes': (tmp_path/'cuda.log').write_text('changed evidence')
    elif mutation == 'normal-runtime': paired['new_runtime_code_sha256'] = '0'*64
    elif mutation == 'normal-parent': paired['parent_checkpoint_sha256'] = '0'*64
    elif mutation == 'nonzero-update': paired['optimizer_updates'] = 1
    elif mutation == 'changed-state': paired['source_credits_unchanged'] = False
    elif mutation == 'geometry': paired['batch']['forward_tokens'] = 1235
    elif mutation == 'timing': paired['new_seconds'] = float('nan')
    elif mutation == 'index-growth': basis['index_memory']['new_attention_index_allocations'] = 2
    elif mutation == 'workspace-bound': basis['index_memory']['host_workspace_extra_bytes_upper_bound'] = 2 << 30
    else: review['metadata_extra_bytes'] = 64 << 20
    paired_path.write_text(json.dumps(paired)); regressions_path.write_text(json.dumps(regressions))
    basis.update(normal_batch_paired=binding(paired_path), cuda_regressions=binding(regressions_path))
    with pytest.raises(ValueError): invoke(tmp_path, old, new, review)
