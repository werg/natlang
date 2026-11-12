"""Native CE mass diagnosis against the existing balanced text objective.

These tests do not adopt a policy or rewrite corpus weights. The exact suffixes
below are authored fixture provenance, never inferred from arbitrary magnitudes.
"""
import pytest
import torch
from torch.nn import functional as F

from natlang_neuralese.data.native_graph import SCHEMA, parse_graph, graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_graph_loss import packed_graph_loss
from natlang_neuralese.train.text_supervision import balanced_position_weights
from test_shared_state import tiny


def declared(encoding='ordinary-suffix-raw-v1', digest='a' * 64):
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY_SCHEMA
    return {'schema': POLICY_SCHEMA, 'sources': [
        {'sha256': digest, 'encoding': encoding, 'context_weight': .05}]}


def ordinary():
    return {'schema': SCHEMA, 'id': 'mass:ordinary', 'split': 'train',
        'source_groups': ['mass:train'], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [{'name': 'text', 'token_ids': [2] * 24 + [3, 4],
                   'target_weights': [0.] + [.05] * 23 + [1., 1.]}],
        'conversion': {'role_interpretation': 'ordinary-text',
                       'source_suffix_start': 24, 'context_weight': .05}}


def test_raw_point_zero_five_is_not_five_percent_context_mass():
    # A real observed shape: a short complete action after a long tool prompt.
    raw = torch.tensor([[0.] + [.05] * 6505 + [1.] * 35])
    target = torch.arange(raw.shape[1]) >= 6506
    assert float(raw[:, target].sum() / raw.sum()) < .10
    balanced = balanced_position_weights(raw, [6506], context_weight=.05, exclude=raw == 0)
    assert float(balanced[:, target].sum() / balanced.sum()) > .95
    assert balanced[0, 0] == 0 and (balanced[raw > 0] > 0).all()


@pytest.mark.parametrize('family', ['lfm'])
@pytest.mark.parametrize('policy', ['raw', 'balanced-reference'])
def test_complete_graph_ce_and_all_derivatives_match_declared_mass(family, policy):
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    row = {'schema': SCHEMA, 'id': 'source-mass-fixture', 'split': 'train',
        'source_groups': ['independently-authored-train'], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [2] * 24 + [3, 4],
             'target_weights': [0.] + [.05] * 23 + [1., 1.], 'producing': [[22, 26]]},
            {'name': 'consumer', 'token_ids': [5] * 20 + [6, 7],
             'target_weights': [0., 0.] + [.05] * 18 + [1., 1.],
             'imports': [{'at': 1, 'source': 'producer', 'start': 22, 'stop': 26}]}]}
    row['source_pairing'] = {'pair_id': 'actual-fixture-pair', 'payloads': [{'invocation': 'producer'}],
        'consumed_occurrences': [{'consumer_import_at': 1}], 'source_ids': ['writer', 'reader']}
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    effective, oracle = [], []
    for node, suffix in zip(row['nodes'], [24, 20]):
        raw = torch.tensor([node['target_weights']])
        valid = (raw > 0).float()
        target = valid * (torch.arange(raw.shape[1]) >= suffix)
        if policy == 'balanced-reference':
            actual = balanced_position_weights(raw, [suffix], context_weight=.05, exclude=raw == 0)
            actual *= raw.sum() / actual.sum()
            # Independent normalized derivative oracle, not the implementation.
            expected = (.05 * valid / valid.sum() + target / target.sum()) / 1.05
            expected *= raw.sum()
        else:
            actual, expected = raw, raw
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(), raw.sum())
        assert torch.equal(actual == 0, raw == 0)
        effective.extend(actual[0].tolist())
        oracle.extend(expected[0].tolist())
    from natlang_neuralese.train.native_supervision import apply_policy
    weighted = (apply_policy(example, row, source_sha256='a' * 64,
                             policy=declared('witnessed-pair-raw-v1'))
                if policy == 'balanced-reference' else example)
    assert torch.equal(torch.tensor(weighted.target_weights), torch.tensor(effective))
    if policy == 'raw':
        assert weighted.target_weights == example.target_weights
    result = model(example.plan, retain_states=False)
    queries, labels, weights = [], [], []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        for local in range(1, len(node.token_ids)):
            if oracle[offset + local]:
                queries.append(offset + local - 1)
                labels.append(node.token_ids[local])
                weights.append(oracle[offset + local])
    reference_weights = torch.tensor(weights)
    ce = F.cross_entropy(model.logits(result['h_final'][0, queries]).float(),
                         torch.tensor(labels), reduction='none')
    expected_loss = (ce * reference_weights).sum() / reference_weights.sum()
    expected_gradients = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    for loss in [graph_loss(model, weighted, checkpoint_layers=True, chunk_tokens=3)[0],
                 packed_graph_loss(model, [weighted], checkpoint_layers=True, chunk_tokens=3)[0]]:
        torch.testing.assert_close(loss, expected_loss)
        actual = torch.autograd.grad(loss, parameters, allow_unused=True)
        for observed, expected in zip(actual, expected_gradients):
            assert (observed is None) == (expected is None)
            if expected is not None:
                torch.testing.assert_close(observed, expected)
        assert any(gradient is not None and gradient.norm() > 0 for gradient in actual)


def test_only_explicit_exact_source_encoding_can_change_weights(tmp_path):
    import copy
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.native_supervision import apply_policy, load_policy
    row = ordinary()
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    policy = declared()
    with pytest.raises(ValueError, match='missing from explicit supervision policy'):
        apply_policy(example, row, source_sha256='b' * 64, policy=policy)
    assert apply_policy(example, row, source_sha256='b' * 64,
                        policy={**policy, 'require_eligible_sources': False}) is example
    with pytest.raises(ValueError, match='authenticated immutable source'):
        apply_policy(example, row, source_sha256=None, policy=policy)
    for mutation in ['missing-suffix', 'coefficient', 'approximate', 'already-balanced']:
        bad = copy.deepcopy(row)
        if mutation == 'missing-suffix':
            bad['conversion'].pop('source_suffix_start')
        elif mutation == 'coefficient':
            bad['conversion']['context_weight'] = .1
        elif mutation == 'approximate':
            bad['nodes'][0]['target_weights'][5] = .050001
        else:
            bad['conversion']['supervision_policy'] = {'name': 'segmented'}
        parsed = parse_graph(bad, tokenizer_fingerprint='f' * 64)
        with pytest.raises(ValueError):
            apply_policy(parsed, bad, source_sha256='a' * 64, policy=policy)
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    binding = {'path': str(path), 'sha256': sha256_file_hex(path)}
    assert load_policy(binding) == policy
    path.write_text(json.dumps(declared(digest='b' * 64)))
    with pytest.raises(ValueError, match='SHA-256'):
        load_policy(binding)
    row['conversion']['supervision_policy'] = {'weighting':'segmented_position_weights'}
    assert apply_policy(example, row, source_sha256='b' * 64, policy=policy) is example


def test_draw_time_policy_keeps_raw_row_binding_and_original_held_weights(tmp_path):
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.data.native_sources import NativeGraphSource
    source = tmp_path / 'graphs.jsonl'
    source.write_text(json.dumps(ordinary()) + '\n')
    raw = NativeGraphSource(source, 'f' * 64, required_split='train')
    binding = dict(raw.bindings)
    original = next(iter(raw))
    raw.set_ce_policy(declared(digest=sha256_file_hex(source)))
    balanced = next(iter(raw))
    assert raw.bindings == binding
    assert balanced.id == original.id and balanced.plan.invocations == original.plan.invocations
    assert balanced.target_weights != original.target_weights
    assert sum(balanced.target_weights[-2:]) / sum(balanced.target_weights) > .95
    # A second consumer with no selected policy (held/eval) sees immutable raw weights.
    held = NativeGraphSource(source, 'f' * 64, required_split='train')
    assert next(iter(held)).target_weights == original.target_weights


# Written for one optimizer update per packed batch; accumulation is tested in test_native_update_accumulation.py.
pytestmark = pytest.mark.usefixtures('per_batch_native_updates')
