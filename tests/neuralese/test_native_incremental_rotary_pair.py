"""The serving optimization retains independent rotary outputs, captures and gradients."""
import copy
import pytest
import torch

from test_shared_state import tiny
from natlang_neuralese.model import shared_state


@pytest.mark.parametrize('family', ['lfm', 'mellum', 'qwen'])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_cached_rotary_pair_matches_independent_execution(monkeypatch, family, dtype, device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(110)
    port = tiny(family).to(device=device, dtype=dtype)
    vocab = port.embedding_weight.shape[0]
    fast = shared_state.SharedStateModel(port, tuple(range(vocab-4, vocab)), rank=4)
    torch.nn.init.normal_(fast.adapter.up.weight, std=.03)
    slow = copy.deepcopy(fast)
    counts = {'fast': 0, 'slow': 0}
    for label, model in [('fast', fast), ('slow', slow)]:
        rope = model.port._rope
        def counted(*args, _rope=rope, _label=label, **kwargs):
            counts[_label] += 1
            return _rope(*args, **kwargs)
        monkeypatch.setattr(model.port, '_rope', counted)

    def execute(model):
        cache = None
        outputs = []
        for ids, producing in [([2, 3, 4], False), ([5], True), ([6, 7], False)]:
            row = shared_state.native_step(model, torch.tensor([ids], device=device), cache, producing=producing)
            cache = row['cache']
            outputs.append(row)
        return outputs

    actual = execute(fast)
    def independent(port, query, key, positions, layer, *, tables=None):
        # The old independent execution is the reference, including its two
        # positional table calls and discarded paired rotary outputs.
        def one(tensor):
            dummy = tensor.new_empty(1, len(positions), port.embedding_weight.shape[1])
            cos, sin = port._rope(dummy, positions[None])
            if hasattr(port, '_apply_rope'):
                return port._apply_rope(layer, tensor, tensor, cos, sin)[0]
            from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb
            return apply_rotary_pos_emb(tensor, tensor, cos, sin)[0]
        return one(query), one(key)
    monkeypatch.setattr(shared_state, '_rotate_pair', independent)
    # The explicit old path constructs its own two tables per layer. Avoid
    # counting the new preparation, which this reference intentionally ignores.
    monkeypatch.setattr(shared_state, '_rotary_tables', lambda *args: (None, None))
    expected = execute(slow)
    assert counts['slow'] == 6*sum(port.is_attention(i) for i in range(port.num_layers))
    assert counts['fast'] == 3
    for a, b in zip(actual, expected):
        for name in ['h_final', 'embeddings', 'positions', 'token_ids']:
            torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
        assert a['cache'].lengths == b['cache'].lengths
        for left, right in zip(a['layers'], b['layers']):
            assert left['kind'] == right['kind']
            for name in left.keys()-{'kind'}:
                torch.testing.assert_close(left[name], right[name], rtol=0, atol=0)
    weights = torch.randn_like(torch.cat([x['h_final'] for x in actual], dim=1))
    for model, outputs in [(fast, actual), (slow, expected)]:
        (torch.cat([x['h_final'] for x in outputs], dim=1)*weights).float().sum().backward()
    for (name, a), (other, b) in zip(fast.named_parameters(), slow.named_parameters()):
        assert name == other
        assert (a.grad is None) == (b.grad is None), name
        if a.grad is not None:
            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0, msg=name)
