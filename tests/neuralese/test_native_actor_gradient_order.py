from types import SimpleNamespace
import torch
from torch import nn
import pytest

from natlang_neuralese.model.native_gradient_partitions import GradientPartitions
from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope, stored_linear
from natlang_neuralese.model.native_graph_plan import Invocation, compile_graph
import natlang_neuralese.model.shared_state as shared


class FakeBank:
    def __init__(self):
        self.active = None
        self.validated = False
        self.calls = []
    def row_groups(self, plan, device):
        self.validated = True
        # The old bucketed dispatch grouped invocations A0 and A2.
        return [('A0', torch.tensor([0, 2], device=device)), ('B1', torch.tensor([1], device=device))]
    def execute(self, name, fn):
        self.calls.append(name)
        old = self.active
        self.active = name
        try:
            return fn()
        finally:
            self.active = old
    def receipt(self):
        return {'test': True}


class FakePort:
    def __init__(self):
        self.embedding_weight = torch.zeros((32, 1), dtype=torch.bfloat16)
        self.layers = [SimpleNamespace(conv=SimpleNamespace(conv=SimpleNamespace(weight=torch.zeros((1, 1, 1)))))]
        self.num_layers = 1
        self.checkpoint_layers = False
        self.ffn_chunk_tokens = 0
        self.native_attention_precision = 'ieee'
    def embed(self, ids):
        return torch.zeros((1, ids.shape[1], 1), dtype=torch.bfloat16)
    def is_attention(self, index): return False
    def window(self, index): return None


def test_shared_state_actor_dispatch_preserves_interleaved_cotangent_after_prior_microbatch(monkeypatch, tmp_path):
    port = FakePort()
    bank = FakeBank()
    monkeypatch.setattr(shared, '_external_layers', lambda *args: ())
    monkeypatch.setattr(shared, '_convolve', lambda layer, gated, c, plan, rows, **kwargs: gated)
    monkeypatch.setattr(shared, '_finish', lambda port, layer, h, out, attention: h + out)
    import natlang_neuralese.model.native_actor_graph as actor_graph
    monkeypatch.setattr(actor_graph, 'invocation_actor_bank', lambda port_arg: bank)

    weight = nn.Parameter(torch.zeros((1, 1), dtype=torch.bfloat16))
    def layer_inputs(_port, _layer, h, _attention):
        name = bank.active
        if name == 'A0':
            factors = [2.0] if h.shape[1] == 1 else [2.0, 1.0]
        elif name == 'B1':
            factors = [-1.0e8]
        elif name == 'A2':
            factors = [1.0]
        else:
            raise AssertionError(name)
        x = torch.tensor(factors, dtype=torch.bfloat16).reshape(1, -1, 1)
        gated = stored_linear(x, weight, storage_dtype=weight.dtype)
        return gated, torch.zeros_like(gated)
    monkeypatch.setattr(shared, '_layer_inputs', layer_inputs)

    plan = compile_graph([Invocation('A0', (1,)), Invocation('B1', (2,)), Invocation('A2', (3,))])
    with ParameterCotangentScope([('shared.weight', weight)], mode='update', directory=tmp_path) as scope:
        prior = stored_linear(torch.ones((1, 1, 1), dtype=torch.bfloat16), weight, storage_dtype=weight.dtype)
        prior.backward(torch.tensor([[[1.0e8]]], dtype=torch.bfloat16))
        scope.new_backward()
        result = shared.forward_graph(port, plan)
        result['h_final'].float().sum().backward()
        scope.finish()
    assert bank.validated
    assert bank.calls == ['A0', 'B1', 'A2', 'A0', 'B1', 'A2'], bank.calls
    # Chronological cached-reference fold: prior, A2, B1, A0 in FP32.
    # 1e8 + 1 rounds to 1e8, then -1e8 cancels, then +2 remains.
    assert weight.grad.item() == 2.0, weight.grad
