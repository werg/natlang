"""Whole-graph and bounded-memory certificates for long FFN checkpointing."""
from contextlib import nullcontext
import pytest
import torch
from torch import nn
from test_native_inline_qat import base, graph, loss_of, model_with_points  # noqa: F401 (fixture)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA unavailable'))])
@pytest.mark.parametrize('backend', ['plain', 'vjp'])
@pytest.mark.parametrize('qat', [None, 'fp8', 'nvfp4'])
def test_lfm3_nested_graph_checkpoint_preserves_all_gradients(base, device, backend, qat, monkeypatch):
    from natlang_neuralese.model import lfm2_port
    from natlang_neuralese.model.native_weight_carrier import ParameterCotangentScope, plain_training_arithmetic
    torch.manual_seed(112)
    model, quantization = model_with_points(base, profile='ffn-stable-v1', device=device)
    model.port.checkpoint_layers = True
    model.port.checkpoint_layer_group_size = 4
    model.port.ffn_chunk_tokens = 2
    example = graph()
    plan = example.plan
    if qat == 'nvfp4':
        quantization.calibrate(lambda batch: loss_of(model, batch[0]), [[example]], 0)
    for p in model.parameters(): p.requires_grad_(True)
    params = tuple(p for p in model.parameters() if p.requires_grad)
    source = torch.randn(1, plan.token_count, model.port.embedding_weight.shape[1], device=device,
                         dtype=torch.bfloat16) * .1
    upstream = torch.randn_like(source) * .05
    def run(enabled):
        monkeypatch.setattr(lfm2_port, 'FFN_CHECKPOINT_MIN_TOKENS', 0 if enabled else 100000)
        x=source.detach().clone().requires_grad_()
        scope=(ParameterCotangentScope.for_model(model,mode='vjp',accumulator_backend='device_fp32' if device=='cuda' else 'cpu_file')
               if backend=='vjp' else plain_training_arithmetic())
        precision=quantization.context(qat, .5) if qat else nullcontext()
        with scope, precision:
            out=model(plan,embeddings=x,retain_states=False)['h_final']
            grads=torch.autograd.grad(out, (x,*params), upstream, allow_unused=True)
            if backend=='vjp': scope.finish()
        return out.detach(),tuple(None if g is None else g.detach() for g in grads)
    actual,expected=run(True),run(False)
    torch.testing.assert_close(actual[0],expected[0],atol=1e-4,rtol=0)
    for a,b in zip(actual[1],expected[1]):
        assert (a is None)==(b is None)
        if a is not None: torch.testing.assert_close(a,b,atol=1e-5,rtol=0)


def test_chunk_checkpoint_higher_order_preserves_reference(monkeypatch):
    from natlang_neuralese.model import lfm2_port
    from natlang_neuralese.common.native_higher_order import reference_arithmetic
    torch.manual_seed(3)
    class FF(nn.Module):
        def __init__(self):
            super().__init__();self.w1=nn.Linear(8,16);self.w3=nn.Linear(8,16);self.w2=nn.Linear(16,8)
        def forward(self,x):return self.w2(torch.nn.functional.silu(self.w1(x))*self.w3(x))
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.ffn_norm=nn.LayerNorm(8);self.post_ffn_norm=nn.LayerNorm(8);self.feed_forward=FF()
    layer=Layer();source=torch.randn(1,9,8)*.1
    def run(enabled):
        monkeypatch.setattr(lfm2_port,'FFN_CHECKPOINT_MIN_TOKENS',0 if enabled else 1000)
        x=source.clone().requires_grad_();params=(x,*layer.parameters())
        with reference_arithmetic():
            output=lfm2_port.feed_forward_residual(layer,x,2,sandwich=True)
            first=torch.autograd.grad(output.square().sum(),params,create_graph=True)
            second=torch.autograd.grad(sum(g.square().sum() for g in first),params)
        return output,first,second
    actual,expected=run(True),run(False)
    torch.testing.assert_close(actual[0],expected[0],atol=1e-4,rtol=0)
    for groups in zip(actual[1:],expected[1:]):
        for a,b in zip(*groups):torch.testing.assert_close(a,b,atol=1e-5,rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA memory certificate')
def test_chunk_checkpoint_reduces_cuda_saved_activation_peak(monkeypatch):
    from natlang_neuralese.model import lfm2_port
    class FF(nn.Module):
        def __init__(self):
            super().__init__();self.w1=nn.Linear(256,2048,bias=False);self.w3=nn.Linear(256,2048,bias=False);self.w2=nn.Linear(2048,256,bias=False)
        def forward(self,x):return self.w2(torch.nn.functional.silu(self.w1(x))*self.w3(x))
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.ffn_norm=nn.Identity();self.post_ffn_norm=nn.Identity();self.feed_forward=FF()
    torch.manual_seed(83);layer=Layer().cuda().bfloat16();source=torch.randn(1,8192,256,device='cuda',dtype=torch.bfloat16)*.1
    upstream=torch.randn_like(source)*.01;parameters=tuple(layer.parameters())
    def run(enabled):
        monkeypatch.setattr(lfm2_port,'FFN_CHECKPOINT_MIN_TOKENS',0 if enabled else 100000)
        x=source.clone().requires_grad_();torch.cuda.synchronize();start=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats()
        output=lfm2_port.feed_forward_residual(layer,x,256,sandwich=True)
        grads=torch.autograd.grad(output,(x,*parameters),upstream)
        torch.cuda.synchronize();peak=torch.cuda.max_memory_allocated()-start
        return output.detach(),tuple(g.detach() for g in grads),peak
    eager=run(False);bounded=run(True)
    torch.testing.assert_close(eager[0],bounded[0],atol=1e-4,rtol=0)
    for a,b in zip(eager[1],bounded[1]):torch.testing.assert_close(a,b,atol=1e-5,rtol=0)
    print('CUDA peak bytes eager / bounded:',eager[2],bounded[2])
    assert bounded[2] < eager[2]*.75


def test_nested_chunk_checkpoint_preserves_stochastic_state(monkeypatch):
    from natlang_neuralese.model import lfm2_port
    from torch.utils.checkpoint import checkpoint
    class Layer(nn.Module):
        def __init__(self):
            super().__init__();self.ffn_norm=nn.Identity();self.post_ffn_norm=nn.Identity()
            self.feed_forward=nn.Sequential(nn.Linear(8,16),nn.Dropout(.25),nn.SiLU(),nn.Linear(16,8))
    torch.manual_seed(314);layer=Layer().train();source=torch.randn(1,13,8);upstream=torch.randn_like(source)
    def run(enabled):
        monkeypatch.setattr(lfm2_port,'FFN_CHECKPOINT_MIN_TOKENS',0 if enabled else 1000)
        torch.manual_seed(975);x=source.clone().requires_grad_()
        output=checkpoint(lambda z: lfm2_port.feed_forward_residual(layer,z,3,sandwich=True),x,use_reentrant=False)
        grads=torch.autograd.grad(output,(x,*layer.parameters()),upstream)
        return output.detach(),tuple(g.detach() for g in grads),torch.get_rng_state().clone()
    actual,expected=run(True),run(False)
    assert torch.equal(actual[0],expected[0]) and torch.equal(actual[2],expected[2])
    for a,b in zip(actual[1],expected[1]):assert torch.equal(a,b)
