"""A compiled-site release (header state + xs cores) served by the Python native engine decodes as the training executor
does: the state is spliced as stored (declared ``same-base-adapter-state``, never re-encoded from its text) and the
adapter runs factored in FP32 beside the base projections, as training differentiates it (``factored_factors``)."""
import importlib.util
from dataclasses import replace
from pathlib import Path

import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.compiler.compiled_site import (CompiledSiteConfig, CompiledSiteHead, compile_release,
                                                      create_compiler_adapter, init_header_state, items_graph,
                                                      place_header, site_loss, state_layout)
from natlang_neuralese.model.native_adapters import create_adapter, target_parameters

ROOT = Path(__file__).resolve().parents[2]
HEADER = 't20 t21 t22 t23'
MESSAGES = [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': 't12 ' + HEADER + ' t13'}]


def parity_script():
    spec = importlib.util.spec_from_file_location('check_compiled_site_parity',
                                                  ROOT / 'scripts/neuralese_data/check_compiled_site_parity.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_served_release_splices_its_state_and_decodes_as_the_training_executor(tmp_path):
    parity = parity_script()
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    state = init_header_state(engine, MESSAGES, [], HEADER)
    block = engine.store.get_native_state(state).block
    from natlang_neuralese.serve.targets import _items
    rendered = engine.render_request(place_header(MESSAGES, HEADER, state), None)
    prompt = _items(engine, rendered.segments, rendered.blocks, rendered.escape_nonce)
    graph = items_graph(engine, prompt, [('tok', token) for token in (40, 41, 42, 43)], state, len(block.token_ids),
                        graph_id='t', source_groups=['g'])
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    config = CompiledSiteConfig(layout=tuple(state_layout(block)), targets=targets, rank=2, num_slots=4, value_dim=16,
                                key_dim=8, num_heads=2)
    torch.manual_seed(0)
    head = CompiledSiteHead(config)
    adapter = create_compiler_adapter(engine, targets, 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=3e-2)
    for _ in range(6):
        optimizer.zero_grad()
        site_loss(engine, head, {'graph': graph, 'state_id': state}, adapter_id=adapter, init_block=block, backward=True)
        optimizer.step()
    head.eval()
    result = compile_release(engine, head, adapter_id=adapter, init_block=block, out=tmp_path / 'release', site_id='s',
                             provenance={})
    released = result['softStates']['soft:instructions']['state']
    stack = [{'id': item['artifact'], 'scale': 1.0} for item in result['artifacts']]
    compiled = place_header(MESSAGES, HEADER, released)
    reference = parity.reference_greedy(engine, compiled, None, stack, 8)   # the graph executor's import of the span
    engine.start()
    try:
        served = parity.generate(engine, compiled, None, stack, 8, released)
    finally:
        engine.stop()
    assert served['import_modes'] == ['same-base-adapter-state'], 'the release state is spliced, not re-encoded'
    assert served['tokens'] == reference['tokens']


def bf16_engine(tmp_path):
    from test_native_serve import _tiny_source
    from natlang_neuralese.model.native_model import load_native_model, model_state, CHECKPOINT_SCHEMA
    from natlang_neuralese.serve import load_native_engine
    (tmp_path / 'base').mkdir()
    _tiny_source(tmp_path / 'base')
    model, _, _, config = load_native_model(tmp_path / 'base', dtype=torch.bfloat16, rank=4)
    torch.save({'schema': CHECKPOINT_SCHEMA, 'model_config': config, **model_state(model)}, tmp_path / 'native.pt')
    return load_native_engine(str(tmp_path / 'native.pt'), device='cpu', dtype=torch.bfloat16)


def test_bf16_served_adapter_is_the_trained_factored_arithmetic(tmp_path):
    """Serving merged ``W + delta`` in BF16 rounded a small trained delta onto the weight grid; served logits must equal
    the grad-enabled (training) forward's."""
    engine = bf16_engine(tmp_path)
    names = sorted(target_parameters(engine.shared_model))
    record = create_adapter(engine, {'kind': 'xs', 'rank': 2, 'seed': 0, 'parameter_names': names})
    torch.manual_seed(1)
    values = {name: torch.randn_like(value) * 3e-3 for name, value in record.artifact.tensors.items()}
    adapter = engine.store.put_native_tensor_artifact(replace(record.artifact, tensors=values)).id
    ids = torch.tensor([[4, 10, 11, 12, 20, 21, 22, 13, 5]])

    def logits(stack):
        from natlang_neuralese.model.shared_state import native_step
        return engine._native_run(lambda: engine.shared_model.logits(
            native_step(engine.shared_model, ids, None)['h_final'][:, -1]).float(), stack)

    stack = [{'id': adapter, 'scale': 1.0}]
    with torch.no_grad():
        served, base = logits(stack), logits([])
    with torch.enable_grad():
        trained = logits(stack).detach()
    assert not torch.equal(trained, base), 'the adapter changes the logits'
    assert torch.equal(served, trained)


cuda = __import__('pytest').mark.skipif(not torch.cuda.is_available(), reason='the batch-invariant kernel is CUDA')


@cuda
def test_factored_lora_delta_is_row_invariant_fp32():
    """The low-rank delta of a row is the same in a one-row call and a long one, and FP32-exact (cuBLAS ran it in TF32
    with a reduction chosen by the row count: the native engine's decode drifted from the whole-prompt graph pass)."""
    from natlang_neuralese.model.native_adapters import FactoredLoRALinear
    torch.manual_seed(0)
    base = torch.nn.Linear(512, 384, bias=False).to('cuda', torch.bfloat16)
    factor_a, factor_b = torch.randn(16, 512, device='cuda') * 0.05, torch.randn(384, 16, device='cuda') * 0.05
    x = torch.randn(1, 700, 512, device='cuda').to(torch.bfloat16)
    layer = FactoredLoRALinear(base, factor_b, factor_a)
    with torch.no_grad():
        whole = layer(x)
        pieces = torch.cat([layer(x[:, :1]), layer(x[:, 1:64]), layer(x[:, 64:65]), layer(x[:, 65:])], dim=1)
    assert torch.equal(whole, pieces)
    wide = FactoredLoRALinear(torch.nn.Linear(512, 384, bias=False).cuda(), factor_b, factor_a)
    with torch.no_grad():       # FP32 end to end: the delta is IEEE FP32, not TF32 (relative error ~1e-3)
        got = wide(x.float()).double()
        exact = x.double() @ wide.weight.double().T + (x.double() @ factor_a.double().T) @ factor_b.double().T
    assert float((got - exact).abs().max() / exact.abs().max()) < 1e-5


@cuda
def test_chunked_native_prefill_under_a_stack_equals_one_pass(tmp_path):
    """The native engine prefills a prompt holding a state in chunks (tokens, consume_open, splice, consume_close,
    tokens) and decodes one token at a time; under an adapter stack every chunking gives the one-pass logits."""
    from natlang_neuralese.model.shared_state import native_step
    engine = bf16_engine(tmp_path)
    engine.shared_model.to('cuda')
    names = sorted(target_parameters(engine.shared_model))
    record = create_adapter(engine, {'kind': 'xs', 'rank': 2, 'seed': 0, 'parameter_names': names})
    torch.manual_seed(1)
    values = {name: torch.randn_like(value) * 3e-2 for name, value in record.artifact.tensors.items()}
    stack = [{'id': engine.store.put_native_tensor_artifact(replace(record.artifact, tensors=values)).id, 'scale': 1.0}]
    ids = [4, *range(10, 50), *range(10, 50), 5]

    def logits(cuts):
        def run():
            cache, h = None, None
            for start, stop in zip((0, *cuts), (*cuts, len(ids))):
                out = native_step(engine.shared_model, torch.tensor([ids[start:stop]], device='cuda'), cache)
                cache, h = out['cache'], out['h_final']
            return engine.shared_model.logits(h[:, -1]).float()
        with torch.no_grad():
            return engine._native_run(run, stack)

    whole = logits(())
    assert torch.equal(whole, logits((40, 41, 81)))
    assert torch.equal(whole, logits((len(ids) - 1,)))


@cuda
def test_chunked_native_base_decode_equals_one_pass(tmp_path):
    """The unadapted arm: under the checkpoint's legacy ``source-v1`` profile the FFN projections ran HF ``nn.Linear``
    (cuBLAS, reduction chosen by the row count), so the engine's chunked prefill and one-token decode drifted from the
    graph's whole-prompt pass (0.12-0.22 nats per step on real weights); they run ``_fixed_ffn`` by default now."""
    from natlang_neuralese.model.shared_execution import _fixed_ffn
    from natlang_neuralese.model.shared_state import native_step
    engine = bf16_engine(tmp_path)
    assert engine.model_config.get('native_shared_execution', 'source-v1') == 'source-v1'
    engine.shared_model.to('cuda')
    ffn = [module for name, module in engine.shared_model.port.named_modules()
           if type(module) is torch.nn.Linear and '.feed_forward.' in name]
    assert ffn and all(getattr(module.forward, '__func__', None) is _fixed_ffn for module in ffn)
    ids = [4, *range(10, 50), *range(10, 50), 5]

    def logits(cuts):
        cache, h = None, None
        with torch.no_grad():
            for start, stop in zip((0, *cuts), (*cuts, len(ids))):
                out = native_step(engine.shared_model, torch.tensor([ids[start:stop]], device='cuda'), cache)
                cache, h = out['cache'], out['h_final']
            return engine.shared_model.logits(h[:, -1]).float()

    whole = logits(())
    assert torch.equal(whole, logits((40, 41, 81)))
    assert torch.equal(whole, logits(tuple(range(70, len(ids)))))     # one-token steps, as decode runs


@cuda
def test_source_ffn_projection_is_row_invariant_at_lfm3_width():
    """``_fixed_ffn`` at LFM3-1.7B's FFN shapes: a row's output is the same in a one-row call and a long prefill."""
    from types import MethodType
    from natlang_neuralese.model.shared_execution import _fixed_ffn
    torch.manual_seed(0)
    for width, out in ((2048, 8192), (8192, 2048)):
        layer = torch.nn.Linear(width, out, bias=False).to('cuda', torch.bfloat16)
        layer.forward = MethodType(_fixed_ffn, layer)
        x = torch.randn(1, 513, width, device='cuda').to(torch.bfloat16)
        with torch.no_grad():
            whole = layer(x)
            pieces = torch.cat([layer(x[:, :1]), layer(x[:, 1:200]), layer(x[:, 200:201]), layer(x[:, 201:])], dim=1)
        assert torch.equal(whole, pieces)


@cuda
def test_factored_lora_factor_gradients_are_ieee_fp32():
    """The adapter backward: dW of the FP32 factors ran ``grad.T @ x`` on cuBLAS in TF32 (relative error ~1e-3) under an
    exact FP32 forward; in a plain (vendor) training update the forward and both adjoints did. All are IEEE FP32 now."""
    from natlang_neuralese.model.native_adapters import FactoredLoRALinear
    from natlang_neuralese.model.native_weight_carrier import (plain_training_arithmetic,
                                                                set_training_projection_arithmetic,
                                                                training_projection_arithmetic)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(0)
    base = torch.nn.Linear(1024, 768, bias=False).to('cuda', torch.bfloat16).requires_grad_(False)
    x = torch.randn(1, 600, 1024, device='cuda').to(torch.bfloat16)
    seed_a, seed_b = torch.randn(16, 1024, device='cuda') * 0.05, torch.randn(768, 16, device='cuda') * 0.05
    weight = torch.randn(1, 600, 768, device='cuda').to(torch.bfloat16).float()   # the BF16 output's cotangent is exact
    xd, wd = x.double()[0], weight.double()[0]
    exact_b = wd.T @ (xd @ seed_a.double().T)                 # d/dB of sum(weight * (x A^T) B^T)
    exact_a = (wd @ seed_b.double()).T @ xd                    # d/dA
    previous = training_projection_arithmetic()
    try:
        for mode in ('invariant', 'vendor'):
            for plain in (False, True):
                set_training_projection_arithmetic(mode)
                factor_a, factor_b = seed_a.clone().requires_grad_(), seed_b.clone().requires_grad_()
                layer = FactoredLoRALinear(base, factor_b, factor_a)
                context = plain_training_arithmetic() if plain else __import__('contextlib').nullcontext()
                with context:
                    (layer(x).float() * weight).sum().backward()
                for got, exact in ((factor_b.grad, exact_b), (factor_a.grad, exact_a)):
                    error = float((got.double() - exact).abs().max() / exact.abs().max())
                    assert error < 1e-5, (mode, plain, error)
    finally:
        set_training_projection_arithmetic(previous)
