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
