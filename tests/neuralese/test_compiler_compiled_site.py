"""Compiled sites: the compiler emits the tuner's stack (header soft state + xs cores) and trains through the native
graph executor's own consumption of the spliced state (``compiler/compiled_site.py``)."""
import json

import pytest
import torch

from test_native_serving_adapters import engine_fixture
from natlang_neuralese.compiler.compiled_site import (CompiledSiteConfig, CompiledSiteHead, compile_release,
                                                      create_compiler_adapter, init_block_values, init_header_state,
                                                      items_graph, site_header, site_loss, state_layout)
from natlang_neuralese.model.native_adapters import target_parameters

HEADER = 't20 t21 t22'


def call(header=HEADER):
    return [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': 't12 ' + header + ' t13'}]


def setup(tmp_path):
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    state = init_header_state(engine, call(), [], HEADER)
    block = engine.store.get_native_state(state).block
    # the fixture's template has no ChatML generation prefix: the call's items rendered by hand around the state
    graph = items_graph(engine, [('tok', 4), ('block', state), ('tok', 5)], [('tok', 8), ('tok', 9)], state,
                        len(block.token_ids), graph_id='t', source_groups=['g'])
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    config = CompiledSiteConfig(layout=tuple(state_layout(block)), targets=targets, rank=2, num_slots=4, value_dim=16,
                                key_dim=8, num_heads=2)
    torch.manual_seed(0)
    head = CompiledSiteHead(config)
    adapter = create_compiler_adapter(engine, targets, 2)
    return engine, block, {'graph': graph, 'state_id': state}, head, adapter


def test_site_header_is_the_rendered_signature_and_instructions():
    user = ('You are inside this call: f(x: string): string\n\nInstructions:\nDo it.\n\nIn eval you can use x.')
    assert site_header([{'role': 'system', 'content': 's'}, {'role': 'user', 'content': user}]) == (
        'You are inside this call: f(x: string): string\n\nInstructions:\nDo it.')
    assert site_header([{'role': 'user', 'content': 'no header'}]) is None


def test_step_zero_is_the_text_state_and_gradients_reach_the_encoder(tmp_path):
    engine, block, site, head, adapter = setup(tmp_path)
    assert graph_imports_state(site)
    base = site_loss(engine, head, site, adapter_id=adapter, init_block=block, kv=False, adapter=False)
    joint = site_loss(engine, head, site, adapter_id=adapter, init_block=block)
    assert joint == base, 'zero-initialized outputs: the compiled stack is the text state and a zero core'
    values = head(init_block_values(block)).state_values
    for name, value in init_block_values(block).items():
        assert torch.equal(values[name], value.float())
    # one backward: the zero output layers get gradient; one update later the encoder does too
    head.zero_grad()
    site_loss(engine, head, site, adapter_id=adapter, init_block=block, backward=True)
    out_grad = sum(float(p.grad.abs().sum()) for p in head.state_out.parameters() if p.grad is not None)
    core_grad = sum(float(p.grad.abs().sum()) for p in head.core_out.parameters() if p.grad is not None)
    assert out_grad > 0 and core_grad > 0
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    losses = []
    for _ in range(6):
        optimizer.zero_grad()
        losses.append(site_loss(engine, head, site, adapter_id=adapter, init_block=block, backward=True))
        optimizer.step()
    encoder_grad = sum(float(p.grad.abs().sum()) for p in head.encoder.parameters() if p.grad is not None)
    assert encoder_grad > 0
    assert losses[-1] < losses[0]
    assert all(p.dtype == torch.float32 for p in head.parameters())


def graph_imports_state(site):
    node = site['graph']['nodes'][0]
    return node['imports'] and site['graph']['external_states']['soft:instructions']['id'] == site['state_id']


def test_the_release_is_exactly_what_was_trained(tmp_path):
    engine, block, site, head, adapter = setup(tmp_path)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(3):
        optimizer.zero_grad()
        site_loss(engine, head, site, adapter_id=adapter, init_block=block, backward=True)
        optimizer.step()
    head.eval()
    trained = site_loss(engine, head, site, adapter_id=adapter, init_block=block)
    result = compile_release(engine, head, adapter_id=adapter, init_block=block, out=tmp_path / 'release',
                             site_id='s', provenance={})
    assert json.loads((tmp_path / 'release' / 'result.json').read_text()) == result
    state = result['softStates']['soft:instructions']['state']
    artifact = result['artifacts'][0]['artifact']
    assert (tmp_path / 'release' / 'states' / f'{state}.nzs').exists()
    assert (tmp_path / 'release' / 'artifacts' / f'{artifact}.nnt').exists()
    stored = engine.store.get_native_state(state).block
    assert stored.producer_kind is None and stored.token_ids.tolist() == block.token_ids.tolist()
    assert any(not torch.equal(a, b) for a, b in zip(init_block_values(stored).values(), init_block_values(block).values()))
    # served as stored: the stored state and the stored cores, no overrides, give the trained loss
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    graph = json.loads(json.dumps(site['graph']).replace(site['state_id'], state))
    body = {'schema': 'natlang.native-gradient-request/1', 'native_identity': engine.native_identity, 'graph': graph,
            'invocation_adapters': {'turn': [{'id': artifact, 'scale': 1.0}]}, 'adapter_gradient_precision': 'fp32',
            'native_state_modes': {state: 'same-base-adapter-state'}}
    served = native_graph_gradient(engine, body, None, evaluation_callback=lambda loss, groups, meta: float(loss.detach()),
                                   evaluation_target_specs={'s': {'kind': 'native-state', 'id': state}})
    assert served == pytest.approx(trained, rel=1e-5, abs=1e-6)
