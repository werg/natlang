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


def test_prepare_keeps_real_header_calls_and_counts_episode_state_imports(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.compiler import run_t2
    from natlang_neuralese.rl import held_eval
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda cases: frozenset({'held'}))
    user = 'You are inside this call: f(x: string): string\n\nInstructions:\nDo it.\n\nIn eval you can use x.'
    plain = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': user}]
    stated = [{'role': 'system', 'content': 's'}, {'role': 'user', 'content': user},
              {'role': 'tool', 'content': [{'type': 'native_state', 'id': {'id': 'nzs1_' + 'a' * 52}}]}]
    turn = lambda messages: {'messages': messages, 'tools': [], 'target': {'role': 'assistant', 'content': 'ok'},
                             'sampling': {'prompt': [{'token_ids': [1, 2, 3]}]}}
    group = tmp_path / 'c' / 'groups' / 'g000000'
    group.mkdir(parents=True)
    (group / 'cases.jsonl').write_text('\n'.join(json.dumps({'id': t}) for t in ('t1', 'held')) + '\n')
    rollouts = [{'task_id': 't1', 'outcome': 'task_success', 'sample': i, 'seed': i, 'turns': [turn(plain), turn(stated)]}
                for i in range(3)]
    rollouts += [{'task_id': 'held', 'outcome': 'task_success', 'sample': 9, 'seed': 9, 'turns': [turn(plain)]},
                 {'task_id': 't1', 'outcome': 'semantic_mismatch', 'sample': 8, 'seed': 8, 'turns': [turn(plain)]}]
    (group / 'rollouts.jsonl').write_text('\n'.join(json.dumps(r) for r in rollouts) + '\n')
    out = tmp_path / 'rows.jsonl'
    run_t2.prepare(SimpleNamespace(collections=[str(tmp_path / 'c')], out=str(out), max_prompt_tokens=4096, per_site=8,
                                   min_per_site=2, dev_site_percent=0))
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    funnel = json.loads(out.with_suffix('.summary.json').read_text())['funnel']
    assert funnel['imports_native_state'] == 3 and funnel['held_excluded_rollouts'] == 1 and funnel['failed_rollouts'] == 1
    assert len(rows) == 3 and not any(run_t2.imports_native_state(row['messages']) for row in rows)
    assert {row['split'] for row in rows} == {'train', 'dev-seen'}
