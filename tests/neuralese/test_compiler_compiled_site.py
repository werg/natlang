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


def test_learned_basis_kb_slots_tier15_and_per_call_compile(tmp_path):
    """Learned (B, A) basis, [z_site, z_kb, null] slots with the Live-KB writer, Tier-1.5 refinement on z_site, and the
    per-call compiler (written by another agent 2026-11-25, adapted in the integration review)."""
    from natlang_neuralese.compiler.compiled_site import compose_site_cores, create_learned_compiler_basis
    from natlang_neuralese.compiler.live_kb_writer import LiveExperienceWriter, _engine_embedding_weight
    from natlang_neuralese.compiler.per_call_hkb import PerCallCompileRequest, PerCallNeuralHKBCompiler
    from natlang_neuralese.tune.latent_ttt import LatentTTTConfig, refine_compiled_site

    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    state = init_header_state(engine, call(), [], HEADER)
    block = engine.store.get_native_state(state).block
    graph = items_graph(engine, [('tok', 4), ('block', state), ('tok', 5)], [('tok', 8), ('tok', 9)], state,
                        len(block.token_ids), graph_id='t-kb', source_groups=['g'])
    site = {'graph': graph, 'state_id': state}
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    basis = create_learned_compiler_basis(engine, targets, rank=2)
    config = CompiledSiteConfig(layout=tuple(state_layout(block)), targets=targets, rank=2, num_slots=4,
                                value_dim=16, key_dim=8, num_heads=2, adapter_inner_rank=1)
    torch.manual_seed(7)
    head = CompiledSiteHead(config)
    writer = LiveExperienceWriter(int(_engine_embedding_weight(engine).shape[-1]), value_dim=16, key_dim=8, num_heads=2,
                                  hidden_dim=16)
    v_kb = torch.ones((1, 1), dtype=torch.bool)

    # step 0: the text state and a zero core whatever the slots hold
    base = site_loss(engine, head, site, adapter_id=basis.adapter_id, init_block=block, basis_module=basis,
                     kv=False, adapter=False)
    z_kb, k_kb = writer.encode_experience_texts(engine, ['t30 t31 t32'])
    with_kb = site_loss(engine, head, site, adapter_id=basis.adapter_id, init_block=block, basis_module=basis,
                        z_kb=z_kb.unsqueeze(0), k_kb=k_kb.unsqueeze(0), w_kb=torch.ones((1, 1)), kb_valid=v_kb)
    assert with_kb == pytest.approx(base, rel=1e-6, abs=1e-7)

    # joint training: gradients reach the head, the learned basis and the writer
    opt = torch.optim.Adam(list(head.parameters()) + list(basis.parameters()) + list(writer.parameters()), lr=1e-2)
    losses = []
    for _ in range(4):
        opt.zero_grad()
        z, k = writer.encode_experience_texts(engine, ['t30 t31 t32'])
        losses.append(site_loss(engine, head, site, adapter_id=basis.adapter_id, init_block=block, basis_module=basis,
                                z_kb=z.unsqueeze(0), k_kb=k.unsqueeze(0), w_kb=torch.ones((1, 1)), kb_valid=v_kb,
                                backward=True))
        opt.step()
    assert losses[-1] < losses[0]
    assert sum(float(p.grad.abs().sum()) for p in basis.parameters() if p.grad is not None) > 0
    assert sum(float(p.grad.abs().sum()) for p in writer.parameters() if p.grad is not None) > 0

    c1 = head(init_block_values(block)).cores
    assert all(torch.allclose(compose_site_cores(c1, c1)[k], 2 * c1[k]) for k in c1)
    with pytest.raises(ValueError):
        compose_site_cores(c1, {name: c1[name] for name in list(c1)[1:]})

    # the release carries the learned basis at its current value and serves the trained loss
    head.eval()
    trained = site_loss(engine, head, site, adapter_id=basis.adapter_id, init_block=block, basis_module=basis)
    rel = compile_release(engine, head, adapter_id=basis.adapter_id, init_block=block, basis_module=basis,
                          out=tmp_path / 'rel', site_id='s', provenance={})
    released = rel['softStates']['soft:instructions']['state']
    artifact = rel['artifacts'][0]['artifact']
    from natlang_neuralese.serve.native_grad import native_graph_gradient
    body = {'schema': 'natlang.native-gradient-request/1', 'native_identity': engine.native_identity,
            'graph': json.loads(json.dumps(graph).replace(state, released)),
            'invocation_adapters': {'turn': [{'id': artifact, 'scale': 1.0}]}, 'adapter_gradient_precision': 'fp32',
            'native_state_modes': {released: 'same-base-adapter-state'}}
    served = native_graph_gradient(engine, body, None, evaluation_callback=lambda loss, groups, meta: float(loss.detach()),
                                   evaluation_target_specs={'s': {'kind': 'native-state', 'id': released}})
    assert served == pytest.approx(trained, rel=1e-5, abs=1e-6)

    # Tier 1.5: only z_site moves, the head and basis do not
    before = [p.detach().clone() for p in [*head.parameters(), *basis.parameters()]]
    examples = [{'graph': graph, 'state_id': state, 'group': f'g{i}'} for i in range(4)]
    result, release = refine_compiled_site(
        engine, head, adapter_id=basis.adapter_id, init_block=block, basis_module=basis, site_examples=examples,
        out_dir=tmp_path / 'rel-t15', site_id='s-t15',
        cfg=LatentTTTConfig(steps=6, lr=0.1, prox_lambda=0.01, eval_every=2, min_query_gain=1e-6))
    assert result.accepted and result.best_query_loss < result.tier1_query_loss
    assert release is not None and (tmp_path / 'rel-t15' / 'result.json').exists()
    assert all(torch.equal(a, b) for a, b in zip(before, [*head.parameters(), *basis.parameters()]))

    # per-call compile: exact provenance is required, the cache keys on it
    with pytest.raises(TypeError):
        PerCallCompileRequest(site_id='f', header=HEADER, messages=call(), tools=[])
    svc = PerCallNeuralHKBCompiler(engine, head, adapter_id=basis.adapter_id, release_root=tmp_path / 'svc',
                                   basis_module=basis)
    req = PerCallCompileRequest(site_id='demo::fn', header=HEADER, messages=call(), tools=[], compiler_sha='c1',
                                policy_sha='p1', scope_headers=('t10 t11',))
    assert svc.compile_call(req)['cacheKey'] == svc.compile_call(req)['cacheKey']
    other = PerCallCompileRequest(site_id='demo::fn', header=HEADER, messages=call(), tools=[], compiler_sha='c2',
                                  policy_sha='p1', scope_headers=('t10 t11',))
    assert svc.compile_call(other)['cacheKey'] != svc.compile_call(req)['cacheKey']


def test_factorized_cores_leave_the_zero_saddle_and_config_is_strict(tmp_path):
    engine, block, site, head, adapter = setup(tmp_path)
    config = CompiledSiteConfig(layout=head.config.layout, targets=head.config.targets, rank=2, num_slots=4,
                                value_dim=16, key_dim=8, num_heads=2, adapter_inner_rank=1)
    head = CompiledSiteHead(config)
    assert all(float(c.detach().abs().max()) == 0 for c in head(init_block_values(block)).cores.values())
    head.zero_grad()
    site_loss(engine, head, site, adapter_id=adapter, init_block=block, backward=True)
    # raw = [diag | U | V] per row: U starts nonzero and V at zero, so V's output weights get gradient at step 0
    grads = [layer.weight.grad.view(2, 3, -1) for layer in head.core_out.values() if layer.weight.grad is not None]
    assert sum(float(g[:, 2].abs().sum()) for g in grads) > 0
    with pytest.raises(TypeError):
        CompiledSiteConfig.from_dict({**config.to_dict(), 'kb_state_slots': 8})
    assert CompiledSiteConfig.from_dict(config.to_dict()) == config


def test_live_kb_peers_skip_the_task_and_its_siblings_and_restraint_keeps_distractors(tmp_path):
    from natlang_neuralese.compiler.live_kb_writer import LiveExperienceWriter, _engine_embedding_weight, build_live_kb_slots
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    writer = LiveExperienceWriter(int(_engine_embedding_weight(engine).shape[-1]), value_dim=16, key_dim=8, num_heads=2,
                                  hidden_dim=16)
    row = {'key': 'r', 'task_id': 'a', 'family': 'f', 'source_groups': ['sg1'], 'header': 't20 t21'}
    peer = lambda key, task, family, groups: {'key': key, 'task_id': task, 'family': family, 'source_groups': groups,
                                               'experience_text': f't{30 + len(key)} t31'}
    pool = [peer('r', 'a', 'f', ['sg1']), peer('same-task', 'a', 'f', []), peer('sibling', 'b', 'f', ['sg1']),
            peer('ok', 'c', 'f', ['sg2']), peer('other', 'd', 'g', ['sg3'])]
    z, k, w, valid = build_live_kb_slots(engine, writer, row, pool, k_pos=4, k_neg=4)
    assert z.shape[1] == 2 and bool(valid.all())       # 'ok' (positive) and 'other' (distractor) only
    assert z.requires_grad
    z_null, _, w_null, _ = build_live_kb_slots(engine, writer, row, pool, k_pos=4, k_neg=4, null_restraint=True)
    assert z_null.shape[1] == 1 and float(w_null.sum()) > 0     # the distractor at its real weight


def test_fo_maml_and_compositional_steps_reach_the_encoder(tmp_path):
    from natlang_neuralese.compiler.run_t2 import Sites, _compositional_step, _fo_maml_episode_step
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    header = 't20\nt21 t22'
    messages = [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': 't12 ' + header + ' t13'}]
    rows = [{'key': f'k{i}', 'site': 's', 'header': header, 'scope_header': 't20',
             'body_header': 't21 t22', 'messages': messages, 'tools': [], 'task_id': f't{i}', 'source_groups': [],
             'target': {'role': 'assistant', 'content': f't{8 + i % 2}'}} for i in range(3)]
    sites = Sites(engine, tmp_path / 'run')
    sites.path.parent.mkdir(parents=True, exist_ok=True)
    block = sites.block(rows[0])
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    config = CompiledSiteConfig(layout=tuple(state_layout(block)), targets=targets, rank=2, num_slots=4, value_dim=16,
                                key_dim=8, num_heads=2, adapter_inner_rank=1)
    torch.manual_seed(1)
    head = CompiledSiteHead(config)
    adapter = create_compiler_adapter(engine, targets, 2)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-2)
    for _ in range(2):   # one update so the zero output layers open a path back to the encoder
        optimizer.zero_grad()
        site_loss(engine, head, sites.row(rows[0]), adapter_id=adapter, init_block=block, backward=True)
        optimizer.step()
    head.zero_grad()
    q = _fo_maml_episode_step(engine, head, sites, rows, adapter_id=adapter, basis_module=None, inner_steps=2,
                              inner_lr=.05, prox_lambda=.01, reptile_weight=.1, scale=1.)
    assert q > 0
    assert sum(float(p.grad.abs().sum()) for p in head.encoder.parameters() if p.grad is not None) > 0
    head.zero_grad()
    comp = _compositional_step(engine, head, sites, rows[0], adapter_id=adapter, basis_module=None, scale=1.)
    assert comp is not None and comp > 0
    assert sum(float(p.grad.abs().sum()) for p in head.core_out.parameters() if p.grad is not None) > 0


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
