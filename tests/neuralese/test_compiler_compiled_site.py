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


def test_prepare_cases_unwrap_authored_rows_for_the_held_check_and_fail_closed(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from natlang_neuralese.compiler import run_t2
    from natlang_neuralese.rl import held_eval
    seen = []
    monkeypatch.setattr(held_eval, 'training_exclusion_ids', lambda cases: seen.extend(cases) or frozenset())
    authored = tmp_path / 'authored.jsonl'
    authored.write_text(json.dumps({'id': 'inv:p:a:1', 'source': 'invented-reference',
                                    'task': {'id': 'inv:p:a:1', 'source_groups': ['invented:p'], 'split': 'train'}}) + '\n')
    args = dict(collections=[], call_level=[], out=str(tmp_path / 'rows.jsonl'), max_prompt_tokens=4096, per_site=8,
                min_per_site=2, dev_site_percent=0)
    run_t2.prepare(SimpleNamespace(cases=[str(authored)], **args))
    assert seen == [{'id': 'inv:p:a:1', 'source_groups': ['invented:p'], 'split': 'train'}]
    with pytest.raises(FileNotFoundError):
        run_t2.prepare(SimpleNamespace(cases=[str(tmp_path / 'missing.jsonl')], **args))

def test_pplx_warmstart_writer_dual_lr_groups_and_strict_manifest(tmp_path):
    from types import SimpleNamespace
    from natlang_neuralese.compiler.live_kb_writer import LiveExperienceWriter, PplxWarmstartWriter

    # Fallback embed writer still exposes fresh_parameters() / warm_parameters() cleanly
    embed_w = LiveExperienceWriter(embed_dim=16, value_dim=8, key_dim=8, num_heads=2, hidden_dim=16)
    assert len(embed_w.warm_parameters()) == 0
    assert len(embed_w.fresh_parameters()) == sum(1 for _ in embed_w.parameters())

    # PplxWarmstartWriter.from_warmstart fails fast on missing writer_H.pt / query_Q.pt (no silent random fallback)
    with pytest.raises(FileNotFoundError, match='writer_H.pt'):
        PplxWarmstartWriter.from_warmstart(
            revision='0000000',
            warmstart_dir=str(tmp_path / 'missing'),
        )

    # Synthetic stub PPLX encoders verifying dual-LR partition (warm backbone vs fresh value_head/poolers)
    class _StubPplx(torch.nn.Module):
        def __init__(self, with_value=True):
            super().__init__()
            self.backbone = torch.nn.Linear(8, 8)
            self.value_head = torch.nn.Linear(8, 12) if with_value else None

        def encode_blocks(self, texts, direction='full'):
            out = []
            for _ in texts:
                h = self.backbone(torch.ones(3, 8))
                v = self.value_head(h) if self.value_head is not None else None
                out.append(SimpleNamespace(keys=h, values=v))
            return out

    w_stub, q_stub = _StubPplx(with_value=True), _StubPplx(with_value=False)
    writer = PplxWarmstartWriter(w_stub, q_stub, value_dim=12, key_dim=8)
    warm_ids = {id(p) for p in writer.warm_parameters()}
    fresh_ids = {id(p) for p in writer.fresh_parameters()}
    assert warm_ids and fresh_ids and warm_ids.isdisjoint(fresh_ids)
    assert all(id(p) in fresh_ids for p in w_stub.value_head.parameters())
    assert all(id(p) in warm_ids for p in w_stub.backbone.parameters())
    assert all(id(p) in warm_ids for p in q_stub.backbone.parameters())
    z_kb, k_kb = writer.encode_experience_texts(None, ['r1', 'r2'])
    q = writer.encode_query_texts(None, ['ctx'])
    assert z_kb.shape == (2, 12) and k_kb.shape == (2, 8) and q.shape == (1, 8)
    # the encoders keep computing deterministically (eval) while their parameters train
    writer.train()
    assert writer.training and not w_stub.training and not q_stub.training



def test_latent_ttt_proximal_l2_matches_fo_maml_sum_scale():
    import inspect
    from natlang_neuralese.tune import latent_ttt
    src = inspect.getsource(latent_ttt.refine_site_latents) + inspect.getsource(latent_ttt.refine_compiled_site)
    assert '.pow(2).sum()' in src and '.pow(2).mean()' not in src


def test_step_verifier_progress_weighting_upweights_late_boundaries(tmp_path):
    from natlang_neuralese.train.step_verifier import VerifierTrainConfig, train_t0
    feat_dir = tmp_path / 'feats'
    feat_dir.mkdir()
    torch.manual_seed(0)
    # 20 episodes x 4 boundaries: early boundaries (0, 1) are pure noise, late boundaries (2, 3) are separable
    xs, meta = [], []
    for ep in range(20):
        label = ep % 2
        is_val = ep >= 14
        for turn in range(4):
            vec = torch.randn(16)
            if turn >= 2:
                vec[0] = 3.0 if label else -3.0
            xs.append(vec)
            meta.append({'episode': f'ep-{ep}', 'task_id': f't-{ep}', 'family': 'famA', 'turn': turn,
                         'boundary': turn, 'success': bool(label), 'validation': is_val})
    torch.save({'hidden': torch.stack(xs), 'meta': meta}, feat_dir / 'shard-000.pt')
    receipt = train_t0(feat_dir, tmp_path / 'out', cfg=VerifierTrainConfig(epochs=5, batch_size=16), device='cpu',
                       progress_weighting=True)
    assert receipt['auroc']['validation'] > 0.65


class _ThinkTokens:
    """A tokenizer stand-in for a thinking template: ids index a fixed vocabulary."""
    vocab = ['<|im_start|>', '<think>', '</think>', '"', '<|im_end|>', 'plan', '<tool_call>', 'yes']

    def convert_tokens_to_ids(self, token):
        return self.vocab.index(token) if token in self.vocab else None

    def decode(self, ids):
        return ''.join(self.vocab[i] for i in ids)


def test_site_graph_masks_a_teacher_empty_think_block_and_keeps_own_reasoning(tmp_path, monkeypatch):
    # 2026-11-26 T2: teacher rows' `<think></think>` was trained (compiled_site.items_graph weighted every target 1);
    # its </think> carried ~all of a teacher row's base CE and the whole compiled gain
    import natlang_neuralese.serve.targets as targets
    from natlang_neuralese.compiler.compiled_site import site_graph
    monkeypatch.delenv('NATLANG_EMPTY_THINK_FIRST', raising=False)
    monkeypatch.delenv('NATLANG_TUNE_OWN_EMPTY_THINK', raising=False)
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    state = init_header_state(engine, call(), [], HEADER)
    block = engine.store.get_native_state(state).block
    rendered = {}
    monkeypatch.setattr(targets, 'target_items', lambda engine, messages, tools, reply, **_: (
        [('tok', 4), ('block', state), ('tok', 5)], [('tok', t) for t in rendered['ids']]))
    monkeypatch.setattr(engine, 'tokenizer', _ThinkTokens())

    def weights(reply, ids, own, **kw):
        rendered['ids'] = ids
        graph = site_graph(engine, call(), [], reply, HEADER, state, len(block.token_ids), graph_id='g',
                           source_groups=['g'], own=own, **kw)
        return graph['nodes'][0]['target_weights'][-len(ids):]

    teacher = {'role': 'assistant', 'content': '"yes"'}
    answer = [2, 3, 7, 3, 4]                          # </think> " yes " <|im_end|>
    # the empty think block and the token predicted at its </think> are context; the answer trains
    assert weights(teacher, answer, own=False) == [0., 0., 1., 1., 1.]
    # the policy's own reasoning keeps its weight, think block included
    own = {**teacher, 'reasoning_content': 'plan'}
    assert weights(own, [5] + answer, own=True) == [1.] * 6
    # an own reply without reasoning: the unsampled empty block is context too
    assert weights(teacher, answer, own=True) == [0., 0., 1., 1., 1.]
    # a teacher's reasoning is context
    assert weights(own, [5] + answer, own=False) == [0., 0., 1., 1., 1., 1.]
    # opt-out
    assert weights(teacher, answer, own=False, mask_context=False) == [1.] * 5


def test_row_loss_scales_count_target_tokens_with_a_cap():
    from natlang_neuralese.compiler.run_t2 import row_loss_scales
    assert row_loss_scales([3, 3000], 'rows', 512) == [.5, .5]
    assert row_loss_scales([3, 3000], 'tokens', 0) == pytest.approx([3 / 3003, 3000 / 3003])
    assert row_loss_scales([3, 3000], 'tokens', 512) == pytest.approx([3 / 515, 512 / 515])


def _kb_fixture(tmp_path):
    from natlang_neuralese.compiler.live_kb_writer import LiveExperienceWriter, _engine_embedding_weight
    engine = engine_fixture(tmp_path, 'lfm', service_template=True)
    torch.manual_seed(3)
    writer = LiveExperienceWriter(int(_engine_embedding_weight(engine).shape[-1]), value_dim=16, key_dim=8, num_heads=2,
                                  hidden_dim=16)
    pool = [{'key': f'p{i}', 'task_id': f'task{i}', 'family': f'fam{i % 3}', 'site': f's{i}', 'source_groups': [],
             'experience_text': f't{30 + i} t{40 + i}', 'header': f't{20 + i}', 'split': 'train'} for i in range(9)]
    return engine, writer, pool


def test_live_kb_fills_positives_by_similarity_and_trains_distractors(tmp_path):
    from natlang_neuralese.compiler.live_kb_writer import KbIndex, build_live_kb_slots
    engine, writer, pool = _kb_fixture(tmp_path)
    row = {'key': 'r', 'task_id': 'r', 'family': 'lonely', 'site': 'sr', 'source_groups': [], 'header': 't20 t21'}
    index = KbIndex(engine, writer, pool)
    chosen = []
    z, _, _, _ = build_live_kb_slots(engine, writer, row, pool, k_pos=2, k_neg=2, index=index, chosen=chosen)
    # no same-family or same-site peer: both positives come from the index (the nearest by query), then 2 distractors
    q = writer.encode_query_texts(engine, [row['header']]).detach()
    assert chosen[:2] == [p['key'] for p in index.nearest(q, pool, 2)]
    assert z.shape[1] == 4 and len(set(chosen)) == 4
    # distractors carry gradient into the writer unless opted out
    writer.zero_grad()
    z[:, 2:].sum().backward()
    assert sum(float(p.grad.abs().sum()) for p in writer.parameters() if p.grad is not None) > 0
    z_off, _, _, _ = build_live_kb_slots(engine, writer, row, pool, k_pos=0, k_neg=2, neg_grad=False)
    assert not z_off.requires_grad
    # a same-site peer is a labelled relative
    sited = {**row, 'site': 's4'}
    chosen = []
    build_live_kb_slots(engine, writer, sited, pool, k_pos=1, k_neg=0, chosen=chosen)
    assert chosen == ['p4']


def test_eval_kb_draw_is_per_row_and_repeatable(tmp_path):
    from natlang_neuralese.compiler.live_kb_writer import build_live_kb_slots
    from natlang_neuralese.compiler.run_t2 import eval_rng
    engine, writer, pool = _kb_fixture(tmp_path)
    rows = [{'key': f'd{i}', 'task_id': f'dev{i}', 'family': 'fam0', 'site': f'ds{i}', 'source_groups': [],
             'header': 't20'} for i in range(4)]
    drawn = []
    for row in rows:
        chosen = []
        build_live_kb_slots(engine, writer, row, pool, k_pos=2, k_neg=2, rng=eval_rng(0, row), chosen=chosen)
        drawn.append(tuple(chosen))
    # each row its own seeded sample (rng=None took the first peers in pool order for every row: identical KBs)
    assert len(set(drawn)) > 1
    again = []
    build_live_kb_slots(engine, writer, rows[0], pool, k_pos=2, k_neg=2, rng=eval_rng(0, rows[0]), chosen=again)
    assert tuple(again) == drawn[0]


def test_ablation_reports_site_blind_and_whole_site_controls_by_source(tmp_path):
    from natlang_neuralese.compiler.live_kb_writer import KbIndex
    from natlang_neuralese.compiler.run_t2 import Sites, ablation
    engine, writer, _ = _kb_fixture(tmp_path)
    headers = ['t20\nt21 t22', 't23\nt24 t25']
    rows = []
    for i in range(6):
        header = headers[i % 2]
        messages = [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': 't12 ' + header + ' t13'}]
        rows.append({'key': f'k{i}', 'site': f's{i % 2}', 'header': header, 'messages': messages, 'tools': [],
                     'task_id': f't{i}', 'family': f'f{i % 3}', 'source_groups': [], 'split': 'dev-seen',
                     'source_kind': 'rollout' if i < 3 else 'call_level_teacher',
                     'experience_text': f't{30 + i} t{31 + i}',
                     'target': {'role': 'assistant', 'content': f't{8 + i % 2}'}})
    sites = Sites(engine, tmp_path / 'run')
    sites.path.parent.mkdir(parents=True, exist_ok=True)
    block = sites.block(rows[0])
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    config = CompiledSiteConfig(layout=tuple(state_layout(block)), targets=targets, rank=2, num_slots=4, value_dim=16,
                                key_dim=8, num_heads=2, adapter_inner_rank=1)
    head = CompiledSiteHead(config)
    adapter = create_compiler_adapter(engine, targets, 2)
    result = ablation(engine, head, sites, rows, adapter, 0, rows, writer=writer, peer_pool=rows, seed=0,
                      index=KbIndex(engine, writer, rows), k_pos=1, k_neg=1)
    assert set(result['mean_ce_by_source']) == {'rollout', 'call_level_teacher'}
    for arm in ('base', 'joint', 'mean_core', 'shuffled', 'shuffled_slots'):
        assert result['mean_ce'][arm] is not None and result['mean_ce_tokens'][arm] is not None
    by_key = {r['key']: r for r in result['per_row']}
    assert all(r['target_tokens'] > 0 for r in result['per_row'])
    # shuffled_kb never reuses the row's own peers; a row without such a donor is listed
    for r in result['per_row']:
        assert ('shuffled_kb' in r) != (r['key'] in result['shuffled_kb_unavailable'])
    assert len({tuple(r['kb_peers']) for r in by_key.values()}) > 1
    assert result['z_site_mean_cos'] is not None


def test_compiled_site_bounded_state_residual_and_site_decorrelation():
    from natlang_neuralese.compiler.run_t2 import site_decorrelation_loss
    layout = (('layers.0.raw_k', (2, 4)), ('layers.0.v', (2, 4)))
    cfg = CompiledSiteConfig(layout=layout, targets=('model.layers.0.self_attn.q_proj.weight',), rank=2, num_slots=4,
                             value_dim=8, key_dim=4, num_heads=2, adapter_inner_rank=1, state_residual_scale=0.1)
    head = CompiledSiteHead(cfg)
    assert head.state_scale is not None
    # Backward-compat: saved dict without state_residual_scale loads as 0.0 (unbounded legacy)
    legacy_dict = cfg.to_dict()
    legacy_dict.pop('state_residual_scale')
    legacy_cfg = CompiledSiteConfig.from_dict(legacy_dict)
    assert legacy_cfg.state_residual_scale == 0.0 and CompiledSiteHead(legacy_cfg).state_scale is None

    # Bounded residual stays within +-state_scale even under huge linear weights
    init_vals = {name: torch.zeros(3, *shape) for name, shape in layout}
    with torch.no_grad():
        for layer in head.state_out.values():
            layer.bias.fill_(50.0)
    out = head(init_vals, apply_dropout=False)
    for name, _ in layout:
        assert float(out.state_values[name].abs().max()) <= 0.1 + 1e-5

    # Site decorrelation penalizes collapsed identical sites strictly more than orthogonal sites
    z_same = [torch.ones(1, 4, 8), torch.ones(1, 4, 8)]
    z_orth = [torch.tensor([[[1., 0., 0., 0., 0., 0., 0., 0.]] * 4]),
              torch.tensor([[[0., 1., 0., 0., 0., 0., 0., 0.]] * 4])]
    assert float(site_decorrelation_loss(z_same)) > float(site_decorrelation_loss(z_orth))


def test_step_verifier_pairwise_within_task_ranking_cancels_task_difficulty(tmp_path):
    from natlang_neuralese.train.step_verifier import VerifierTrainConfig, train_t0
    feat_dir = tmp_path / 'feats'
    feat_dir.mkdir()
    torch.manual_seed(7)
    # Every task has a large confounding task-level offset on dims 2..15, while true pass/fail signal is on dim 0.
    # Within-task pairwise ranking should cleanly recover dim 0 and beat random on validation_within_task!
    xs, meta = [], []
    for task in range(24):
        task_offset = torch.randn(16) * 4.0
        task_offset[0] = 0.0
        is_val = task >= 18
        for ep in range(4):
            label = ep % 2
            for turn in range(2):
                vec = task_offset.clone() + torch.randn(16) * 0.5
                vec[0] = 2.0 if label else -2.0
                xs.append(vec)
                meta.append({'episode': f't{task}-e{ep}', 'task_id': f'task-{task}', 'family': 'famA',
                             'turn': turn, 'boundary': turn, 'success': bool(label), 'validation': is_val})
    torch.save({'hidden': torch.stack(xs), 'meta': meta}, feat_dir / 'shard-000.pt')
    receipt = train_t0(feat_dir, tmp_path / 'out',
                       cfg=VerifierTrainConfig(epochs=6, batch_size=32, pairwise_weight=2.0, dropout=0.1),
                       device='cpu', progress_weighting=True)
    assert receipt['auroc']['validation_within_task'] > 0.85
    assert receipt['rows']['validation_mixed_tasks'] == 6


def test_step_verifier_rich_score_boundary_matches_feature_dim():
    from natlang_neuralese.train.step_verifier import StepVerifierHead
    h = torch.randn(10, 8)
    rich_head = StepVerifierHead(3 * 8 + 5, mlp_dim=16)
    last_head = StepVerifierHead(8, mlp_dim=16)
    p_rich = rich_head.score_boundary(h, prompt_end=4, boundary=9, scalars=[0.5, 0.2, -0.1, -0.5, 1.0])
    p_last = last_head.score_boundary(h, prompt_end=4, boundary=9)
    assert 0.0 <= p_rich <= 1.0 and 0.0 <= p_last <= 1.0


def test_t3_mode_p_http_server_serves_compiled_site(tmp_path):
    import threading
    import urllib.request
    from natlang_neuralese.compiler.per_call_hkb import PerCallNeuralHKBCompiler
    from natlang_neuralese.compiler.t3_server import CompiledSiteService, make_http_server

    engine, _, _ = _kb_fixture(tmp_path)
    header = 'You are inside this call: f(x: string): string\n\nInstructions:\nCheck x.'
    messages = [{'role': 'system', 'content': 't10 t11'}, {'role': 'user', 'content': header}]
    from natlang_neuralese.compiler.compiled_site import init_header_state
    init_id = init_header_state(engine, messages, [], header)
    init_block = engine.store.get_native_state(init_id).block
    targets = tuple(sorted(target_parameters(engine.shared_model)))
    cfg = CompiledSiteConfig(layout=tuple(state_layout(init_block)), targets=targets, rank=2, num_slots=4,
                             value_dim=16, key_dim=8, num_heads=2, adapter_inner_rank=1)
    head = CompiledSiteHead(cfg)
    adapter = create_compiler_adapter(engine, targets, 2)
    compiler = PerCallNeuralHKBCompiler(engine, head, adapter_id=adapter, release_root=tmp_path / 'releases')
    service = CompiledSiteService(compiler, compiler_sha='c' * 64, policy_sha='p' * 64)
    server = make_http_server(service, host='127.0.0.1', port=0)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urllib.request.urlopen(f'http://127.0.0.1:{port}/health') as resp:
            health = json.loads(resp.read().decode('utf-8'))
        assert health['ok'] is True and health['mode'] == 'mode-p'
        req = urllib.request.Request(
            f'http://127.0.0.1:{port}/v1/natlang/compile-site',
            data=json.dumps({'site_id': 'site-f', 'header': header, 'messages': messages}).encode('utf-8'),
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req) as resp:
            compiled = json.loads(resp.read().decode('utf-8'))
        assert compiled['mode'] == 'mode-p'
        assert compiled['adapterId'] and compiled['nativeStateId'] and compiled['cacheKey']
    finally:
        server.shutdown()
        server.server_close()
