"""Weight EMA (BF16, stochastic rounding), LR anneal branch and matrix/attention-logit telemetry."""

import hashlib
import json
import math
from types import SimpleNamespace

import pytest
import torch

from natlang_neuralese.train.weight_ema import WeightEMA, decay_at, parse_weight_ema


# EMA math -------------------------------------------------------------------------------------------------------

def _param(values):
    return torch.nn.Parameter(values, requires_grad=False)


def test_weight_ema_spec_and_warmup_schedule():
    assert parse_weight_ema(None) is None and parse_weight_ema('off') == {'off': True}
    spec = parse_weight_ema('{"decay": 0.999, "warmup": 10, "eval": "both"}')
    assert spec == {'decay': 0.999, 'warmup': 10, 'eval': 'both', 'seed': 0}
    for bad in ('{"decay": 1.0}', '{"decay": 0.9, "warmup": 1}', '{"decay": 0.9, "eval": "x"}', '{"decay": 0.9, "z": 1}'):
        with pytest.raises(ValueError):
            parse_weight_ema(bad)
    assert decay_at(0.999, 0, 5) == 0.999
    assert decay_at(0.999, 10, 0) == pytest.approx(0.1)
    assert decay_at(0.999, 10, 90) == pytest.approx(0.91)
    assert decay_at(0.999, 10, 100000) == 0.999


def test_bf16_stochastic_rounding_ema_tracks_the_fp64_ema_where_nearest_rounding_stalls():
    elements, steps, decay = 20000, 2000, 0.999
    weight = _param(torch.ones(elements, dtype=torch.bfloat16))
    ema = WeightEMA([('w', weight)], decay=decay, seed=7)
    with torch.no_grad():
        weight.copy_(torch.full((elements,), 1.03125, dtype=torch.bfloat16))  # 4 ulps; exactly representable
    nearest = torch.ones(elements, dtype=torch.bfloat16)
    exact = 1.0
    global_rng = torch.get_rng_state()
    for _ in range(steps):
        ema.update()
        # Deterministic nearest rounding of the same recurrence: each increment is far below half an ulp.
        nearest.copy_(torch.lerp(nearest.float(), weight.float(), 1 - decay))
        exact = decay * exact + (1 - decay) * 1.03125
    assert torch.equal(torch.get_rng_state(), global_rng), 'the EMA must not consume the global RNG'
    sr = ema.tensors['w'].float()
    assert sr.dtype == torch.float32 and ema.tensors['w'].dtype == torch.bfloat16
    assert (nearest.float() == 1.0).all()  # small updates are lost without stochastic rounding
    assert abs(float(sr.mean()) - exact) < 4e-4  # unbiased: matches the FP64 EMA in expectation
    assert abs(exact - 1.0) > 0.02
    assert ema.updates == steps


def test_bf16_ema_follows_a_moving_target_in_expectation():
    generator = torch.Generator().manual_seed(1)
    elements, steps, decay = 8192, 3000, 0.99
    weight = _param(torch.randn(elements, generator=generator).to(torch.bfloat16))
    ema = WeightEMA([('w', weight)], decay=decay, warmup=10, seed=3)
    reference = weight.detach().double().clone()
    for step in range(steps):
        with torch.no_grad():
            weight.add_((0.01 * torch.randn(elements, generator=generator)).to(torch.bfloat16))
        ema.update()
        d = decay_at(decay, 10, step)
        reference = d * reference + (1 - d) * weight.detach().double()
    error = ema.tensors['w'].double() - reference
    assert abs(float(error.mean())) < 2e-3
    assert float(error.abs().mean()) < 0.05  # per-element SR noise stays at BF16-ulp scale


def test_ema_state_roundtrip_is_bit_exact_and_continues_identically(tmp_path):
    def run(steps_before, resume):
        generator = torch.Generator().manual_seed(11)
        weight = _param(torch.randn(3000, generator=generator).to(torch.bfloat16))
        wide = _param(torch.randn(40, 5, generator=generator))  # an FP32 parameter
        named = [('a', weight), ('b', wide)]
        ema = WeightEMA(named, decay=0.9, warmup=4, seed=5)
        for step in range(6):
            with torch.no_grad():
                weight.add_((0.02 * torch.randn(3000, generator=generator)).to(torch.bfloat16))
                wide.add_(0.02 * torch.randn(40, 5, generator=generator))
            ema.update()
            if resume and step + 1 == steps_before:
                torch.save(ema.state_dict(), tmp_path / 'ema.pt')
                saved = torch.load(tmp_path / 'ema.pt', weights_only=True)
                ema = WeightEMA.restore(named, saved)
        return ema
    uninterrupted, resumed = run(3, False), run(3, True)
    assert uninterrupted.updates == resumed.updates == 6
    assert uninterrupted.tensors['a'].dtype == torch.bfloat16 and uninterrupted.tensors['b'].dtype == torch.float32
    for name in ('a', 'b'):
        assert torch.equal(uninterrupted.tensors[name], resumed.tensors[name])
    assert resumed.provenance['events'][-1]['event'] == 'restored'


def test_ema_restore_reports_changed_inventory_and_rejects_shape_changes():
    one, two = _param(torch.ones(4)), _param(torch.ones(2, 2, dtype=torch.bfloat16))
    saved = WeightEMA([('one', one), ('gone', two)], decay=0.9).state_dict()
    restored = WeightEMA.restore([('one', one), ('new', two)], saved)
    event = restored.provenance['events'][-1]
    assert event['reinitialized'] == ['new'] and event['dropped'] == ['gone']
    with pytest.raises(ValueError, match='shape'):
        WeightEMA.restore([('one', _param(torch.ones(5)))], saved)
    recast = WeightEMA.restore([('one', _param(torch.ones(4, dtype=torch.bfloat16)))], saved)
    assert recast.tensors['one'].dtype == torch.bfloat16
    assert recast.provenance['events'][-1]['recast_to_parameter_dtype'] == ['one']


def test_applied_swaps_the_ema_in_and_the_raw_weights_back_exactly():
    weight = _param(torch.randn(100).to(torch.bfloat16))
    ema = WeightEMA([('w', weight)], decay=0.5)
    with torch.no_grad():
        weight.mul_(2)
    ema.update()
    raw, averaged = weight.detach().clone(), ema.tensors['w'].clone()
    with ema.applied():
        assert torch.equal(weight, averaged)
    assert torch.equal(weight, raw) and torch.equal(ema.tensors['w'], averaged)


# LR anneal ------------------------------------------------------------------------------------------------------

def test_anneal_shapes_and_group_scaling():
    from natlang_neuralese.train.lr_anneal import LrAnneal, multiplier, parse_lr_anneal

    linear = parse_lr_anneal('{"steps": 4, "shape": "linear", "floor": 0.2}')
    assert [multiplier(linear, k) for k in range(5)] == pytest.approx([1.0, 0.8, 0.6, 0.4, 0.2])
    cosine = parse_lr_anneal('{"steps": 4, "shape": "cosine"}')
    assert multiplier(cosine, 0) == 1.0 and multiplier(cosine, 2) == pytest.approx(0.5)
    assert multiplier(cosine, 4) == pytest.approx(0.0, abs=1e-12)
    for bad in ('{"steps": 0}', '{"steps": 3, "shape": "x"}', '{"steps": 3, "floor": 1.0}', '{"steps": 3, "q": 1}'):
        with pytest.raises(ValueError):
            parse_lr_anneal(bad)
    first, second = torch.nn.Parameter(torch.ones(2)), torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([{'params': [first], 'lr': 1e-3}, {'params': [second], 'lr': 4e-3}])
    anneal = LrAnneal(optimizer, linear, start_global_step=10)
    assert anneal.remaining(10) == 4 and anneal.remaining(14) == 0
    assert anneal.scale(12) == pytest.approx(0.6)
    assert [g['lr'] for g in optimizer.param_groups] == pytest.approx([6e-4, 2.4e-3])
    anneal.restore()
    assert [g['lr'] for g in optimizer.param_groups] == [1e-3, 4e-3]



def test_warmup_is_linear_by_global_step_and_composes_with_an_anneal():
    from natlang_neuralese.train.lr_anneal import LrAnneal, LrWarmup, parse_lr_anneal

    first = torch.nn.Parameter(torch.ones(2))
    optimizer = torch.optim.AdamW([{'params': [first], 'lr': 1e-3}])
    warmup = LrWarmup(optimizer, 4)
    assert [warmup.factor(k) for k in range(6)] == pytest.approx([.25, .5, .75, 1., 1., 1.])
    anneal = LrAnneal(optimizer, parse_lr_anneal('{"steps": 4}'), start_global_step=0)
    anneal.scale(1)
    assert warmup.scale(1) == .5 and optimizer.param_groups[0]['lr'] == pytest.approx(1e-3 * .75 * .5)
    warmup.restore()
    anneal.restore()
    assert optimizer.param_groups[0]['lr'] == 1e-3
    assert warmup.scale(7) == 1.0 and optimizer.param_groups[0]['lr'] == 1e-3

# Telemetry ------------------------------------------------------------------------------------------------------

def test_matrix_kinds_are_name_based_for_lfm2_lfm3_and_interface_names():
    from natlang_neuralese.train.matrix_telemetry import matrix_kind

    assert matrix_kind('backbone.model.layers.3.self_attn.q_proj.weight') == 'self_attn.q_proj'
    assert matrix_kind('backbone.layers.15.self_attn.gate_proj.weight') == 'self_attn.gate_proj'
    assert matrix_kind('backbone.model.layers.0.feed_forward.w2.weight') == 'feed_forward.w2'
    assert matrix_kind('backbone.model.layers.1.conv.in_proj.weight') == 'conv.in_proj'
    assert matrix_kind('backbone.model.layers.1.conv.out_proj.weight') == 'conv.out_proj'
    assert matrix_kind('interface.input_rows') == 'embedding'


def test_attention_probe_matches_brute_force_including_imported_keys_and_consumer_rows():
    from natlang_neuralese.model.attention_probe import AttentionLogitProbe, observe_attention, probing
    from natlang_neuralese.model.native_graph_plan import ImportSpan, Invocation, compile_graph

    producer = Invocation('a', tuple(range(4)), producing=((0, 2),))
    consumer = Invocation('b', tuple(range(3)), imports=(ImportSpan(at=1, source='a', start=0, stop=2),))
    plan = compile_graph([producer, consumer])
    generator = torch.Generator().manual_seed(4)
    heads, kv_heads, width, scale = 4, 2, 8, 0.7
    slots = len(plan.slot_sources)
    q = torch.randn(1, heads, plan.token_count, width, generator=generator)
    k = torch.randn(1, kv_heads, slots, width, generator=generator)
    assert plan.token_count == 7
    for window in (None, 3):
        probe = AttentionLogitProbe([5], chunk_elements=64)
        with probing(probe):
            observe_attention(5, q, k, plan, window, scale)
            observe_attention(5, q * 100, k, plan, window, scale)  # a layer is measured once per probe
            observe_attention(6, q, k, plan, window, scale)  # not selected
        measured = probe.finalize()
        assert list(measured) == [5]
        allowed = [list(row) for row in plan.allowed]
        if window is not None:
            allowed = [row[-window:] for row in allowed]
        imported_slots = {plan.slot_offsets[1] + 1, plan.slot_offsets[1] + 2}
        best = {'all': -math.inf, 'imported_keys': -math.inf, 'consumer_rows': -math.inf}
        consumer_rows = 0
        for row, visible in enumerate(allowed):
            sees_import = bool(imported_slots & set(visible))
            consumer_rows += sees_import
            for head in range(heads):
                for slot in visible:
                    logit = float(scale * (q[0, head, row] @ k[0, head // (heads // kv_heads), slot]))
                    best['all'] = max(best['all'], logit)
                    if slot in imported_slots:
                        best['imported_keys'] = max(best['imported_keys'], logit)
                    if sees_import:
                        best['consumer_rows'] = max(best['consumer_rows'], logit)
        for key, value in best.items():
            expected = None if value == -math.inf else value
            if expected is None:
                assert measured[5][key] is None
            else:
                assert measured[5][key] == pytest.approx(expected, rel=1e-5, abs=1e-6), (window, key)
        assert measured[5]['consumer_row_count'] == consumer_rows
        assert measured[5]['imported_slots'] == 2
    assert consumer_rows >= 1


def test_probe_failure_never_raises_into_training():
    from natlang_neuralese.model.attention_probe import AttentionLogitProbe

    probe = AttentionLogitProbe()
    plan = SimpleNamespace(node_ids=(0, 0, 0), token_count=3, slot_sources=(0, 1, 2), invocations=(), offsets=(),
                           sequences=(), slot_offsets=(), bounds=lambda *a: (_ for _ in ()).throw(RuntimeError('boom')))
    probe.observe(0, torch.zeros(1, 4, 3, 8), torch.zeros(1, 2, 3, 8), plan, None, 1.0)
    assert 'boom' in probe.finalize()[0]['error']


# Trainer integration (tiny native fixture on CPU) -----------------------------------------------------------------

def _fixture(tmp_path, dtype='bfloat16'):
    from test_native_serve import _tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model

    base = tmp_path / 'base'
    _tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=getattr(torch, dtype), rank=4)
    paths = {}
    for split in ('train', 'test'):
        row = dict(graph(split), split=split, source_groups=[split], tokenizer_fingerprint=config['tokenizer_fingerprint'])
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text(json.dumps(row) + '\n')
    return base, paths


def _argv(base, paths, out, *, epochs, dtype='bfloat16', extra=()):
    return ['--base', str(base), '--dtype', dtype, '--rank', '4', '--device', 'cpu',
            '--native-graphs', str(paths['train']), '--held-native-graphs', str(paths['test']),
            '--epochs', str(epochs), '--eval-every', '100', '--out', str(out), *extra]


def _train(argv, monkeypatch, *, stop_after=None):
    from natlang_neuralese.train import native_state_sft as trainer

    with monkeypatch.context() as patch:
        if stop_after is not None:
            calls = []

            def stop(policy, step, **kwargs):
                calls.append(step)
                if len(calls) >= stop_after:
                    policy.stop.requested = True
                return False
            patch.setattr(trainer.CheckpointPolicy, 'due', stop)
        assert trainer.main(argv) == 0


def _load(path):
    return torch.load(path, weights_only=False)


def _same_weights(left, right):
    def equal(a, b):
        if torch.is_tensor(a):
            return torch.equal(a, b)
        if isinstance(a, dict):
            return a.keys() == b.keys() and all(equal(a[key], b[key]) for key in a)
        return a == b
    return all(equal(left[part][key], right[part][key])
               for part in ('backbone_state', 'interface_state') for key in left[part])


def test_trainer_ema_resumes_bit_exactly_earns_no_credit_and_leaves_training_untouched(tmp_path, monkeypatch):
    base, paths = _fixture(tmp_path)
    spec = '{"decay": 0.9, "warmup": 3, "eval": "both"}'
    plain = tmp_path / 'plain'
    _train(_argv(base, paths, plain, epochs=4, extra=['--weight-ema', 'off']), monkeypatch)
    straight = tmp_path / 'straight'
    _train(_argv(base, paths, straight, epochs=4, extra=['--weight-ema', spec]), monkeypatch)
    halted = tmp_path / 'halted'
    argv = _argv(base, paths, halted, epochs=4, extra=['--weight-ema', spec])
    _train(argv, monkeypatch, stop_after=2)
    midway = _load(halted / 'checkpoint.pt')['training']
    assert midway['global_step'] == 2 and midway['ema']['updates'] == 2
    _train(argv, monkeypatch)
    a, b, c = (_load(path / 'checkpoint.pt') for path in (plain, straight, halted))
    assert _same_weights(a, b), 'enabling the EMA must not change the trained weights'
    assert _same_weights(b, c), 'a resumed run must equal the uninterrupted one'
    for key in b['training']['ema']['tensors']:
        assert torch.equal(b['training']['ema']['tensors'][key], c['training']['ema']['tensors'][key])
    for state in (b, c):
        training = state['training']
        assert training['step'] == training['global_step'] == 4 and training['exposure_counts'] == {'train': 4}
        assert training['ema']['updates'] == 4 and json.loads(training['options']['weight_ema'])['decay'] == 0.9
    assert 'ema' not in a['training']
    assert c['training']['metrics']['held_ema']['evaluated_weights'] == 'ema'
    assert 'evaluated_weights' not in c['training']['metrics']['held']
    # The EMA differs from the raw weights and the export holds exactly the saved EMA.
    from natlang_neuralese.model.native_model import load_native_checkpoint
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters
    from natlang_neuralese.train.weight_ema import export_checkpoint
    export = export_checkpoint(halted / 'checkpoint.pt', tmp_path / 'ema-export', base=base)
    assert export['source_global_step'] == 4 and export['ema']['updates'] == 4
    exported, *_ = load_native_checkpoint(tmp_path / 'ema-export' / 'checkpoint.pt', base=base, allow_weights_only=True)
    raw, *_ = load_native_checkpoint(halted / 'checkpoint.pt', base=base)
    saved = c['training']['ema']['tensors']
    differing = 0
    for (name, parameter), (_, original) in zip(native_trainable_parameters(exported), native_trainable_parameters(raw)):
        assert torch.equal(parameter, saved[name])
        differing += not torch.equal(parameter, original)
    assert differing > 0


def test_trainer_continues_a_carried_ema_by_default_and_discards_it_only_explicitly(tmp_path, monkeypatch):
    base, paths = _fixture(tmp_path)
    first = tmp_path / 'first'
    _train(_argv(base, paths, first, epochs=1, extra=['--weight-ema', '{"decay": 0.9, "eval": "ema"}']), monkeypatch)
    # Omitted: the carried EMA continues with its recorded decay, warmup, seed and evaluation mode.
    second = tmp_path / 'second'
    _train(_argv(base, paths, second, epochs=2, extra=['--continue-from', str(first / 'checkpoint.pt')]), monkeypatch)
    training = _load(second / 'checkpoint.pt')['training']
    assert training['ema']['updates'] == 2 and training['ema']['decay'] == 0.9
    assert json.loads(training['options']['weight_ema'])['eval'] == 'ema'
    assert training['options']['weight_ema_resolution'] == {'source': 'carried'}
    continued = tmp_path / 'third'
    _train(_argv(base, paths, continued, epochs=2, extra=['--continue-from', str(first / 'checkpoint.pt'),
                                                        '--weight-ema', '{"decay": 0.9}']), monkeypatch)
    training = _load(continued / 'checkpoint.pt')['training']
    events = training['ema']['provenance']['events']
    assert training['ema']['updates'] == 2 and events[-1]['event'] == 'restored' and events[-1]['updates'] == 1
    discarded = tmp_path / 'fourth'
    _train(_argv(base, paths, discarded, epochs=2, extra=['--continue-from', str(first / 'checkpoint.pt'),
                                                        '--weight-ema', 'off']), monkeypatch)
    assert 'ema' not in _load(discarded / 'checkpoint.pt')['training']


def test_trainer_telemetry_matches_direct_computation_and_brute_force_logits(tmp_path, monkeypatch):
    from natlang_neuralese.model import shared_state
    from natlang_neuralese.model.native_model import load_native_checkpoint, load_native_model
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters

    base, paths = _fixture(tmp_path)
    captured = []
    real = shared_state.graph_attention

    def spy(q, k, v, plan, window, scale, **kwargs):
        if hasattr(plan, 'node_ids') and not captured:
            with torch.no_grad():
                scores = (q.detach().float() @ k.detach().float().repeat_interleave(
                    q.shape[1] // k.shape[1], 1).transpose(-1, -2)) * scale
                visible = plan.attention_mask(q.device, window)
                captured.append(float(scores.masked_fill(~visible, -math.inf).amax()))
        return real(q, k, v, plan, window, scale, **kwargs)
    monkeypatch.setattr(shared_state, 'graph_attention', spy)
    out = tmp_path / 'run'
    _train(_argv(base, paths, out, epochs=2, extra=['--matrix-telemetry', '{"every": 1}']), monkeypatch)
    records = [json.loads(line) for line in (out / 'matrix-telemetry.jsonl').read_text().splitlines()]
    assert [record['global_step'] for record in records] == [1, 2]
    assert records[0]['attention_logits']['layers_observed'] == [1]
    assert records[0]['attention_logits']['layers']['1']['all'] == pytest.approx(captured[0], rel=1e-5)
    # Norms against a direct FP64 computation from the saved weights and the base weights.
    model, *_ = load_native_checkpoint(out / 'checkpoint.pt', base=base)
    fresh, *_ = load_native_model(base, dtype=torch.bfloat16, rank=4)
    fresh_norms = {name: float(parameter.detach().double().norm())
                   for name, parameter in native_trainable_parameters(fresh) if parameter.ndim >= 2}
    options = _load(out / 'checkpoint.pt')['training']['options']
    reference = options['matrix_norm_reference']['norms']
    for name, value in fresh_norms.items():  # backbone weights are the base weights; the adapter init is random per load
        if name.startswith('backbone.'):
            assert reference[name] == pytest.approx(value, rel=1e-5), name
    final = records[-1]['norms']['matrices']
    kinds = set()
    for name, parameter in native_trainable_parameters(model):
        if parameter.ndim < 2:
            continue
        direct = float(parameter.detach().double().norm())
        assert final[name]['norm'] == pytest.approx(direct, rel=1e-5), name
        if reference[name] == 0:  # a zero-initialized matrix has no growth ratio
            assert final[name]['ratio'] is None, name
        else:
            assert final[name]['ratio'] == pytest.approx(direct / reference[name], rel=1e-4), name
        kinds.add(final[name]['kind'])
    assert {'self_attn.q_proj', 'self_attn.k_proj', 'self_attn.v_proj', 'conv.in_proj', 'conv.out_proj',
            'feed_forward.w1', 'feed_forward.w2', 'feed_forward.w3'} <= kinds
    assert 'vocabulary.embedding' in final
    assert options['matrix_norm_reference']['source'] == 'base-initialization'
    progress = json.loads((out / 'progress.json').read_text())
    assert progress['matrix_telemetry']['file'] == 'matrix-telemetry.jsonl'


def _tree_hashes(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(root.rglob('*')) if path.is_file()}


def test_anneal_branch_is_a_separate_continuation_that_leaves_the_main_lineage_untouched(tmp_path, monkeypatch):
    from natlang_neuralese.model.native_model import load_native_checkpoint
    from natlang_neuralese.train import lr_anneal
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.train.native_state_sft import native_trainable_parameters

    base, paths = _fixture(tmp_path)
    ema = '{"decay": 0.9}'
    reference = tmp_path / 'reference'
    _train(_argv(base, paths, reference, epochs=6, extra=['--weight-ema', ema]), monkeypatch)
    main = tmp_path / 'main'
    argv = _argv(base, paths, main, epochs=6, extra=['--weight-ema', ema])
    _train(argv, monkeypatch, stop_after=2)
    before = _tree_hashes(main)
    parent = _load(main / 'checkpoint.pt')['training']
    assert parent['global_step'] == 2

    rates = []
    real = trainer.native_training_update

    def spy(model, optimizer, batch, **kwargs):
        rates.append(sorted({group['lr'] for _, child in __import__(
            'natlang_neuralese.train.optim_restore', fromlist=['_children'])._children(optimizer)
            for group in child.param_groups}))
        return real(model, optimizer, batch, **kwargs)
    monkeypatch.setattr(trainer, 'native_training_update', spy)
    with pytest.raises(ValueError, match='own namespace'):
        lr_anneal.plan_branch(main / 'checkpoint.pt', main / 'branch', {'steps': 2, 'shape': 'linear', 'floor': 0.0})
    branch = tmp_path / 'branch'
    assert lr_anneal.main(['run', '--from', str(main / 'checkpoint.pt'), '--out', str(branch), '--steps', '2',
                           '--shape', 'linear', '--floor', '0.25', '--base', str(base)]) == 0
    monkeypatch.setattr(trainer, 'native_training_update', real)

    assert _tree_hashes(main) == before, 'the main lineage directory must be untouched'
    state = _load(branch / 'branch' / 'checkpoint.pt')
    training = state['training']
    base_rate = rates[0][-1]
    assert [max(r) for r in rates] == pytest.approx([base_rate, 0.625 * base_rate])
    assert training['global_step'] == 4 and training['ema']['updates'] == 4
    assert training['options']['lr_anneal']['branch'] is True
    assert training['options']['lr_anneal']['adopt_as_lineage_parent'] is False
    assert training['options']['lr_anneal']['parent_checkpoint_sha256'] == \
        json.loads((branch / 'anneal-provenance.json').read_text())['parent_checkpoint_sha256']
    progress = [json.loads(line) for line in (branch / 'branch' / 'train.jsonl').read_text().splitlines()]
    assert [round(item['lr_anneal']['multiplier'], 6) for item in progress] == [1.0, 0.625]
    assert training['exposure_counts'] == {'train': 4}, 'the branch continues the live cursor and exposure'
    # The release candidate is the EMA continued through the anneal.
    export = json.loads((branch / 'anneal-provenance.json').read_text())
    assert export['main_lineage_modified'] is False and export['parent_has_weight_ema'] is True
    candidate, *_ = load_native_checkpoint(branch / 'release-candidate-ema' / 'checkpoint.pt', base=base,
                                           allow_weights_only=True)
    for name, parameter in native_trainable_parameters(candidate):
        assert torch.equal(parameter, training['ema']['tensors'][name])
    # The main lineage continues exactly as if the branch never existed.
    _train(argv, monkeypatch)
    resumed, straight = _load(main / 'checkpoint.pt'), _load(reference / 'checkpoint.pt')
    assert _same_weights(resumed, straight)
    assert resumed['training']['global_step'] == 6 and resumed['training']['options']['lr_anneal'] is None


def test_weight_ema_and_telemetry_are_on_by_default_and_a_continuation_without_an_ema_starts_one(tmp_path, monkeypatch):
    from natlang_neuralese.train.matrix_telemetry import DEFAULT as TELEMETRY, parse_matrix_telemetry
    from natlang_neuralese.train.weight_ema import DEFAULT, resolve_weight_ema

    assert parse_matrix_telemetry(None) == TELEMETRY and parse_matrix_telemetry('off') is None
    assert resolve_weight_ema(None, None) == (DEFAULT, {'source': 'default', 'started_from': 'current-weights'})
    assert resolve_weight_ema(None, None, fp8_training=True)[0]['eval'] == 'raw'
    assert resolve_weight_ema(None, None, diloco=True)[0] is None
    assert resolve_weight_ema(None, {'updates': 3, 'decay': .9, 'warmup': 0, 'seed': 1}, diloco=True)[0] == {'off': True}
    base, paths = _fixture(tmp_path)
    fresh = tmp_path / 'fresh'
    _train(_argv(base, paths, fresh, epochs=1), monkeypatch)
    training = _load(fresh / 'checkpoint.pt')['training']
    assert training['ema']['updates'] == 1 and json.loads(training['options']['weight_ema']) == DEFAULT
    assert training['options']['weight_ema_resolution']['source'] == 'default'
    assert json.loads(training['options']['matrix_telemetry']) == TELEMETRY
    assert 'held_ema' in training['metrics']
    # A historical parent without an EMA: its continuation adopts the default from the parent's weights at that step.
    old = tmp_path / 'old'
    _train(_argv(base, paths, old, epochs=1, extra=['--weight-ema', 'off', '--matrix-telemetry', 'off']), monkeypatch)
    assert 'ema' not in _load(old / 'checkpoint.pt')['training']
    adopted = tmp_path / 'adopted'
    _train(_argv(base, paths, adopted, epochs=2, extra=['--continue-from', str(old / 'checkpoint.pt')]), monkeypatch)
    training = _load(adopted / 'checkpoint.pt')['training']
    initialized = training['ema']['provenance']['events'][0]
    assert initialized['event'] == 'initialized' and initialized['global_step'] == 1
    assert initialized['checkpoint_sha256'] == hashlib.sha256((old / 'checkpoint.pt').read_bytes()).hexdigest()
    assert training['ema']['updates'] == 1
    reference = training['options']['matrix_norm_reference']
    assert reference['source'] == 'parent-continuation' and reference['global_step'] == 1
