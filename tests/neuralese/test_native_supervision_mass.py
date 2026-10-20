"""Native CE mass diagnosis against the existing balanced text objective.

These tests do not adopt a policy or rewrite corpus weights. The exact suffixes
below are authored fixture provenance, never inferred from arbitrary magnitudes.
"""
import pytest
import torch
from torch.nn import functional as F

from natlang_neuralese.data.native_graph import SCHEMA, parse_graph, graph_loss
from natlang_neuralese.model.shared_state import SharedStateModel
from natlang_neuralese.train.native_state_sft import packed_graph_loss
from natlang_neuralese.train.text_supervision import balanced_position_weights
from test_shared_state import tiny


def declared(encoding='ordinary-suffix-raw-v1', digest='a' * 64):
    from natlang_neuralese.train.native_supervision import SCHEMA as POLICY_SCHEMA
    return {'schema': POLICY_SCHEMA, 'sources': [
        {'sha256': digest, 'encoding': encoding, 'context_weight': .05}]}


def ordinary():
    return {'schema': SCHEMA, 'id': 'mass:ordinary', 'split': 'train',
        'source_groups': ['mass:train'], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [{'name': 'text', 'token_ids': [2] * 24 + [3, 4],
                   'target_weights': [0.] + [.05] * 23 + [1., 1.]}],
        'conversion': {'role_interpretation': 'ordinary-text',
                       'source_suffix_start': 24, 'context_weight': .05}}


def test_raw_point_zero_five_is_not_five_percent_context_mass():
    # A real observed shape: a short complete action after a long tool prompt.
    raw = torch.tensor([[0.] + [.05] * 6505 + [1.] * 35])
    target = torch.arange(raw.shape[1]) >= 6506
    assert float(raw[:, target].sum() / raw.sum()) < .10
    balanced = balanced_position_weights(raw, [6506], context_weight=.05, exclude=raw == 0)
    assert float(balanced[:, target].sum() / balanced.sum()) > .95
    assert balanced[0, 0] == 0 and (balanced[raw > 0] > 0).all()


@pytest.mark.parametrize('family', ['lfm', 'mellum'])
@pytest.mark.parametrize('policy', ['raw', 'balanced-reference'])
def test_complete_graph_ce_and_all_derivatives_match_declared_mass(family, policy):
    model = SharedStateModel(tiny(family), marker_ids=(58, 59, 60, 61), rank=4)
    row = {'schema': SCHEMA, 'id': 'source-mass-fixture', 'split': 'train',
        'source_groups': ['independently-authored-train'], 'tokenizer_fingerprint': 'f' * 64,
        'nodes': [
            {'name': 'producer', 'token_ids': [2] * 24 + [3, 4],
             'target_weights': [0.] + [.05] * 23 + [1., 1.], 'producing': [[22, 26]]},
            {'name': 'consumer', 'token_ids': [5] * 20 + [6, 7],
             'target_weights': [0., 0.] + [.05] * 18 + [1., 1.],
             'imports': [{'at': 1, 'source': 'producer', 'start': 22, 'stop': 26}]}]}
    row['source_pairing'] = {'pair_id': 'actual-fixture-pair', 'payloads': [{'invocation': 'producer'}],
        'consumed_occurrences': [{'consumer_import_at': 1}], 'source_ids': ['writer', 'reader']}
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    parameters = tuple(parameter for parameter in model.parameters() if parameter.requires_grad)
    effective, oracle = [], []
    for node, suffix in zip(row['nodes'], [24, 20]):
        raw = torch.tensor([node['target_weights']])
        valid = (raw > 0).float()
        target = valid * (torch.arange(raw.shape[1]) >= suffix)
        if policy == 'balanced-reference':
            actual = balanced_position_weights(raw, [suffix], context_weight=.05, exclude=raw == 0)
            actual *= raw.sum() / actual.sum()
            # Independent normalized derivative oracle, not the implementation.
            expected = (.05 * valid / valid.sum() + target / target.sum()) / 1.05
            expected *= raw.sum()
        else:
            actual, expected = raw, raw
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.sum(), raw.sum())
        assert torch.equal(actual == 0, raw == 0)
        effective.extend(actual[0].tolist())
        oracle.extend(expected[0].tolist())
    from natlang_neuralese.train.native_supervision import apply_policy
    weighted = (apply_policy(example, row, source_sha256='a' * 64,
                             policy=declared('witnessed-pair-raw-v1'))
                if policy == 'balanced-reference' else example)
    assert torch.equal(torch.tensor(weighted.target_weights), torch.tensor(effective))
    if policy == 'raw':
        assert weighted.target_weights == example.target_weights
    result = model(example.plan, retain_states=False)
    queries, labels, weights = [], [], []
    for node, offset in zip(example.plan.invocations, example.plan.offsets):
        for local in range(1, len(node.token_ids)):
            if oracle[offset + local]:
                queries.append(offset + local - 1)
                labels.append(node.token_ids[local])
                weights.append(oracle[offset + local])
    reference_weights = torch.tensor(weights)
    ce = F.cross_entropy(model.logits(result['h_final'][0, queries]).float(),
                         torch.tensor(labels), reduction='none')
    expected_loss = (ce * reference_weights).sum() / reference_weights.sum()
    expected_gradients = torch.autograd.grad(expected_loss, parameters, allow_unused=True)
    for loss in [graph_loss(model, weighted, checkpoint_layers=True, chunk_tokens=3)[0],
                 packed_graph_loss(model, [weighted], checkpoint_layers=True, chunk_tokens=3)[0]]:
        torch.testing.assert_close(loss, expected_loss)
        actual = torch.autograd.grad(loss, parameters, allow_unused=True)
        for observed, expected in zip(actual, expected_gradients):
            assert (observed is None) == (expected is None)
            if expected is not None:
                torch.testing.assert_close(observed, expected)
        assert any(gradient is not None and gradient.norm() > 0 for gradient in actual)


def test_only_explicit_exact_source_encoding_can_change_weights(tmp_path):
    import copy
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.native_supervision import apply_policy, load_policy
    row = ordinary()
    example = parse_graph(row, tokenizer_fingerprint='f' * 64)
    policy = declared()
    with pytest.raises(ValueError, match='missing from explicit supervision policy'):
        apply_policy(example, row, source_sha256='b' * 64, policy=policy)
    assert apply_policy(example, row, source_sha256='b' * 64,
                        policy={**policy, 'require_eligible_sources': False}) is example
    with pytest.raises(ValueError, match='authenticated immutable source'):
        apply_policy(example, row, source_sha256=None, policy=policy)
    for mutation in ['missing-suffix', 'coefficient', 'approximate', 'already-balanced']:
        bad = copy.deepcopy(row)
        if mutation == 'missing-suffix':
            bad['conversion'].pop('source_suffix_start')
        elif mutation == 'coefficient':
            bad['conversion']['context_weight'] = .1
        elif mutation == 'approximate':
            bad['nodes'][0]['target_weights'][5] = .050001
        else:
            bad['conversion']['supervision_policy'] = {'name': 'segmented'}
        parsed = parse_graph(bad, tokenizer_fingerprint='f' * 64)
        with pytest.raises(ValueError):
            apply_policy(parsed, bad, source_sha256='a' * 64, policy=policy)
    path = tmp_path / 'policy.json'
    path.write_text(json.dumps(policy))
    binding = {'path': str(path), 'sha256': sha256_file_hex(path)}
    assert load_policy(binding) == policy
    path.write_text(json.dumps(declared(digest='b' * 64)))
    with pytest.raises(ValueError, match='SHA-256'):
        load_policy(binding)
    row['conversion']['supervision_policy'] = {'weighting':'segmented_position_weights'}
    assert apply_policy(example, row, source_sha256='b' * 64, policy=policy) is example


def test_draw_time_policy_keeps_raw_row_binding_and_original_held_weights(tmp_path):
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.data.native_sources import NativeGraphSource
    source = tmp_path / 'graphs.jsonl'
    source.write_text(json.dumps(ordinary()) + '\n')
    raw = NativeGraphSource(source, 'f' * 64, required_split='train')
    binding = dict(raw.bindings)
    original = next(iter(raw))
    raw.set_ce_policy(declared(digest=sha256_file_hex(source)))
    balanced = next(iter(raw))
    assert raw.bindings == binding
    assert balanced.id == original.id and balanced.plan.invocations == original.plan.invocations
    assert balanced.target_weights != original.target_weights
    assert sum(balanced.target_weights[-2:]) / sum(balanced.target_weights) > .95
    # A second consumer with no selected policy (held/eval) sees immutable raw weights.
    held = NativeGraphSource(source, 'f' * 64, required_split='train')
    assert next(iter(held)).target_weights == original.target_weights


def test_actual_interrupted_training_continues_same_optimizer_and_row_credits(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train import native_state_sft as trainer
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train.checkpoint_offload import _same_values
    from test_native_serve import _tiny_source
    base = tmp_path / 'base'
    _tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    paths = {}
    for split in ['train', 'test']:
        row = ordinary()
        row.update(id=split, split=split, source_groups=[split],
                   tokenizer_fingerprint=config['tokenizer_fingerprint'])
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text(json.dumps(row) + '\n')
    out = tmp_path / 'out'
    argv = ['--base', str(base), '--dtype', 'float32', '--rank', '4', '--device', 'cpu',
        '--native-graphs', str(paths['train']), '--held-native-graphs', str(paths['test']),
        '--epochs', '3', '--eval-every', '100', '--out', str(out)]
    def stop_after_one(policy, step, **kwargs):
        policy.stop.requested = True
        return False
    monkeypatch.setattr(trainer.CheckpointPolicy, 'due', stop_after_one)
    assert trainer.main(argv) == 0
    before = torch.load(out / 'checkpoint.pt', weights_only=False)['training']
    policy = tmp_path / 'policy.json'
    policy.write_text(json.dumps(declared(digest=sha256_file_hex(paths['train']))))
    factory = trainer.make_native_optimizer
    restored = []
    def observe(*args, **kwargs):
        optimizer, names = factory(*args, **kwargs)
        restore = optimizer.load_state_dict
        def load(state, **options):
            _same_values(state, before['optimizer'])
            result = restore(state, **options)
            restored.append(True)
            return result
        optimizer.load_state_dict = load
        return optimizer, names
    monkeypatch.setattr(trainer, 'make_native_optimizer', observe)
    assert trainer.main(argv + ['--ce-supervision-policy', str(policy)]) == 0
    after = torch.load(out / 'checkpoint.pt', weights_only=False)['training']
    assert restored == [True]
    assert after['step'] == before['step'] + 1 == 2
    assert after['global_step'] == before['global_step'] + 1
    assert after['exposure_counts'] == {'train': 2}
    assert after['source_row_bindings'] == before['source_row_bindings']
    history = after['options']['ce_supervision_policy_history']
    assert history[-1]['from'] is None and history[-1]['at_global_step'] == 1
    assert history[-1]['to']['sha256'] == sha256_file_hex(policy)
    assert history[-1]['held_ce'] == 'original-corpus-weights'
    monkeypatch.setattr(trainer, 'make_native_optimizer', factory)
    assert trainer.main(argv) == 0
    inherited = torch.load(out / 'checkpoint.pt', weights_only=False)['training']
    assert inherited['step'] == 3
    assert inherited['options']['ce_supervision_policy_binding'] == history[-1]['to']
    assert inherited['options']['ce_supervision_policy_history'] == history


def test_materializer_binds_actual_container_inventory_and_retains_segmented(tmp_path):
    import json
    from scripts.neuralese_data.build_native_ce_supervision_policy import build
    from natlang_neuralese.common.hashing import sha256_file_hex
    source = tmp_path / 'ordinary.jsonl'
    source.write_text(json.dumps(ordinary()) + '\n')
    wt = ordinary()
    wt['conversion']['supervision_policy'] = {'weighting':'segmented_position_weights'}
    segmented = tmp_path / 'segmented.jsonl'
    segmented.write_text(json.dumps(wt) + '\n')
    result = build([source, segmented], legacy_pair_context_weight=.05)
    assert result['require_eligible_sources'] is True
    assert result['sources'] == declared(digest=sha256_file_hex(source))['sources']
    assert result['retained_sources'][0]['sha256'] == sha256_file_hex(segmented)
    from scripts.neuralese_data.build_native_ce_supervision_policy import main
    out = tmp_path / 'policy.json'
    argv = ['--source',str(source),'--legacy-pair-context-weight','.05','--out',str(out)]
    assert main(argv) == 0 and main(argv) == 0
    with pytest.raises(ValueError, match='new explicit continuation artifact'):
        main(argv[:-2] + ['--source',str(segmented),'--out',str(out)])


def test_recipe_provider_transport_exact_policy_and_reject_unbound_selection(tmp_path):
    import json
    from scripts.coreweave.provider_dependencies import dependency_closure
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train.recipe import stage_input_args
    policy = tmp_path / 'ce.json'
    policy.write_text(json.dumps(declared()))
    binding = {'path':str(policy),'sha256':sha256_file_hex(policy)}
    assert stage_input_args({'ce_supervision_policy':binding}, 'native_state_sft') == [
        '--ce-supervision-policy', str(policy)]
    config = tmp_path / 'config.json'
    config.write_text(json.dumps({'ce_supervision_policy':binding,
        'trainer_parameters':{'ce_supervision_policy':str(policy)}}))
    closure = dependency_closure({'config':{'path':str(config),'sha256':sha256_file_hex(config)}},
                                 namespace=str(tmp_path))
    assert {item['path'] for item in closure['files']} == {str(config),str(policy)}
    config.write_text(json.dumps({'trainer_parameters':{'ce_supervision_policy':str(policy)}}))
    with pytest.raises(ValueError, match='exact bound supervision'):
        dependency_closure({'config':{'path':str(config),'sha256':sha256_file_hex(config)}},
                           namespace=str(tmp_path))


def test_actual_active_recipe_handoff_accepts_only_bound_ce_continuation(tmp_path):
    import copy
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train import recipe
    sources = {}
    for role in ['native_graphs', 'held_native_graphs']:
        path = tmp_path / (role + '.jsonl')
        path.write_text(json.dumps(ordinary()) + '\n')
        sources[role] = {'path':str(path),'sha256':sha256_file_hex(path),'binding':role}
    policy = tmp_path / 'ce.json'
    policy.write_text(json.dumps(declared(digest=sources['native_graphs']['sha256'])))
    extended = {**sources,'ce_supervision_policy':
                {'path':str(policy),'sha256':sha256_file_hex(policy),'binding':'ce'}}
    def plan(roles):
        return {'recipe':{'stages':[{'id':'full','kind':'native_state_sft',
            'inputs':{role:value['binding'] for role,value in roles.items()}}],
            'input_bindings':{value['binding']:{'path':value['path'],'sha256':value['sha256']}
                              for value in roles.values()}},
            'stage_inputs':{'full':roles},'device':'cuda','recipe_sha256':'preserved-lineage',
            'inputs':{value['path']:value['sha256'] for value in roles.values()}}
    before, after = plan(sources), plan(extended)
    changes = recipe.unstarted_stage_input_changes(before, after, tmp_path, started_stage_ids=('full',))
    assert changes['full']['kind'] == 'ce_supervision_continuation'
    frozen = tmp_path / 'run/runtime/natlang_neuralese'
    frozen.mkdir(parents=True)
    (frozen / 'old.py').write_text('old = True')
    current = tmp_path / 'current'
    current.mkdir()
    (current / 'new.py').write_text('new = True')
    before['code'] = {'old.py':recipe.sha(frozen/'old.py')}
    updated = recipe.lineage_handoff(before, after, frozen, current, tmp_path/'plan.json',
        tmp_path/'run', dry_run=True, started_stage_ids=('full',))
    assert updated['recipe_sha256'] == 'preserved-lineage'
    assert updated['stage_inputs']['full'] == extended
    altered = copy.deepcopy(extended)
    altered['held_native_graphs']['sha256'] = 'b' * 64
    assert not recipe._ce_supervision_input_continuation(sources, altered)
    policy.write_text('changed declaration')
    with pytest.raises(ValueError, match='SHA-256'):
        recipe._ce_supervision_input_continuation(sources, extended)


def test_semantic_control_consumer_distillation_use_same_bound_policy():
    import copy
    from natlang_neuralese.train.native_semantic import joint_loss
    from natlang_neuralese.train.native_semantic_stage import SourceBoundPair
    from natlang_neuralese.train.native_supervision import apply_policy
    from test_native_semantic import setup_pair
    participants, raw, vae = setup_pair()
    row = SourceBoundPair(raw)
    row.source_sha256 = 'a' * 64
    for view in row['models'].values():
        graph = view['graph']
        graph['nodes'][0]['name'] = 'producer'
        graph['nodes'][1]['imports'][0]['source'] = 'producer'
        view['export']['node'] = 'producer'
        graph['nodes'][0]['target_weights'][1] = .05
        graph['nodes'][1]['target_weights'][1] = .05
        graph['source_pairing'] = {'pair_id':'fixture', 'payloads':[{'invocation':'producer'}],
            'consumed_occurrences':[{'consumer_import_at':2}], 'source_ids':['w','r']}
    policy = declared('witnessed-pair-raw-v1')
    manual = copy.deepcopy(dict(row))
    for view in manual['models'].values():
        graph = view['graph']
        example = parse_graph(graph, tokenizer_fingerprint=graph['tokenizer_fingerprint'])
        balanced = apply_policy(example, graph, source_sha256='a' * 64, policy=policy)
        for node, offset in zip(graph['nodes'], example.plan.offsets):
            node['target_weights'] = list(balanced.target_weights[offset:offset+len(node['token_ids'])])
    parameters = tuple(parameter for owner in [vae, *[p.model for p in participants.values()]]
                       for parameter in owner.parameters() if parameter.requires_grad)
    for phase in ['joint', 'distill']:
        torch.manual_seed(29)
        expected = joint_loss(vae, participants, manual, phase=phase)[0]
        gradients = torch.autograd.grad(expected, parameters, allow_unused=True)
        torch.manual_seed(29)
        actual = joint_loss(vae, participants, row, phase=phase,
                            options={'_ce_supervision_policy':policy})[0]
        torch.testing.assert_close(actual, expected)
        for observed, reference in zip(torch.autograd.grad(actual, parameters, allow_unused=True), gradients):
            assert (observed is None) == (reference is None)
            if reference is not None:
                torch.testing.assert_close(observed, reference)


def test_actual_shared_semantic_resume_and_policy_extension_preserve_optimizer(tmp_path, monkeypatch):
    import json
    from natlang_neuralese.common.hashing import sha256_file_hex
    from natlang_neuralese.train import native_semantic_stage as stage
    from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
    from test_native_semantic_stage import _fixture, _assert_same
    config_path, participants_path, config, row = _fixture(tmp_path, monkeypatch)
    for view in row['models'].values():
        graph = view['graph']
        graph['nodes'][0]['name'] = 'producer'
        graph['nodes'][1]['imports'][0]['source'] = 'producer'
        view['export']['node'] = 'producer'
        graph['nodes'][0]['target_weights'][1] = .05
        graph['nodes'][1]['target_weights'][1] = .05
        graph['source_pairing'] = {'pair_id':'fixture', 'payloads':[{'invocation':'producer'}],
            'consumed_occurrences':[{'consumer_import_at':2}], 'source_ids':['w','r']}
    source = tmp_path / 'train.jsonl'
    source.write_text(json.dumps(row) + '\n')
    config['pairs']['sha256'] = sha256_file_hex(source)
    first = declared('witnessed-pair-raw-v1', digest=sha256_file_hex(source))
    policy = tmp_path / 'ce-first.json'
    policy.write_text(json.dumps(first))
    config['ce_supervision_policy'] = {'path':str(policy),'sha256':sha256_file_hex(policy)}
    config_path.write_text(json.dumps(config))
    torch.manual_seed(731)
    paused = stage.run(config_path, participants_path, tmp_path/'continued', max_updates=2, device='cpu')
    assert not paused['completed'] and paused['step'] == 2
    second = tmp_path / 'ce-extended.json'
    first['sources'].extend(declared('ordinary-suffix-raw-v1', digest='c'*64)['sources'])
    second.write_text(json.dumps(first))
    config['ce_supervision_policy'] = {'path':str(second),'sha256':sha256_file_hex(second)}
    config_path.write_text(json.dumps(config))
    checkpoint = tmp_path/'continued/checkpoint.pt'
    finished = stage.run(config_path, participants_path, tmp_path/'continued', resume=checkpoint, device='cpu')
    assert finished['completed'] and finished['step'] == 3
    saved = CheckpointPolicy.load(checkpoint)
    torch.manual_seed(731)
    assert stage.run(config_path, participants_path, tmp_path/'uninterrupted', device='cpu')['completed']
    uninterrupted = CheckpointPolicy.load(tmp_path/'uninterrupted/checkpoint.pt')
    _assert_same(saved['vae_optimizer'], uninterrupted['vae_optimizer'])
    _assert_same(saved['vae_state'], uninterrupted['vae_state'])
    _assert_same(saved['training']['rng'], uninterrupted['training']['rng'])
    for name, participant in saved['participants'].items():
        assert participant['training_options']['ce_supervision_policy_binding'] == config['ce_supervision_policy']
        _assert_same(participant['optimizer'], uninterrupted['participants'][name]['optimizer'])
        _assert_same(participant['model_state'], uninterrupted['participants'][name]['model_state'])
    config.pop('ce_supervision_policy')
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='cannot silently drop'):
        stage.run(config_path, participants_path, tmp_path/'continued', resume=checkpoint, device='cpu')
