"""Real participant-selective restore keeps one joint identity and all recovery state."""
import json
from pathlib import Path
import shutil

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_shared_checkpoint import shared_manifest, shared_dependencies
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


def equal_tree(left, right):
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left: equal_tree(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert type(left) is type(right) and len(left) == len(right)
        for a, b in zip(left, right): equal_tree(a, b)
    else:
        assert left == right


def actual_shared(tmp_path):
    from test_native_semantic_runtime import fixture
    from natlang_neuralese.train.native_state_sft import make_native_optimizer
    from natlang_neuralese.train.native_semantic import SemanticParticipant
    from natlang_neuralese.train.native_semantic_stage import save_shared
    from natlang_neuralese.model.shared_state import Invocation, compile_graph
    participants, channel = fixture(tmp_path)
    trained = {}
    for name, entry in participants.items():
        optimizer, _ = make_native_optimizer(entry.model, lr=.001, in_backward=False)
        roles = entry.model.marker_ids
        plan = compile_graph([Invocation('answer', (4, roles[0], 5, roles[1], 6), producing=((2, 3),))])
        result = entry.model(plan)
        logits = entry.model.logits(result['h_final'])[0]
        torch.nn.functional.cross_entropy(logits[:-1].float(), torch.tensor([roles[0], 5, roles[1], 6])).backward()
        optimizer.step(); optimizer.zero_grad(set_to_none=True)
        from natlang_neuralese.train.recipe import native_recipe_binding
        recipe=tmp_path/(name+'-recipe.json');recipe.write_text(json.dumps({'model':name}))
        reference=native_recipe_binding(recipe,('a' if name=='lfm' else 'b')*64)
        trained[name] = SemanticParticipant(entry.model, entry.tokenizer, entry.config,
            channel.bindings[name], optimizer, sft_progress={'stage': {'exposure_counts': {'earned': 3}}}, global_step=7,recipe_binding=reference)
    optimizer = torch.optim.AdamW(channel.vae.parameters(), lr=.001)
    sum(value.square().mean() for value in channel.vae.parameters()).backward()
    optimizer.step(); optimizer.zero_grad(set_to_none=True)
    policy = CheckpointPolicy(tmp_path/'shared', prune_best=False)
    rng = torch.get_rng_state().clone()
    export = save_shared(policy, trained, channel.vae, optimizer,
        training={'step': 2, 'exposure': {'actual-pair': 2}, 'config_history': []}, device='cpu')
    assert torch.equal(rng, torch.get_rng_state())
    return policy, trained, channel, optimizer, export


def test_actual_both_ports_restore_selected_bundle_and_complete_joint_state(tmp_path):
    from natlang_neuralese.model.native_bundle import export_bundle, verify_bundle
    from natlang_neuralese.model.native_model import load_native_checkpoint
    from natlang_neuralese.model.semantic_channel import SemanticChannel
    policy, participants, channel, optimizer, export = actual_shared(tmp_path)
    manifest = shared_manifest(policy.slot)
    assert set(manifest['participants']) == {'lfm', 'mellum'}
    full = CheckpointPolicy.load(policy.slot)
    equal_tree(full['vae_optimizer']['state'], optimizer.state_dict())
    assert full['training']['exposure'] == {'actual-pair': 2}
    for name, entry in participants.items():
        equal_tree(full['participants'][name]['optimizer'], entry.optimizer.state_dict())
        assert full['participants'][name]['sft_progress'] == entry.sft_progress
        assert full['participants'][name]['training_options']['recipe_binding']==entry.recipe_binding
        assert export['models'][name]['recipe_binding']==entry.recipe_binding
        from natlang_neuralese.train.recipe import checkpoint_recipe_binding
        assert checkpoint_recipe_binding(policy.slot,participant=name)==entry.recipe_binding
    full_bundle = export_bundle(policy.slot, tmp_path/'full-bundle',
                                semantic_adapter=policy.out/'semantic-adapter.json')
    peer = Path(manifest['participants']['mellum']['path'])
    hidden = peer.with_name(peer.name + '.not-staged'); peer.rename(hidden)
    try:
        selected = CheckpointPolicy.load(policy.slot, shared_participant='lfm')
        assert set(selected['participants']) == {'lfm'}
        SemanticChannel.load(policy.out/'semantic-adapter.json')
        partial = export_bundle(policy.slot, tmp_path/'lfm-bundle', participant='lfm',
                                semantic_adapter=policy.out/'semantic-adapter.json')
        assert manifest['participants']['mellum']['sha256'] not in {x['sha256'] for x in partial['files']}
        with pytest.raises((ValueError, FileNotFoundError)):
            CheckpointPolicy.load(policy.slot)
    finally:
        hidden.rename(peer)
    digest = export['checkpoint']['sha256']
    assert partial['checkpoint']['sha256'] == full_bundle['checkpoint']['sha256'] == digest
    shutil.rmtree(policy.out)
    portable = verify_bundle(tmp_path/'lfm-bundle')
    model, _, _, _, state = load_native_checkpoint(portable['checkpoint']['resolved'],
        participant='lfm', base=tmp_path/'lfm', dtype=torch.float32)
    for name, parameter in participants['lfm'].model.named_parameters():
        assert torch.equal(parameter, dict(model.named_parameters())[name])
    equal_tree(state['shared_training']['optimizer'], participants['lfm'].optimizer.state_dict())
    assert state['shared_training']['sft_progress'] == participants['lfm'].sft_progress
    assert state['shared_training']['options']['recipe_binding']==participants['lfm'].recipe_binding
    restored_channel = SemanticChannel.load(portable['semantic_adapter']['resolved'])
    equal_tree(restored_channel.vae.state_dict(), channel.vae.state_dict())
    complete = CheckpointPolicy.load(full_bundle['checkpoint']['resolved'])
    for name in participants:
        equal_tree(complete['participants'][name], full['participants'][name])
    equal_tree(complete['training']['rng'], full['training']['rng'])


def test_provider_stages_only_selected_shared_participant_and_semantic_payloads(tmp_path):
    from scripts.coreweave.provider_dependencies import dependency_closure
    policy, _, _, _, _ = actual_shared(tmp_path)
    manifest = shared_manifest(policy.slot)
    def binding(path):
        return {'path': str(path), 'sha256': sha256_file_hex_cached(path), 'bytes': path.stat().st_size}
    request = {'kind': 'round', 'native_participant': 'lfm',
               'parent_checkpoint': binding(policy.slot),
               'semantic_adapter': binding(policy.out/'semantic-adapter.json')}
    files = {entry['path'] for entry in dependency_closure(request, namespace=str(tmp_path))['files']}
    assert manifest['participants']['lfm']['path'] in files
    assert manifest['participants']['mellum']['path'] not in files
    assert manifest['semantic']['path'] in files and manifest['training']['path'] in files


def test_changed_selected_payload_and_wrong_participant_fail_closed(tmp_path):
    policy, _, _, _, _ = actual_shared(tmp_path)
    manifest = shared_manifest(policy.slot)
    with pytest.raises(ValueError, match='requested participant'):
        shared_dependencies(manifest, policy.slot, participant='unknown')
    selected = Path(manifest['participants']['lfm']['path'])
    selected.write_bytes(b'changed exact payload')
    with pytest.raises(ValueError, match='changed|differs'):
        CheckpointPolicy.load(policy.slot, shared_participant='lfm')


@pytest.mark.parametrize('after_commit', [False, True])
def test_payload_cleanup_follows_actual_global_commit(tmp_path, monkeypatch, after_commit):
    from natlang_neuralese.model.native_shared_checkpoint import save_shared_checkpoint
    policy = CheckpointPolicy(tmp_path/'transaction', prune_best=False)
    def state(value):
        return {'schema': 'natlang.neuralese-native-shared-checkpoint/1',
                'participants': {'lfm': {'model_state': {'weight': torch.tensor([value])},
                                         'optimizer': {}, 'sft_progress': {'earned': value}}},
                'vae_specification': {}, 'vae_state': {}, 'vae_optimizer': {},
                'training': {'step': value, 'rng': torch.get_rng_state()}}
    save_shared_checkpoint(policy, state(1))
    old_sha = sha256_file_hex_cached(policy.slot)
    old_objects = set((policy.out/'objects').iterdir())
    save = policy.save
    def fail(manifest):
        if after_commit: save(manifest)
        raise RuntimeError('injected metadata failure' if after_commit else 'injected before commit')
    monkeypatch.setattr(policy, 'save', fail)
    with pytest.raises(RuntimeError, match='injected'):
        save_shared_checkpoint(policy, state(2))
    restored = CheckpointPolicy.load(policy.slot)
    assert restored['training']['step'] == (2 if after_commit else 1)
    if after_commit:
        assert sha256_file_hex_cached(policy.slot) != old_sha
    else:
        assert sha256_file_hex_cached(policy.slot) == old_sha
        assert set((policy.out/'objects').iterdir()) == old_objects


def test_full_and_selected_pins_survive_object_retention(tmp_path):
    from natlang_neuralese.model.native_shared_checkpoint import pin_shared_checkpoint, prune_shared_objects
    from natlang_neuralese.model.native_bundle import verify_bundle
    policy, _, _, _, _ = actual_shared(tmp_path)
    old = shared_manifest(policy.slot)
    full = pin_shared_checkpoint(policy.slot)
    selected = pin_shared_checkpoint(policy.slot, participant='lfm')
    pinned_sha = full['checkpoint']['sha256']
    state = CheckpointPolicy.load(policy.slot)
    state['training']['step'] += 1
    for participant in state['participants'].values():
        participant['global_step'] += 1
    from natlang_neuralese.model.native_shared_checkpoint import save_shared_checkpoint
    save_shared_checkpoint(policy, state)
    pruned = prune_shared_objects(policy.out)
    assert not Path(old['training']['path']).exists()
    assert not pruned['removed']  # successful save already reclaimed superseded paths
    for bundle in [full, selected]:
        checkpoint = Path(bundle['checkpoint']['resolved'])
        verified = verify_bundle(checkpoint.parent)
        assert verified['checkpoint']['sha256'] == pinned_sha
        restored = CheckpointPolicy.load(checkpoint, shared_participant=verified.get('participant'))
        assert restored['training']['step'] == 2
    assert CheckpointPolicy.load(policy.slot)['training']['step'] == 3


def test_complete_shared_export_survives_original_removal_and_rejects_partial_private_receipt(tmp_path):
    from scripts.coreweave.shared_export import prepare, release
    from scripts.coreweave.source_archive import restore
    policy, participants, channel, optimizer, export = actual_shared(tmp_path)
    expected = CheckpointPolicy.load(policy.slot)
    from natlang_neuralese.model.native_shared_checkpoint import pin_shared_checkpoint
    from natlang_neuralese.model.native_bundle import verify_bundle
    peer_consumer_pin = pin_shared_checkpoint(policy.slot, consumer='independent-full-ferry')
    prepared = prepare(policy.slot, tmp_path/'exports', reserve_bytes=0)
    assert len(prepared['closure_files']) == 6  # Global, four payloads, bundle inventory.
    assert prepared['checkpoint']['sha256'] == export['checkpoint']['sha256']
    pin = Path(prepared['pin'])
    private = {'private': True, 'source': prepared['archive']['path'],
               'identity': {key: prepared['archive'][key] for key in ('sha256','bytes')},
               'independent_verification': {'sha256': '0'*64, 'bytes': prepared['archive']['bytes']}}
    with pytest.raises(ValueError, match='independently verified'):
        release(prepared, private)
    assert pin.exists()
    private['independent_verification'] = dict(private['identity'])
    result = release(prepared, private)
    assert result['pin_released'] and not pin.exists()
    verify_bundle(Path(peer_consumer_pin['checkpoint']['resolved']).parent)
    shutil.rmtree(policy.out)
    target = tmp_path/'restored'
    restore(prepared['spool'], prepared['name'], target)
    restored = CheckpointPolicy.load(target/'checkpoint.pt')
    equal_tree(expected, restored)


def test_shared_export_refuses_to_seal_a_missing_peer(tmp_path):
    from scripts.coreweave.shared_export import prepare
    policy, participants, channel, optimizer, export = actual_shared(tmp_path)
    manifest = shared_manifest(policy.slot)
    Path(manifest['participants']['mellum']['path']).unlink()
    with pytest.raises((FileNotFoundError, ValueError)):
        prepare(policy.slot, tmp_path/'exports', reserve_bytes=0)
    assert not list((tmp_path/'exports').glob('*.tar.zst'))


def test_shared_export_release_waits_for_same_consumer_pack_without_blocking_training_commit(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    import fcntl
    import threading
    from scripts.coreweave import shared_export
    policy, participants, channel, optimizer, export = actual_shared(tmp_path)
    first = shared_export.prepare(policy.slot, tmp_path/'first', reserve_bytes=0)
    private = {'private': True, 'source': first['archive']['path'],
               'identity': {key: first['archive'][key] for key in ('sha256','bytes')},
               'independent_verification': {key: first['archive'][key] for key in ('sha256','bytes')}}
    entered, continue_pack, release_started, released = (threading.Event() for _ in range(4))
    pack = shared_export.source_archive.pack
    def blocked_pack(*args, **kwargs):
        entered.set()
        assert continue_pack.wait(10)
        return pack(*args, **kwargs)
    def releasing():
        release_started.set()
        result = shared_export.release(first, private)
        released.set()
        return result
    monkeypatch.setattr(shared_export.source_archive, 'pack', blocked_pack)
    with ThreadPoolExecutor(2) as pool:
        preparing = pool.submit(shared_export.prepare, policy.slot, tmp_path/'second', reserve_bytes=0)
        assert entered.wait(10)
        ending = pool.submit(releasing)
        assert release_started.wait(10)
        try:
            assert not released.wait(.1)
            assert Path(first['pin']).is_dir()
            # Packing owns only the export lifecycle lock, leaving training free to commit.
            with (policy.out/'.shared-checkpoint.lock').open('a+b') as stream:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                fcntl.flock(stream, fcntl.LOCK_UN)
        finally:
            continue_pack.set()
        second = preparing.result(timeout=20)
        assert ending.result(timeout=20)['pin_released']
    assert second['checkpoint']['sha256'] == first['checkpoint']['sha256']
    assert Path(second['archive']['path']).is_file()


def test_semantic_continuation_authenticates_participant_lineage_against_actual_checkpoint(tmp_path):
    import copy
    from natlang_neuralese.rl.native_supervised import bind_semantic_recipe_lineage,continuation_config
    policy,participants,_,_,export=actual_shared(tmp_path)
    current=tmp_path/'current.json'
    current.write_text(json.dumps({'stages':[]}))
    for name,entry in participants.items():
        reference=export['models'][name]['recipe_binding']
        # Historical policy files are provenance, not a new runtime dependency.
        Path(reference['path']).unlink()
        config={'production_recipe':str(current),'trainer_parameters':{},'seed_trainer_parameters':{}}
        bind_semantic_recipe_lineage(config,policy.slot,name,reference)
        assert config['recipe_lineage_sha256']==entry.recipe_binding['lineage_sha256']
        for key in ('trainer_parameters','seed_trainer_parameters'):
            bound=config[key]['recipe_binding']
            assert bound['lineage_sha256']==entry.recipe_binding['lineage_sha256']
            assert bound['sha256']!=reference['sha256']
        previous=json.loads(json.dumps(config))
        current.write_text(json.dumps({'stages':[],'changed_current_policy':name}))
        bind_semantic_recipe_lineage(config,policy.slot,name,reference,preserve_policy=True)
        assert config==previous  # Polling cannot rewrite an already sealed policy binding.
        for field,value in [('lineage_sha256','c'*64),('sha256','d'*64)]:
            forged=copy.deepcopy(reference);forged[field]=value
            with pytest.raises(ValueError,match='exact participant recipe provenance'):
                bind_semantic_recipe_lineage({'production_recipe':str(current)},policy.slot,name,forged)
        with pytest.raises(ValueError,match='exact participant recipe provenance'):
            bind_semantic_recipe_lineage({'production_recipe':str(current)},policy.slot,name,None)


def test_real_semantic_participant_load_preserves_and_authenticates_recipe_metadata(tmp_path):
    from natlang_neuralese.train.native_semantic_stage import load_participant
    policy,participants,_,_,export=actual_shared(tmp_path)
    name='lfm'
    binding={'checkpoint':{**export['checkpoint'],'participant':name},
        'native_identity':export['models'][name]['native_identity'],
        'recipe':export['models'][name]['recipe_binding']}
    config={'backbone_lr':.001,'models':{name:{'base':str(tmp_path/'lfm'),'fp8_training':'off'}}}
    loaded=load_participant(name,binding,config,'cpu')
    assert loaded.recipe_binding==participants[name].recipe_binding
    equal_tree(loaded.optimizer.state_dict(),participants[name].optimizer.state_dict())
    assert loaded.sft_progress==participants[name].sft_progress
    forged=json.loads(json.dumps(binding));forged['recipe']['lineage_sha256']='c'*64
    with pytest.raises(ValueError,match='actual checkpoint'):
        load_participant(name,forged,config,'cpu')


@pytest.mark.parametrize('boundary',['stage-complete','communication-foundation'])
def test_legacy_recipe_backfill_uses_real_readiness_and_original_stage_source_policy(tmp_path,boundary):
    import copy
    from test_native_readiness import ready
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA,model_state,native_identity
    from natlang_neuralese.train.native_state_sft import TRAINING_SCHEMA
    from natlang_neuralese.train.native_semantic_stage import load_participant
    from natlang_neuralese.train.native_readiness import validate_ready
    from natlang_neuralese.train.native_continuation import communication_foundation,progress_obligations,collect_sft_progress
    from test_native_continuation import progress
    from test_native_state_sft import source_manifest,authenticated_pair,graph
    from natlang_neuralese.data.native_sources import NativeGraphSource
    policy,participants,_,_,_=actual_shared(tmp_path)
    entry=participants['lfm']
    # Actual old-style model+optimizer state has no recipe metadata. Readiness
    # carries exact bound protocol evidence; this test does not claim that its
    # synthetic exposure/runtime receipt fields qualify real production models.
    training=progress();training.update(schema=TRAINING_SCHEMA,optimizer=entry.optimizer.state_dict(),fp8=None,
        options={'stage_id':'graph','epochs':1})
    training['sft_progress']={'graph':copy.deepcopy(training)}
    parent=tmp_path/'legacy.pt'
    torch.save({'schema':CHECKPOINT_SCHEMA,'model_config':entry.config,**model_state(entry.model),
        'training':training},parent)
    identity=native_identity(entry.model,entry.config,checkpoint_sha256=sha256_file_hex_cached(parent))
    evidence=tmp_path/'evidence';evidence.mkdir()
    path,receipt,requirement=ready(evidence,'lfm')
    receipt['checkpoint']={'schema':CHECKPOINT_SCHEMA,'path':str(parent),'sha256':identity['checkpoint_sha256']}
    receipt['native_identity']=identity
    receipt['recipe'].update(file_sha256=receipt['recipe']['sha256'],lineage_sha256='a'*64)
    requirement['recipe_lineage_sha256']='a'*64
    for role in ('runtime_certificate','task_execution_receipt'):
        proof_path=Path(receipt[role]['path']);proof=json.loads(proof_path.read_text())
        proof['native_identity']=identity;proof_path.write_text(json.dumps(proof))
        receipt[role]['sha256']=sha256_file_hex_cached(proof_path)
    report_path=Path(receipt['training_receipt']['path']);report=json.loads(report_path.read_text())
    report['checkpoint_sha256']=identity['checkpoint_sha256']
    if boundary=='communication-foundation':
        source_dir=tmp_path/'foundation';source_dir.mkdir()
        manifest=source_manifest(source_dir,[dict(graph('ordinary'),tokenizer_fingerprint=identity['tokenizer_fingerprint'])],[authenticated_pair('a',roles=entry.model.marker_ids,
            fingerprint=identity['tokenizer_fingerprint'])])
        source=NativeGraphSource(manifest,identity['tokenizer_fingerprint'],required_split='train',role_ids=entry.model.marker_ids)
        foundation=communication_foundation(source,{'lfm:a':1},epochs=1,trained_bindings=source.bindings)
        mandatory=progress_obligations(collect_sft_progress(training))
        report.update(completed=False,foundation_ready=True,communication_foundation=foundation,
            mandatory_sft=mandatory,model_config={'marker_ids':list(entry.model.marker_ids)})
        receipt.update(completed=False,foundation_ready=True,communication_foundation=foundation,mandatory_sft=mandatory)
        requirement.update(readiness_boundary=boundary,foundation_epochs=1,
            foundation_sources={'path':str(manifest),'sha256':sha256_file_hex_cached(manifest)})
    report_path.write_text(json.dumps(report));receipt['training_receipt']['sha256']=sha256_file_hex_cached(report_path)
    path.write_text(json.dumps(receipt))
    selected=validate_ready(path,requirement)
    assert selected['readiness_requirement']['stage_ids']==requirement['stage_ids']
    assert selected['readiness_requirement']['forward_atol']==requirement['forward_atol']
    config={'backbone_lr':.001,'models':{'lfm':{'base':str(tmp_path/'lfm'),'fp8_training':'off'}}}
    loaded=load_participant('lfm',selected,config,'cpu')
    assert loaded.recipe_binding==receipt['recipe']
    equal_tree(loaded.optimizer.state_dict(),entry.optimizer.state_dict())
    for damage in ('missing-policy','wrong-stage','loosened-stage'):
        forged=copy.deepcopy(selected)
        if damage=='missing-policy':forged.pop('readiness_requirement')
        elif damage=='wrong-stage':forged['stage_id']='not-declared'
        else:forged['readiness_requirement']['stage_ids']=['full']
        with pytest.raises(ValueError,match='original readiness policy|declared required native stage'):
            load_participant('lfm',forged,config,'cpu')
    if boundary=='communication-foundation':
        assert selected['readiness_requirement']['foundation_sources']==requirement['foundation_sources']
        # A replaced protected/source projection cannot inherit the old source
        # binding just because the old readiness JSON still reports complete.
        Path(requirement['foundation_sources']['path']).write_text('{}')
        with pytest.raises(ValueError,match='artifact changed'):
            load_participant('lfm',selected,config,'cpu')
