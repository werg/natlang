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
        trained[name] = SemanticParticipant(entry.model, entry.tokenizer, entry.config,
            channel.bindings[name], optimizer, sft_progress={'stage': {'exposure_counts': {'earned': 3}}}, global_step=7)
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
