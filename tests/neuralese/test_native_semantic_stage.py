"""Actual VAE/joint/distill optimizer updates, complete exposure and resume."""
import copy
import json
from pathlib import Path

import torch

from natlang_neuralese.common.hashing import canonical_json_sha256_hex, sha256_file_hex
from natlang_neuralese.train import native_semantic_stage as stage
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_state_sft import make_native_optimizer
from test_native_semantic import setup_pair


def _fixture(tmp_path, monkeypatch):
    participants, row, _ = setup_pair()
    parent = tmp_path/'parent'; parent.write_bytes(b'actual test parent identity')
    bindings = []
    for name, participant in participants.items():
        participant.config['architecture'] = participant.identity['architecture']
        bindings.append({'model_id': name, 'checkpoint': {'path': str(parent), 'sha256': sha256_file_hex(parent)},
                         'native_identity': participant.identity})
    manifest = tmp_path/'participants.json'
    manifest.write_text(json.dumps({'schema': 'natlang.native-shared-participants/1', 'models': bindings,
                                    'signature': canonical_json_sha256_hex(bindings)}))
    def artifact(name, value):
        path = tmp_path/name; path.write_text(json.dumps(value) + '\n')
        return {'path': str(path), 'sha256': sha256_file_hex(path)}
    held = copy.deepcopy(row); held['id'] = 'held:payload'; held['split'] = 'test'; held['source_groups'] = ['held:one']
    for name, view in held['models'].items():
        view['graph'].update(id=name + ':held', split='test', source_groups=['held:one'])
    config = {'schema': stage.CONFIG_SCHEMA, 'pairs': artifact('train.jsonl', row),
        'held_pairs': artifact('held.jsonl', held), 'backbone_lr': 1e-4, 'vae_lr': 1e-3,
        'latent_width': 6, 'hidden_width': 12, 'checkpoint_every': 1, 'eval_every': 1,
        'phases': [{'id': kind, 'kind': kind, 'epochs': 1} for kind in ('vae', 'joint', 'distill')]}
    config_path = tmp_path/'config.json'; config_path.write_text(json.dumps(config))
    def load(name, binding, config, device):
        fresh = setup_pair()[0][name]
        fresh.config['architecture'] = fresh.identity['architecture']
        if name == 'mellum':
            # The tiny fixture uses dense experts/attention in FP32 for CPU
            # exactness. Select the dense latent policy, never ternary codes.
            fresh.model.port.hf.natlang_precision = 'bf16'
        fresh.optimizer = make_native_optimizer(fresh.model, lr=config['backbone_lr'], in_backward=False)[0]
        return fresh
    monkeypatch.setattr(stage, 'load_participant', load)
    return config_path, manifest, config, row


def _assert_same(left, right):
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_same(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_same(a, b)
    else:
        assert left == right


def test_complete_curriculum_and_interrupted_resume_are_exact(tmp_path, monkeypatch):
    config, manifest, _, row = _fixture(tmp_path, monkeypatch)
    torch.manual_seed(1234)
    uninterrupted = stage.run(config, manifest, tmp_path/'whole', device='cpu')
    assert uninterrupted['completed'] and uninterrupted['step'] == 3
    torch.manual_seed(1234)
    interrupted = stage.run(config, manifest, tmp_path/'continued', max_updates=2, device='cpu')
    assert not interrupted['completed'] and interrupted['missing_exposure'][0]['phase'] == 'distill'
    checkpoint = tmp_path/'continued'/'checkpoint.pt'
    resumed = stage.run(config, manifest, tmp_path/'continued', resume=checkpoint, device='cpu')
    assert resumed['completed'] and resumed['step'] == 3
    whole = CheckpointPolicy.load(tmp_path/'whole'/'checkpoint.pt')
    restored = CheckpointPolicy.load(checkpoint)
    _assert_same(whole['vae_state'], restored['vae_state'])
    _assert_same(whole['vae_optimizer'], restored['vae_optimizer'])
    for name in whole['participants']:
        _assert_same(whole['participants'][name]['model_state'], restored['participants'][name]['model_state'])
        _assert_same(whole['participants'][name]['optimizer'], restored['participants'][name]['optimizer'])
    _assert_same(whole['training']['rng'], restored['training']['rng'])
    assert all(counts[stage.pair_key('lfm', 'mellum')] == 1
               for values in resumed['exposure'].values() for counts in values.values())
    export = json.loads((tmp_path/'continued'/'semantic-adapter.json').read_text())
    assert export['checkpoint']['sha256'] == sha256_file_hex(checkpoint)
    assert {value['native_identity']['checkpoint_sha256'] for value in export['models'].values()} == {
        export['checkpoint']['sha256']}


def test_appended_data_and_larger_epochs_are_not_mistaken_for_finished(tmp_path, monkeypatch):
    path, manifest, config, row = _fixture(tmp_path, monkeypatch)
    first = stage.run(path, manifest, tmp_path/'run', device='cpu')
    assert first['completed']
    added = copy.deepcopy(row); added['id'] = 'additional:payload'
    train = Path(config['pairs']['path'])
    train.write_text(train.read_text() + json.dumps(added) + '\n')
    config['pairs']['sha256'] = sha256_file_hex(train)
    config['phases'][0]['epochs'] = 2
    path.write_text(json.dumps(config))
    report = stage.run(path, manifest, tmp_path/'run', resume=tmp_path/'run'/'checkpoint.pt', device='cpu')
    assert report['completed'] and report['step'] == 8
    for counts in report['exposure']['vae'].values():
        assert counts[stage.pair_key('lfm', 'mellum')] == 2
