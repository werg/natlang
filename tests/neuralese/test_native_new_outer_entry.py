import hashlib
import json
import shutil

import pytest
import torch


def test_native_entry_saves_full_transition_before_readiness(tmp_path, monkeypatch):
    from test_native_serve import _tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer, native_diloco

    base = tmp_path / 'base'
    _tiny_source(base)
    _, _, _, config = load_native_model(base, dtype=torch.float32, rank=4)
    paths = {}
    for split in ('train', 'test'):
        paths[split] = tmp_path / (split + '.jsonl')
        paths[split].write_text(json.dumps(dict(graph(split), split=split, source_groups=[split],
                                                tokenizer_fingerprint=config['tokenizer_fingerprint'])) + '\n')
    out = tmp_path / 'run'
    argv = ['--base',str(base), '--dtype','float32', '--rank','4', '--device','cpu',
            '--native-graphs',str(paths['train']), '--held-native-graphs',str(paths['test']),
            '--epochs','1', '--out',str(out)]
    assert trainer.main(argv) == 0
    slot = out / 'checkpoint.pt'
    seed = tmp_path / 'seed.pt'
    shutil.copyfile(slot, seed)
    old = tmp_path / 'old.pt'
    shutil.copyfile(slot, old)
    def ref(path):
        return {'path':str(path), 'bytes':path.stat().st_size,
                'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    original = torch.load(slot, weights_only=True)
    original['training']['options']['standalone_diloco_history'] = [{'checkpoint':ref(old)}]
    torch.save(original, slot)
    parent = tmp_path / 'peer.pt'
    shutil.copyfile(slot, parent)
    declaration = tmp_path / 'reanchor.json'
    declaration.write_text(json.dumps({'schema':'natlang.diloco-new-outer/1', 'worker':0,
        'parent':ref(parent), 'historical_outer_checkpoint':ref(old), 'seed_sha256':ref(seed)['sha256'],
        'source_hash':original['training']['source_hash'], 'held_sha256':ref(paths['test'])['sha256']}))
    class Ready(Exception): pass
    def same(left, right):
        if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)): return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
        return left == right
    class FakeLearner:
        worker=0; shard=(0,4); local_steps=0; reanchor_deviations=[]; manifest={'fingerprint':'f'*64}
        def state_dict(self, **kwargs): return {'local_steps':0, 'pending':{}}
        def await_reanchor_cohort(self, digest, **kwargs):
            saved = torch.load(slot, weights_only=True)
            assert ref(slot)['sha256'] == digest
            assert saved['training']['options']['new_outer_reanchor']['parent'] == ref(parent)
            assert saved['training']['diloco']['local_steps'] == 0
            assert saved['training']['exposure_counts'] == original['training']['exposure_counts']
            assert same(saved['training']['optimizer'], original['training']['optimizer'])
            assert same(saved['training']['rng'], original['training']['rng'])
            assert same(saved['backbone_state'], original['backbone_state'])
            assert same(saved['interface_state'], original['interface_state'])
            raise Ready()
    monkeypatch.setattr(native_diloco, 'attach_learner', lambda *args, **kwargs: FakeLearner())
    settings = json.dumps({'run_id':'new', 'workers':4, 'worker_id':0, 'coordinator_url':'http://127.0.0.1:1'})
    with pytest.raises(Ready):
        trainer.main([*argv, '--continue-from',str(seed), '--diloco',settings,
                      '--diloco-new-outer',str(declaration)])
