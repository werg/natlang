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
    rotation={'schema':'natlang.pending-batch-rotation/1','anchor_id':'train','anchor_binding':original['training']['source_row_bindings']['train'],'parent_checkpoint_sha256':ref(old)['sha256']}
    original['training']['options']['pending_batch_rotation']=rotation
    torch.save(original, slot)
    parent = tmp_path / 'peer.pt'
    shutil.copyfile(slot, parent)
    peer_declaration = tmp_path / 'peer.json'
    peer_declaration.write_text(json.dumps({'schema':'natlang.diloco-new-outer/1', 'worker':0,
        'parent':ref(parent), 'historical_outer_checkpoint':ref(old), 'seed_sha256':ref(seed)['sha256'],
        'source_hash':original['training']['source_hash'], 'held_sha256':ref(paths['test'])['sha256']}))
    declaration = tmp_path / 'reanchor.json'
    declaration.write_text(json.dumps({'schema':'natlang.diloco-new-outer-manifest/1',
                                       'workers':4, 'peers':[ref(peer_declaration)]*4}))
    class Ready(Exception): pass
    def same(left, right):
        if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)): return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
        return left == right
    class FakeLearner:
        worker=0; shard=(0,4); local_steps=0; reanchor_deviations=[]; manifest={'fingerprint':'f'*64}
        def state_dict(self, **kwargs): return {'local_steps':0, 'pending':{}}
        def await_update_capacity(self, **kwargs):
            saved=torch.load(slot,weights_only=True)['training']['options']
            assert len(saved['pending_batch_rotation_history'])==1
            assert 'pending_batch_rotation' not in saved
            raise Ready()
        def await_reanchor_cohort(self, digest, **kwargs):
            saved = torch.load(slot, weights_only=True)
            assert ref(slot)['sha256'] == digest
            assert saved['training']['options']['new_outer_reanchor']['parent'] == ref(parent)
            options=saved['training']['options'];history=options['pending_batch_rotation_history']
            assert 'pending_batch_rotation' not in options and len(history)==1
            assert history[0]['rotation']==rotation and history[0]['new_outer_parent']==ref(parent)
            assert history[0]['new_outer_declaration_sha256']==ref(declaration)['sha256']
            assert history[0]['subsequent_schedule']=='normal-sharded-interleave'
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
    with pytest.raises(Ready):
        trainer.main([*argv, '--continue-from',str(seed), '--diloco',settings,
                      '--diloco-new-outer',str(declaration)])


def test_existing_full_slot_wins_over_original_continue_from(tmp_path, monkeypatch):
    from test_native_serve import _tiny_source
    from test_native_state_sft import graph
    from natlang_neuralese.model.native_model import load_native_model
    from natlang_neuralese.train import native_state_sft as trainer
    base=tmp_path/'base';_tiny_source(base)
    from natlang_neuralese.model import lfm2_port
    from transformers import AutoModelForCausalLM,AutoTokenizer
    def local_backbone(source,lora=None,dtype=torch.bfloat16,device='cpu'):
        model=AutoModelForCausalLM.from_pretrained(str(source),dtype=dtype).to(device).eval()
        for parameter in model.parameters():parameter.requires_grad_(False)
        return model,AutoTokenizer.from_pretrained(str(source))
    monkeypatch.setattr(lfm2_port,'load_backbone',local_backbone)
    original_from_pretrained=AutoModelForCausalLM.from_pretrained
    def cpu_load(*args,**kwargs):
        kwargs.pop('device_map',None)
        return original_from_pretrained(*args,**kwargs)
    monkeypatch.setattr(AutoModelForCausalLM,'from_pretrained',cpu_load)
    from transformers import AutoConfig
    model_type=type(AutoModelForCausalLM.from_config(AutoConfig.from_pretrained(str(base))))
    original_model_load=model_type.from_pretrained
    def cpu_model_load(*args,**kwargs):
        kwargs.pop('device_map',None)
        return original_model_load(*args,**kwargs)
    monkeypatch.setattr(model_type,'from_pretrained',cpu_model_load)
    _,_,_,config=load_native_model(base,dtype=torch.float32,rank=4)
    paths={}
    for split in ('train','test'):
        paths[split]=tmp_path/(split+'.jsonl')
        paths[split].write_text(json.dumps(dict(graph(split),split=split,source_groups=[split],tokenizer_fingerprint=config['tokenizer_fingerprint']))+'\n')
    out=tmp_path/'run';argv=['--base',str(base),'--dtype','float32','--rank','4','--device','cpu',
        '--native-graphs',str(paths['train']),'--held-native-graphs',str(paths['test']),'--epochs','1','--out',str(out)]
    assert trainer.main(argv)==0
    slot=out/'checkpoint.pt';parent=tmp_path/'parent.pt';shutil.copyfile(slot,parent)
    full=torch.load(slot,weights_only=True);old_step=full['training']['global_step'];full['training']['global_step']=old_step+11;torch.save(full,slot)
    class Loaded(Exception):pass
    actual_loader=trainer.load_native_checkpoint
    def observe(path,*args,**kwargs):
        model,tokenizer,roles,configuration,state=actual_loader(path,*args,**kwargs)
        assert path==slot and state['training']['global_step']==old_step+11
        def same(a,b):
            if isinstance(a,torch.Tensor):return isinstance(b,torch.Tensor) and torch.equal(a,b)
            if isinstance(a,dict):return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
            if isinstance(a,(tuple,list)):return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
            return a==b
        for key in ('optimizer','rng','exposure_counts'):assert full['training'][key] and same(state['training'][key],full['training'][key])
        for key in ('backbone_state','interface_state'):assert same(state[key],full[key])
        raise Loaded()
    monkeypatch.setattr(trainer,'load_native_checkpoint',observe)
    with pytest.raises(Loaded):trainer.main([*argv,'--continue-from',str(parent)])


def test_first_outer_saves_full_transition_before_readiness(tmp_path, monkeypatch):
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
    assert not original['training']['options'].get('standalone_diloco_history')
    rotation={'schema':'natlang.pending-batch-rotation/1','anchor_id':'train','anchor_binding':original['training']['source_row_bindings']['train'],'parent_checkpoint_sha256':ref(old)['sha256']}
    original['training']['options']['pending_batch_rotation']=rotation
    torch.save(original, slot)
    parent = tmp_path / 'peer.pt'
    shutil.copyfile(slot, parent)
    peer_declaration = tmp_path / 'peer.json'
    peer_declaration.write_text(json.dumps({'schema':'natlang.diloco-first-outer/1', 'worker':0,
        'parent':ref(parent), 'historical_outer_checkpoint':None, 'seed_sha256':ref(seed)['sha256'],
        'source_hash':original['training']['source_hash'], 'held_sha256':ref(paths['test'])['sha256']}))
    declaration = tmp_path / 'reanchor.json'
    declaration.write_text(json.dumps({'schema':'natlang.diloco-new-outer-manifest/1',
                                       'workers':4, 'peers':[ref(peer_declaration)]*4}))
    class Ready(Exception): pass
    def same(left, right):
        if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)): return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
        return left == right
    class FakeLearner:
        worker=0; shard=(0,4); local_steps=0; reanchor_deviations=[]; manifest={'fingerprint':'f'*64}
        def state_dict(self, **kwargs): return {'local_steps':0, 'pending':{}}
        def await_update_capacity(self, **kwargs):
            saved=torch.load(slot,weights_only=True)['training']['options']
            assert len(saved['pending_batch_rotation_history'])==1
            assert 'pending_batch_rotation' not in saved
            raise Ready()
        def await_reanchor_cohort(self, digest, **kwargs):
            saved = torch.load(slot, weights_only=True)
            assert ref(slot)['sha256'] == digest
            assert saved['training']['options']['new_outer_reanchor']['parent'] == ref(parent)
            options=saved['training']['options'];history=options['pending_batch_rotation_history']
            assert 'pending_batch_rotation' not in options and len(history)==1
            assert history[0]['rotation']==rotation and history[0]['new_outer_parent']==ref(parent)
            assert history[0]['new_outer_declaration_sha256']==ref(declaration)['sha256']
            assert history[0]['subsequent_schedule']=='normal-sharded-interleave'
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
    with pytest.raises(Ready):
        trainer.main([*argv, '--continue-from',str(seed), '--diloco',settings,
                      '--diloco-new-outer',str(declaration)])



def test_first_outer_clone_rng_commits_once_before_barrier(tmp_path, monkeypatch):
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
    assert not original['training']['options'].get('standalone_diloco_history')
    rotation={'schema':'natlang.pending-batch-rotation/1','anchor_id':'train','anchor_binding':original['training']['source_row_bindings']['train'],'parent_checkpoint_sha256':ref(old)['sha256']}
    original['training']['options']['pending_batch_rotation']=rotation
    torch.save(original, slot)
    parent = tmp_path / 'peer.pt'
    shutil.copyfile(slot, parent)
    shutil.copyfile(slot, seed)
    peer_declaration = tmp_path / 'peer.json'
    peer_declaration.write_text(json.dumps({'schema':'natlang.diloco-first-outer/1', 'worker':0,
        'parent':ref(parent), 'historical_outer_checkpoint':None, 'seed_sha256':ref(seed)['sha256'],
        'rng_fork_seed':native_diloco.first_outer_rng_seed(ref(seed)['sha256'],0),
        'source_hash':original['training']['source_hash'], 'held_sha256':ref(paths['test'])['sha256']}))
    declaration = tmp_path / 'reanchor.json'
    declaration.write_text(json.dumps({'schema':'natlang.diloco-new-outer-manifest/1',
                                       'workers':4, 'peers':[ref(peer_declaration)]*4}))
    class Ready(Exception): pass
    def same(left, right):
        if isinstance(left, torch.Tensor): return isinstance(right, torch.Tensor) and torch.equal(left, right)
        if isinstance(left, dict): return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
        if isinstance(left, (list, tuple)): return len(left) == len(right) and all(same(a,b) for a,b in zip(left,right))
        return left == right
    class FakeLearner:
        worker=0; shard=(0,4); local_steps=0; reanchor_deviations=[]; manifest={'fingerprint':'f'*64}
        def state_dict(self, **kwargs): return {'local_steps':0, 'pending':{}}
        def await_update_capacity(self, **kwargs):
            saved=torch.load(slot,weights_only=True)['training']['options']
            assert len(saved['pending_batch_rotation_history'])==1
            assert 'pending_batch_rotation' not in saved
            raise Ready()
        def await_reanchor_cohort(self, digest, **kwargs):
            saved = torch.load(slot, weights_only=True)
            assert ref(slot)['sha256'] == digest
            assert saved['training']['options']['new_outer_reanchor']['parent'] == ref(parent)
            options=saved['training']['options'];history=options['pending_batch_rotation_history']
            assert 'pending_batch_rotation' not in options and len(history)==1
            assert history[0]['rotation']==rotation and history[0]['new_outer_parent']==ref(parent)
            assert history[0]['new_outer_declaration_sha256']==ref(declaration)['sha256']
            assert history[0]['subsequent_schedule']=='normal-sharded-interleave'
            assert saved['training']['diloco']['local_steps'] == 0
            assert saved['training']['exposure_counts'] == original['training']['exposure_counts']
            assert same(saved['training']['optimizer'], original['training']['optimizer'])
            assert not same(saved['training']['rng'], original['training']['rng'])
            assert saved['training']['options']['new_outer_reanchor']['rng_fork']['policy']=='rank-derived-clone/1'
            assert same(saved['backbone_state'], original['backbone_state'])
            assert same(saved['interface_state'], original['interface_state'])
            raise Ready()
    monkeypatch.setattr(native_diloco, 'attach_learner', lambda *args, **kwargs: FakeLearner())
    settings = json.dumps({'run_id':'new', 'workers':4, 'worker_id':0, 'coordinator_url':'http://127.0.0.1:1'})
    with pytest.raises(Ready):
        trainer.main([*argv, '--continue-from',str(seed), '--diloco',settings,
                      '--diloco-new-outer',str(declaration)])
    with pytest.raises(Ready):
        trainer.main([*argv, '--continue-from',str(seed), '--diloco',settings,
                      '--diloco-new-outer',str(declaration)])

