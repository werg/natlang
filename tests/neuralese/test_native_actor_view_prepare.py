"""natlang pins prepare a compact actor view in one call; the server loads it under the full checkpoint's identity."""
import json
import subprocess
import sys

import pytest
import torch

from natlang_neuralese.model.native_bundle import load_actor_view, prepare_actor_view
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy
from natlang_neuralese.train.native_continuation import TRAINING_SCHEMA
from natlang_neuralese.train.loop import capture_training_rng_state


def complete(tmp_path):
    from test_native_serving_adapters import engine_fixture
    engine = engine_fixture(tmp_path, 'lfm')
    state = CheckpointPolicy.load(engine.native_checkpoint)
    state.update(step=12, training={'schema': TRAINING_SCHEMA, 'global_step': 12,
        'optimizer': {'state': {'moment': torch.arange(7.)}}, 'rng': capture_training_rng_state('cpu'),
        'source_receipts': {'source': 'retained'}})
    path = tmp_path/'full.pt'
    torch.save(state, path)
    engine.store.close()
    return path


def test_prepared_view_serves_the_full_checkpoint_identity(tmp_path):
    path = complete(tmp_path)
    reference = prepare_actor_view(path, tmp_path/'views')
    assert prepare_actor_view(path, tmp_path/'views') == reference, 'a second preparation reuses the immutable view'
    view = load_actor_view(path, reference)
    assert 'training' not in view and view['actor_view']['training_resume'] is False
    full = load_native_engine(str(path), device='cpu')
    compact = load_native_engine(str(path), device='cpu', actor_view=reference)
    assert compact.native_inference_only and not full.native_inference_only
    prompt = list(range(5, 17))
    with torch.no_grad():
        a = full._native_logits(full._native_forward(prompt, None)['h_final'])
        b = compact._native_logits(compact._native_forward(prompt, None)['h_final'])
    assert torch.equal(a, b)


def test_cli_prints_the_receipt_and_refuses_incomplete_sources(tmp_path):
    path = complete(tmp_path)
    out = subprocess.run([sys.executable, '-m', 'natlang_neuralese.model.native_bundle', 'actor-view',
                          '--checkpoint', str(path), '--out', str(tmp_path/'views')], capture_output=True, text=True, check=True)
    reference = json.loads(out.stdout.strip().splitlines()[-1])
    assert set(reference) == {'path', 'sha256', 'bytes'}
    state = CheckpointPolicy.load(path)
    del state['training']['optimizer']
    torch.save(state, tmp_path/'partial.pt')
    with pytest.raises(ValueError, match='optimizer'):
        prepare_actor_view(tmp_path/'partial.pt', tmp_path/'views')


def tiny_complete(tmp_path):
    from natlang_neuralese.model.native_model import CHECKPOINT_SCHEMA
    state={'schema':CHECKPOINT_SCHEMA,'model_config':{'architecture':'fixture'},
        'backbone_state':{'weight':torch.arange(64,dtype=torch.float32).reshape(8,8)},
        'interface_state':{'adapter':torch.arange(8,dtype=torch.float32)},'step':12,
        'training':{'optimizer':{'state':{'moment':torch.ones(8)}},'rng':{'seed':12},
                    'source_receipts':{'source':'retained'}}}
    path=tmp_path/'full.pt';torch.save(state,path);return path


def test_reuse_rechecks_values_without_second_serialization(tmp_path,monkeypatch):
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);root=tmp_path/'views'
    first=prepare_actor_view(path,root)
    def no_write(*args,**kwargs):raise AssertionError('cached view must not be serialized again')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',no_write)
    second=prepare_actor_view(path,root)
    assert second==first
    assert len(list((root/first_source_sha(path)).glob('*.pt')))==1


def first_source_sha(path):
    from natlang_neuralese.common.hashing import sha256_file_hex
    return sha256_file_hex(path)


def test_cached_receipt_and_view_corruption_fail_closed(tmp_path):
    path=tiny_complete(tmp_path);root=tmp_path/'views';ref=prepare_actor_view(path,root)
    proof=__import__('pathlib').Path(ref['path']);original=proof.read_bytes()
    proof.write_bytes(original+b' ')
    with pytest.raises(ValueError,match='receipt digest'):prepare_actor_view(path,root)
    proof.write_bytes(original)
    receipt=json.loads(original);view=proof.parent/receipt['view']['path']
    with view.open('r+b') as stream:
        stream.seek(-1,2);old=stream.read(1);stream.seek(-1,2);stream.write(bytes([old[0]^1]))
    with pytest.raises((ValueError,RuntimeError,FileNotFoundError)):
        prepare_actor_view(path,root)


def test_bounded_comparison_handles_strides_and_detects_late_difference():
    from natlang_neuralese.model.native_bundle import _same_values_bounded
    a=torch.arange(8192,dtype=torch.float32).reshape(64,128).t()
    b=a.clone();_same_values_bounded({'tensor':a},{'tensor':b},max_bytes=256)
    b[-1,-1]+=1
    with pytest.raises(RuntimeError,match='actor view differs'):
        _same_values_bounded({'tensor':a},{'tensor':b},max_bytes=256)


def test_forged_receipt_for_changed_source_fails_tensor_comparison(tmp_path):
    import hashlib,os
    from pathlib import Path
    path=tiny_complete(tmp_path);root=tmp_path/'views';ref=prepare_actor_view(path,root)
    old_proof=Path(ref['path']);old=json.loads(old_proof.read_text())
    state=CheckpointPolicy.load(path);name=next(k for k,v in state['backbone_state'].items() if v.is_floating_point() and v.numel())
    state['backbone_state'][name]=state['backbone_state'][name].clone();state['backbone_state'][name].view(-1)[0]+=1
    changed=tmp_path/'changed.pt';torch.save(state,changed)
    newdir=root/first_source_sha(changed);newdir.mkdir()
    os.link(old_proof.parent/old['view']['path'],newdir/old['view']['path'])
    old['source']={'path':str(changed.resolve()),'sha256':first_source_sha(changed),'bytes':changed.stat().st_size}
    raw=(json.dumps(old,sort_keys=True,indent=2)+'\n').encode()
    (newdir/(hashlib.sha256(raw).hexdigest()+'.json')).write_bytes(raw)
    with pytest.raises(RuntimeError,match='actor view differs'):prepare_actor_view(changed,root)


def test_transport_reuses_view_but_binds_new_runtime(tmp_path,monkeypatch):
    from natlang_neuralese.model.native_bundle import prepare_actor_transport
    from natlang_neuralese.common import runtime_inventory
    from natlang_neuralese.train import trajectory_state
    path=tiny_complete(tmp_path);root=tmp_path/'views'
    monkeypatch.setattr(runtime_inventory,'python_runtime_binding',lambda:{'sha256':'a'*64})
    first=prepare_actor_transport(path,root,scope='diagnostic-local-pin')
    def no_write(*args,**kwargs):raise AssertionError('unexpected second serialization')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',no_write)
    monkeypatch.setattr(runtime_inventory,'python_runtime_binding',lambda:{'sha256':'b'*64})
    second=prepare_actor_transport(path,root,scope='diagnostic-local-pin')
    before=json.loads(__import__('pathlib').Path(first['path']).read_text())
    after=json.loads(__import__('pathlib').Path(second['path']).read_text())
    assert before['view']==after['view'] and first['sha256']!=second['sha256']
    assert after['python_runtime']['sha256']=='b'*64
