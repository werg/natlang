import copy
import json
from pathlib import Path

import pytest
import torch

from natlang_neuralese.common.hashing import sha256_file_hex_cached
from natlang_neuralese.model.native_bundle import write_actor_view,load_actor_view
from natlang_neuralese.model.native_model import load_native_checkpoint
from natlang_neuralese.model.shared_state import Invocation,ImportSpan,compile_graph
from natlang_neuralese.serve import load_native_engine
from natlang_neuralese.serve.native_optim import _native_optimizer_step_impl
from natlang_neuralese.train.checkpoint_policy import CheckpointPolicy


@pytest.fixture(params=['lfm','mellum'])
def source(tmp_path,request):
    from test_native_serving_adapters import engine_fixture
    engine=engine_fixture(tmp_path,request.param)
    state=CheckpointPolicy.load(engine.native_checkpoint)
    state.update(step=71,training={'optimizer':{'actual_moment':torch.arange(19.)},
        'rng':{'cpu':torch.get_rng_state()},'source_receipts':{'train':{'sha256':'a'*64}},
        'sft_progress':{'native_full_sft':{'exposure_counts':{'row-'+str(i):1 for i in range(10000)}}}})
    path=tmp_path/'full.pt';torch.save(state,path)
    stat=path.stat();signature=(stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns)
    state=CheckpointPolicy.load(path)
    yield engine,path,state,signature
    engine.store.close()


def test_both_ports_exact_actor_view_without_schedule_decode(source,tmp_path,monkeypatch):
    engine,path,state,signature=source
    before=sha256_file_hex_cached(path)
    ref=write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    assert write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)==ref
    receipt=json.loads(Path(ref['path']).read_text())
    assert receipt['source']['sha256']==before and receipt['training_resume']is False
    assert receipt['view']['bytes']<path.stat().st_size
    original_load=CheckpointPolicy.load
    seen=[]
    def view_only(p,*a,**kw):
        assert Path(p).resolve()!=path.resolve(),'Actor must not decode the full training state'
        seen.append(Path(p));return original_load(p,*a,**kw)
    monkeypatch.setattr(CheckpointPolicy,'load',view_only)
    restored=load_native_engine(str(path),device='cpu',actor_view=ref)
    try:
        assert len(seen)==1
        assert restored.native_identity['checkpoint_sha256']==before
        assert restored.rl_policy_binding['actor_view']['view']['sha256']==receipt['view']['sha256']
        assert restored.native_training_state=={}
        for name,value in engine.shared_model.named_parameters():
            target=dict(restored.shared_model.named_parameters())[name]
            assert value.dtype==target.dtype and torch.equal(value,target)
        plan=compile_graph([Invocation('p',(4,5,6,7),producing=((1,3),)),
            Invocation('c',(8,9),imports=(ImportSpan(1,'p',1,3),))])
        with torch.no_grad():
            a=engine.shared_model.logits(engine.shared_model(plan)['h_final'])
            b=restored.shared_model.logits(restored.shared_model(plan)['h_final'])
        assert torch.equal(a,b)
        with pytest.raises(ValueError,match='read-only'):
            load_native_checkpoint(path,actor_view=ref)
        with pytest.raises(Exception,match='complete parent checkpoint'):
            _native_optimizer_step_impl(restored,{},'test')
    finally:restored.store.close()
    assert sha256_file_hex_cached(path)==before
    assert state['training']['sft_progress']['native_full_sft']['exposure_counts']['row-9999']==1


@pytest.mark.parametrize('field',['source','receipt','view','profile','training'])
def test_bound_actor_view_tampering_fails(source,tmp_path,field):
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    proof=Path(ref['path']);receipt=json.loads(proof.read_text())
    if field=='source':
        other=tmp_path/'other.pt';torch.save({**state,'step':72},other)
        with pytest.raises(ValueError,match='another complete'):
            load_actor_view(other,ref)
        return
    if field=='receipt':
        proof.write_text(proof.read_text()+' ')
        with pytest.raises(ValueError,match='receipt binding'):
            load_actor_view(path,ref)
        return
    view=proof.parent/receipt['view']['path']
    if field=='view':
        view.write_bytes(b'changed')
        with pytest.raises((ValueError,FileNotFoundError)):
            load_actor_view(path,ref)
        return
    value=CheckpointPolicy.load(view)
    if field=='profile':value['model_config']={**value['model_config'],'native_norm_execution':'wrong-profile'}
    else:value['training']={'optimizer':state['training']['optimizer']}
    replacement=view.with_suffix('.replacement');torch.save(value,replacement);replacement.replace(view)
    receipt['view'].update(sha256=sha256_file_hex_cached(view),bytes=view.stat().st_size)
    proof.write_text(json.dumps(receipt));ref.update(sha256=sha256_file_hex_cached(proof),bytes=proof.stat().st_size)
    with pytest.raises(ValueError,match='payload differs'):
        load_actor_view(path,ref)


def test_unsupported_view_retains_existing_full_loader(tmp_path):
    source=tmp_path/'composite.json';source.write_text('{}')
    assert write_actor_view(source,tmp_path/'views',validated_state={'schema':'unsupported-shared'})is None
    assert not(tmp_path/'views').exists()


def test_replaced_source_after_decode_cannot_receive_old_weights(source,tmp_path):
    _,path,state,signature=source
    other=path.with_suffix('.replacement');torch.save({**state,'step':72},other);other.replace(path)
    with pytest.raises(ValueError,match='unchanged file'):
        write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    assert not(tmp_path/'views').exists()


def test_actual_cursor_reuses_one_full_decode_and_replays_existing_pin(source,tmp_path,monkeypatch,capsys):
    import importlib.util
    import shutil
    import sys
    module_path=Path(__file__).parents[2]/'scripts/coreweave/cursor_checkpoint_export.py'
    spec=importlib.util.spec_from_file_location('actual_actor_view_cursor',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,path,_,_=source
    root=tmp_path/'production';slot=root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True);shutil.copyfile(path,slot)
    pins=tmp_path/'pins';views=tmp_path/'views'
    original=torch.load;loads=[]
    def counted(p,*a,**kwargs):
        loads.append(Path(p));return original(p,*a,**kwargs)
    monkeypatch.setattr(torch,'load',counted)
    code=module.remote_code(role='lfm',root=root,known={},last=-1,extra=None,
        pin_directory=pins,minimum_available_bytes=0,actor_view_directory=views,
        actor_view_runtime=Path(__file__).parents[2]/'training/neuralese')
    results=[]
    for _ in range(2):
        scope={}
        try:exec(compile(code,'<actual-cursor>','exec'),scope)
        finally:
            if 'remote_lock' in scope:scope['remote_lock'].close()
        result=json.loads(capsys.readouterr().out.splitlines()[-1]);results.append(result)
    assert results[0]['chosen']['actor_view']==results[1]['chosen']['actor_view']
    assert len(loads)==4  # one original full mmap decode + one compact verification per observation
    assert sum('cursor-observation' in p.name for p in loads)==2
    chosen=results[-1]['chosen'];state=load_actor_view(chosen['pin'],chosen['actor_view'])
    assert state['step']==71 and state['inference_only']is True
    assert CheckpointPolicy.load(chosen['pin'])['training']['optimizer']


def test_actual_actor_cli_and_provider_consumed_closure(source,tmp_path,monkeypatch,capsys):
    import importlib.util
    import sys
    from types import SimpleNamespace
    engine,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'views',validated_state=state,decoded_source_stat=signature)
    root=Path(__file__).parents[2]
    spec=importlib.util.spec_from_file_location('real_actor_cli',root/'scripts/neuralese_data/serve_native_checkpoint.py')
    cli=importlib.util.module_from_spec(spec);spec.loader.exec_module(cli)
    seen=[]
    def serve(actual,host,port):
        assert actual.native_identity['checkpoint_sha256']==sha256_file_hex_cached(path)
        assert actual.native_actor_view['receipt']==ref
        seen.append(actual.native_identity)
        return SimpleNamespace(server_address=(host,port),serve_forever=lambda:None,server_close=lambda:None)
    monkeypatch.setattr(cli,'serve',serve)
    monkeypatch.setattr(sys,'argv',['serve','--native-state','--checkpoint',str(path),
        '--checkpoint-sha256',sha256_file_hex_cached(path),'--native-actor-view',ref['path'],
        '--native-actor-view-sha256',ref['sha256'],'--device','cpu','--threads','1'])
    cli.main();assert len(seen)==1
    spec=importlib.util.spec_from_file_location('actual_view_provider',root/'scripts/coreweave/provider_dependencies.py')
    provider=importlib.util.module_from_spec(spec);spec.loader.exec_module(provider)
    config=tmp_path/'config.json';config.write_text(json.dumps({'native_actor_view':ref}))
    closure=provider.dependency_closure({'config':{'path':str(config),'sha256':sha256_file_hex_cached(config)}},namespace=str(tmp_path))
    actual={x['path']for x in closure['files']}
    view=json.loads(Path(ref['path']).read_text())['view']
    assert {str(config),ref['path'],str(Path(ref['path']).parent/view['path']),str(path)}==actual
    capsys.readouterr()


@pytest.mark.parametrize('failure',['extraction','import','pin'])
def test_optional_view_failure_keeps_real_full_pin_and_export_highwater(source,tmp_path,monkeypatch,capsys,failure):
    import importlib.util
    import shutil
    from natlang_neuralese.model import native_bundle
    spec=importlib.util.spec_from_file_location('failed_view_cursor',Path(__file__).parents[2]/'scripts/coreweave/cursor_checkpoint_export.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,path,_,_=source
    root=tmp_path/'production';slot=root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True);shutil.copyfile(path,slot)
    def failed(actual,*a,**kw):
        if failure=='pin':
            replacement=Path(actual).with_suffix('.replacement');replacement.write_bytes(b'changed pin');replacement.replace(actual)
        raise OSError('deliberate optional extraction failure')
    monkeypatch.setattr(native_bundle,'write_actor_view',failed)
    code=module.remote_code(role='lfm',root=root,known={},last=-1,extra=None,
        pin_directory=tmp_path/'pins',minimum_available_bytes=0,actor_view_directory=tmp_path/'views',
        actor_view_runtime=(tmp_path/'wrong-runtime'if failure=='import'else Path(__file__).parents[2]/'training/neuralese'))
    scope={}
    try:
        if failure=='pin':
            with pytest.raises(ValueError,match='full pin changed'):
                exec(compile(code,'<actual-cursor>','exec'),scope)
            return
        exec(compile(code,'<actual-cursor>','exec'),scope)
    finally:scope['remote_lock'].close()
    observed=json.loads(capsys.readouterr().out.splitlines()[-1]);chosen=observed['chosen']
    assert chosen['actor_view_error']=={'type':'ValueError'if failure=='import'else'OSError','fallback':'complete-checkpoint','assigned_repair':True}
    assert 'actor_view'not in chosen
    assert sha256_file_hex_cached(chosen['pin'])==sha256_file_hex_cached(path)
    assert CheckpointPolicy.load(chosen['pin'])['training']['optimizer']
    def export(actual,receipt_path):
        assert actual==chosen
        identity={'sha256':sha256_file_hex_cached(actual['pin']),'bytes':Path(actual['pin']).stat().st_size}
        return {'source':actual['pin'],'identity':identity,'independent_verification':dict(identity)}
    result=module.run_once(tmp_path/'exports','lfm',observe=lambda **kw:observed,export=export)
    assert result['verified_step']==71
    state=json.loads(module.paths_for(tmp_path/'exports','lfm')['state'].read_text())
    assert state['last_verified_step']==71 and state['actor_view_error']==chosen['actor_view_error']


@pytest.mark.parametrize('problem',[None,'wrong-sha','import-error','changes-during-write'])
def test_actual_cursor_bound_writer_sidecar_preserves_full_backup(source,tmp_path,capsys,problem):
    import hashlib,importlib.util,shutil
    module_path=Path(__file__).parents[2]/'scripts/coreweave/cursor_checkpoint_export.py'
    spec=importlib.util.spec_from_file_location('sidecar_actor_cursor',module_path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    _,path,_,_=source
    root=tmp_path/'production';slot=root/'rounds/lfm/production/round-000/learner/checkpoint.pt'
    slot.parent.mkdir(parents=True);shutil.copyfile(path,slot)
    sidecar=tmp_path/'immutable-native_bundle.py'
    raw=Path(__file__).parents[2].joinpath('training/neuralese/natlang_neuralese/model/native_bundle.py').read_bytes()
    if problem=='import-error':raw=b"raise RuntimeError('optional writer unavailable')\n"
    if problem=='changes-during-write':
        raw+=b"\noriginal_writer=write_actor_view\ndef write_actor_view(*args,**kwargs):\n result=original_writer(*args,**kwargs)\n Path(__file__).write_bytes(b'changed')\n return result\n"
    sidecar.write_bytes(raw);sha=hashlib.sha256(raw).hexdigest()
    code=module.remote_code(role='lfm',root=root,known={},last=-1,extra=None,
        pin_directory=tmp_path/'pins',minimum_available_bytes=0,
        actor_view_directory=tmp_path/'views',actor_view_runtime=Path(__file__).parents[2]/'training/neuralese',
        actor_view_sidecar=sidecar,actor_view_sidecar_sha256='0'*64 if problem=='wrong-sha'else sha)
    scope={}
    try:exec(compile(code,'<sidecar-cursor>','exec'),scope)
    finally:
        if 'remote_lock'in scope:scope['remote_lock'].close()
    result=json.loads(capsys.readouterr().out.splitlines()[-1]);chosen=result['chosen']
    assert Path(chosen['pin']).read_bytes()==slot.read_bytes()
    assert CheckpointPolicy.load(chosen['pin'])['training']['optimizer']
    if problem:
        assert 'actor_view'not in chosen and chosen['actor_view_error']['fallback']=='complete-checkpoint'
    else:
        receipt=json.loads(Path(chosen['actor_view']['path']).read_text())
        assert receipt['mechanism']['sha256']==sha and Path(receipt['mechanism']['path'])==sidecar
        assert load_actor_view(chosen['pin'],chosen['actor_view'])['step']==71


def test_physical_inference_export_cannot_publish_optimizer_updates(source,tmp_path):
    _,path,state,signature=source
    before=sha256_file_hex_cached(path)
    ref=write_actor_view(path,tmp_path/'physical-views',validated_state=state,decoded_source_stat=signature)
    receipt=json.loads(Path(ref['path']).read_text())
    view=Path(ref['path']).parent/receipt['view']['path']
    restored=load_native_engine(str(view),device='cpu')
    try:
        assert restored.native_identity['checkpoint_sha256']==receipt['view']['sha256']
        assert restored.rl_policy_binding['inference_source']['training_checkpoint_sha256']==before
        assert restored.native_training_state=={}
        with pytest.raises(Exception,match='complete parent checkpoint'):
            _native_optimizer_step_impl(restored,{},'test')
    finally:restored.store.close()
    assert sha256_file_hex_cached(path)==before
    assert state['training']['sft_progress']['native_full_sft']['exposure_counts']['row-9999']==1


@pytest.mark.parametrize('mutation',[None,'weights','step','truncated','symlink','outside'])
def test_interrupted_view_reuses_only_exact_fully_verified_output(source,tmp_path,monkeypatch,mutation):
    from natlang_neuralese.train import trajectory_state
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'original',validated_state=state,decoded_source_stat=signature)
    receipt=json.loads(Path(ref['path']).read_text())
    original=Path(ref['path']).parent/receipt['view']['path']
    out=tmp_path/'recovery';folder=out/receipt['source']['sha256'];folder.mkdir(parents=True)
    pending=folder/'.actor-view-interrupted.pt'
    import shutil
    shutil.copyfile(original,pending)
    if mutation in ('weights','step'):
        bad=copy.deepcopy(CheckpointPolicy.load(pending))
        if mutation=='step':bad['step']+=1
        else:
            key=next(iter(bad['backbone_state']))
            bad['backbone_state'][key]=bad['backbone_state'][key]+1
        torch.save(bad,pending)
    elif mutation=='truncated':pending.write_bytes(b'partial checkpoint')
    elif mutation=='symlink':
        pending.unlink();pending.symlink_to(original)
    elif mutation=='outside':
        pending=tmp_path/'.actor-view-outside.pt';shutil.copyfile(original,pending)
    def forbidden(*args,**kwargs):raise AssertionError('Recovery must not serialize actor weights again')
    monkeypatch.setattr(trajectory_state,'atomic_checkpoint',forbidden)
    source_hash=sha256_file_hex_cached(path)
    if mutation:
        with pytest.raises((ValueError,RuntimeError,AssertionError,EOFError,IndexError,__import__('pickle').UnpicklingError)):
            write_actor_view(path,out,validated_state=state,decoded_source_stat=signature,prepared_view=pending)
        assert pending.exists()
        assert not list(folder.glob('*.json'))
    else:
        recovered=write_actor_view(path,out,validated_state=state,decoded_source_stat=signature,prepared_view=pending)
        value=json.loads(Path(recovered['path']).read_text())
        assert value['view']['sha256']==receipt['view']['sha256']
        assert value['source']==receipt['source']
        assert not pending.exists()
        assert load_actor_view(path,recovered)['step']==state['step']
    assert sha256_file_hex_cached(path)==source_hash


def test_recovered_actor_view_retains_candidate_if_receipt_publication_fails(source,tmp_path,monkeypatch):
    import shutil
    from natlang_neuralese.model import native_composite
    _,path,state,signature=source
    ref=write_actor_view(path,tmp_path/'original',validated_state=state,decoded_source_stat=signature)
    value=json.loads(Path(ref['path']).read_text())
    folder=tmp_path/'recover'/value['source']['sha256'];folder.mkdir(parents=True)
    pending=folder/'.actor-view-recover.pt'
    shutil.copyfile(Path(ref['path']).parent/value['view']['path'],pending)
    def fail(*args,**kwargs):raise OSError('simulated receipt storage failure')
    monkeypatch.setattr(native_composite,'_durable_immutable',fail)
    with pytest.raises(OSError,match='receipt storage'):
        write_actor_view(path,tmp_path/'recover',validated_state=state,decoded_source_stat=signature,prepared_view=pending)
    assert pending.exists()
