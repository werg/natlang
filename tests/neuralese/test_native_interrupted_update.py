"""Real process custody and full saved state for an interrupted versioned worker."""
import copy
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
import pytest
import torch
from test_native_owned_operation_reconciliation import owned
from test_native_ordinary_runtime_handoff import ordinary
from test_native_bare_runtime_handoff import bare
from test_native_later_runtime_handoff import later
from natlang_neuralese.rl import native_raw_handoff as h, native_bridge as bridge
from natlang_neuralese.rl import native_interrupted_update as interrupted
from natlang_neuralese.train.native_resident import binding, code_inventory
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import write_json_durable as write


@pytest.fixture
def stopped(owned,tmp_path,request):
    x=owned;W=tmp_path/'interrupted';W.mkdir();out=x['slot'].parent
    boundary_options=getattr(request,'param',{})
    runtime=W/'runtime';package=runtime/'training/neuralese/natlang_neuralese'
    (package/'train').mkdir(parents=True)
    for p in (package/'__init__.py',package/'train/__init__.py'):p.write_text('')
    signal_proof=W/'actual-trainer-signal.txt';ready=W/'trainer-ready'
    (package/'train/native_state_sft.py').write_text(
        'import sys,signal\nfrom pathlib import Path\n'
        'signal.signal(signal.SIGTERM,lambda *_:Path('+repr(str(signal_proof))+').write_text("SIGTERM"))\n'
        'Path('+repr(str(ready))+').touch()\nsys.stdin.buffer.read()')
    entry=W/'worker_entry-v6.py';entry.write_text("import subprocess,sys,json;p=subprocess.Popen(json.loads(sys.argv[-1]),stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate();sys.exit(1)")
    parent=json.loads(x['owner_continuation'].read_text())['parent']
    config=W/'config.json';write(config,{'fixture':'actual process protocol; no production capacity credit'})
    request={'schema':bridge.REQUEST_SCHEMA,'kind':'round','out':str(out.parent),'attempt_id':'interrupted',
        'config':binding(config),'parent_checkpoint':parent,'native_actor_mode':'raw-state'}
    request['request_id']=digest(request);request_path=out.parent/'stopped.request.json';write(request_path,request)
    N=W/'resident';N.mkdir();capacity=W/'capacity.json';result_path=W/'failed-result.json'
    rb=x['saved']['training']['options']['recipe_binding']
    child=[sys.executable,'-m','natlang_neuralese.train.native_state_sft','--out',str(out),
        '--native-graphs',str(x['slot']),
        '--held-native-graphs',str(x['slot']),
        '--stage-id','sft','--recipe-binding',json.dumps(rb),'--resident-data-directory',str(N),
        '--resident-geometry-qualification',str(capacity)]
    if boundary_options.get('quantum',3)is not None:
        child.extend(['--refresh-after-updates',str(boundary_options.get('quantum',3))])
    argv=[sys.executable,'-u',str(entry),'--request',str(request_path),'--result',str(result_path),'--fixture-child',json.dumps(child)]
    spec=W/'worker-spec.json';write(spec,{'request_id':request['request_id'],'kind':'round','runtime':str(runtime),
        'argv':argv,'env':{'NATLANG_WORKER_ENTRY_SHA256':binding(entry)['sha256']}})
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,
        env={**os.environ,'PYTHONPATH':str(package.parent),'NATLANG_WORKER_ENTRY_SHA256':binding(entry)['sha256']})
    try:
        assert process.stdout.readline().strip()=='ready'
        for _ in range(200):
            if ready.exists():break
            time.sleep(.01)
        assert ready.exists()
        launch=W/'launch.json';write(launch,{'pid':process.pid,'argv':argv})
        observed=W/'observed.json';write(observed,{'at':time.time(),'pid':process.pid,'argv':argv,
            'identity':h.observe(process.pid),'request':request['request_id'],'original_provider_launch':binding(launch)})
        intent=h.capture_drain_intent(binding(observed),W/'intent.json')
        captured=json.loads(Path(intent['path']).read_text());trainer=captured['children'][0]
        environment=interrupted.capture_provider_environment(captured['owner'],W/'provider-environment.json')
        stop=W/'stop.json';interrupted.request_operational_stop(intent,trainer,binding(spec),
            reason='apply a better task evaluator without resetting earned work',out=stop)
        observed_signal=None
        for _ in range(200):
            try:observed_signal=signal_proof.read_text()
            except FileNotFoundError:pass
            if observed_signal=='SIGTERM':break
            time.sleep(.01)
        assert observed_signal=='SIGTERM'
        process.communicate();assert process.returncode==1
    finally:
        if process.poll()is None:process.communicate()
    saved=copy.deepcopy(x['saved']);training=saved['training'];training['step']=3
    training.setdefault('source_row_bindings',{key:digest(key)for key in training['exposure_counts']})
    training.setdefault('held_row_bindings',{'held':digest('held')})
    training.setdefault('sft_progress',{})
    saved['step']=training['global_step'];training['source_row_order']=list(training['source_row_bindings'])
    training['options']['continuation']={'checkpoint_sha256':parent['sha256']}
    torch.save(saved,x['slot']);shutil.copyfile(x['slot'],x['pin']);sha=binding(x['pin'])['sha256']
    receipt=json.loads(x['operation_receipt'].read_text());receipt.update(checkpoint=binding(x['slot']),completed=False)
    receipt['continuation_boundary'].update(reached=boundary_options.get('reached',False),
        kind=boundary_options.get('kind','semantic-refresh'),checkpoint_sha256=sha)
    receipt['native_identity']['checkpoint_sha256']=sha;write(x['operation_receipt'],receipt)
    identity={'model_config':saved['model_config']};evidence=W/'capacity-proof.json';write(evidence,{'scope':'CPU protocol fixture, not GPU capacity'})
    write(capacity,{'schema':'natlang.native-resident-geometry-qualification/1','passed':True,'identity':identity,
        'runtime_code_sha256':digest(code_inventory(package)),'evidence':binding(evidence),'limits':{'forward_tokens':12,'kv_slots':128}})
    plan=N/'recipe-plan.json';write(plan,{'code':code_inventory(package),'stage_inputs':{'sft':{
        name:training['source_receipts'][i]for i,name in enumerate(('native_graphs','held_native_graphs'))}}})
    initial=N/'initial.json';write(initial,{'owner':trainer,'geometry_identity':identity,'geometry_qualification':binding(capacity),
        'plan_sha256':digest(json.loads(plan.read_text())),'stage_id':'sft','rows':len(training['source_row_bindings']),
        'held_rows':len(training['held_row_bindings'])})
    report=out/'report.json';write(report,{'schema':'natlang.neuralese-native-sft-report/1','completed':False,
        'step':3,'checkpoint':str(x['slot']),'checkpoint_sha256':sha,'continuation':{'checkpoint_sha256':parent['sha256']},'model_config':saved['model_config']})
    certificate=out/'native-runtime-certificate.json';write(certificate,{'passed':False,'deferred':True,
        'reason':'operational stop; qualification deferred until resume','native_identity':{'checkpoint_sha256':sha}})
    write(out/'progress.json',{'step':3,'global_step':training['global_step']})
    observation=interrupted.observe_checkpoint(binding(x['pin']),W/'saved-frame.json')
    write(result_path,{'schema':bridge.RESULT_SCHEMA,'kind':'round','status':'failed','request_id':request['request_id'],
        'request_sha256':binding(request_path)['sha256'],'error':'actual stopped protocol owner','artifacts':bridge.inventory(out.parent)})
    exit_path=W/'exit.json';write(exit_path,{'at':time.time(),'exit_code':1})
    terminal=W/'terminal.json';write(terminal,{'at':json.loads(exit_path.read_text())['at'],'returncode':1,
        'actual_provider_exit':binding(exit_path),'actual_operation_result':binding(result_path)})
    recovery=W/'recovery.json';write(recovery,{'schema':interrupted.SCHEMA,'at':time.time(),
        'captured_owner_intent':intent,'actual_failed_terminal':binding(terminal),'actual_failed_result':binding(result_path),
        'original_request':binding(spec),'provider_entry':binding(entry),'provider_environment':environment,'operational_stop':binding(stop),
        'original_operation_still_failed':True,'complete_optimizer_rng_and_ordinary_exposure_verified':True,
        'actual_new_optimizer_updates':3,'saved_global_step':training['global_step'],'saved_stage_step':3,
        'checkpoint':binding(x['pin']),'original_slot':binding(x['slot']),'checkpoint_bytes':x['pin'].stat().st_size,
        'independent_pin_inode':[x['pin'].stat().st_dev,x['pin'].stat().st_ino],
        'original_slot_inode':[x['slot'].stat().st_dev,x['slot'].stat().st_ino],
        'report':binding(report),'runtime_certificate':binding(certificate),'saved_frame_observation':observation,
        'resident':{'directory':str(N),'initial':binding(initial),'plan':binding(plan),'current':None,'generation':None}})
    return dict(recovery=recovery,receipt=binding(x['operation_receipt']),ack=W/'ack.json',result=result_path,
        observation=observation,pin=x['pin'],saved=saved,stop=stop,capacity=capacity,entry=entry)


def test_versioned_interruption_recovers_earned_state_without_completed_boundary(stopped):
    x=stopped;original=x['result'].read_bytes();pin=x['pin'].read_bytes()
    ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    record=h.verify_saved_update_acknowledgement(ack,receipt=x['receipt'])
    assert record['outcome']=={'status':'failed','returncode':1,'recovered_optimizer_updates':3,'communication_runtime_qualified':False}
    assert record['resident']['current']is None
    assert json.loads(Path(ack['path']).read_text())['payload_verification']['independent']is True
    assert x['result'].read_bytes()==original and x['pin'].read_bytes()==pin
    with pytest.raises(ValueError):h.verify_handoff_controls(ack)


@pytest.mark.parametrize('stopped',[{'reached':True,'quantum':3}],indirect=True)
def test_reached_refresh_with_interrupted_evaluation_retains_failed_operation(stopped):
    x=stopped;failed=x['result'].read_bytes();pin=x['pin'].read_bytes()
    ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    record=h.verify_saved_update_acknowledgement(ack,receipt=x['receipt'])
    assert record['outcome']=={'status':'failed','returncode':1,'recovered_optimizer_updates':3,
                              'communication_runtime_qualified':False}
    assert x['result'].read_bytes()==failed and x['pin'].read_bytes()==pin
    with pytest.raises(ValueError):h.verify_handoff_controls(ack)


@pytest.mark.parametrize('stopped',[
    {'reached':True,'quantum':4},{'reached':True,'quantum':0},
    {'reached':True,'quantum':None},{'reached':True,'quantum':3,'kind':'completed'},
    {'reached':1,'quantum':3},{'reached':'true','quantum':3},
],indirect=True)
def test_reached_operational_boundary_requires_boolean_and_actual_earned_quantum(stopped):
    x=stopped
    with pytest.raises(ValueError):
        h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    assert not x['ack'].exists()


@pytest.mark.parametrize('tamper',[None,'history','rows','source','profile'])
def test_adopted_generation_and_executed_profile_are_bound(stopped,tamper):
    from natlang_neuralese.train import native_resident as resident
    x=stopped;r=json.loads(x['recovery'].read_text());e=r['resident']
    initial=json.loads(Path(e['initial']['path']).read_text())
    plan=json.loads(Path(e['plan']['path']).read_text())
    observation=json.loads(Path(x['observation']['path']).read_text())
    payload={'plan':plan,'rows':observation['train_rows'],'held_rows':observation['held_rows'],
        'train':observation['source_receipts'][0],'held':observation['source_receipts'][1]}
    history=[{'generation':'actual-prior-source'}]
    current={'schema':resident.SCHEMA,'payload':payload,'generation':digest(payload),
        'at_global_step':observation['global_step'],'history':history}
    observation['resident_adoption']={'generation':current['generation'],'history':history}
    if tamper=='history':observation['resident_adoption']['history']=[]
    if tamper=='rows':current['payload']['rows']+=1
    if tamper=='source':current['payload']['train']={**payload['train'],'sha256':'0'*64}
    if tamper=='profile':
        capacity_path=Path(initial['geometry_qualification']['path'])
        capacity=json.loads(capacity_path.read_text())
        capacity['identity']['batch_graphs']=999
        initial['geometry_identity']['batch_graphs']=999
        write(capacity_path,capacity);initial['geometry_qualification']=binding(capacity_path)
        initial_path=Path(e['initial']['path']);write(initial_path,initial);e['initial']=binding(initial_path)
    current_path=Path(e['directory'])/'current.json';write(current_path,current)
    e.update(current=binding(current_path),generation=current['generation'])
    observation_path=Path(x['observation']['path']);write(observation_path,observation)
    r['saved_frame_observation']=binding(observation_path);write(x['recovery'],r)
    if tamper is None:
        ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
        assert h.verify_saved_update_acknowledgement(ack,receipt=x['receipt'])['resident']['generation']==current['generation']
    else:
        with pytest.raises(ValueError):h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
        assert not x['ack'].exists()


@pytest.mark.parametrize('tamper',['missing','changed'])
def test_historical_ack_requires_its_first_payload_verification(stopped,tamper):
    x=stopped;ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    value=json.loads(Path(ack['path']).read_text())
    if tamper=='missing':value.pop('payload_verification')
    else:value['payload_verification']['bytes']+=1
    write(x['ack'],value)
    with pytest.raises(ValueError):h.verify_saved_update_acknowledgement(binding(x['ack']),receipt=x['receipt'])


@pytest.mark.parametrize('tamper',['entry','stop','capacity','observation','boundary','pin','missing-pin','pin-inode'])
def test_interruption_does_not_relax_process_profile_or_actual_state(stopped,tamper):
    x=stopped
    if tamper=='entry':x['entry'].write_text('# replaced entry')
    elif tamper=='stop':x['stop'].write_text('{}')
    elif tamper=='capacity':x['capacity'].write_text('{}')
    elif tamper=='observation':Path(x['observation']['path']).write_text('{}')
    elif tamper=='pin':x['pin'].write_bytes(b'changed saved state')
    elif tamper=='missing-pin':x['pin'].unlink()
    elif tamper=='pin-inode':
        replacement=x['pin'].with_suffix('.replacement');shutil.copyfile(x['pin'],replacement);os.replace(replacement,x['pin'])
    else:
        r=json.loads(x['recovery'].read_text());r['actual_new_optimizer_updates']=999;write(x['recovery'],r)
    with pytest.raises((ValueError,KeyError,FileNotFoundError)):h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['receipt'],x['ack'])
    assert not x['ack'].exists()


@pytest.mark.parametrize('actual_sha',[None,'0'*64])
def test_worker_spec_cannot_replace_missing_or_wrong_actual_entry_environment(tmp_path,actual_sha):
    entry=tmp_path/'entry.py';entry.write_text("import sys;print('ready',flush=True);sys.stdin.buffer.read()")
    argv=[sys.executable,'-u',str(entry)];env=dict(os.environ);env.pop('NATLANG_WORKER_ENTRY_SHA256',None)
    if actual_sha is not None:env['NATLANG_WORKER_ENTRY_SHA256']=actual_sha
    process=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
    try:
        assert process.stdout.readline().strip()=='ready'
        owner=h.observe(process.pid)
        assert owner and h.process_argv(owner)==argv
        intent={'owner':owner,'at':time.time()};out=tmp_path/'environment.json'
        if actual_sha is None:
            with pytest.raises(ValueError):interrupted.capture_provider_environment(owner,out)
            assert not out.exists()
        else:
            reference=interrupted.capture_provider_environment(owner,out)
            stop=tmp_path/'stop.json';write(stop,{'at':time.time()})
            recovery={'provider_entry':binding(entry),'provider_environment':reference,'operational_stop':binding(stop)}
            spec={'env':{'NATLANG_WORKER_ENTRY_SHA256':binding(entry)['sha256']}}
            assert interrupted.bridge_invocation(spec,argv,recovery,intent)is False
    finally:process.communicate()
