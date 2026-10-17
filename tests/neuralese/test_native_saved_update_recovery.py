"""A genuinely failed process can retain earned work without a successful handoff."""
import copy,json,os,shutil,subprocess,sys,time
from pathlib import Path
import pytest
from test_native_owned_operation_reconciliation import owned
from test_native_ordinary_runtime_handoff import ordinary
from test_native_bare_runtime_handoff import bare
from test_native_later_runtime_handoff import later
from natlang_neuralese.rl import native_raw_handoff as h,native_raw_rounds as raw,native_bridge as bridge
from natlang_neuralese.train.native_resident import binding,code_inventory
from natlang_neuralese.common.hashing import canonical_json_sha256_hex as digest
from natlang_neuralese.common.jsonio import write_json_durable as write

@pytest.fixture
def recovered(owned,tmp_path):
 x=owned;W=tmp_path/'failed-worker';W.mkdir();root=x['slot'].parent.parent;out=x['slot'].parent
 runtime=tmp_path/'failed-runtime';package=runtime/'training/neuralese/natlang_neuralese';package.mkdir(parents=True)
 (package/'__init__.py').write_text('');(package/'rl').mkdir();(package/'rl/__init__.py').write_text('')
 (package/'rl/native_bridge.py').write_text("import subprocess,sys,json;p=subprocess.Popen(json.loads(sys.argv[-1]),stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate();sys.exit(1)")
 parent=json.loads(x['owner_continuation'].read_text())['parent'];config=root/'failed-config.json';write(config,{'fixture':'actual failed owner control only'})
 req={'schema':bridge.REQUEST_SCHEMA,'kind':'round','out':str(root),'attempt_id':'real-failed-fixture','config':binding(config),'parent_checkpoint':parent,'native_actor_mode':'raw-state'};req['request_id']=digest(req);request=root/'failed.request.json';write(request,req)
 result_path=W/'result.json';rb=x['saved']['training']['options']['recipe_binding'];child=[sys.executable,'-c','import sys;sys.stdin.buffer.read()', '--out',str(out),'--stage-id','sft','--recipe-binding',json.dumps(rb),'--task-eval','']
 argv=[sys.executable,'-u','-m','natlang_neuralese.rl.native_bridge','--request',str(request),'--result',str(result_path),'--fixture-child',json.dumps(child)]
 spec=W/'worker-spec.json';write(spec,{'request_id':req['request_id'],'kind':'round','runtime':str(runtime),'argv':argv})
 env={**os.environ,'PYTHONPATH':str(package.parent)};p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,env=env)
 try:
  assert p.stdout.readline().strip()=='ready';owner=h.observe(p.pid);launch=W/'child-launch.json';write(launch,{'pid':p.pid,'argv':argv})
  observed=W/'observed.json';write(observed,{'at':time.time(),'pid':p.pid,'argv':argv,'identity':owner,'request':req['request_id'],'original_provider_launch':binding(launch)})
  intent=h.capture_drain_intent(binding(observed),W/'drain-intent.json');p.communicate();assert p.returncode==1
 finally:
  if p.poll()is None:p.communicate()
 exit_path=W/'child-exit.json';write(exit_path,{'at':time.time(),'exit_code':1})
 receipt=json.loads(x['operation_receipt'].read_text());receipt['continuation_boundary']['reached']=True;write(x['operation_receipt'],receipt)
 sha=binding(x['pin'])['sha256'];step=3;global_step=x['saved']['training']['global_step']
 write(out/'progress.json',{'step':step,'global_step':global_step})
 report=out/'report.json';write(report,{'schema':'natlang.neuralese-native-sft-report/1','step':step,'checkpoint':str(x['slot']),'checkpoint_sha256':sha,'continuation':{'checkpoint_sha256':parent['sha256']},'model_config':x['saved']['model_config']})
 cert=out/'native-runtime-certificate.json';write(cert,{'passed':False,'deferred':True,'native_identity':{'checkpoint_sha256':sha}})
 quality=out/'task-eval.jsonl';quality.write_text('{}\n')
 identity={'model_config':x['saved']['model_config']};request_memory=out/'memory-request.json';write(request_memory,{'identity':identity,'runtime_code_sha256':digest(code_inventory(package))})
 proof=out/'memory-proof.json';proofvalue={'passed':True,'parent_checkpoint_sha256':parent['sha256'],'request':binding(request_memory),'identity':identity,'runtime_code_sha256':digest(code_inventory(package))};write(proof,proofvalue)
 memory=out/'memory-qualification.json';remembered={'passed':True,'parent_checkpoint_sha256':parent['sha256'],'request':binding(request_memory)};write(memory,{**remembered,'proof':binding(proof)})
 observation=W/'saved-frame.json';write(observation,{'model_config':x['saved']['model_config'],'memory_qualification':remembered})
 result={'schema':bridge.RESULT_SCHEMA,'kind':'round','status':'failed','request_id':req['request_id'],'request_sha256':binding(request)['sha256'],'error':'actual fixture exit1 after saved update','artifacts':bridge.inventory(root)};write(result_path,result)
 terminal=W/'terminal.json';write(terminal,{'at':json.loads(exit_path.read_text())['at'],'returncode':1,'actual_provider_exit':binding(exit_path),'actual_operation_result':binding(result_path)})
 recovery=W/'saved-update-recovery.json';write(recovery,{'schema':h.SAVED_UPDATE_RECOVERY,'at':time.time(),'captured_owner_intent':intent,'actual_failed_terminal':binding(terminal),'actual_failed_result':binding(result_path),'original_request':binding(spec),'original_operation_still_failed':True,'complete_optimizer_rng_and_ordinary_exposure_verified':True,'actual_new_optimizer_updates':3,'saved_global_step':global_step,'saved_stage_step':step,'checkpoint':binding(x['pin']),'original_slot':binding(x['slot']),'checkpoint_bytes':x['pin'].stat().st_size,'independent_pin_inode':[x['pin'].stat().st_dev,x['pin'].stat().st_ino],'original_slot_inode':[x['slot'].stat().st_dev,x['slot'].stat().st_ino],'report':binding(report),'runtime_certificate':binding(cert),'partial_quality':binding(quality),'saved_frame_observation':binding(observation)})
 x.update(recovery=recovery,failed_result=result_path,failed_bytes=result_path.read_bytes(),failed_receipt=binding(x['operation_receipt']),ack=W/'ack.json',proof=proof)
 return x

def test_actual_failed_owner_acknowledgement_preserves_failure_and_earned_updates(recovered):
 x=recovered;ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])
 value=h.verify_saved_update_acknowledgement(ack,receipt=x['failed_receipt'])
 assert value['schema']==h.SAVED_UPDATE_RECOVERY and value['outcome']['returncode']==1
 assert value['outcome']['recovered_optimizer_updates']==3 and not value['outcome']['communication_runtime_qualified']
 assert x['failed_result'].read_bytes()==x['failed_bytes']
 with pytest.raises(ValueError):h.verify_handoff_controls(ack)
 with pytest.raises(ValueError):h.verify_handoff_controls(binding(x['recovery']))
 assert h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])==ack

@pytest.mark.parametrize('tamper',['updates','failed-status','inventory','memory-proof','live-owner'])
def test_recovery_cannot_borrow_unearned_or_unqualified_state(recovered,tamper,monkeypatch):
 x=recovered
 if tamper=='updates':
  v=json.loads(x['recovery'].read_text());v['actual_new_optimizer_updates']=99;write(x['recovery'],v)
 elif tamper in('failed-status','inventory'):
  result=json.loads(x['failed_result'].read_text())
  if tamper=='failed-status':result['status']='completed'
  else:result['artifacts']=[r for r in result['artifacts']if r['path']!=x['failed_receipt']['path']]
  write(x['failed_result'],result);v=json.loads(x['recovery'].read_text());v['actual_failed_result']=binding(x['failed_result']);terminal=Path(v['actual_failed_terminal']['path']);tv=json.loads(terminal.read_text());tv['actual_operation_result']=v['actual_failed_result'];write(terminal,tv);v['actual_failed_terminal']=binding(terminal);write(x['recovery'],v)
 elif tamper=='memory-proof':write(x['proof'],{'passed':False})
 else:
  intent=json.loads(Path(json.loads(x['recovery'].read_text())['captured_owner_intent']['path']).read_text());monkeypatch.setattr(h,'observe',lambda pid:intent['owner']if pid==intent['owner']['pid']else None)
 with pytest.raises((ValueError,KeyError)):h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])
 assert not x['ack'].exists()

def test_recovery_is_never_a_final_current_handoff(recovered):
 x=recovered;ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])
 with pytest.raises(ValueError,match='historical only'):
  raw._owned_operation_chain([{'saved_update_recovery':ack,'receipt':x['failed_receipt']}],{'sha256':'0'*64},'fixture','fixture')

@pytest.mark.parametrize('later',['lfm','mellum'],indirect=True)
def test_both_port_recovery_ancestry_rejoins_a_real_later_full_save(recovered,tmp_path):
 import torch
 from test_native_later_runtime_handoff import _step,_actual_state
 x=recovered;protected=x['pin'].read_bytes();ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])
 parent=json.loads(x['owner_continuation'].read_text())['parent'];saved=copy.deepcopy(x['saved'])
 # This is a real next optimizer update on each native port, separate from the
 # harmless process-custody fixture. No production capacity claim is made.
 _step(x['model'],x['optimizer']);_actual_state(saved,x['model'],x['optimizer'],x['model_config'])
 saved['training']['global_step']+=1;saved['training']['options']['continuation']={'checkpoint_sha256':binding(x['pin'])['sha256']}
 N=tmp_path/'later-success';N.mkdir();out=N/'learner';out.mkdir();rb=saved['training']['options']['recipe_binding']
 if isinstance(rb,str):rb=json.loads(rb)
 child=[sys.executable,'-c','import sys;sys.stdin.buffer.read()','--out',str(out),'--stage-id','sft','--recipe-binding',json.dumps(rb),'--task-eval','']
 script="import subprocess,sys,json;p=subprocess.Popen(json.loads(sys.argv[1]),stdin=subprocess.PIPE);print('ready',flush=True);sys.stdin.buffer.read();p.communicate()"
 argv=[sys.executable,'-c',script,json.dumps(child)];p=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True)
 try:
  assert p.stdout.readline().strip()=='ready';owner=h.observe(p.pid);launch=N/'launch.json';write(launch,{'at':time.time(),'pid':p.pid,'argv':argv,'identity':owner});intent=h.capture_drain_intent(binding(launch),N/'intent.json')
  slot=out/'checkpoint.pt';pin=N/'full.pt';torch.save(saved,slot);shutil.copyfile(slot,pin)
  p.communicate();assert p.returncode==0;exit_path=N/'exit.json';write(exit_path,{'at':time.time(),'returncode':0})
  drained=h.complete_drain(intent,binding(exit_path),binding(pin),N/'handoff.json',original_slot=binding(slot))
 finally:
  if p.poll()is None:p.communicate()
 receipt=copy.deepcopy(json.loads(Path(x['failed_receipt']['path']).read_text()));receipt['checkpoint']=binding(slot);receipt['native_identity']['checkpoint_sha256']=binding(pin)['sha256'];receipt['continuation']['checkpoint_sha256']=binding(x['pin'])['sha256'];receipt['continuation_boundary'].update(new_optimizer_updates=1,checkpoint_sha256=binding(pin)['sha256']);rp=out/'native-training-receipt.json';write(rp,receipt)
 operations=[{'saved_update_recovery':ack,'receipt':x['failed_receipt']},{'handoff':drained,'receipt':binding(rp)}]
 chain=raw._owned_operation_chain(operations,{**parent,'bytes':Path(parent['path']).stat().st_size},receipt['model_id'],rb['lineage_sha256'])
 assert [r['new_optimizer_updates']for r in chain]==[3,1]
 assert chain[0]['outcome']['status']=='failed'and chain[0]['original_failed_result']==binding(x['failed_result'])
 assert chain[1]['checkpoint']==binding(pin)and chain[1]['parent']['sha256']==binding(x['pin'])['sha256']
 assert x['pin'].read_bytes()==protected and x['failed_result'].read_bytes()==x['failed_bytes']

def test_captured_actual_launch_survives_mutable_provider_bookkeeping_pruning(recovered):
 x=recovered;recovery=json.loads(x['recovery'].read_text());intent=json.loads(Path(recovery['captured_owner_intent']['path']).read_text());launch=json.loads(Path(intent['launch']['path']).read_text())
 Path(launch['original_provider_launch']['path']).unlink()
 ack=h.prepare_saved_update_acknowledgement(binding(x['recovery']),x['failed_receipt'],x['ack'])
 assert h.verify_saved_update_acknowledgement(ack,receipt=x['failed_receipt'])['outcome']['status']=='failed'
