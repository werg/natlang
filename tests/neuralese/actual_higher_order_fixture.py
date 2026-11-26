"""TEST ONLY actual Base capture -> canonical offline admission. No registry writes."""
import hashlib,json,sys,subprocess
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'scripts/neuralese_data')]
from live_teacher_sources import load_episode
root=Path(sys.argv[1]).resolve(); base=Path(sys.argv[2]).resolve()
capture=json.loads((root/'actual-capture.json').read_text())
assert capture['testOnly'] is True and capture['noModelInference'] is True
assert capture['result']==10

def artifact(name,value):
 path=root/name;path.write_text(json.dumps(value,ensure_ascii=False)+'\n')
 return {'path':name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
details=capture['details']
# Native Records getCall carries the original reduction trace as JSONL bytes.
for d in details:
 if isinstance(d['events'],str):d['events']=[json.loads(line) for line in d['events'].splitlines() if line.strip()]
roots=[d for d in details if d['record']['parent_call_id'] is None]
assert len(roots)==1 and len(details)==4
r=roots[0]['record'];goal=capture['goal']
entry={'program':'skills/hof/runUserGoal.nl','callable':'runUserGoal','arguments':{'goal':goal},'argumentJson':{'goal':capture['values'][r['inputs']['goal']['hash']]}}
interaction={'kind':'open-user-goal','goal':goal,'program':entry['program'],'entry':entry}
world={'instance':'test-only-local-scripted-gateway','run_id':'test-only-hof-run','context_id':'test-only-hof','case_id':'test-only-higher-order-scalar'}
checks={'nativeScalarTen':capture['result']==10,'fourActualCapturedCalls':len(details)==4}
if capture.get('effectful'):
 make=next(d['record'] for d in details if d['record']['definition']['name']=='make')
 consume=next(d['record'] for d in details if d['record']['definition']['name']=='consume')
 escaped=next(d for d in details if d['record']['parent_call_id']==consume['call_id'])
 effects=escaped['record']['effects']
 pairs=sorted([(json.loads(capture['values'][e['args']['hash']])[0],json.loads(capture['values'][e['result']['hash']])) for e in effects])
 for effect in effects:
  for ref in (effect['args'],effect['result']):
   encoded=capture['values'][ref['hash']].encode()
   assert ref['complete'] is True and len(encoded)==ref['bytes'] and hashlib.sha256(encoded).hexdigest()==ref['hash']
 requests=[e for e in escaped['events'] if e.get('kind')=='effect' and e.get('phase')=='requested' and e.get('service')=='docs']
 checks.update(nativeEffectsOnCurrentObserver=len(effects)==2 and all(e['service']=='docs' and e['method']=='describe' for e in effects),
               creatorAndConsumerUnmodified=not make['effects'] and not consume['effects'] and not r['effects'],
               exactNativePairs=pairs==[('3',{'id':'3','surface':'workspace','qualifiedName':'test-only-hof/3','title':'3'}),('4',{'id':'4','surface':'workspace','qualifiedName':'test-only-hof/4','title':'4'})],
               uniqueObservedRequestSequences=len(requests)==2 and len({e['seq'] for e in requests})==2)
 assert all(checks.values()),checks

proof={'goal':goal,'entry':entry,'contextId':world['context_id'],'caseId':world['case_id'],'passed':True,'checks':checks}
value=capture['values'][r['output']['hash']];assert json.loads(value)==capture['result']
verification={'schema':'natlang.live-task-verification/1','task_id':'test-only-higher-order-scalar','run_id':world['run_id'],'world':world,'status':'passed','checks':checks,'root_call_ids':[r['call_id']],'calls':[d['summary'] for d in details],'interaction':interaction,
 'validator':artifact('validator-source.json',{'testOnly':True,'source':(base/'packages/natlang/src/higherOrderCaptureIntegration.test.ts').read_text()}),
 'evidence':[{**artifact('actual-scalar-proof.json',{'proof':proof,'applicationReturn':{'root_call_id':r['call_id'],'valueJson':value,'selectedResult':capture['result']}}),'kind':'workspace'}]}
# Provenance identifies the clean committed capture fixture and its pinned vendor SDK.
sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=base,text=True).strip()
assert not subprocess.check_output(['git','status','--porcelain'],cwd=base,text=True).strip(), 'commit the actual capture fixture before final conversion'
spec={'schema':'natlang.live-teacher-task/1','id':'test-only-higher-order-scalar','family':'test-only-higher-order','source_namespace':'test-only:offline:higher-order','source_ids':['test-only-hof'],'source_groups':['test-only-hof'],'source_revisions':[sha],'split':'train','learner_kinds':['natlang-executor'],'code':{'git_commit':sha,'dirty':False},'world':world,'required_effect_kinds':['workspace'],'interaction':interaction,'calls':[artifact(f'call-{i}.json',d) for i,d in enumerate(details)],'verification':artifact('verification.json',verification),'testOnly':True,'registryRegistrationForbidden':True}
artifact('manifest.json',spec)
episode=load_episode(root/'manifest.json')
assert len(episode['turns'])==4
assert len(episode['call_tree'])==4
(root/'episode.json').write_text(json.dumps(episode)+'\n')
(root/'native-input.jsonl').write_text(json.dumps(episode['trajectory_ir'])+'\n')
print(json.dumps({'testOnly':True,'turns':len(episode['turns']),'calls':len(episode['call_tree']),'verification_mode':episode['verification_mode'],'task_completed':episode['trajectory_ir']['outcome']['task_completed']}))
# Optional real tokenizer-only rendering; never loads model weights or runs inference.
import os
if tokenizer:=os.environ.get('NATLANG_TEST_HOF_TOKENIZER'):
 import sft_sources as s
 import numpy as np
 renderer=s.TemplateRenderer(tokenizer)
 masks=[]
 for index,turn in enumerate(episode['turns']):
  row,context=s.call_row(renderer,'teacher',episode['key'],index,turn)
  weights=row.get('weight',np.ones(len(row['sample']),np.float32))
  assert len(row['prompt'])>0 and len(row['sample'])>context
  assert np.all(weights[:context]==0) and np.all(weights[context:]==1)
  masks.append({'invocation_id':turn['invocation_id'],'prompt_ids':row['prompt'].tolist(),'sample_ids':row['sample'].tolist(),'sample_weights':weights.tolist(),'zero_weight_lead':int(context),'prompt_tokens':len(row['prompt']),'trained_action_tokens':int(np.sum(weights))})
 (root/'training-mask-evidence.json').write_text(json.dumps({'testOnly':True,'modelInference':False,'renderer':renderer.identity,'rows':masks})+'\n')
 print(json.dumps({'tokenizer':renderer.identity,'masks':[{'invocation_id':m['invocation_id'],'prompt_tokens':m['prompt_tokens'],'zero_weight_lead':m['zero_weight_lead'],'trained_action_tokens':m['trained_action_tokens']} for m in masks]}))
