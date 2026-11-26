import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {verifyObservedWorkspaceEvidence,verifyOriginalRejection,reviewedDecisionOrdinals} from '../scripts/observed-workspace-evidence.mjs';
import {materializeNativeRows} from '../dist/teacher/native-materializer.js';
const repo = resolve(import.meta.dirname, '../..');
const python = join(repo, '.venv-neuralese/bin/python');
const fixtureCode = String.raw`
import copy,hashlib,json,sys
from pathlib import Path
sys.path[:0]=[str(Path.cwd()/'tests/neuralese'),str(Path.cwd()/'scripts/neuralese_data')]
from test_live_teacher_sources import fixture,tree_fixture,partial_fixture,open_goal,artifact,rewrite
import live_teacher_sources as live
root=Path(sys.argv[1]);kind=sys.argv[2]
if kind.startswith('partial'):
 path,spec,verification=partial_fixture(root)
 if kind=='partial-assessor':verification['training_scope']['assessor']=artifact(root,'assessor-source.json',{'code':'native byte comparison'})
 if kind=='partial-error':
  child=json.loads((root/'child.json').read_text());child['record']['outcome']='error'
  spec['calls'][1]=artifact(root,'child.json',child)
  verification['training_scope']['invocations'][0]['ordinals']=[0]
 spec['verification']=artifact(root,'verification.json',verification);path.write_text(json.dumps(spec))
else:
 if kind in ('caught-child','incidental-nested','goal-void-error','goal-nonvoid-error'):
  path,spec,_,verification=tree_fixture(root);detail=json.loads((root/'call.json').read_text())
  child=json.loads((root/'child.json').read_text());child['events']=[{'kind':'action','outcome':'completed','tool_call_id':'actual-tool-id'}]
  spec['calls'][1]=artifact(root,'child.json',child)
 else:path,spec,detail,verification=fixture(root)
 interaction=open_goal(detail,spec,verification)
 detail['events']=[{'kind':'action','outcome':'completed','tool_call_id':'actual-tool-id'}]
 if kind in ('goal-root-error','goal-root-error-text','goal-void-error','goal-nonvoid-error'):
  detail['record']['outcome']='error'
  if kind in ('goal-void-error','goal-nonvoid-error'):
   entry={'program':'skills/open-user-goal/apply.ts','callable':'apply','resultType':('SourceAttributionReport' if kind=='goal-nonvoid-error' else 'void'),'arguments':{'goal':interaction['goal']},'argumentJson':{'goal':json.dumps(interaction['goal'])}}
   interaction.update(program=entry['program'],entry=entry);verification['interaction']=copy.deepcopy(interaction)
   detail['record'].update(program_root='skills/open-user-goal',executor={'kind':'crisp'},model_turns=[])
   detail['record']['definition'].update(name='apply',source='apply.ts',id='ts:apply.ts#default',returns=entry['resultType'])
   detail['summary']['definition_id']='ts:apply.ts#default'
   verification['calls'][0]=copy.deepcopy(detail['summary'])
  verification['world'].update(context_id='actual-context',case_id='actual-case')
  spec['world']=copy.deepcopy(verification['world'])
  proof={'goal':interaction['goal'],'contextId':'actual-context','caseId':'actual-case','passed':True,'checks':{'actualEffect':True}}
  if interaction.get('entry'):proof['entry']=copy.deepcopy(interaction['entry'])
  payload={'proof':proof}
  if kind=='goal-nonvoid-error':
   proof.update(semanticReviewRequired=True,outcomeDimensions={'effects':{'passed':True},'returnedValue':{'required':False,'passed':True}})
   payload['semanticReview']={'schema':'natlang.live-workflow-review/1','model':'TEST-only-local-review','input':{'actual':{'retainedText':'hello'},'expected':{'goal':interaction['goal']}},'verdict':{'accepted':True},'exchange':{'request':{},'reply':{}}}
  verification['evidence']=[{**artifact(root,'actual-goal.json',payload),'kind':'workspace'}]
  if kind=='goal-root-error-text':
   extra=copy.deepcopy(detail['record']['model_turns'][0])
   extra['reply'].update(text='A later unexecuted conclusion',raw_calls=[],calls=[])
   extra['reply']['raw_response']['vibestudio']['response']['content']=[{'type':'text','text':'A later unexecuted conclusion'}]
   detail['record']['model_turns'].append(extra)
 if kind in ('reasoned-readout','redacted-thinking','historical-redacted-thinking'):
  exchange=detail['record']['model_turns'][0]
  if kind=='reasoned-readout':
   reasoning=copy.deepcopy(exchange);reasoning['request']['decision_reasoning']={'schema':'natlang.decision-reasoning/1'}
   reasoning['reply'].update(text='Read the current facts before deciding.',raw_calls=[],calls=[])
   reasoning['reply']['raw_response']['vibestudio']['response']['content']=[{'type':'text','text':reasoning['reply']['text']}]
   detail['record']['model_turns'].append(reasoning)
  else:
   exchange['reply']['raw_response']['vibestudio']['response']['content'].insert(0,{'type':'thinking','thinking':'[redacted]','redacted':True})
   if kind=='historical-redacted-thinking':exchange['reply']['reasoning']='[redacted]'
 if kind in ('typed','typed-masked'):
  texts={'approval':'{"revision":"approved","limit":1e-7}','draft':'"new draft"'}
  entry={'program':'skills/documents/reconcileDraft.nl','callable':'reconcileDraft','arguments':{name:json.loads(text) for name,text in texts.items()},'argumentJson':texts}
  interaction.update(program=entry['program'],entry=entry);verification['interaction']=copy.deepcopy(interaction)
  detail['record'].update(program_root='skills/documents',inputs={name:{'complete':True,'bytes':len(text.encode()),'hash':hashlib.sha256(text.encode()).hexdigest()} for name,text in texts.items()})
  detail['record']['definition'].update(name=entry['callable'],source='reconcileDraft.nl')
 if 'masked' in kind or kind.startswith('incidental'):
  terminal=copy.deepcopy(detail['record']['model_turns'][0]);args={'status':'success','value':{'wrong':True}}
  terminal['reply']['raw_calls']=[{'id':'terminal-id','type':'function','function':{'name':'return_result','arguments':json.dumps(args)}}]
  terminal['reply']['calls']=[['return_result',args]]
  terminal['reply']['raw_response']['vibestudio']['response']['content']=[{'type':'toolCall','id':'terminal-id','name':'return_result','arguments':args}]
  detail['record']['model_turns']=([detail['record']['model_turns'][0]] if kind!='masked-only' else [])+[terminal]
  detail['events'].append({'kind':'action','outcome':'completed','tool_call_id':'terminal-id'})
  verification['evidence']=[{**artifact(root,'optional-readback.json',{'proof':{'outcomeDimensions':{'returnedValue':{'required':False,'passed':False}}}}),'kind':'workspace'}]
 if kind.startswith('incidental'):
  detail['record']['definition']['source']='runUserGoal.nl'
  detail['record']['output']={'complete':True,'hash':hashlib.sha256(b'{"wrong":true}').hexdigest(),'bytes':14}
  text='{"wrong":true}'
  selector={'definition':copy.deepcopy(detail['record']['definition']),'valueJson':text,'sha256':hashlib.sha256(text.encode()).hexdigest()}
  if kind=='incidental-forged-hash':selector['sha256']='0'*64
  if kind=='incidental-foreign-call':selector['definition']['source']='foreign.nl'
  proof={'actualResult':{'wrong':True},'outcomeDimensions':{'incidentalValues':{'passed':False,'incorrectReturns':[selector,selector,{**selector,'required':False,'definition':{'name':'absentHelper','source':'absent.nl'}}]}}}
  if kind=='incidental-format-only':
   selector.update(required=False,trainingEvidence='format-only',contribution={'schema':'natlang.application-return-contribution/1','kind':'root','root_call_id':detail['record']['call_id'],'producer_call_id':detail['record']['call_id'],'root_value_json':text,'root_sha256':selector['sha256'],'root_path':[],'producer_path':[],'checks':['optionalFindingFormat'],'lineage':[detail['record']['call_id']]})
   proof={'actualResult':{'wrong':True},'outcomeContract':{'returnedValue':{'required':False,'trainingEvidence':'format-only'}},'outcomeDimensions':{'returnedValue':{'required':False,'passed':True,'checks':{'optionalFindingFormat':True},'trainingEvidence':'format-only','unreviewedReturns':[selector]}}}
  if kind=='incidental-nested':
   child['record']['definition']['source']='owned/recoverAndWrite.nl'
   child['record']['output']=copy.deepcopy(detail['record']['output'])
   child['record']['model_turns']=[copy.deepcopy(terminal)]
   child['events']=[{'kind':'action','outcome':'completed','tool_call_id':'terminal-id'}]
   proof['outcomeDimensions']['incidentalValues']['incorrectReturns'].append({**selector,'definition':copy.deepcopy(child['record']['definition'])})
   spec['calls'][1]=artifact(root,'child.json',child)
  verification['evidence']=[{**artifact(root,'incidental.json',{'proof':proof}),'kind':'workspace'}]
 if kind=='goal-reference-review':
  from test_review_evidence_roles import review_fixture
  review,source_input=review_fixture()
  source_input['actual']['goal']=source_input['expected']['goal']=interaction['goal']
  source_input['actual']['requestedReturn']=None
  source_input['expected']['requestedReturnRequired']=False
  review['input']=copy.deepcopy(source_input)
  review['exchange']['request']['messages'][1]['content']=json.dumps({'rubric':source_input['rubric'],'reference':source_input['expected'],'candidate':source_input['actual']})
  verification['world'].update(context_id='actual-context',case_id='actual-case');spec['world']=copy.deepcopy(verification['world'])
  proof={'goal':interaction['goal'],'contextId':'actual-context','caseId':'actual-case','passed':True,'checks':{'actualEffect':True},
   'semanticReviewRequired':True,'semanticReviewInput':source_input,'outcomeDimensions':{'effects':{'passed':True},'returnedValue':{'required':False,'passed':True}}}
  verification['evidence']=[{**artifact(root,'reference-review.json',{'proof':proof,'semanticReview':review}),'kind':'workspace'}]
 if kind.startswith('decision-review'):
  binding=artifact(root,'call.json',detail)
  selector={'call_id':detail['record']['call_id'],'ordinal':0,'definition':copy.deepcopy(detail['record']['definition']),'call_sha256':binding['sha256'],'reason':'Source-bound semantic audit of this exact captured decision.'}
  if kind=='decision-review-hash':selector['call_sha256']='0'*64
  if kind=='decision-review-score':selector['ordinal']=1
  verification['evidence']=[{**artifact(root,'decision-review.json',{'proof':{'decisionReview':{'schema':'natlang.live-decision-review/1','excluded':[selector]}}}),'kind':'workspace'}]
 if kind=='native-bridge':
  selected={'question':'Actual source-backed brief','statements':[]}
  actual={'results':[{'kind':'domain','report':selected}],'remaining':[]}
  text=json.dumps(actual,separators=(',',':'))
  detail['record']['output']={'complete':True,'bytes':len(text.encode()),'hash':hashlib.sha256(text.encode()).hexdigest()}
  verification['evidence']=[{**artifact(root,'native-return.json',{'applicationReturn':{'root_call_id':'call-one','valueJson':text,'selectedResult':selected}}),'kind':'workspace'}]
 if kind in ('caught-child','incidental-nested','goal-void-error','goal-nonvoid-error'):
  spec['calls'][0]=artifact(root,'call.json',detail);spec['verification']=artifact(root,'verification.json',verification);path.write_text(json.dumps(spec))
 else:rewrite(root,spec,detail,verification)
if kind in ('completed','completed-roles','completed-review','completed-promoted'):
 import recheck_live_trajectories as recovery
 proof={'goal':interaction['goal'],'checks':{'native_state':True,'old_oracle':False}}
 packet={'test':{'name':'draft'},'result':{'passed':False},'execution':{'diagnostics':{
  'openGoalVerification':proof,'natlangTeacher':{'programPath':interaction['program'],'calls':[detail],
  'rootCallIds':['call-one'],'report':{'value':'actual'}}}}}
 raw=artifact(root,'original.json',packet)
 artifact(root,'pins.json',{'repositories':{'natlang':{'git_commit':'a'*40,'dirty':False}},'instance':'owned-instance','run_id':'run'})
 artifact(root,'catalog.jsonl',{'scenario':'draft','taskId':'live-draft','family':'draft','split':'train','sourceGroups':['drafts']})
 outcome={'schema':'natlang.live-outcome-reassessment/1','task_id':'live-draft','original_sha256':raw['sha256'],
  'goal':interaction['goal'],'entry':None,'actualResult':{'value':'actual'},'passed':True,
  'checks':{'native_state':True,'old_oracle':True},'rationale':'Independently recomputed native outcome.',
  'checker_sources':[artifact(root,'checker-source.json',{'code':'original checker'})],
  'evidence':[artifact(root,'native-state.json',{'state':'native'})]}
 if kind=='completed-roles':
  contract={'effects':{'checks':['native_state'],'rationale':'Native state.'},'returnedValue':{'checks':[],'required':False,'rationale':'No answer requested.'},'incidentalValues':{'checks':['old_oracle'],'rationale':'An old convention, with no incorrect emitted value.'}}
  outcome.update(checks={'native_state':True},outcomeContract=contract,outcomeDimensions={'effects':{'passed':True,'checks':{'native_state':True}},'returnedValue':{'required':False,'passed':True,'checks':{}},'incidentalValues':{'passed':False,'checks':{'old_oracle':False},'incorrectReturns':[]}})
  outcome['checker_sources'].append({**artifact(root,'contract.json',contract),'role':'outcome-contract'})
 if kind in ('completed-review','completed-promoted'):
  native_input={'actual':{'documents':[{'path':'note.md','text':'Checked draft'}]},'expected':{'goal':interaction['goal']},'rubric':'Check content'}
  packet['result']['passed']=True
  packet['execution']['diagnostics']['openGoalVerification'].update(preparedContentProject={'artifact':{'semanticReviewInput':native_input}},checks={'native_state':True,'old_oracle':True})
  raw=artifact(root,'original.json',packet);outcome['original_sha256']=raw['sha256']
  review={'schema':'natlang.live-workflow-review/1','input':native_input,'verdict':{'accepted':False,'verdict':'Content is correct; baseline handle mistaken for a new commit.'},'exchange':{'request':{},'reply':{}}}
  outcome['evidence'].append({**artifact(root,'original-review.json',review),'role':'original-rejected-review'})
 if kind=='completed-promoted':
  old_contract={'effects':{'checks':['native_state'],'rationale':'Native state.'},'returnedValue':{'checks':['answer'],'required':False,'rationale':'Originally optional answer.'},'incidentalValues':{'checks':['receipt'],'rationale':'Incidental copy.'}}
  old_dimensions={'effects':{'passed':True,'checks':{'native_state':True}},'returnedValue':{'required':False,'passed':True,'checks':{'answer':True}},'incidentalValues':{'passed':True,'checks':{'receipt':True},'incorrectReturns':[]}}
  packet['execution']['diagnostics']['openGoalVerification'].update(checks={'native_state':True},observedChecks={'native_state':True,'answer':True,'receipt':True},outcomeContract=old_contract,outcomeDimensions=old_dimensions)
  raw=artifact(root,'original.json',packet);outcome['original_sha256']=raw['sha256']
  contract=copy.deepcopy(old_contract);contract['returnedValue']['required']=True
  dimensions=copy.deepcopy(old_dimensions);dimensions['returnedValue']['required']=True
  outcome.update(checks={'native_state':True,'answer':True},observedChecks={'native_state':True,'answer':True,'receipt':True},outcomeContract=contract,outcomeDimensions=dimensions)
  outcome['checker_sources'].append({**artifact(root,'promoted-contract.json',contract),'role':'outcome-contract'})
 completed=artifact(root,'completed.json',outcome)
 artifact(root,'assessment.json',{'schema':'natlang.live-reassessment/1','task_id':'live-draft',
  'original_sha256':raw['sha256'],'rationale':outcome['rationale'],'completed_outcome':completed})
 recovery.held_closure=lambda _:{};recovery.held_hit=lambda *_:False
 result=recovery.recover(root/'original.json',root/'pins.json',root/'assessment.json',root/'catalog.jsonl',root/'held.json',root/'derivative')
 path=Path(result['manifest'])
if kind.startswith('partial'):
 def test_only_uncertified_scope(verification,records,root,details=None):
  return {i['call_id']:set(i['ordinals']) if i.get('ordinals') is not None else None for i in verification['training_scope']['invocations']}
 live.verified_training_scope=test_only_uncertified_scope
row=live.load_episode(path)['trajectory_ir']
if kind=='masked-historical':row['outcome'].pop('excluded_return_decisions',None)
(root/'input.jsonl').write_text(json.dumps(row)+'\n')
`;
async function fixture(kind) {
  const root = await mkdtemp(join(tmpdir(), 'natlang-observed-materialize-'));
  const result = spawnSync(python, ['-c', fixtureCode, root, kind], {cwd: repo, encoding: 'utf8',
    env: {...process.env, PYTHONPATH: `${repo}/training/neuralese:${repo}`, OMP_NUM_THREADS: '1', MKL_NUM_THREADS: '1'}});
  assert.equal(result.status, 0, result.stderr);
  return {root, row: JSON.parse(await readFile(join(root, 'input.jsonl'), 'utf8'))};
}
for (const [kind, expected] of [['incidental-format-only',1],['goal-reference-review',1],['goal-nonvoid-error',1],['goal-void-error',1],['goal-root-error',1],['goal-root-error-text',1],['reasoned-readout',1],['redacted-thinking',1],['historical-redacted-thinking',1],['decision-review',0],['legacy',1],['native-bridge',1],['typed',1],['masked',1],['typed-masked',1],['masked-historical',1],['masked-only',0],['caught-child',2],['completed',1],['completed-roles',1],['completed-review',1],['completed-promoted',1],['incidental',1],['incidental-nested',1]]) {
  test(`Python native IR -> disk verifier -> TS materializer -> CLI: ${kind}`, async () => {
    const {root,row} = await fixture(kind);
    try {
      await verifyObservedWorkspaceEvidence(row);
      if(kind==='incidental-format-only'){
        assert.equal(row.trajectory[0].training_admission.approved,true);
        assert.equal(row.trajectory.at(-1).training_admission.approved,false);
        assert.match(row.trajectory.at(-1).training_admission.reason,/lacks independent correctness/);
        const payload=JSON.parse(await readFile(join(root,'incidental.json'),'utf8'));
        assert.equal(payload.proof.outcomeDimensions.returnedValue.passed,true);
        assert.equal(payload.proof.outcomeDimensions.returnedValue.checks.optionalFindingFormat,true);
      }
      assert.equal(materializeNativeRows([row]).turns.length, expected);
      const run = spawnSync(process.execPath,[join(repo,'ts-host/scripts/materialize-native-teacher.mjs'),join(root,'input.jsonl'),join(root,'output.jsonl')],{encoding:'utf8'});
      assert.equal(run.status,0,run.stderr);
      const rows=(await readFile(join(root,'output.jsonl'),'utf8')).split('\n').filter(Boolean).map(JSON.parse);
      assert.equal(rows.length,expected);
      assert.ok(rows.every(row=>!JSON.stringify(row.target).includes('terminal-id')));
      if(kind.startsWith('partial')) {
        assert.ok(rows.every(row=>row.outcome.status==='partial' && row.outcome.task_completed===false));
        assert.equal(JSON.parse(run.stdout).partial_rows,1);
        assert.equal(JSON.parse(run.stdout).accepted_rows,0);
      }
    } finally {await rm(root,{recursive:true,force:true});}
  });
}
for (const kind of ['masked','incidental','incidental-format-only']) {
  test(`rejects forged decision admission from independently excluded ${kind} evidence`, async () => {
    const {root,row}=await fixture(kind);
    try {
      const excluded=row.trajectory.at(-1);
      excluded.training_admission={approved:true};
      await assert.rejects(verifyObservedWorkspaceEvidence(row),/observed decision differs/);
      await writeFile(join(root,'forged.jsonl'),JSON.stringify(row)+'\n');
      const run=spawnSync(process.execPath,[join(repo,'ts-host/scripts/materialize-native-teacher.mjs'),join(root,'forged.jsonl'),join(root,'forged-output.jsonl')],{encoding:'utf8'});
      assert.notEqual(run.status,0);
      await assert.rejects(readFile(join(root,'forged-output.jsonl')), {code:'ENOENT'});
      assert.throws(()=>materializeNativeRows([row]),/outside independent admission/);
      if(kind==='masked') {
        delete row.outcome.excluded_return_decisions;
        await assert.rejects(verifyObservedWorkspaceEvidence(row));
      }
    } finally {await rm(root,{recursive:true,force:true});}
  });
}
for (const forgery of ['input','path','callable']) test(`rejects self-consistent forged application ${forgery}`,async()=>{
  const {root,row}=await fixture('typed');
  try {
    const manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
    const receipt=JSON.parse(await readFile(join(root,'verification.json'),'utf8'));
    const entry=row.task.interaction.entry;
    if(forgery==='input') {entry.arguments.draft='forged draft';entry.argumentJson.draft='"forged draft"';}
    if(forgery==='path') {entry.program='skills/documents/unrelated.nl';row.task.interaction.program=entry.program;}
    if(forgery==='callable') entry.callable='unrelatedFunction';
    manifest.interaction=structuredClone(row.task.interaction);receipt.interaction=structuredClone(row.task.interaction);
    const bind=async(name,value)=>{const bytes=Buffer.from(JSON.stringify(value));await writeFile(join(root,name),bytes);return {path:name,sha256:createHash('sha256').update(bytes).digest('hex')};};
    manifest.verification=await bind('verification.json',receipt);row.provenance.verification=manifest.verification;
    const source=await bind('manifest.json',manifest);row.provenance.source_manifest={...source,path:join(root,source.path)};
    await assert.rejects(verifyObservedWorkspaceEvidence(row),/input differs|actual source|actual application entry/);
  } finally {await rm(root,{recursive:true,force:true});}
});

test('actual recovered archived Luna capture materializes from its original bound evidence',
  {skip: !process.env.NATLANG_RECOVERED_OBSERVED_MANIFEST},async()=>{
    const root=await mkdtemp(join(tmpdir(),'natlang-real-observed-'));
    try {
      const code='import json,sys;from pathlib import Path;from live_teacher_sources import load_episode;Path(sys.argv[2]).write_text(json.dumps(load_episode(sys.argv[1])["trajectory_ir"])+"\\n")';
      const input=join(root,'real-input.jsonl');
      const result=spawnSync(python,['-c',code,process.env.NATLANG_RECOVERED_OBSERVED_MANIFEST,input],{cwd:repo,encoding:'utf8',
        env:{...process.env,PYTHONPATH:`${repo}/scripts/neuralese_data:${repo}/training/neuralese:${repo}`}});
      assert.equal(result.status,0,result.stderr);
      const row=JSON.parse(await readFile(input,'utf8'));
      await verifyObservedWorkspaceEvidence(row);
      const expected=row.trajectory.filter(turn=>turn.training_admission.approved).length;
      const run=spawnSync(process.execPath,[join(repo,'ts-host/scripts/materialize-native-teacher.mjs'),input,join(root,'real-output.jsonl')],{encoding:'utf8'});
      assert.equal(run.status,0,run.stderr);
      assert.equal(JSON.parse(run.stdout).training_decisions,expected);
      assert.equal(row.outcome.task_completed,process.env.NATLANG_RECOVERED_COMPLETED==='1');
    } finally {await rm(root,{recursive:true,force:true});}
  });

for(const mutation of ['actual-result','native-evidence','completion-status'])test(`completed reassessment rejects ${mutation} forgery`,async()=>{
 const {root,row}=await fixture('completed');
 try {
  const manifest=JSON.parse(await readFile(row.provenance.source_manifest.path,'utf8'));
  const base=join(root,'derivative');
  if(mutation==='completion-status')row.outcome.reassessment=null;
  else {
   const receipt=JSON.parse(await readFile(join(base,manifest.verification.path),'utf8'));
   const workspace=receipt.evidence.find(item=>item.kind==='workspace');
   const proof=JSON.parse(await readFile(join(base,workspace.path),'utf8')).proof;
   if(mutation==='native-evidence')await writeFile(join(base,proof.evidence[0].path),'forged witness');
   else {
    proof.actualResult={value:'invented'};const bytes=JSON.stringify({proof});await writeFile(join(base,workspace.path),bytes);
    // Updating a local nested binding cannot rewrite the attested verification artifact.
   }
  }
  await assert.rejects(verifyObservedWorkspaceEvidence(row));
 }finally{await rm(root,{recursive:true,force:true});}
});

for(const kind of ['partial','partial-error','partial-assessor'])test(`rejects uncertified legacy private decisions: ${kind}`,async()=>{
 const {root,row}=await fixture(kind);
 try{
  await assert.rejects(verifyObservedWorkspaceEvidence(row),/partial training scope/);
  assert.throws(()=>materializeNativeRows([row]),/independent training scope/);
  const run=spawnSync(process.execPath,[join(repo,'ts-host/scripts/materialize-native-teacher.mjs'),join(root,'input.jsonl'),join(root,'output.jsonl')],{encoding:'utf8'});
  assert.notEqual(run.status,0);
  await assert.rejects(readFile(join(root,'output.jsonl')),{code:'ENOENT'});
 }finally{await rm(root,{recursive:true,force:true})}
});

for(const kind of ['incidental-forged-hash','incidental-foreign-call','decision-review-hash','decision-review-score'])test(`rejects ${kind} output exclusion`,async()=>{
 await assert.rejects(fixture(kind),/incorrect return|decision review/);
});
for(const mutation of ['hash','foreign-source','value'])test(`independent disk verifier rejects incidental ${mutation} forgery`,async()=>{
 const {root,row}=await fixture('incidental');
 try {
  const manifest=JSON.parse(await readFile(join(root,'manifest.json'),'utf8'));
  const receipt=JSON.parse(await readFile(join(root,'verification.json'),'utf8'));
  const proof=JSON.parse(await readFile(join(root,'incidental.json'),'utf8'));
  const selector=proof.proof.outcomeDimensions.incidentalValues.incorrectReturns[0];
  if(mutation==='hash')selector.sha256='0'.repeat(64);
  if(mutation==='foreign-source')selector.definition.source='foreign.nl';
  if(mutation==='value'){
   selector.valueJson='{"invented":true}';
   selector.sha256=createHash('sha256').update(selector.valueJson).digest('hex');
  }
  const bind=async(name,value)=>{const bytes=Buffer.from(JSON.stringify(value));await writeFile(join(root,name),bytes);return {path:name,sha256:createHash('sha256').update(bytes).digest('hex')};};
  receipt.evidence[0]={...await bind('incidental.json',proof),kind:'workspace'};
  manifest.verification=await bind('verification.json',receipt);row.provenance.verification=manifest.verification;
  const source=await bind('manifest.json',manifest);row.provenance.source_manifest={...source,path:join(root,source.path)};
  await assert.rejects(verifyObservedWorkspaceEvidence(row),/incorrect return/);
 } finally {await rm(root,{recursive:true,force:true});}
});

import {validateReassessmentCheckRoles} from '../scripts/observed-workspace-evidence.mjs';
test('source-bound check roles preserve omitted false observations and reject incomplete or forged roles', async()=>{
 const root=await mkdtemp(join(tmpdir(),'natlang-reassessment-roles-'));
 try{
  const contract={effects:{checks:['native'],rationale:'Native state.'},returnedValue:{checks:['summary'],required:false,rationale:'No summary requested.'}};
  const raw=JSON.stringify(contract);await writeFile(join(root,'contract.json'),raw);
  const proof={checks:{native:true},outcomeContract:contract,checker_sources:[{path:'contract.json',sha256:createHash('sha256').update(raw).digest('hex'),role:'outcome-contract'}],
   outcomeDimensions:{effects:{passed:true,checks:{native:true}},returnedValue:{required:false,passed:false,checks:{summary:false}}}};
  const original={native:true,summary:false};await validateReassessmentCheckRoles(proof,original,root);
  for(const mutate of [p=>delete p.outcomeContract,p=>p.checker_sources=[],
   p=>p.outcomeDimensions.returnedValue.checks.summary=true,p=>p.outcomeDimensions.returnedValue.passed=true,
   p=>p.outcomeContract.returnedValue.required=true,p=>p.checks.unknown=true]){
   const candidate=structuredClone(proof);mutate(candidate);await assert.rejects(validateReassessmentCheckRoles(candidate,original,root));
  }
 }finally{await rm(root,{recursive:true,force:true});}
});

for (const mutation of ['root','hash','selected']) test(`native bridge disk verification rejects ${mutation} forgery`, async()=>{
  const {root,row}=await fixture('native-bridge');
  try {
    const manifestBinding=row.provenance.source_manifest;
    // Bind the changed evidence consistently; the independent root must still disagree.
    const manifest=JSON.parse(await readFile(manifestBinding.path,'utf8'));
    const receiptPath=resolve(root,manifest.verification.path);
    const receipt=JSON.parse(await readFile(receiptPath,'utf8'));
    const evidence=receipt.evidence.find(item=>item.kind==='workspace');
    const evidencePath=resolve(root,evidence.path);
    const payload=JSON.parse(await readFile(evidencePath,'utf8'));
    if(mutation==='root')payload.applicationReturn.root_call_id='foreign';
    if(mutation==='hash')payload.applicationReturn.valueJson='{}';
    if(mutation==='selected')payload.applicationReturn.selectedResult={question:'forged'};
    await writeFile(evidencePath,JSON.stringify(payload));
    evidence.sha256=createHash('sha256').update(await readFile(evidencePath)).digest('hex');
    await writeFile(receiptPath,JSON.stringify(receipt));
    manifest.verification.sha256=createHash('sha256').update(await readFile(receiptPath)).digest('hex');
    await writeFile(manifestBinding.path,JSON.stringify(manifest));
    manifestBinding.sha256=createHash('sha256').update(await readFile(manifestBinding.path)).digest('hex');
    row.provenance.verification=manifest.verification;
    await assert.rejects(verifyObservedWorkspaceEvidence(row),/application return|domain return/u);
  } finally {await rm(root,{recursive:true,force:true});}
});

test('disk verifier distinguishes rejected original review from original native success',async()=>{
 const root=await mkdtemp(join(tmpdir(),'rejected-review-boundary-'));
 try{
  const input={actual:{documents:[{text:'Actual draft'}]},expected:{goal:'Prepare draft'},rubric:'Check content'};
  const packet={result:{passed:true},execution:{diagnostics:{openGoalVerification:{semanticReviewInput:input}}}};
  const review={schema:'natlang.live-workflow-review/1',input,verdict:{accepted:false},exchange:{request:{},reply:{}}};
  const file=join(root,'review.json');
  async function binding(){await writeFile(file,JSON.stringify(review));return {evidence:[{role:'original-rejected-review',path:'review.json',sha256:createHash('sha256').update(await readFile(file)).digest('hex')}]};}
  assert.equal(await verifyOriginalRejection(packet,await binding(),root),true);
  review.verdict.accepted=true;assert.equal(await verifyOriginalRejection(packet,await binding(),root),false);
  review.verdict.accepted=false;review.input={...input,expected:{goal:'Other goal'}};assert.equal(await verifyOriginalRejection(packet,await binding(),root),false);
  review.input=input;review.exchange={};assert.equal(await verifyOriginalRejection(packet,await binding(),root),false);
  review.exchange={request:{}};assert.equal(await verifyOriginalRejection(packet,await binding(),root),false);
  assert.equal(await verifyOriginalRejection(packet,{evidence:[]},root),false);
 }finally{await rm(root,{recursive:true,force:true});}
});


test('independent decision-quality holds bind exact captured bytes and cannot release or redirect targets',()=>{
 const record={call_id:'actual',definition:{name:'judge',source:'judge.nl'},model_turns:[{kind:'generation',reply:{calls:[]}}]};
 const artifact={sha256:'a'.repeat(64)};
 const selector={call_id:'actual',definition:record.definition,ordinal:0,call_sha256:artifact.sha256,reason:'Unsupported actual decision'};
 assert.deepEqual([...reviewedDecisionOrdinals([selector],record,artifact,new Set())],[0]);
 for(const changed of [{...selector,call_sha256:'b'.repeat(64)},{...selector,ordinal:2},{...selector,definition:{name:'foreign'}},{...selector,reason:''}])
  assert.throws(()=>reviewedDecisionOrdinals([changed],record,artifact,new Set()),/decision review/);
 assert.throws(()=>reviewedDecisionOrdinals([selector,selector],record,artifact,new Set()),/decision review/);
});

test('goal completion does not supervise unexecuted prose after an executor error',async()=>{
  const {root,row}=await fixture('goal-root-error-text');
  try {
    const last=row.trajectory.at(-1);
    assert.equal(last.training_admission.approved,false);
    last.training_admission={approved:true};
    await assert.rejects(verifyObservedWorkspaceEvidence(row),/observed decision differs/);
    assert.throws(()=>materializeNativeRows([row]),/ineligible for supervision/);
  } finally {await rm(root,{recursive:true,force:true});}
});
