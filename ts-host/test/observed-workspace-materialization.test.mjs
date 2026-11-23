import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {mkdtemp, readFile, writeFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join, resolve} from 'node:path';
import {createHash} from 'node:crypto';
import {verifyObservedWorkspaceEvidence} from '../scripts/observed-workspace-evidence.mjs';
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
 if kind=='partial-error':
  child=json.loads((root/'child.json').read_text());child['record']['outcome']='error'
  spec['calls'][1]=artifact(root,'child.json',child)
  verification['training_scope']['invocations'][0]['ordinals']=[0]
 spec['verification']=artifact(root,'verification.json',verification);path.write_text(json.dumps(spec))
else:
 if kind=='caught-child':
  path,spec,_,verification=tree_fixture(root);detail=json.loads((root/'call.json').read_text())
  child=json.loads((root/'child.json').read_text());child['events']=[{'kind':'action','outcome':'completed','tool_call_id':'actual-tool-id'}]
  spec['calls'][1]=artifact(root,'child.json',child)
 else:path,spec,detail,verification=fixture(root)
 interaction=open_goal(detail,spec,verification)
 detail['events']=[{'kind':'action','outcome':'completed','tool_call_id':'actual-tool-id'}]
 if kind in ('typed','typed-masked'):
  texts={'approval':'{"revision":"approved","limit":1e-7}','draft':'"new draft"'}
  entry={'program':'skills/documents/reconcileDraft.nl','callable':'reconcileDraft','arguments':{name:json.loads(text) for name,text in texts.items()},'argumentJson':texts}
  interaction.update(program=entry['program'],entry=entry);verification['interaction']=copy.deepcopy(interaction)
  detail['record'].update(program_root='skills/documents',inputs={name:{'complete':True,'bytes':len(text.encode()),'hash':hashlib.sha256(text.encode()).hexdigest()} for name,text in texts.items()})
  detail['record']['definition'].update(name=entry['callable'],source='reconcileDraft.nl')
 if 'masked' in kind:
  terminal=copy.deepcopy(detail['record']['model_turns'][0]);args={'status':'success','value':{'wrong':True}}
  terminal['reply']['raw_calls']=[{'id':'terminal-id','type':'function','function':{'name':'return_result','arguments':json.dumps(args)}}]
  terminal['reply']['calls']=[['return_result',args]]
  terminal['reply']['raw_response']['vibestudio']['response']['content']=[{'type':'toolCall','id':'terminal-id','name':'return_result','arguments':args}]
  detail['record']['model_turns']=([detail['record']['model_turns'][0]] if kind!='masked-only' else [])+[terminal]
  detail['events'].append({'kind':'action','outcome':'completed','tool_call_id':'terminal-id'})
  verification['evidence']=[{**artifact(root,'optional-readback.json',{'proof':{'outcomeDimensions':{'returnedValue':{'required':False,'passed':False}}}}),'kind':'workspace'}]
 if kind=='caught-child':
  spec['calls'][0]=artifact(root,'call.json',detail);spec['verification']=artifact(root,'verification.json',verification);path.write_text(json.dumps(spec))
 else:rewrite(root,spec,detail,verification)
if kind=='completed':
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
 completed=artifact(root,'completed.json',outcome)
 artifact(root,'assessment.json',{'schema':'natlang.live-reassessment/1','task_id':'live-draft',
  'original_sha256':raw['sha256'],'rationale':outcome['rationale'],'completed_outcome':completed})
 recovery.held_closure=lambda _:{};recovery.held_hit=lambda *_:False
 result=recovery.recover(root/'original.json',root/'pins.json',root/'assessment.json',root/'catalog.jsonl',root/'held.json',root/'derivative')
 path=Path(result['manifest'])
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
for (const [kind, expected] of [['legacy',1],['typed',1],['partial',1],['partial-error',1],['masked',1],['typed-masked',1],['masked-historical',1],['masked-only',0],['caught-child',2],['completed',1]]) {
  test(`Python native IR -> disk verifier -> TS materializer -> CLI: ${kind}`, async () => {
    const {root,row} = await fixture(kind);
    try {
      await verifyObservedWorkspaceEvidence(row);
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
for (const kind of ['partial','partial-error','masked']) {
  test(`rejects forged decision admission from independently excluded ${kind} evidence`, async () => {
    const {root,row}=await fixture(kind);
    try {
      const excluded=kind==='masked'?row.trajectory.at(-1):row.trajectory.find(turn=>turn.invocation_id==='call-one'&&turn.phase==='action');
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
