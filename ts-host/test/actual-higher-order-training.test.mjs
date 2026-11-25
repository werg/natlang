/** Opt-in actual Base capture: scripted TEST-only local gateway, never registry data. */
import test from 'node:test';
import assert from 'node:assert/strict';
import {spawnSync} from 'node:child_process';
import {readFile, writeFile, mkdtemp, copyFile, rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,resolve} from 'node:path';
import {verifyObservedWorkspaceEvidence} from '../scripts/observed-workspace-evidence.mjs';
import {materializeNativeRows} from '../dist/teacher/native-materializer.js';
const repo=resolve(import.meta.dirname,'../..');
const capture=process.env.NATLANG_TEST_HOF_CAPTURE;
const base=process.env.NATLANG_TEST_HOF_BASE;
test('actual escaped callable capture retains every independently verified training decision',
 {skip:!capture||!base},async()=>{
 const root=await mkdtemp(join(tmpdir(),'natlang-actual-hof-'));
 try{
  await copyFile(capture,join(root,'actual-capture.json'));
  const converted=spawnSync(join(repo,'.venv-neuralese/bin/python'),[join(repo,'tests/neuralese/actual_higher_order_fixture.py'),root,base],{cwd:repo,encoding:'utf8',env:{...process.env,PYTHONPATH:`${repo}/training/neuralese:${repo}`,OMP_NUM_THREADS:'1',MKL_NUM_THREADS:'1'}});
  assert.equal(converted.status,0,converted.stderr);
  const episode=JSON.parse(await readFile(join(root,'episode.json'),'utf8'));
  const native=episode.trajectory_ir;
  await verifyObservedWorkspaceEvidence(native);
  const rows=materializeNativeRows([native]).turns;
  assert.equal(rows.length,4);
  assert.equal(native.outcome.task_completed,true);
  const raw=JSON.parse(await readFile(capture,'utf8'));
  assert.equal(raw.testOnly,true);assert.equal(raw.noModelInference,true);
  const records=new Map(raw.details.map(d=>[d.record.call_id,d.record]));
  const make=[...records.values()].find(r=>r.definition.name==='make');
  const consume=[...records.values()].find(r=>r.definition.name==='consume');
  const escaped=[...records.values()].find(r=>r.parent_call_id===consume.call_id);
  assert.equal(make.output.complete,false);assert.equal(make.output.reason,'nonportable');
  assert.equal(consume.inputs.fn.complete,false);assert.equal(consume.inputs.fn.reason,'nonportable');
  assert.equal(JSON.parse(raw.values[escaped.captures.delta.hash]),7);
  assert.match(JSON.stringify(consume.model_turns[0].request.messages),/live value, use it in eval/);
  assert.match(JSON.stringify(escaped.model_turns[0].request.messages),/const delta: number = 7/);
  assert.equal(escaped.parent_call_id,consume.call_id);
  if(raw.effectful){
   assert.equal(escaped.effects.length,2);assert.deepEqual(make.effects,[]);assert.deepEqual(consume.effects,[]);
   const pairs=escaped.effects.map(e=>[JSON.parse(raw.values[e.args.hash])[0],JSON.parse(raw.values[e.result.hash])]).sort((a,b)=>a[0]-b[0]);
   assert.deepEqual(pairs, [['3',{id:'3',surface:'workspace',qualifiedName:'test-only-hof/3',title:'3'}],['4',{id:'4',surface:'workspace',qualifiedName:'test-only-hof/4',title:'4'}]]);
   for(const effect of escaped.effects)for(const ref of [effect.args,effect.result]){
    const text=raw.values[ref.hash];assert.equal(Buffer.byteLength(text),ref.bytes);
    const{createHash}=await import('node:crypto');assert.equal(createHash('sha256').update(text).digest('hex'),ref.hash);
   }
   const decision=native.trajectory.find(d=>d.invocation_id===escaped.call_id);
   const artifact=join(root,decision.live_exchange.call_artifact.path);
   const originalBytes=await readFile(artifact,'utf8');
   const forged=JSON.parse(originalBytes);forged.record.effects[0].result.hash='0'.repeat(64);
   try{await writeFile(artifact,JSON.stringify(forged));await assert.rejects(verifyObservedWorkspaceEvidence(native));}
   finally{await writeFile(artifact,originalBytes);}
   await verifyObservedWorkspaceEvidence(native);
  }

  assert.notEqual(escaped.parent_call_id,make.call_id);
  assert.ok(Date.parse(make.ended_at)<=Date.parse(consume.started_at));
  if(process.env.NATLANG_TEST_HOF_TOKENIZER){
   const evidence=JSON.parse(await readFile(join(root,'training-mask-evidence.json'),'utf8'));
   assert.equal(evidence.renderer.model_type,'lfm3');
   assert.equal(evidence.rows.length,4);
   for(const row of evidence.rows){
    assert.ok(row.prompt_tokens>0&&row.trained_action_tokens>0);
    assert.ok(row.sample_weights.slice(0,row.zero_weight_lead).every(w=>w===0));
    assert.ok(row.sample_weights.slice(row.zero_weight_lead).every(w=>w===1));
   }
  }
  for(const row of rows){
   const id=row.source_ref.invocation_id;
   const original=records.get(id).model_turns[0];
   assert.deepEqual(row.messages,original.request.messages);
   assert.deepEqual(row.tools,original.request.tools);
   assert.deepEqual(row.target.tool_calls,original.reply.raw_calls);
   assert.equal(row.source_ref.parent_invocation_id,records.get(id).parent_call_id);
   assert.equal(row.training_admission.approved,true);
   assert.equal(row.decision.action_receipts.length,1);
  }
  const forged=structuredClone(native);
  forged.task.interaction.entry.argumentJson.goal='"forged input"';
  await assert.rejects(verifyObservedWorkspaceEvidence(forged));
  const wrongParent=structuredClone(native);
  wrongParent.trajectory.find(d=>d.invocation_id===escaped.call_id).parent_invocation_id=make.call_id;
  await assert.rejects(verifyObservedWorkspaceEvidence(wrongParent));
 }finally{await rm(root,{recursive:true,force:true});}
});
