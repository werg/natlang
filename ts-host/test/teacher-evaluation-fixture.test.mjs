import test from 'node:test';
import { createServer } from 'node:http';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import assert from 'node:assert/strict';
import { fingerprint } from '../dist/adaptation/identity.js';
import { executeProgram, validateFocusedRecord, collectBatch, nativeJobRunner, defaultToolSurfaceHash } from '../dist/teacher/collector.js';

function record() {
  const selected = {id:'selected',group:'fixture-world',split:'train',input:{n:3},expected:4};
  return {version:'natlang.program/2',id:'fixture-entry',kind:'evaluation_fixture',source:'test',split:'train',
    source_ids:['fixture-entry'],source_groups:['fixture-world'],license:'test',semantics:{root:'increment.nl',
    files:{'increment.nl':'---\nargs: { n: number }\nreturns: number\n---\nReturn n plus one.\n',
      'natlang.ts':'import increment from "./increment.nl"; export { increment };\n'},
    inputs:{n:3},expected:4,operation:'constraints',evaluation_fixture:{schema:'natlang.evaluation-fixture/1',
      program_root:'skill',suite_module:'skill/suite.ts',program_entry:'natlang.ts',case:selected,
      root_role:'entry',entry:{kind:'natlang-root',module:'increment.nl',export:'default',input_arguments:['n'],export_path:['increment']},service_inputs:[],
      host_files:{'skill/asset.json':'{"offset":1}', 'skill/suite.ts':`
import { defineEvaluationSuite } from '@workspace/natlang/evaluation';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
export default defineEvaluationSuite<any,any>({id:'portable',program:{root:'.',entry:'natlang.ts',id:'portable'},
components:'lambda-only',executorIdentity:{id:'original-student',configuration:{}},systemPrompt:'Authored fixture system prompt.',
policy:{codeEdits:'deny',systemPromptHash:${JSON.stringify(fingerprint('Authored fixture system prompt.','natlang.system-prompt/v1'))},settings:{temperature:0.3}},cases:[],budget:{maxRollouts:5,maxModelCalls:10,maxProposals:0},
fixture:{async create(c,context){const app=await context.loadFreshProgram() as {increment(n:number):Promise<number>};const asset=JSON.parse(readFileSync(join(context.programRoot,'asset.json'),'utf8'));return {
execute:()=>context.runtime.run(()=>app.increment(c.input.n)),observe:(result,outcome)=>({...(result===undefined?{}:{result}),offset:asset.offset})};}},
score(c,observation:any){return {quality:observation.result===c.expected && observation.offset===1 ? 1:0,gates:{world:observation.offset===1}};}});
`}}}};
}
const options={rootSeed:9,runId:'fixture-run',calls:false};
const teacher=()=>Object.assign(async request=>({calls:[['eval',{code:'n + 1',finish:true}]],completion_tokens:2}),{model:'real-teacher'});

test('canonical fixture executes the original root, assets and scorer with argument and source evidence',async()=>{
  const task=record(), requests=[];
  const driver=Object.assign(async request=>{requests.push(structuredClone(request));return teacher()(request);},{model:'real-teacher'});
  const run=await executeProgram(task,driver,options);
  assert.equal(run.outcome.accepted,true,JSON.stringify(run.outcome));
  assert.equal(run.outcome.value,4); assert.equal(requests.length,1);
  assert.match(JSON.stringify(requests[0].messages),/Authored fixture system prompt/);
  assert.equal(run.outcome.evaluation.fixture_entry.calls,1);
  assert.equal(run.outcome.evaluation.outcome.value,4);
  assert.deepEqual(run.outcome.evaluation.entry,task.semantics.evaluation_fixture.entry);
  assert.equal(run.outcome.invocation_ledger.length,1);
  assert.ok(run.outcome.invocation_ledger[0].definition_source.endsWith('increment.nl'));
  assert.match(JSON.stringify(run.trace),/real-teacher/);
});

test('IR inputs are enforced on the actual callable before model execution',async()=>{
 const task=record();task.semantics.inputs.n=99;let calls=0;
 const run=await executeProgram(task,async()=>{calls++;return teacher()();},options);
 assert.equal(calls,0);assert.equal(run.outcome.accepted,false);
 assert.match(run.outcome.evaluation.outcome.error.message,/differs from the program IR input/);
 assert.equal(run.outcome.checks.fixture_entry_binding,false);
});

test('changing original host-world assets changes the original scorer result',async()=>{
 const task=record();task.semantics.evaluation_fixture.host_files['skill/asset.json']='{"offset":7}';
 const run=await executeProgram(task,teacher(),options);
 assert.equal(run.outcome.accepted,false);assert.equal(run.outcome.evaluation.gates.world,false);
});

test('fixture descriptor rejects wrong kind, split, oracle files and missing export binding',()=>{
 for(const mutate of [r=>r.kind='lambda_source',r=>r.split='test',r=>r.semantics.files['oracle.ts']='export {};',
 r=>r.semantics.evaluation_fixture.entry.export_path=[],r=>r.semantics.evaluation_fixture.host_files['../escape']='bad']){
 const task=record();mutate(task);assert.throws(()=>validateFocusedRecord(task));}
});

test('authored module keeps its semantic anchor without inventing a model invocation',async()=>{
 const task=record();task.semantics.files['natlang.ts']='export async function increment(n:number):Promise<number>{return n+1;}';
 task.semantics.evaluation_fixture.entry={kind:'authored-module',module:'natlang.ts',export:'increment',input_arguments:['n'],export_path:['increment']};
 task.semantics.evaluation_fixture.root_role='semantic-anchor';
 const run=await executeProgram(task,async()=>{throw Error('typed branch must not invoke model');},options);
 assert.equal(run.outcome.accepted,true,JSON.stringify(run.outcome));assert.equal(run.trace.length,0);
 assert.equal(run.outcome.invocation_ledger.length,0);
});

test('finite decision functions collect real generated turns rather than provider decision scores',async()=>{
 const task=record();task.semantics.files['increment.nl']=task.semantics.files['increment.nl'].replace('returns: number','returns: 4 | 5\nreadout: decision');
 let generated=0;
 const driver=Object.assign(async()=>{generated++;return {text:'4',calls:[],completion_tokens:2};},
 {model:'finite-teacher',decide:async()=>{throw Error('a teacher demonstration must not substitute decision scores');}});
 const run=await executeProgram(task,driver,options);
 assert.equal(run.outcome.accepted,true,JSON.stringify(run.outcome));assert.equal(generated,1);
});

test('explicit collection settings override the authored settings and preserve teacher identity',async()=>{
 const requests=[];const driver=Object.assign(async request=>{requests.push(request);return teacher()(request);},{model:'override-teacher'});
 const run=await executeProgram(record(),driver,{...options,systemPrompt:'Collection prompt override.',temperature:0.7,contextTokens:8192,promptRendering:'full'});
 assert.equal(run.outcome.accepted,true,JSON.stringify(run.outcome));
 assert.match(JSON.stringify(requests[0].messages),/Collection prompt override/);
 assert.doesNotMatch(JSON.stringify(requests[0].messages),/Authored fixture system prompt/);
 assert.equal(requests[0].temperature,0.7);
});


test('repeated canonical execution renders stable real requests and source identity',async()=>{
 const requests=[];const driver=Object.assign(async request=>{requests.push(structuredClone(request));return teacher()(request);},{model:'replay-teacher'});
 const a=await executeProgram(record(),driver,options),b=await executeProgram(record(),driver,options);
 assert.equal(a.outcome.accepted,true);assert.equal(b.outcome.accepted,true);
 assert.deepEqual(requests[0].messages,requests[1].messages);
 assert.deepEqual(requests[0].tools,requests[1].tools);
 assert.equal(a.outcome.evaluation.suite_hash,b.outcome.evaluation.suite_hash);
 assert.doesNotMatch(JSON.stringify(requests[0]),/natlang-evaluation-fixture-/);
});


test('native collector preserves generated decision requests and plain JSON replies',async()=>{
 const requests=[];
 const server=createServer((request,response)=>{let body='';request.setEncoding('utf8');request.on('data',chunk=>body+=chunk);request.on('end',()=>{requests.push(JSON.parse(body));
 response.writeHead(200,{'content-type':'application/json'});response.end(JSON.stringify({choices:[{message:{role:'assistant',content:'4',tool_calls:[]}}],usage:{prompt_tokens:10,completion_tokens:2}}));});});
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 const directory=await mkdtemp(join(tmpdir(),'fixture-canonical-'));
 try{
 const task=record();task.semantics.files['increment.nl']=task.semantics.files['increment.nl'].replace('returns: number','returns: 4 | 5\nreadout: decision');
 const config={jobs:join(directory,'jobs'),output:join(directory,'out.jsonl'),workers:1,
 modelId:'canonical-teacher',endpoint:`http://127.0.0.1:${server.address().port}`,rootSeed:9,systemPrompt:'Teacher context.',contextTokens:8192,
 maxTurns:3,transportRetries:0,toolSurfaceSha256:await defaultToolSurfaceHash()};
 const status=await collectBatch([{index:0,record:task}],config,nativeJobRunner(config));assert.equal(status.completed,1);
 const row=JSON.parse((await readFile(config.output,'utf8')).trim());
 assert.equal(row.version,'natlang.teacher_trajectory.native/1');
 assert.deepEqual(row.task.program_ir,task);assert.equal(row.outcome.accepted,true,JSON.stringify(row.outcome));
 assert.equal(row.trajectory.length,1);
 assert.equal(row.trajectory[0].invocation_id,row.outcome.invocation_ledger[0].invocation_id);
 assert.match(JSON.stringify(row.outcome.execution_graph),/canonical-teacher/);
 assert.equal(row.outcome.evaluation.quality,1);
 assert.deepEqual(row.trajectory[0].tools_offered,[]);assert.equal(row.trajectory[0].model_response.text,'4');
 if(process.env.NATLANG_TEST_FIXTURE_ROW_OUTPUT){const {writeFile}=await import('node:fs/promises');await writeFile(process.env.NATLANG_TEST_FIXTURE_ROW_OUTPUT,JSON.stringify({row,requests,decision_system_prompt:(await import('../dist/native/decision.js')).DECISION_SYSTEM_PROMPT}));}
 }finally{await new Promise(resolve=>server.close(resolve));await rm(directory,{recursive:true,force:true});}
});

test('collector capabilities execute the portable fixture frontend before advertising support',async()=>{
 const {teacherCollectorCapabilities}=await import('../dist/teacher/runtime-capabilities.js');
 const capabilities=await teacherCollectorCapabilities();
 assert.equal(capabilities.evaluation_fixture,'natlang.evaluation-fixture/1');assert.equal(capabilities.probes.evaluation_fixture,true);
});
