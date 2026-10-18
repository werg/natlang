import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createHash} from 'node:crypto';
import {buildScaffolds,specs} from '../../scripts/neuralese_data/build_rl_task_curriculum.mjs';
const runtime=process.env.NATLANG_CONTRACT_RUNTIME;
if(!runtime)throw Error('NATLANG_CONTRACT_RUNTIME must name the built runtime');
const {pathToFileURL}=await import('node:url');
const {resolve}=await import('node:path');
const {validateCurriculum,admitRow}=await import(pathToFileURL(resolve(runtime,'dist/teacher/curriculum.js')).href);
const {referenceRow}=await import('../../ts-host/scripts/inline-curriculum/references.mjs');

test('independent complete static tasks retain exact oracles, private calls and pending qualification',async()=>{
 const rows=await buildScaffolds(runtime);assert.equal(rows.length,12);
 assert.equal(new Set(rows.map(row=>row.id)).size,12);
 for(const row of rows){
  validateCurriculum(row);
  assert.equal(row.split,'train');assert.equal(row.semantics.operation,'exact');
  assert.equal(row.curriculum.rl.qualification,'pending-live-48');
  assert.ok(row.semantics.files[row.semantics.root]);
  const reference=row.curriculum.reference.root[0][1].code;
  if(row.curriculum.trajectory_contract){
   assert.equal(row.curriculum.trajectory_contract.authored_root_sha256,createHash('sha256').update(reference).digest('hex'));
   assert.equal(row.curriculum.trajectory_contract.min_state_edges_from_children,0);
   assert.match(reference,/Neuralese</);
  }else{
   const value=Function(reference)();assert.deepEqual(value,row.semantics.expected);
  }
 }
 assert.deepEqual([...new Set(rows.map(row=>row.curriculum.rl.level))],[0,1,2,3]);
 assert.equal(rows.filter(row=>row.curriculum.trajectory_contract).length,3);
 assert.equal(specs[6].expected,45);assert.equal(specs[7].expected,80);
});

import {createServer} from 'node:http';
import {mkdtemp,mkdir,rm,readFile,writeFile} from 'node:fs/promises';
import {execFile} from 'node:child_process';
import {promisify} from 'node:util';
import {tmpdir} from 'node:os';
const {nativeJobRunner,expectedProvenance,defaultToolSurfaceHash}=await import(pathToFileURL(resolve(runtime,'dist/teacher/collector.js')).href);

test('real natlang host verifier accepts ordinary tasks and private writer public return, and rejects wrong answer',async()=>{
 const dir=await mkdtemp(resolve(tmpdir(),'rl-curriculum-contract-'));let answer;
 const server=createServer((request,response)=>{
  request.resume();request.on('end',()=>{
   response.writeHead(200,{'content-type':'application/json'});
   response.end(JSON.stringify({choices:[{message:{role:'assistant',content:'',tool_calls:[{
    id:'contract-return',type:'function',function:{name:'return_result',arguments:JSON.stringify({status:'success',value:answer})}}]}}]}));
  });
 });
 await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
 try{
  const rows=await buildScaffolds(runtime);let accepted=0;
  for(const [index,row] of rows.slice(0,10).entries()){
   answer=row.semantics.expected;const jobs=resolve(dir,'case-'+index);await mkdir(jobs,{recursive:true});
   const options={jobs,output:resolve(jobs,'out.jsonl'),workers:1,modelId:'local-contract-fixture',
    rootSeed:7,systemPrompt:'Execute this typed task.',contextTokens:8192,
    toolSurfaceSha256:await defaultToolSurfaceHash(),endpoint:`http://127.0.0.1:${server.address().port}`,
    collectionRole:'teacher',transportRetries:0};
   const item={index:0,record:row};const result=await nativeJobRunner(options)(item,expectedProvenance(row,options));
   assert.equal(result.outcome.accepted,true,JSON.stringify(result.outcome));accepted++;
  }
  assert.equal(accepted,10);
  const row=rows[0];answer=18;const jobs=resolve(dir,'wrong');await mkdir(jobs,{recursive:true});
  const options={jobs,output:resolve(jobs,'out.jsonl'),workers:1,modelId:'local-contract-fixture',
   rootSeed:7,systemPrompt:'Execute this typed task.',contextTokens:8192,
   toolSurfaceSha256:await defaultToolSurfaceHash(),endpoint:`http://127.0.0.1:${server.address().port}`,
   collectionRole:'teacher',transportRetries:0};
  const wrong=await nativeJobRunner(options)({index:0,record:row},expectedProvenance(row,options));
  assert.equal(wrong.outcome.accepted,false);assert.ok(wrong.outcome.rejection_reasons.includes('answer'));
 }finally{await new Promise(resolve=>server.close(resolve));await rm(dir,{recursive:true,force:true});}
});


test('all twelve exact sources pass scripted text transport and unchanged admission',async()=>{
 const options={modelId:'scripted-source-reference',rootSeed:909,systemPrompt:'Execute this typed task.',
  contextTokens:65536,maxTurns:80,toolSurfaceSha256:await defaultToolSurfaceHash(),
  collectionRole:'reference',textNeuraleseEmulation:true};
 for(const [index,record] of (await buildScaffolds(runtime)).entries()){
  const actual=await referenceRow(record,index,options);
  assert.equal(actual.outcome.accepted,true,JSON.stringify({id:record.id,outcome:actual.outcome}));
  assert.equal(admitRow(actual).admitted,true,JSON.stringify({id:record.id,admission:admitRow(actual)}));
 }
});

// Golden complete original programs from repaired079b12, projected only to the
// old descriptor fields. New schema metadata and scripted child witnesses may
// be added; original task text, files, oracle, root, groups and RL status may not.
const originalProgramDigests={
  "rl-execution-curriculum-20261016:integer-return": "a6941da501db217c7c2280558004c92419c109621f49ba76f7bc2c22f12f01da",
  "rl-execution-curriculum-20261016:string-return": "46403900098a7b324a88e027bdad801f60fcff778170e5d8cd3c3606ca6a3195",
  "rl-execution-curriculum-20261016:boolean-return": "54fbe1807299d2e5f477de6a5ef6b0ef52bbdd567d645acac265b7c8b4d39e5c",
  "rl-execution-curriculum-20261016:addition": "0a3a27858439501d04a886db5df5999acd1dc66fed9a223854ae1252cc596b19",
  "rl-execution-curriculum-20261016:sort": "c31f8d4fc2378d4176bec1e54a4b11f26ecebfab1eed7001b233d939dbf86421",
  "rl-execution-curriculum-20261016:field-extract": "7577299206d0398f25e2975764a24543cef67a09b4a5a349c0425128e1785e67",
  "rl-execution-curriculum-20261016:invoice-total": "3dfbf9b9362fe5d1a4cafecfa0f83fa8aba5ea4e6f75bf75f069de20a1837927",
  "rl-execution-curriculum-20261016:filter-sum": "80d7724d97ee01784488fdacd4cde547629ad3a62e51cf11a33f30b3368876c4",
  "rl-execution-curriculum-20261016:normalize-join": "93064665bb053d9f835a32c0da9e5e41c1234739e642f3324b1bc08012dc8e81",
  "rl-execution-curriculum-20261016:private-writer-return": "81bfc5e2d3e108e292daf9416ab329697437e30a8bf405b4666f8d36dbc3c615",
  "rl-execution-curriculum-20261016:private-writer-reader": "46c71bd3400a209034df61d8085ac5ad20c87fcc25e4f04ddd32b3bca50d141d",
  "rl-execution-curriculum-20261016:private-two-inputs": "1a19989ccaab747dfe9b4c9c8ad913762bd6f044e626d5d39363bed83b369992"
};
function canonical(value){
 if(Array.isArray(value))return value.map(canonical);
 if(value!==null&&typeof value==='object')return Object.fromEntries(Object.keys(value).sort().map(k=>[k,canonical(value[k])]));
 return value;
}
test('schema repair preserves every original model-facing and oracle program field',async()=>{
 for(const row of await buildScaffolds(runtime)){
  const c=row.curriculum;
  assert.equal(c.reference.root.length,2);assert.deepEqual(c.reference.root[1],['reply',{text:'done'}]);
  const old={...row,curriculum:{version:c.version,family:c.family,reference:{root:c.reference.root.slice(0,1)},rl:c.rl,
   ...(c.trajectory_contract?{trajectory_contract:c.trajectory_contract}:{})}};
  const sha=createHash('sha256').update(JSON.stringify(canonical(old))).digest('hex');
  assert.equal(sha,originalProgramDigests[row.id],row.id);
 }
});

const {readTextFromRenderedOpening,scriptedTextReferenceDriver}=await import('../../ts-host/scripts/inline-curriculum/text-reference-driver.mjs');
test('scripted text reads require one exact visible opening and preserve ordinary source dispatch',async()=>{
 const block=(id,value)=>`[[Neuralese text block id=${id}; type=Neuralese<string>; exact JSON string body=${JSON.stringify(value)}]]`;
 const body='quote " and ]] plus \\ and \n';const a=block('a',body);
 assert.equal(readTextFromRenderedOpening({messages:[{content:a}]}),body);
 assert.equal(readTextFromRenderedOpening({messages:[{content:a},{content:a}]}),body);
 for(const messages of [[],[{content:'[[Neuralese text block id=a; type=Neuralese<string>; exact JSON string body=invalid]]'}],
  [{content:block('a','one')+block('b','two')}],[{content:'[[Neuralese text block id=a; type=Neuralese<boolean>; exact JSON string body="x"]]'}]])
  assert.throws(()=>readTextFromRenderedOpening({messages}),/exactly one visible exact JSON string opening/);
 const ordinary={messages:[{content:'ordinary source request'}]};let calls=0;
 const ref=scriptedTextReferenceDriver(async request=>{calls++;assert.equal(request,ordinary);return {calls:[['reply',{text:'done'}]]};});
 assert.deepEqual(await ref(ordinary),{calls:[['reply',{text:'done'}]]});assert.equal(calls,1);
 const read=await ref({messages:[{content:'This is a typed Neuralese read call. '+a}]});
 assert.equal(read.calls[0][1].value,body);assert.equal(calls,1);
 await assert.rejects(ref({messages:[{content:'This is a typed Neuralese read call.'}]}),/exactly one visible/);
});
