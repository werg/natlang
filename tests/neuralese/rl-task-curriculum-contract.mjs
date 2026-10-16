import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createHash} from 'node:crypto';
import {buildScaffolds,specs} from '../../scripts/neuralese_data/build_rl_task_curriculum.mjs';
const runtime=process.env.NATLANG_CONTRACT_RUNTIME;
if(!runtime)throw Error('NATLANG_CONTRACT_RUNTIME must name the built runtime');
const {pathToFileURL}=await import('node:url');
const {resolve}=await import('node:path');

test('independent complete static tasks retain exact oracles, private calls and pending qualification',async()=>{
 const rows=await buildScaffolds(runtime);assert.equal(rows.length,12);
 assert.equal(new Set(rows.map(row=>row.id)).size,12);
 for(const row of rows){
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
import {mkdtemp,mkdir,rm} from 'node:fs/promises';
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
