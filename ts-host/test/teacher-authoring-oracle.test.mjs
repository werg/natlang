import assert from 'node:assert/strict';
import {test} from 'node:test';
import {createServer} from 'node:http';
import {mkdtemp,readFile,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {collectBatch,nativeJobRunner,defaultSystemPrompt,defaultToolSurfaceHash} from '../dist/teacher/collector.js';

for (const correct of [true,false]) test(`collector uses actual executable authoring oracle, passing=${correct}`,async()=>{
  const dir=await mkdtemp(join(tmpdir(),'authoring-oracle-'));
  const server=createServer((request,response)=>{
    request.resume();request.on('end',()=>{response.writeHead(200,{'content-type':'application/json'});
      response.end(JSON.stringify({choices:[{message:{role:'assistant',content:'',tool_calls:[{id:'done',type:'function',function:{name:'return_result',arguments:JSON.stringify({status:'success',value:'done'})}}]}}]}));});
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  try {
    const program={version:'natlang.program/2',id:'authoring:test',kind:'lambda_source',source:'fixture',split:'train',source_ids:['authoring:test'],source_groups:['authoring:test'],license:'test',semantics:{
      root:'implement.nl',files:{'implement.nl':'---\nargs: {}\nreturns: string\nkind: directory-reducer\n---\nComplete the implementation.\n'},inputs:{},expected:null,
      folder_files:{'app.ts':`export async function double(value: number): Promise<number> { return value * ${correct?2:3}; }`},
      authoring:{module:'app.ts',export:'double',runs:[{args:[3],expected:6},{args:[8],expected:16}]}}};
    const options={jobs:join(dir,'jobs'),output:join(dir,'out.jsonl'),workers:1,modelId:'teacher',rootSeed:7,systemPrompt:defaultSystemPrompt,contextTokens:16384,
      endpoint:`http://127.0.0.1:${server.address().port}`,transportRetries:0,toolSurfaceSha256:await defaultToolSurfaceHash()};
    await collectBatch([{index:0,record:program}],options,nativeJobRunner(options));
    const row=JSON.parse((await readFile(options.output,'utf8')).trim());
    assert.equal(row.outcome.value,'done');
    assert.equal(row.outcome.authoring.ok,correct);
    assert.deepEqual(row.outcome.oracle,{accepted:correct,level:'exact',verdict:'executable-authoring/1'});
    assert.equal(row.outcome.accepted,correct);
  } finally {await new Promise(resolve=>server.close(resolve));await rm(dir,{recursive:true,force:true});}
});
