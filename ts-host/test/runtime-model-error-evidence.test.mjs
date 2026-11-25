import assert from 'node:assert/strict';
import test from 'node:test';
import {createNatlangRuntime,nodePlatform} from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {loadVirtualNatlang} from '../dist/runtime/virtual-project.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
for(const mode of ['node','workerd'])test(`${mode} actual model driver failure keeps operational cause in durable call events`,async()=>{
 const records=[],events=[],blobs=new Map();const store={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){},record(record,values,text){records.push(record);for(const [k,v]of values)blobs.set(k,v);events.push(...(text??'').trim().split('\n').filter(Boolean).map(JSON.parse));}};
 const failure=new TypeError('fetch failed',{cause:Object.assign(new Error('private request'),{code:'ECONNRESET',syscall:'read'})});let requested=0;
 const platform=mode==='node'?nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const runtime=createNatlangRuntime({platform,calls:store,model:async()=>{requested++;throw failure;}});
 try{const fn=loadVirtualNatlang({'read.nl':'---\nargs: {}\nreturns: string\n---\nRead the assigned value.'},'read.nl','/read',platform);await assert.rejects(runtime.run(()=>fn()));assert.equal(requested,1);
 const event=events.find(e=>e.kind==='model_request'&&e.phase==='error');assert.ok(event,'durable error event exists');assert.deepEqual(event.error_metadata,{name:'TypeError',cause:{name:'Error',code:'ECONNRESET',syscall:'read'}});assert.ok(!JSON.stringify(event.error_metadata).includes('private'));assert.equal(records[0].output,null);assert.notEqual(records[0].outcome,'done');
 }finally{runtime.close();}
});
