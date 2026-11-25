import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame} from '../dist/runtime/context.js';
import {loadVirtualNatlang} from '../dist/runtime/virtual-project.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
function memory(){const records=new Map(),events=new Map();return{records,events,settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:(r,_blobs,text)=>{records.set(r.call_id,r);events.set(r.call_id,(text??"").trim().split("\n").filter(Boolean).map(line=>JSON.parse(line)));},version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};}
const nl=body=>`---\nargs: {}\nreturns: string\n---\n${body}`;
for(const mode of ['node','workerd'])for(const independent of [false,true])test(`${mode} nested service task preserves causal trace with fresh authority (independent=${independent})`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const store=memory();let programs,bridgeFrame,outerTask;const nestedTasks=[];const parentAbort=new AbortController();
 const runAgent=async session=>{const f=session.runtime.frame;if(session.lam.functionName==='check'){nestedTasks.push(f.task);assert.deepEqual(Object.keys(session.availableServices()),['own']);assert.notEqual(f.signal,bridgeFrame.signal);assert.equal(f.task.modelBudget.usage().requests,0);}else outerTask=f.task;
 const result=await session.applyAsync('eval',{code:session.lam.functionName==='outer'?'await bridge.read(); return await bridge.read();':'return own.read();',finish:true});assert.equal(result.kind,'completed',result.text);};
 const child=core.createNatlangRuntime({platform,calls:store,programRoot:'/loaded',services:{own:{read:()=> 'actual'}},serviceDeclarations:{own:'export function read():string;'},agent:runAgent});
 const runtime=core.createNatlangRuntime({platform,calls:store,programRoot:'/caller',signal:parentAbort.signal,services:{bridge:{read:async()=>{bridgeFrame=currentFrame();assert.ok(bridgeFrame?.parentCallId);return child.run(()=>programs.check(),{traceParent:!independent,serviceScopes:{own:['check.nl']}});}},forbidden:{read:()=> 'secret'}},serviceDeclarations:{bridge:'export function read():Promise<string>;',forbidden:'export function read():string;'},agent:runAgent});
 try{const files={'outer.nl':nl('Read bridge.'),'check.nl':nl('Read own.')};programs={outer:loadVirtualNatlang(files,'outer.nl','/causal',platform),check:loadVirtualNatlang(files,'check.nl','/causal',platform)};assert.equal(await runtime.run(()=>programs.outer()),'actual');
 const rows=[...store.records.values()],outer=rows.find(r=>r.definition.name==='outer'),nested=rows.filter(r=>r.definition.name==='check');assert.ok(outer);assert.equal(nested.length,2);assert.equal(new Set(nested.map(r=>r.task_id)).size,2);
 for(const row of nested){assert.notEqual(row.task_id,outer.task_id);assert.equal(row.program_root,'/loaded');assert.equal(row.parent_call_id,independent?null:outer.call_id);assert.equal(row.parent_action_index,independent?null:0);}
 assert.equal(outer.program_root,'/caller');assert.deepEqual(outer.graph_custody.child_call_ids,independent?[]:nested.map(r=>r.call_id));
 assert.deepEqual(nested.map(r=>store.events.get(r.call_id).find(e=>e.kind==='manifest').sibling_ordinal),independent?[0,0]:[0,1]);
 for(const task of nestedTasks){assert.notEqual(outerTask,task);assert.notEqual(outerTask.modelBudget,task.modelBudget);assert.equal(task.runtime,child);}

 }finally{runtime.close();child.close();}
});
