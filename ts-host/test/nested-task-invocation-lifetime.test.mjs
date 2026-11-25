import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame,runInFrame} from '../dist/runtime/context.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
import {NodeNativeRuntime} from '../dist/node-runtime.js';
const nl='---\nargs: {value: number}\nreturns: number\n---\nRead the actual own service.';
function store(){const records=new Map();return{records,settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.set(r.call_id,r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};}
function setup(mode,recording){
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const calls=store();let bridge,outerTask;const childTasks=[];
 const child=core.createNatlangRuntime({platform,calls,programRoot:'/loaded',recording,services:{own:{read:value=>value*2}},agent:async session=>{const frame=currentFrame();childTasks.push(frame.task);assert.deepEqual(Object.keys(session.availableServices()),['own']);const result=await session.applyAsync('eval',{code:'return own.read(value);',finish:true});assert.equal(result.kind,'completed',result.text);}});
 const outer=core.createNatlangRuntime({platform,calls,programRoot:'/owner',services:{bridge:{run:async value=>{outerTask=currentFrame().task;return bridge(value);}}}});
 const files={'services.d.ts':'declare module "natlang:services" {export const bridge:{run(value:number):Promise<number>};}',
  'outer.ts':'import {bridge} from "natlang:services";export default async function outer(value:number):Promise<number>{return await bridge.run(value);}',
  'check.ts':'import judge from "./judge.nl";import helper from "./helper";export async function check(value:number):Promise<number>{return await judge(helper(value));}',
  'helper.ts':'export default function helper(value:number):number{return value+1;}', 'judge.nl':nl};
 const project=core.compileVirtualProject({files,root:'/loaded-reference'},core,{platform});assert.equal(project.ok,true,JSON.stringify(project.diagnostics));
 const modules={outer:project.require('outer.ts').default,check:project.require('check.ts').check};
 return {platform,calls,child,outer,modules,childTasks,setBridge(fn){bridge=fn;},get outerTask(){return outerTask;},close(){outer.close();child.close();}};
}
for(const mode of ['node','workerd'])test(`${mode} actual loaded TS entry remains a crisp child with its NL child and fresh scopes`,async()=>{
 const s=setup(mode);s.setBridge(value=>s.child.run(()=>s.modules.check(value)));
 try{
  assert.equal(await s.outer.run(()=>s.modules.outer(2)),6);
  const rows=[...s.calls.records.values()],outer=rows.find(row=>row.definition.name==='outer'),check=rows.find(row=>row.definition.name==='check'),judge=rows.find(row=>row.definition.name==='judge');
  assert.ok(outer&&check&&judge);assert.equal(rows.length,3);assert.equal(check.executor.kind,'crisp');assert.equal(check.parent_call_id,outer.call_id);assert.equal(judge.parent_call_id,check.call_id);assert.notEqual(check.task_id,outer.task_id);assert.equal(check.program_root,'/loaded');assert.deepEqual(outer.graph_custody.child_call_ids,[check.call_id]);assert.deepEqual(check.graph_custody.child_call_ids,[judge.call_id]);assert(!rows.some(row=>row.definition.name==='helper'));assert.equal(s.childTasks[0].isInvocationActive(check.call_id),false);
 }finally{s.close();}
});
for(const mode of ['node','workerd'])test(`${mode} nativeRoots opt-out preserves real NL causality and service scopes`,async()=>{
 const s=setup(mode,{nativeRoots:false});s.setBridge(value=>s.child.run(()=>s.modules.check(value)));
 try{assert.equal(await s.outer.run(()=>s.modules.outer(2)),6);const rows=[...s.calls.records.values()],outer=rows.find(row=>row.definition.name==='outer'),judge=rows.find(row=>row.definition.name==='judge');assert.equal(rows.length,2);assert.equal(judge.parent_call_id,outer.call_id);assert.equal(judge.program_root,'/loaded');}finally{s.close();}
});
for(const mode of ['node','workerd'])for(const admission of ['late','deferred'])test(`${mode} ${admission} task entry cannot attach to a completed invocation in an open task`,async()=>{
 const s=setup(mode);let release,completion,saved;
 const gate=new Promise(resolve=>{release=resolve;});
 s.setBridge(()=>{saved=currentFrame();completion=admission==='late'?gate.then(()=>s.child.run(()=>s.modules.check(4))):s.child.run(async()=>{await gate;return s.modules.check(4);});return 1;});
 try{
  await s.outer.run(async()=>{assert.equal(await s.modules.outer(1),1);assert.equal(saved.task.isClosed,false);const before=[...s.calls.records.values()].find(row=>row.definition.name==='outer');assert.deepEqual(before.graph_custody.child_call_ids,[]);release();assert.equal(await completion,10);});
  const rows=[...s.calls.records.values()],outer=rows.find(row=>row.definition.name==='outer'),check=rows.find(row=>row.definition.name==='check');assert.equal(check.parent_call_id,null);assert.deepEqual(outer.graph_custody.child_call_ids,[]);
 }finally{release();await completion?.catch(()=>{});s.close();}
});
for(const mode of ['node','workerd'])test(`${mode} an aborted parent grants neither stale ancestry nor cancellation to an explicitly fresh task`,async()=>{
 const s=setup(mode);s.setBridge(value=>{currentFrame().task.cancel(Error('parent stopped'));return s.child.run(()=>s.modules.check(value));});
 try{assert.equal(await s.outer.run(()=>s.modules.outer(2)),6);const rows=[...s.calls.records.values()],check=rows.find(row=>row.definition.name==='check');assert.equal(check.parent_call_id,null);assert.equal(s.childTasks[0].signal.aborted,true); // closed after successful independent completion
 }finally{s.close();}
});
test('runtime.bind retains lexical task authority after its creator invocation ends',async()=>{
 const s=setup('node');let bound,taskBound;
 s.setBridge(()=>{bound=s.outer.bind(()=>42);return 1;});
 try{await s.outer.run(async()=>{taskBound=s.outer.bind(()=>42);await s.modules.outer(1);assert.equal(taskBound(),42);assert.equal(bound(),42);});}finally{s.close();}
});
test('standalone native runtime owns its real recorded root lifetime for nested TS entry',async()=>{
 const {buildPending}=await import('../dist/native/values.js'),{toHost}=await import('../dist/runtime/kernel.js');
 const s=setup('node');let sourceFrame;
 const native=new NodeNativeRuntime({calls:s.calls,runId:'standalone-root',services:{bridge:{read:()=>{sourceFrame=currentFrame();return s.child.run(()=>s.modules.check(2));}}},declarations:{bridge:'export function read():Promise<number>;'},agent:async session=>{const result=await session.applyAsync('eval',{code:'return await bridge.read();',finish:true});assert.equal(result.kind,'completed',result.text);}});
 try{const result=await native.run(buildPending({$lambda:{type:'() => number',instructions:'Read native bridge.'}}));assert.equal(toHost(result.value),6);const check=[...s.calls.records.values()].find(row=>row.definition.name==='check');assert.equal(check.parent_call_id,'standalone-root');assert.equal(sourceFrame.task.isInvocationActive('standalone-root'),false);}finally{native.environment.close();native.frame.task.close();native.frame.task.runtime.close();s.close();}
});
