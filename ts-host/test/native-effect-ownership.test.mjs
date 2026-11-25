/** TEST ONLY local scripted executor: no provider, model inference, or teacher registry. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame,runInFrame} from '../dist/runtime/context.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
import {recordingServices,unrecordedServices,ONCE_EFFECTS} from '../dist/native/effects.js';
import {checkServiceResults} from '../dist/native/refinement.js';

for(const mode of ['node','workerd'])
test(`${mode} native entry to NL descendant assigns each physical effect to its actual executing call`,async()=>{
 const records=[];let physicalCalls=0,forbiddenCalls=0;
 const calls={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.push(r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const own={async read(){physicalCalls++;return 10;}},forbidden={read(){forbiddenCalls++;return 99;}};
 const runtime=core.createNatlangRuntime({platform,calls,services:{own,forbidden},serviceDeclarations:{own:'export function read():Promise<number>;',forbidden:'export function read():number;'},agent:async session=>{
  const result=await session.applyAsync('eval',{code:'return await own.read();',finish:true});
  assert.equal(result.kind,'completed',result.text);
 }});
 try{
  const built=core.compileVirtualProject({root:'/test-only-effect-owner',files:{
   'entry.ts':'import judge from "./judge.nl";export async function check():Promise<number>{return await judge();}',
   'judge.nl':'---\nargs: {}\nreturns: number\n---\nRead the owned value.'}},core,{platform});
  assert.equal(built.ok,true,JSON.stringify(built.diagnostics));
  const fn=built.require('entry.ts').check;
  assert.deepEqual(await Promise.all([1,2].map(()=>runtime.run(()=>runInFrame({...currentFrame(),services:{own}},()=>fn())))),[10,10]);
  assert.equal(physicalCalls,2);assert.equal(forbiddenCalls,0);assert.equal(records.length,4);
  const roots=records.filter(r=>r.definition.name==='check'),children=records.filter(r=>r.definition.name==='judge');
  assert.equal(roots.length,2);assert.equal(children.length,2);
  for(const root of roots)assert.equal(root.effects.length,0);
  for(const child of children){
   assert.ok(roots.some(root=>root.call_id===child.parent_call_id));
   assert.equal(child.effects.length,1);assert.equal(child.effects[0].service,'own');assert.equal(child.effects[0].by,'agent');
  }
  assert.ok(records.every(r=>r.cost.model_requests===0));
 }finally{runtime.close();}
});

test('observer provenance preserves exact grants, native receiver, once scope, and each invocation’s result checks',async()=>{
 let physical=0;const parent=[],child=[],checks=[];
 class Port {#value=10;[ONCE_EFFECTS]=['read'];get value(){return this.#value;}read(){physical++;return this.#value;}}
 const port=Object.freeze(new Port()),source={own:port};
 const ancestor=recordingServices(source,e=>parent.push(e));
 source.forbidden={read(){throw Error('not granted');}};
 const lexical=unrecordedServices({own:ancestor.own});
 assert.deepEqual(Object.keys(lexical),['own']);assert.equal(lexical.own,port);
 assert.deepEqual(Object.keys(unrecordedServices(ancestor)),['own']);
 const checked=checkServiceResults(lexical,{'own.read':{}},async value=>{checks.push(value);});
 const descendant=recordingServices(checked,e=>child.push(e),lexical);
 assert.equal(descendant.own.value,10);assert.equal(await descendant.own.read(),10);
 // A further invocation replaces this observer/checker, preserving the same lexical native port.
 const nextSource=unrecordedServices(descendant);
 const next=recordingServices(checkServiceResults(nextSource,{'own.read':{}},async value=>checks.push(value+1)),e=>child.push(e),nextSource);
 assert.equal(await next.own.read(),10);
 assert.equal(physical,1);assert.deepEqual(checks,[10,11]);assert.equal(parent.length,0);
 assert.deepEqual(child.map(e=>e.phase),['requested','completed','requested','completed']);
 assert.equal(await ancestor.own.read(),10);assert.equal(physical,1);assert.equal(parent.length,2);
 assert.deepEqual(Object.keys(unrecordedServices(recordingServices({},()=>{}))),[]);
});

test('cached effects still run each current result check and failed native effects remain retryable',async()=>{
 let physical=0;const events=[];
 const port={[ONCE_EFFECTS]:['read'],read(){if(++physical===1)throw Error('native refusal');return 10;}};
 const source={own:port};
 const invocation=check=>recordingServices(checkServiceResults(source,{'own.read':{}},check),e=>events.push(e),source);
 await assert.rejects(invocation(async()=>{}).own.read(),/native refusal/);
 await assert.rejects(invocation(async()=>{throw Error('current authority denied result');}).own.read(),/current authority denied result/);
 assert.equal(await invocation(async value=>assert.equal(value,10)).own.read(),10);
 assert.equal(physical,2);
 assert.deepEqual(events.map(e=>e.phase),['requested','failed','requested','failed','requested','completed']);
});
