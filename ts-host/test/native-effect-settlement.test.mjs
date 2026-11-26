/** TEST ONLY scripted execution and manually deferred Native ports; no provider. */
import test from 'node:test';
import assert from 'node:assert/strict';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
import {NativeEffectOwner,recordingServices} from '../dist/native/effects.js';
const turn=()=>new Promise(resolve=>setImmediate(resolve));
const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
const platformOf=mode=>mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
for(const mode of ['node','workerd'])for(const variant of ['nl','ts','unrecorded-ts','no-store-ts','opt-out'])
test(`${mode} ${variant} discarded Native RPC settles before loaded entry returns`,{timeout:15000},async()=>{
 const platform=platformOf(mode),gate=deferred(),entered=deferred(),records=[];let physical=false,returned=false,task;
 const calls={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.push(r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const rt=core.createNatlangRuntime({platform,...(variant==='no-store-ts'?{}:{calls}),recording:{nativeRoots:variant!=='unrecorded-ts'},settleNativeEffects:variant!=='opt-out',services:{owned:{async read(){entered.resolve();await gate.promise;physical=true;return 10;}}},serviceDeclarations:{owned:'export function read():Promise<number>;'},agent:async session=>{
  const result=await session.applyAsync('eval',{code:'void owned.read(); return 7;',finish:true});assert.equal(result.kind,'completed',result.text);
 }});
 const nl=variant==='nl'||variant==='opt-out';
 const files=nl?{'entry.ts':'import read from "./read.nl";export async function run():Promise<number>{return await read();}','read.nl':'---\nargs: {}\nreturns: number\n---\nStart the owned read and return seven.'}:{'entry.ts':'import {owned} from "natlang:services";export async function run():Promise<number>{void owned.read();return 7;}'};
 if(!nl)files['services.d.ts']='declare module "natlang:services" {export const owned:{read():Promise<number>};}';
 try{
  const built=core.compileVirtualProject({root:'/test-only-settlement',files},core,{platform});assert.equal(built.ok,true,JSON.stringify(built.diagnostics));
  task=rt.run(async()=>{const value=await built.require('entry.ts').run();returned=true;assert.equal(physical,variant!=='opt-out');return value;});
  await Promise.race([entered.promise,task.then(()=>{throw Error('Entry returned before entering the port');})]);await turn();assert.equal(returned,variant==='opt-out');
  if(variant!=='opt-out')assert.equal(records.some(r=>r.outcome==='done'),false);
  gate.resolve();assert.equal(await task,7);assert.equal(physical,true);
  if(variant==='unrecorded-ts'||variant==='no-store-ts')assert.equal(records.length,0);
  if(variant==='nl'){const child=records.find(r=>r.definition.name==='read');assert.equal(child.effects.length,1);assert.equal(child.effects[0].error,undefined);assert.equal(child.outcome,'done');}
 }finally{gate.resolve();if(task)await task;rt.close();}
});
test('effect owner preserves real rejection, joins concurrent accepted RPCs, and refuses expired use',async()=>{
 const owner=new NativeEffectOwner(),a=deferred(),b=deferred(),events=[];
 const services=recordingServices({owned:{read(id){return id==='a'?a.promise:b.promise;}}},e=>events.push(e),undefined,undefined,owner);
 const pa=services.owned.read('a'),pb=services.owned.read('b');let drained=false;
 const joined=owner.drain().then(()=>drained=true);a.resolve(1);assert.equal(await pa,1);await turn();assert.equal(drained,false);
 const rejected=assert.rejects(pb,/actual refusal/);b.reject(Error('actual refusal'));await rejected;await joined;
 assert.deepEqual(events.map(e=>e.phase),['requested','requested','completed','failed']);owner.close();assert.throws(()=>services.owned.read('a'),/invocation.*finished/);
});

for(const mode of ['node','workerd'])test(`${mode} standalone NativeRuntime joins discarded RPC at eval boundary`,{timeout:15000},async()=>{
 const {kernelHooks}=await import('../dist/runtime/hooks.js');
 const {NativeRuntime,NativeSession}=await import('../dist/native/runtime.js');
 const {TypeEnv}=await import('../dist/native/types.js');
 const {buildPending}=await import('../dist/native/values.js');
 const platform=platformOf(mode),gate=deferred(),entered=deferred();let done=false;
 const runtime=new NativeRuntime({environment:platform.environment({}),hooks:kernelHooks,services:{owned:{async read(){entered.resolve();await gate.promise;return 9;}}},declarations:{owned:'export function read():Promise<number>;'}});
 const lam=buildPending({$lambda:{type:'() => number',instructions:'Return seven.'}}),session=new NativeSession(runtime,lam,new TypeEnv());
 const action=session.applyAsync('eval',{code:'void owned.read();return 7;',finish:true}).then(r=>{done=true;return r;});
 try{await Promise.race([entered.promise,action.then(r=>{throw Error(r.text);})]);await turn();assert.equal(done,false);gate.resolve();assert.equal((await action).kind,'completed');}
 finally{gate.resolve();await action;runtime.environment.close();}
});
for(const mode of ['node','workerd'])for(const record of [true,false])test(`${mode} unawaited TS helper owns both RPCs across an unrelated await, trace ${record}`,{timeout:15000},async()=>{
 const platform=platformOf(mode),first=deferred(),middle=deferred(),seen=deferred();let n=0,returned=false,task;
 const rt=core.createNatlangRuntime({platform,calls:false,recording:{nativeRoots:record},services:{owned:{async read(){n++;seen.resolve();if(n===1)await first.promise;return n;}}}});
 const files={'entry.ts':'import helper from "./helper";export async function run(pause:Promise<void>):Promise<number>{void helper(pause);return 7;}', 'helper.ts':'import {owned} from "natlang:services";export default async function helper(pause:Promise<void>):Promise<void>{await owned.read();await pause;await owned.read();}', 'services.d.ts':'declare module "natlang:services" {export const owned:{read():Promise<number>};}'};
 try{const built=core.compileVirtualProject({root:'/test-only-helper-settlement',files},core,{platform});assert.equal(built.ok,true,JSON.stringify(built.diagnostics));task=rt.run(async()=>{const value=await built.require('entry.ts').run(middle.promise);returned=true;return value;});await Promise.race([seen.promise,task.then(()=>{throw Error('missing helper');})]);first.resolve();await turn();assert.equal(n,1);assert.equal(returned,false);middle.resolve();assert.equal(await task,7);assert.equal(n,2);}
 finally{first.resolve();middle.resolve();await task;rt.close();}
});

for(const mode of ['node','workerd'])test(`${mode} inert promise construction does not deadlock Native action settlement`,{timeout:15000},async()=>{
 const platform=platformOf(mode),runtime=core.createNatlangRuntime({platform,calls:false,agent:async session=>{const result=await session.applyAsync('eval',{code:'void new Promise<void>(()=>{});return 7;',finish:true});assert.equal(result.kind,'completed',result.text);}});
 try{const fn=core.loadVirtualNatlang({'root.nl':'---\nargs: {}\nreturns: number\n---\nReturn seven.'},'root.nl','/test-only-inert-promise',platform);assert.equal(await runtime.run(()=>fn()),7);}finally{runtime.close();}
});
for(const mode of ['node','workerd'])test(`${mode} synchronous TS entry refuses known async Native dispatch before effects`,async()=>{
 const platform=platformOf(mode);let physical=0;
 const rt=core.createNatlangRuntime({platform,calls:false,services:{owned:{async read(){physical++;return 1;}}}});
 const files={'entry.ts':'import {owned} from "natlang:services";export function run():number{void owned.read();return 7;}','services.d.ts':'declare module "natlang:services" {export const owned:{read():Promise<number>};}'};
 try{const built=core.compileVirtualProject({root:'/test-only-sync-native',files},core,{platform});assert.equal(built.ok,true,JSON.stringify(built.diagnostics));await rt.run(()=>assert.throws(()=>built.require('entry.ts').run(),/synchronous native entry cannot start asynchronous/));assert.equal(physical,0);}finally{rt.close();}
});
test('standalone NodeNativeRuntime finishes its record only after the actual Native effect settles',{timeout:15000},async()=>{
 const {NodeNativeRuntime}=await import('../dist/node-runtime.js'),{buildPending}=await import('../dist/native/values.js');
 const gate=deferred(),entered=deferred();let done=false;
 const runtime=new NodeNativeRuntime({calls:false,services:{owned:{async read(){entered.resolve();await gate.promise;return 9;}}},declarations:{owned:'export function read():Promise<number>;'},agent:async session=>{const result=await session.applyAsync('eval',{code:'void owned.read();return 7;',finish:true});assert.equal(result.kind,'completed',result.text);}});
 const task=runtime.run(buildPending({$lambda:{type:'() => number',instructions:'Return seven.'}})).then(r=>{done=true;return r;});
 try{await Promise.race([entered.promise,task.then(()=>{throw Error('missing Native read');})]);await turn();assert.equal(done,false);gate.resolve();assert.equal((await task).value,7);}
 finally{gate.resolve();await task;runtime.environment.close();runtime.frame.task.close();runtime.frame.task.runtime.close();}
});
for(const mode of ['node','workerd'])test(`${mode} cancellation settles a real rejected Native RPC without inventing completion`,{timeout:15000},async()=>{
 const {currentFrame}=await import('../dist/runtime/context.js');
 const platform=platformOf(mode),entered=deferred(),records=[];let owner,settled=false;
 const calls={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.push(r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const rt=core.createNatlangRuntime({platform,calls,services:{owned:{async read(){owner=currentFrame().task;entered.resolve();try{await new Promise((_resolve,reject)=>owner.signal.addEventListener('abort',()=>reject(owner.signal.reason),{once:true}));}finally{settled=true;}return 9;}}},serviceDeclarations:{owned:'export function read():Promise<number>;'},agent:async session=>{const result=await session.applyAsync('eval',{code:'void owned.read();return 7;',finish:true});if(result.kind!=='completed')throw Error(result.text);}});
 const fn=core.loadVirtualNatlang({'root.nl':'---\nargs: {}\nreturns: number\n---\nReturn seven.'},'root.nl','/test-only-aborted-native',platform);
 const task=rt.run(()=>fn()),rejected=assert.rejects(task,/actual caller cancelled|stopped|cancelled/);
 try{await entered.promise;owner.cancel(Error('actual caller cancelled'));await rejected;assert.equal(settled,true);const row=records.find(r=>r.definition.name==='root');assert.notEqual(row.outcome,'done');assert.match(row.effects[0].error,/actual caller cancelled/);}
 finally{owner?.cancel(Error('test cleanup'));await task.catch(()=>{});rt.close();}
});

test('ordinary generator syntax and synchronous iterator results remain unchanged',async()=>{
 const platform=core.nodePlatform(),rt=core.createNatlangRuntime({platform,calls:false});
 try{const built=core.compileVirtualProject({root:'/test-only-generator-regression',files:{'entry.ts':'export function* sequence():Generator<number>{yield Math.abs(-1);yield 2;}'}},core,{platform});assert.equal(built.ok,true,JSON.stringify(built.diagnostics));await rt.run(()=>assert.deepEqual([...built.require('entry.ts').sequence()],[1,2]));}finally{rt.close();}
});
