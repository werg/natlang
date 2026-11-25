/** Controlled real browser-slot runtime in Node; no page/UI or model inference claim. */
import assert from 'node:assert/strict';
import test from 'node:test';
import * as browser from '../dist/browser/index.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';
const h=(args,returns,body)=>`---\nargs: ${args}\nreturns: '${returns}'\n---\n${body}`;
for(const variant of ['native-callback','escaped-effectful-lambda'])test(`browser slot ${variant} preserves caller observation after await`,async()=>{
 const records=[];let physical=0;const calls={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:r=>records.push(r),version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const runtime=browser.createNatlangRuntime({calls,services:{own:{async read(id){physical++;await Promise.resolve();return id.length;}}},agent:async session=>{
  const name=session.lam.functionName;
  const code=name==='root'?variant==='native-callback'?'return await observe(readOwned);':'const retained=await make();return await consumer(retained);':name==='make'?'return nl`Use the captured native readOwned on id.`;':name==='observe'?'const a=await read("before");await Promise.resolve();return a+await read("after");':name==='consumer'?'return await fn("later");':'return await readOwned(id);';
  const result=await session.applyAsync('eval',{code,finish:true});assert.equal(result.kind,'completed',result.text);
 }});
 const files={'root.nl':h('{}','number','Use owned read callbacks.'),'root/observe.nl':h('{read:"(id:string) => number"}','number','Inspect the actual owned values.'),'root/readOwned.ts':'import {own} from "natlang:services";export default async function readOwned(id:string):Promise<number>{return await own.read(id);}','root/consumer.nl':h('{fn:"(id:string) => number"}','number','Use the retained function.'),'root/make.nl':h('{}','(id:string) => number','Retain an owned native reader.'),'root/make/readOwned.ts':'import {own} from "natlang:services";export default async function readOwned(id:string):Promise<number>{return await own.read(id);}'};
 const root=browser.loadVirtualNatlang(files,'root.nl','/browser-slot-native-callback');
 try{
  // Browser host callbacks need no asynchronous ambient propagation here: the
  // actual language program's compiled awaits restore its slot throughout.
  const value=await runtime.run(()=>root());
  if(variant==='native-callback'){assert.equal(value,11);const observer=records.find(r=>r.definition.name==='observe');assert.equal(observer.effects.length,2);assert.equal(physical,2);}
  else{assert.equal(value,5);const consumed=records.find(r=>r.definition.name==='consumer'),child=records.find(r=>r.parent_call_id===consumed.call_id);assert.equal(child.effects.length,1);assert.equal(physical,1);assert.equal(records.find(r=>r.definition.name==='make').effects.length,0);}
  assert.equal(records.find(r=>r.definition.name==='root').effects.length,0);
 }finally{runtime.close();}
});
