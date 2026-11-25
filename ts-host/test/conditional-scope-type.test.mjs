import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {formatType} from '../dist/native/types.js';

const receipt={path:'native-receipt',allowed:false,count:0,note:''};
const recordType='{path:string;allowed:boolean;count:number;note:string}';
async function run(mode,code,{flag=true,returns='Receipt | null',badReceipt=false,badPredicate=false,continueCode}={}){
 let effects=0;const steps=[],localTypes=[];
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const declarations={ports:`export type Receipt=${recordType}; export function stage():Promise<Receipt>;`};
 const runtime=core.createNatlangRuntime({platform,calls:false,services:{ports:{async stage(){effects++;return badReceipt?'wrong receipt':receipt;}}},serviceDeclarations:declarations,serviceScopes:{ports:['root/issueReceipt.ts']},agent:async session=>{
  const child=session.lam.functionName==='predicate';
  const result=await session.applyAsync('eval',{code:child?(badPredicate?'return "wrong boolean";':'return flag;'):code,finish:child||!continueCode});
  steps.push(result);
  if(!child){localTypes.push(Object.fromEntries(Object.entries(session.lam.letTypes).map(([name,type])=>[name,formatType(type)])));
   if(continueCode&&!['error','rejected','completed'].includes(result.kind))steps.push(await session.applyAsync('eval',{code:continueCode,finish:true}));
  }
 }});
 try{
  const compiled=core.compileVirtualProject({root:'/conditional-bindings',files:{
   'types.ts':`export type Receipt=${recordType};`,
   'main.ts':'import root from "./root.nl"; export default async function main(flag:boolean){return root(flag);}',
   'root.nl':`---\nargs: {flag: boolean}\nreturns: ${returns}\n---\nReturn actual requested native receipt or the supported negative result.\n`,
   'root/predicate.nl':'---\nargs: {flag: boolean}\nreturns: boolean\n---\nReturn the actual supplied flag unchanged.\n',
   'root/issueReceipt.ts':'import {ports} from "natlang:services"; import type {Receipt} from "../types.js"; export default async function issueReceipt():Promise<Receipt>{return ports.stage();}',
  }},core,{platform,programId:'conditional-binding-program',declarations});
  assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
  try{return {value:await runtime.run(()=>compiled.require('main.ts').default(flag)),effects,steps,localTypes};}
  catch(error){return {error,effects,steps,localTypes};}
 }finally{runtime.close();}
}
for(const mode of ['node','workerd']){
 for(const flag of [true,false])test(`${mode} conditional local keeps receipt|null rather than predicate boolean (${flag})`,async()=>{
  const out=await run(mode,'const handoff=await predicate(flag)?await issueReceipt():null;return handoff;',{flag});
  assert.equal(out.error,undefined,String(out.error));assert.deepEqual(out.value,flag?receipt:null);assert.equal(out.effects,flag?1:0);
  assert.match(out.localTypes[0].handoff,/null/);assert.doesNotMatch(out.localTypes[0].handoff,/^boolean$/);
 });
 for(const fallback of ['false','0'])test(`${mode} conditional preserves actual ${fallback} and all falsey native receipt fields`,async()=>{
  for(const flag of [true,false]){const out=await run(mode,`const handoff=await predicate(flag)?await issueReceipt():${fallback};return handoff;`,{flag,returns:'Receipt | boolean | number'});
   assert.equal(out.error,undefined,String(out.error));assert.deepEqual(out.value,flag?receipt:fallback==='false'?false:0);assert.equal(out.effects,flag?1:0);}
 });
 test(`${mode} a retained conditional local can subsequently hold its other actual branch type`,async()=>{
  const out=await run(mode,'let handoff=await predicate(flag)?await issueReceipt():null;',{continueCode:'handoff=null;return handoff;'});
  assert.equal(out.error,undefined,String(out.error));assert.equal(out.value,null);assert.equal(out.effects,1);
 });
 test(`${mode} logical operands and projected calls do not supply the whole initializer type`,async()=>{
  const projected=await run(mode,'const path=(await issueReceipt()).path;return path;',{returns:'string'});
  assert.equal(projected.error,undefined,String(projected.error));assert.equal(projected.value,receipt.path);
  for(const flag of [true,false]){const logical=await run(mode,'const handoff=await predicate(flag)&&await issueReceipt();return handoff;',{flag,returns:'Receipt | boolean'});
   assert.equal(logical.error,undefined,String(logical.error));assert.deepEqual(logical.value,flag?receipt:false);assert.equal(logical.effects,flag?1:0);}
 });
 test(`${mode} explicit incompatible annotations and bad actual callee returns remain rejected`,async()=>{
  const annotated=await run(mode,'const handoff:boolean=await predicate(flag)?await issueReceipt():null;return handoff;');
  assert.ok(annotated.error);assert.equal(annotated.effects,1); // Runtime annotation validation follows the actual expression; its completed effect is not undone.
  const bad=await run(mode,'const handoff=await predicate(flag)?await issueReceipt():null;return handoff;',{badReceipt:true});assert.ok(bad.error);assert.equal(bad.effects,1);
  const predicate=await run(mode,'const handoff=await predicate(flag)?await issueReceipt():null;return handoff;',{badPredicate:true});assert.ok(predicate.error);assert.equal(predicate.effects,0);
 });
}
