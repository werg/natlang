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

for(const mode of ['node','workerd']) {
 for(const prefix of ['let handoff=null;handoff=await issueReceipt();','let handoff=null;await Promise.resolve();handoff=await issueReceipt();'])test(`${mode} unannotated null local evolves within its declaring eval: ${prefix}`,async()=>{
  const out=await run(mode,prefix+'return handoff;');
  assert.equal(out.error,undefined,JSON.stringify(out.steps));assert.deepEqual(out.value,receipt);assert.equal(out.effects,1);
 });
 test(`${mode} saved null local may evolve in a later eval`,async()=>{
  const out=await run(mode,'let handoff=null;',{continueCode:'handoff=await issueReceipt();return handoff;'});
  assert.equal(out.error,undefined,JSON.stringify(out.steps));assert.deepEqual(out.value,receipt);assert.equal(out.effects,1);
 });
 test(`${mode} explicit null and stable nonnull local types still refuse incompatible final values`,async()=>{
  for(const code of ['let handoff:null=null;handoff=await issueReceipt();return handoff;','let handoff=0;handoff=await issueReceipt();return handoff;']){
   const out=await run(mode,code);assert.ok(out.error);assert.equal(out.effects,1);
  }
 });
 test(`${mode} unchanged null local keeps its actual null result`,async()=>{
  const out=await run(mode,'let handoff=null;return handoff;');assert.equal(out.error,undefined,JSON.stringify(out.steps));assert.equal(out.value,null);assert.equal(out.effects,0);
 });
}
