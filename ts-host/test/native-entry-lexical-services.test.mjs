import assert from 'node:assert/strict';
import test from 'node:test';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
import {currentFrame,runInFrame} from '../dist/runtime/context.js';
import {DEFAULT_SETTINGS} from '../dist/calls/types.js';

for(const mode of ['node','workerd'])for(const grant of ['narrow','empty','task-default'])
test(`${mode} authored native entry preserves ${grant} lexical service grants and actual effect recording`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const records=[],blobs=new Map();let forbiddenCalls=0;
 const calls={settings:()=>({...DEFAULT_SETTINGS,specialization:'off'}),record:(record,values)=>{records.push(record);for(const [hash,text]of values)blobs.set(hash,text);},version:()=>0,currentCompilation:()=>undefined,caseServed(){},enqueue(){}};
 const allowed={read:()=>10},forbidden={read:()=>{forbiddenCalls++;return 99;}};
 const runtime=core.createNatlangRuntime({platform,calls,services:{allowed,forbidden}});
 const built=core.compileVirtualProject({root:'/lexical-native',files:{
  'entry.ts':'import {inspect} from "native-test";export function check():{keys:string[];value:number|null;hasForbidden:boolean}{return inspect();}',
  'native-test.d.ts':'declare module "native-test" {export function inspect():{keys:string[];value:number|null;hasForbidden:boolean};}',
 }},core,{platform,modules:{'native-test':{inspect(){
  const services=currentFrame().services;
  const hasForbidden=Object.hasOwn(services,'forbidden');
  if(hasForbidden)services.forbidden.read();
  return{keys:Object.keys(services),value:services.allowed?.read()??null,hasForbidden};
 }}}});
 assert.equal(built.ok,true,JSON.stringify(built.diagnostics));
 const check=built.require('entry.ts').check;
 try{
  const actual=await runtime.run(()=>{
   const frame=currentFrame();
   return runInFrame({...frame,...(grant==='task-default'?{}:{services:grant==='empty'?{}:{allowed}})},()=>check());
  });
  const expectedKeys=grant==='empty'?[]:grant==='narrow'?['allowed']:['allowed','forbidden'];
  assert.deepEqual(actual,{keys:expectedKeys,value:grant==='empty'?null:10,hasForbidden:grant==='task-default'});
  assert.equal(forbiddenCalls,grant==='task-default'?1:0);
  assert.equal(records.length,1);const record=records[0];
  assert.equal(record.executor.kind,'crisp');assert.equal(record.cost.model_requests,0);
  assert.deepEqual(JSON.parse(blobs.get(record.output.hash)),actual);
  assert.deepEqual(record.effects.map(effect=>effect.service).sort(),expectedKeys.slice().sort());
  assert.ok(record.effects.every(effect=>effect.by==='crisp'));
 }finally{runtime.close();}
});
