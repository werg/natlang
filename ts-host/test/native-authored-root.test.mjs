import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
const files={
 'entry.ts':'import describe from "./describe.nl"; export async function summarize(value:number):Promise<{description:string;doubled:number}>{return {description:await describe(value),doubled:value*2};} export function twice(value:number):number{return value*2;} export function fail(value:string):never{throw new Error(value);}',
 'describe.nl':'---\nargs: { value: number }\nreturns: string\n---\nDescribe the number.',
};
function setup(platform=core.nodePlatform(),recording){
 const dir=mkdtempSync(join(tmpdir(),'native-entry-actual-')),store=core.CallStore.open(dir);
 store.writeSettings({minFreeBytes:0,specialization:'off'});
 const rt=core.createNatlangRuntime({platform,calls:store,recording,model:async()=>({calls:[['return_result',{status:'success',value:'seven'}]]})});
 const compiled=core.compileVirtualProject({files,root:'/reference'},core,{platform});
 assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
 return {rt,store,platform,module:compiled.require('entry.ts'),close(){rt.close();store.close();rmSync(dir,{recursive:true,force:true});}};
}
for(const mode of ['node','workerd'])test(`actual ${mode} compiled TS root owns typed NL child and exact return`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const env=setup(platform);
 try{
  assert.deepEqual(await env.rt.run(()=>env.module.summarize(7)),{description:'seven',doubled:14});
  const rows=env.store.calls().map(row=>env.store.call(row.call_id));
  const root=rows.find(row=>row.parent_call_id===null),child=rows.find(row=>row.definition.name==='describe');
  assert.equal(root.definition.name,'summarize');assert.equal(root.executor.kind,'crisp');assert.equal(root.outcome,'done');
  assert.deepEqual(root.definition.params,[{name:'value',type:'number'}]);
  assert.equal(env.store.value(root.inputs.value),7);assert.deepEqual(env.store.value(root.output),{description:'seven',doubled:14});
  assert.equal(child.parent_call_id,root.call_id);assert.ok(env.store.events(child.call_id).some(event=>event.kind==='model_request')); 
  assert.deepEqual(root.graph_custody.child_call_ids,[child.call_id]);
 }finally{env.close();}
});
test('native entry preserves synchronous return and failure semantics',async()=>{
 const env=setup();try{
  await env.rt.run(()=>{assert.equal(env.module.twice(4),8);assert.throws(()=>env.module.fail('actual failure'),/actual failure/);});
  const rows=env.store.calls().map(row=>env.store.call(row.call_id));
  assert.equal(rows.find(row=>row.definition.name==='twice').outcome,'done');
  const failure=rows.find(row=>row.definition.name==='fail');assert.equal(failure.outcome,'error');assert.match(failure.detail,/actual failure/);assert.equal(failure.output,null);
 }finally{env.close();}
});
test('callable-folder TS root and compiled TS root use the same tracing boundary',async()=>{
 const env=setup();try{
  const folder=core.loadVirtualCallables(files,'','/reference',env.platform);
  assert.equal(await env.rt.run(()=>folder.entry.twice(3)),6);
  assert.equal(env.store.calls().length,1);assert.equal(env.store.call(env.store.calls()[0].call_id).definition.name,'twice');
 }finally{env.close();}
});
test('native root capture has an explicit opt-out',async()=>{
 const env=setup(core.nodePlatform(),{nativeRoots:false});try{
  await env.rt.run(()=>env.module.summarize(7));
  assert.deepEqual(env.store.calls().map(row=>env.store.call(row.call_id).definition.name),['describe']);
 }finally{env.close();}
});
test('already compiled inferred TS and re-exported functions retain behavior and are captured',async()=>{
 const env=setup();try{
  const compiled=core.compileVirtualProject({files:{'main.ts':'export { triple } from "./helper.js"; export const constant=3;','helper.ts':'export function triple(value:number){return value*3;}'},root:'/loose'},core,{platform:env.platform,constrained:false});
  assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
  const module=compiled.require('main.ts');assert.equal(module.constant,3);
  assert.equal(await env.rt.run(()=>module.triple(3)),9);
  const record=env.store.call(env.store.calls()[0].call_id);assert.equal(record.definition.name,'triple');assert.equal(env.store.value(record.inputs.arg0),3);assert.equal(env.store.value(record.output),9);
 }finally{env.close();}
});
