import test from 'node:test';
import assert from 'node:assert/strict';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import * as core from '../dist/index.js';
import {workerdPlatform} from '../dist/core/workerd.js';
const files={
 'entry.ts':'import describe from "./describe.nl"; export async function summarize(value:number):Promise<{description:string;doubled:number}>{return {description:await describe(value),doubled:value*2};} export async function withOptional(value:number,label?:string):Promise<string>{return (label??"prefix")+await describe(value);} export async function restful(prefix:string,...values:number[]):Promise<string>{return prefix+values.join(",")+":"+await describe(values.length);} export function twice(value:number):number{return value*2;} export function fail(value:string):never{throw new Error(value);}',
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
for(const mode of ['node','workerd'])test(`actual ${mode} TS optional input stays optional in replay identity`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const env=setup(platform);try{
  assert.equal(await env.rt.run(()=>env.module.withOptional(7)),'prefixseven');
  const rows=env.store.calls().map(row=>env.store.call(row.call_id));
  const root=rows.find(row=>row.parent_call_id===null),child=rows.find(row=>row.definition.name==='describe');
  assert.equal(root.definition.id,'ts:entry.ts#withOptional');
  assert.deepEqual(root.definition.params.map(({name,optional})=>({name,...(optional?{optional}: {})})),[{name:'value'},{name:'label',optional:true}]);
  assert.equal(Object.hasOwn(root.inputs,'label'),false);
  assert.equal(env.store.value(root.inputs.value),7);
  assert.equal(child.parent_call_id,root.call_id);
  assert.deepEqual(root.graph_custody.child_call_ids,[child.call_id]);
 }finally{env.close();}
});
for(const mode of ['node','workerd'])test(`actual ${mode} TS rest captures every trailing value and keeps child lineage`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const env=setup(platform);try{
  assert.equal(await env.rt.run(()=>env.module.restful('p:',1,2,3)),'p:1,2,3:seven');
  const folder=core.loadVirtualCallables(files,'','/reference',platform);
  assert.equal(await env.rt.run(()=>folder.entry.restful('empty:')),'empty::seven');
  const rows=env.store.calls().map(row=>env.store.call(row.call_id));
  const roots=rows.filter(row=>row.parent_call_id===null);assert.equal(roots.length,2);
  for(const root of roots){
   assert.equal(root.definition.params[1].name,'values');assert.equal(root.definition.params[1].rest,true);
   const prefix=env.store.value(root.inputs.prefix);
   assert.deepEqual(env.store.value(root.inputs.values),prefix==='p:'?[1,2,3]:[]);
   const child=rows.find(row=>row.parent_call_id===root.call_id);assert.equal(child.definition.name,'describe');
   assert.deepEqual(root.graph_custody.child_call_ids,[child.call_id]);
  }
 }finally{env.close();}
});
test('NL eval sees the actual rest signature of a TS helper',async()=>{
 const dir=mkdtempSync(join(tmpdir(),'native-rest-scope-')),store=core.CallStore.open(dir);
 store.writeSettings({minFreeBytes:0,specialization:'off'});
 const platform=core.nodePlatform(),rt=core.createNatlangRuntime({platform,calls:store,model:async()=>({calls:[['eval',{code:'return helper.join("p:",1,2,3);',finish:true}]]})});
 const program={'main.nl':'---\nargs: {}\nreturns: string\n---\nJoin the supplied values using helper.',
 'main/helper.ts':'export function join(prefix:string,...values:number[]):string{return prefix+values.join(",");}'};
 try{const main=core.loadVirtualNatlang(program,'main.nl','/reference',platform);assert.equal(await rt.run(()=>main()),'p:1,2,3');}
 finally{rt.close();store.close();rmSync(dir,{recursive:true,force:true});}
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

for(const mode of ['node','workerd'])test(`overlapping ${mode} native roots preserve reverse-completing children and exact effect ownership`,async()=>{
 const platform=mode==='node'?core.nodePlatform():workerdPlatform({realm:{call(body,params,args){return Function(...params,body)(...args);}}});
 const env=setup(platform),gates=[],seen=[];
 let both;const entered=new Promise(resolve=>{both=resolve;});
 const services={journal:{async note(id,phase){seen.push({id,phase});return `${id}:${phase}`;}}};
 const rt=core.createNatlangRuntime({platform,calls:env.store,services,model:async()=>{
  await new Promise(resolve=>{gates.push(resolve);if(gates.length===2)both();});
  return {calls:[['eval',{code:'await journal.note(id,"child"); return id + "-checked";',finish:true}]]};
 }});
 const sources={
  'entry.ts':'import { journal } from "natlang:services"; import describe from "./describe.nl"; export async function run(id:string):Promise<{id:string;description:string}>{ await journal.note(id,"before"); const description=await describe(id); await journal.note(id,"after"); return {id,description}; }',
  'describe.nl':'---\nargs: { id: string }\nreturns: string\n---\nInspect the given id using journal and describe it.',
  'services.d.ts':'declare module "natlang:services" { export const journal: { note(id:string,phase:string):Promise<string> }; }',
 };
 const compiled=core.compileVirtualProject({files:sources,root:'/concurrent'},core,{platform});assert.equal(compiled.ok,true,JSON.stringify(compiled.diagnostics));
 const module=compiled.require('entry.ts');
 try{
  const first=rt.run(()=>module.run('first')),second=rt.run(()=>module.run('second'));
  await entered;gates[1]();assert.deepEqual(await second,{id:'second',description:'second-checked'});
  gates[0]();assert.deepEqual(await first,{id:'first',description:'first-checked'});
  assert.deepEqual(seen,[{id:'first',phase:'before'},{id:'second',phase:'before'},{id:'second',phase:'child'},{id:'second',phase:'after'},{id:'first',phase:'child'},{id:'first',phase:'after'}]);
  const records=env.store.calls().map(row=>env.store.call(row.call_id)),roots=records.filter(row=>row.parent_call_id===null);
  assert.equal(records.length,4);assert.equal(roots.length,2);
  for(const root of roots){
   const id=env.store.value(root.inputs.id),children=records.filter(row=>row.parent_call_id===root.call_id);
   assert.equal(root.definition.name,'run');assert.deepEqual(env.store.value(root.output),{id,description:id+'-checked'});
   assert.equal(children.length,1);assert.equal(children[0].definition.name,'describe');assert.equal(env.store.value(children[0].inputs.id),id);
   assert.deepEqual(root.effects.map(effect=>env.store.value(effect.args)),[[id,'before'],[id,'after']]);
   assert.deepEqual(children[0].effects.map(effect=>env.store.value(effect.args)),[[id,'child']]);
   assert.deepEqual(root.graph_custody.child_call_ids,[children[0].call_id]);
  }
 }finally{for(const release of gates)release();rt.close();env.close();}
});
