import assert from 'node:assert/strict';
import {test} from 'node:test';
import {mkdtempSync,rmSync} from 'node:fs';
import {tmpdir} from 'node:os';
import {join} from 'node:path';
import {setImmediate as tick} from 'node:timers/promises';
import {createNatlangRuntime,loadVirtualNatlang,CallStore} from '../dist/index.js';
import {scriptedModel} from './support/natlang.mjs';
const header='---\nargs: { id: string }\nreturns: { id: string; value: string }\n---\n';
const helper='import { notes } from "natlang:services"; export default async function readNative(id:string):Promise<{id:string;value:string}>{ return notes.read(id); }';
function setup(){const dir=mkdtempSync(join(tmpdir(),'native-service-import-'));const store=CallStore.open(dir);store.writeSettings({minFreeBytes:0});return {store,close(){store.close();rmSync(dir,{recursive:true,force:true});}};}
function root(){return loadVirtualNatlang({'root.nl':header+'Use readNative for the exact owned value.','root/readNative.ts':helper},'root.nl');}
function rows(store){return store.calls().map(row=>store.call(row.call_id));}
test('explicit TS native imports record exact values while the service is hidden from NL tools',async()=>{
 const env=setup();try{
  const notes={prefix:'actual',async read(id){return {id,value:this.prefix};}};
  const rt=createNatlangRuntime({calls:env.store,model:scriptedModel(()=> 'return await readNative(id)').driver,services:{notes},serviceScopes:{notes:[]}});
  assert.deepEqual(await rt.run(()=>root()('owned')),{id:'owned',value:'actual'});
  const record=rows(env.store)[0];assert.equal(record.effects.length,1);assert.equal(record.effects[0].method,'read');
  assert.deepEqual(env.store.value(record.effects[0].args),['owned']);assert.deepEqual(env.store.value(record.effects[0].result),{id:'owned',value:'actual'});
 }finally{env.close();}
});
test('nested and concurrent invocations keep native effects on the correct caller without duplicate wrapping',async()=>{
 const env=setup();try{
  const fn=loadVirtualNatlang({'root.nl':header+'Use child.','root/child.nl':header+'Child reads the owned value.','root/child/readNative.ts':helper},'root.nl');
  let release;const both=new Promise(resolve=>{release=resolve;});let count=0;
  const rt=createNatlangRuntime({calls:env.store,services:{notes:{async read(id){if(++count===2)release();await both;return {id,value:'native'};}}},serviceScopes:{notes:[]},model:scriptedModel(opening=>opening.includes('Child reads')?'return await readNative(id)':'return await child(id)').driver});
  assert.deepEqual(await Promise.all(['one','two'].map(id=>rt.run(()=>fn(id)))),[{id:'one',value:'native'},{id:'two',value:'native'}]);
  const records=rows(env.store);assert.equal(records.length,4);
  for(const record of records){if(record.definition.name==='root'){assert.equal(record.effects.length,0);continue;}assert.equal(record.effects.length,1);const id=env.store.value(record.inputs.id);assert.deepEqual(env.store.value(record.effects[0].args),[id]);assert.equal(env.store.value(record.effects[0].result).id,id);}
 }finally{env.close();}
});
test('imported async native results wait for their exact checkpoint before another model turn',async()=>{
 const env=setup();let release;const durable=new Promise(resolve=>{release=resolve;});let entered;const seen=new Promise(resolve=>{entered=resolve;});let requests=0;
 try{
  const checkpoint=env.store.checkpoint.bind(env.store);env.store.checkpoint=(...args)=>{if(args[1].effects.length){entered();return durable.then(()=>checkpoint(...args));}return checkpoint(...args);};
  const rt=createNatlangRuntime({calls:env.store,services:{notes:{async read(id){return {id,value:'native'};}}},serviceScopes:{notes:[]},model:async()=>({calls:[['eval',{code:++requests===1?'const observed = await readNative(id)':'return observed',finish:requests>1}]]})});
  const pending=rt.run(()=>root()('owned'));await seen;await tick();assert.equal(requests,1);release();assert.deepEqual(await pending,{id:'owned',value:'native'});
 }finally{release();env.close();}
});
test('required imported-effect persistence failures reject the native result at the TS call site',async()=>{
 const env=setup();let requests=0;try{
  env.store.checkpointRequired=true;const checkpoint=env.store.checkpoint.bind(env.store);env.store.checkpoint=(...args)=>args[1].effects.length?Promise.reject(new Error('required native journal unavailable')):checkpoint(...args);
  const fn=loadVirtualNatlang({'root.nl':'---\nargs: { id: string }\nreturns: { error: string }\n---\nObserve the native persistence failure.','root/readNative.ts':helper},'root.nl');
  const rt=createNatlangRuntime({calls:env.store,services:{notes:{async read(id){return {id,value:'native'};}}},serviceScopes:{notes:[]},model:async()=>{requests++;return {calls:[['eval',{code:'try { await readNative(id); return { error: "missing rejection" }; } catch (error) { return { error: String(error) }; }',finish:true}]]};}});
  const result=await rt.run(()=>fn('owned'));assert.match(result.error,/required native journal unavailable/);assert.equal(requests,1);
 }finally{env.close();}
});
