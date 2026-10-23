import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { MemoryNeuraleseStore } from '../../dist/native/neuralese-store.js';
import { nativeAdapterCreate, nativeAdapterOptimizerStep, nativeGraphGradient } from '../../dist/neuralese/native-learning-client.js';
import { restoreNativeTensorArtifacts } from '../../dist/neuralese/native-tensor-client.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { neuraleseRef } from '../../dist/native/neuralese.js';
import { withAdapters } from '../../dist/neuralese/learning.js';
import { nativeTextOperation } from '../../dist/model/native-state-client.js';
import { saveNz, loadNz } from '../../dist/native/nz-file.js';
const [endpoint, archive, fixturePath] = process.argv.slice(2);
const fixture=JSON.parse(await readFile(fixturePath,'utf8'));
const info=await (await fetch(endpoint+'/v1/neuralese/info')).json();
assert.ok(info.capabilities.includes('native.adapters'));
assert.equal(info.adapters,undefined);
assert.equal(info.grad_order,1);
assert.ok(info.native_state.consumption_modes.includes('same-base-adapter-state'));
assert.equal(info.projections,undefined);
const store=await FileNeuraleseStore.open(archive,{residentBytes:0});
let adapter=await nativeAdapterCreate(endpoint,store,{nativeIdentity:fixture.identity,
  parameterNames:[fixture.target],rank:2});
const driver=neuraleseServerModelTurn({endpoint,model:'native-fixture',store});
const request={messages:[{role:'user',content:'t4 t5'}],options:['t6 t8','t7 t9']};
const base=await driver.decide(request);
const zero=await withAdapters(neuraleseRef('Adapter',adapter.id),()=>driver.decide(request));
assert.deepEqual(zero.log_probs,base.log_probs);
assert.deepEqual(zero.tokens,base.tokens);
assert.ok(base.tokens.every(count=>count>0),'decision fixture must score actual reply tokens');
let state;
for(let step=1;step<=2;step++){
  const gradient=await nativeGraphGradient(endpoint,store,{graph:fixture.graph,nativeIdentity:fixture.identity,
    target:{kind:'adapter',id:adapter.id},adapters:[adapter.id]});
  const updated=await nativeAdapterOptimizerStep(endpoint,store,{gradient:gradient.gradient,
    nativeIdentity:fixture.identity,optimizer:{name:'adam',learning_rate:.2},optimizerStateId:state});
  assert.equal(updated.step,step);
  assert.notEqual(updated.adapter.id,adapter.id);
  adapter=updated.adapter;state=updated.optimizer_state_id;
}
const adapted=await withAdapters(neuraleseRef('Adapter',adapter.id),()=>driver.decide(request));
assert.notDeepEqual(adapted.log_probs,base.log_probs);
const written=await nativeTextOperation({endpoint,store},'write-text',{text:'t5 t6',adapters:[adapter.id]});
const bytes=await saveNz({memo:{type:'Neuralese<string>',value:neuraleseRef('Neuralese<string>',written.result.id)},adapter:{type:'Adapter',value:neuraleseRef('Adapter',adapter.id)},
  optimizer:{type:'Neuralese<unknown>',value:neuraleseRef('Neuralese<unknown>',state)}},
  {store,dialect:'native-tensor/adapter-coefficients'});
const portable=new MemoryNeuraleseStore();
const loaded=await loadNz(bytes,{store:portable});
assert.equal(loaded.exports.adapter.$neuralese.id,adapter.id);
assert.equal(loaded.nativeStates.size,1);
assert.ok(loaded.nativeTensors.size>=3,'coefficients, exact bases, optimizer continuation travel together');
await portable.collect(new Set([adapter.id,state]));
assert.ok(await portable.has(adapter.metadata.source_artifact_id));
// Drop remote data, reopen the durable archive and restore from the portable file.
await driver.blocks.collect(new Set());
assert.equal(await driver.blocks.has(adapter.id),false);
assert.deepEqual(await restoreNativeTensorArtifacts(endpoint,portable,[adapter.id,state]),[]);
const reopened=await FileNeuraleseStore.open(archive,{residentBytes:0});
const again=neuraleseServerModelTurn({endpoint,model:'native-fixture',store:reopened});
const restored=await withAdapters(neuraleseRef('Adapter',adapter.id),()=>again.decide(request));
assert.deepEqual(restored.log_probs,adapted.log_probs);
assert.deepEqual(restored.tokens,adapted.tokens);
assert.deepEqual((await again.decide(request)).log_probs,base.log_probs,'adaptation remains scoped after reload');
await withAdapters(neuraleseRef('Adapter',adapter.id),()=>again({messages:request.messages,tools:[],max_tokens:1}));
console.log(JSON.stringify({adapter:adapter.id,state,portable:loaded.nativeTensors.size,restored:true}));
