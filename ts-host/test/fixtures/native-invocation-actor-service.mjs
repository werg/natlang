import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FileNeuraleseStore} from '../../dist/neuralese/node-block-store.js';
import {nativeGraphGradient,nativeAdapterOptimizerStep} from '../../dist/neuralese/native-learning-client.js';
import {archiveNativeTensorArtifact} from '../../dist/neuralese/native-tensor-client.js';
import {HttpNeuraleseStore} from '../../dist/model/neuralese-server.js';
const [endpoint,archive,fixturePath]=process.argv.slice(2);
const fixture=JSON.parse(await readFile(fixturePath,'utf8'));
let store=await FileNeuraleseStore.open(archive,{residentBytes:0});
let bindings=fixture.bindings;
let parent=fixture.source,state;
const baseRequest={messages:[{role:'user',content:'t4 t5'}],options:['t7','t8']};
const scoreBase=async()=>{
  const response=await fetch(endpoint+'/v1/neuralese/decide',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(baseRequest)});
  assert.equal(response.status,200);
  return (await response.json()).log_probs;
};
const base=await scoreBase();
await assert.rejects(nativeGraphGradient(endpoint,store,{graph:fixture.graph,nativeIdentity:fixture.identity,
  target:{kind:'adapter',id:parent,names:fixture.target_names},invocationAdapters:{...bindings,missing:[parent]}}),/actual invocation names/);
await assert.rejects(nativeGraphGradient(endpoint,store,{graph:{...fixture.graph,split:'test'},nativeIdentity:fixture.identity,
  target:{kind:'adapter',id:parent,names:fixture.target_names},invocationAdapters:bindings}),/train rows/);
for(let step=1;step<=2;step++){
  const gradient=await nativeGraphGradient(endpoint,store,{graph:fixture.graph,nativeIdentity:fixture.identity,
    target:{kind:'adapter',id:parent,names:fixture.target_names},adapterGradientPrecision:'fp32',invocationAdapters:bindings});
  assert.equal(gradient.gradient.metadata.provenance.effective_native_identity,undefined);
  assert.equal(gradient.gradient.metadata.provenance.adapter_binding,undefined);
  assert.ok(Object.values(gradient.gradient.tensors).some(tensor => {
    const view=new DataView(tensor.data.buffer,tensor.data.byteOffset,tensor.data.byteLength);
    for(let offset=0;offset<view.byteLength;offset+=4) if(view.getFloat32(offset,true)!==0) return true;
    return false;
  }),'consumer CE must reach a producer value-projection coefficient');
  const before=gradient.gradient.metadata.provenance.invocation_actors;
  assert.notEqual(before.invocations.producer.native_identity.checkpoint_sha256,before.invocations.reader.native_identity.checkpoint_sha256);
  const updated=await nativeAdapterOptimizerStep(endpoint,store,{gradient:gradient.gradient,nativeIdentity:fixture.identity,
    optimizer:{name:'adam',learning_rate:.01},optimizerStateId:state});
  assert.equal(updated.step,step);
  assert.equal(updated.adapter_binding,undefined);
  assert.equal(updated.effective_native_identity,undefined);
  assert.notEqual(updated.adapter.id,parent);
  assert.deepEqual(updated.invocation_actors.invocations.reader,before.invocations.reader);
  assert.equal(updated.invocation_actors.invocations.producer.adapter_binding.adapters[0].id,updated.adapter.id);
  assert.equal(updated.invocation_actors.invocations.peer.adapter_binding.adapters[0].id,updated.adapter.id);
  assert.notEqual(updated.invocation_actors.invocations.producer.native_identity.checkpoint_sha256,before.invocations.producer.native_identity.checkpoint_sha256);
  const prior=await store.getNativeTensorArtifact(parent);
  assert.notDeepEqual(updated.adapter.tensors,prior.tensors);
  bindings=Object.fromEntries(Object.entries(bindings).map(([name,entries])=>[name,entries.map(id=>id===parent?updated.adapter.id:id)]));
  parent=updated.adapter.id;state=updated.optimizer_state_id;
  await archiveNativeTensorArtifact(endpoint,store,fixture.target);
  await new HttpNeuraleseStore(endpoint).collect(new Set());
  store=await FileNeuraleseStore.open(archive,{residentBytes:0});
  assert.deepEqual(await scoreBase(),base,'graph actor scopes must restore the unadapted live model');
}
console.log(JSON.stringify({passed:true,parent,state}));
