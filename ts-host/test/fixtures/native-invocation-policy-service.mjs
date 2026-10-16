import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FileNeuraleseStore} from '../../dist/neuralese/node-block-store.js';
import {nativeGraphGradient,nativeAdapterOptimizerStep} from '../../dist/neuralese/native-learning-client.js';
import {HttpNeuraleseStore} from '../../dist/model/neuralese-server.js';
const [endpoint,archive,path]=process.argv.slice(2),fixture=JSON.parse(await readFile(path,'utf8'));
let store=await FileNeuraleseStore.open(archive,{residentBytes:0}),parent=fixture.parent,state,bindings=fixture.bindings;
for(let step=1;step<=2;step++){
  const gradient=await nativeGraphGradient(endpoint,store,{graph:fixture.graph,nativeIdentity:fixture.identity,
    target:{kind:'adapter',id:parent,names:fixture.target_names},adapterGradientPrecision:'fp32',
    invocationAdapters:bindings,objective:fixture.objective,nativeStateModes:fixture.native_state_modes});
  assert.equal(gradient.objective,'native-policy-objective/2');
  assert.equal(gradient.gradient.metadata.provenance.behavior_policy_sha256,fixture.behavior_policy_sha256);
  assert.equal(gradient.gradient.metadata.provenance.effective_native_identity,undefined);
  const updated=await nativeAdapterOptimizerStep(endpoint,store,{gradient:gradient.gradient,nativeIdentity:fixture.identity,
    optimizer:{name:'adam',learning_rate:.01},optimizerStateId:state});
  assert.equal(updated.step,step);
  assert.deepEqual(updated.invocation_source_names,gradient.gradient.metadata.provenance.invocation_source_names);
  bindings=Object.fromEntries(Object.entries(bindings).map(([name,ids])=>[name,ids.map(id=>id===parent?updated.adapter.id:id)]));
  parent=updated.adapter.id;state=updated.optimizer_state_id;
  await new HttpNeuraleseStore(endpoint).collect(new Set());
  store=await FileNeuraleseStore.open(archive,{residentBytes:0});
}
assert.ok(await store.getNativeTensorArtifact(state));
console.log(JSON.stringify({passed:true,parent,state}));
