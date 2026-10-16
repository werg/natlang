import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {FileNeuraleseStore} from '../../dist/neuralese/node-block-store.js';
import {nativeGraphGradient,nativeStateOptimizerStep} from '../../dist/neuralese/native-learning-client.js';
import {HttpNeuraleseStore,restoreBlocks} from '../../dist/model/neuralese-server.js';
const [endpoint,directory,fixturePath]=process.argv.slice(2),fixture=JSON.parse(await readFile(fixturePath,'utf8'));
let store=await FileNeuraleseStore.open(directory,{residentBytes:0}),ident=fixture.id,history;
const request=id=>({graph:{...fixture.graph,external_states:{prior:{id,token_count:2}}},
  nativeIdentity:fixture.identity,target:{kind:'native-state',id}});
for(let step=1;step<=2;step++){
  const gradient=await nativeGraphGradient(endpoint,store,request(ident));
  const updated=await nativeStateOptimizerStep(endpoint,store,{gradient:gradient.gradient,nativeIdentity:fixture.identity,
    optimizer:{name:'adam',learning_rate:.01},optimizerStateId:history});
  assert.equal(updated.step,step);assert.equal(updated.native_state.metadata.derivation.parent_state_id,ident);
  assert.deepEqual(updated.native_state.metadata.derivation.origin_state_ids,[fixture.id]);
  ident=updated.native_state_id;history=updated.optimizer_state_id;
}
const before=await nativeGraphGradient(endpoint,store,request(ident));
const gc=await fetch(endpoint+'/v1/neuralese/collect',{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({referenced:[]})});
assert.equal(gc.status,200);
store=await FileNeuraleseStore.open(directory,{residentBytes:0});
const after=await nativeGraphGradient(endpoint,store,request(ident));assert.equal(after.loss,before.loss);
assert.equal(after.gradient.id,before.gradient.id);
const restored=await restoreBlocks(new HttpNeuraleseStore(endpoint),store,[history]);assert.deepEqual(restored,[]);
console.log(JSON.stringify({passed:true,restored:true,id:ident,history}));
