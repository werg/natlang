import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {decodeNativeTensorArtifact,encodeNativeTensorArtifact,MemoryNeuraleseStore,nativeTensorArtifactDependencies,
  validateNativeInvocationBehaviorPolicy} from '../../dist/native/neuralese-store.js';
import {saveNz,loadNz} from '../../dist/native/nz-file.js';
import {nativeTensorRef} from '../../dist/native/neuralese.js';
const [paths,gradientId,stateId]=process.argv.slice(2);
const store=new MemoryNeuraleseStore();
for(const path of JSON.parse(paths)){
  const raw=new Uint8Array(await readFile(path));
  const artifact=decodeNativeTensorArtifact(raw);
  assert.deepEqual(encodeNativeTensorArtifact(artifact),raw);
  const {id,...input}=artifact;
  assert.equal((await store.putNativeTensorArtifact(input)).id,id);
}
const gradient=await store.getNativeTensorArtifact(gradientId);
const policy=gradient.metadata.provenance.behavior_policy;
validateNativeInvocationBehaviorPolicy(policy,gradient.metadata.native_identity);
assert.throws(()=>validateNativeInvocationBehaviorPolicy({...policy,collector_policy_role:'effective-model'},gradient.metadata.native_identity));
const forged={...gradient,canonicalMetadata:undefined,metadata:{...gradient.metadata,provenance:{...gradient.metadata.provenance,behavior_policy_sha256:'0'.repeat(64)}}};
assert.throws(()=>encodeNativeTensorArtifact(forged),/descriptor digest/);
const bytes=await saveNz({gradient:{type:'NativeTensor',value:nativeTensorRef(gradientId,'adapter-gradient')},
  state:{type:'NativeTensor',value:nativeTensorRef(stateId,'optimizer-state')}},{store,dialect:'native-tensor/adapter-gradient'});
const restored=new MemoryNeuraleseStore();await loadNz(bytes,{store:restored});
const pending=[gradientId,stateId],seen=new Set();
while(pending.length){
  const id=pending.pop();if(seen.has(id))continue;seen.add(id);
  const artifact=await restored.getNativeTensorArtifact(id);assert.ok(artifact,id);
  pending.push(...nativeTensorArtifactDependencies(artifact));
}
assert.ok(seen.size>4);
console.log(JSON.stringify({passed:true,dependencies:seen.size}));
