import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { decodeNativeTensorArtifact, encodeNativeTensorArtifact, MemoryNeuraleseStore,
  nativeTensorArtifactDependencies, validateNativeInvocationActorReceipt } from '../../dist/native/neuralese-store.js';
import { saveNz, loadNz } from '../../dist/native/nz-file.js';
import { neuraleseRef } from '../../dist/native/neuralese.js';
const [paths, gradientId] = process.argv.slice(2);
const store = new MemoryNeuraleseStore();
for (const path of JSON.parse(paths)) {
  const raw = new Uint8Array(await readFile(path));
  const artifact = decodeNativeTensorArtifact(raw);
  assert.deepEqual(encodeNativeTensorArtifact(artifact), raw);
  const {id,...input}=artifact;
  assert.equal((await store.putNativeTensorArtifact(input)).id,id);
}
const gradient=await store.getNativeTensorArtifact(gradientId);
const receipt=gradient.metadata.provenance.invocation_actors;
validateNativeInvocationActorReceipt(receipt,gradient.metadata.native_identity);
assert.equal(nativeTensorArtifactDependencies(gradient).length,2);
for (const change of [ {base_native_identity:{...gradient.metadata.native_identity,checkpoint_sha256:'0'.repeat(64)}},
  {invocations:{...receipt.invocations, writer:{...receipt.invocations.writer,native_identity:receipt.invocations.reader.native_identity}}} ])
  assert.throws(()=>validateNativeInvocationActorReceipt({...receipt,...change},gradient.metadata.native_identity));
const bytes=await saveNz({gradient:{type:'Neuralese<unknown>',value:neuraleseRef('Neuralese<unknown>',gradientId)}},
  {store,dialect:'native-tensor/adapter-gradient'});
const portable=new MemoryNeuraleseStore();
await loadNz(bytes,{store:portable});
for (const id of nativeTensorArtifactDependencies(gradient)) {
  const actor=await portable.getNativeTensorArtifact(id);
  assert.ok(actor);
  assert.ok(await portable.getNativeTensorArtifact(actor.metadata.source_artifact_id));
}
