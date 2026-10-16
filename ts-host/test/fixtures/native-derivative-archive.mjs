import {readFile,writeFile} from 'node:fs/promises';
import assert from 'node:assert/strict';
import {decodeNativeState,encodeNativeState,decodeNativeTensorArtifact,encodeNativeTensorArtifact,
  MemoryNeuraleseStore,archiveNativeArtifactFrom,nativeArtifactDependencyClosure} from '../../dist/native/neuralese-store.js';
import {FileNeuraleseStore} from '../../dist/neuralese/node-block-store.js';
import {OpfsNeuraleseStore} from '../../dist/browser/neuralese-opfs-store.js';
import {fakeOpfs} from './fake-opfs.mjs';
import {neuraleseRef} from '../../dist/native/neuralese.js';
import {saveNz,loadNz,decodeNz} from '../../dist/native/nz-file.js';
import {hexDigest} from '../../dist/native/hash.js';
const [input,output,directory]=process.argv.slice(2),data=JSON.parse(await readFile(input,'utf8'));
const states=new Map(),tensors=new Map(),wires=new Map();
for(const [id,body] of Object.entries(data.artifacts)){
  const wire=new Uint8Array(Buffer.from(body,'base64')),artifact=id.startsWith('nzs1_')?decodeNativeState(wire):decodeNativeTensorArtifact(wire);
  assert.equal(artifact.id,id);wires.set(id,wire);(id.startsWith('nzs1_')?states:tensors).set(id,artifact);
}
const remote={getNativeState:async id=>states.get(id),getNativeTensorArtifact:async id=>tensors.get(id)};
const file=await FileNeuraleseStore.open(directory,{residentBytes:0});
await archiveNativeArtifactFrom(remote,file,data.root);
const closure=await nativeArtifactDependencyClosure(file,[data.root]);
assert(closure.size>5);
const portable=await saveNz({state:{type:'Neuralese<string>',value:neuraleseRef('Neuralese<string>',data.root)}},
  {store:file,dialect:`native-state/${states.get(data.root).metadata.architecture}`});
const parsed=decodeNz(portable);assert.equal(parsed.nativeStates.size+parsed.nativeTensors.size,closure.size);
const memory=new MemoryNeuraleseStore();await loadNz(portable,{store:memory});
const {storage}=fakeOpfs();let browser=await OpfsNeuraleseStore.open({storage,residentBytes:0});
await browser.importNz(portable);browser=await OpfsNeuraleseStore.open({storage,residentBytes:0});
assert.deepEqual(await browser.collect(new Set([data.root])),[]);
assert.deepEqual(await file.collect(new Set([data.root])),[]);
const browserPortable=await browser.exportNz([data.root]);assert.equal(decodeNz(browserPortable).nativeTensors.size,parsed.nativeTensors.size);
const result={};
for(const id of closure){
  for(const store of [file,memory,browser]){
    const artifact=id.startsWith('nzs1_')?await store.getNativeState(id):await store.getNativeTensorArtifact(id);
    const wire=id.startsWith('nzs1_')?encodeNativeState(artifact):encodeNativeTensorArtifact(artifact);
    assert.deepEqual(wire,wires.get(id));result[id]=Buffer.from(wire).toString('base64');
  }
}
const bad=structuredClone(states.get(data.root));delete bad.id;
if(bad.metadata.derivation.kind==='optimizer')bad.metadata.derivation.step+=1;
else bad.metadata.derivation.source_native_identity.checkpoint_sha256='b'.repeat(64);
await assert.rejects(()=>memory.putNativeState(bad),/optimizer|binding|history|source|producer/);
const declaration=[...tensors.values()].find(value=>value.metadata.provenance.schema==='natlang.native-derivative-program/1');
if(declaration){
const omitted=structuredClone(declaration);delete omitted.id;omitted.metadata.provenance.dependencies=[];
assert.throws(()=>encodeNativeTensorArtifact(omitted),/dependency|binding/);
const forward=structuredClone(declaration);delete forward.id;
const program=JSON.parse(forward.metadata.provenance.program_json);program.nodes[0].value=program.nodes.at(-1).name;
forward.metadata.provenance.program_json=JSON.stringify(program);
forward.metadata.provenance.program_sha256=hexDigest(forward.metadata.provenance.program_json);
assert.throws(()=>encodeNativeTensorArtifact(forward),/finite topology/);
const inner=[...tensors.values()].find(value=>value.metadata.provenance.native_derivative_target?.stage==='inner');
assert(inner);const wrongOrigin=structuredClone(inner);delete wrongOrigin.id;
wrongOrigin.metadata.provenance.native_derivative_target.variable='absent';
await assert.rejects(()=>memory.putNativeTensorArtifact(wrongOrigin),/declared inner/);
const wrongGraph=structuredClone(inner);delete wrongGraph.id;
const graph=JSON.parse(wrongGraph.metadata.provenance.graph_json);
graph.nodes[0].token_ids[0]+=1;
wrongGraph.metadata.provenance.graph_json=JSON.stringify(graph);
wrongGraph.metadata.provenance.graph_sha256=hexDigest(wrongGraph.metadata.provenance.graph_json);
await assert.rejects(()=>memory.putNativeTensorArtifact(wrongGraph),/declared objective graph/);
if(inner.metadata.provenance.query_objective_json){
  const wrongQuery=structuredClone(inner);delete wrongQuery.id;
  const objective=JSON.parse(wrongQuery.metadata.provenance.query_objective_json);
  objective.terms[0].weight=-.5;
  wrongQuery.metadata.provenance.query_objective_json=JSON.stringify(objective);
  wrongQuery.metadata.provenance.query_objective_sha256=hexDigest(wrongQuery.metadata.provenance.query_objective_json);
  await assert.rejects(()=>memory.putNativeTensorArtifact(wrongQuery),/declared query objective/);
}
}
await writeFile(output,JSON.stringify(result));console.log(JSON.stringify({passed:true,artifacts:closure.size,states:parsed.nativeStates.size}));
