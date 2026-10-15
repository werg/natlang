/** HTTP/archive contract controls; numerical behavior is covered by actual Python ports. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { MemoryNeuraleseStore, encodeNativeTensorArtifact, decodeNativeTensorArtifact } from '../dist/native/neuralese-store.js';
import { nativeGraphGradient, nativeOptimizerStep } from '../dist/neuralese/native-learning-client.js';

test('native learning client preserves named backbone artifacts across server/store restart and identity key order', async () => {
  const remote = new Map(), store = new MemoryNeuraleseStore();
  const identity = { checkpoint_sha256:'1'.repeat(64),tokenizer_fingerprint:'2'.repeat(64),embedding_weight_sha256:'3'.repeat(64),architecture:'lfm2' };
  const shuffled = { architecture:identity.architecture,embedding_weight_sha256:identity.embedding_weight_sha256,
    tokenizer_fingerprint:identity.tokenizer_fingerprint,checkpoint_sha256:identity.checkpoint_sha256 };
  const name = 'port.hf.model.layers.0.self_attn.q_proj.weight';
  const artifact = (kind,source=null) => decodeNativeTensorArtifact(encodeNativeTensorArtifact({
    metadata:{schema:'natlang.neuralese-native-tensor-artifact/1',kind,native_identity:identity,source_artifact_id:source,
      provenance:{contract_fixture:true},tensors:[{name,dtype:'F32',shape:[1]}]},
    tensors:{[name]:{dtype:'F32',shape:[1],data:new Uint8Array(new Float32Array([1]).buffer)}} }));
  const gradient = artifact('parameter-gradient'), parameters = artifact('parameter-update',gradient.id), state = artifact('optimizer-state',parameters.id);
  remote.set(gradient.id,gradient); remote.set(parameters.id,parameters); remote.set(state.id,state);
  let requests = 0, corruptCheckpoint = false;
  const server = createServer(async (req,res) => {
    const chunks=[]; for await(const chunk of req) chunks.push(chunk);
    const raw=Buffer.concat(chunks), id=req.url.split('/').at(-1);
    if(req.url.includes('/native-tensor-artifacts/')) {
      if(req.method==='PUT') { const value=decodeNativeTensorArtifact(raw);remote.set(value.id,value);res.end(JSON.stringify({id:value.id}));return; }
      if(!remote.has(id)) {res.statusCode=404;res.end();return;}
      if(req.method==='GET')res.end(encodeNativeTensorArtifact(remote.get(id)));else res.end();
      return;
    }
    const body=JSON.parse(raw);
    if(req.url.endsWith('/grad')) {res.end(JSON.stringify({loss:1,stats:{},gradient:{id:gradient.id},native_identity:shuffled,objective:'native-graph-token-ce/1'}));return;}
    assert.equal(body.gradient_id,gradient.id);assert.equal(body.optimizer.momentum,.8);
    assert.ok(remote.has(gradient.id));assert.ok(remote.has(state.id));requests++;
    res.end(JSON.stringify({checkpoint:{path:'/fixture/native.json',sha256:'4'.repeat(64),schema:'natlang.neuralese-native-composite-checkpoint/1',native_identity:{...identity,checkpoint_sha256:'4'.repeat(64)}},
      parameters:{id:parameters.id},optimizer_state_id:state.id,native_identity:{...identity,checkpoint_sha256:(corruptCheckpoint?'5':'4').repeat(64)},step:2}));
  });
  await new Promise(resolve=>server.listen(0,'127.0.0.1',resolve));
  const endpoint=`http://127.0.0.1:${server.address().port}`;
  try {
    const value=await nativeGraphGradient(endpoint,store,{graph:{schema:'natlang.neuralese-native-graph/1',split:'train'},target:{kind:'model-parameters',names:[name]},nativeIdentity:identity});
    assert.equal(value.gradient.id,gradient.id);
    await store.putNativeTensorArtifact(state);
    remote.delete(gradient.id);remote.delete(state.id); // restarted serving store
    const updated=await nativeOptimizerStep(endpoint,store,{gradient:value.gradient,nativeIdentity:shuffled,
      optimizer:{name:'sgd',learning_rate:.01,momentum:.8},optimizerStateId:state.id});
    assert.equal(requests,1);assert.equal(updated.parameters.id,parameters.id);
    assert.equal((await store.getNativeTensorArtifact(state.id)).id,state.id);
    assert.equal((await store.getNativeTensorArtifact(parameters.id)).metadata.tensors[0].name,name);
    corruptCheckpoint = true;
    await assert.rejects(nativeOptimizerStep(endpoint,store,{gradient:value.gradient,nativeIdentity:shuffled,
      optimizer:{name:'sgd',momentum:.8},optimizerStateId:state.id}),/checkpoint identity differs/);
  } finally {await new Promise(resolve=>server.close(resolve));}
});
