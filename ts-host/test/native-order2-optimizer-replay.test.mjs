import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import test from 'node:test';
import { createLearning, learningService } from '../dist/neuralese/learning.js';
import { activeNativeAdapters, activeRecorder } from '../dist/native/model-recording.js';
import { MemoryNeuraleseStore, encodeNativeTensorArtifact, decodeNativeTensorArtifact, nativeCanonicalJson } from '../dist/native/neuralese-store.js';
import { nativeAdapterRef } from '../dist/native/neuralese.js';

const identity = { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
  embedding_weight_sha256: '3'.repeat(64), architecture: 'fixture-native' };
const tensor = values => ({ dtype: 'F32', shape: [values.length], data: new Uint8Array(new Float32Array(values).buffer) });

function artifact(kind, source, name, values, provenance) {
  return decodeNativeTensorArtifact(encodeNativeTensorArtifact({ metadata: {
    schema: 'natlang.neuralese-native-tensor-artifact/1', kind, native_identity: identity,
    source_artifact_id: source, provenance, tensors: [{ name, dtype: 'F32', shape: [values.length] }],
  }, tensors: { [name]: tensor(values) } }));
}

test('order-two SGD replay records momentum in its optimizer node', async () => {
  const name = 'fixture.weight';
  const adapterSpec = { adapter_spec: { kind: 'xs', rank: 1, dim: 1, parameter_names: [name] } };
  const basis = artifact('adapter-basis', null, `left.${name}`, [1], adapterSpec);
  const initial = artifact('adapter-coefficients', basis.id, name, [0.25], adapterSpec);
  const gradient = artifact('adapter-gradient', initial.id, name, [0.5], { fixture: 'inner-gradient' });
  const updated = artifact('adapter-coefficients', basis.id, name, [0.2], adapterSpec);
  const optimizerState = artifact('optimizer-state', updated.id, name, [0.5], {
    gradient_id: gradient.id, parent_adapter_id: initial.id, step: 1,
  });
  const adapterBinding = { schema: 'natlang.native-adapter/1', base_native_identity: identity,
    adapters: [{ id: updated.id, scale_hex: '0x1.0000000000000p+0' }] };
  const effectiveIdentity = { ...identity, checkpoint_sha256: createHash('sha256')
    .update(nativeCanonicalJson(adapterBinding)).digest('hex') };
  const store = new MemoryNeuraleseStore();
  for (const value of [basis, initial, gradient, updated, optimizerState])
    await store.putNativeTensorArtifact({ metadata: value.metadata, tensors: value.tensors });

  const remote = new Map([basis, initial, gradient, updated, optimizerState].map(value => [value.id, value]));
  let derivativeProgram;
  const server = createServer(async (request, response) => {
    const chunks = [];
    for await (const chunk of request) chunks.push(chunk);
    const bytes = Buffer.concat(chunks);
    const match = request.url.match(/^\/v1\/neuralese\/native-tensor-artifacts\/(nnt1_[a-z2-7]{52})(\/meta)?$/);
    if (match) {
      const value = remote.get(match[1]);
      if (!value) { response.writeHead(404); response.end(); return; }
      if (request.method === 'PUT' && !match[2]) {
        const uploaded = decodeNativeTensorArtifact(bytes);
        remote.set(uploaded.id, uploaded);
        response.writeHead(201, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ id: uploaded.id }));
      } else if (match[2]) {
        response.writeHead(200, { 'content-type': 'application/json' });
        response.end(JSON.stringify({ id: value.id, ...value.metadata }));
      } else {
        response.writeHead(200, { 'content-type': 'application/octet-stream' });
        response.end(encodeNativeTensorArtifact(value));
      }
      return;
    }
    if (request.url === '/v1/neuralese/grad') {
      const body = JSON.parse(bytes.toString());
      if (body.native_derivative_program) {
        derivativeProgram = body.native_derivative_program;
        const root = derivativeProgram.wrt[0];
        response.end(JSON.stringify({ loss: 0.4, execution_id: 'fixture-execution', gradients: { [root]: { id: gradient.id } } }));
      } else {
        const targets = body.native_recorded_objective.targets;
        response.end(JSON.stringify({ loss: 0.5, stats: {},
          gradients: Object.fromEntries(targets.map(target => [target.root, { id: gradient.id }])),
          native_identity: identity, objective: 'native-recorded-objective/1' }));
      }
      return;
    }
    if (request.url === '/v1/neuralese/optim') {
      response.end(JSON.stringify({ adapter: { id: updated.id }, optimizer_state_id: optimizerState.id,
        step: 1, native_identity: identity, adapter_binding: adapterBinding, effective_native_identity: effectiveIdentity }));
      return;
    }
    response.writeHead(404); response.end();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const learning = createLearning(learningService({ endpoint: `http://127.0.0.1:${server.address().port}`, store }));
  let turnNumber = 0;
  const episode = async actor => {
    const recorder = activeRecorder();
    const id = `momentum-turn-${++turnNumber}`, node = `${id}:model-turn`;
    recorder.recordGraphNode({ node, kind: 'model_turn', call_id: id });
    recorder.record({ messages: [{ role: 'user', content: 'score this actor' }], tools: [],
      reply: { role: 'assistant', content: 'done' }, blocks: [], adapters: activeNativeAdapters(),
      native: { schema: 'natlang.native-recorded-turn/1', id, invocation_id: id, retry: 0,
        kind: 'generation', sampling: { schema: 'natlang.native-rl-sampling/4', prompt: [], writes: [], native_state_imports: [] },
        graph: { graph_turn: node, written_nodes: {}, read_bindings: {} } } });
    recorder.linkNativeTurn(id, node);
    return { id, turns: [recorder.turns.at(-1)] };
  };

  try {
    await learning.valueAndGrad(async actor => {
      const inner = await learning.valueAndGrad(async candidate =>
        learning.objectives.logLikelihood(await learning.withNativeAdapters(candidate, () => episode(candidate))), actor);
      const optimizer = learning.optimizers.sgd({ lr: 0.1, momentum: 0.8 });
      const update = await optimizer.step({ value: actor, opt: optimizer.init(actor) }, inner.grad);
      return learning.objectives.logLikelihood(await learning.withNativeAdapters(update.value, () => episode(update.value)));
    }, nativeAdapterRef(initial.id), { order: 2 });

    assert.ok(derivativeProgram, 'the outer gradient must be sent as a derivative program');
    const optimizerNode = derivativeProgram.nodes.find(node => node.kind === 'optimizer');
    assert.ok(optimizerNode, 'the inner SGD update must appear in the replay graph');
    assert.deepEqual(optimizerNode.optimizer, { name: 'sgd', learning_rate: 0.1, momentum: 0.8 },
      'replay must retain every optimizer setting used by the forward update');
  } finally {
    await new Promise(resolve => server.close(resolve));
  }
});
