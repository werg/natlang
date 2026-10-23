import assert from 'node:assert/strict';
import { createServer } from 'node:http';
import { test } from 'node:test';
import { createLearning, learningService } from '../dist/neuralese/learning.js';
import { activeRecorder } from '../dist/native/model-recording.js';
import { MemoryNeuraleseStore, encodeNativeTensorArtifact, decodeNativeTensorArtifact } from '../dist/native/neuralese-store.js';
import { nativeAdapterRef } from '../dist/native/neuralese.js';
import { nativeStateFixture } from './fixtures/native-state.mjs';

const identity = { checkpoint_sha256: '1'.repeat(64), tokenizer_fingerprint: '2'.repeat(64),
  embedding_weight_sha256: '3'.repeat(64), architecture: 'fixture-native' };
const tensor = value => ({ dtype: 'F32', shape: [1], data: new Uint8Array(new Float32Array([value]).buffer) });

// Exercise the actual native objective builders with content-addressed nnt1/nzs1
// values and the current recorded-turn envelope. The fake server only answers
// the transport contract; numerical execution is tested by the Python ports.
async function requestTerms(build, targetActor = 'student') {
  const store = new MemoryNeuraleseStore();
  const actors = {};
  for (const [name, value] of [['student', 0.1], ['teacher', 0.2], ['sample', 0.3]]) {
    actors[name] = await store.putNativeTensorArtifact({ metadata: {
      schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-coefficients',
      native_identity: identity, source_artifact_id: null,
      provenance: { adapter_spec: { kind: 'xs', rank: 1, dim: 1, parameter_names: ['fixture.weight'] } },
      tensors: [{ name: 'fixture.weight', dtype: 'F32', shape: [1] }],
    }, tensors: { 'fixture.weight': tensor(value) } });
  }
  const raw = nativeStateFixture(identity.architecture);
  raw.metadata.checkpoint_sha256 = identity.checkpoint_sha256;
  raw.metadata.tokenizer_fingerprint = identity.tokenizer_fingerprint;
  raw.metadata.embedding_weight_sha256 = identity.embedding_weight_sha256;
  const state = await store.putNativeState(raw);
  const target = actors[targetActor];
  const gradient = await store.putNativeTensorArtifact({ metadata: {
    schema: 'natlang.neuralese-native-tensor-artifact/1', kind: 'adapter-gradient',
    native_identity: identity, source_artifact_id: target.id, provenance: { fixture: true },
    tensors: [{ name: 'fixture.weight', dtype: 'F32', shape: [1] }],
  }, tensors: { 'fixture.weight': tensor(0.4) } });

  const remote = new Map();
  let captured;
  const server = createServer(async (req, res) => {
    const chunks = []; for await (const chunk of req) chunks.push(chunk);
    const bytes = Buffer.concat(chunks);
    const match = req.url.match(/^\/v1\/neuralese\/native-tensor-artifacts\/(nnt1_[a-z2-7]{52})(\/meta)?$/);
    if (match && req.method === 'PUT' && !match[2]) {
      const artifact = decodeNativeTensorArtifact(bytes);
      remote.set(artifact.id, artifact);
      res.writeHead(201, { 'content-type': 'application/json' }); res.end(JSON.stringify({ id: artifact.id })); return;
    }
    if (match && req.method === 'GET' && match[2]) {
      const artifact = remote.get(match[1]);
      if (!artifact) { res.writeHead(404); res.end(); return; }
      const { id, metadata } = artifact;
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ id, ...metadata })); return;
    }
    if (match && req.method === 'GET' && !match[2]) {
      const artifact = remote.get(match[1]);
      if (!artifact) { res.writeHead(404); res.end(); return; }
      res.writeHead(200, { 'content-type': 'application/octet-stream' });
      res.end(encodeNativeTensorArtifact(artifact)); return;
    }
    if (req.url === '/v1/neuralese/grad' && req.method === 'POST') {
      captured = JSON.parse(bytes.toString());
      const targets = captured.native_recorded_objective?.targets ?? [];
      res.writeHead(200, { 'content-type': 'application/json' });
      res.end(JSON.stringify({ loss: 0.25, stats: { graph_nodes: 1 },
        gradients: Object.fromEntries(targets.map(item => [item.root, { id: gradient.id }])),
        native_identity: identity, objective: 'native-recorded-objective/1' })); return;
    }
    res.writeHead(404); res.end();
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const endpoint = `http://127.0.0.1:${server.address().port}`;
  try {
    const learning = createLearning(learningService({ endpoint, store }));
    const result = await learning.valueAndGrad(() => build(learning, actors, state), nativeAdapterRef(target.id));
    assert.equal(+result.loss, 0.25);
    assert.ok(result.grad.$gradient.$gradientBlock);
    assert.ok(captured?.native_recorded_objective, 'native recorded objective reaches the shared gradient endpoint');
    return captured.native_recorded_objective;
  } finally { await new Promise(resolve => server.close(resolve)); }
}

function output(id, actors, state, { decision = false, includeState = false } = {}) {
  const actor = actors[id === 'student' ? 'student' : id === 'teacher' ? 'teacher' : 'sample'];
  return async () => {
    const recorder = activeRecorder();
    const node = `turn-node:${id}`;
    const prompt = includeState ? [{ kind: 'native_state', id: state.id }] : [];
    const reads = includeState ? [{ node: `read:${id}`, kind: 'block_read', turn: node, block: state.id,
      inputs: [{ block: state.id, node: 'external' }] }] : [];
    const receipt = includeState ? [{ id: state.id, mode: 'same-base-adapter-state',
      observed_native_state_id: state.id }] : [];
    recorder.recordGraphNode({ node, kind: 'model_turn', call_id: id, ...(decision ? { readout: 'decision' } : {}) });
    for (const event of reads) recorder.recordGraphNode(event);
    recorder.record({ messages: [{ role: 'system', content: includeState ? [{ type: 'native_state', id: state.id }] : 'system' },
      { role: 'user', content: id }],
      tools: [{ type: 'function', function: { name: 'tool-' + id, parameters: { type: 'object' } } }],
      reply: { role: 'assistant', content: 'answer' }, blocks: [],
      adapters: [{ id: actor.id, scale: 0.25 }],
      ...(includeState ? { native_state_modes: { [state.id]: 'same-base-adapter-state' } } : {}),
      ...(decision ? { decision: { options: ['"yes"', '"no"'] } } : {}),
      native: { schema: 'natlang.native-recorded-turn/1', id, invocation_id: id, retry: 0,
        ...(decision ? { kind: 'decision', decision: { prompt, writes: [], native_state_imports: receipt } } :
          { kind: 'generation', sampling: { schema: 'natlang.native-rl-sampling/4', prompt, writes: [],
            native_state_imports: receipt } }) } });
    recorder.linkNativeTurn(id, node);
    return recorder.turns.at(-1);
  };
}

test('generated distillation retains exact independent teacher occurrence and native transport contract', async () => {
  for (const conditioned of [false, true]) {
    const request = await requestTerms((learning, actors, state) => conditioned
      ? learning.objectives.conditionedDistill(output('student', actors, state), output('teacher', actors, state, { includeState: true }))
      : learning.objectives.selfDistill(output('student', actors, state), output('teacher', actors, state, { includeState: true })));
    const term = request.terms[0];
    assert.equal(term.native_turn_id, 'student');
    assert.equal(term.teacher_native_turn_id, 'teacher');
    assert.deepEqual(term.teacher_native_state_modes, { [request.turns[1].native.sampling.native_state_imports[0].id]: 'same-base-adapter-state' });
    assert.deepEqual(term.teacher_adapters, [{ id: request.turns[1].adapters[0].id, scale: 0.25 }]);
    assert.deepEqual(request.turns.map(turn => turn.native.id), ['student', 'teacher']);
  }
});

test('decision distillation binds its actual teacher query without borrowing student modes', async () => {
  const request = await requestTerms((learning, actors, state) => learning.objectives.conditionedDistill(
    output('student', actors, state, { decision: true }), output('teacher', actors, state, { decision: true, includeState: true })));
  const term = request.terms[0];
  assert.equal(term.native_turn_id, 'student');
  assert.equal(term.teacher.native_turn_id, 'teacher');
  assert.deepEqual(term.teacher.native_state_modes, { [request.turns[1].native.decision.native_state_imports[0].id]: 'same-base-adapter-state' });
  assert.deepEqual(term.teacher.adapters, [{ id: request.turns[1].adapters[0].id, scale: 0.25 }]);
  assert.equal(term.tools[0].function.name, 'tool-student');
  assert.equal(term.teacher.tools[0].function.name, 'tool-teacher');
  assert.equal(term.native_state_modes, undefined);
});

test('REINFORCE and conditional likelihood retain distinct probability laws', async () => {
  const request = await requestTerms(async (learning, actors, state) => learning.objectives.sum(
    await learning.objectives.logLikelihood({ id: 'trajectory', turns: [await output('sample', actors, state)()] }, 3),
    await learning.objectives.policyGradient([{ trajectory: { id: 'trajectory', turns: [await output('sample', actors, state)()] }, reward: 5 }], { baseline: 2 })),
    'sample');
  assert.deepEqual(request.terms.map(term => [term.likelihood_law, term.weight, term.native_turn_id]),
    [['model-conditional', 3, 'sample'], ['sampled-policy', 3, 'sample']]);
});
