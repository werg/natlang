// Actual native HTTP turns and producer/consumer occurrences through public natlang:learning.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { NativeToolAgent } from '../../dist/native/agent.js';
import { nativeStateConsumerFromInfo, neuraleseRef } from '../../dist/native/neuralese.js';
import { session as open } from '../support/natlang.mjs';
import { createLearning, learningService } from '../../dist/neuralese/learning.js';
import { activeNativeAdapters, activeNativeStates, activeRecorder } from '../../dist/native/model-recording.js';

const [endpoint, archive, binding] = process.argv.slice(2);
const fixture = JSON.parse(await readFile(binding, 'utf8'));
const info = await fetch(endpoint + '/v1/neuralese/info').then(response => response.json());
const consumer = nativeStateConsumerFromInfo(info);
const store = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
const learning = createLearning(learningService({ endpoint, store }));
const driverContexts = [];
let expectedRecorder;
const promptStateIds = value => {
  if (Array.isArray(value)) return value.flatMap(promptStateIds);
  if (!value || typeof value !== 'object') return [];
  if (value.type === 'native_state' && typeof value.id === 'string') return [value.id];
  return Object.values(value).flatMap(promptStateIds);
};
const adapter = await learning.nativeAdapters.create(fixture.identity, { parameterNames: [fixture.target], rank: 2 });
const server = neuraleseServerModelTurn({ endpoint, model: 'native-recorded-learning', store });
const driverFor = selectedServer => Object.assign(async request => {
  driverContexts.push({ path: 'generation', hasRecorder: !!activeRecorder(), sameRecorder: activeRecorder() === expectedRecorder,
    stateIds: activeNativeStates().map(state => state.id), promptStateIds: promptStateIds(request.messages) });
  let raw;
  try { raw = await selectedServer({ ...request, max_tokens: 8 }); }
  catch (error) {
    driverContexts.push({ path: 'generation-error', message: error instanceof Error ? error.message : String(error) });
    throw error;
  }
  const content = raw.raw_response?.choices?.[0]?.message?.content;
  const produced = Array.isArray(content) && content.some(part => part.type === 'native_state');
  return { ...raw, text: undefined, calls: [['return_result', { status: 'success', value: produced ? content : raw.text || 'consumed' }]] };
}, { neuralese: true, decide: async (...args) => {
  driverContexts.push({ path: 'decide', hasRecorder: !!activeRecorder(), sameRecorder: activeRecorder() === expectedRecorder,
    stateIds: activeNativeStates().map(state => state.id) });
  try { return await selectedServer.decide(...args); }
  catch (error) {
    driverContexts.push({ path: 'decide-error', message: error instanceof Error ? error.message : String(error) });
    throw error;
  }
} });
const driver = driverFor(server);

async function episode() {
  const trajectory = await learning.trajectory(async () => {
    const child = open({ type: '() => Neuralese<string>', instructions: 'Produce a useful memo.' }, { neuralese: {
      store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(child);
    const parent = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Consume the child memo.',
      args: { memo: child.lam.return } }, { neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(parent);
    assert.equal(parent.lam.return, 'consumed');
  });
  assert.ok(trajectory.turns.some(turn => turn.native?.graph?.written_nodes && Object.keys(turn.native.graph.written_nodes).length),
    'captured producer turn must retain its actual written native-state occurrence');
  assert.ok(trajectory.turns.some(turn => turn.native?.graph?.read_bindings && Object.keys(turn.native.graph.read_bindings).length),
    'captured consumer turn must retain the actual producer import binding');
  assert.ok(trajectory.turns.every(turn => turn.adapters?.length === 1 && turn.native.sampling.adapter_binding?.adapters?.length === 1),
    'each captured producer/consumer turn must retain the effective native actor binding');
  return trajectory;
}

async function consumeStoredState(state) {
  const contextStart = driverContexts.length;
  expectedRecorder = activeRecorder();
  const runResults = [];
  let producedStateId;
  let mixedCallId, scopedOnlyCallId;
  const trajectory = await learning.trajectory(async () => {
    const stateId = state.$neuralese.id;
    assert.ok(expectedRecorder, 'state objective must run in the active high-level recorder');
    const scopedServer = neuraleseServerModelTurn({ endpoint, model: 'native-recorded-learning', store,
      onExchange: () => driverContexts.push({ path: 'exchange', hasRecorder: !!activeRecorder(), sameRecorder: activeRecorder() === expectedRecorder,
        stateIds: activeNativeStates().map(state => state.id) }),
      nativeStateModes: { [stateId]: 'same-base-adapter-state' } });
    const scopedDriver = driverFor(scopedServer);
    const child = open({ type: '() => Neuralese<string>', instructions: 'Produce a useful memo.' }, { neuralese: {
      store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    const agentOptions = { maxTurns: 2, neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } };
    assert.equal(child.completed, false, 'state scoped producer must be a fresh session');
    const beforeChild = driverContexts.length;
    runResults.push(await learning.withNativeStates(stateId, async () => {
      assert.deepEqual(activeNativeStates(), [{ id: stateId }]);
      return new NativeToolAgent(scopedDriver, agentOptions).run(child);
    }));
    const afterChild = driverContexts.length;
    producedStateId = child.lam.return?.$neuralese?.id;
    assert.ok(producedStateId?.startsWith('nzs1_'), 'the in-trajectory producer must return its actual archived native state');
    const producedArtifact = await store.getNativeState(producedStateId);
    assert.ok(producedArtifact, `producer state ${producedStateId} must remain in archive custody`);
    const producedIdentity = { architecture: producedArtifact.metadata.architecture, checkpoint_sha256: producedArtifact.metadata.checkpoint_sha256,
      tokenizer_fingerprint: producedArtifact.metadata.tokenizer_fingerprint, embedding_weight_sha256: producedArtifact.metadata.embedding_weight_sha256 };
    const consumerIdentity = { architecture: consumer.architecture, checkpoint_sha256: consumer.checkpoint_sha256,
      tokenizer_fingerprint: consumer.tokenizer_fingerprint, embedding_weight_sha256: consumer.embedding_weight_sha256 };
    const adaptedBase = producedArtifact.metadata.adapter_binding?.base_native_identity;
    const matches = identity => identity && ['architecture', 'checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256']
      .every(key => identity[key] === consumerIdentity[key]);
    assert.ok(matches(producedIdentity) || matches(adaptedBase),
      `producer state must authenticate this exact reader, directly or through its adapter base: ${JSON.stringify(producedArtifact.metadata)}`);
    const session = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Use the memo argument and stored guidance.',
      args: { memo: child.lam.return } },
    { neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    mixedCallId = session.runtime.options.runId;
    assert.equal(session.completed, false, 'state scoped consumer must be a fresh session');
    runResults.push(await learning.withNativeStates([stateId, producedStateId], async () => {
      assert.deepEqual(activeNativeStates(), [{ id: stateId }, { id: producedStateId }]);
      return new NativeToolAgent(scopedDriver, agentOptions).run(session);
    }));
    const scopedOnly = open({ type: '() => string', instructions: 'Use the child state supplied by scope.' },
      { neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    scopedOnlyCallId = scopedOnly.runtime.options.runId;
    runResults.push(await learning.withNativeStates([stateId, stateId, producedStateId], () =>
      new NativeToolAgent(scopedDriver, agentOptions).run(scopedOnly)));
    driverContexts.push({ path: 'session-boundaries', beforeChild, afterChild, afterParent: driverContexts.length,
      childCompleted: child.completed, parentCompleted: session.completed, scopedOnlyCompleted: scopedOnly.completed });
    assert.equal(session.lam.return, 'consumed');
    assert.equal(scopedOnly.lam.return, 'consumed');
  });
  const hasStateRead = (turn) => turn.native?.graph?.read_bindings?.[state.$neuralese.id] !== undefined ||
    turn.native?.graph?.read_occurrences?.some(occurrence => occurrence.source_state_id === state.$neuralese.id);
  assert.equal(runResults.length, 3, `all scoped agent calls must complete: ${JSON.stringify({ failure: trajectory.failure, contexts: driverContexts.slice(contextStart) })}`);
  assert.ok(trajectory.turns.filter(hasStateRead).length >= 2,
    `the stored state must retain exact imports in the actual producer and consumer calls: ${JSON.stringify({ turns: trajectory.turns.map(turn => ({ native: turn.native?.graph, modes: turn.native_state_modes })), contexts: driverContexts.slice(contextStart) })}`);
  const producedTurn = trajectory.turns.find(turn => turn.native?.graph?.written_nodes?.[producedStateId] ||
    turn.native?.graph?.written_occurrences?.some(occurrence => occurrence.id === producedStateId));
  const consumerTurn = trajectory.turns.find(turn => turn.native?.graph?.read_occurrences?.some(occurrence =>
    occurrence.id === producedStateId && occurrence.source !== 'external'));
  assert.ok(producedTurn && consumerTurn,
    'a scoped state produced in the same trajectory must become a recorded consumer read');
  const producedNodes = new Set([...(producedTurn.native.graph.written_nodes ? Object.values(producedTurn.native.graph.written_nodes) : []),
    ...(producedTurn.native.graph.written_occurrences ?? []).map(occurrence => occurrence.node)]);
  assert.ok(consumerTurn.native.graph.read_occurrences.some(occurrence => producedNodes.has(occurrence.source)),
    'the scoped consumer import must bind to the actual producer write node');
  const mixedReads = [...expectedRecorder.graphNodes.values()].filter(event => event.kind === 'block_read' &&
    event.call_id === mixedCallId && event.block === producedStateId);
  assert.equal(mixedReads.length, 1, 'an already-present typed import must not be duplicated by withNativeStates');
  assert.ok(mixedReads[0].positions[0] > 0,
    'the typed import position must include the actual prepended scoped system message');
  const scopedOnlyTurn = trajectory.turns.find(turn => turn.native?.graph?.graph_turn.startsWith(`${scopedOnlyCallId}#`));
  const scopedOnlySource = scopedOnlyTurn?.native?.graph?.read_bindings?.[producedStateId] ??
    scopedOnlyTurn?.native?.graph?.read_occurrences?.find(occurrence => occurrence.id === producedStateId)?.source;
  assert.ok(scopedOnlySource && producedNodes.has(scopedOnlySource),
  'an ID-only scoped import must resolve to the actual earlier writer rather than an external root');
  const scopedOnlyReads = [...expectedRecorder.graphNodes.values()].filter(event => event.kind === 'block_read' &&
    event.call_id === scopedOnlyCallId);
  const scopedReadsByTurn = Map.groupBy(scopedOnlyReads, event => event.turn);
  assert.ok(scopedReadsByTurn.size >= 1, 'the scoped-only call must record its actual model turns');
  for (const [turn, reads] of scopedReadsByTurn) {
    const guidanceReads = reads.filter(event => event.block === state.$neuralese.id);
    const scopedWriterRead = reads.find(event => event.block === producedStateId);
    assert.deepEqual(guidanceReads.map(event => event.positions), [[0, 0], [0, 1]],
      `turn ${turn} must preserve both same-ID scope occurrences`);
    assert.deepEqual(scopedWriterRead?.positions, [0, 2],
      `turn ${turn} must preserve the distinct scoped import's actual system-message part position`);
  }
  return trajectory;
}

async function separatelyScopedEpisode(producerAdapter, consumerAdapter) {
  const trajectory = await learning.trajectory(async () => {
    const child = open({ type: '() => Neuralese<string>', instructions: 'Produce a useful memo.' }, { neuralese: {
      store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    await learning.withNativeAdapters(producerAdapter, () =>
      new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(child));
    const stateId = child.lam.return?.$neuralese?.id;
    assert.ok(stateId?.startsWith('nzs1_'), 'the first scoped call must produce an exact native state artifact');
    const parent = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Consume the child memo.',
      args: { memo: child.lam.return } }, { neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    const parentServer = neuraleseServerModelTurn({ endpoint, model: 'native-recorded-learning', store,
      nativeStateModes: { [stateId]: 'same-base-adapter-state' } });
    await learning.withNativeAdapters(consumerAdapter, () =>
      new NativeToolAgent(driverFor(parentServer), { maxTurns: 2,
        neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(parent));
    assert.equal(parent.lam.return, 'consumed');
    const control = open({ type: '() => string', instructions: 'Return the unbound control result.' }, { neuralese: {
      store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    await new NativeToolAgent(driver, { maxTurns: 2,
      neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(control);
  });
  const modelTurns = trajectory.turns.filter(turn => turn.native);
  const actorIds = modelTurns.map(turn => turn.adapters?.[0]?.id ?? null);
  assert.deepEqual(actorIds.slice(0, 2), [producerAdapter.$nativeAdapter.id, consumerAdapter.$nativeAdapter.id],
    'producer and consumer turns must retain their own exact actor scopes');
  assert.ok(actorIds.length >= 3 && actorIds.slice(2).every(id => id === null),
    'every turn of the unbound control must remain unbound');
  return trajectory;
}

const initial = await learning.valueAndGrad(async actor =>
  learning.objectives.logLikelihood(await learning.withNativeAdapters(actor, episode)), adapter);
assert.deepEqual(activeNativeAdapters(), [], 'native actor bindings must not escape their asynchronous scope');
assert.ok(Number.isFinite(initial.loss.valueOf()) && initial.loss.valueOf() > 0);
const optimizer = learning.optimizers.adam({ lr: 0.08 });
const state = optimizer.init(adapter);
const updated = await optimizer.step({ value: adapter, opt: state }, initial.grad);
assert.notEqual(updated.value.$nativeAdapter.id, adapter.$nativeAdapter.id);
const after = await learning.valueAndGrad(async actor =>
  learning.objectives.logLikelihood(await learning.withNativeAdapters(actor, episode)), updated.value);
assert.ok(after.loss.valueOf() < initial.loss.valueOf(), `Adam objective must decrease: ${initial.loss.valueOf()} -> ${after.loss.valueOf()}`);
const higherOrder = await learning.valueAndGrad(async actor => {
  const inner = await learning.valueAndGrad(async candidate =>
    learning.objectives.logLikelihood(await learning.withNativeAdapters(candidate, episode)), actor);
  const metaOptimizer = learning.optimizers.adam({ lr: 0.02 });
  const adapted = await metaOptimizer.step({ value: actor, opt: metaOptimizer.init(actor) }, inner.grad);
  return learning.objectives.logLikelihood(await learning.withNativeAdapters(adapted.value, episode));
}, adapter, { order: 2 });
const higherOrderEntry = higherOrder.grad.$gradient.$gradientBlock;
const higherOrderArtifact = await store.getNativeTensorArtifact(higherOrderEntry.id);
assert.equal(higherOrderArtifact.metadata.kind, 'adapter-gradient');
assert.equal(higherOrderArtifact.metadata.provenance.native_derivative_target.stage, 'outer');
assert.ok(higherOrderArtifact.metadata.provenance.derivative_execution_id,
  'order-2 adapter gradients must come from the native derivative-program executor');
const secondAdapter = await learning.nativeAdapters.create(fixture.identity, { parameterNames: [fixture.target], rank: 1 });
const multiTarget = await learning.valueAndGrad(async actors =>
  learning.objectives.logLikelihood(await separatelyScopedEpisode(actors.producer, actors.consumer)),
  { producer: adapter, consumer: secondAdapter });
const producerGradient = multiTarget.grad.$gradient.producer.$gradientBlock;
const consumerGradient = multiTarget.grad.$gradient.consumer.$gradientBlock;
assert.equal(producerGradient.of, adapter.$nativeAdapter.id);
assert.equal(consumerGradient.of, secondAdapter.$nativeAdapter.id);
assert.notEqual(producerGradient.id, consumerGradient.id,
  'separately scoped native actors must keep their exact gradient tree bindings');
const multiTargetValue = { producer: adapter, consumer: secondAdapter };
const multiTargetUpdate = await optimizer.step({ value: multiTargetValue, opt: optimizer.init(multiTargetValue) }, multiTarget.grad);
assert.notEqual(multiTargetUpdate.value.producer.$nativeAdapter.id, adapter.$nativeAdapter.id);
assert.notEqual(multiTargetUpdate.value.consumer.$nativeAdapter.id, secondAdapter.$nativeAdapter.id);
const stateRef = neuraleseRef('NativeState', fixture.state_id);
const stateValue = { memo: stateRef, backup: stateRef };
const stateResult = await learning.valueAndGrad(async state =>
  learning.objectives.logLikelihood(await consumeStoredState(state.memo)), stateValue);
const memoGradient = stateResult.grad.$gradient.memo.$gradientBlock;
assert.equal(memoGradient.type, 'NativeState');
assert.equal(stateResult.grad.$gradient.backup.$gradientBlock.id, memoGradient.id,
  'repeated state roots share one exact named gradient artifact');
const stateGradient = await store.getNativeTensorArtifact(memoGradient.id);
assert.equal(stateGradient.metadata.kind, 'state-gradient');
const stateUpdated = await optimizer.step({ value: stateValue, opt: optimizer.init(stateValue) }, stateResult.grad);
assert.equal(stateUpdated.value.memo.$neuralese.id, stateUpdated.value.backup.$neuralese.id);
assert.notEqual(stateUpdated.value.memo.$neuralese.id, fixture.state_id);
const jointInput = { actor: adapter, memo: stateRef, backup: stateRef };
const joint = await learning.valueAndGrad(async values =>
  learning.objectives.logLikelihood(await learning.withNativeAdapters(values.actor, () => consumeStoredState(values.memo))), jointInput);
assert.equal(joint.grad.$gradient.actor.$gradientBlock.type, 'NativeAdapter');
assert.equal(joint.grad.$gradient.memo.$gradientBlock.type, 'NativeState');
const jointUpdated = await optimizer.step({ value: jointInput, opt: optimizer.init(jointInput) }, joint.grad);
assert.notEqual(jointUpdated.value.actor.$nativeAdapter.id, adapter.$nativeAdapter.id);
assert.notEqual(jointUpdated.value.memo.$neuralese.id, fixture.state_id);
const directory = archive + '-reload';
const archivePath = directory + '/learned-adapter.nz';
const { mkdir } = await import('node:fs/promises');
await mkdir(directory, { recursive: true });
await learning.save(archivePath, { actor: updated.value });
const reloadedStore = await FileNeuraleseStore.open(directory + '/runtime', { residentBytes: 0 });
const reloadedLearning = createLearning(learningService({ endpoint, store: reloadedStore }));
const reloaded = await reloadedLearning.load(archivePath);
assert.equal(reloaded.exports.actor.$nativeAdapter.id, updated.value.$nativeAdapter.id);
assert.ok(await reloadedStore.getNativeTensorArtifact(reloaded.exports.actor.$nativeAdapter.id), 'reload must restore nnt1 coefficients');
const delta = await learning.deltas.diff(updated.value, adapter);
const restored = await learning.deltas.apply(adapter, delta);
assert.ok(restored.$nativeAdapter.id.startsWith('nnt1_'), 'native delta application must produce a durable coefficient artifact');
console.log(JSON.stringify({ passed: true, initial_loss: initial.loss.valueOf(), updated_loss: after.loss.valueOf(),
  adapter: updated.value.$nativeAdapter.id, delta: delta.$nativeAdapter.id, producer_consumer: true,
  scoped_actor_isolated: true, saved_reloaded: true, native_state_gradient: true, joint_gradient: true,
  multi_target_scope: true, higher_order: true, scoped_state_writer: true, mixed_import_positions: true,
  repeated_scope_imports: true }));
