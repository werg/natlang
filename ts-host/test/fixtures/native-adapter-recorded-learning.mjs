// Actual native HTTP turns and producer/consumer occurrences through public natlang:learning.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { NativeToolAgent } from '../../dist/native/agent.js';
import { nativeStateConsumerFromInfo, neuraleseRef } from '../../dist/native/neuralese.js';
import { session as open } from '../support/natlang.mjs';
import { createLearning, learningService } from '../../dist/neuralese/learning.js';
import { activeNativeAdapters } from '../../dist/neuralese/recording.js';

const [endpoint, archive, binding] = process.argv.slice(2);
const fixture = JSON.parse(await readFile(binding, 'utf8'));
const info = await fetch(endpoint + '/v1/neuralese/info').then(response => response.json());
const consumer = nativeStateConsumerFromInfo(info);
const store = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
const learning = createLearning(learningService({ endpoint, store }));
const adapter = await learning.nativeAdapters.create(fixture.identity, { parameterNames: [fixture.target], rank: 2 });
const server = neuraleseServerModelTurn({ endpoint, model: 'native-recorded-learning', store,
  nativeStateModes: { [fixture.state_id]: 'same-base-adapter-state' } });
const driver = Object.assign(async request => {
  const raw = await server({ ...request, max_tokens: 8 });
  const content = raw.raw_response?.choices?.[0]?.message?.content;
  const produced = Array.isArray(content) && content.some(part => part.type === 'native_state');
  return { ...raw, text: undefined, calls: [['return_result', { status: 'success', value: produced ? content : raw.text || 'consumed' }]] };
}, { neuralese: true, decide: server.decide });

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
  const trajectory = await learning.trajectory(async () => {
    const parent = open({ type: '(memo: Neuralese<string>, backup: Neuralese<string>) => string',
      instructions: 'Consume both exact state inputs.', args: { memo: state, backup: state } }, { neuralese: {
      store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).session;
    await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store, dialect: info.dialects[0], nativeStateConsumer: consumer } }).run(parent);
    assert.equal(parent.lam.return, 'consumed');
  });
  assert.ok(trajectory.turns.some(turn => turn.native?.sampling?.prompt?.filter(segment =>
    segment.kind === 'native_state' && segment.id === fixture.state_id).length === 2),
    'the state target must retain both actual import occurrences');
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
  scoped_actor_isolated: true, saved_reloaded: true, native_state_gradient: true, joint_gradient: true, higher_order: true }));
