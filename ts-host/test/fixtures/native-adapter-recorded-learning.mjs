// Actual native HTTP turns and producer/consumer occurrences through public natlang:learning.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { NativeToolAgent } from '../../dist/native/agent.js';
import { nativeStateConsumerFromInfo } from '../../dist/native/neuralese.js';
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
const server = neuraleseServerModelTurn({ endpoint, model: 'native-recorded-learning', store });
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
  scoped_actor_isolated: true, saved_reloaded: true }));
