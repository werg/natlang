// Real native HTTP forwards and state splices; deterministic tool readout is not a quality claim.
import assert from 'node:assert/strict';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
import { neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { NativeToolAgent } from '../../dist/native/agent.js';
import { session as open } from '../support/natlang.mjs';
import { nativeStateConsumerFromInfo } from '../../dist/native/neuralese.js';
const [endpoint, archive] = process.argv.slice(2);
const info = await fetch(endpoint+'/v1/neuralese/info').then(r => r.json());
const store = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
const neuralese = { store, dialect: info.dialects[0], nativeStateConsumer: nativeStateConsumerFromInfo(info) };
const server = neuraleseServerModelTurn({ endpoint, model: 'native-fixture', store });
const driver = Object.assign(async request => {
  const raw = await server({ ...request, max_tokens: 8 });
  return { ...raw, text: undefined, calls: [['return_result', { status: 'success', value: raw.text || 'consumed' }]] };
}, { neuralese: true });
const child = open({ type: '() => Neuralese<string>', instructions: 'Produce a useful memo.' }, { neuralese }).session;
await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(child);
assert.ok(child.lam.return?.$neuralese, JSON.stringify(child.lam));
const id = child.lam.return.$neuralese.id;
assert.ok(id.startsWith('nzs1_'));
const write = child.runtime.trace.events.find(event => event.kind === 'block_write' && event.block === id);
assert.ok(write?.native_identity, 'native producer is recorded through the shared block-write graph');
const reopened = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
assert.equal((await reopened.getNativeState(id)).id, id);
assert.deepEqual(await server.blocks.collect(new Set()), [id]);
assert.equal(await server.blocks.hasNativeState(id), false, 'parent must restore exact native state from its persistent archive');
const parent = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Consume the child memo.',
  args: { memo: child.lam.return } }, { neuralese }).session;
await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(parent);
assert.equal(parent.lam.return, 'consumed');
const read = parent.runtime.trace.events.find(event => event.kind === 'block_read' && event.block === id);
assert.ok(read?.inputs.some(input => input.node === write.node), 'parent reads the actual child producer, not an invented external edge');
console.log(JSON.stringify({ state_id: id, parent_result: parent.lam.return, archived: true }));
