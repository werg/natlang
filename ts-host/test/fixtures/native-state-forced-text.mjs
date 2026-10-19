import assert from 'node:assert/strict';
import { FileNeuraleseStore } from '../../dist/neuralese/node-block-store.js';
const [endpoint, archive] = process.argv.slice(2);
const info = await fetch(endpoint+'/v1/neuralese/info').then(r => r.json());
const reopened = await FileNeuraleseStore.open(archive, { residentBytes: 0 });
// Forced typed writes execute the actual producer; contextual init preserves imported source receipts.
const { nativeStateRuntimeOptions } = await import('../../dist/model/native-state-client.js');
const { initBodyInContext } = await import('../../dist/neuralese/text-init.js');
const { saveNz, decodeNz } = await import('../../dist/native/nz-file.js');
const neuralese = await nativeStateRuntimeOptions({ endpoint, store: reopened, info });
const port = neuralese.port;
const forced = await port.write('t5', { type: 'Neuralese<string>' });
assert.ok(forced.id.startsWith('nzs1_'));
assert.equal((await reopened.getNativeState(forced.id)).metadata.text, 't5');
assert.ok(forced.producer.native_source_capture);
const { NativeToolAgent } = await import('../../dist/native/agent.js');
const { session: open } = await import('../support/natlang.mjs');
for (const text of ['t5', '']) {
  const computed = open({ type: '() => Neuralese<string>', instructions: 'Return the computed string.' }, { neuralese }).session;
  const driver = async () => ({ calls: [['return_result', { status: 'success', value: text }]] });
  await new NativeToolAgent(driver, { maxTurns: 1, neuralese }).run(computed);
  assert.ok(computed.lam.return?.$neuralese, JSON.stringify(computed.lam));
  const artifact = await reopened.getNativeState(computed.lam.return.$neuralese.id);
  assert.equal(artifact.metadata.text, text);
  assert.ok(computed.runtime.trace.events.some(event => event.kind === 'block_write' && event.source_capture));
}

const initialized = await initBodyInContext({ endpoint, store: reopened, text: 't6', type: 'Neuralese<string>',
  render: async (placeholder, capture) => capture({ messages: [{ role: 'user', content: [
    { type: 'native_state', id: forced.id }, { type: 'text', text: ' t5' },
    { type: 'native_state', id: placeholder }, { type: 'text', text: 't8 ' }] }], tools: [] }) });
assert.equal(initialized.init.contextual, true);
assert.equal(initialized.init.source_capture.segments.filter(s => s.kind === 'native_state').length, 1);
const bytes = await saveNz({ body: { type: 'Neuralese<string>', value: { $neuralese: { type: 'Neuralese<string>', id: initialized.id } } } },
  { store: reopened, dialect: info.dialects[0], provenance: { init: initialized.init } });
const saved = decodeNz(bytes);
assert.ok(saved.nativeStates.has(forced.id), 'contextual source dependencies survive library publication');
assert.ok(saved.nativeStates.has(initialized.id));


const zero = await port.write('', { type: 'Neuralese<string>' });
assert.equal((await reopened.getNativeState(zero.id)).tensors.token_ids.shape[0], 0);
const zeroBody = await initBodyInContext({ endpoint, store: reopened, text: '', type: 'Neuralese<string>', placeholderOccurrence: 1,
  render: async (placeholder, capture) => capture({ messages: [{ role: 'user', content: [
    { type: 'native_state', id: zero.id }, { type: 'text', text: 't7 ' }, { type: 'native_state', id: placeholder }, { type: 'text', text: ' t8' }] }], tools: [] }) });
assert.equal(zeroBody.init.tokens, 0);
const emptyReply = await fetch(endpoint + '/v1/chat/completions', { method: 'POST',
  headers: { 'content-type': 'application/json' }, body: JSON.stringify({ messages: [{ role: 'user', content: 'empty-fixture' }], max_tokens: 8 }) }).then(r => r.json());
assert.ok(emptyReply.native_state?.blocks?.[0]?.id, JSON.stringify(emptyReply));
const { HttpNeuraleseStore } = await import('../../dist/model/neuralese-server.js');
const sampledEmpty = await new HttpNeuraleseStore(endpoint).getNativeState(emptyReply.native_state.blocks[0].id);
assert.equal(sampledEmpty.tensors.token_ids.shape[0], 0);
const { serverViewer } = await import('../../dist/neuralese/view.js');
let viewTurns;
const viewer = serverViewer({ endpoint, store: reopened, window: 2, onSourceTurns(turns) { viewTurns = turns; } });
const viewed = await viewer({ value: 't5 t6 t7', instructions: 'keep identifiers' });
assert.ok(viewed.$neuralese.id.startsWith('nzs1_'));
assert.equal(viewTurns.length, 3, 'two actual part writers and their actual combine writer');
assert.equal(viewTurns[2].messages[1].content.filter(part => part.type === 'native_state').length, 2);
for (const turn of viewTurns) for (const write of turn.response.x_natlang_rl_capture.writes)
  assert.ok(await reopened.getNativeState(write.state_id));
const { neuraleseServerModelTurn } = await import('../../dist/model/neuralese-server.js');
const actor = neuraleseServerModelTurn({ endpoint, model: 'actual-native-receipt', store: reopened,
  request: { temperature: 1, x_natlang_rl_capture: true, neuralese_template: {
    call: 'return_result', arguments: { status: 'success' }, argument: 'value', value: 'write' } } });
const response = await actor({ messages: [{ role: 'user', content: [{ type: 'native_state', id: viewed.$neuralese.id }] }], tools: [], max_tokens: 64 });
const sampling = response.raw_response.x_natlang_rl_capture;
assert.ok(sampling.writes.length);
for (const write of sampling.writes) for (const stateId of [write.state_id, write.raw_state_id].filter(Boolean))
  assert.ok(await reopened.getNativeState(stateId), 'ordinary server client archives raw and compact producer receipt artifacts');
for (const observation of sampling.prompt) if (observation.kind === 'native_state')
  assert.ok(await reopened.getNativeState(observation.id), 'ordinary server client archives actual decoded observations');
console.log(JSON.stringify({ forced: true, contextual: true, view_parts: 2 }));
