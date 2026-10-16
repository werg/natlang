import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { MemoryNeuraleseStore, validateNativeAdapterStateConsumption } from '../../dist/native/neuralese-store.js';
import { setRecorderSource } from '../../dist/neuralese/recording.js';
import { HttpNeuraleseStore, neuraleseServerModelTurn } from '../../dist/model/neuralese-server.js';
import { nativeTextOperation, archiveNativeState } from '../../dist/model/native-state-client.js';
import { saveNz, loadNz } from '../../dist/native/nz-file.js';
import { neuraleseRef } from '../../dist/native/neuralese.js';
const [endpoint, fixturePath] = process.argv.slice(2);
const body = JSON.parse(await readFile(fixturePath, 'utf8'));
const store = new MemoryNeuraleseStore();
const { result, meta } = await nativeTextOperation({ endpoint, store }, 'init-body', body);
const receipt = result.source_capture.imports[0].adapter_state_consumption;
assert.notDeepEqual(receipt.source_native_identity, receipt.target_native_identity);
const source = await store.getNativeState(receipt.source_state_id);
const target = await store.getNativeState(result.id);
validateNativeAdapterStateConsumption(receipt, source, target.metadata);
for (const change of [ { source_state_id: result.id }, { tokens: receipt.tokens + 1 },
  { source_native_identity: receipt.target_native_identity }, { target_adapter_binding: null } ])
  assert.throws(() => validateNativeAdapterStateConsumption({ ...receipt, ...change }, source, target.metadata));
const bytes = await saveNz({ value: { type: 'Neuralese<string>', value: neuraleseRef('Neuralese<string>', result.id) } }, { store, dialect: meta.dialect, provenance: meta.producer });
const portable = new MemoryNeuraleseStore();
await loadNz(bytes, { store: portable });
assert.ok(await portable.getNativeState(result.id));
assert.ok(await portable.getNativeState(receipt.source_state_id));
for (const id of [...receipt.source_adapter_binding.adapters, ...receipt.target_adapter_binding.adapters].map(entry => entry.id))
  assert.ok(await portable.getNativeTensorArtifact(id));
assert.deepEqual(meta.producer.native_source_capture, result.source_capture);
await archiveNativeState({endpoint,store}, body.placeholder);
await new HttpNeuraleseStore(endpoint).collect(new Set());
const restored = await nativeTextOperation({endpoint,store}, 'init-body', body);
assert.equal(restored.result.id, result.id);
const turns = [];
setRecorderSource(() => ({ record: turn => turns.push(turn) }));
const driver = neuraleseServerModelTurn({endpoint, model:'native-fixture', store, nativeStateModes:body.native_state_modes});
await driver.decide({messages:body.messages, options:['t7','t8'], adapters:body.adapters.map(id => ({id,scale:1}))});
assert.deepEqual(turns[0].native_state_modes, body.native_state_modes);
setRecorderSource(() => undefined);
console.log(JSON.stringify({ passed: true }));
