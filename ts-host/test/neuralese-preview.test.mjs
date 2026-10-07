import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NativeToolAgent } from '../dist/native/agent.js';
import { NodeNativeRuntime } from '../dist/node-runtime.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { TypeEnv, parseType } from '../dist/native/types.js';
import { lambda } from './support/natlang.mjs';
import { createTextNeuraleseEmulation } from '../dist/model/text-neuralese-emulation.js';
import { resolveNeuralesePreviews } from '../dist/native/neuralese-preview.js';

const TYPE = 'Neuralese<string>';
const NOTE = 'This is only a placeholder; no evidence has been reviewed.';
const label = (id, type = TYPE, body = NOTE) =>
  `[[Neuralese text block id=${id} type=${type}; exact JSON string body=${JSON.stringify(body)}]]`;

async function emulatedBlock(body = NOTE) {
  const emulation = createTextNeuraleseEmulation();
  const meta = await emulation.port.write(body, { producer: { marker_context: 'return-result', result_type: TYPE } });
  return { emulation, meta, ref: neuraleseRef(TYPE, meta.id) };
}

test('NativeToolAgent resolves a V87-shaped exact preview back to the visible typed result reference', async () => {
  const { emulation, meta, ref } = await emulatedBlock();
  const agent = new NativeToolAgent(emulation.wrap(async () => ({ calls: [['return_result', {
    status: 'success', value: label(meta.id) }]] })), { neuralese: emulation.runtime });
  const runtime = new NodeNativeRuntime({ runId: 'neuralese-preview-test', agent: session => agent.run(session) });
  const result = await runtime.run(lambda({ type: `(prior: ${TYPE}) => ${TYPE}`, args: { prior: ref },
    instructions: 'Return prior.' }));
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, ref);
  const events = runtime.trace.events.filter(event => event.kind === 'neuralese_preview_resolution');
  assert.equal(events.length, 1);
  assert.equal(events[0].schema, 'natlang.neuralese-preview-resolution/1');
  assert.deepEqual(events[0].resolutions.map(({ path, id, type }) => ({ path, id, type })),
    [{ path: 'return', id: meta.id, type: TYPE }]);
  assert.equal(events[0].resolutions[0].body_sha256, meta.producer.text_body_sha256);
});

test('NativeToolAgent resolves a nested preview while preserving raw proposal and applied action evidence', async () => {
  const { emulation, meta, ref } = await emulatedBlock();
  const rawValue = { note: label(meta.id), count: 2 };
  const agent = new NativeToolAgent(emulation.wrap(async () => ({ calls: [['return_result', {
    status: 'success', value: rawValue }]] })), { neuralese: emulation.runtime });
  const runtime = new NodeNativeRuntime({ runId: 'neuralese-preview-nested-test', agent: session => agent.run(session) });
  const result = await runtime.run(lambda({ type: `(prior: ${TYPE}) => { note: ${TYPE}; count: number }`,
    args: { prior: ref }, instructions: 'Return prior and count.' }));
  assert.equal(result.outcome.kind, 'done');
  assert.deepEqual(result.value, { note: ref, count: 2 });

  const proposal = runtime.trace.events.find(event => event.kind === 'proposal' && event.phase === 'released');
  assert.deepEqual(proposal.calls[0][1].value, rawValue, 'proposal trace keeps the model-proposed preview text');
  const action = runtime.trace.events.find(event => event.kind === 'action' && event.name === 'return_result');
  assert.deepEqual(action.arguments.value, { note: ref, count: 2 }, 'action trace records the normalized typed value actually applied');
  assert.deepEqual(runtime.trace.events.filter(event => event.kind === 'neuralese_preview_resolution')[0].resolutions
    .map(({ path, id, type }) => ({ path, id, type })), [{ path: 'return/note', id: meta.id, type: TYPE }]);
  assert.equal(runtime.trace.events.filter(event => event.kind === 'neuralese_preview_resolution').length, 1);
});

test('preview identity resolution handles a typed Neuralese leaf inside a result record', async () => {
  const { emulation, meta, ref } = await emulatedBlock();
  const value = { note: label(meta.id), count: 2 };
  const expected = parseType('{ note: Neuralese<string>; count: number }');
  const result = await resolveNeuralesePreviews(value, expected, new TypeEnv(), [{ note: ref }], emulation.store);
  assert.deepEqual(result.value, { note: ref, count: 2 });
  assert.deepEqual(result.resolutions.map(item => item.path), ['return/note']);
});

test('preview copying preserves an own __proto__ record field without changing the result prototype', async () => {
  const { emulation, meta, ref } = await emulatedBlock();
  const value = JSON.parse(`{"note":${JSON.stringify(label(meta.id))},"__proto__":"data"}`);
  const expected = parseType('{ note: Neuralese<string> }');
  const result = await resolveNeuralesePreviews(value, expected, new TypeEnv(), [{ note: ref }], emulation.store);
  assert.equal(Object.getPrototypeOf(result.value), Object.prototype);
  assert.equal(Object.hasOwn(result.value, '__proto__'), true);
  assert.equal(result.value.__proto__, 'data');
  assert.equal(result.value.note.$neuralese.id, ref.$neuralese.id);
});

test('preview resolution rejects unknown or nonvisible IDs, altered body/type, trailing text, and opaque backend blocks', async () => {
  const { emulation, meta, ref } = await emulatedBlock();
  const foreign = await emulation.port.write('A different block.', {
    producer: { marker_context: 'return-result', result_type: TYPE } });
  const expected = parseType(TYPE), env = new TypeEnv(), visible = [ref];
  const escapedEquivalent = label(meta.id).replace('"This', '"\\u0054his');
  const escaped = await resolveNeuralesePreviews(escapedEquivalent, expected, env, visible, emulation.store);
  assert.deepEqual(escaped.value, ref, 'equivalent JSON escapes resolve by decoded body identity');
  assert.equal(escaped.resolutions.length, 1);
  const rejectedCases = [
    label('nz1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'),
    label(foreign.id, TYPE, 'A different block.'),
    label(meta.id, 'Neuralese<number>'),
    `${label(meta.id)} extra`,
    `prefix ${label(meta.id)}`,
    `[[Neuralese text block id=${meta.id} type=${TYPE}; exact JSON string body=not-json]]`,
  ];
  for (const candidate of rejectedCases) {
    const result = await resolveNeuralesePreviews(candidate, expected, env, visible, emulation.store);
    assert.equal(result.value, candidate);
    assert.deepEqual(result.resolutions, []);
  }

  const opaqueStore = new MemoryNeuraleseStore();
  const opaqueMeta = await opaqueStore.put({ dialect: 'nd:server', length: 0, width: 1, dtype: 'f32',
    data: new Uint8Array(0), type: TYPE, producer: { kind: 'server' } });
  const opaqueRef = neuraleseRef(TYPE, opaqueMeta.id), opaqueLabel = label(opaqueMeta.id);
  const opaque = await resolveNeuralesePreviews(opaqueLabel, expected, env, [opaqueRef], opaqueStore);
  assert.equal(opaque.value, opaqueLabel);
  assert.deepEqual(opaque.resolutions, []);
});
