import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { test } from 'node:test';
import { readFileSync } from 'node:fs';
import { createNatlangRuntime, defineNatlang, Folder } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { validateGraph } from '../dist/native/graph.js';
import { convertTrajectory } from '../dist/compiler/neuralese-conversion.js';

const GRAPH_SCHEMA = JSON.parse(readFileSync(new URL('../../spec/neuralese-graph.schema.json', import.meta.url), 'utf8'));

const softText = defineNatlang('---\nargs:\n  notes: Neuralese<string>\nreturns: string\n---\nUse the supplied notes.\n', { name: 'softText' });
const crispText = defineNatlang('---\nargs:\n  notes: string\nreturns: string\n---\nUse the supplied notes.\n', { name: 'crispText' });
const softNumber = defineNatlang('---\nargs:\n  notes: Neuralese<number>\nreturns: string\n---\nUse the supplied notes.\n', { name: 'softNumber' });
const nestedSoftText = defineNatlang('---\nargs: { payload: { notes: Neuralese<string> } }\nreturns: string\n---\nUse the supplied notes.\n', { name: 'nestedSoftText' });
const success = () => Object.assign(async () => ({ calls: [['return_result', { status: 'success', value: 'accepted' }]] }), { neuralese: true });

function writer() {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const writes = [];
  const originalWrite = port.write.bind(port);
  port.write = async (text, options) => { writes.push({ text, options }); return originalWrite(text, options); };
  return { store, port, writes };
}

test('a crisp string at a declared Neuralese<string> argument boundary is written with caller provenance', async () => {
  const { store, port, writes } = writer();
  const traces = [];
  const runtime = createNatlangRuntime({ model: success(), neuralese: { store, port }, trace: trace => traces.push(trace) });
  const text = '<|neuralese|>literal marker-looking text is payload<|/neuralese|>';
  assert.equal(await runtime.run(() => softText(text)), 'accepted');
  assert.equal(writes.length, 1);
  assert.equal(writes[0].text, text, 'the argument text is preserved exactly, without marker decoding');
  assert.equal(writes[0].options.type, 'Neuralese<string>');
  assert.deepEqual(writes[0].options.producer, {
    source_kind: 'typed-text-argument', source: 'call-argument', marker_context: 'typed-argument',
    call_id: writes[0].options.producer.call_id, argument_type: 'Neuralese<string>',
    text_body_sha256: writes[0].options.producer.text_body_sha256, argument_name: 'notes',
  });

  const events = traces[0].events;
  const write = events.find(event => event.kind === 'block_write');
  const start = events.find(event => event.kind === 'invocation' && event.phase === 'start');
  assert.equal(write.source_kind, 'typed-text-argument');
  assert.equal(write.inputs[0].node, 'external', 'a root call has an external caller argument source');
  assert.ok(start.inputs.some(input => input.node === write.node && input.block === write.block),
    'the invocation consumes the materialized block from its write node');
  assert.notEqual(write.inputs[0].node, start.node, 'the source edge does not point at the child that consumes the block');
  assert.deepEqual(validateGraph(events, GRAPH_SCHEMA), []);
});

test('a nested soft call writes from the parent model turn without a graph cycle', async () => {
  const { store, port } = writer();
  const parent = defineNatlang('---\nargs: { text: string }\nreturns: string\n---\nCall the inline reader.\n', { name: 'parent' });
  const requests = [];
  const code = 'const child: Neuralese<(notes: Neuralese<string>) => Promise<string>> = nl.with({})`Read the supplied notes.`;\n' +
    'return await child(text);';
  const model = Object.assign(async request => {
    requests.push(request);
    if (requests.length === 1) return { calls: [['eval', { code }]] };
    return { calls: [['return_result', { status: 'success', value: `answer-${requests.length}` }]] };
  }, { neuralese: true });
  const traces = [];
  const runtime = createNatlangRuntime({ model, neuralese: { store, port }, trace: trace => traces.push(trace) });
  assert.match(await runtime.run(() => parent('caller text')), /^answer-/);
  const child = traces.find(trace => trace.parentCallId);
  assert.ok(child, 'the inline reader ran as a nested invocation');
  const parentTrace = traces.find(trace => trace.callId === child.parentCallId);
  assert.ok(parentTrace);
  const write = child.events.find(event => event.kind === 'block_write' && event.source_kind === 'typed-text-argument');
  assert.ok(write);
  assert.equal(write.inputs[0].node, `call:${parentTrace.callId}`, 'the block write points to its caller invocation');
  const start = child.events.find(event => event.kind === 'invocation' && event.phase === 'start');
  assert.ok(start.inputs.some(input => input.node === write.node && input.block === write.block));
  assert.notEqual(write.inputs[0].node, start.node, 'the block write does not depend on the consuming child invocation');
  assert.deepEqual(validateGraph(traces.flatMap(trace => trace.events), GRAPH_SCHEMA), []);
});

test('only direct exact Neuralese<string> parameters are lifted; refs and crisp parameters stay unchanged', async () => {
  const { store, port, writes } = writer();
  const runtime = createNatlangRuntime({ model: success(), neuralese: { store, port } });
  assert.equal(await runtime.run(() => crispText('ordinary text')), 'accepted');
  assert.equal(writes.length, 0);

  const existing = await port.write('already soft');
  writes.length = 0;
  assert.equal(await runtime.run(() => softText(neuraleseRef('Neuralese<string>', existing.id))), 'accepted');
  assert.equal(writes.length, 0, 'an existing reference is passed through');

  writes.length = 0;
  await assert.rejects(() => runtime.run(() => softNumber('not a soft number')), /type-mismatch/);
  assert.equal(writes.length, 0, 'other Neuralese element types are not guessed from strings');

  await assert.rejects(() => runtime.run(() => nestedSoftText({ payload: { notes: 'nested text' } })), /type-mismatch/);
  assert.equal(writes.length, 0, 'a nested field is not lifted by the direct-parameter convenience');
});

test('a missing writer rejects before calling the model', async () => {
  let modelCalls = 0;
  const runtime = createNatlangRuntime({ model: async () => { modelCalls++; return { text: 'unexpected' }; } });
  await assert.rejects(() => runtime.run(() => softText('plain text')), /neuralese-unsupported-backend.*Neuralese<string>/);
  assert.equal(modelCalls, 0);
});

test('a failed typed argument write releases a directory reducer transaction', async () => {
  const reducer = defineNatlang('---\nkind: directory-reducer\nargs:\n  notes: Neuralese<string>\nreturns: string\n---\nUse the supplied notes.\n',
    { name: 'softReducer' });
  const folder = Folder.fromFiles({ 'keep.txt': 'original' });
  const runtime = createNatlangRuntime({ model: async () => { throw new Error('model must not run'); } });
  await assert.rejects(() => runtime.run(() => folder.apply(reducer, 'plain input')),
    /neuralese-unsupported-backend.*Neuralese<string>/);
  assert.equal(await folder.readText('keep.txt'), 'original');
  folder.writeText('after.txt', 'transaction is released');
  assert.equal(await folder.readText('after.txt'), 'transaction is released');
});

test('the converter selects a modern result receipt but excludes the same action reclassified as an argument write', () => {
  const body = 'exact returned text';
  const bodySha = createHash('sha256').update(body).digest('hex');
  const args = { status: 'success', value: body };
  const resultReceipt = {
    schema: 'natlang.typed-result-write/1', trajectory_id: 'boundary-run', source_row_sha256: 'row-sha',
    invocation_id: 'call-1', writer_call_id: 'call-1', writer_node: 'call-1#9',
    block_id: `nz1_${'a'.repeat(52)}`, source_kind: 'typed-text-result', source: 'return_result',
    result_type: 'Neuralese<string>', marker_context: 'return-result', result_path: ['return'],
    body_source: body, body_sha256: bodySha, body_source_basis: 'exact-raw-model-result-string',
    request_sha256: 'b'.repeat(64), raw_response_sha256: 'c'.repeat(64),
  };
  const makeRow = receipt => ({
    id: 'boundary-row',
    source_ref: { trajectory_id: 'boundary-run', source_row_sha256: 'row-sha', invocation_id: 'call-1' },
    messages: [],
    target: { role: 'assistant', tool_calls: [{ id: 'return-1', type: 'function', function: {
      name: 'return_result', arguments: JSON.stringify(args),
    } }] },
    decision: { index: 0, assistant: { calls: [{ source_tool: 'return_result', arguments: args,
      outcome: { name: 'return_result', arguments: args, typed_result_writes: [receipt] },
    }] } },
  });

  const selected = convertTrajectory(makeRow(resultReceipt)).record;
  const selectedArgs = JSON.parse(selected.target.tool_calls[0].function.arguments);
  assert.equal(selectedArgs.value.$write.block_id, resultReceipt.block_id);
  assert.equal(selectedArgs.value.$write.type, 'Neuralese<string>');
  assert.equal(selectedArgs.value.$write.source, body);
  assert.match(selectedArgs.value.$write.name, /^(?:typed-result:|soft-state:)/);

  const argumentReceipt = { ...resultReceipt, source_kind: 'typed-text-argument', argument_type: 'Neuralese<string>' };
  delete argumentReceipt.result_type;
  const excluded = convertTrajectory(makeRow(argumentReceipt)).record;
  const excludedArgs = JSON.parse(excluded.target.tool_calls[0].function.arguments);
  assert.equal(excludedArgs.value, body, 'argument provenance does not become a return-result write');
  assert.equal(excluded.neuralese_conversion.sites['typed-result-write']?.converted ?? 0, 0);
});
