// A Neuralese text write is written at the end of its producing call (DECISIONS 2026-11-25): the runtime hands the
// port the call's context, the Python servers get it as write-text fields, and the grammar engines (managed
// llama-server, page) force the text after the call's reply.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { createNatlangRuntime } from '../dist/index.js';
import { loadVirtualNatlang } from '../dist/runtime/virtual-project.js';

const ID = 'nzs1_' + 'b'.repeat(52);

test('a typed Neuralese result is written at the end of the turn that returned it', async () => {
  const writes = [];
  const port = { dialect: 'native-state/test', forcedText: true, async write(text, options) {
    writes.push({ text, options });
    return { id: ID, dialect: 'native-state/test', type: 'Neuralese<string>', length: 2, width: 8 };
  } };
  const driver = Object.assign(async () => ({ calls: [['return_result', { status: 'success', value: 'Gate C opens at 06:40.' }]],
    completion_tokens: 3, raw_response: { x_natlang_rl_capture: { prompt: [{ kind: 'tokens', token_ids: [1, 2, 3] }],
      generated_inputs: [{ token_id: 9 }, { token_id: 10 }, { token_id: 11 }] } } }), { neuralese: true });
  const runtime = createNatlangRuntime({ calls: false, model: driver,
    neuralese: { store: new MemoryNeuraleseStore(), dialect: 'native-state/test', port } });
  const fn = loadVirtualNatlang({ 'note.nl': '---\nargs: {}\nreturns: Neuralese<string>\n---\nWrite the gate note.\n' }, 'note.nl');
  try { await runtime.run(() => fn()); } catch { /* the fixture state is not archived */ }
  assert.equal(writes.length, 1);
  const [{ text, options }] = writes;
  assert.equal(text, 'Gate C opens at 06:40.');
  assert.deepEqual([options.context.prompt, options.context.reply], [[1, 2, 3], [9, 10, 11]]);
  assert.ok(Array.isArray(options.context.messages) && options.context.messages.length >= 1);
});

test('the write port sends the producing call to write-text; with no ids, its messages', async () => {
  const { nativeStateWritePort } = await import('../dist/model/native-state-client.js');
  const info = { dialects: ['native-state/fixture'], capabilities: ['native.text-write'], native_state: {
    enabled: true, architecture: 'fixture', checkpoint_sha256: 'a'.repeat(64), tokenizer_fingerprint: 'b'.repeat(64),
    embedding_weight_sha256: 'c'.repeat(64), consumption_modes: ['exact-native-state'], role_marker_ids: [1, 2, 3, 4] } };
  const bodies = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    if (String(url).endsWith('/write-text')) bodies.push(JSON.parse(init.body));
    return new Response(JSON.stringify({ error: 'stop here' }), { status: 500 });
  };
  try {
    const port = await nativeStateWritePort({ endpoint: 'http://native', store: new MemoryNeuraleseStore(), info });
    await assert.rejects(port.write('v', { type: 'Neuralese<string>', context: { messages: [], tools: [], prompt: [5], reply: [6] } }));
    await assert.rejects(port.write('v', { context: { messages: [{ role: 'user', content: 'm' }], tools: [] } }));
    await assert.rejects(port.write('v'));
  } finally { globalThis.fetch = realFetch; }
  assert.deepEqual([bodies[0].prompt_token_ids, bodies[0].reply_token_ids, bodies[0].context], [[5], [6], undefined]);
  assert.deepEqual([bodies[1].messages, bodies[1].tools, bodies[1].prompt_token_ids], [[{ role: 'user', content: 'm' }], [], undefined]);
  assert.deepEqual(Object.keys(bodies[2]).sort(), ['text']);
});

test('the grammar write forces the text after the producing reply, closing an open thought first', async () => {
  const { grammarCapture, grammarWritePort } = await import('../dist/model/native-operator.js');
  const requests = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const body = init?.body ? JSON.parse(init.body) : undefined;
    requests.push({ url: String(url), body });
    if (String(url).endsWith('/tokenize'))
      return new Response(JSON.stringify({ tokens: body.content === '<think>' ? [6] : body.content === '</think>' ? [7] :
        body.content === '<|im_end|>' ? [2] : body.content === 'PROMPT' ? [1, 6] : [50, 51] }));
    if (String(url).endsWith('/apply-template')) return new Response(JSON.stringify({ prompt: 'PROMPT' }));
    return new Response(JSON.stringify({ error: 'stop here' }), { status: 500 });
  };
  const info = { native_state: { enabled: true, production: true, architecture: 'lfm3', role_marker_ids: [100, 101, 102, 103] } };
  const forced = () => requests.filter(r => r.url.endsWith('/v1/chat/completions')).at(-1).body;
  try {
    const capture = grammarCapture({ endpoint: 'http://fork', store: new MemoryNeuraleseStore(), info });
    // after a recorded reply (its end of turn dropped)
    await assert.rejects(capture({ messages: [{ role: 'user', content: 'x' }], tools: [], prompt: [1, 6], reply: [6, 7, 12, 2],
      write: 'the value' }), /forced span failed \(500\)/);
    assert.equal(forced().grammar, 'root ::= <[6]> <[7]> <[12]> <[100]> <[50]> <[51]> <[101]>');
    // no producing call: the port's own call (the text as the user's message), its empty thought closed
    const port = grammarWritePort({ endpoint: 'http://fork', store: new MemoryNeuraleseStore(), info });
    assert.equal(port.dialect, 'native-state/lfm3');
    await assert.rejects(port.write('the value'), /forced span failed \(500\)/);
    assert.equal(forced().grammar, 'root ::= <[7]> <[100]> <[50]> <[51]> <[101]>');
    assert.deepEqual(forced().messages, [{ role: 'user', content: 'the value' }]);
  } finally { globalThis.fetch = realFetch; }
});

test('a production engine without native.text-write gets the grammar write port', async () => {
  const { nativeStateRuntimeOptions } = await import('../dist/model/native-state-client.js');
  const info = { dialects: ['native-state/lfm3'], capabilities: [], native_state: { enabled: true, production: true,
    architecture: 'lfm3', checkpoint_sha256: 'a'.repeat(64), tokenizer_fingerprint: 'b'.repeat(64),
    embedding_weight_sha256: 'c'.repeat(64), consumption_modes: ['exact-native-state'], role_marker_ids: [1, 2, 3, 4] } };
  const options = await nativeStateRuntimeOptions({ endpoint: 'http://fork', store: new MemoryNeuraleseStore(), info });
  assert.equal(options.port.dialect, 'native-state/lfm3');
  assert.equal(options.port.forcedText, true);
});

test('a grammar capture of the reply asks for producer_kind capture when the server serves it; a write never does', async () => {
  const { grammarCapture } = await import('../dist/model/native-operator.js');
  const requests = [];
  const realFetch = globalThis.fetch;
  globalThis.fetch = async (url, init) => {
    const body = init?.body ? JSON.parse(init.body) : undefined;
    requests.push({ url: String(url), body });
    if (String(url).endsWith('/tokenize'))
      return new Response(JSON.stringify({ tokens: body.content === '<think>' ? [6] : body.content === '</think>' ? [7] :
        body.content === '<|im_end|>' ? [2] : [50, 51] }));
    return new Response(JSON.stringify({ error: 'stop here' }), { status: 500 });
  };
  const forced = () => requests.filter(r => r.url.endsWith('/v1/chat/completions')).at(-1).body;
  const native_state = { enabled: true, production: true, architecture: 'lfm3', role_marker_ids: [100, 101, 102, 103] };
  const call = { messages: [{ role: 'user', content: 'x' }], tools: [], prompt: [1, 6], reply: [6, 7, 12, 2] };
  try {
    const serving = grammarCapture({ endpoint: 'http://fork', store: new MemoryNeuraleseStore(),
      info: { native_state, capabilities: ['native.producer-kind.capture'] } });
    await assert.rejects(serving(call), /forced span failed \(500\)/);
    assert.equal(forced().x_natlang_producer_kind, 'capture');
    await assert.rejects(serving({ ...call, write: 'the value' }), /forced span failed \(500\)/);
    assert.equal(forced().x_natlang_producer_kind, undefined);
    // an older fork serves no capture kind: the request carries no field it would refuse
    const older = grammarCapture({ endpoint: 'http://fork', store: new MemoryNeuraleseStore(), info: { native_state, capabilities: [] } });
    await assert.rejects(older(call), /forced span failed \(500\)/);
    assert.equal(forced().x_natlang_producer_kind, undefined);
  } finally { globalThis.fetch = realFetch; }
});

test('the store keeps a capture-kind state (a reply captured in its own context) with its kind in the content id', async () => {
  const { nativeStateFixture } = await import('./fixtures/native-state.mjs');
  const { encodeNativeStateWire, decodeNativeState } = await import('../dist/native/neuralese-store.js');
  const kind = producer_kind => { const value = nativeStateFixture('lfm3'); value.metadata.producer_kind = producer_kind; return encodeNativeStateWire(value); };
  const capture = kind('capture'), span = kind('span');
  assert.notEqual(capture.id, span.id);
  assert.equal(decodeNativeState(capture.wire, capture.id).metadata.producer_kind, 'capture');
  assert.throws(() => kind('other'), /producer_kind/);
});
