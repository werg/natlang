import test from 'node:test';
import { createNativeStateFixturePort } from './fixtures/native-state-port.mjs';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'node:http';
import { neuraleseServerModelTurn, requestNativeStateIds } from '../dist/model/neuralese-server.js';
import { nativeStateRuntimeOptions } from '../dist/model/native-state-client.js';
import { join } from 'node:path';
import { MemoryNeuraleseStore, encodeNativeState } from '../dist/native/neuralese-store.js';
import { neuraleseRef, neuraleseSentinel, isNeuraleseRef, soleSentinel, sourceWithLiteralCalls,
  textToParts, partsToText, sentinelIds, decodeTurnValue, dialectBinding, nativeStateConsumerFromInfo } from '../dist/native/neuralese.js';
import { TypeEnv, parseType } from '../dist/native/types.js';
import { coerce } from '../dist/native/values.js';
import { saveNz, loadNz, loadNzSync, registerImportedNativeArtifacts, adoptImportedNativeArtifacts } from '../dist/native/nz-file.js';
import { FileNeuraleseStore } from '../dist/neuralese/node-block-store.js';
import { OpfsNeuraleseStore } from '../dist/browser/neuralese-opfs-store.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { session as open } from './support/natlang.mjs';
import { fakeOpfs } from './fixtures/fake-opfs.mjs';
import { nativeStateFixture } from './fixtures/native-state.mjs';

const runtimeFor = (store, architecture='fixture') => ({ store, dialect: `native-state/${architecture}` });
const envFor = options => { const env = new TypeEnv(); env.dialects = dialectBinding(options); return env; };

test('native server runtime binding supplies the real typed-state writer and refuses unsupported required writers', async () => {
  const store = new MemoryNeuraleseStore();
  const info = { dialects: ['native-state/fixture'], capabilities: ['native.text-write'], native_state: {
    enabled: true, architecture: 'fixture', checkpoint_sha256: 'a'.repeat(64), tokenizer_fingerprint: 'b'.repeat(64),
    embedding_weight_sha256: 'c'.repeat(64), consumption_modes: ['exact-native-state'], role_marker_ids: [1, 2, 3, 4],
  } };
  const runtime = await nativeStateRuntimeOptions({ endpoint: 'http://native-fixture', store, info });
  assert.equal(runtime.store, store);
  assert.equal(runtime.dialect, 'native-state/fixture');
  assert.equal(runtime.port.dialect, 'native-state/fixture');
  assert.equal(runtime.nativeStateConsumer.checkpoint_sha256, 'a'.repeat(64));

  const readOnly = await nativeStateRuntimeOptions({ endpoint: 'http://native-fixture', store,
    info: { ...info, capabilities: [] } }, { write: 'available' });
  assert.equal(readOnly.port, undefined);
  await assert.rejects(() => nativeStateRuntimeOptions({ endpoint: 'http://native-fixture', store,
    info: { ...info, capabilities: [] } }), /neuralese-native-text-write-unavailable/);
  await assert.rejects(() => nativeStateRuntimeOptions({ endpoint: 'http://native-fixture', store,
    info: { ...info, capabilities: undefined } }), /neuralese-native-text-write-unavailable/);
  await assert.rejects(() => nativeStateRuntimeOptions({ endpoint: 'http://native-fixture', store,
    info: { ...info, native_state: { ...info.native_state, consumption_modes: undefined } } }, { write: 'available' }),
    /neuralese-native-state-unavailable/);
});

test('short native ID in historical tool arguments stays ordinary data', () => {
  const full = 'nzs1_' + 'a'.repeat(52);
  const short = { type: 'native_state', id: 'nzs1_w42ancp7k5' };
  const valid = { type: 'native_state', id: full };
  const history = value => [{ role: 'assistant', tool_calls: [{ function: {
    name: 'return_result', arguments: JSON.stringify({ status: 'success', value }),
  } }] }];
  assert.deepEqual(requestNativeStateIds(history([short])), []);
  assert.deepEqual(requestNativeStateIds(history([valid])), [full]);
  assert.deepEqual(requestNativeStateIds(history([short, valid])), []);
  assert.deepEqual(requestNativeStateIds(history([valid, valid])), [full]);
  assert.deepEqual(requestNativeStateIds(history({ nested: [short] })), []);
  assert.deepEqual(requestNativeStateIds(history({ nested: [valid] })), [full]);
  assert.equal(short.id, 'nzs1_w42ancp7k5');
});

for (const mixed of [false, true]) test(`malformed historical state arguments preserve the next transport request (mixed=${mixed})`, async t => {
  const requests = [];
  const parts = [{ type: 'native_state', id: 'nzs1_w42ancp7k5' }];
  if (mixed) parts.push({ type: 'native_state', id: 'nzs1_' + 'a'.repeat(52) });
  const original = JSON.stringify({ status: 'success', value: parts });
  t.mock.method(globalThis, 'fetch', async (url, options) => {
    if (String(url).endsWith('/v1/neuralese/info'))
      return new Response(JSON.stringify({ native_state: { enabled: true }, capabilities: ['chat'], stream: false }));
    assert.ok(String(url).endsWith('/v1/chat/completions'), 'invalid history must not trigger artifact lookup');
    requests.push(JSON.parse(options.body));
    return new Response(JSON.stringify({ choices: [{ message: { role: 'assistant', content: 'Recovered.' },
      finish_reason: 'stop' }] }));
  });
  const driver = neuraleseServerModelTurn({ endpoint: 'http://native-fixture', model: 'fixture' });
  const reply = await driver({ messages: [{ role: 'assistant', tool_calls: [{ id: 'prior', type: 'function',
    function: { name: 'return_result', arguments: original } }] },
    { role: 'tool', tool_call_id: 'prior', content: 'Return the required ordinary string array.' }], tools: [] });
  assert.equal(reply.text, 'Recovered.');
  assert.equal(requests.length, 1);
  assert.deepEqual(JSON.parse(requests[0].messages[0].tool_calls[0].function.arguments), JSON.parse(original));
});

test('native state is the existing typed value through parts, nested values, eval and soft bodies', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const ref = neuraleseRef('Neuralese<string>', artifact.id), marker = neuraleseSentinel(artifact.id);
  assert.ok(isNeuraleseRef(ref)); assert.equal(soleSentinel(marker), artifact.id);
  assert.deepEqual(sentinelIds(`prefix${marker}suffix`), [artifact.id]);
  assert.deepEqual(textToParts(marker), [{ type: 'native_state', id: artifact.id }]);
  assert.equal(partsToText(textToParts(marker)), marker);
  assert.equal(await decodeTurnValue([{ type: 'native_state', id: artifact.id }], undefined, {}), marker);
  assert.equal(sourceWithLiteralCalls(`return ${marker};`), `return __neuralese(${JSON.stringify(artifact.id)});`);
  assert.equal(sourceWithLiteralCalls('`'+marker+'`'), '`'+'${__neuralese.body('+JSON.stringify(artifact.id)+')}`');
  const result = coerce({ memo: marker }, parseType('{ memo: Neuralese<string> }'), envFor(runtimeFor(store)));
  assert.equal(result.memo.$neuralese.id, artifact.id);
  assert.equal(store.peek(artifact.id).dialect, 'native-state/fixture');
});

test('foreign native default values require declared adaptation; named dialects stay strict', async () => {
  const store = new MemoryNeuraleseStore(), source = await store.putNativeState(nativeStateFixture('source'));
  const ref = neuraleseRef('Neuralese<string>', source.id), runtime = runtimeFor(store, 'target');
  assert.throws(() => coerce(ref, parseType('Neuralese<string>'), envFor(runtime)), /neuralese-native-consumer-unavailable/);
  runtime.nativeStateConsumer = nativeStateConsumerFromInfo({ native_state: { enabled: true,
    ...nativeStateFixture('target').metadata, consumption_modes: ['exact-native-state', 'best-effort-text-retokenization'] } });
  assert.equal(coerce(ref, parseType('Neuralese<string>'), envFor(runtime)), ref);
  assert.throws(() => coerce(neuraleseRef('Neuralese<string, "native-state/target">', source.id),
    parseType('Neuralese<string, "native-state/target">'), envFor(runtime)), /neuralese-dialect-mismatch/);
  assert.throws(() => nativeStateConsumerFromInfo({ native_state: { enabled: true } }), /identity/);
});

test('typed child return reaches parent as native-state content without writing or flattening it', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const neuralese = runtimeFor(store), child = open({ type: '() => Neuralese<string>', instructions: 'Produce a memo.' }, { neuralese }).session;
  const produce = Object.assign(async () => ({ calls: [['return_result', { status: 'success',
    value: [{ type: 'native_state', id: artifact.id }] }]] }), { neuralese: true });
  await new NativeToolAgent(produce, { maxTurns: 2, neuralese }).run(child);
  assert.equal(child.lam.return.$neuralese.id, artifact.id);
  const parent = open({ type: '(memo: Neuralese<string>) => string', instructions: 'Consume the child memo.',
    args: { memo: child.lam.return } }, { neuralese }).session;
  let imports = [];
  const consume = Object.assign(async request => {
    imports = request.messages.flatMap(message => Array.isArray(message.content) ? message.content : []);
    return { calls: [['return_result', { status: 'success', value: 'consumed' }]] };
  }, { neuralese: true });
  await new NativeToolAgent(consume, { maxTurns: 2, neuralese }).run(parent);
  assert.equal(parent.lam.return, 'consumed');
  assert.ok(imports.some(part => part.type === 'native_state' && part.id === artifact.id));
  assert.ok(!JSON.stringify(imports).includes(artifact.metadata.text), 'the producer text is not flattened into the parent prompt');
});

test('assistant native completion is decoded against the declared return type', async () => {
  const store = new MemoryNeuraleseStore(), neuralese = runtimeFor(store);
  const makeDriver = async text => {
    const artifact = await store.putNativeState({ ...nativeStateFixture(), metadata: {
      ...nativeStateFixture().metadata, text,
    } });
    const marker = neuraleseSentinel(artifact.id);
    const parts = [{ type: 'native_state', id: artifact.id }];
    const driver = Object.assign(async () => ({ text: marker, calls: [], raw_response: {
      choices: [{ message: { role: 'assistant', content: parts } }],
    } }), { neuralese: true });
    return { driver, artifact, marker };
  };
  const run = async (type, sourceText, expected) => {
    const { driver, artifact } = await makeDriver(sourceText);
    const session = open({ type: `() => ${type}`, instructions: 'Return the requested value.' }, { neuralese }).session;
    await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(session);
    assert.deepEqual(session.lam.return, expected);
    return artifact;
  };

  await run('string', 'ordinary text result', 'ordinary text result');
  await run('number', '42', 42);
  await run('{ count: number; label: string }', '{"count":7,"label":"ready"}', { count: 7, label: 'ready' });
  const unknown = await makeDriver('opaque value');
  const unknownSession = open({ type: '() => unknown', instructions: 'Return the requested value.' }, { neuralese }).session;
  await new NativeToolAgent(unknown.driver, { maxTurns: 2, neuralese }).run(unknownSession);
  assert.equal(unknownSession.lam.return.$neuralese.id, unknown.artifact.id);

  const listArtifacts = await Promise.all(['first', 'second'].map(async text => store.putNativeState({
    ...nativeStateFixture(), metadata: { ...nativeStateFixture().metadata, text },
  })));
  const listParts = listArtifacts.map(artifact => ({ type: 'native_state', id: artifact.id }));
  const listDriver = Object.assign(async () => ({ calls: [], raw_response: {
    choices: [{ message: { role: 'assistant', content: listParts } }],
  } }), { neuralese: true });
  const listSession = open({ type: '() => Neuralese<string>[]', instructions: 'Return both values.' }, { neuralese }).session;
  await new NativeToolAgent(listDriver, { maxTurns: 2, neuralese }).run(listSession);
  assert.deepEqual(listSession.lam.return.map(value => value.$neuralese.id), listArtifacts.map(value => value.id));

  const recordArtifact = await store.putNativeState({ ...nativeStateFixture(), metadata: {
    ...nativeStateFixture().metadata, text: 'record memo',
  } });
  const recordDriver = Object.assign(async () => ({ calls: [['return_result', { status: 'success', value: {
    memo: [{ type: 'native_state', id: recordArtifact.id }],
  } }]] }), { neuralese: true });
  const recordSession = open({ type: '() => { memo: Neuralese<string>[] }', instructions: 'Return the memo.' },
    { neuralese }).session;
  await new NativeToolAgent(recordDriver, { maxTurns: 2, neuralese }).run(recordSession);
  assert.deepEqual(recordSession.lam.return.memo.map(value => value.$neuralese.id), [recordArtifact.id]);

  const native = await makeDriver('soft memo');
  const softSession = open({ type: '() => Neuralese<string>', instructions: 'Return the requested soft value.' },
    { neuralese }).session;
  await new NativeToolAgent(native.driver, { maxTurns: 2, neuralese }).run(softSession);
  assert.equal(softSession.lam.return.$neuralese.id, native.artifact.id);

  const mixedArtifact = await store.putNativeState({ ...nativeStateFixture(), metadata: {
    ...nativeStateFixture().metadata, text: 'middle',
  } });
  const mixedParts = [{ type: 'text', text: 'before ' }, { type: 'native_state', id: mixedArtifact.id },
    { type: 'text', text: ' after' }];
  const mixedDriver = Object.assign(async () => ({ text: partsToText(mixedParts), calls: [], raw_response: {
    choices: [{ message: { role: 'assistant', content: mixedParts } }],
  } }), { neuralese: true });
  const mixedSession = open({ type: '() => string', instructions: 'Return the requested text.' }, { neuralese }).session;
  await new NativeToolAgent(mixedDriver, { maxTurns: 2, neuralese }).run(mixedSession);
  assert.equal(mixedSession.lam.return, 'before middle after');
});

test('native typed artifacts survive file/OPFS restart, pin lifecycle, .nz and compiled-import adoption', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-native-store-'));
  const { storage } = fakeOpfs();
  try {
    for (const create of [() => FileNeuraleseStore.open(directory, { residentBytes: 0 }),
      () => OpfsNeuraleseStore.open({ storage, residentBytes: 0 })]) {
      const store = await create(), artifact = await store.putNativeState(nativeStateFixture());
      const reopened = await create();
      assert.equal(reopened.peek(artifact.id).length, 2);
      assert.deepEqual(encodeNativeState(await reopened.getNativeState(artifact.id)), encodeNativeState(artifact));
      await reopened.pin(artifact.id); assert.deepEqual(await reopened.collect(new Set()), []);
      const ref = neuraleseRef('Neuralese<string>', artifact.id);
      const wire = await saveNz({ memo: { type: 'Neuralese<string>', value: ref } }, { store: reopened, dialect: 'native-state/fixture' });
      const restored = new MemoryNeuraleseStore(), loaded = await loadNz(wire, { store: restored });
      assert.equal(loaded.exports.memo.$neuralese.id, artifact.id);
      assert.deepEqual(encodeNativeState(await restored.getNativeState(artifact.id)), encodeNativeState(artifact));
      const compiled = loadNzSync(wire), adopted = new MemoryNeuraleseStore();
      registerImportedNativeArtifacts(compiled.nativeStates.values(), compiled.nativeTensors.values()); await adoptImportedNativeArtifacts(adopted);
      assert.equal((await adopted.getNativeState(artifact.id)).id, artifact.id);
      const corrupted = wire.slice(); corrupted[corrupted.length-1] ^= 1;
      assert.throws(() => loadNzSync(corrupted), /neuralese-block-integrity/);
      await reopened.unpin(artifact.id); assert.deepEqual(await reopened.collect(new Set()), [artifact.id]);
      assert.equal(await reopened.hasNativeState(artifact.id), false);
    }
    const file = await FileNeuraleseStore.open(directory, { residentBytes: 0 });
    const artifact = await file.putNativeState(nativeStateFixture());
    const wire = await readFile(join(directory, artifact.id)); wire[wire.length-1] ^= 1; await writeFile(join(directory, artifact.id), wire);
    await assert.rejects(() => file.getNativeState(artifact.id), /neuralese-block-integrity/);
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('native server capability gaps fail before sending unsupported guidance or typed templates', async t => {
  const { neuraleseServerModelTurn } = await import('../dist/model/neuralese-server.js');
  let chatRequests = 0;
  t.mock.method(globalThis, 'fetch', async url => {
    if (String(url).endsWith('/v1/neuralese/info'))
      return new Response(JSON.stringify({ dialects: ['native-state/fixture'], capabilities: ['chat'], stream: false }));
    chatRequests++; throw new Error('unsupported operation reached the server');
  });
  const driver = neuraleseServerModelTurn({ endpoint: 'http://native-fixture', model: 'fixture' });
  const messages = [{ role: 'user', content: 'Use the typed interface.' }];
  await assert.rejects(() => driver({ messages, tools: [], guidance: true }), /neuralese-guidance-unavailable/);
  await assert.rejects(() => driver({ messages, tools: [], template: { call: 'return_result', value: 'write', value_type: 'string' } }),
    /neuralese-template-unavailable/);
  assert.equal(chatRequests, 0);
});

test('zero-token native identity preserves exact layer layout through persistent and nz archives', async () => {
  const input = nativeStateFixture(); input.metadata.text = '';
  for (const tensor of Object.values(input.tensors)) { tensor.shape[0] = 0; tensor.data = new Uint8Array(); }
  const store = new MemoryNeuraleseStore(), empty = await store.putNativeState(input);
  const bytes = await saveNz({ identity: { type: 'Neuralese<string>', value: neuraleseRef('Neuralese<string>', empty.id) } },
    { store, dialect: 'native-state/fixture' });
  const loaded = await loadNz(bytes, { store });
  assert.equal(loaded.nativeStates.get(empty.id).tensors.token_ids.shape[0], 0);
  const directory = await mkdtemp(join(tmpdir(), 'native-empty-'));
  try {
    const disk = await FileNeuraleseStore.open(directory, { residentBytes: 0 });
    await disk.putNativeState(input);
    const restored = await (await FileNeuraleseStore.open(directory)).getNativeState(empty.id);
    assert.equal(restored.id, empty.id);
    for (const [name, tensor] of Object.entries(restored.tensors)) {
      assert.deepEqual(tensor.shape, empty.tensors[name].shape);
      assert.equal(tensor.data.length, 0);
    }
  } finally { await rm(directory, { recursive: true, force: true }); }
});

test('shared semantic defaults bind their actual adapter and preserve strict named types', async () => {
  const store = new MemoryNeuraleseStore(), input = nativeStateFixture('source');
  input.metadata = { ...input.metadata, schema: 'neuralese.semantic-state/1', layer_kinds: [],
    source_model: 'source', adapter_sha256: 'a'.repeat(64), offsets: [[0,3], [3,5]], posterior: 'mean', offset_policy: 'exact-tokenizer-offsets' };
  for (const name of Object.keys(input.tensors)) if (name.startsWith('layers.')) delete input.tensors[name];
  input.tensors.code = { dtype: 'F32', shape: [2, 2], data: new Uint8Array(new Float32Array([1,2,3,4]).buffer) };
  const compact = await store.putNativeState(input), raw = await store.putNativeState(nativeStateFixture('source'));
  assert.equal(store.peek(compact.id).dialect, 'semantic-state/' + 'a'.repeat(64));
  const consumer = nativeStateConsumerFromInfo({ native_state: { enabled: true, ...nativeStateFixture('target').metadata,
    semantic_adapter_sha256: 'a'.repeat(64), consumption_modes: ['learned-semantic-state-deltas'] } });
  const runtime = { store, dialect: 'semantic-state/' + 'a'.repeat(64), nativeStateConsumer: consumer };
  for (const id of [raw.id, compact.id]) {
    const ref = neuraleseRef('Neuralese<string>', id);
    assert.equal(coerce(ref, parseType('Neuralese<string>'), envFor(runtime)), ref);
  }
  assert.throws(() => coerce(neuraleseRef('Neuralese<string>', raw.id),
    parseType('Neuralese<string, "semantic-state/' + 'a'.repeat(64) + '">'), envFor(runtime)), /dialect-mismatch/);
  const wrong = { ...runtime, dialect: 'semantic-state/' + 'b'.repeat(64),
    nativeStateConsumer: { ...consumer, semantic_adapter_sha256: 'b'.repeat(64) } };
  assert.throws(() => coerce(neuraleseRef('Neuralese<string>', compact.id), parseType('Neuralese<string>'), envFor(wrong)), /consumer-unavailable/);
});

test('shared semantic standard libraries retain their dialect and archived zero identity', async () => {
  const { COMBINATORS, loadStandardLibrary, createNeuraleseLibrary } = await import('../dist/neuralese/combinators.js');
  const store = new MemoryNeuraleseStore(), input = nativeStateFixture();
  input.metadata = { ...input.metadata, schema: 'neuralese.semantic-state/1', layer_kinds: [],
    source_model: 'source', adapter_sha256: 'c'.repeat(64), offsets: [[0,3], [3,5]], posterior: 'mean', offset_policy: 'exact-tokenizer-offsets' };
  for (const name of Object.keys(input.tensors)) if (name.startsWith('layers.')) delete input.tensors[name];
  input.tensors.code = { dtype: 'F32', shape: [2, 2], data: new Uint8Array(new Float32Array([1,2,3,4]).buffer) };
  const body = await store.putNativeState(input), emptyInput = structuredClone(input);
  emptyInput.metadata.text = ''; emptyInput.metadata.offsets = [];
  for (const tensor of Object.values(emptyInput.tensors)) { tensor.shape[0] = 0; tensor.data = new Uint8Array(); }
  const empty = await store.putNativeState(emptyInput), exports = {};
  for (const [name, entry] of Object.entries(COMBINATORS)) {
    const type = `Neuralese<${entry.type}>`;
    exports[name] = { type, value: { kind: 'soft-function', type, body: body.id, captures: {} } };
  }
  exports.empty = { type: 'Neuralese<unknown>', value: neuraleseRef('Neuralese<unknown>', empty.id) };
  const dialect = 'semantic-state/' + 'c'.repeat(64);
  const bytes = await saveNz(exports, { store, dialect });
  const library = await loadStandardLibrary(bytes, store);
  assert.equal(library.dialect, dialect);
  assert.equal(createNeuraleseLibrary(library).empty().$neuralese.id, empty.id);
});


for (const format of ['native_state']) {
  test(`${format} transport retains its identity at nested value boundaries and rejects ordinary strings`, async () => {
    const store = new MemoryNeuraleseStore();
    const port = createNativeStateFixturePort(store);
    const artifact = await port.write('opaque native payload');
    const options = { store, port, dialect: port.dialect };
    const parts = [{ type: format, id: artifact.id }], env = envFor(options);
    const decoded = await decodeTurnValue({ rows: [{ memo: parts }] }, port, { marker_context: 'return-result' });
    assert.deepEqual(decoded.rows[0].memo, parts);
    const result = coerce(decoded, parseType('{ rows: { memo: Neuralese<string> }[] }'), env);
    assert.equal(result.rows[0].memo.$neuralese.id, artifact.id);
    for (const type of ['string', 'Blob', '{ type: string, id: string }[]'])
      assert.throws(() => coerce(parts, parseType(type), env), /type-mismatch/);
    const unknown = coerce(parts, parseType('unknown'), env);
    assert.ok(isNeuraleseRef(unknown)); assert.equal(unknown.$neuralese.id, artifact.id);
    const union = coerce(parts, parseType('string | Neuralese<string>'), env);
    assert.equal(union.$neuralese.id, artifact.id);
    const mixed = [{ type: 'text', text: 'prefix ' }, ...parts];
    for (const type of ['string', 'Neuralese<string>', 'unknown'])
      assert.throws(() => coerce(mixed, parseType(type), env), /type-mismatch/);
    const literal = neuraleseSentinel(artifact.id);
    assert.equal(coerce(literal, parseType('string'), env), literal);
    assert.equal(await decodeTurnValue([{ type: 'text', text: literal }], port,
      { marker_context: 'other-tool-argument' }), literal);
    assert.equal(await decodeTurnValue(parts, port, { marker_context: 'assistant-text' }), literal);
    assert.equal(await decodeTurnValue(parts, port, { marker_context: 'eval-code' }), literal);
  });

  test(`${format} genuine producer output cannot escape as an opaque-ID string result`, async () => {
    const store = new MemoryNeuraleseStore();
    const port = createNativeStateFixturePort(store);
    const artifact = await port.write('opaque payload');
    const neuralese = { store, port, dialect: port.dialect };
    const invocation = open({ type: '() => string', instructions: 'Return the answer text.' }, { neuralese }).session;
    let turns = 0, feedback = '';
    const driver = Object.assign(async request => {
      turns++; if (turns === 2) feedback = JSON.stringify(request.messages);
      return { calls: [['return_result', { status: 'success', value: turns === 1 ?
        [{ type: format, id: artifact.id }] : 'actual answer' }]] };
    }, { neuralese: true });
    await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(invocation);
    assert.equal(turns, 2); assert.equal(invocation.lam.return, 'actual answer');
    assert.match(feedback, /type-mismatch/);
  });
}

test('nested declared literal strings remain exact while typed legacy literals retain transport provenance', async () => {
  const store = new MemoryNeuraleseStore(), port = createNativeStateFixturePort(store);
  const neuralese = { store, port, dialect: port.dialect }, env = envFor(neuralese);
  const literal = '<|neuralese|>literal user text<|/neuralese|>';
  const type = parseType('{ literal: string, items: { memo: Neuralese<string> }[], tags: { [key: string]: string } }');
  const decoded = await decodeTurnValue({ literal, items: [{ memo: '<|neuralese|>memo<|/neuralese|>' }],
    tags: { source: literal } }, port, { marker_context: 'return-result' }, [], { type, env });
  assert.equal(decoded.literal, literal); assert.equal(decoded.tags.source, literal);
  assert.equal(decoded.items[0].memo[0].type, 'native_state');
  const value = coerce(decoded, type, env);
  assert.equal(value.literal, literal); assert.ok(isNeuraleseRef(value.items[0].memo));
});


test('declared union discriminants preserve the selected nested literal/Neuralese contract', async () => {
  const store = new MemoryNeuraleseStore(), port = createNativeStateFixturePort(store);
  const env = envFor({ store, port, dialect: port.dialect });
  const type = parseType('{ mode: "state", memo: Neuralese<string> } | { mode: "text", memo: string }');
  const literal = '<|neuralese|>literal<|/neuralese|>';
  const text = await decodeTurnValue({ mode: 'text', memo: literal }, port,
    { marker_context: 'return-result' }, [], { type, env });
  assert.equal(coerce(text, type, env).memo, literal);
  const state = await decodeTurnValue({ mode: 'state', memo: literal }, port,
    { marker_context: 'return-result' }, [], { type, env });
  assert.ok(isNeuraleseRef(coerce(state, type, env).memo));
  const refined = { kind: 'refined', base: parseType('string'), predicate: 'an exact literal' };
  assert.equal(await decodeTurnValue(literal, port, { marker_context: 'return-result' }, [],
    { type: refined, env }), literal);
  const untrusted = { kind: 'untrusted', base: parseType('string') };
  assert.equal(await decodeTurnValue(literal, port, { marker_context: 'return-result' }, [],
    { type: untrusted, env }), literal);
  const id = state.memo[0].id;
  const argument = await decodeTurnValue({ nested: [[{ type: 'native_state', id }]] }, port,
    { marker_context: 'other-tool-argument' });
  assert.equal(coerce(argument, parseType('{ nested: Neuralese<string>[] }'), env).nested[0].$neuralese.id, id);
});

test('typed native-state completion preserves refined, untrusted, and union-list alternatives', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const env = envFor(runtimeFor(store));
  const parts = [{ type: 'native_state', id: artifact.id }];
  const refined = parseType('Is<Neuralese<string>, "a useful memo">');
  const refinedValue = await decodeTurnValue(parts, undefined, { marker_context: 'return-result' }, [],
    { type: refined, env });
  assert.deepEqual(refinedValue, parts);
  assert.equal(coerce(refinedValue, refined, env).$neuralese.id, artifact.id);

  const untrusted = { kind: 'untrusted', base: parseType('Neuralese<string>') };
  const untrustedValue = await decodeTurnValue(parts, undefined, { marker_context: 'return-result' }, [],
    { type: untrusted, env });
  assert.deepEqual(untrustedValue, parts);
  assert.equal(coerce(untrustedValue, untrusted, env).$neuralese.id, artifact.id);

  const unionList = parseType('Neuralese<string>[] | null');
  const listValue = await decodeTurnValue(parts, undefined, { marker_context: 'return-result' }, [],
    { type: unionList, env });
  assert.deepEqual(listValue, [neuraleseSentinel(artifact.id)]);
  assert.equal(coerce(listValue, unionList, env)[0].$neuralese.id, artifact.id);
});


for (const format of ['native_state']) for (const channel of ['tool', 'assistant']) {
  test(`${format} ${channel} real HTTP normalization preserves producer provenance through agent rejection`, async () => {
    const store = new MemoryNeuraleseStore();
    const artifact = await store.putNativeState(nativeStateFixture());
    const neuralese = runtimeFor(store);
    const requests = [];
    let replyMode = 'block';
    const server = createServer(async (request, response) => {
      if (request.url === '/v1/neuralese/info') {
        response.setHeader('content-type', 'application/json');
        response.end(JSON.stringify({ dialects: [neuralese.dialect], stream: false })); return;
      }
      if (request.url?.endsWith('/meta')) {
        response.setHeader('content-type', 'application/json'); response.end('{}'); return;
      }
      if (request.url === '/v1/neuralese/native-states/' + artifact.id) {
        response.end(encodeNativeState(artifact)); return;
      }
      let body = ''; for await (const chunk of request) body += chunk;
      requests.push(JSON.parse(body));
      response.setHeader('content-type', 'application/json');
      const parts = [{ type: 'native_state', id: artifact.id }];
      const value = requests.length !== 1 ? 'actual HTTP answer' : replyMode === 'literal' ?
        neuraleseSentinel(artifact.id) : replyMode === 'mixed' ? [{ type: 'text', text: '<|neuralese|>literal text<|/neuralese|> ' }, ...parts] : parts;
      const message = channel === 'assistant' ? { role: 'assistant', content: value } :
        { role: 'assistant', content: '', tool_calls: [{ id: 'reply-' + requests.length, type: 'function',
          function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value }) } }] };
      response.end(JSON.stringify({ choices: [{ message, finish_reason: requests.length === 1 && replyMode === 'truncated' ? 'length' : 'stop' }], ...(requests.length !== 1 ? {} :
        { native_state: { blocks: [{ ...artifact.metadata, id: artifact.id }] } }) }));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    try {
      const endpoint = `http://127.0.0.1:${server.address().port}`;
      const driver = neuraleseServerModelTurn({ endpoint, model: 'exact-protocol-fixture', store });
      const invocation = open({ type: '() => string', instructions: 'Return actual answer text.' }, { neuralese }).session;
      await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(invocation);
      const assistantNativeText = channel === 'assistant';
      assert.equal(invocation.lam.return, assistantNativeText ? 'the balance is 42' : 'actual HTTP answer');
      assert.equal(requests.length, assistantNativeText ? 1 : 2);
      if (!assistantNativeText) assert.match(JSON.stringify(requests[1].messages), /type-mismatch/);
      const child = open({ type: '() => Neuralese<string>', instructions: 'Return the produced value.' }, { neuralese }).session;
      requests.length = 0;
      await new NativeToolAgent(driver, { maxTurns: 1, neuralese }).run(child);
      assert.equal(child.lam.return.$neuralese.id, artifact.id);
      if (channel === 'assistant') {
        for (replyMode of ['mixed', 'truncated', 'literal']) {
          requests.length = 0;
          const target = open({ type: '() => string', instructions: 'Return actual answer text.' }, { neuralese }).session;
          await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(target);
          const mixedNativeText = replyMode === 'mixed';
          assert.equal(target.lam.return, replyMode === 'literal' ? neuraleseSentinel(artifact.id) : mixedNativeText ?
            '<|neuralese|>literal text<|/neuralese|> the balance is 42' : 'actual HTTP answer');
          assert.equal(requests.length, replyMode === 'literal' || mixedNativeText ? 1 : 2);
          if (replyMode === 'mixed' && !mixedNativeText) assert.match(JSON.stringify(requests[1].messages), /type-mismatch/);
        }
      }
    } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  });
}

for (const format of ['native_state']) {
  test(`${format} assistant producer reply to compaction cannot finish the function`, async () => {
    const store = new MemoryNeuraleseStore();
    const note = await store.putNativeState(nativeStateFixture());
    const fixture = nativeStateFixture();
    const answer = await store.putNativeState({ ...fixture,
      metadata: { ...fixture.metadata, text: 'actual answer' } });
    const neuralese = runtimeFor(store);
    const invocation = open({ type: '() => Neuralese<string>', instructions: 'Return the actual answer.' }, { neuralese }).session;
    let turns = 0, compactions = 0;
    const driver = Object.assign(async request => {
      turns++;
      const prompt_tokens = Math.round((JSON.stringify(request.messages).length + JSON.stringify(request.tools).length) / 4);
      if (request.tools.map(tool => tool.function.name).join() === 'compact_history') {
        compactions++;
        return { text: neuraleseSentinel(note.id), prompt_tokens,
          raw_response: { choices: [{ message: { role: 'assistant', content: [{ type: format, id: note.id }] } }] } };
      }
      return turns < 10 ? { calls: [['eval', { code: `console.log('y'.repeat(2000)); ${turns}` }]], prompt_tokens } :
        { calls: [['return_result', { status: 'success', value: [{ type: format, id: answer.id }] }]], prompt_tokens };
    }, { neuralese: true });
    await new NativeToolAgent(driver, { contextTokens: 4096, maxTurns: 30, neuralese }).run(invocation);
    assert.ok(compactions > 0, 'the actual agent entered its compaction-only turn');
    assert.equal(invocation.lam.return.$neuralese.id, answer.id);
  });
}

test('per-use native constants survive parts, contextual coercion, eval and .nz soft bodies', async () => {
  const { stopGradient } = await import('../dist/neuralese/learning.js');
  const { neuraleseRefSentinel, soleSentinelUse } = await import('../dist/native/neuralese.js');
  const { softFunction, softFunctionOf, save, nzExports, Context } = await import('../dist/runtime/contexts.js');
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const live = neuraleseRef('Neuralese<string>', artifact.id), constant = stopGradient(live);
  assert.equal(live.$neuralese.stop_gradient, undefined);
  assert.equal(constant.$neuralese.stop_gradient, true);
  const marker = neuraleseRefSentinel(constant);
  assert.deepEqual(soleSentinelUse(marker), {id:artifact.id,stop_gradient:true});
  assert.deepEqual(textToParts(marker), [{type:'native_state',id:artifact.id,stop_gradient:true}]);
  assert.equal(partsToText(textToParts(marker)), marker);
  assert.equal(coerce(marker,parseType('Neuralese<string>'),envFor(runtimeFor(store))).$neuralese.stop_gradient,true);
  assert.equal(sourceWithLiteralCalls(`return ${marker};`),`return __neuralese(${JSON.stringify(artifact.id)}, true);`);
  assert.equal(sourceWithLiteralCalls('`'+marker+'`'),'`${__neuralese.body('+JSON.stringify(artifact.id)+', true)}`');
  assert.notEqual(Context.empty().with({memo:live}).id,Context.empty().with({memo:constant}).id);
  const fn = softFunction({type:'() => string',body:constant});
  assert.equal(softFunctionOf(fn).bodyStopGradient,true);
  const wire = await save({memo:constant,fn},{store,dialect:'native-state/fixture'});
  const loaded = await loadNz(wire), exports = nzExports(loaded);
  assert.equal(exports.memo.$neuralese.stop_gradient,true);
  assert.equal(softFunctionOf(exports.fn).bodyStopGradient,true);
  assert.equal(exports.memo.$neuralese.id,live.$neuralese.id);
});

test('agent request and graph preserve a native constant use without changing the artifact', async () => {
  const { stopGradient } = await import('../dist/neuralese/learning.js');
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const constant=stopGradient(neuraleseRef('Neuralese<string>',artifact.id));
  const neuralese=runtimeFor(store), parent=open({type:'(memo: Neuralese<string>) => string',instructions:'Use the memo.',
    args:{memo:constant}},{neuralese}).session;
  let uses=[];
  const consume=Object.assign(async request=>{
    uses=request.messages.flatMap(message=>Array.isArray(message.content)?message.content:[]).filter(part=>part.type==='native_state');
    return {calls:[['return_result',{status:'success',value:'done'}]]};
  },{neuralese:true});
  await new NativeToolAgent(consume,{maxTurns:2,neuralese}).run(parent);
  assert.equal(uses.length,1);assert.deepEqual(uses[0],{type:'native_state',id:artifact.id,stop_gradient:true});
  const reads=parent.runtime.trace.events.filter(event=>event.kind==='block_read');
  assert.equal(reads.length,1);assert.equal(reads[0].stop_gradient,true);
  assert.equal((await store.getNativeState(artifact.id)).id,artifact.id);
});

test('equal native IDs retain independent writer origins through typed values, contexts and archives', async () => {
  const { neuraleseRefSentinel, soleSentinelUse }=await import('../dist/native/neuralese.js');
  const { save,nzExports,Context,softFunction,softFunctionOf }=await import('../dist/runtime/contexts.js');
  const { blockInput }=await import('../dist/native/graph.js');
  const { stopGradient }=await import('../dist/neuralese/learning.js');
  const store=new MemoryNeuraleseStore(),artifact=await store.putNativeState(nativeStateFixture());
  const first=neuraleseRef('Neuralese<string>',artifact.id,false,'call:first#write-0');
  const second=neuraleseRef('Neuralese<string>',artifact.id,false,'call:second#write-0');
  const stopped=stopGradient(first);
  assert.equal(stopped.$neuralese.source_occurrence,first.$neuralese.source_occurrence);
  assert.notEqual(neuraleseRefSentinel(first),neuraleseRefSentinel(second));
  assert.deepEqual(soleSentinelUse(neuraleseRefSentinel(stopped)),{id:artifact.id,stop_gradient:true,source_occurrence:'call:first#write-0'});
  assert.equal(blockInput(artifact.id,'read',first.$neuralese.source_occurrence).node,'external',
    'an occurrence without current execution evidence is not attributed to a writer');
  assert.notEqual(Context.empty().with({memo:first}).id,Context.empty().with({memo:second}).id);
  const body=softFunction({type:'() => string',body:stopped});
  const wire=await save({first,second,body},{store,dialect:'native-state/fixture'});
  const exports=nzExports(await loadNz(wire));
  assert.equal(exports.first.$neuralese.source_occurrence,'call:first#write-0');
  assert.equal(exports.second.$neuralese.source_occurrence,'call:second#write-0');
  assert.equal(softFunctionOf(exports.body).bodySourceOccurrence,'call:first#write-0');
  assert.equal(sourceWithLiteralCalls(`return ${neuraleseRefSentinel(first)};`),`return __neuralese(${JSON.stringify(artifact.id)}, undefined, "call:first#write-0");`);
});

test('actual eval compiler preserves native constant and source metadata in a typed result', async()=>{
  const {neuraleseRefSentinel}=await import('../dist/native/neuralese.js');
  const store=new MemoryNeuraleseStore(),artifact=await store.putNativeState(nativeStateFixture());
  const reference=neuraleseRef('Neuralese<string>',artifact.id,true,'exact-source#write');
  const target=open({type:'() => Neuralese<string>',instructions:'Return the typed memo.'},{neuralese:runtimeFor(store)}).session;
  const result=await target.applyAsync('eval',{code:`const memo: Neuralese<string> = ${neuraleseRefSentinel(reference)}; memo;`,finish:true});
  assert.deepEqual(target.lam.return,reference,JSON.stringify(result));
});

test('tool JSON history records every actual native import with its writer and gradient use', async () => {
  const store=new MemoryNeuraleseStore(),artifact=await store.putNativeState(nativeStateFixture());
  const invocation=open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const agent=new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  const part={type:'native_state',id:artifact.id,source_occurrence:'actual-writer#6',stop_gradient:true};
  const messages=[{role:'assistant',tool_calls:[{function:{name:'return_result',arguments:JSON.stringify({value:[part],nested:{again:[part]}})}}]}];
  const before=structuredClone(messages);
  const {graphNode}=await import('../dist/native/graph.js');
  graphNode(invocation.runtime.trace,'block_write',{call_id:'actual-writer',turn:'actual-writer#turn6',block:artifact.id},[],'actual-writer#6');
  await agent.modelTurnNode(invocation,messages,{},2);
  const reads=invocation.runtime.trace.events.filter(event=>event.kind==='block_read');
  assert.equal(reads.length,2);
  assert.ok(reads.every(event=>event.block===artifact.id&&event.stop_gradient===true&&event.inputs[0].node==='actual-writer#6'));
  assert.deepEqual(messages,before);
  const {nativeCaptureGraphBinding}=await import('../dist/native/model-recording.js');
  const name=invocation.runtime.trace.events.find(event=>event.kind==='model_turn').node;
  const nodes=new Map(invocation.runtime.trace.events.filter(event=>event.node).map(event=>[event.node,event]));
  assert.equal(nativeCaptureGraphBinding(name,{prompt:[{kind:'native_state',id:artifact.id,source_occurrence:'actual-writer#6',stop_gradient:true},{kind:'native_state',id:artifact.id,source_occurrence:'actual-writer#6',stop_gradient:true}]},nodes).read_occurrences.length,2);
  const prior=invocation.runtime.trace.events.filter(event=>event.kind==='block_read').length;
  await agent.modelTurnNode(invocation,[{tool_calls:[{function:{arguments:JSON.stringify({singleton:part,text:JSON.stringify(part)})}}]}],{},3);
  assert.equal(invocation.runtime.trace.events.filter(event=>event.kind==='block_read').length,prior);
  await assert.rejects(()=>agent.modelTurnNode(invocation,[{tool_calls:[{function:{arguments:'not json'}}]}],{},4),SyntaxError);
});

test('captured sampler prompt adds omitted external native context to the whole-turn graph', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const invocation = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const agent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  const prompt = [{kind:'native_state',id:artifact.id,native_identity:{
    architecture:artifact.metadata.architecture,checkpoint_sha256:artifact.metadata.checkpoint_sha256,
    tokenizer_fingerprint:artifact.metadata.tokenizer_fingerprint,
    embedding_weight_sha256:artifact.metadata.embedding_weight_sha256}}, {kind:'tokens',token_ids:[1,2]}];
  await agent.modelTurnNode(invocation,[{role:'system',content:'context was supplied outside the message list'}],{
    raw_response:{x_natlang_rl_capture:{prompt,writes:[]}},text:'',calls:[]},1);
  const read=invocation.runtime.trace.events.find(event=>event.kind==='block_read');
  assert.ok(read);
  assert.equal(read.block,artifact.id);
  assert.equal(read.inputs[0].node,'external');
  assert.deepEqual(read.positions,[0,0]);
  assert.equal(read.position_source,'captured_prompt_segment');
  const turn=invocation.runtime.trace.events.find(event=>event.kind==='model_turn');
  assert.equal(turn.inputs.filter(input=>input.port==='read').length,1);
});

test('captured prompt reconciles existing native reads and rejects unexplained graph reads', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const invocation = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const agent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  const part={type:'native_state',id:artifact.id};
  await agent.modelTurnNode(invocation,[{role:'user',content:[part]}],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'native_state',id:artifact.id}],writes:[]}},text:'',calls:[]},1);
  assert.equal(invocation.runtime.trace.events.filter(event=>event.kind==='block_read').length,1);
  const other=await store.putNativeState(nativeStateFixture('second-fixture'));
  await assert.rejects(()=>agent.modelTurnNode(invocation,[{role:'user',content:[{type:'native_state',id:other.id}]}],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'tokens',token_ids:[1]}],writes:[]}},text:'',calls:[]},2),
  /request graph contains native reads absent from the captured model prompt/);
});

test('captured-only native reads bind a unique in-episode writer and reject an unknown declared writer', async () => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture());
  const invocation = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const agent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  const {graphNode}=await import('../dist/native/graph.js');
  const writer = graphNode(invocation.runtime.trace,'block_write',{call_id:'producer',turn:'producer#turn1',block:artifact.id},[],'producer#write1');
  await agent.modelTurnNode(invocation,[],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'native_state',id:artifact.id}],writes:[]}},text:'',calls:[]},1);
  const read=invocation.runtime.trace.events.find(event=>event.kind==='block_read');
  assert.equal(read.inputs[0].node,writer);
  const second = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const secondAgent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  await assert.rejects(()=>secondAgent.modelTurnNode(second,[],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'native_state',id:artifact.id,source_occurrence:'missing#writer'}],writes:[]}},
    text:'',calls:[]},1), /names no matching in-episode writer/);
});

test('captured native reads retain the exact completed sibling writer and reject a prior execution writer', async (t) => {
  const store = new MemoryNeuraleseStore(), artifact = await store.putNativeState(nativeStateFixture('sibling-occurrence'));
  const invocation = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store)}).session;
  const agent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  const {blockInput,graphNode,hasProducerOccurrence,registerTrace,releaseTrace}=await import('../dist/native/graph.js');
  const {NativeTraceRecorder}=await import('../dist/native/trace.js');
  const { setRecorderSource } = await import('../dist/native/model-recording.js');
  const root = invocation.runtime.options.runId.split('/')[0];
  const firstWriter = `${root}/sibling-1#write`;
  const laterWriter = `${root}/sibling-2#write`;
  const firstTrace = new NativeTraceRecorder({run_id:`${root}/sibling-1`});
  const laterTrace = new NativeTraceRecorder({run_id:`${root}/sibling-2`});
  registerTrace(`${root}/sibling-1`, firstTrace, invocation.runtime.episodeBudget);
  registerTrace(`${root}/sibling-2`, laterTrace, invocation.runtime.episodeBudget);
  graphNode(firstTrace, 'block_write', {call_id:`${root}/sibling-1`,turn:'sibling-1#turn1',block:artifact.id}, [], firstWriter);
  graphNode(laterTrace, 'block_write', {call_id:`${root}/sibling-2`,turn:'sibling-2#turn1',block:artifact.id}, [], laterWriter);
  assert.equal(hasProducerOccurrence(artifact.id, firstWriter, invocation.runtime.episodeBudget), true);
  assert.equal(blockInput(artifact.id, 'block', firstWriter, invocation.runtime.episodeBudget).node, firstWriter);
  releaseTrace(`${root}/sibling-1`); releaseTrace(`${root}/sibling-2`);
  await agent.modelTurnNode(invocation,[],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'native_state',id:artifact.id,source_occurrence:firstWriter}],writes:[]}},
    text:'',calls:[]},1);
  const read=invocation.runtime.trace.events.find(event=>event.kind==='block_read');
  assert.equal(read.inputs[0].node, firstWriter, 'a later equal-content sibling does not replace the declared producer occurrence');

  const next = open({type:'() => boolean',instructions:'Return a boolean.'},{neuralese:runtimeFor(store),
    runId:invocation.runtime.options.runId}).session;
  const nextAgent = new NativeToolAgent(Object.assign(async()=>({}),{neuralese:true}),{neuralese:runtimeFor(store)});
  await assert.rejects(()=>nextAgent.modelTurnNode(next,[],{
    raw_response:{x_natlang_rl_capture:{prompt:[{kind:'native_state',id:artifact.id,source_occurrence:firstWriter}],writes:[]}},
    text:'',calls:[]},1), /names no matching in-episode writer/);

  const conflictTrace = new NativeTraceRecorder({run_id:`${root}/conflict`});
  registerTrace(`${root}/conflict`, conflictTrace, invocation.runtime.episodeBudget);
  t.after(() => { setRecorderSource(() => undefined); releaseTrace(`${root}/conflict`); });
  const reusedNode = `${root}/conflict#same`;
  const { ModelTurnRecorder } = await import('../dist/native/model-recording.js');
  const recorder = new ModelTurnRecorder();
  setRecorderSource(() => recorder);
  graphNode(conflictTrace, 'block_write', {call_id:`${root}/conflict`,turn:'conflict#turn1',block:artifact.id}, [], reusedNode);
  const other = await store.putNativeState(nativeStateFixture('different-writer-block'));
  const traceLength = conflictTrace.events.length, recorderNodes = recorder.graphNodes.size;
  assert.throws(()=>graphNode(conflictTrace, 'block_write', {call_id:`${root}/conflict`,turn:'conflict#turn2',block:other.id}, [], reusedNode),
    /writer occurrence .* reused for a different native block/);
  assert.equal(conflictTrace.events.length, traceLength, 'rejected writer did not mutate the trace');
  assert.equal(recorder.graphNodes.size, recorderNodes, 'rejected writer did not mutate the active recorder');
  setRecorderSource(() => undefined);
  releaseTrace(`${root}/conflict`);
});

test('implicit block inputs resolve writers only inside the caller execution scope', async (t) => {
  const { blockInput, graphNode, registerTrace, releaseTrace, valueInputs } = await import('../dist/native/graph.js');
  const { NativeTraceRecorder } = await import('../dist/native/trace.js');
  const block = 'soft-block:shared-value';
  const firstScope = {}, secondScope = {};
  const firstTrace = new NativeTraceRecorder({run_id:'implicit-first'});
  const secondTrace = new NativeTraceRecorder({run_id:'implicit-second'});
  registerTrace('implicit-first', firstTrace, firstScope);
  registerTrace('implicit-second', secondTrace, secondScope);
  t.after(() => { releaseTrace('implicit-first'); releaseTrace('implicit-second'); });
  const writer = graphNode(firstTrace, 'block_write', {block}, [], 'implicit-first#write');
  const nativeBlock = `nzs1_${'a'.repeat(52)}`;
  const nativeWriter = graphNode(firstTrace, 'block_write', {block:nativeBlock}, [], 'implicit-first#native-write');
  assert.equal(blockInput(block, 'arg:value', undefined, firstScope).node, writer,
    'the execution can resolve its own implicit sibling producer');
  assert.equal(blockInput(block, 'arg:value', undefined, secondScope).node, 'external',
    'another execution reusing the block ID cannot inherit the first execution writer');
  assert.equal(valueInputs(neuraleseRef('Neuralese<string>', nativeBlock, false, 'foreign#writer'), 'arg', firstScope)[0].node,
    'external', 'a stale declared occurrence cannot be rebound to the latest local producer');
  assert.equal(valueInputs(neuraleseRef('Neuralese<string>', nativeBlock, false, nativeWriter), 'arg', firstScope)[0].node,
    nativeWriter, 'an exact declared occurrence is preserved when the current scope observed it');
  assert.equal(blockInput(block, 'arg:value').node, 'external',
    'callers without execution scope do not consult ambient producer state');
});
