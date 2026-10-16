import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { createServer } from 'node:http';
import { neuraleseServerModelTurn, encodeBlockBody } from '../dist/model/neuralese-server.js';
import { join } from 'node:path';
import { MemoryNeuraleseStore, encodeNativeState, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { neuraleseRef, neuraleseSentinel, isNeuraleseRef, soleSentinel, sourceWithLiteralCalls,
  textToParts, partsToText, sentinelIds, decodeTurnValue, dialectBinding, nativeStateConsumerFromInfo } from '../dist/native/neuralese.js';
import { TypeEnv, parseType } from '../dist/native/types.js';
import { coerce } from '../dist/native/values.js';
import { saveNz, loadNz, loadNzSync, registerImportedBlocks, adoptImportedBlocks } from '../dist/native/nz-file.js';
import { FileNeuraleseStore } from '../dist/neuralese/node-block-store.js';
import { OpfsNeuraleseStore } from '../dist/browser/neuralese-opfs-store.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { session as open } from './support/natlang.mjs';
import { fakeOpfs } from './fixtures/fake-opfs.mjs';
import { nativeStateFixture } from './fixtures/native-state.mjs';

const runtimeFor = (store, architecture='fixture') => ({ store, dialect: `native-state/${architecture}` });
const envFor = options => { const env = new TypeEnv(); env.dialects = dialectBinding(options); return env; };

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
      registerImportedBlocks(compiled.blocks.values(), compiled.nativeStates.values()); await adoptImportedBlocks(adopted);
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


for (const format of ['native_state', 'neuralese']) {
  test(`${format} transport retains its identity at nested value boundaries and rejects ordinary strings`, async () => {
    const store = new MemoryNeuraleseStore();
    const port = format === 'neuralese' ? new StandInNeuralesePort(store, hashingEmbedder(8), 8) : undefined;
    const artifact = port ? await port.write('opaque legacy payload') : await store.putNativeState(nativeStateFixture());
    const options = port ? { store, port, dialect: port.dialect } : runtimeFor(store);
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
    const port = format === 'neuralese' ? new StandInNeuralesePort(store, hashingEmbedder(8), 8) : undefined;
    const artifact = port ? await port.write('opaque payload') : await store.putNativeState(nativeStateFixture());
    const neuralese = port ? { store, port, dialect: port.dialect } : runtimeFor(store);
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
  const store = new MemoryNeuraleseStore(), port = new StandInNeuralesePort(store, hashingEmbedder(8), 8);
  const neuralese = { store, port, dialect: port.dialect }, env = envFor(neuralese);
  const literal = '<|neuralese|>literal user text<|/neuralese|>';
  const type = parseType('{ literal: string, items: { memo: Neuralese<string> }[], tags: { [key: string]: string } }');
  const decoded = await decodeTurnValue({ literal, items: [{ memo: '<|neuralese|>memo<|/neuralese|>' }],
    tags: { source: literal } }, port, { marker_context: 'return-result' }, [], { type, env });
  assert.equal(decoded.literal, literal); assert.equal(decoded.tags.source, literal);
  assert.equal(decoded.items[0].memo[0].type, 'neuralese');
  const value = coerce(decoded, type, env);
  assert.equal(value.literal, literal); assert.ok(isNeuraleseRef(value.items[0].memo));
});


test('declared union discriminants preserve the selected nested literal/Neuralese contract', async () => {
  const store = new MemoryNeuraleseStore(), port = new StandInNeuralesePort(store, hashingEmbedder(8), 8);
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
  const argument = await decodeTurnValue({ nested: [[{ type: 'neuralese', id }]] }, port,
    { marker_context: 'other-tool-argument' });
  assert.equal(coerce(argument, parseType('{ nested: Neuralese<string>[] }'), env).nested[0].$neuralese.id, id);
});


for (const format of ['native_state', 'neuralese']) for (const channel of ['tool', 'assistant']) {
  test(`${format} ${channel} real HTTP normalization preserves producer provenance through agent rejection`, async () => {
    const store = new MemoryNeuraleseStore();
    const port = format === 'neuralese' ? new StandInNeuralesePort(store, hashingEmbedder(8), 8) : undefined;
    const artifact = port ? await port.write('opaque transport payload') : await store.putNativeState(nativeStateFixture());
    const neuralese = port ? { store, port, dialect: port.dialect } : runtimeFor(store);
    let incidentalWrites = 0;
    if (port) { const write = port.write.bind(port); port.write = async (...args) => { incidentalWrites++; return write(...args); }; }
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
      if (request.url === '/v1/neuralese/blocks/' + artifact.id) {
        response.end(encodeBlockBody(artifact, (await store.get(artifact.id)).data)); return;
      }
      if (request.url === '/v1/neuralese/native-states/' + artifact.id) {
        response.end(encodeNativeState(artifact)); return;
      }
      let body = ''; for await (const chunk of request) body += chunk;
      requests.push(JSON.parse(body));
      response.setHeader('content-type', 'application/json');
      const parts = [{ type: format, id: artifact.id }];
      const value = requests.length !== 1 ? 'actual HTTP answer' : replyMode === 'literal' ?
        neuraleseSentinel(artifact.id) : replyMode === 'mixed' ? [{ type: 'text', text: '<|neuralese|>literal text<|/neuralese|> ' }, ...parts] : parts;
      const message = channel === 'assistant' ? { role: 'assistant', content: value } :
        { role: 'assistant', content: '', tool_calls: [{ id: 'reply-' + requests.length, type: 'function',
          function: { name: 'return_result', arguments: JSON.stringify({ status: 'success', value }) } }] };
      response.end(JSON.stringify({ choices: [{ message, finish_reason: requests.length === 1 && replyMode === 'truncated' ? 'length' : 'stop' }], ...(requests.length !== 1 ? {} : format === 'native_state' ?
        { native_state: { blocks: [{ ...artifact.metadata, id: artifact.id }] } } : { neuralese: { blocks: [artifact] } }) }));
    });
    await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
    try {
      const endpoint = `http://127.0.0.1:${server.address().port}`;
      const driver = neuraleseServerModelTurn({ endpoint, model: 'exact-protocol-fixture', store });
      const invocation = open({ type: '() => string', instructions: 'Return actual answer text.' }, { neuralese }).session;
      await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(invocation);
      assert.equal(invocation.lam.return, 'actual HTTP answer'); assert.equal(requests.length, 2);
      assert.match(JSON.stringify(requests[1].messages), /type-mismatch/);
      const child = open({ type: '() => Neuralese<string>', instructions: 'Return the produced value.' }, { neuralese }).session;
      requests.length = 0;
      await new NativeToolAgent(driver, { maxTurns: 1, neuralese }).run(child);
      assert.equal(child.lam.return.$neuralese.id, artifact.id);
      if (channel === 'assistant') {
        for (replyMode of ['mixed', 'truncated', 'literal']) {
          requests.length = 0;
          const target = open({ type: '() => string', instructions: 'Return actual answer text.' }, { neuralese }).session;
          await new NativeToolAgent(driver, { maxTurns: 2, neuralese }).run(target);
          assert.equal(target.lam.return, replyMode === 'literal' ? neuraleseSentinel(artifact.id) : 'actual HTTP answer');
          assert.equal(requests.length, replyMode === 'literal' ? 1 : 2);
          if (replyMode === 'mixed') assert.match(JSON.stringify(requests[1].messages), /type-mismatch/);
          assert.equal(incidentalWrites, 0, 'sourced text runs must not create an incidental producer');
        }
      }
    } finally { server.closeAllConnections(); await new Promise(resolve => server.close(resolve)); }
  });
}
