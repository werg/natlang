import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { MemoryNeuraleseStore, encodeNativeState } from '../dist/native/neuralese-store.js';
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
