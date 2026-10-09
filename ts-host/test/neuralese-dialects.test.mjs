import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { dialectBinding, neuraleseRef, neuraleseSentinel, readerDialect } from '../dist/native/neuralese.js';
import { fitsType, parseType, TypeEnv } from '../dist/native/types.js';
import { coerce, Reject } from '../dist/native/values.js';

// Dialects are static facts (DECISIONS.md 2026-10-09): DefaultDialect is the runtime's reader dialect, and a block's
// stored dialect is checked wherever a value is coerced to a Neuralese type.

const softText = defineNatlang('---\nargs:\n  notes: Neuralese<string>\nreturns: string\n---\nUse the supplied notes.\n', { name: 'softText' });
const otherText = defineNatlang('---\nargs:\n  notes: Neuralese<string, "nd:other@1">\nreturns: string\n---\nUse the supplied notes.\n', { name: 'otherText' });
const success = () => Object.assign(async () => ({ calls: [['return_result', { status: 'success', value: 'accepted' }]] }), { neuralese: true });

const boundEnv = (options) => { const env = new TypeEnv(); env.dialects = dialectBinding(options); return env; };

test('the reader dialect is the declared dialect, else the port\'s, and null for a text-only runtime', () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  assert.equal(createNatlangRuntime({ neuralese: { store, port } }).readerDialect(), 'nd:natlang@1');
  assert.equal(createNatlangRuntime({ neuralese: { store, dialect: 'nd:served@2' } }).readerDialect(), 'nd:served@2');
  assert.equal(createNatlangRuntime({ neuralese: { store } }).readerDialect(), null);
  assert.equal(createNatlangRuntime({}).readerDialect(), null);
  assert.throws(() => createNatlangRuntime({ neuralese: { store, port, dialect: 'nd:served@2' } }),
    /neuralese-dialect-mismatch: .*"nd:served@2".*"nd:natlang@1"/);
});

test('the stand-in port binds DefaultDialect to its own dialect', () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8);
  assert.equal(readerDialect({ store, port }), 'nd:standin@0');
  const env = boundEnv({ store, port });
  assert.ok(fitsType(parseType('Neuralese<string>'), parseType('Neuralese<string, "nd:standin@0">'), env));
  assert.ok(fitsType(parseType('Neuralese<string, "nd:standin@0">'), parseType('Neuralese<string>'), env));
  assert.ok(!fitsType(parseType('Neuralese<string, "nd:other@1">'), parseType('Neuralese<string>'), env));
  // Unbound (no runtime), DefaultDialect is a name of its own, as before.
  assert.ok(!fitsType(parseType('Neuralese<string>'), parseType('Neuralese<string, "nd:standin@0">')));
});

test('coercing a block of another dialect to a Neuralese type is rejected with what to do', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const foreign = await new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:other@1').write('elsewhere');
  const own = await port.write('here');
  const env = boundEnv({ store, port });
  const wanted = parseType('Neuralese<string>');
  assert.deepEqual(coerce(neuraleseRef('Neuralese<string>', own.id), wanted, env), neuraleseRef('Neuralese<string>', own.id));
  assert.throws(() => coerce(neuraleseRef('Neuralese<string>', foreign.id), wanted, env, 'notes'), error => {
    assert.ok(error instanceof Reject);
    assert.equal(error.diagnostics[0].code, 'neuralese-dialect-mismatch');
    assert.match(error.message, /notes: neuralese-dialect-mismatch, expected Neuralese<string> in dialect "nd:natlang@1"/);
    assert.match(error.message, /written in dialect "nd:other@1"/);
    assert.match(error.message, /convert\(value, "nd:natlang@1"\)/);
    return true;
  });
  // A sole block marker (a literal in a tool argument) is checked the same way.
  assert.throws(() => coerce(neuraleseSentinel(foreign.id), wanted, env), /neuralese-dialect-mismatch/);
  // A block the store does not hold is checked by its reference type only.
  const absent = (await new StandInNeuralesePort(new MemoryNeuraleseStore(), hashingEmbedder(8), 8, 'nd:other@1').write('gone')).id;
  assert.ok(coerce(neuraleseRef('Neuralese<string>', absent), wanted, env));
  // Text-only (unbound): nothing to check against.
  assert.ok(coerce(neuraleseRef('Neuralese<string>', foreign.id), wanted, boundEnv({ store })));
});

test('a call argument of another dialect fails before the model runs; one of the reader dialect is accepted', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const foreign = await new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:other@1').write('elsewhere');
  const own = await port.write('here');
  let turns = 0;
  const model = Object.assign(async (...args) => { turns++; return success()(...args); }, { neuralese: true });
  const runtime = createNatlangRuntime({ model, neuralese: { store, port } });
  assert.equal(await runtime.run(() => softText(neuraleseRef('Neuralese<string>', own.id))), 'accepted');
  assert.equal(await runtime.run(() => softText(neuraleseRef('Neuralese<string, "nd:natlang@1">', own.id))), 'accepted');
  const before = turns;
  await assert.rejects(runtime.run(() => softText(neuraleseRef('Neuralese<string>', foreign.id))),
    /neuralese-dialect-mismatch.*"nd:natlang@1".*"nd:other@1"/s);
  assert.equal(turns, before, 'the mismatch is found at the boundary, not by the model');
});

test('text at a slot of another dialect is not written by the runtime\'s port', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(8), 8, 'nd:natlang@1');
  const runtime = createNatlangRuntime({ model: success(), neuralese: { store, port } });
  await assert.rejects(runtime.run(() => otherText('plain notes')),
    /neuralese-dialect-mismatch.*"nd:other@1".*writes in dialect "nd:natlang@1"/s);
  assert.equal(store.size, 0, 'nothing was written');
  assert.equal(await runtime.run(() => softText('plain notes')), 'accepted');
  assert.equal(store.size, 1);
});
