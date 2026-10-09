import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NEURALESE_TYPE_DOCUMENTATION } from '../dist/compiler/intrinsics.js';
import { BUILT_IN_DOCS } from '../dist/native/runtime.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../dist/native/neuralese-store.js';
import { session as open } from './support/natlang.mjs';

test('Neuralese read guidance separates the callable read operation from body metadata', () => {
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /read<T>\(value: Neuralese<T>\)/);
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /read\(text: string\): Promise<string>/);
  assert.match(BUILT_IN_DOCS.read, /await read\(value\)/);
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /neuralese\.bodies\.read is only the configured reader body ID/);
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /neuralese\.textReadSource is read-only provenance metadata/);
  assert.match(BUILT_IN_DOCS['neuralese.textReadSource'], /Use the supported typed operation\s+read\(value\)/);
  assert.match(BUILT_IN_DOCS['neuralese.textReadSource'], /descriptor itself does not perform it/);
});

test('eval neuralese.read compatibility entry delegates to the configured typed readout', async () => {
  const readBody = 'nz1_bbbbbbbbbbbbbbbbbbbb';
  let written = 0, read = 0;
  const { session } = open({ type: '() => string', instructions: 'Return the text read from the child.' }, {
    services: { neuralese: { dialect: 'test', width: 4, bodies: { read: readBody } } },
    neuralese: { store: { has: async id => id === readBody } },
    agent: child => {
      if (child.lam.type.returns.kind === 'neuralese') {
        written++;
        child.lam.return = neuraleseRef('Neuralese<string>', 'nz1_aaaaaaaaaaaaaaaaaaaa');
      } else {
        read++;
        child.lam.return = 'answer text';
      }
    },
  });
  const declaration = await session.applyAsync('eval', { code: `await read_code('neuralese');` });
  assert.equal(declaration.kind, 'ok', declaration.text);
  assert.match(declaration.value, /read\": <T>\(value: Neuralese<T>\) => Promise<T>/);
  const result = await session.applyAsync('eval', { code: `
    const child = nl<Neuralese<string>>\`Return the answer as a soft string.\`;
    const soft = await child({});
    const text: string = await neuralese.read(soft);
    text;
  ` });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, 'answer text');
  assert.equal(written, 1);
  assert.equal(read, 1, 'the configured read body is invoked exactly once');
  const crisp = await session.applyAsync('eval', { code: 'await neuralese.read("already crisp");' });
  assert.equal(crisp.kind, 'ok', crisp.text);
  assert.equal(crisp.value, 'already crisp');
  assert.equal(read, 1, 'crisp text is an identity read and does not invoke the configured reader');
  const invalid = await session.applyAsync('eval', { code: 'await neuralese.read({ value: "not text" });' });
  assert.equal(invalid.kind, 'error');
  assert.match(invalid.text, /Neuralese reference/);
  assert.equal(read, 1, 'non-string objects are still rejected without invoking the reader');
});

test('a computed crisp child string can be returned directly as Neuralese<string>', async () => {
  const store = new MemoryNeuraleseStore();
  const port = new StandInNeuralesePort(store, hashingEmbedder(4), 4);
  const writes = [], write = port.write.bind(port);
  port.write = async (text, options) => { writes.push({ text, options }); return write(text, options); };
  const { session } = open({ type: '() => Neuralese<string>', instructions: 'Return the note.' }, {
    neuralese: { store, port },
    agent: child => { child.lam.return = 'computed note'; },
  });
  const result = await session.applyAsync('eval', { finish: true, code: `
    const update = await nl<string>\`Write the computed note.\`();
    return update;
  ` });
  assert.equal(result.kind, 'completed', result.text);
  assert.equal(session.lam.return.$neuralese.type, 'Neuralese<string>');
  assert.equal(writes.length, 1, 'the task result writer materializes the returned crisp child string once');
  assert.equal(writes[0].text, 'computed note');
});

test('eval read(value) returns typed string, number and object payloads and rejects non-Neuralese values', async () => {
  const readBody = 'nz1_bbbbbbbbbbbbbbbbbbbb';
  let written = 0, read = 0;
  const answers = ['ordinary text', 17, { status: 'ready', count: 3 }];
  const { session } = open({ type: '() => unknown', instructions: 'Return the typed values read from the children.' }, {
    services: { neuralese: { dialect: 'test', width: 4, bodies: { read: readBody } } },
    neuralese: { store: { has: async id => id === readBody } },
    agent: child => {
      if (child.lam.type.returns.kind === 'neuralese') {
        written++;
        const types = ['Neuralese<string>', 'Neuralese<number>', 'Neuralese<{ status: string; count: number }>'];
        child.lam.return = neuraleseRef(types[written - 1], `nz1_${String.fromCharCode(96 + written).repeat(20)}`);
      } else {
        child.lam.return = answers[read++];
      }
    },
  });
  const result = await session.applyAsync('eval', { code: `
    const textValue = await nl<Neuralese<string>>\`Write a text value.\`({});
    const numberValue = await nl<Neuralese<number>>\`Write a number.\`({});
    const objectValue = await nl<Neuralese<{ status: string; count: number }>>\`Write the object.\`({});
    const text: string = await read(textValue);
    const number: number = await read(numberValue);
    const object: { status: string; count: number } = await read(objectValue);
    let rejected = false;
    try { await (read as (value: unknown) => Promise<unknown>)(17); }
    catch (error) { rejected = typeof error?.message === 'string' && /Neuralese reference/.test(error.message); }
    ({ text, number, object, rejected });
  ` });
  assert.equal(result.kind, 'ok', result.text);
  assert.deepEqual(result.value, { text: 'ordinary text', number: 17,
    object: { status: 'ready', count: 3 }, rejected: true });
  assert.equal(written, 3);
  assert.equal(read, 3, 'each typed read invokes the configured reader once');
});
