import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNativeStateFixturePort } from './fixtures/native-state-port.mjs';
import { NEURALESE_TYPE_DOCUMENTATION } from '../dist/compiler/intrinsics.js';
import { BUILT_IN_DOCS } from '../dist/native/runtime.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
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
  const readBody = 'nzs1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  let written = 0, read = 0;
  const { session } = open({ type: '() => string', instructions: 'Return the text read from the child.' }, {
    services: { neuralese: { dialect: 'test', width: 4, bodies: { read: readBody } } },
    neuralese: { store: { has: async id => id === readBody } },
    agent: child => {
      if (child.lam.type.returns.kind === 'neuralese') {
        written++;
        child.lam.return = neuraleseRef('Neuralese<string>', 'nzs1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa');
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
  const port = createNativeStateFixturePort(store);
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
  const readBody = 'nzs1_bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb';
  let written = 0, read = 0;
  const answers = ['ordinary text', 17, { status: 'ready', count: 3 }];
  const { session } = open({ type: '() => unknown', instructions: 'Return the typed values read from the children.' }, {
    services: { neuralese: { dialect: 'test', width: 4, bodies: { read: readBody } } },
    neuralese: { store: { has: async id => id === readBody } },
    agent: child => {
      if (child.lam.type.returns.kind === 'neuralese') {
        written++;
        const types = ['Neuralese<string>', 'Neuralese<number>', 'Neuralese<{ status: string; count: number }>'];
        child.lam.return = neuraleseRef(types[written - 1], `nzs1_${String.fromCharCode(96 + written).repeat(52)}`);
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

test('eval read(value) without a standard library reads the value with the executing model as a typed call', async () => {
  // Held collection runs no services.neuralese; the opaque-access error advises read(value), which then failed.
  const reads = [];
  const { session } = open({ type: '() => string', instructions: 'Return the text read from the child.' }, {
    agent: child => {
      if (child.lam.type.returns.kind === 'neuralese')
        child.lam.return = neuraleseRef('Neuralese<string>', `nzs1_${'c'.repeat(52)}`);
      else { reads.push(child); child.lam.return = 'The garden'; }
    },
  });
  const result = await session.applyAsync('eval', { code: `
    const soft = await nl<Neuralese<string>>\`Write where the ball is.\`({});
    const text: string = await read(soft);
    text;
  ` });
  assert.equal(result.kind, 'ok', result.text);
  assert.equal(result.value, 'The garden');
  assert.equal(reads.length, 1, 'one model readout call');
  assert.equal(reads[0].lam.type.params.fields[0].name, 'value');
  assert.match(String(reads[0].lam.instructions ?? reads[0].lam.body ?? ''), /Read the supplied Neuralese value/);
});

test('a Neuralese value printed or stored in eval reads as one plain line', async () => {
  const { session } = open({ type: '() => string', instructions: 'Return the text read from the child.' }, {
    agent: child => { child.lam.return = neuraleseRef('Neuralese<string>', 'nzs1_aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'); },
  });
  const result = await session.applyAsync('eval', { code: `
    const child = nl<Neuralese<string>>\`Return the answer as a soft string.\`;
    const note = await child({});
    console.log(note); console.log('note:', note); console.log({ notes: [note] });
  ` });
  assert.equal(result.kind, 'ok', result.text);
  const line = '[Neuralese<string> value: opaque in code; its content is where it was given to you, so pass it on as is, ' +
    'or write await read(value) for an ordinary copy]';
  assert.ok(result.text.includes(`console:\n${line}\nnote: ${line}\n{"notes":["${line}"]}\n`), result.text);
  assert.doesNotMatch(result.text.split('Stored local')[0], /\$neuralese|nzs1_/);
  assert.match(result.text, /local note = \S+ \(Neuralese<string>: pass it on as is, or write await read\(note\) for an ordinary copy\)/);
});

test('the wrong forms of using a Neuralese input fail fast with what to do instead', async () => {
  // held-v3 after sft-v5: calls that received a Neuralese value probed it ~7.6 model calls each (DECISIONS 2026-11-21)
  const { session } = open({ type: '(notes: Neuralese<string>) => string', instructions: 'Return the notes as text.' }, {
    agent: child => { child.lam.return = 'unused'; },
  });
  // read_code on the input's name says where its content is and how to get an ordinary copy
  const code = await session.applyAsync('read_code', { name: 'notes' });
  assert.equal(code.kind, 'rejected', code.text);
  assert.match(code.text, /notes is this call's Neuralese input: its content is in your opening message, so use it from there, or write await read\(notes\) in eval/);
  // a value that is one placeholder would be kept as those characters
  for (const value of ['${value}', '${notes}', ' ${In the notes: two open items.} ']) {
    const placeholder = await session.applyAsync('return_result', { status: 'success', value });
    assert.equal(placeholder.kind, 'rejected', placeholder.text);
    assert.match(placeholder.text, /return_result fills in no placeholder, so write out the text or value you mean/);
  }
  // a promise prints as what it is, not [object Promise]
  const pending = await session.applyAsync('eval', { code: 'const p = Promise.resolve(1); console.log(p); await p;' });
  assert.equal(pending.kind, 'ok', pending.text);
  assert.match(pending.text, /console:\n\[a Promise: not awaited yet; write await before the call to get its value\]/);
  assert.doesNotMatch(pending.text, /object Promise/);
});
