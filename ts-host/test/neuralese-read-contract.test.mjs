import assert from 'node:assert/strict';
import { test } from 'node:test';
import { NEURALESE_TYPE_DOCUMENTATION } from '../dist/compiler/intrinsics.js';
import { BUILT_IN_DOCS } from '../dist/native/runtime.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { session as open } from './support/natlang.mjs';

test('Neuralese read guidance separates the callable read operation from body metadata', () => {
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /neuralese\.read<T>\(value: Neuralese<T>\)/);
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /neuralese\.bodies\.read is only the configured reader body ID/);
  assert.match(NEURALESE_TYPE_DOCUMENTATION, /neuralese\.textReadSource is read-only provenance metadata/);
  assert.match(BUILT_IN_DOCS['neuralese.textReadSource'], /Use the supported typed operation\s+neuralese\.read\(value\)/);
  assert.match(BUILT_IN_DOCS['neuralese.textReadSource'], /descriptor itself does not perform it/);
});

test('eval neuralese.read delegates to the configured typed readout', async () => {
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
});
