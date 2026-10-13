import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { test } from 'node:test';
import { externalModule } from '../dist/native/external.js';

test('trusted host service may import a Node builtin without adding Node to the browser loader', () => {
  const source = "import { createHash } from 'node:crypto'; export function digest(text: string): string { return createHash('sha256').update(text).digest('hex'); }";
  const service = externalModule('digestService', source, createRequire(import.meta.url));
  assert.equal(service.exports.digest('abc'),
    'ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad');
  assert.match(service.declaration, /digest\(text: string\): string/);
});
