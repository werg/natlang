import assert from 'node:assert/strict';
import { test } from 'node:test';
import { runInNewContext } from 'node:vm';
import { snapshot } from '../dist/calls/index.js';

test('plain JSON crossing a compartment keeps the exact snapshot', () => {
  const foreign = runInNewContext('({ goal: "cite evidence", work: [{ input: { pages: [1] } }] })');
  const same = { goal: 'cite evidence', work: [{ input: { pages: [1] } }] };
  assert.deepEqual(snapshot(foreign, 1024), snapshot(same, 1024));
  assert.equal(snapshot(foreign, 1024).text, JSON.stringify(same));
});

test('a foreign class, custom prototype and nonportable leaves remain excluded', () => {
  for (const code of [
    'new class Object { constructor() { this.page = 1; } }()',
    'Object.create({ constructor: Object }, { page: { enumerable: true, value: 1 } })',
    '({ invalid: undefined })',
    '({ item: new Map() })',
  ]) {
    const { ref, text } = snapshot(runInNewContext(code), 1024);
    assert.equal(ref.complete, false, code);
    assert.equal(ref.reason, 'nonportable', code);
    assert.equal(text, undefined, code);
  }
});

test('sparse and decorated arrays are not silently certified as exact JSON', () => {
  for (const value of [new Array(2), [1,,3], Object.assign([1,2],{note:'must not vanish'}), runInNewContext('[1,,3]')]) {
    const taken=snapshot(value,1024);assert.equal(taken.ref.complete,false);assert.equal(taken.ref.reason,'nonportable');assert.equal(taken.text,undefined);
  }
  assert.equal(snapshot([1,null,3],1024).ref.complete,true);
});
