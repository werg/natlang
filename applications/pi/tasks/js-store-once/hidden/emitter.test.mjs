import { test } from 'node:test';
import assert from 'node:assert/strict';
import { pathToFileURL } from 'node:url';
const { Emitter } = await import(pathToFileURL(`${process.cwd()}/src/emitter.js`).href);
const { createStore } = await import(pathToFileURL(`${process.cwd()}/src/store.js`).href);
test('several once listeners in a row', () => {
  const e = new Emitter(); const seen = [];
  e.once('x', () => seen.push('a')); e.once('x', () => seen.push('b')); e.on('x', () => seen.push('c')); e.once('x', () => seen.push('d'));
  e.emit('x'); e.emit('x');
  assert.deepEqual(seen, ['a', 'b', 'c', 'd', 'c']);
  assert.equal(e.count('x'), 1);
});
test('removers', () => {
  const e = new Emitter(); const seen = [];
  const stop = e.once('x', (v) => seen.push(v)); stop(); e.emit('x', 1);
  const stop2 = e.on('y', (v) => seen.push(v)); e.emit('y', 2); stop2(); e.emit('y', 3);
  assert.deepEqual(seen, [2]);
  assert.equal(e.count('x') + e.count('y'), 0);
});
test('listener unsubscribing itself during emit', () => {
  const store = createStore(0); const seen = [];
  const self = (v) => { seen.push(`self ${v}`); store.unsubscribe(self); };
  store.subscribe(self); store.subscribe((v) => seen.push(`next ${v}`));
  store.set(1); store.set(2);
  assert.deepEqual(seen, ['self 1', 'next 1', 'next 2']);
});
test('off of once fn with the same function elsewhere', () => {
  const e = new Emitter(); const seen = [];
  const f = (v) => seen.push(v);
  e.once('x', f); e.on('z', f); e.off('x', f); e.emit('x', 1); e.emit('z', 2);
  assert.deepEqual(seen, [2]);
});
