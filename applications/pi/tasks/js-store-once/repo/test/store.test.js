import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createStore } from '../src/store.js';

test('subscribers see changes', () => {
  const store = createStore(1);
  const seen = [];
  store.subscribe((next, previous) => seen.push([previous, next]));
  store.set(2);
  store.set(3);
  assert.deepEqual(seen, [[1, 2], [2, 3]]);
});

test('one-time subscriber does not hide the next one', () => {
  const store = createStore(0);
  const seen = [];
  store.subscribeOnce((value) => seen.push(`once ${value}`));
  store.subscribe((value) => seen.push(`always ${value}`));
  store.set(1);
  store.set(2);
  assert.deepEqual(seen, ['once 1', 'always 1', 'always 2']);
});

test('unsubscribe a one-time subscriber', () => {
  const store = createStore(0);
  const seen = [];
  const listener = (value) => seen.push(value);
  store.subscribeOnce(listener);
  store.unsubscribe(listener);
  store.set(1);
  assert.deepEqual(seen, []);
  assert.equal(store.subscribers(), 0);
});
