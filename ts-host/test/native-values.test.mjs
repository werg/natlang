import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType } from '../dist/index.js';
import { MISSING, Reject, buildPending, coerce, dump, problems, unboundParts } from '../dist/native/values.js';

test('native values distinguish missing, Null and empty records', () => {
  const type = parseType('{ name: Text, note?: Text }');
  const draft = coerce({}, type, new TypeEnv(), 'return');
  assert.deepEqual(draft, {});
  assert.deepEqual(problems(draft, type, new TypeEnv(), 'return').holes.map(d => d.path), ['return/name']);
  assert.deepEqual(coerce({ name: 'Ada', note: null }, type, new TypeEnv()), { name: 'Ada' });
  assert.throws(() => coerce({ name: null }, type, new TypeEnv()), Reject);
  assert.notEqual(MISSING, null);
});

test('native pending construction preserves typed Lambda and combinator parts', () => {
  const lam = buildPending({ $lambda: { type: 'Lambda<{ item: Num }, Num>', code: 'return args.item * 2;' } });
  assert.equal(lam.nodeKind, 'lambda');
  assert.equal(lam.body, 'return args.item * 2;\n');
  assert.deepEqual(unboundParts(lam, new TypeEnv(), '').map(d => d.path), ['/args/item']);
  const map = buildPending({ $map: { type: 'Map<Num, Num>', over: [1, 2],
    fn: { $lambda: { type: 'Lambda<{ item: Num }, Num>', code: 'return args.item * 2;' } } } });
  assert.equal(map.nodeKind, 'map');
  assert.equal(map.fn.nodeKind, 'lambda');
  assert.equal(dump(map).$map.type, 'Map<Num, Num>');
  assert.throws(() => buildPending({ $map: { type: 'Map<Num, Num>', over: ['x'] } }), Reject);
});
