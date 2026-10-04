/** Residual updates (LEARNING_CONTINUUM.md §5): block deltas and file patches. */
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { MemoryNeuraleseStore } from '../dist/native/neuralese-store.js';
import { neuraleseRef } from '../dist/native/neuralese.js';
import { diff, apply, compose, blockFloats, diffFiles, applyPatch, composePatches, DeltaError } from '../dist/neuralese/deltas.js';

const store = new MemoryNeuraleseStore();
const stores = { read: id => store.get(id), write: store };
const block = async (values, dialect = 'nd:natlang@1', type = 'Neuralese<string>', dtype = 'f32') => {
  const data = dtype === 'f32' ? new Uint8Array(new Float32Array(values).buffer) : (() => {
    const out = new Uint8Array(values.length * 2), view = new DataView(out.buffer), scratch = new DataView(new ArrayBuffer(4));
    values.forEach((v, i) => { scratch.setFloat32(0, v); view.setUint16(i * 2, scratch.getUint32(0) >>> 16, true); });
    return out; })();
  const meta = await store.put({ dialect, length: values.length / 2, width: 2, dtype, data, type });
  return neuraleseRef(type, meta.id);
};
const floats = async ref => [...blockFloats(await store.get(ref.$neuralese.id))];

test('diff, apply and compose: base + scale · delta, deltas sum, and a delta is not a value', async () => {
  const base = await block([1, 2, 3, 4]), after = await block([2, 2, 1, 4]);
  const delta = await diff(stores, after, base);
  assert.equal((await store.get(delta.$neuralese.id)).meta.dialect, 'nd:natlang@1#delta');
  assert.equal(delta.$neuralese.type, 'Delta<Neuralese<string>>');
  assert.deepEqual(await floats(delta), [1, 0, -2, 0]);
  assert.equal((await apply(stores, base, delta)).$neuralese.id, after.$neuralese.id, 'apply(base, diff(after, base)) = after');
  assert.deepEqual(await floats(await apply(stores, base, delta, 0.5)), [1.5, 2, 2, 4]);
  assert.equal(await apply(stores, base, delta, 0), base, 'scale 0 is the base itself');
  const other = await diff(stores, await block([1, 3, 3, 5]), base);
  assert.deepEqual(await floats(await compose(stores, [delta, { delta: other, scale: 2 }])), [1, 2, -2, 2]);
  await assert.rejects(() => apply(stores, base, after), /needs a delta/);
  await assert.rejects(() => diff(stores, delta, base), /takes values/);
  await assert.rejects(async () => diff(stores, await block([1, 2]), base), /delta-shape/);
  await assert.rejects(async () => diff(stores, await block([1, 2, 3, 4], 'other@1'), base), /delta-dialect/);
});

test('bf16 blocks decode for arithmetic; adapter deltas keep the adapter spec', async () => {
  const half = await block([1.5, -2, 0.25, 8], 'nd:natlang@1', 'Neuralese<string>', 'bf16');
  assert.deepEqual(await floats(half), [1.5, -2, 0.25, 8]);
  const spec = 'adapter/1;base=abc;kind=xs;r=1;u=0;layers=6-6;targets=out;seed=0';
  const zero = await block([0, 0, 0, 0], spec, 'Adapter'), tuned = await block([0.1, 0, 0, -0.2], spec, 'Adapter');
  const delta = await diff(stores, tuned, zero);
  assert.equal((await store.get(delta.$neuralese.id)).meta.dialect, spec + '#delta');
  const halfway = await apply(stores, zero, delta, 0.5);
  assert.equal(halfway.$neuralese.type, 'Adapter');
  assert.equal((await store.get(halfway.$neuralese.id)).meta.dialect, spec);
});

test('file patches: diff, apply, merge; conflicting patches do not compose', () => {
  const base = { 'SKILL.md': 'v1', 'notes.md': 'keep', 'old.md': 'gone soon' };
  const a = diffFiles({ ...base, 'SKILL.md': 'v2' }, base);
  const { 'old.md': _, ...withoutOld } = base;
  const b = diffFiles({ ...withoutOld, 'new.md': 'hello' }, base);
  assert.deepEqual(a.changes, { 'SKILL.md': 'v2' });
  assert.deepEqual(b.changes, { 'new.md': 'hello', 'old.md': null });
  const merged = composePatches([a, b]);
  assert.deepEqual(applyPatch(base, merged, { strict: true }), { 'SKILL.md': 'v2', 'notes.md': 'keep', 'new.md': 'hello' });
  assert.throws(() => composePatches([a, diffFiles({ ...base, 'SKILL.md': 'v3' }, base)]), /patch-conflict/);
  assert.throws(() => applyPatch({ x: 'y' }, a, { strict: true }), DeltaError);
});
