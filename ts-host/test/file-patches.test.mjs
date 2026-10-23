import assert from 'node:assert/strict';
import { test } from 'node:test';
import { diffFiles, applyPatch, composePatches, FilePatchError } from '../dist/native/file-patches.js';

test('ordinary source-file patches preserve diff, strict apply, merge, and conflict behavior', () => {
  const base = { 'SKILL.md': 'v1', 'notes.md': 'keep', 'old.md': 'gone soon' };
  const a = diffFiles({ ...base, 'SKILL.md': 'v2' }, base);
  const { 'old.md': _, ...withoutOld } = base;
  const b = diffFiles({ ...withoutOld, 'new.md': 'hello' }, base);
  assert.deepEqual(a.changes, { 'SKILL.md': 'v2' });
  assert.deepEqual(b.changes, { 'new.md': 'hello', 'old.md': null });
  const merged = composePatches([a, b]);
  assert.deepEqual(applyPatch(base, merged, { strict: true }), { 'SKILL.md': 'v2', 'notes.md': 'keep', 'new.md': 'hello' });
  assert.throws(() => composePatches([a, diffFiles({ ...base, 'SKILL.md': 'v3' }, base)]), /patch-conflict/);
  assert.throws(() => applyPatch({ x: 'y' }, a, { strict: true }), FilePatchError);
});
