import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { NodeFileTree } from '../dist/index.js';

test('NodeFileTree resolves directory metadata and file contents on demand', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-files-'));
  mkdirSync(join(root, 'src'));
  writeFileSync(join(root, 'src', 'note.txt'), 'old');
  const tree = new NodeFileTree(root);
  writeFileSync(join(root, 'src', 'note.txt'), 'new\nsecond\nthird');
  assert.deepEqual(tree.read('src'), { kind: 'directory', path: 'src',
    entries: [{ name: 'note.txt', kind: 'file', bytes: 16 }] });
  assert.deepEqual(tree.read('src/note.txt', 2, 2), { kind: 'text', path: 'src/note.txt', text: 'second',
    start: 2, end: 2, truncated: true, bytes: 16 });
});

test('NodeFileTree reports binary files and rejects escapes through symlinks', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-files-'));
  writeFileSync(join(root, 'binary'), Buffer.from([1, 0, 2]));
  symlinkSync(tmpdir(), join(root, 'outside'));
  const tree = new NodeFileTree(root);
  assert.deepEqual(tree.read('binary'), { kind: 'binary', path: 'binary', bytes: 3 });
  assert.throws(() => tree.read('outside'), /escapes its root/);
  assert.throws(() => tree.read('../outside'), /escapes its root/);
});
