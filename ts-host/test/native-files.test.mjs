import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { commitFileWrites, lazyDict, NodeFileTree, validateFileWrites } from '../dist/index.js';

test('lazy dictionaries reject a path that is both a leaf and a branch', () => {
  assert.throws(() => lazyDict({ collision: 1, 'collision/child': 2 }), /also a branch/);
});

test('NodeFileTree resolves directory metadata and file contents on demand', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-files-'));
  mkdirSync(join(root, 'src'));
  writeFileSync(join(root, 'src', 'note.txt'), 'old');
  const tree = new NodeFileTree(root);
  writeFileSync(join(root, 'src', 'note.txt'), 'new\nsecond\nthird');
  assert.deepEqual(tree.entries(), [{ name: 'src', kind: 'branch' }]);
  const src = tree.child('src');
  assert.deepEqual(src.entries(), [{ name: 'note.txt', kind: 'leaf' }]);
  assert.deepEqual(src.child('note.txt'), { kind: 'text', text: 'new\nsecond\nthird', bytes: 16 });
  writeFileSync(join(root, 'src', 'note.txt'), 'changed after observation');
  assert.deepEqual(src.child('note.txt'), { kind: 'text', text: 'new\nsecond\nthird', bytes: 16 });
});

test('NodeFileTree reports binary files and rejects escapes through symlinks', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-files-'));
  writeFileSync(join(root, 'binary'), Buffer.from([1, 0, 2]));
  symlinkSync(tmpdir(), join(root, 'outside'));
  const tree = new NodeFileTree(root);
  assert.deepEqual(tree.child('binary'), { kind: 'binary', bytes: 3 });
  assert.throws(() => tree.child('outside'), /escapes its root/);
  assert.throws(() => tree.child('../outside'), /no such host-tree entry/);
});

test('portable file-write plans validate and commit beneath their root', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-writes-'));
  writeFileSync(join(root, 'existing.txt'), 'old');
  const plan = [{ path: 'nested/new.txt', text: 'hello' }, { path: 'existing.txt', text: 'new' }];
  assert.deepEqual(validateFileWrites(plan), plan);
  assert.deepEqual(commitFileWrites(root, plan), [
    { path: 'nested/new.txt', bytes: 5, overwritten: false },
    { path: 'existing.txt', bytes: 3, overwritten: true },
  ]);
  assert.throws(() => commitFileWrites(root, [{ path: '../outside', text: 'x' }]), /invalid relative/);
  assert.throws(() => validateFileWrites([
    { path: 'same', text: 'x' }, { path: 'same', text: 'y' },
  ]), /duplicate/);
});
