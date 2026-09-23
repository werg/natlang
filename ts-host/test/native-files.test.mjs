import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, readFileSync, existsSync, symlinkSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { openFolder, saveFolder } from '../dist/index.js';

test('a disk folder lists once, reads contents lazily, and skips escaping symlinks', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-files-'));
  mkdirSync(join(root, 'src')); mkdirSync(join(root, 'node_modules'));
  writeFileSync(join(root, 'src', 'note.txt'), 'old');
  writeFileSync(join(root, 'node_modules', 'skip.js'), '');
  symlinkSync(tmpdir(), join(root, 'outside'));
  const folder = openFolder(root);
  writeFileSync(join(root, 'src', 'note.txt'), 'new\nsecond\nthird');
  assert.deepEqual(folder.list().map(entry => [entry.path, entry.kind]), [['src', 'folder']]);
  assert.equal(await folder.readText('src/note.txt', 2), 'second\n');
  writeFileSync(join(root, 'src', 'note.txt'), 'changed after observation');
  assert.equal(await folder.readText('src/note.txt'), 'new\nsecond\nthird', 'a read is stable once observed');
  await assert.rejects(folder.readText('outside/anything'), /not found/);
  assert.throws(() => folder.writeText('x', 'y'), /read-only/);
});

test('saveFolder writes a folder\'s changes back beneath its root', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-writes-'));
  writeFileSync(join(root, 'existing.txt'), 'old'); writeFileSync(join(root, 'gone.txt'), 'bye');
  const folder = openFolder(root, 'write');
  const transaction = await folder.beginTransaction();
  transaction.folder.writeText('nested/new.txt', 'hello');
  transaction.folder.writeText('existing.txt', 'new');
  transaction.folder.remove('gone.txt');
  await transaction.commit();
  assert.deepEqual(saveFolder(root, folder).map(change => [change.path, change.kind]),
    [['existing.txt', 'modified'], ['gone.txt', 'deleted'], ['nested/new.txt', 'added']]);
  assert.equal(readFileSync(join(root, 'nested/new.txt'), 'utf8'), 'hello');
  assert.equal(readFileSync(join(root, 'existing.txt'), 'utf8'), 'new');
  assert.equal(existsSync(join(root, 'gone.txt')), false);
  const escaping = { changes: [{ path: '../outside', kind: 'added', after: new Uint8Array([1]) }], moves: [] };
  assert.throws(() => saveFolder(root, escaping), /escapes its root/);
});
