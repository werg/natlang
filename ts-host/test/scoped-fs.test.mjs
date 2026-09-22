import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, FolderBusyError, FolderConflictError } from '../dist/index.js';

test('folder handles expose the same overlay and selected-install behavior', async () => {
  const folder = Folder.fromFiles({ 'src/a.ts': 'one\n', 'src/b.ts': 'two\n', 'README.md': 'guide' });
  assert.deepEqual((await folder.list('src')).map(entry => entry.path), ['src/a.ts', 'src/b.ts']);
  assert.deepEqual((await folder.dir('src').folders()).map(entry => entry.relativePath), []);
  const walked = [];
  for await (const entry of folder.root().walk()) walked.push(entry.relativePath);
  assert.deepEqual(walked, ['README.md', 'src', 'src/a.ts', 'src/b.ts']);
  assert.equal(await folder.file('src/a.ts').readText(1, 1), 'one\n');
  assert.deepEqual((await folder.search('two')).map(match => match.path), ['src/b.ts']);

  const child = folder.fork();
  await child.file('src/a.ts').editText('one', 'ONE');
  await child.file('src/b.ts').moveTo(child.dir('src/archive'));
  await child.file('new.txt').writeText('new');
  await child.file('README.md').remove();
  const diff = await child.diff();
  assert.deepEqual(diff.changes.map(item => [item.path, item.kind]), [
    ['README.md', 'deleted'], ['new.txt', 'added'], ['src/a.ts', 'modified'],
    ['src/archive/b.ts', 'added'], ['src/b.ts', 'deleted']]);
  assert.deepEqual(diff.moves, [['src/b.ts', 'src/archive/b.ts']]);
  assert.equal(await folder.file('src/a.ts').readText(), 'one\n');

  const installed = await folder.installFrom(child, ['src/**']);
  assert.deepEqual(installed.changes.map(item => item.path), ['src/a.ts', 'src/archive/b.ts', 'src/b.ts']);
  assert.equal(await folder.file('src/a.ts').readText(), 'ONE\n');
  assert.equal(await folder.file('src/archive/b.ts').readText(), 'two\n');
  assert.equal(await folder.file('README.md').readText(), 'guide');
});

test('fuzzy edits and path safety are aligned with the Python surface', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'alpha   beta\ngamma\n' });
  await folder.file('a.txt').editText('alpha beta', 'changed', true);
  assert.equal(await folder.file('a.txt').readText(), 'changed\ngamma\n');
  assert.throws(() => folder.file('../a.txt'));
  assert.throws(() => folder.file('a\\b'));
  assert.throws(() => folder.writeText('/absolute', 'bad'));
});

test('edit diagnostics distinguish missing and ambiguous spans', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'alpha\nalpha\nbeta\n' });
  await assert.rejects(folder.file('a.txt').editText('missing', 'changed'),
    /no matching span; retry with fuzzy: true/);
  await assert.rejects(folder.file('a.txt').editText('alpha', 'changed'),
    /found 2 matching spans; make find more specific/);
  await assert.rejects(folder.file('a.txt').editText('missing', 'changed', true),
    /no matching span, even with fuzzy: true/);
});

test('selected installs are atomic and writer locks exclude competing writers', async () => {
  const folder = Folder.fromFiles({ a: 'a', b: 'b' });
  const child = folder.fork();
  await child.file('a').writeText('A');
  await child.file('b').writeText('B');
  await folder.file('a').writeText('changed behind child');
  await assert.rejects(folder.installFrom(child), FolderConflictError);
  assert.equal(await folder.file('b').readText(), 'b');

  const release = await folder.writer().acquire(false);
  try { await assert.rejects(folder.writer().acquire(false), FolderBusyError); }
  finally { release(); }
});

test('a transaction holds the parent lock and installs without reacquiring it', async () => {
  const folder = Folder.fromFiles({ a: 'a', b: 'b' });
  const transaction = await folder.beginTransaction(false);
  await transaction.folder.file('a').writeText('A');
  await assert.rejects(folder.writer().acquire(false), FolderBusyError);
  const installed = await transaction.commit(['a']);
  assert.deepEqual(installed.changes.map(item => item.path), ['a']);
  assert.equal(await folder.file('a').readText(), 'A');
  assert.equal(await folder.file('b').readText(), 'b');

  const aborted = await folder.beginTransaction(false);
  await aborted.folder.file('b').writeText('B');
  aborted.abort();
  assert.equal(await folder.file('b').readText(), 'b');

  const nested = await folder.dir('nested').beginTransaction(false);
  await nested.folder.file('inside.txt').writeText('scoped');
  await nested.commit();
  assert.equal(await folder.file('nested/inside.txt').readText(), 'scoped');
});
