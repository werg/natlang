import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder, runFolderBash, checkBashPolicy } from '../dist/index.js';

test('bash reads and writes the folder overlay, with pipelines and changed paths', async () => {
  const folder = Folder.fromFiles({ 'notes/a.txt': 'first\nsecond\n' });
  const result = await runFolderBash(folder, "rg second notes/a.txt | sed 's/second/SECOND/' > notes/result.txt");
  assert.equal(result.exitCode, 0, result.stderr);
  assert.deepEqual(result.changedPaths, ['notes/result.txt']);
  assert.match(await folder.readText('notes/result.txt'), /SECOND/);
  assert.equal((await runFolderBash(folder, 'cat notes/result.txt')).stdout.trim(), 'SECOND');
});

test('bash policy allows finite for loops and rejects open loops and recursion', () => {
  assert.doesNotThrow(() => checkBashPolicy('for f in *.md; do echo "$f"; done'));
  assert.throws(() => checkBashPolicy('while true; do echo x; done'), /While loops/);
  assert.throws(() => checkBashPolicy('for ((i=0;i<10;i++)); do echo x; done'), /CStyleFor loops/);
  assert.throws(() => checkBashPolicy('a(){ b; }; b(){ a; }; a'), /calls itself/);
});

test('bash refuses paths outside its folder root', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'a' });
  const result = await runFolderBash(folder, 'cat /etc/passwd');
  assert.notEqual(result.exitCode, 0);
  assert.match(result.stderr, /folder root|ENOENT|outside|No such file/);
  const unknown = await runFolderBash(folder, 'yes | head -n 1');
  assert.match(unknown.stderr, /yes: command not found/, 'an unknown command is not found, not an escape');
});

test('shell python3 and sqlite3 share its folder and interpreter', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'hello' });
  const python = await runFolderBash(folder, 'python3 -c "print(open(\'a.txt\').read())"');
  assert.equal(python.stdout, 'hello\n');
  const sqlite = await runFolderBash(folder,
    'sqlite3 data.db "create table t(x); insert into t values (3); select x from t;"');
  assert.equal(sqlite.stdout, '3\n');
  assert.deepEqual(sqlite.changedPaths, ['data.db']);
  const separated = await runFolderBash(folder,
    `sqlite3 -header -separator : data.db "insert into t values ('a;b'); select x, 'ok' from t where x = 'a;b';"`);
  assert.equal(separated.exitCode, 0, separated.stderr);
  assert.equal(separated.stdout, "x:'ok'\na;b:ok\n");
});
