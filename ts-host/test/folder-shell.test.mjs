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

test('bash has /dev/null, /dev/zero and a scratch /tmp like a standard shell (H4)', async () => {
  const folder = Folder.fromFiles({ 'a.txt': 'a\n' });
  const silenced = await runFolderBash(folder, 'ls nosuch 2>/dev/null; echo rc=$?; echo hi > /dev/null; cat /dev/null; echo end');
  assert.equal(silenced.exitCode, 0, silenced.stderr);
  assert.equal(silenced.stdout, 'rc=2\nend\n');
  assert.equal(silenced.stderr, '');
  assert.deepEqual(silenced.changedPaths, []);
  assert.equal((await runFolderBash(folder, 'cat < /dev/null | wc -c; test -e /dev/null && echo exists')).stdout.trim().replace(/\s+/g, ' '), '0 exists');
  const scratch = await runFolderBash(folder, 'mkdir -p /tmp/w && echo scratch > /tmp/w/n && cat /tmp/w/n && cp a.txt /tmp/w/a && ls /tmp/w');
  assert.equal(scratch.exitCode, 0, scratch.stderr);
  assert.equal(scratch.stdout, 'scratch\na\nn\n');
  assert.deepEqual(scratch.changedPaths, [], 'scratch files are not part of the folder');
  assert.equal((await runFolderBash(folder, 'cat /etc/passwd')).exitCode !== 0, true);
});

test('python3 in bash starts in the shell directory and sees the same folder (H4)', async () => {
  const folder = Folder.fromFiles({ 'types.ts': 'export type A = string;\n', 'sourcingDesk/a.nl': 'x', 'sourcingDesk/b.nl': 'y' });
  const result = await runFolderBash(folder, 'cd sourcingDesk && python3 -c "import os; print(sorted(os.listdir(\'.\')), os.getcwd(), open(\'/workspace/types.ts\').read().strip())"; ls');
  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.stdout, "['a.nl', 'b.nl'] /workspace/sourcingDesk export type A = string;\na.nl\nb.nl\n");
  const root = await runFolderBash(folder, 'python3 -c "import os; print(sorted(os.listdir(\'.\')))"');
  assert.equal(root.stdout, "['sourcingDesk', 'types.ts']\n");
});
