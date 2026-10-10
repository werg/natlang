/**
 * The browser host's parts, platform-neutral and run here on Node: the ExecutionEnv over a natlang Folder
 * (host/folder-env.ts: file system with Node's error codes and messages, readers, the folder shell, timeouts, spilled
 * output) and pi-durable's SQLite storage over sqlite-wasm (host/sqlite-wasm.ts: transactions, rollback, a reopened
 * session). The companion's workspace services run on that env.
 *
 * Run: node --test applications/pi/test/browser-host.test.mjs (after ts-host's build:node; the modules load from source).
 */
import assert from 'node:assert/strict';
import { test, before } from 'node:test';
import { register } from 'node:module';
import sqlite3InitModule from '@sqlite.org/sqlite-wasm';

// The neutral runtime specifier, bound to the Node runtime as a build binds it: the modules load from their sources.
const runtimeUrl = new URL('../../../ts-host/dist/index.js', import.meta.url).href;
register(`data:text/javascript,${encodeURIComponent(`export async function resolve(specifier, context, next) {
  return specifier === 'natlang:runtime' ? { url: ${JSON.stringify(runtimeUrl)}, shortCircuit: true } : next(specifier, context); }`)}`);
let m;
const context = { abortSignal: undefined };

before(async () => {
  m = { natlang: await import(runtimeUrl), env: await import('../host/folder-env.ts'), sqlite: await import('../host/sqlite-wasm.ts'),
    durable: await import('../vendor/durable/src/index.ts'), envService: await import('../host/env.ts'),
    chord: await import('@earendil-works/chord/context'), ai: await import('@earendil-works/pi-ai'), sqlite3: await sqlite3InitModule() };
});

const folderEnv = (files = {}, options = {}) => {
  const folder = new m.natlang.Folder(files);
  return { folder, env: new m.env.FolderExecutionEnv({ folder, ...options }) };
};
const value = result => { assert.equal(result.ok, true, result.ok ? '' : `${result.error.code}: ${result.error.message}`); return result.value; };
const failure = result => { assert.equal(result.ok, false, 'expected a failure'); return result.error; };

test('the folder is the file system at /workspace, with Node\'s errors', async () => {
  const { folder, env } = folderEnv({ 'src/a.txt': '﻿one\ntwo\n' });
  assert.equal(env.cwd, '/workspace');
  assert.equal(value(await env.absolutePath('src/../src/a.txt', context)), '/workspace/src/a.txt');
  assert.equal(value(await env.absolutePath('~/x', context)), '/workspace/x');
  assert.equal(value(await env.readTextFile('src/a.txt', context)), '﻿one\ntwo\n', 'a byte-order mark stays in the text');
  const info = value(await env.fileInfo('src', context));
  assert.deepEqual([info.kind, info.name, info.path], ['directory', 'src', '/workspace/src']);
  assert.equal(value(await env.fileInfo('src/a.txt', context)).size, 11);
  const missing = failure(await env.readTextFile('nope.txt', context));
  assert.deepEqual([missing.code, missing.message], ['not_found', "ENOENT: no such file or directory, open '/workspace/nope.txt'"]);
  assert.equal(failure(await env.readTextFile('src', context)).code, 'is_directory');
  assert.equal(failure(await env.writeFile('/etc/passwd', 'x', context)).code, 'permission_denied');
  assert.equal(failure(await env.readTextFile('/etc/passwd', context)).code, 'not_found');
  assert.equal(failure(await env.writeFile('src/a.txt/b', 'x', context)).code, 'not_directory');

  value(await env.writeFile('deep/new/file.md', '# hi\n', context));
  assert.equal(await folder.readText('deep/new/file.md'), '# hi\n', 'writes land in the folder, parents implied');
  value(await env.appendFile('deep/new/file.md', 'more\n', context));
  value(await env.truncateFile('deep/new/file.md', 4, context));
  assert.equal(await folder.readText('deep/new/file.md'), '# hi');
  value(await env.renameFile('deep/new/file.md', 'moved.md', context));
  assert.equal(value(await env.exists('deep/new/file.md', context)), false);
  assert.equal(await folder.readText('moved.md'), '# hi');
  value(await env.createDir('empty/inner', { recursive: true }, context));
  assert.deepEqual(value(await env.listDir('.', context)).map(entry => [entry.name, entry.kind]).sort(),
    [['empty', 'directory'], ['moved.md', 'file'], ['src', 'directory']]);
  assert.equal(failure(await env.remove('src', {}, context)).code, 'is_directory');
  value(await env.remove('src', { recursive: true }, context));
  assert.equal(folder.isFolder('src'), false);
  assert.equal(value(await env.exists('empty/inner', context)), true);
  assert.equal(failure(await env.watch([], () => {}, context)).code, 'not_supported');
});

test('readers see the file as it was opened; times advance with each change', async () => {
  const { env } = folderEnv({ 'a.txt': 'l0\nl1\nl2' });
  const reader = value(await env.openBinaryReader('a.txt', undefined, context));
  const before = value(await reader.info(context));
  value(await env.writeFile('a.txt', 'changed', context));
  assert.equal(new TextDecoder().decode(value(await reader.read(3, 2, context))), 'l1');
  const scan = value(await reader.scanLines({ startLine: 1, endLine: 2 }, context));
  assert.deepEqual([scan.newlines, scan.start, scan.end], [2, 3, 5]);
  assert.deepEqual(value(await reader.info(context)), before);
  await reader.close(context);
  assert.ok(value(await env.fileInfo('a.txt', context)).mtimeMs > before.mtimeMs);
  assert.deepEqual(value(await env.readTextLines('a.txt', undefined, context)), ['changed']);
  const lines = value(await env.openTextLineReader('a.txt', context));
  assert.deepEqual(value(await lines.readLine(context)), { text: 'changed', terminated: false });
});

test('commands run in the folder shell over the same folder', async () => {
  const { folder, env } = folderEnv({ 'src/app.js': 'const answer = 41;\n', 'README.md': 'hello\n' });
  let output = '';
  const ran = value(await env.exec('sed -i "s/41/42/" src/app.js && cat src/app.js && echo oops >&2 && exit 3',
    { onOutput: text => { output += text; } }, context));
  assert.equal(ran.exitCode, 3);
  assert.equal(output, 'const answer = 42;\noops\n');
  assert.equal(await folder.readText('src/app.js'), 'const answer = 42;\n');
  assert.ok(value(await env.fileInfo('src/app.js', context)).mtimeMs > 0, 'a command\'s change moves the file\'s time');
  output = '';
  value(await env.exec('pwd && ls', { cwd: 'src', onOutput: text => { output += text; } }, context));
  assert.equal(output, '/workspace/src\napp.js\n');
  // An argv runs without shell parsing: the pattern reaches grep as it is.
  output = '';
  value(await env.exec(['grep', '-rnE', '-e', 'answer = [0-9]+;', '.'], { onOutput: text => { output += text; } }, context));
  assert.match(output, /src\/app\.js:1:const answer = 42;/);
  output = '';
  assert.equal(value(await env.exec('while true; do echo; done', { onOutput: text => { output += text; } }, context)).exitCode, 2);
  assert.match(output, /While loops are not allowed/);
  assert.equal(failure(await env.exec('ls', { cwd: '/etc' }, context)).code, 'spawn_error');
});

test('a long output spills to /tmp; a timeout and an abort end the command', async () => {
  const { env } = folderEnv({});
  const ran = value(await env.exec('seq 1 3000', { spill: { afterBytes: 50 * 1024, afterLines: 2000 } }, context));
  assert.match(ran.spillPath, /^\/tmp\/tmp-[^/]+\/pi-output-.*\.log$/);
  const spilled = value(await env.readTextFile(ran.spillPath, context));
  assert.equal(spilled.split('\n').length, 3001);
  assert.equal(value(await env.exec('seq 1 3', { spill: { afterBytes: 50 * 1024, afterLines: 2000 } }, context)).spillPath, undefined);
  const slow = failure(await env.exec('sleep 2', { timeout: 0.05 }, context));
  assert.deepEqual([slow.code, slow.message], ['timeout', 'timeout:0.05']);
  const controller = new AbortController();
  const aborted = env.exec('sleep 2', {}, { abortSignal: controller.signal });
  setTimeout(() => controller.abort(), 20);
  assert.equal(failure(await aborted).code, 'aborted');
});

test('pi\'s env service runs on the folder environment', async () => {
  const { env } = folderEnv({ 'notes.txt': 'a\n' });
  const outputs = [];
  const api = { env, output: text => outputs.push(text), diagnostic: () => {}, outputWindow: undefined };
  const service = m.envService.envService(api, context);
  assert.equal((await service.readPath('@notes.txt')).value, '/workspace/notes.txt');
  const lock = (await service.lock('notes.txt')).value;
  assert.equal(typeof lock, 'number');
  service.unlock(lock);
  assert.deepEqual((await service.runShell('echo hi')).value, { exitCode: 0 });
  assert.deepEqual(outputs, ['hi\n']);
  await service.release();
});

test('pi-durable sessions persist in sqlite-wasm, transactions roll back as a whole', async () => {
  const db = new m.sqlite3.oo1.DB(':memory:');
  const database = new m.sqlite.WasmSqliteDatabase(db);
  await database.exec('CREATE TABLE t (k TEXT PRIMARY KEY, v BLOB, n INTEGER)');
  await database.run('INSERT INTO t VALUES (?, ?, ?)', 'a', new Uint8Array([1, 2]), 7);
  assert.deepEqual({ ...await database.get('SELECT k, v, n FROM t WHERE k = ?', 'a') }, { k: 'a', v: new Uint8Array([1, 2]), n: 7 });
  await assert.rejects(database.transaction(async tx => {
    await tx.run('INSERT INTO t VALUES (?, ?, ?)', 'b', null, 1);
    throw new Error('no');
  }), /no/);
  assert.equal((await database.all('SELECT k FROM t')).length, 1, 'the failed transaction left nothing');
  let handle;
  assert.equal(await database.transaction(async tx => { handle = tx; await tx.run('INSERT INTO t VALUES (?, ?, ?)', 'c', null, 2); return 'ok'; }), 'ok');
  await assert.rejects(handle.get('SELECT 1'), /no longer active/);
  await database.close();

  // A Harness on the storage: its root conversation lands in the database's own tables.
  const session = new m.sqlite3.oo1.DB(':memory:');
  const ctx = m.chord.BACKGROUND_CONTEXT;
  const harness = await m.durable.Harness.open(await m.sqlite.openWasmSqliteStorage(session),
    { models: m.ai.createModels(), registry: m.durable.createRegistry() }, ctx);
  const conversation = await harness.root(ctx);
  const rows = session.exec({ sql: 'SELECT id FROM conversations', rowMode: 'object', returnValue: 'resultRows' });
  assert.deepEqual(rows.map(row => row.id), [conversation.id]);
  await harness.close(ctx);
});
