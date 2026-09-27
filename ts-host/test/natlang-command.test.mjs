import test from 'node:test';
import assert from 'node:assert/strict';
import { Folder } from '../dist/index.js';
import { executeNatlangAsk, executeNatlangCall, parseAskArgs } from '../dist/native/natlang-command.js';

test('ask input modes preserve order, type JSONL, and filter original inputs', async () => {
  const folder = Folder.fromFiles({ 'notes/a.txt': 'alpha', 'notes/b.txt': 'beta' });
  const calls = [];
  const ask = async (_instruction, returns, input) => {
    calls.push({ returns, input });
    return returns === 'boolean' ? String(input).includes('keep') : typeof input === 'object' &&
      input?.path ? await input.readText() : input;
  };
  assert.equal(await executeNatlangAsk(folder, 'echo', 'first\nsecond\n', { lines: true, jobs: 2 }, ask), 'first\nsecond\n');
  assert.equal(await executeNatlangAsk(folder, 'echo', '{"n":1}\n{"n":2}\n', { jsonl: true }, ask),
    '{"n":1}\n{"n":2}\n');
  assert.deepEqual(calls.slice(2).map(call => call.input), [{ n: 1 }, { n: 2 }]);
  assert.equal(await executeNatlangAsk(folder, 'judge', 'keep\nskip\n', { lines: true, filter: true }, ask), 'keep\n');
  assert.equal(await executeNatlangAsk(folder, 'read', '', { files: 'notes/*.txt' }, ask),
    'notes/a.txt\talpha\nnotes/b.txt\tbeta\n');
  assert.equal(parseAskArgs(['--lines', '--returns', 'boolean', 'Is', 'this', 'urgent?']).instruction,
    'Is this urgent?');
});

test('shell natlang ask uses the shared pipeline', async () => {
  const { runFolderBash } = await import('../dist/index.js');
  const folder = Folder.fromFiles({ 'a.txt': 'hello' });
  const result = await runFolderBash(folder, "printf 'keep\\nskip\\n' | natlang ask --lines --filter 'keep this?'",
    { ask: async (_instruction, returns, input) => returns === 'boolean' && input === 'keep' });
  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.stdout, 'keep\n');
});

test('call input modes are shared by the CLI and shell', async () => {
  const seen = [];
  const call = async (name, input, mode) => { seen.push({ name, input, mode }); return input; };
  assert.equal(await executeNatlangCall('echo', '{"value":3}', 'whole', call), '{"value":3}\n');
  assert.equal(await executeNatlangCall('echo', 'a\nb\n', 'line', call), 'a\nb\n');
  assert.equal(await executeNatlangCall('echo', '{"n":1}\n{"n":2}\n', 'jsonl', call),
    '{"n":1}\n{"n":2}\n');
  assert.deepEqual(seen.map(item => item.mode), ['whole', 'line', 'line', 'jsonl', 'jsonl']);
});

test('shell natlang call and apply use the runtime callbacks', async () => {
  const { runFolderBash } = await import('../dist/index.js');
  const folder = Folder.fromFiles({ 'reports/a.txt': 'initial' });
  const calls = [];
  const call = async (name, input, mode) => { calls.push({ name, input, mode }); return String(input).toUpperCase(); };
  const apply = async (name, path) => { calls.push({ name, path }); folder.writeText(`${path}/a.txt`, 'changed'); return 'done'; };
  const result = await runFolderBash(folder, "printf 'a\\nb\\n' | natlang call upper --lines", { call, apply });
  assert.equal(result.exitCode, 0, result.stderr);
  assert.equal(result.stdout, 'A\nB\n');
  const applied = await runFolderBash(folder, 'natlang apply tidy reports', { call, apply });
  assert.equal(applied.exitCode, 0, applied.stderr);
  assert.equal(applied.stdout, 'done\n');
  assert.equal(await folder.readText('reports/a.txt'), 'changed');
  assert.deepEqual(calls, [
    { name: 'upper', input: 'a', mode: 'line' }, { name: 'upper', input: 'b', mode: 'line' },
    { name: 'tidy', path: 'reports' },
  ]);
});
