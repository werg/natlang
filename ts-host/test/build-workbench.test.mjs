import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost, NodeFileTree } from '../dist/index.js';
import { BuildWorkspace } from '../../applications/build_workbench.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const source = fileURLToPath(new URL('../../codebases/build_workbench/build.nl', import.meta.url));
const node = process.execPath;

async function run(tasks, goal, choose = () => 'source', setup = () => {}) {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-build-'));
  mkdirSync(join(folder, 'out'));
  writeFileSync(join(folder, 'input.txt'), 'hello');
  setup(folder);
  const build = await new BuildWorkspace(folder).open();
  const host = new NatlangHost({ host: { build, drainEvents: () => build.drainEvents() }, mode: 'retained' });
  try {
    const tracePath = join(folder, 'build.trace.jsonl');
    const cyclic = tasks.every(task => task.needs.length);
    const result = await host.run({ source: { kind: 'file', path: source },
      inputs: { tasks, goal, files: new NodeFileTree(folder) }, tracePath,
      modelTurn: request => {
        const prompt = String(request.messages.find(m => m.role === 'user')?.content ?? '');
        if (prompt.includes('function build(')) return evalTurn(request,
          'const initial = await prepare(goal, tasks);\n' +
          'let current = initial;\n' +
          'while (!await finished(current)) current = await step(current, files);\n' +
          'current');
        if (prompt.includes('function step(')) return cyclic ? evalTurn(request,
          'const ready = await ready_tasks(state);\nawait stall(state)') : evalTurn(request,
          'const ready = await ready_tasks(state);\n' +
          'const chosen = await choose(ready, state.goal, files);\n' +
          'await advance(state, chosen)');
        return evalTurn(request, JSON.stringify(choose()));
      } });
    const trace = readFileSync(tracePath, 'utf8').trim().split('\n').map(JSON.parse);
    return { result, folder, trace };
  } finally { host.close(); }
}

test('natlang builds the goal through a real serial process and records outputs', async () => {
  let n = 0;
  const tasks = [
    { id: 'source', needs: [], description: 'prepare source',
      argv: [node, '-e', 'require("fs").copyFileSync("input.txt", "out/source.txt")'],
      inputs: ['input.txt'], outputs: ['out/source.txt'] },
    { id: 'goal', needs: ['source'], description: 'publish goal',
      argv: [node, '-e', 'const fs=require("fs"); fs.writeFileSync("out/goal.txt", fs.readFileSync("out/source.txt", "utf8").toUpperCase())'],
      inputs: ['out/source.txt'], outputs: ['out/goal.txt'] },
    { id: 'unrelated', needs: [], description: 'unrelated',
      argv: [node, '-e', 'require("fs").writeFileSync("out/unrelated.txt", "bad")'],
      inputs: [], outputs: ['out/unrelated.txt'] },
  ];
  const { result, folder, trace } = await run(tasks, 'goal', () => n++ === 0 ? 'source' : 'goal');
  try {
    assert.equal(result.outcome.kind, 'done');
    assert.equal(result.value.status, 'done');
    assert.deepEqual(Array.from(result.value.order), ['source', 'goal']);
    assert.equal(readFileSync(join(folder, 'out/goal.txt'), 'utf8'), 'HELLO');
    assert.ok(result.value.results.every(row => row.status === 'ok' &&
      row.input_sha256.length === 64 && row.output_sha256.length === 64));
    assert.equal(trace.filter(e => e.kind === 'host' && e.event?.operation === 'build.execute').length, 2);
    assert.ok(trace.some(e => e.kind === 'action' && e.name === 'eval'));
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('failed process cannot mark its dependent goal complete', async () => {
  const tasks = [
    { id: 'source', needs: [], description: 'prepare source', argv: [node, '-e', 'process.exit(7)'],
      inputs: ['input.txt'], outputs: ['out/source.txt'] },
    { id: 'goal', needs: ['source'], description: 'publish goal', argv: [node, '-e', 'process.exit(0)'],
      inputs: ['out/source.txt'], outputs: ['out/goal.txt'] },
  ];
  const { result, folder } = await run(tasks, 'goal');
  try {
    assert.equal(result.value.status, 'failed');
    assert.deepEqual(Array.from(result.value.order), []);
    assert.equal(result.value.results[0].exit_code, 7);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('cyclic dependency exposes a blocker without running a process', async () => {
  const tasks = [
    { id: 'source', needs: ['goal'], description: 'source', argv: [node], inputs: [], outputs: ['out/source.txt'] },
    { id: 'goal', needs: ['source'], description: 'goal', argv: [node], inputs: [], outputs: ['out/goal.txt'] },
  ];
  const { result, folder, trace } = await run(tasks, 'goal');
  try {
    assert.equal(result.value.status, 'blocked');
    assert.deepEqual(Array.from(result.value.blocked), ['goal', 'source']);
    assert.equal(trace.filter(e => e.kind === 'host' && e.event?.operation === 'build.execute').length, 0);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});

test('input mutation and preexisting outputs are rejected as verified builds', async () => {
  const mutation = [{ id: 'source', needs: [], description: 'bad generator',
    argv: [node, '-e', 'const fs=require("fs"); fs.writeFileSync("input.txt", "changed"); fs.writeFileSync("out/source.txt", "result")'],
    inputs: ['input.txt'], outputs: ['out/source.txt'] }];
  const first = await run(mutation, 'source');
  try {
    assert.equal(first.result.value.status, 'failed');
    assert.match(first.result.value.detail, /declared input changed/);
  } finally { rmSync(first.folder, { recursive: true, force: true }); }

  const preexisting = [{ id: 'source', needs: [], description: 'bad declaration',
    argv: [node, '-e', 'process.exit(0)'], inputs: ['input.txt'], outputs: ['out/source.txt'] }];
  const second = await run(preexisting, 'source', () => 'source',
    folder => writeFileSync(join(folder, 'out/source.txt'), 'stale'));
  try {
    assert.equal(second.result.value.status, 'failed');
    assert.match(second.result.value.detail, /output already exists/);
  } finally { rmSync(second.folder, { recursive: true, force: true }); }
});

test('exact built-in cache reuses outputs, invalidates changed inputs, and rejects corrupt entries', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-build-cache-test-'));
  const cacheDir = join(root, 'cache');
  const task = { id: 'copy', needs: [], description: 'copy bytes', argv: ['@builtin', 'copy'],
    inputs: ['input.txt'], outputs: ['output.txt'] };
  const execute = async (name, value) => {
    const folder = join(root, name); mkdirSync(folder);
    writeFileSync(join(folder, 'input.txt'), value);
    const workspace = await new BuildWorkspace(folder, { cacheDir }).open();
    const result = await workspace.execute(task);
    return { result, events: workspace.drainEvents(), folder };
  };
  try {
    const first = await execute('first', 'hello');
    assert.equal(first.result.status, 'ok');
    assert.equal(first.result.detail, 'built-in copy');
    const key = first.events.find(e => e.operation === 'build.cache').key;
    const second = await execute('second', 'hello');
    assert.equal(second.result.detail, 'cache hit');
    assert.equal(readFileSync(join(second.folder, 'output.txt'), 'utf8'), 'hello');
    const changed = await execute('changed', 'goodbye');
    assert.equal(changed.result.detail, 'built-in copy');
    assert.notEqual(changed.result.output_sha256, first.result.output_sha256);
    writeFileSync(join(cacheDir, key, 'output.bin'), 'corrupt');
    const repaired = await execute('repaired', 'hello');
    assert.equal(repaired.result.status, 'ok');
    assert.equal(repaired.result.detail, 'built-in copy');
    assert.ok(repaired.events.some(e => e.operation === 'build.cache' && e.status === 'invalid'));
    assert.equal(readFileSync(join(repaired.folder, 'output.txt'), 'utf8'), 'hello');
  } finally { rmSync(root, { recursive: true, force: true }); }
});
