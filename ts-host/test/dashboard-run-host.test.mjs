/**
 * The dashboard's run hosts (dashboard/run-host.js): both deliver the same chunks (output, log and the runtime's live events) to a
 * watcher, so the Runs page draws a run identically from a CLI subprocess and from `runtime.run` in the host's own process.
 */
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadNatlang } from '../dist/index.js';
import { createExecutionModel } from '../dashboard/execution.js';
import { httpRunHost, inProcessRunHost, rerunSpec } from '../dashboard/run-host.js';
import { scriptedModel } from './support/natlang.mjs';

/** double(n) is a natural-language function the scripted model answers with `n * 2`. */
function double() {
  const root = mkdtempSync(join(tmpdir(), 'dashboard-run-host-'));
  mkdirSync(root, { recursive: true });
  writeFileSync(join(root, 'natlang.json'), '{}');
  writeFileSync(join(root, 'double.nl'), '---\nargs:\n  n: number\nreturns: number\n---\nDouble n.\n');
  return loadNatlang(join(root, 'double.nl'));
}
const scripted = () => createNatlangRuntime({ model: scriptedModel(() => 'return n * 2').driver, calls: false });
/** What a Runs page accumulates from a watcher. */
function collector() {
  const seen = { chunks: [], stdout: '', stderr: '', model: createExecutionModel(), live: [], connection: [] };
  return { seen, observer: { update(chunk) {
    seen.chunks.push(chunk); seen.stdout += chunk.stdout; seen.stderr += chunk.stderr;
    for (const event of chunk.live) { seen.live.push(event); seen.model.add(event); }
  }, connection: ok => seen.connection.push(ok), error: error => { seen.error = error; } } };
}
const settled = async (seen) => { while (!seen.chunks.some(chunk => chunk.drained)) await new Promise(resolve => setImmediate(resolve)); };

test('an in-process run delivers its live events, output and final state to a watcher', async () => {
  const fn = double();
  const host = inProcessRunHost(scripted(), { call: spec => fn(spec.inputs.n) });
  assert.deepEqual(host.kinds, ['call']);
  const lists = [];
  const unsubscribe = host.subscribe(runs => lists.push(runs.map(run => run.status)));
  const started = await host.start({ kind: 'call', function: 'double.nl', inputs: { n: 21 }, profile: 'p' });
  assert.equal(started.status, 'running');
  const { seen, observer } = collector();
  const stop = host.watch(started.id, observer);
  await settled(seen);
  const last = seen.chunks.at(-1);
  assert.equal(last.run.status, 'done'); assert.equal(last.run.exitCode, 0); assert.equal(last.drained, true);
  assert.equal(seen.stdout, '42\n');
  // The live events are the runtime's, stamped with `at`, and fold into the execution model the Runs page draws.
  assert.equal(seen.live[0].type, 'call_start'); assert.equal(typeof seen.live[0].at, 'number');
  assert.ok(seen.live.some(event => event.type === 'trace' && event.event.kind === 'model_request'));
  const end = seen.live.find(event => event.type === 'call_end');
  assert.equal(end.outcome, 'done');
  const [call] = seen.model.roots;
  assert.equal(call.name, 'double'); assert.equal(call.status, 'done');
  // A late watcher gets everything so far in one chunk, already drained.
  const late = collector();
  host.watch(started.id, late.observer);
  assert.equal(late.seen.chunks.length, 1);
  assert.equal(late.seen.stdout, '42\n'); assert.equal(late.seen.chunks[0].drained, true);
  assert.deepEqual(late.seen.live, seen.live);
  assert.deepEqual(rerunSpec(last.run), { kind: 'call', function: 'double.nl', inputs: { n: 21 }, profile: 'p', program: undefined });
  assert.deepEqual(lists, [['running'], ['done']]);
  stop(); unsubscribe();
  assert.deepEqual((await host.list()).map(run => run.id), [started.id]);
  await host.remove(started.id);
  assert.deepEqual(await host.list(), []);
});

test('an in-process run that throws fails with its message on stderr; stopping aborts the task', async () => {
  const host = inProcessRunHost(scripted(), {
    ask: () => { throw new Error('the instruction was refused'); },
    run: (_spec, io) => new Promise((_, reject) => io.signal.addEventListener('abort', () => reject(new Error('aborted')))),
  });
  const failing = collector();
  const first = await host.start({ kind: 'ask', instruction: 'x' });
  host.watch(first.id, failing.observer); await settled(failing.seen);
  assert.equal(failing.seen.chunks.at(-1).run.status, 'failed'); assert.equal(failing.seen.chunks.at(-1).run.exitCode, 1);
  assert.equal(failing.seen.stderr, 'the instruction was refused\n');
  const waiting = collector();
  const second = await host.start({ kind: 'run', path: '.' });
  host.watch(second.id, waiting.observer);
  await assert.rejects(() => host.remove(second.id), /stop the run before removing it/);
  assert.equal((await host.stop(second.id)).status, 'stopped');
  await settled(waiting.seen);
  const final = waiting.seen.chunks.at(-1);
  assert.equal(final.run.status, 'stopped'); assert.equal(final.run.exitCode, null); assert.notEqual(final.run.endedAt, null);
  assert.equal(waiting.seen.stderr, '');
  await assert.rejects(() => host.start({ kind: 'build', path: '.' }), /cannot start a build run/);
  await assert.rejects(() => host.input(first.id, 'x'), /does not read input/);
  const missing = collector();
  host.watch('run-99', missing.observer);
  assert.match(missing.seen.error.message, /no run run-99/);
});

test('a disposed watcher hears nothing more', async () => {
  let release;
  const host = inProcessRunHost(scripted(), { run: (_spec, io) => new Promise(resolve => { release = () => { io.stdout('late\n'); resolve(); }; }) });
  const { seen, observer } = collector();
  const run = await host.start({ kind: 'run', path: '.' });
  const stop = host.watch(run.id, observer);
  stop(); const heard = seen.chunks.length;
  release();
  await new Promise(resolve => setImmediate(resolve));
  assert.equal(seen.chunks.length, heard);
});

test('the http run host follows a job by offsets, reports a lost connection and stops after the end', async () => {
  const reads = [], job = status => ({ id: 'job-1', kind: 'run', title: 't', status, exitCode: status === 'running' ? null : 0, startedAt: new Date().toISOString(),
    endedAt: status === 'running' ? null : new Date().toISOString(), interactive: false });
  const script = [
    () => ({ job: job('running'), stdout: { text: 'a', offset: 1 }, stderr: { text: '', offset: 0 }, live: { text: '{"type":"call_start","callId":"c","name":"f"}\n{"type":"call_end"', offset: 40 } }),
    () => { throw new TypeError('fetch failed'); },
    () => ({ job: job('done'), stdout: { text: 'b', offset: 2 }, stderr: { text: '', offset: 0 }, live: { text: '', offset: 40 } }),
    () => ({ job: job('done'), stdout: { text: '', offset: 2 }, stderr: { text: '', offset: 0 }, live: { text: '', offset: 40 } }),
  ];
  const api = { jobs: { list: async () => [job('done')], read: async (id, offsets) => { reads.push({ ...offsets }); const next = script[reads.length - 1]; if (!next) throw new Error('read after the end'); return next(); } } };
  const host = httpRunHost(api);
  const { seen, observer } = collector();
  const stop = host.watch('job-1', observer);
  await settled(seen);
  stop();
  assert.equal(seen.stdout, 'ab');
  assert.deepEqual(seen.live.map(event => event.type), ['call_start'], 'only complete event lines are parsed');
  assert.deepEqual(seen.connection, [true, false, true, true]);
  // Offsets advance from what each read returned; the failed read is repeated from the same place.
  assert.deepEqual(reads, [{ stdout: 0, stderr: 0, live: 0 }, { stdout: 1, stderr: 0, live: 40 }, { stdout: 1, stderr: 0, live: 40 }, { stdout: 2, stderr: 0, live: 40 }]);
  assert.equal(seen.chunks.at(-1).drained, true);
  // A refusal from the server (no such run) is final, not retried.
  const refused = collector();
  const refusing = httpRunHost({ jobs: { read: async () => { throw Object.assign(new Error('no job x'), { status: 404 }); } } });
  refusing.watch('x', refused.observer);
  while (!refused.seen.error) await new Promise(resolve => setImmediate(resolve));
  assert.equal(refused.seen.error.message, 'no job x');
});

test('the vendored dashboard target stages the embeddable modules, styles and types without the standalone shell', async () => {
  const { buildDashboardAssets } = await import('../scripts/dashboard-assets.mjs');
  const { mkdtempSync, rmSync, readFileSync } = await import('node:fs');
  const out = mkdtempSync(join(tmpdir(), 'dashboard-assets-'));
  try {
    const { files, sources } = buildDashboardAssets(out);
    for (const file of ['pages/overview.js', 'pages/runs.js', 'common.js', 'run-host.js', 'api.js', 'pages.js', 'style.css', 'types.d.ts', 'execution.js']) assert.ok(files.includes(file), file);
    assert.ok(!files.includes('app.js') && !files.includes('index.html'));
    assert.deepEqual(sources, files.map(file => `dashboard/${file}`));
    assert.match(readFileSync(join(out, 'style.css'), 'utf8'), /@layer natlang/);
  } finally { rmSync(out, { recursive: true, force: true }); }
});
