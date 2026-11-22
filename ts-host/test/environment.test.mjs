import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile } from 'node:fs/promises';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createNatlangRuntime, loadNatlang, DesktopBindings } from '../dist/index.js';
import { TypeScriptEnvironment } from '../dist/environment.js';
import { scriptedModel } from './support/natlang.mjs';

async function workspace(t, extra = {}) {
  const path = await mkdtemp(join(tmpdir(), 'natlang-workspace-'));
  await writeFile(join(path, 'package.json'), JSON.stringify({ name: 'natlang-fixture', private: true, type: 'module', ...extra }, null, 2));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}
async function installedPackage(root, name, source = 'export const answer = 42;\n') {
  const dir = join(root, 'node_modules', ...name.split('/'));
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'package.json'), JSON.stringify({ name, version: '1.0.0', type: 'module', exports: './index.js', main: './index.js' }));
  await writeFile(join(dir, 'index.js'), source);
  return dir;
}
const evalRequest = code => ({ code, body: true, scope: {}, path: 'test' });

test('eval scope preserves absent optional fields without relaxing portable result validation',()=>{
 const environment=new TypeScriptEnvironment();
 const request={...evalRequest('return self.evidence[0].optional === undefined;'),scope:{evidence:[{optional:undefined,value:1}]}};
 assert.equal(environment.execute(request).result,true);
 environment.close();
});

test('fresh and retained eval contexts, frozen snapshots, live values, and disposal', () => {
  const request = code => ({ code, scope: { args: { n: 3 } }, body: false, path: 'eval' });
  const fresh = new TypeScriptEnvironment({ mode: 'fresh' });
  assert.equal(fresh.execute(request('globalThis.marker = 7; 7')).result, 7);
  assert.equal(fresh.execute(request('globalThis.marker ?? 0')).result, 0);
  const retained = new TypeScriptEnvironment({ mode: 'retained' });
  assert.equal(retained.execute(request('globalThis.marker = 9; 9')).result, 9);
  assert.equal(retained.execute(request('globalThis.marker')).result, 9);
  // In an eval body the scope snapshot and live values are the body's parameters, never globals.
  const body = code => ({ ...request(code), body: true });
  assert.throws(() => retained.execute(body('self.args.n = 10')), /read only|read-only|Cannot assign/);
  assert.equal(retained.execute(body('return typeof globalThis.self')).result, 'undefined');
  assert.equal(Object.prototype.toString.call(retained.execute(request('new Date(0)')).result), '[object Date]');
  const live = { hits: 0 };
  assert.equal(retained.execute({ ...body('__live.counter.hits += 1; return __live.counter.hits'), live: { counter: live } }).result, 1);
  assert.equal(live.hits, 1, 'live values are shared by reference');
  retained.close();
  assert.throws(() => retained.execute(request('1')), /disposed/);
});

test('eval rejects local and built-in imports and validates type-only imports', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'helper.mjs'), 'export const double = value => value * 2;\n');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  for (const code of ["const mod = await import('./helper.mjs'); return mod.double(3);", "import { double } from './helper.mjs'; return double(3);",
    "const fs = await import('node:fs'); return true;", "import type { X } from './local'; return 1;"])
    await assert.rejects(environment.executeAsync(evalRequest(code)), /Only package imports/, code);
  // Type-only imports are erased, so they never load anything.
  assert.equal((await environment.executeAsync(evalRequest("import { type Y } from 'not-installed'; return 1;"))).result, 1);
});

test('eval runs with Node host globals, try/catch, and classes', async t => {
  const environment = new TypeScriptEnvironment();
  t.after(() => environment.close());
  const code = 'class Pair { constructor(a, b) { this.sum = a + b; } }\n' +
    'let parsed; try { parsed = JSON.parse("{"); } catch { parsed = "fallback"; }\n' +
    'await new Promise(done => setTimeout(done, 1));\n' +
    'return [new Pair(2, 3).sum, parsed, Buffer.from("abc").length, typeof process.cwd()];';
  assert.equal(JSON.stringify((await environment.executeAsync(evalRequest(code))).result), JSON.stringify([5, 'fallback', 3, 'string']));
});

test('eval can fetch', async t => {
  const path = await workspace(t);
  const server = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ path: request.url, answer: 42 }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const environment = new TypeScriptEnvironment({ workspace: path, network: true });
  t.after(() => environment.close());
  const result = await environment.executeAsync(evalRequest(`const response = await fetch('http://127.0.0.1:${server.address().port}/data'); return await response.json();`));
  assert.deepEqual(JSON.parse(JSON.stringify(result.result)), { path: '/data', answer: 42 });
});

test('eval imports whatever the workspace can resolve, like any module in it', async t => {
  const path = await workspace(t);
  await installedPackage(path, 'natlang-local-fixture');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  assert.equal((await environment.executeAsync(evalRequest("const pkg = await import('natlang-local-fixture'); return pkg.answer;"))).result, 42);
  assert.equal((await environment.executeAsync(evalRequest("import { answer as result } from 'natlang-local-fixture'; return result;"))).result, 42);
  await assert.rejects(environment.executeAsync(evalRequest("return (await import('not-installed')).answer;")), /Cannot find module 'not-installed'/);
});

test('the runtime workspace supplies eval packages and callable-folder package imports', async t => {
  const path = await workspace(t);
  await installedPackage(path, '@fixture/math', 'export const answer = 42;\nexport default value => value + 1;\n');
  await writeFile(join(path, 'answer.nl'), '---\nargs: {}\nreturns: number\n---\nImport the math dependency and add one using increment.\n');
  await mkdir(join(path, 'answer'));
  await writeFile(join(path, 'answer', 'increment.ts'), "import add from '@fixture/math';\nexport default function increment(value: number): number { return add(value); }\n");
  const model = scriptedModel(() => "import { answer } from '@fixture/math'; return increment(answer)");
  const runtime = createNatlangRuntime({ workspace: path, model: request => model.driver(request) });
  const previous = process.cwd();
  process.chdir(tmpdir());
  try { assert.equal(await runtime.run(() => loadNatlang(join(path, 'answer.nl'))()), 43); }
  finally { process.chdir(previous); }
});

test('callable-folder modules cannot import local files outside their folder', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'helper.mjs'), 'export const double = value => value * 2;\n');
  await writeFile(join(path, 'main.nl'), '---\nargs: {}\nreturns: number\n---\nDouble three.\n');
  await mkdir(join(path, 'main'));
  await writeFile(join(path, 'main', 'twice.ts'), "import { double } from '../helper.mjs';\nexport default function twice(): number { return double(3); }\n");
  const main = loadNatlang(join(path, 'main.nl'));
  const runtime = createNatlangRuntime({ workspace: path });
  await assert.rejects(() => runtime.run(async () => main.twice()), /may import its sibling items and packages only/);
});

test('desktop bindings work as a host service through eval', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-desktop-'));
  const input = join(dir, 'input.txt');
  writeFileSync(input, 'sample');
  const desktop = new DesktopBindings();
  try {
    const root = mkdtempSync(join(tmpdir(), 'natlang-desktop-fn-'));
    writeFileSync(join(root, 'shout.nl'), '---\nargs:\n  path: string\nreturns: string\n---\nRead path and upper-case it with a process.\n');
    const model = scriptedModel(() => 'const content: string = desktop.readText(path);\n' +
      'return desktop.run(["node", "-e", "process.stdout.write(process.argv[1].toUpperCase())", content]).stdout');
    const traces = [];
    const runtime = createNatlangRuntime({ model: model.driver, services: { desktop }, trace: trace => traces.push(trace) });
    assert.equal(await runtime.run(() => loadNatlang(join(root, 'shout.nl'))(input)), 'SAMPLE');
    const effects = traces[0].events.filter(event => event.kind === 'effect' && event.phase === 'completed').map(event => event.capability);
    assert.deepEqual(effects, ['desktop.readText', 'desktop.run']);
  } finally { desktop.close(); rmSync(dir, { recursive: true, force: true }); }
});

test('evaluated code cannot reach native process bindings or end the process', () => {
  // A teacher's probe of process.binding('fs') aborted a 96-worker collector natively (2026-10-07).
  const environment = new TypeScriptEnvironment({ mode: 'fresh' });
  const request = code => ({ code, scope: {}, body: true, path: 'eval' });
  for (const member of ['binding', '_linkedBinding', 'dlopen', 'abort', 'reallyExit', 'exit', 'kill', 'chdir'])
    assert.throws(() => environment.execute(request(`return process.${member}('fs')`)), /not available to evaluated code/, member);
  assert.equal(environment.execute(request('return typeof process.cwd() === "string" && typeof process.env')).result, 'object');
  assert.equal(environment.execute(request('return process.hrtime.bigint() > 0n')).result, true);
  environment.close();
});

test('an eval whose data explodes fails alone with the memory teaching error; its sibling completes', async () => {
  // Sokoban BFS/A* solvers in eval grew a shared collector to OOM, killing every co-hosted episode (2026-11-16).
  const sibling = new TypeScriptEnvironment({ memoryLimitMb: 64 });
  const runaway = new TypeScriptEnvironment({ memoryLimitMb: 64 });
  const pending = sibling.executeAsync(evalRequest('const rows = Array.from({ length: 1000 }, (_, i) => ({ i })); ' +
    'await new Promise(resolve => setTimeout(resolve, 20)); return rows.length;'));
  const limit = /eval stopped: its data grew past the 64 MB memory limit of one eval; use a bounded or more memory-efficient approach/;
  // An unbounded search, a callback that fills, and code that catches the error and keeps growing all fail.
  assert.throws(() => runaway.execute(evalRequest('const seen = new Map();\nfor (let i = 0; ; i++) seen.set("s" + i, [i, i + 1]);')), limit);
  assert.throws(() => runaway.execute(evalRequest('return Array.from({ length: 2e7 }, (_, i) => ({ i, key: "k" + i })).length;')), limit);
  assert.throws(() => runaway.execute(evalRequest('const keep = [];\ntry { for (;;) keep.push(new Array(256).fill(1)); } catch {}\n' +
    'for (;;) keep.push(new Array(256).fill(2));')), limit);
  await assert.rejects(runaway.executeAsync(evalRequest('const keep = [];\nawait null;\nwhile (true) keep.push({ at: keep.length });')), limit);
  assert.equal((await pending).result, 1000);
  // Garbage is not growth: a loop that allocates many times the bound but keeps little passes, and the environment lives on.
  assert.equal(runaway.execute(evalRequest('let total = 0;\nfor (let i = 0; i < 300000; i++) total += new Array(128).fill(i).length;\nreturn total;')).result, 300000 * 128);
  assert.equal(runaway.execute(evalRequest('return 1 + 1;')).result, 2);
  sibling.close(); runaway.close();
});

test('memory checks keep eval semantics and line numbers; the bound is a setting', async () => {
  const { instrumentMemoryChecks } = await import('../dist/native/eval-memory.js');
  const { evalMemoryLimitMb } = await import('../dist/environment.js');
  const code = [
    'class Walker extends Array { constructor(n: number) { super(); for (let i = 0; i < n; i++) this.push(i); }',
    '  get total() { return this.reduce((a, b) => a + b, 0); } }',
    'const pairs = [1, 2].map(x => ({ x }));',
    'let skipped = 0; outer: for (const a of [1, 2, 3]) for (const b of [1, 2]) { if (b === 2) { skipped++; continue outer; } }',
    'let n = 0; do n++; while (n < 5)',
    'function* count(k: number) { for (let i = 0; i < k; i++) yield i; }',
    'return [new Walker(4).total, pairs[1].x, skipped, n, [...count(3)].length, (() => { "use strict"; return 7; })()];'].join('\n');
  const instrumented = instrumentMemoryChecks(`function f() {\n${code}\n}`);
  assert.equal(instrumented.split('\n').length, code.split('\n').length + 2, 'insertions stay on their lines');
  assert.match(instrumented, /do \{__natlang_memory\(\);n\+\+;\} while/);
  for (const memoryLimitMb of [64, 0]) {
    const environment = new TypeScriptEnvironment({ memoryLimitMb });
    assert.equal(JSON.stringify(environment.execute(evalRequest(code)).result), "[6,2,3,5,3,7]");
    environment.close();
  }
  assert.equal(new TypeScriptEnvironment({ memoryLimitMb: 0 }).memoryLimitMb, undefined);
  assert.equal(evalMemoryLimitMb({}), 256);
  assert.equal(evalMemoryLimitMb({ NATLANG_EVAL_MEMORY_MB: '0' }), 0);
  assert.equal(evalMemoryLimitMb({ NATLANG_EVAL_MEMORY_MB: '512' }), 512);
  assert.throws(() => evalMemoryLimitMb({ NATLANG_EVAL_MEMORY_MB: 'lots' }), /NATLANG_EVAL_MEMORY_MB/);
});

test('an eval that computes without pausing fails alone with the time teaching error; co-hosted work goes on', async () => {
  // sft-v7 held candidate 2026-11-22: an eval's runaway recursion held a shared collector's event loop for an hour.
  const { evalSyncLimitSeconds, syncGuard } = await import('../dist/native/eval-memory.js');
  const limit = /eval stopped: it computed for more than 0.2 s without pausing; use a bounded or faster approach/;
  const runaway = new TypeScriptEnvironment({ syncLimitSeconds: 0.2 });
  const sibling = new TypeScriptEnvironment({ syncLimitSeconds: 0.2 });
  let ticks = 0;
  const timer = setInterval(() => ticks++, 20);
  // The exponential search of the incident (no timeout_ms), a stretch after an await, and code that catches the error.
  const search = 'function find(xs: any[]): boolean { if (xs.length === 1) return false;\n' +
    'for (let i = 0; i < xs.length; i++) for (let j = i + 1; j < xs.length; j++) if (find([xs[i] + xs[j], ...xs.filter((_, k) => k !== i && k !== j)])) return true;\n' +
    'return false; }\nreturn find(Array.from({ length: 40 }, (_, i) => i));';
  assert.throws(() => runaway.execute(evalRequest(search)), limit);
  await assert.rejects(runaway.executeAsync(evalRequest('await null;\nlet n = 0;\nwhile (true) n++;')), limit);
  await assert.rejects(runaway.executeAsync(evalRequest('await null;\nfor (;;) { try { for (;;) {} } catch {} }')), limit);
  // A builtin that does not return within the first stretch (catastrophic backtracking) is interrupted too.
  assert.throws(() => runaway.execute(evalRequest('return /^(a+)+$/.test("a".repeat(40) + "b");')), limit);
  // Work that pauses is not one stretch; a smaller timeout_ms keeps its own error; the environment lives on.
  assert.equal((await sibling.executeAsync(evalRequest('let total = 0;\nfor (let r = 0; r < 6; r++) {\n' +
    'const end = performance.now() + 60; while (performance.now() < end) total++;\nawait new Promise(resolve => setTimeout(resolve, 1)); }\n' +
    'return total > 0;'))).result, true);
  assert.throws(() => runaway.execute({ ...evalRequest('for (;;) {}'), timeoutMs: 50 }), /timed out after 50ms/);
  assert.equal(runaway.execute(evalRequest('return 1 + 1;')).result, 2);
  clearInterval(timer);
  assert.ok(ticks > 0);
  // The bound is a setting; 0 turns it off; module guards reset per run.
  assert.equal(evalSyncLimitSeconds({}), 60);
  assert.equal(evalSyncLimitSeconds({ NATLANG_EVAL_SYNC_SECONDS: '0' }), 0);
  assert.equal(evalSyncLimitSeconds({ NATLANG_EVAL_SYNC_SECONDS: '120' }), 120);
  assert.throws(() => evalSyncLimitSeconds({ NATLANG_EVAL_SYNC_SECONDS: 'long' }), /NATLANG_EVAL_SYNC_SECONDS/);
  assert.equal(new TypeScriptEnvironment({ syncLimitSeconds: 0 }).syncLimitSeconds, undefined);
  let clock = 0;
  const once = syncGuard(1, 'stop', false, () => clock);
  once(); clock = 5000; assert.throws(() => { for (let i = 0; i < 200; i++) once(); }, /stop/);
  await null;
  once(); for (let i = 0; i < 200; i++) once();
  runaway.close(); sibling.close();
});

test('callable-folder module code is bounded per synchronous run', async () => {
  const environment = new TypeScriptEnvironment({ mode: 'retained', memoryLimitMb: 64 });
  const module = { exports: {} };
  environment.evaluateModule('module.exports.grow = function () { const keep = []; for (;;) keep.push({ at: keep.length }); };\n' +
    'module.exports.sum = n => { let total = 0; for (let i = 0; i < n; i++) total += i; return total; };', { module, exports: module.exports });
  assert.throws(() => module.exports.grow(), /callable-folder code stopped: its data grew past the 64 MB memory limit of one synchronous run of callable-folder code; use a bounded/);
  await null;
  assert.equal(module.exports.sum(1000), 499500);
  environment.close();
});
