import assert from 'node:assert/strict';
import { mkdtemp, mkdir, rm, writeFile, readFile, access } from 'node:fs/promises';
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
async function fixturePackage(root, name, source = 'export const answer = 42;\n') {
  const dir = join(root, 'fixture-pkg');
  await mkdir(dir, { recursive: true });
  await writeFile(join(dir, 'package.json'), JSON.stringify({ name, version: '1.0.0', type: 'module', exports: './index.js', main: './index.js' }));
  await writeFile(join(dir, 'index.js'), source);
  return dir;
}
const evalRequest = code => ({ code, body: true, scope: {}, path: 'test' });

test('fresh and retained eval contexts, frozen snapshots, live values, and disposal', () => {
  const request = code => ({ code, scope: { args: { n: 3 } }, body: false, path: 'eval' });
  const fresh = new TypeScriptEnvironment({ mode: 'fresh' });
  assert.equal(fresh.execute(request('globalThis.marker = 7; 7')).result, 7);
  assert.equal(fresh.execute(request('globalThis.marker ?? 0')).result, 0);
  const retained = new TypeScriptEnvironment({ mode: 'retained' });
  assert.equal(retained.execute(request('globalThis.marker = 9; 9')).result, 9);
  assert.equal(retained.execute(request('globalThis.marker')).result, 9);
  assert.throws(() => retained.execute(request('self.args.n = 10')), /read only|read-only|Cannot assign/);
  assert.equal(Object.prototype.toString.call(retained.execute(request('new Date(0)')).result), '[object Date]');
  const live = { hits: 0 };
  assert.equal(retained.execute({ ...request('__live.counter.hits += 1; __live.counter.hits'), live: { counter: live } }).result, 1);
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
    await assert.rejects(environment.executeAsync(evalRequest(code)), /Only declared package imports/, code);
  await assert.rejects(environment.executeAsync(evalRequest("import { type Y } from 'not-installed'; return 1;")), /not declared in application package.json/);
});

test('eval fetches when the workspace allows network access', async t => {
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

test('installPackages updates package.json; only declared, installed packages import', async t => {
  const path = await workspace(t);
  await fixturePackage(path, 'natlang-local-fixture');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  await assert.rejects(environment.executeAsync(evalRequest("const pkg = await import('natlang-local-fixture'); return pkg.answer;")));
  assert.equal((await environment.executeAsync(evalRequest("await installPackages(['file:./fixture-pkg']); return true;"))).result, true);
  await access(join(path, 'package-lock.json'));
  assert.deepEqual(JSON.parse(await readFile(join(path, 'package.json'), 'utf8')).dependencies, { 'natlang-local-fixture': 'file:fixture-pkg' });
  assert.equal((await environment.executeAsync(evalRequest("const pkg = await import('natlang-local-fixture'); return pkg.answer;"))).result, 42);
  assert.equal((await environment.executeAsync(evalRequest("import { answer as result } from 'natlang-local-fixture'; return result;"))).result, 42);
  await mkdir(join(path, 'node_modules', 'undeclared'));
  await writeFile(join(path, 'node_modules', 'undeclared', 'package.json'), JSON.stringify({ name: 'undeclared', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(path, 'node_modules', 'undeclared', 'index.js'), 'export const answer = 99;\n');
  await assert.rejects(environment.executeAsync(evalRequest("return (await import('undeclared')).answer;")), /not declared in application package.json/);
});

test('the runtime workspace supplies eval packages, callable-folder package imports, and the dependency prompt', async t => {
  const path = await workspace(t);
  await fixturePackage(path, '@fixture/math', 'export const answer = 42;\nexport default value => value + 1;\n');
  const setup = new TypeScriptEnvironment({ workspace: path });
  await setup.installPackages(['file:./fixture-pkg']);
  setup.close();
  await writeFile(join(path, 'package.json'), JSON.stringify({ name: 'natlang-fixture', private: true, type: 'module',
    dependencies: { '@fixture/math': 'file:fixture-pkg', 'not-installed': '1.0.0' } }));
  await writeFile(join(path, 'answer.nl'), '---\nargs: {}\nreturns: number\n---\nImport the math dependency and add one using increment.\n');
  await mkdir(join(path, 'answer'));
  await writeFile(join(path, 'answer', 'increment.ts'), "import add from '@fixture/math';\nexport default function increment(value: number): number { return add(value); }\n");
  const prompts = [];
  const model = scriptedModel(() => "import { answer } from '@fixture/math'; result = increment(answer)");
  const runtime = createNatlangRuntime({ workspace: path, model: request => { prompts.push(request.messages[0].content); return model.driver(request); } });
  const previous = process.cwd();
  process.chdir(tmpdir());
  try { assert.equal(await runtime.run(() => loadNatlang(join(path, 'answer.nl'))()), 43); }
  finally { process.chdir(previous); }
  assert.match(prompts[0], /Importable application dependencies from package\.json:\n- "@fixture\/math"/);
  assert.doesNotMatch(prompts[0], /not-installed/);
});

test('callable-folder modules cannot import local files outside their folder', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'helper.mjs'), 'export const double = value => value * 2;\n');
  await writeFile(join(path, 'main.nl'), '---\nargs: {}\nreturns: number\n---\nDouble three.\n');
  await mkdir(join(path, 'main'));
  await writeFile(join(path, 'main', 'twice.ts'), "import { double } from '../helper.mjs';\nexport default function twice(): number { return double(3); }\n");
  const main = loadNatlang(join(path, 'main.nl'));
  const runtime = createNatlangRuntime({ workspace: path });
  await assert.rejects(() => runtime.run(async () => main.twice()), /may import its sibling items and declared packages only/);
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
      'result = desktop.run(["node", "-e", "process.stdout.write(process.argv[1].toUpperCase())", content]).stdout');
    const traces = [];
    const runtime = createNatlangRuntime({ model: model.driver, services: { desktop }, trace: trace => traces.push(trace) });
    assert.equal(await runtime.run(() => loadNatlang(join(root, 'shout.nl'))(input)), 'SAMPLE');
    const effects = traces[0].events.filter(event => event.kind === 'effect' && event.phase === 'completed').map(event => event.capability);
    assert.deepEqual(effects, ['desktop.readText', 'desktop.run']);
  } finally { desktop.close(); rmSync(dir, { recursive: true, force: true }); }
});
