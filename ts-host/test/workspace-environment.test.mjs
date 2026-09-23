import assert from 'node:assert/strict';
import { access, mkdtemp, mkdir, readFile, rm, writeFile } from 'node:fs/promises';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { NativeNatlangHost, TypeScriptEnvironment } from '../dist/index.js';

async function workspace(t) {
  const path = await mkdtemp(join(tmpdir(), 'natlang-workspace-'));
  await writeFile(join(path, 'package.json'), JSON.stringify({ name: 'natlang-fixture', private: true, type: 'module' }, null, 2));
  t.after(() => rm(path, { recursive: true, force: true }));
  return path;
}

const evalRequest = code => ({ code, body: true, scope: { let: {} }, path: 'test', effectful: false });

test('project eval rejects local application imports in both forms', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'helper.mjs'), 'export const double = value => value * 2;\n');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  await assert.rejects(environment.executeAsync(evalRequest("const mod = await import('./helper.mjs'); return mod.double(3);")), /Only declared package imports/);
  await assert.rejects(environment.executeAsync(evalRequest("import { double } from './helper.mjs'; return double(3);")), /Only declared package imports/);
  await assert.rejects(environment.executeAsync(evalRequest("const fs = await import('node:fs'); return true;")), /Only declared package imports/);
});

test('workspace eval validates type-only imports before erasing them', async t => {
  const path = await workspace(t);
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  await assert.rejects(environment.executeAsync(evalRequest("import type { X } from './local'; return 1;")), /Only declared package imports/);
  await assert.rejects(environment.executeAsync(evalRequest("import { type Y } from 'not-installed'; return 1;")), /not declared in application package.json/);
});

test('native host rejects local imports from application TypeScript', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'helper.mjs'), 'export const double = value => value * 2;\n');
  await writeFile(join(path, 'main.ts'), "import { double } from './helper.mjs';\nexport default function main(): number { return double(3); }\n");
  const host = new NativeNatlangHost({ workspace: path });
  t.after(() => host.close());
  await assert.rejects(host.run({ source: { kind: 'file', path: join(path, 'main.ts') } }), /bad-import/);
});

test('native host rejects undeclared type-only package imports', async t => {
  const path = await workspace(t);
  await writeFile(join(path, 'main.ts'), "import type { Value } from 'undeclared';\nexport default function main(): number { return 3; }\n");
  const host = new NativeNatlangHost({ workspace: path });
  t.after(() => host.close());
  await assert.rejects(host.run({ source: { kind: 'file', path: join(path, 'main.ts') } }), /bad-import/);
});

test('workspace network capability fetches JSON and returns a portable value', async t => {
  const path = await workspace(t);
  const server = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ path: request.url, answer: 42 }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const address = server.address();
  const environment = new TypeScriptEnvironment({ workspace: path, network: true });
  t.after(() => environment.close());
  const result = await environment.executeAsync(evalRequest(`const response = await fetch('http://127.0.0.1:${address.port}/data'); return await response.json();`));
  assert.deepEqual(JSON.parse(JSON.stringify(result.result)), { path: '/data', answer: 42 });
});

test('workspace installPackages installs a local package into its package manifest', async t => {
  const path = await workspace(t);
  const fixture = join(path, 'fixture-pkg');
  await mkdir(fixture);
  await writeFile(join(fixture, 'package.json'), JSON.stringify({ name: 'natlang-local-fixture', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(fixture, 'index.js'), 'export const answer = 42;\n');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  await assert.rejects(environment.executeAsync(evalRequest("const pkg = await import('natlang-local-fixture'); return pkg.answer;")), /cannot find|resolve|module|package/i);
  const installResult = await environment.executeAsync(evalRequest("await installPackages(['file:./fixture-pkg']); return true;"));
  assert.equal(installResult.result, true);
  await access(join(path, 'package-lock.json'));
  await environment.packages.prepareDependencies();
  assert.deepEqual(JSON.parse(await readFile(join(path, 'package.json'), 'utf8')).dependencies, { 'natlang-local-fixture': 'file:fixture-pkg' });
  const result = await environment.executeAsync(evalRequest("const pkg = await import('natlang-local-fixture'); return pkg.answer;"));
  assert.equal(result.result, 42);
  await mkdir(join(path, 'node_modules', 'undeclared'));
  await writeFile(join(path, 'node_modules', 'undeclared', 'package.json'), JSON.stringify({ name: 'undeclared', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(path, 'node_modules', 'undeclared', 'index.js'), 'export const answer = 99;\n');
  await assert.rejects(environment.executeAsync(evalRequest("return (await import('undeclared')).answer;")), /not declared in application package.json/);
});

test('workspace dependencies resolve in dynamic and static eval imports and project TypeScript from any process cwd', async t => {
  const path = await workspace(t);
  const fixture = join(path, 'fixture-pkg');
  await mkdir(fixture);
  await writeFile(join(fixture, 'package.json'), JSON.stringify({ name: '@natlang/fixture', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(fixture, 'index.js'), 'export const answer = 42; export default value => value + 1;\n');
  const environment = new TypeScriptEnvironment({ workspace: path });
  t.after(() => environment.close());
  await environment.installPackages(['file:./fixture-pkg']);
  assert.deepEqual(environment.packages.listDeclaredDependencies(), ['@natlang/fixture']);
  assert.deepEqual(environment.packages.listAvailableDependencies(), ['@natlang/fixture']);
  const previousCwd = process.cwd();
  process.chdir(tmpdir());
  try {
    const dynamic = await environment.executeAsync(evalRequest("const pkg = await import('@natlang/fixture'); return pkg.answer;"));
    assert.equal(dynamic.result, 42);
    const staticallyImported = await environment.executeAsync(evalRequest("import { answer as result } from '@natlang/fixture'; return result;"));
    assert.equal(staticallyImported.result, 42);
    const nested = join(path, 'src', 'nested');
    await mkdir(nested, { recursive: true });
    await writeFile(join(nested, 'main.ts'), "import { answer } from '@natlang/fixture';\nexport default function main(): number { return answer; }\n");
    const host = new NativeNatlangHost({ workspace: path });
    t.after(() => host.close());
    const result = await host.run({ source: { kind: 'file', path: join(nested, 'main.ts') } });
    assert.equal(result.outcome.kind, 'done', JSON.stringify(result.outcome));
    assert.equal(result.value, 42);
  } finally { process.chdir(previousCwd); }
});

test('native host runs an application file importing an installed local package', async t => {
  const path = await workspace(t);
  const fixture = join(path, 'fixture-pkg');
  await mkdir(fixture);
  await writeFile(join(fixture, 'package.json'), JSON.stringify({ name: 'natlang-file-package', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(fixture, 'index.js'), 'export const answer = 42;\n');
  const environment = new TypeScriptEnvironment({ workspace: path });
  await environment.executeAsync(evalRequest("await installPackages(['file:./fixture-pkg']); return true;"));
  await environment.close();
  await writeFile(join(path, 'main.ts'), "import { answer } from 'natlang-file-package';\nexport default function main(): number { return answer; }\n");
  const host = new NativeNatlangHost({ workspace: path });
  t.after(() => host.close());
  const result = await host.run({ source: { kind: 'file', path: join(path, 'main.ts') } });
  assert.equal(result.outcome.kind, 'done', JSON.stringify(result.outcome));
  assert.equal(result.value, 42);
});

test('native model eval erases type-only imports without package resolution', async t => {
  const path = await workspace(t);
  const environment = new TypeScriptEnvironment({ workspace: path });
  const host = new NativeNatlangHost({ environment });
  t.after(() => { host.close(); environment.close(); });
  let turn = 0;
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: '() => number', instructions: 'Return one.'
  } } }, modelTurn: () => ++turn === 1
    ? { calls: [['eval', { code: "import type { X } from 'not-installed'; import { type Y } from 'also-not-installed'; return 1;" }]], completion_tokens: 1 }
    : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 }, options: { model: { max_turns: 3 } } });
  assert.equal(result.outcome.kind, 'done');
  assert.equal(result.value, 1);
});

test('native eval consumes imported namespaces and fetch responses within each call', async t => {
  const path = await workspace(t);
  const fixture = join(path, 'fixture-pkg');
  await mkdir(fixture);
  await writeFile(join(fixture, 'package.json'), JSON.stringify({ name: 'natlang-staged-package', version: '1.0.0', type: 'module', exports: './index.js' }));
  await writeFile(join(fixture, 'index.js'), 'export const answer = 42;\n');
  const server = createServer((request, response) => {
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ answer: 42 }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  const { port } = server.address();
  const environment = new TypeScriptEnvironment({ workspace: path, network: true });
  const host = new NativeNatlangHost({ environment });
  t.after(() => { host.close(); environment.close(); });
  let turn = 0;
  const result = await host.run({ source: { kind: 'program', program: { $lambda: {
    type: '() => number', instructions: 'Use the installed package and fetched JSON to return 42.'
  } } }, modelTurn: () => ++turn === 1
    ? { calls: [['eval', { code: "const pkg = await import('natlang-staged-package'); return pkg.answer;" }]], completion_tokens: 1 }
    : turn === 2
      ? { calls: [['eval', { code: `const response = await fetch('http://127.0.0.1:${port}/data'); const data = await response.json(); return data.answer;` }]], completion_tokens: 1 }
      : { calls: [['mark_lines', { start: 1 }]], completion_tokens: 1 }, options: { model: { max_turns: 4 } } });
  assert.equal(result.outcome.kind, 'done');
  assert.equal(result.value, 42);
  assert.equal(turn, 3);
});

test('default TypeScript environment uses the current project without an import mode', async () => {
  const environment = new TypeScriptEnvironment();
  try {
    assert.equal(environment.scopeCapabilities.allowModules, true);
    assert.ok(environment.packages?.listAvailableDependencies().includes('yaml'));
    const result = await environment.executeAsync(evalRequest("import YAML from 'yaml'; return typeof YAML.parse;"));
    assert.equal(result.result, 'function');
    await assert.rejects(environment.executeAsync(evalRequest("return (await import('node:fs')).existsSync;")), /Only declared package imports/);
  } finally { environment.close(); }
});
