import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync, rmSync, symlinkSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { spawn } from 'node:child_process';
import { createServer } from 'node:http';
import { createProgram, readExamples, saveExample } from '../dist/cli/authoring.js';
import { parsePackageManifest } from '../dist/package/manifest.js';
import { startDashboard } from '../dist/dashboard/server.js';

const cli = new URL('../bin/natlang.mjs', import.meta.url).pathname;
function run(args, cwd, env) {
  return new Promise((resolve, reject) => {
    const child = spawn(process.execPath, [cli, ...args], { cwd, env, stdio: ['ignore', 'pipe', 'pipe'] });
    let out = '', err = '';
    child.stdout.on('data', chunk => { out += chunk; }); child.stderr.on('data', chunk => { err += chunk; });
    child.once('error', reject); child.once('close', code => resolve({ code, out, err }));
  });
}

test('a newly scaffolded program checks, runs, and checks saved expected results', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-authoring-')), program = join(root, 'my-program');
  let answer = 'billing', requests = 0;
  const server = createServer((request, response) => {
    request.resume(); request.on('end', () => {
      response.setHeader('content-type', 'application/json');
      if (request.url.endsWith('/models')) return response.end(JSON.stringify({ data: [{ id: 'fixture' }] }));
      requests++;
      response.end(JSON.stringify({ choices: [{ message: { content: JSON.stringify(answer), tool_calls: [] }, finish_reason: 'stop' }], usage: { completion_tokens: 1 } }));
    });
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const env = { ...process.env, NATLANG_SERVER: `http://127.0.0.1:${server.address().port}`, NATLANG_MODEL: 'fixture',
    NATLANG_CALL_STORE: 'off', NATLANG_HASH_CACHE: join(root, 'hashes.jsonl'), NATLANG_CONFIG_HOME: root, NATLANG_STATE_HOME: join(root, 'state') };
  try {
    const init = await run(['init', program], root, env); assert.equal(init.code, 0, init.err);
    const original = readFileSync(join(program, 'answer.nl'), 'utf8');
    assert.notEqual((await run(['init', program], root, env)).code, 0);
    assert.equal(readFileSync(join(program, 'answer.nl'), 'utf8'), original, 'init never overwrites a project');
    const checked = await run(['check'], program, env); assert.equal(checked.code, 0, checked.err);
    assert.equal(requests, 0, 'creation and checking never contact the model');
    const result = await run(['run'], program, env); assert.equal(result.code, 0, result.err); assert.equal(JSON.parse(result.out), 'billing');
    const manifestPath = join(program, 'natlang.json'), manifest = JSON.parse(readFileSync(manifestPath, 'utf8'));
    manifest.model = { endpoint: env.NATLANG_SERVER, model: 'fixture' }; writeFileSync(manifestPath, JSON.stringify(manifest));
    const noOverrides = { ...env }; delete noOverrides.NATLANG_SERVER; delete noOverrides.NATLANG_MODEL;
    const called = await run(['call', join(program, 'answer.nl'), '--inputs', join(program, 'sample.json')], root, noOverrides);
    assert.equal(called.code, 0, called.err); assert.equal(called.out.trim(), 'billing', 'calling a file from another directory respects its program model');
    const passed = await run(['test'], program, env); assert.equal(passed.code, 0, passed.err); assert.match(passed.out, /1\/1 passed/);
    answer = 'other';
    const failed = await run(['test'], program, env); assert.equal(failed.code, 1); assert.match(failed.out, /expected: "billing"/); assert.match(failed.out, /received: "other"/);
    const healthy = await run(['doctor', '--json'], root, env); assert.equal(healthy.code, 0, healthy.err);
    assert.equal(JSON.parse(healthy.out).inferenceTested, false);
    const unreachable = await run(['doctor', '--json'], root, { ...env, NATLANG_SERVER: 'http://127.0.0.1:1' });
    assert.equal(unreachable.code, 1); assert.equal(JSON.parse(unreachable.out).connection.reachable, false);
    saveExample(program, { function: 'answer.nl', inputs: { input: 'test' }, expected: 'other' });
    saveExample(program, { function: 'answer.nl', inputs: { input: 'test' }, expected: 'billing' });
    assert.equal(readExamples(program).length, 2, 'saving the same input updates its expectation');
    assert.throws(() => saveExample(program, { function: '../escape.nl', inputs: {}, expected: null }), /inside/);
    writeFileSync(join(program, 'examples.json'), '{}');
    assert.notEqual((await run(['test'], program, env)).code, 0, 'malformed examples cannot silently pass');
  } finally { await new Promise(resolve => server.close(resolve)); rmSync(root, { recursive: true, force: true }); }
});

test('Playground instructions and typed return values survive scaffold serialization', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-scaffold-'));
  try {
    const program = createProgram(join(root, 'from-playground'), { instruction: 'Summarize input.\nKeep it short.', input: 'hello', returns: '{ summary: string }', profile: 'chosen-profile' });
    assert.match(readFileSync(join(program, 'answer.nl'), 'utf8'), /returns: "\{ summary: string \}"/);
    assert.deepEqual(JSON.parse(readFileSync(join(program, 'sample.json'), 'utf8')), { input: 'hello' });
    assert.deepEqual(readExamples(program), []);
    const manifest = JSON.parse(readFileSync(join(program, 'natlang.json'), 'utf8'));
    assert.equal(manifest.model.profile, 'chosen-profile');
    assert.equal(parsePackageManifest(manifest).targets.main.arguments[0].type, 'string');
    manifest.targets.main.arguments[0].type = 'not-a-type';
    assert.throws(() => parsePackageManifest(manifest), /argument type/);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

test('dashboard creation and edits preserve existing files, reject stale writes, and keep paths in the workspace', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-dashboard-authoring-'));
  const environment = { ...process.env, NATLANG_CONFIG_HOME: join(root, '.config'), NATLANG_STATE_HOME: join(root, '.state'), NATLANG_CALL_STORE: 'off' };
  const dashboard = await startDashboard({ root, environment, version: 'test', port: 0, log: { write() {} } });
  const url = new URL(dashboard.url), token = url.searchParams.get('token');
  const api = async (method, path, body) => {
    const response = await fetch(url.origin + path, { method, headers: { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`, 'content-type': 'application/json' }, ...(body ? { body: JSON.stringify(body) } : {}) });
    return { status: response.status, body: await response.json() };
  };
  try {
    assert.equal((await api('POST', '/api/programs', { name: 'new-program' })).status, 200);
    assert.equal((await api('POST', '/api/programs', { name: 'new-program' })).status, 409);
    assert.equal((await api('POST', '/api/programs', { name: '../escape' })).status, 400);
    const path = 'new-program/answer.nl', original = readFileSync(join(root, path), 'utf8');
    assert.equal((await api('PUT', '/api/programs/source', { path, previous: original, text: original + '\nBe concise.\n' })).status, 200);
    assert.equal((await api('PUT', '/api/programs/source', { path, previous: original, text: original })).status, 409);
    const saved = await api('POST', '/api/programs/example', { path: 'new-program', example: { function: 'answer.nl', inputs: { input: 'new' }, expected: 'other' } });
    assert.equal(saved.status, 200); assert.equal(saved.body.examples, 2);
    symlinkSync(tmpdir(), join(root, 'outside'));
    assert.equal((await api('GET', '/api/programs/source?path=outside/anything.nl')).status, 400);
    const custom = await api('POST', '/api/programs', { name: 'custom', instruction: 'Summarize input.', input: 'example', returns: '{ summary: string }' });
    assert.equal(custom.status, 200);
    assert.equal(JSON.parse(readFileSync(join(root, 'custom', 'sample.json'), 'utf8')).input, 'example');
  } finally { await dashboard.close(); rmSync(root, { recursive: true, force: true }); }
});

test('dev reruns input changes and stops cleanly without reacting to build output', { timeout: 30000 }, async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-dev-'));
  writeFileSync(join(root, 'main.ts'), `import { readFileSync } from 'node:fs';\nexport function main() { console.log(readFileSync(${JSON.stringify(join(root, 'input.json'))}, 'utf8')); }\n`);
  writeFileSync(join(root, 'input.json'), 'first');
  const env = { ...process.env, NATLANG_SERVER: 'http://127.0.0.1:1', NATLANG_MODEL: 'fixture', NATLANG_CALL_STORE: 'off', NATLANG_HASH_CACHE: join(root, 'hashes.jsonl'), NATLANG_CONFIG_HOME: root };
  const child = spawn(process.execPath, [cli, 'dev', join(root, 'main.ts')], { cwd: root, env, stdio: ['ignore', 'pipe', 'pipe'] });
  let out = '', err = ''; child.stdout.on('data', chunk => { out += chunk; }); child.stderr.on('data', chunk => { err += chunk; });
  const until = async predicate => { const deadline = Date.now() + 20000; while (!predicate()) { if (Date.now() > deadline) throw new Error(`watch timed out: ${out}\n${err}`); await new Promise(resolve => setTimeout(resolve, 50)); } };
  try {
    await until(() => out.includes('first'));
    writeFileSync(join(root, 'input.json'), 'second');
    await until(() => out.includes('second'));
    child.kill('SIGINT');
    assert.equal(await new Promise(resolve => child.once('close', resolve)), 0);
    assert.equal((out.match(/first/g) ?? []).length, 1);
    assert.equal((out.match(/second/g) ?? []).length, 1);
  } finally { if (child.exitCode === null) { child.kill('SIGKILL'); await new Promise(resolve => child.once('close', resolve)); } rmSync(root, { recursive: true, force: true }); }
});
