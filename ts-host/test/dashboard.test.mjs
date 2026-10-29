import test from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { startDashboard } from '../dist/dashboard/server.js';

function workspace() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-dashboard-'));
  const home = join(root, 'home');
  const project = join(root, 'project', 'greeter');
  mkdirSync(project, { recursive: true });
  writeFileSync(join(project, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'greeter', version: '0.1.0',
    description: 'Greets.', include: ['greet.nl'], model: { stock: 'lfm3-1.7b-q4_k_m' },
    warmup: [{ function: 'greet.nl', inputs: { name: 'Ada' } }] }));
  writeFileSync(join(project, 'greet.nl'), '---\ndescription: Greet someone.\nargs:\n  name: string\nreturns: string\n---\nGreet name warmly.\n');
  const environment = { ...process.env, NATLANG_CONFIG_HOME: join(home, 'config'), NATLANG_STATE_HOME: join(home, 'state'),
    NATLANG_CACHE_HOME: join(home, 'cache'), NATLANG_HOME: join(home, 'data'), NATLANG_CALL_STORE: 'off', NATLANG_RUNTIME_HOME: join(home, 'runtime') };
  return { root: join(root, 'project'), environment, home };
}

test('the dashboard refuses requests without its session token or from another host name', async () => {
  const { root, environment } = workspace();
  const dashboard = await startDashboard({ root, version: 'test', environment, port: 0, log: { write: () => true } });
  try {
    const base = new URL(dashboard.url), origin = `${base.protocol}//${base.host}`;
    assert.equal((await fetch(`${origin}/api/status`)).status, 401);
    const { request } = await import('node:http');
    const rebound = await new Promise((resolve, reject) => request({ host: base.hostname, port: base.port, path: '/api/status',
      headers: { host: 'attacker.example' } }, response => { response.resume(); resolve(response.statusCode); }).on('error', reject).end());
    assert.equal(rebound, 403);
    const token = base.searchParams.get('token');
    const redirect = await fetch(dashboard.url, { redirect: 'manual' });
    assert.equal(redirect.status, 302);
    assert.match(redirect.headers.get('set-cookie') ?? '', /natlang_dashboard=/);
    // The header alone (no cookie) is not enough either.
    assert.equal((await fetch(`${origin}/api/status`, { headers: { 'x-natlang-token': token } })).status, 401);
  } finally { await dashboard.close(); }
});

test('the dashboard lists programs, saves validated profiles and stores provider keys privately', async () => {
  const { root, environment, home } = workspace();
  const dashboard = await startDashboard({ root, version: 'test', environment, port: 0, log: { write: () => true } });
  try {
    const base = new URL(dashboard.url), origin = `${base.protocol}//${base.host}`, token = base.searchParams.get('token');
    const call = async (method, path, body) => {
      const response = await fetch(origin + path, { method, headers: { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`,
        ...(body ? { 'content-type': 'application/json' } : {}) }, ...(body ? { body: JSON.stringify(body) } : {}) });
      return { status: response.status, value: await response.json() };
    };
    const programs = await call('GET', '/api/programs');
    assert.equal(programs.value.length, 1);
    assert.equal(programs.value[0].name, 'greeter'); assert.equal(programs.value[0].functions, 1);
    const detail = await call('GET', '/api/programs/detail?path=greeter');
    assert.deepEqual(detail.value.functions.map(fn => [fn.name, fn.params, fn.returns]), [['greet', ['name: string'], 'string']]);
    assert.equal(detail.value.manifest.model.stock, 'lfm3-1.7b-q4_k_m');
    assert.equal((await call('GET', '/api/programs/detail?path=../..')).status, 400, 'paths outside the root are refused');

    assert.equal((await call('PUT', '/api/profiles/fast', { profile: { local: { stock: 'lfm3-1.7b-q4_k_m' } }, makeDefault: true })).status, 200);
    assert.equal((await call('PUT', '/api/profiles/broken', { profile: { local: { stock: 'no-such-model' } } })).status, 400);
    const config = JSON.parse(readFileSync(join(home, 'config', 'config.json'), 'utf8'));
    assert.equal(config.defaultProfile, 'fast');
    assert.equal(config.profiles.broken, undefined);
    const status = await call('GET', '/api/status');
    assert.equal(status.value.model.profile, 'fast'); assert.equal(status.value.model.release.id, 'lfm3-1.7b-q4_k_m');

    assert.equal((await call('PUT', '/api/credentials/env:MY_KEY', { key: 'secret-value' })).status, 200);
    const { storedApiKey } = await import('../dist/model/config.js');
    assert.equal(storedApiKey('MY_KEY', environment), 'secret-value');
    const listed = await call('GET', '/api/config');
    assert.deepEqual(listed.value.credentials, [{ id: 'env:MY_KEY', type: 'api_key' }]);
    assert.ok(!JSON.stringify(listed.value).includes('secret-value'), 'keys are never sent back');
  } finally { await dashboard.close(); }
});

test('a dashboard job runs the CLI, records its output and writes the build lock with the program model', async () => {
  const { root, environment } = workspace();
  const dashboard = await startDashboard({ root, version: 'test', environment, port: 0, log: { write: () => true } });
  try {
    const base = new URL(dashboard.url), origin = `${base.protocol}//${base.host}`, token = base.searchParams.get('token');
    const headers = { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`, 'content-type': 'application/json' };
    const job = await (await fetch(`${origin}/api/jobs`, { method: 'POST', headers, body: JSON.stringify({ kind: 'build', path: 'greeter' }) })).json();
    assert.equal(job.status, 'running');
    let state;
    for (let attempt = 0; attempt < 600; attempt++) {
      state = await (await fetch(`${origin}/api/jobs/${job.id}`, { headers })).json();
      if (state.job.status !== 'running') break;
      await new Promise(resolve => setTimeout(resolve, 100));
    }
    assert.equal(state.job.status, 'done', state.stderr.text);
    assert.match(state.stdout.text, /lock: .*natlang\.lock\.json/);
    const lock = JSON.parse(readFileSync(join(root, 'greeter', '.natlang', 'build', 'natlang.lock.json'), 'utf8'));
    assert.equal(lock.model.release.id, 'lfm3-1.7b-q4_k_m', 'the program model wins over the machine default');
    assert.deepEqual(lock.functions.map(fn => fn.name), ['greet']);
    assert.equal(lock.warm, null);
  } finally { await dashboard.close(); }
});
