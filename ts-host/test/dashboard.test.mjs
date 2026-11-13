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

test('a run whose local model is not downloaded asks first; a confirmed run passes --yes; new functions come from a template', async () => {
  const { root, environment } = workspace();
  const dashboard = await startDashboard({ root, version: 'test', environment, port: 0, log: { write: () => true } });
  try {
    const base = new URL(dashboard.url), origin = `${base.protocol}//${base.host}`, token = base.searchParams.get('token');
    const headers = { 'x-natlang-token': token, cookie: `natlang_dashboard=${token}`, 'content-type': 'application/json' };
    const post = async (path, body) => { const response = await fetch(origin + path, { method: 'POST', headers, body: JSON.stringify(body) });
      return { status: response.status, value: await response.json() }; };
    const readiness = await (await fetch(`${origin}/api/programs/model?path=greeter&profile=`, { headers })).json();
    assert.equal(readiness.kind, 'managed-local');
    assert.equal(readiness.setup.bytes, 1099627808, 'the program model (Q4_K_M) is not downloaded');
    const refused = await post('/api/jobs', { kind: 'run', path: 'greeter' });
    assert.equal(refused.status, 409);
    assert.match(refused.value.error, /downloads the local model \(1\.0 GB\) first/);
    assert.deepEqual(refused.value.setup, readiness.setup);
    assert.equal((await post('/api/jobs', { kind: 'call', function: 'greeter/greet.nl', inputs: { name: 'Ada' }, program: 'greeter' })).status, 409);
    assert.equal((await post('/api/jobs', { kind: 'test', path: 'greeter' })).status, 409);
    // Confirmed: the job carries --yes (greeter has no saved examples, so this test job ends before any model work).
    const confirmed = await post('/api/jobs', { kind: 'test', path: 'greeter', download: true });
    assert.equal(confirmed.status, 200); assert.ok(confirmed.value.argv.includes('--yes'));
    assert.ok(!(await post('/api/jobs', { kind: 'build', path: 'greeter' })).value.argv.includes('--yes'), 'a build without warm-up downloads nothing');
    // A hosted or external model needs no download, so nothing is asked.
    const external = await startDashboard({ root, version: 'test', environment: { ...environment, NATLANG_SERVER: 'http://127.0.0.1:1', NATLANG_MODEL: 'x' }, port: 0, log: { write: () => true } });
    try {
      const url = new URL(external.url), externalToken = url.searchParams.get('token');
      const response = await fetch(`${url.protocol}//${url.host}/api/programs/model?path=greeter`, { headers: { 'x-natlang-token': externalToken, cookie: `natlang_dashboard=${externalToken}` } });
      assert.equal((await response.json()).setup, null);
    } finally { await external.close(); }

    const created = await post('/api/programs/function', { path: 'greeter', file: 'greet/polite.nl' });
    assert.equal(created.status, 200); assert.equal(created.value.path, join('greeter', 'greet', 'polite.nl'));
    assert.match(readFileSync(join(root, 'greeter', 'greet', 'polite.nl'), 'utf8'), /^---\ndescription: .*\nargs:\n  text: string\nreturns: string\n---\n/);
    assert.equal((await post('/api/programs/function', { path: 'greeter', file: 'greet/polite.nl' })).status, 409, 'never overwrites');
    assert.equal((await post('/api/programs/function', { path: 'greeter', file: 'bad name.nl' })).status, 400);
    assert.equal((await post('/api/programs/function', { path: 'greeter', file: '../escape.nl' })).status, 400);
  } finally { await dashboard.close(); }
});
