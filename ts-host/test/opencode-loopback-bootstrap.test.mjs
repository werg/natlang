import test from 'node:test';
import assert from 'node:assert/strict';
import { chmod, mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { dirname, resolve } from 'node:path';
import { spawn } from 'node:child_process';
import { once } from 'node:events';
import { fileURLToPath } from 'node:url';

const repo = resolve(dirname(fileURLToPath(import.meta.url)), '../..');
const bootstrap = resolve(repo, 'scripts/opencode-loopback-bootstrap.mjs');

async function fixture(t, { blockStartup = false } = {}) {
  const root = await mkdtemp(resolve(tmpdir(), 'natlang-opencode-bootstrap-'));
  t.after(() => rm(root, { recursive: true, force: true }));
  const sdk = resolve(root, 'fake-sdk.mjs');
  const clientDir = resolve(root, 'bin');
  const binary = resolve(clientDir, 'opencode');
  const output = resolve(root, 'fresh-output');
  const started = resolve(root, 'sdk-started.json');
  const stopped = resolve(root, 'sdk-stopped');
  await import('node:fs/promises').then(fs => fs.mkdir(clientDir));
  await writeFile(binary, '#!/bin/sh\nexit 0\n');
  await chmod(binary, 0o755);
  const sdkSource = `import { writeFileSync } from 'node:fs';
export async function createOpencode({ signal }) {
  writeFileSync(${JSON.stringify(started)}, JSON.stringify({ cwd: process.cwd() }));
  ${blockStartup ? `if (signal.aborted) throw signal.reason;
  const holdEventLoop = setInterval(() => {}, 1000);
  try { await new Promise((resolve, reject) => {
    signal.addEventListener('abort', () => reject(signal.reason), { once: true });
  }); } finally { clearInterval(holdEventLoop); }` : ''}
  return {
    client: { session: { create(){}, prompt(){}, messages(){}, delete(){}, abort(){} } },
    server: { close() { writeFileSync(${JSON.stringify(stopped)}, 'closed'); } }
  };
}`;
  await writeFile(sdk, sdkSource);
  const child = spawn(process.execPath, [bootstrap, '--sdk-module', sdk, '--client-bin', binary,
    '--out', output, '--model', 'exo-free'], {
    env: { ...process.env, OPENCODE_API_KEY: 'test-only-placeholder', OPENCODE_TEST_STARTED: started },
    stdio: ['ignore', 'pipe', 'pipe']
  });
  let stdout = '', stderr = '';
  child.stdout.setEncoding('utf8').on('data', chunk => { stdout += chunk; });
  child.stderr.setEncoding('utf8').on('data', chunk => { stderr += chunk; });
  t.after(() => { if (child.exitCode === null) child.kill('SIGKILL'); });
  return { root, sdk, binary, output, started, stopped, child, get stdout() { return stdout; }, get stderr() { return stderr; } };
}

async function waitFor(predicate, label) {
  const deadline = Date.now() + 5_000;
  while (Date.now() < deadline) {
    if (await predicate()) return;
    await new Promise(resolveWait => setTimeout(resolveWait, 20));
  }
  throw new Error(`timed out waiting for ${label}`);
}

test('bootstraps from isolated scratch and closes adapter and SDK on SIGTERM', async t => {
  const fx = await fixture(t);
  await waitFor(async () => fx.stdout.includes('"endpoint"'), 'loopback endpoint');
  const scratch = resolve(fx.output, 'scratch');
  const started = JSON.parse(await readFile(fx.started, 'utf8'));
  assert.equal(started.cwd, scratch);
  const config = JSON.parse(await readFile(resolve(fx.output, 'bootstrap-config.json'), 'utf8'));
  assert.equal(config.provider_availability, 'not-probed');
  assert.equal(config.model_id, 'exo-free');
  assert.equal(config.server_bind.hostname, '127.0.0.1');
  assert.equal(JSON.stringify(config).includes('test-only-placeholder'), false);
  const exited = once(fx.child, 'exit');
  fx.child.kill('SIGTERM');
  const [code, signal] = await exited;
  assert.equal(code, 0);
  assert.equal(signal, null);
  assert.equal(await readFile(fx.stopped, 'utf8'), 'closed');
  assert.equal(JSON.parse(await readFile(resolve(fx.output, 'lifecycle.json'), 'utf8')).status, 'stopped');
  assert.equal(fx.stderr, '');
});

test('SIGTERM during SDK startup aborts startup before adapter creation', async t => {
  const fx = await fixture(t, { blockStartup: true });
  await waitFor(async () => {
    try { await readFile(fx.started); return true; } catch { return false; }
  }, 'SDK startup');
  await new Promise(resolveWait => setTimeout(resolveWait, 30));
  const exited = once(fx.child, 'exit');
  fx.child.kill('SIGTERM');
  const [code, signal] = await exited;
  assert.ok(code === 0 || code === 1, `unexpected startup-abort exit code: ${code}`);
  assert.equal(signal, null);
  assert.equal(fx.stdout.includes('"endpoint"'), false);
  assert.equal(JSON.parse(await readFile(resolve(fx.output, 'lifecycle.json'), 'utf8')).status, 'stopped');
  assert.equal(fx.stderr.includes('test-only-placeholder'), false);
});
