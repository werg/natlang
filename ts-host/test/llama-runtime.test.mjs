import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { execFileSync, spawnSync } from 'node:child_process';
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createServer } from 'node:http';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { discoverLlamaRuntime, inspectLlamaServer, installManagedLlamaRuntime,
  isCompatibleLlamaVersion } from '../dist/model/index.js';

function fakeServer(path, version = '0.4.1', build = 10964) {
  writeFileSync(path, `#!${process.execPath}\nconsole.log('version: ${version} (build ${build}, commit b29c606e2)');\n`);
  chmodSync(path, 0o755);
}

test('llama.cpp compatibility checks version and build boundaries', () => {
  assert.equal(isCompatibleLlamaVersion('0.4.1', 10964), true);
  assert.equal(isCompatibleLlamaVersion('0.4.1-dev', 10964), true);
  assert.equal(isCompatibleLlamaVersion('0.4.1', 10963), false);
  assert.equal(isCompatibleLlamaVersion('0.5.0', 20000), false);
});

test('runtime discovery validates explicit and PATH executables', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-discovery-'));
  const compatible = join(root, 'llama-server'), old = join(root, 'old-server');
  fakeServer(compatible); fakeServer(old, '0.3.9', 9000);
  const found = discoverLlamaRuntime({ ...process.env, PATH: root, NATLANG_RUNTIME_HOME: join(root, 'managed') });
  assert.equal(found.selected?.path, compatible);
  assert.equal(found.selected?.source, 'path');
  const rejected = discoverLlamaRuntime({ ...process.env, NATLANG_LLAMA_SERVER: old,
    NATLANG_RUNTIME_HOME: join(root, 'managed') });
  assert.equal(rejected.selected, null);
  assert.equal(rejected.candidates[0]?.compatible, false);
  assert.match(inspectLlamaServer(old, 'explicit').reason, /needs/);
});

test('managed runtime downloads, verifies, extracts, and is rediscovered', { skip: process.platform === 'win32' }, async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-install-'));
  const payload = join(root, 'payload'), binaryDirectory = join(payload, 'build', 'bin');
  mkdirSync(binaryDirectory, { recursive: true }); fakeServer(join(binaryDirectory, 'llama-server'));
  const archive = join(root, 'runtime.tar.gz');
  execFileSync('tar', ['-czf', archive, '-C', payload, '.']);
  const bytes = readFileSync(archive), sha256 = createHash('sha256').update(bytes).digest('hex');
  const server = createServer((_request, response) => { response.writeHead(200, { 'content-length': bytes.length }); response.end(bytes); });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  const address = server.address(); assert.ok(address && typeof address !== 'string');
  const release = { version: '0.4.1', build: 10964, commit: 'b29c606e2',
    compatible: { minimumVersion: '0.4.1', maximumVersionExclusive: '0.5.0', minimumBuild: 10964 },
    artifacts: [{ key: 'fixture', platform: process.platform, arch: process.arch, backend: 'cpu', archive: 'tar.gz',
      url: `http://127.0.0.1:${address.port}/runtime.tar.gz`, bytes: bytes.length, sha256 }] };
  const environment = { ...process.env, PATH: '', NATLANG_RUNTIME_HOME: join(root, 'managed') };
  try {
    const installed = await installManagedLlamaRuntime({ environment, release });
    assert.equal(installed.source, 'managed'); assert.equal(installed.compatible, true);
    const discovered = discoverLlamaRuntime(environment, release);
    assert.equal(discovered.selected?.source, 'managed');
    assert.equal(discovered.selected?.path, installed.path);
  } finally { await new Promise(resolve => server.close(resolve)); }
});

test('CLI setup never downloads without consent in a noninteractive process', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-llama-consent-'));
  const cli = new URL('../dist/cli/main.js', import.meta.url);
  const result = spawnSync(process.execPath, [cli.pathname, '--setup'], {
    env: { ...process.env, PATH: '', NATLANG_RUNTIME_HOME: root, NATLANG_HOME: join(root, 'data') },
    encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'], timeout: 10000,
  });
  assert.equal(result.status, 0);
  assert.match(result.stdout, /runtime unavailable/);
  assert.equal(discoverLlamaRuntime({ ...process.env, PATH: '', NATLANG_RUNTIME_HOME: root }).selected, null);
});
