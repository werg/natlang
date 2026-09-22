import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { PackageRegistry } from '../../applications/package_registry.mjs';
import { evalTurn } from './support/eval-turn.mjs';

const resolvePath = fileURLToPath(new URL('../../codebases/packages/resolve.nl', import.meta.url));
const installPath = fileURLToPath(new URL('../../codebases/packages/install.nl', import.meta.url));
const packages = [
  { name: 'core', version: '1.0.0', description: 'Normalize text', dependencies: {},
    engines: ['typescript-host'], definitions: { normalize: { args: { text: 'string' }, returns: 'string',
      engine: 'typescript-host', code: 'return text.trim().toUpperCase();' } } },
  { name: 'core', version: '2.0.0', description: 'Changed normalization contract', dependencies: {},
    engines: ['typescript-host'], definitions: { normalize: { args: { text: 'string' }, returns: 'string',
      engine: 'typescript-host', code: 'return text.toLowerCase();' } } },
  ...['1.0.0', '1.1.0'].map(version => ({ name: 'textutil', version,
    description: `string utility ${version}`, dependencies: { core: '^1.0.0' },
    engines: ['typescript-host'], definitions: { main: { args: { text: 'string' },
      returns: 'string', instructions: 'Call normalize on text and return its result.',
      uses: { normalize: 'core/normalize' } } } })),
];

test('natlang resolves and installs a checked offline source bundle', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-packages-'));
  const registry = new PackageRegistry(packages, folder);
  const host = new NatlangHost({ host: { packages: registry,
    drainEvents: () => registry.drainEvents() }, mode: 'retained' });
  const request = { name: 'textutil', range: '^1.0.0', engine: 'typescript-host',
    purpose: 'Use the latest compatible text utility.' };
  const modelTurn = turn => {
    const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
    if (prompt.includes('function resolve(')) return evalTurn(turn,
      'const locks = await solutions(request);\n' +
      'const selected = await choose(request, locks);\n' +
      'await finalize(request, locks, selected)');
    if (prompt.includes('Choose the ID')) return evalTurn(turn,
      'JSON.parse(' + JSON.stringify(JSON.stringify(registry.solutions(request)[0].id)) + ')');
    if (prompt.includes('function install(')) return evalTurn(turn,
      'const checked = await validate(lock);\nawait publish(lock, target)');
    return evalTurn(turn, 'await normalize(text)');
  };
  try {
    const resolved = await host.run({ source: { kind: 'file', path: resolvePath },
      inputs: { request }, modelTurn });
    assert.equal(resolved.outcome.kind, 'done');
    assert.equal(resolved.value.status, 'resolved');
    assert.equal(resolved.value.alternatives, 2);
    assert.deepEqual(Array.from(resolved.value.lock.packages).map(pin => pin.version), ['1.0.0', '1.1.0']);
    const installed = await host.run({ source: { kind: 'file', path: installPath },
      inputs: { lock: resolved.value.lock, target: 'textapp' }, modelTurn });
    assert.equal(installed.value.status, 'installed');
    const workspace = await registry.load('textapp');
    let turns = 0;
    const result = await workspace.invoke('textutil__main', { text: ' hi ' }, {
      modelTurn: request => evalTurn(request, 'await normalize(text)') });
    assert.equal(result.outcome, 'done');
    assert.equal(result.value, 'HI');
    await assert.rejects(registry.install(resolved.value.lock, 'textapp'), /already exists/);
    const bundlePath = join(realpathSync(join(folder, 'textapp')), 'bundle.json');
    const prior = readFileSync(bundlePath, 'utf8');
    writeFileSync(bundlePath, prior.replace('trim().toUpperCase()', 'toLowerCase()'));
    await assert.rejects(registry.load('textapp'), /installed bundle changed/);
  } finally { host.close(); rmSync(folder, { recursive: true, force: true }); }
});

test('conflicts, engine mismatch, cycles, tampered locks and path traversal fail exactly', async () => {
  const folder = mkdtempSync(join(tmpdir(), 'natlang-packages-'));
  try {
    const registry = new PackageRegistry(packages, folder);
    assert.equal(registry.solutions({ name: 'textutil', range: '^2.0.0', engine: 'typescript-host' }).length, 0);
    assert.equal(registry.solutions({ name: 'textutil', range: '*', engine: 'quickjs-isolated' }).length, 0);
    const lock = registry.solutions({ name: 'textutil', range: '*', engine: 'typescript-host' })[0];
    assert.throws(() => registry.bundle({ ...lock, id: 'tampered' }), /lock does not match/);
    await assert.rejects(registry.install(lock, '../escape'), /invalid install target/);
    const cycle = new PackageRegistry([
      { name: 'a', version: '1.0.0', engines: ['typescript-host'], dependencies: { b: '*' },
        definitions: { main: { returns: 'number', engine: 'typescript-host', code: 'return 1;' } } },
      { name: 'b', version: '1.0.0', engines: ['typescript-host'], dependencies: { a: '*' },
        definitions: { main: { returns: 'number', engine: 'typescript-host', code: 'return 2;' } } },
    ], folder);
    assert.equal(cycle.solutions({ name: 'a', range: '*', engine: 'typescript-host' }).length, 0);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
