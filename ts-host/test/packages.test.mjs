import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, readFileSync, realpathSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { NatlangHost } from '../dist/index.js';
import { PackageRegistry } from '../../applications/package_registry.mjs';

const resolvePath = fileURLToPath(new URL('../../codebases/packages/resolve.nl', import.meta.url));
const installPath = fileURLToPath(new URL('../../codebases/packages/install.nl', import.meta.url));
const packages = [
  { name: 'core', version: '1.0.0', description: 'Normalize text', dependencies: {},
    engines: ['typescript-host'], definitions: { normalize: { args: { text: 'Text' }, returns: 'Text',
      engine: 'typescript-host', code: 'return args.text.trim().toUpperCase();' } } },
  { name: 'core', version: '2.0.0', description: 'Changed normalization contract', dependencies: {},
    engines: ['typescript-host'], definitions: { normalize: { args: { text: 'Text' }, returns: 'Text',
      engine: 'typescript-host', code: 'return args.text.toLowerCase();' } } },
  ...['1.0.0', '1.1.0'].map(version => ({ name: 'textutil', version,
    description: `Text utility ${version}`, dependencies: { core: '^1.0.0' },
    engines: ['typescript-host'], definitions: { main: { args: { text: 'Text' },
      returns: 'Text', instructions: 'Call normalize on text and return its result.',
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
    if (turn.messages.filter(m => m.role === 'assistant').length > 1)
      return { calls: [], text: 'done', completion_tokens: 1 };
    const prompt = String(turn.messages.find(m => m.role === 'user')?.content ?? '');
    if (prompt.includes('function resolve(')) return { calls: [
      ['call', { function: 'solutions', to: 'let/locks', inputs: { request: 'args/request' } }],
      ['call', { function: 'choose', to: 'let/selected', inputs: {
        request: 'args/request', locks: 'let/locks' } }],
      ['call', { function: 'finalize', to: 'return', inputs: {
        request: 'args/request', locks: 'let/locks', selected: 'let/selected' } }],
    ], completion_tokens: 1 };
    if (prompt.includes('Choose the ID')) return { calls: [['write', {
      path: 'return', value: registry.solutions(request)[0].id }]], completion_tokens: 1 };
    if (prompt.includes('function install(')) return { calls: [
      ['call', { function: 'validate', to: 'let/checked', inputs: { lock: 'args/lock' } }],
      ['call', { function: 'publish', to: 'return', inputs: {
        lock: 'args/lock', target: 'args/target' } }],
    ], completion_tokens: 1 };
    return { calls: [], text: 'done', completion_tokens: 1 };
  };
  try {
    const resolved = await host.run({ source: { kind: 'file', path: resolvePath },
      inputs: { request }, modelTurn, options: { model: { segment_turns: 2 } } });
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
      modelTurn: () => ++turns === 1 ? { calls: [['call', { function: 'normalize',
        to: 'return', inputs: { text: 'args/text' } }]], completion_tokens: 1 } :
        { calls: [], text: 'done', completion_tokens: 1 } });
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
        definitions: { main: { returns: 'Num', engine: 'typescript-host', code: 'return 1;' } } },
      { name: 'b', version: '1.0.0', engines: ['typescript-host'], dependencies: { a: '*' },
        definitions: { main: { returns: 'Num', engine: 'typescript-host', code: 'return 2;' } } },
    ], folder);
    assert.equal(cycle.solutions({ name: 'a', range: '*', engine: 'typescript-host' }).length, 0);
  } finally { rmSync(folder, { recursive: true, force: true }); }
});
