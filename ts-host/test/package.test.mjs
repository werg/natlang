import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import test from 'node:test';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive,
  satisfiesVersion, writePackageArchive } from '../dist/index.js';
import { main as cliMain } from '../dist/cli/main.js';

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-package-'));
  mkdirSync(join(root, 'program'), { recursive: true });
  writeFileSync(join(root, 'program', 'main.nl'), 'function main: Lambda<{}, Text>\nreturn "hello"\n');
  writeFileSync(join(root, 'program', 'pixel.bin'), Buffer.from([0, 255, 17, 128]));
  const manifest = { schema: 'natlang.package/v1', name: 'example', version: '1.2.3',
    include: ['program'], exports: { main: 'program/main.nl' } };
  return { root, manifest };
}

test('native package archives are deterministic and preserve binary assets', () => {
  const { root, manifest } = fixture();
  const first = createPackageArchive(manifest, root), second = createPackageArchive(manifest, root);
  assert.deepEqual(first, second);
  assert.equal(first.files.find(file => file.path.endsWith('.bin')).content, 'AP8RgA==');
  const path = join(root, 'example.nlpkg'); writePackageArchive(path, first);
  assert.equal(parsePackageArchive(JSON.parse(readFileSync(path, 'utf8'))).digest, first.digest);
});

test('archive verification rejects traversal and modified bytes', () => {
  const { root, manifest } = fixture();
  assert.throws(() => createPackageArchive({ ...manifest, include: ['../secret'] }, root), /escapes/);
  const archive = structuredClone(createPackageArchive(manifest, root));
  archive.files[0].content = Buffer.from('changed').toString('base64');
  assert.throws(() => parsePackageArchive(archive), /checksum mismatch/);
});

test('immutable store installs, resolves, lists, and rejects version rebinding', () => {
  const { root, manifest } = fixture(), store = new NatlangPackageStore(join(root, 'store'));
  const archive = createPackageArchive(manifest, root), installed = store.install(archive);
  assert.equal(installed.digest, archive.digest);
  assert.equal(readFileSync(join(installed.root, 'program', 'pixel.bin')).toString('hex'), '00ff1180');
  assert.deepEqual(store.list().map(row => `${row.name}@${row.version}`), ['example@1.2.3']);
  writeFileSync(join(root, 'program', 'main.nl'), 'changed');
  assert.throws(() => store.install(createPackageArchive(manifest, root)), /already bound/);
});

test('store resolution detects changed installed content', () => {
  const { root, manifest } = fixture(), store = new NatlangPackageStore(join(root, 'store'));
  const installed = store.install(createPackageArchive(manifest, root));
  const path = join(installed.root, 'program', 'main.nl'); chmodSync(path, 0o644); writeFileSync(path, 'tampered');
  assert.throws(() => store.resolve('example@1.2.3'), /content changed/);
});

test('dependency ranges are checked before a package is installed', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-dependencies-'));
  writeFileSync(join(root, 'main.nl'), 'return null');
  const make = (name, version, dependencies = {}) => createPackageArchive({ schema: 'natlang.package/v1',
    name, version, dependencies, include: ['main.nl'] }, root);
  const store = new NatlangPackageStore(join(root, 'store'));
  assert.throws(() => store.install(make('app', '1.0.0', { library: '^2.0.0' })), /needs library/);
  store.installMany([make('app', '1.0.0', { library: '^2.0.0' }), make('library', '2.3.0')]);
  assert.equal(store.resolve('library@2.3.0').version, '2.3.0');
  assert.equal(store.resolve('app@1.0.0').dependencies.library.digest,
    store.resolve('library@2.3.0').digest);
  assert.equal(satisfiesVersion('0.2.4', '^0.2.1'), true);
  assert.equal(satisfiesVersion('0.3.0', '^0.2.1'), false);
});

test('dependency locks choose a stable version and cycles are rejected', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-locks-'));
  writeFileSync(join(root, 'main.nl'), 'return null');
  const make = (name, packageVersion, dependencies = {}) => createPackageArchive({ schema: 'natlang.package/v1',
    name, version: packageVersion, dependencies, include: ['main.nl'] }, root);
  const store = new NatlangPackageStore(join(root, 'store'));
  store.installMany([make('library', '1.0.0'), make('library', '1.4.0'),
    make('app', '1.0.0', { library: '^1.0.0' })]);
  assert.equal(store.resolve('app@1.0.0').dependencies.library.version, '1.4.0');
  store.install(make('library', '1.8.0'));
  assert.equal(store.resolve('app@1.0.0').dependencies.library.version, '1.4.0');
  assert.throws(() => store.installMany([make('a', '1.0.0', { b: '*' }),
    make('b', '1.0.0', { a: '*' })]), /dependency cycle/);
});

test('CLI packs, installs, and runs a target from the content store', async () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-cli-package-'));
  writeFileSync(join(root, 'direct.ts'), `/*---
description: Return a fixture number.
args: {}
returns: Num
---*/
return 7`);
  writeFileSync(join(root, 'helper.ts'), `/*---
description: Return text from the local codebase.
args: {}
returns: Text
---*/
return "local helper"`);
  writeFileSync(join(root, 'ordinary.ts'), 'export const ordinaryHostCode = true;\n');
  writeFileSync(join(root, 'project-notes.txt'), 'non-source project context');
  const administrativeNames = ['apps', 'inspect', 'packages', 'package', 'setup', 'runtime', 'doctor'];
  for (const name of administrativeNames) {
    mkdirSync(join(root, name));
    writeFileSync(join(root, name, 'main.ts'), `/*---
description: Prove administrative words remain valid source paths.
args: {}
returns: Text
---*/
return "source named ${name}"`);
  }
  writeFileSync(join(root, 'target.mjs'), `export function createTarget(context) {
    return { run() { context.io.output.write(context.package.name + ':' + context.args.join(',') + ':' + Object.keys(context.dependencies).length); } };
  }`);
  writeFileSync(join(root, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v1',
    name: 'cli-fixture', version: '1.0.0', include: ['target.mjs'], targets: {
      hello: { kind: 'command', entry: 'target.mjs' },
    } }));
  const archive = join(root, 'fixture.nlpkg'), store = join(root, 'store');
  const cli = join(import.meta.dirname, '..', 'bin', 'natlang.mjs');
  const help = execFileSync(process.execPath, [cli], { encoding: 'utf8' });
  assert.match(help, /natlang SOURCE/); assert.doesNotMatch(help, /natlang app run/);
  const empty = execFileSync(process.execPath, [cli, '--packages', '--store', store], { encoding: 'utf8' });
  assert.match(empty, /No distribution packages are installed/);
  const direct = execFileSync(process.execPath, [cli, join(root, 'direct.ts')], { encoding: 'utf8' });
  assert.equal(direct, '7\n');
  for (const name of administrativeNames) {
    const collidingPath = execFileSync(process.execPath, [cli, name], { encoding: 'utf8', cwd: root });
    assert.equal(collidingPath, `"source named ${name}"\n`);
  }
  const originalFetch = globalThis.fetch, originalWrite = process.stdout.write, originalCwd = process.cwd();
  const previousServer = process.env.NATLANG_SERVER, previousModel = process.env.NATLANG_MODEL;
  let wire, anonymousOutput = '', anonymousTurn = 0;
  globalThis.fetch = async (_url, init) => {
    wire = JSON.parse(init.body);
    anonymousTurn++;
    const call = anonymousTurn === 1 ? { name: 'read', arguments: JSON.stringify({ path: 'files/project-notes.txt' }) } :
      anonymousTurn === 2 ? { name: 'write', arguments: JSON.stringify({
        path: 'return', type: 'Text', value: 'anonymous result', done: 1 }) } : undefined;
    return new Response(JSON.stringify({ choices: [{ message: { content: '', tool_calls: call ? [{
      id: `anonymous-${anonymousTurn}`, type: 'function', function: call }] : [] } }], usage: { completion_tokens: 1 } }), { status: 200,
      headers: { 'content-type': 'application/json' } });
  };
  process.env.NATLANG_SERVER = 'http://model.test'; process.env.NATLANG_MODEL = 'fixture';
  process.stdout.write = chunk => { anonymousOutput += String(chunk); return true; };
  process.chdir(root);
  try { assert.equal(await cliMain(['answer from this codebase', '--timeout', '5000']), 0); }
  finally {
    process.chdir(originalCwd); process.stdout.write = originalWrite; globalThis.fetch = originalFetch;
    if (previousServer === undefined) delete process.env.NATLANG_SERVER; else process.env.NATLANG_SERVER = previousServer;
    if (previousModel === undefined) delete process.env.NATLANG_MODEL; else process.env.NATLANG_MODEL = previousModel;
  }
  assert.equal(anonymousOutput, 'anonymous result\n');
  assert.match(JSON.stringify(wire), /answer from this codebase/);
  assert.match(JSON.stringify(wire), /helper/);
  assert.match(JSON.stringify(wire), /non-source project context/);
  const local = execFileSync(process.execPath, [cli, root, '--', 'local'], { encoding: 'utf8' });
  assert.equal(local, 'cli-fixture:local:0');
  const localManifest = execFileSync(process.execPath, [cli, join(root, 'natlang.json'), '--', 'path'],
    { encoding: 'utf8', cwd: tmpdir() });
  assert.equal(localManifest, 'cli-fixture:path:0');
  const inspected = JSON.parse(execFileSync(process.execPath, [cli, '--inspect', root, '--json'], { encoding: 'utf8' }));
  assert.equal(inspected.target, 'hello');
  const discovered = JSON.parse(execFileSync(process.execPath, [cli, '--apps', root, '--json'], { encoding: 'utf8' }));
  assert.equal(discovered[0].name, 'cli-fixture');
  const ignored = spawnSync(process.execPath, [cli, '--apps', root, '--store', store], { encoding: 'utf8' });
  assert.equal(ignored.status, 1); assert.match(ignored.stderr, /option --store is not valid here/);
  const emptyDirectory = join(root, 'not-an-app'); mkdirSync(emptyDirectory);
  const invalidApp = spawnSync(process.execPath, [cli, '--package', 'pack', emptyDirectory], { encoding: 'utf8' });
  assert.equal(invalidApp.status, 1);
  assert.match(invalidApp.stderr, /does not contain a valid natlang\.json/);
  assert.doesNotMatch(invalidApp.stderr, /EISDIR/);
  execFileSync(process.execPath, [cli, '--package', 'pack', join(root, 'natlang.json'), '--out', archive]);
  execFileSync(process.execPath, [cli, '--package', 'install', archive, '--store', store]);
  const listing = execFileSync(process.execPath, [cli, '--packages', '--store', store], { encoding: 'utf8' });
  assert.match(listing, /cli-fixture@1\.0\.0/);
  const result = execFileSync(process.execPath, [cli, 'cli-fixture@1.0.0#hello',
    '--store', store, '--', 'one', 'two'], { encoding: 'utf8' });
  assert.equal(result, 'cli-fixture:one,two:0');
  const convenient = execFileSync(process.execPath, [cli, 'cli-fixture',
    '--store', store, '--', 'three'], { encoding: 'utf8' });
  assert.equal(convenient, 'cli-fixture:three:0');
});
