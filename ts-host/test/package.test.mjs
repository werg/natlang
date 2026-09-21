import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive,
  satisfiesVersion, writePackageArchive } from '../dist/index.js';

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
  assert.equal(satisfiesVersion('0.2.4', '^0.2.1'), true);
  assert.equal(satisfiesVersion('0.3.0', '^0.2.1'), false);
});

test('CLI packs, installs, and runs a target from the content store', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-cli-package-'));
  writeFileSync(join(root, 'target.mjs'), `export function createTarget(context) {
    return { run() { context.io.output.write(context.package.name + ':' + context.args.join(',')); } };
  }`);
  writeFileSync(join(root, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v1',
    name: 'cli-fixture', version: '1.0.0', include: ['target.mjs'], targets: {
      hello: { kind: 'command', entry: 'target.mjs' },
    } }));
  const archive = join(root, 'fixture.nlpkg'), store = join(root, 'store');
  const cli = join(import.meta.dirname, '..', 'bin', 'natlang.mjs');
  execFileSync(process.execPath, [cli, 'package', 'pack', join(root, 'natlang.json'), '--out', archive]);
  execFileSync(process.execPath, [cli, 'package', 'install', archive, '--store', store]);
  const result = execFileSync(process.execPath, [cli, 'run', 'cli-fixture@1.0.0#hello',
    '--store', store, '--', 'one', 'two'], { encoding: 'utf8' });
  assert.equal(result, 'cli-fixture:one,two');
});
