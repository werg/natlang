import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import test from 'node:test';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive,
  writePackageArchive } from '../dist/index.js';

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
