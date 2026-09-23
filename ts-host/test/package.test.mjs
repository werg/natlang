import assert from 'node:assert/strict';
import { chmodSync, mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { execFileSync, spawnSync } from 'node:child_process';
import test from 'node:test';
import { createPackageArchive, NatlangPackageStore, parsePackageArchive,
  satisfiesVersion, writePackageArchive } from '../dist/index.js';
import { createServer } from 'node:http';
import { promisify } from 'node:util';
import { execFile as execFileCallback } from 'node:child_process';
const execFile = promisify(execFileCallback);

function fixture() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-package-'));
  mkdirSync(join(root, 'program'), { recursive: true });
  writeFileSync(join(root, 'program', 'main.nl'), 'function main: () => string\nreturn "hello"\n');
  writeFileSync(join(root, 'program', 'pixel.bin'), Buffer.from([0, 255, 17, 128]));
  const manifest = { schema: 'natlang.package/v2', name: 'example', version: '1.2.3',
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
  const make = (name, version, dependencies = {}) => createPackageArchive({ schema: 'natlang.package/v2',
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
  const make = (name, packageVersion, dependencies = {}) => createPackageArchive({ schema: 'natlang.package/v2',
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

async function modelServer(t, respond) {
  const wire = [];
  const server = createServer(async (request, response) => {
    let text = ''; for await (const chunk of request) text += chunk;
    const body = JSON.parse(text); wire.push(body);
    const call = respond(body);
    response.writeHead(200, { 'content-type': 'application/json' });
    response.end(JSON.stringify({ choices: [{ message: { content: '', tool_calls: call ? [{ id: `call-${wire.length}`, type: 'function',
      function: { name: call[0], arguments: JSON.stringify(call[1]) } }] : [] } }], usage: { completion_tokens: 1 } }));
  });
  await new Promise(resolve => server.listen(0, '127.0.0.1', resolve));
  t.after(() => new Promise(resolve => { server.closeAllConnections(); server.close(resolve); }));
  return { wire, env: { ...process.env, NATLANG_SERVER: `http://127.0.0.1:${server.address().port}`, NATLANG_MODEL: 'fixture' } };
}

test('CLI builds and runs TypeScript entries, packs and installs applications, and answers instructions', async t => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-cli-'));
  const files = {
    'package.json': JSON.stringify({ name: 'cli-fixture-app', private: true, type: 'module' }),
    'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext',
      strict: true, skipLibCheck: true, outDir: 'dist' }, include: ['*.ts', 'natlang.d/**/*.ts'] }),
    'main.ts': "import { nl } from '@natlang/node';\nimport type { TargetContext } from '@natlang/node';\n" +
      'export async function main(context: TargetContext): Promise<number> {\n' +
      '  const greeting = await context.runtime.run(() => nl<string>`Greet the people named in context.args.`(context.args));\n' +
      '  context.io.output.write(greeting);\n  return 0;\n}\n',
    'natlang.d/helper.ts': 'export function helper(): string { return "local helper"; }\n',
    'project-notes.txt': 'non-source project context',
    'natlang.json': JSON.stringify({ schema: 'natlang.package/v2', name: 'cli-fixture', version: '1.0.0',
      include: ['main.ts', 'natlang.d', 'package.json', 'tsconfig.json'], targets: { hello: { entry: 'main.ts', description: 'Greets.' } } }),
  };
  for (const [path, text] of Object.entries(files)) { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); }
  const { wire, env } = await modelServer(t, body => {
    const opening = String(body.messages[1]?.content ?? ''), turn = body.messages.length;
    const code = opening.includes('Greet the people') ? 'result = "hello " + input.join(" and ")' :
      opening.includes('answer from this codebase') ? 'result = helper.helper() + ": " + (await project.file("project-notes.txt").readText())' : undefined;
    if (!code) return null;
    return turn === 2 ? ['eval', { code }] : turn === 4 ? ['mark_lines', { start: 1 }] : null;
  });
  const cli = join(import.meta.dirname, '..', 'bin', 'natlang.mjs');
  const natlang = (args, options = {}) => execFile(process.execPath, [cli, ...args], { encoding: 'utf8', env, ...options }).then(result => result.stdout);
  const help = await natlang([]);
  assert.match(help, /natlang run \[SOURCE\]/); assert.match(help, /natlang check \[PROJECT\]/);
  assert.equal(await natlang(['check', root]), 'ok\n');
  assert.equal(await natlang(['run', root, '--', 'Ada', 'Grace']), 'hello Ada and Grace');
  assert.equal(await natlang(['run', join(root, 'main.ts'), '--', 'Linus']), 'hello Linus');
  assert.equal(await natlang(['ask', 'answer from this codebase'], { cwd: root }), 'local helper: non-source project context\n');
  assert.ok(wire.some(body => /helper\(\): string {2}# TypeScript/.test(JSON.stringify(body))), 'natlang.d is listed for ask');
  const inspected = JSON.parse(execFileSync(process.execPath, [cli, 'inspect', root, '--json'], { encoding: 'utf8' }));
  assert.equal(inspected.target, 'hello'); assert.equal(inspected.entry, 'main.ts');
  const discovered = JSON.parse(execFileSync(process.execPath, [cli, 'apps', root, '--json'], { encoding: 'utf8' }));
  assert.equal(discovered[0].name, 'cli-fixture');
  const store = join(root, 'store'), archive = join(root, 'fixture.nlpkg');
  assert.match(execFileSync(process.execPath, [cli, 'packages', '--store', store], { encoding: 'utf8' }), /No distribution packages/);
  execFileSync(process.execPath, [cli, 'package', 'pack', root, '--out', archive]);
  execFileSync(process.execPath, [cli, 'package', 'install', archive, '--store', store]);
  assert.match(execFileSync(process.execPath, [cli, 'packages', '--store', store], { encoding: 'utf8' }), /cli-fixture@1\.0\.0/);
  assert.equal(await natlang(['run', 'cli-fixture@1.0.0#hello', '--store', store, '--', 'Barbara']), 'hello Barbara');
  const bad = spawnSync(process.execPath, [cli, 'apps', root, '--store', store], { encoding: 'utf8' });
  assert.equal(bad.status, 1); assert.match(bad.stderr, /option --store is not valid here/);
});
