import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject, checkProject, parsePackageManifest, parseRefinementSettings } from '../dist/index.js';

const RUNTIME = { url: new URL('../dist/index.js', import.meta.url).href,
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

function project(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-refined-'));
  for (const [path, text] of Object.entries(files)) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

const FILES = {
  'package.json': JSON.stringify({ name: 'fixture-refined', private: true, type: 'module' }),
  'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext',
    strict: true, skipLibCheck: true, rootDir: 'src', outDir: 'dist' }, include: ['src/**/*.ts'] }),
  'src/reply.nl': '---\nargs:\n  complaint: string\nreturns: \'Is<string, "a reply that is polite">\'\n---\nAnswer the complaint.\n',
};

test('a refined result is a string everywhere and the declaration emits the Is brand', () => {
  const root = project({ ...FILES, 'src/app.ts': `import reply from './reply.nl';
import type { Is } from '@natlang/node';
export async function run(complaint: string): Promise<string> {
  const answer = await reply(complaint);
  const text: string = answer;
  const evidence: Is<string, "a reply that is polite"> = answer;
  return text + evidence;
}
` });
  const result = buildProject({ project: root, runtimeModule: RUNTIME });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics, null, 1));
  const declaration = readFileSync(join(root, 'src/reply.d.nl.ts'), 'utf8');
  assert.match(declaration, /import type \{ NatlangFunction, Folder as FolderStore, FolderHandle, Is \}/);
  assert.match(declaration, /NatlangFunction<\[complaint: string\], Is<string, "a reply that is polite">>/);
});

test('crisp TypeScript cannot pass a plain string where a refined value is expected', () => {
  const root = project({ ...FILES, 'src/app.ts': `import type { Is } from '@natlang/node';
export function send(body: Is<string, "a reply that is polite">): string { return body; }
export const bad = send("plain string");
` });
  const result = checkProject(root, { runtimeModule: RUNTIME });
  assert.equal(result.ok, false);
  assert.ok(result.diagnostics.some(item => item.file.endsWith('app.ts') && /not assignable/.test(item.message)), JSON.stringify(result.diagnostics));
});

test('natlang.json carries refinements settings, validated where the runtime is configured', () => {
  const base = { schema: 'natlang.package/v2', name: 'demo', version: '1.0.0', include: ['src'] };
  const manifest = parsePackageManifest({ ...base, refinements: { threshold: 0.6, predicates: { 'one   line': { threshold: 0.9 } } } });
  assert.equal(manifest.refinements.threshold, 0.6);
  assert.deepEqual(parseRefinementSettings(manifest.refinements).predicates, { 'one line': { threshold: 0.9 } });
  assert.throws(() => parseRefinementSettings({ threshold: 3 }), /threshold/);
  assert.throws(() => parseRefinementSettings({ colour: 1 }), /unknown refinements field/);
  assert.throws(() => parsePackageManifest({ ...base, refinements: [] }), /refinements must be an object/);
});
