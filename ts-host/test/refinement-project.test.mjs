import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { buildProject, checkProject, parsePackageManifest, parseRefinementSettings, __natlang } from '../dist/index.js';
import { refinements as crispTable } from '../dist/native/types.js';
import { pathToFileURL } from 'node:url';

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
  assert.deepEqual(result.refinedSlots, [{ source: 'src/reply.nl', function: 'reply', slot: 'return', predicate: 'a reply that is polite' }]);
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

test('refine<Is<T, P>>(value) and assume<R>(value) are lowered to the predicate form', () => {
  const root = project({ ...FILES, 'src/app.ts': `import { refine, assume } from '@natlang/node';
import type { Is } from '@natlang/node';
type Polite = Is<string, "a reply that is polite">;
export async function run(text: string): Promise<string> {
  const checked: Polite = await refine<Polite>(text);
  const taken = assume<Is<Is<string, "short">, "lowercase">>(text);
  return checked + taken;
}
` });
  const result = buildProject({ project: root, runtimeModule: RUNTIME });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics, null, 1));
  const emitted = readFileSync(join(root, 'dist/app.js'), 'utf8');
  assert.match(emitted, /refine\(text, "a reply that is polite"\)/);
  assert.match(emitted, /assume\(text, "short; and lowercase"\)/);
});

test('a refined alias in types.ts is the same type as the one in the generated declaration, with no import of Is', () => {
  const root = project({ ...FILES,
    'src/reply.nl': '---\nargs:\n  complaint: string\nreturns: Polite\n---\nAnswer the complaint.\n',
    'src/types.ts': 'export type Polite = Is<string, "a reply that is polite">;\nexport type Pair = { first: Polite, second: Is<number, "a whole number"> };\n',
    'src/app.ts': `import reply from './reply.nl';
import type { Polite, Pair } from './types.js';
export async function run(complaint: string): Promise<Pair['first']> {
  const answer: Polite = await reply(complaint);
  return answer;
}
` });
  const result = buildProject({ project: root, runtimeModule: RUNTIME });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics, null, 1));
});

test('the compiled .nl modules register the nearest refinements.ts, whatever hosts them', async () => {
  const predicate = 'a reply that is polite';
  const root = project({ ...FILES,
    'src/refinements.ts': `export const refinements = { "a  reply that is polite": (value: unknown) => typeof value === "string" && !/fault/.test(value) };\n`,
    'src/app.ts': "import reply from './reply.nl';\nexport default reply;\n" });
  const result = buildProject({ project: root, runtimeModule: RUNTIME });
  assert.equal(result.ok, true, JSON.stringify(result.diagnostics, null, 1));
  assert.match(readFileSync(join(root, 'dist/reply.nl.js'), 'utf8'), /registerCrisp\(__crisp, "\.\/refinements\.js"\)/);
  assert.equal(crispTable[predicate], undefined);
  await import(pathToFileURL(join(root, 'dist/reply.nl.js')).href);
  assert.equal(crispTable[predicate]('Sorry for the delay.'), true);
  assert.equal(crispTable[predicate]('your fault'), false);
  assert.throws(() => __natlang.registerCrisp({ [predicate]: value => value === 'something else' }, 'other/refinements.js'), /already registered with different code/);
  __natlang.registerCrisp({ [predicate]: crispTable[predicate] }, 'again');
  assert.throws(() => __natlang.registerCrisp({ p: 3 }, 'bad.js'), /must be a function/);
  assert.throws(() => __natlang.registerCrisp(undefined, 'bad.js'), /must export `refinements`/);
  delete crispTable[predicate];
});
