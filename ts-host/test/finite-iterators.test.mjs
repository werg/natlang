import assert from 'node:assert/strict';
import { test } from 'node:test';
import { mkdtempSync, mkdirSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { checkProject } from '../dist/index.js';
import { finite, finiteArrayIterator } from '../dist/runtime/lowered.js';

const RUNTIME = { url: new URL('../dist/index.js', import.meta.url).href,
  types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };

function project(files) {
  const root = mkdtempSync(join(tmpdir(), 'natlang-iter-'));
  for (const [path, text] of Object.entries({
    'package.json': JSON.stringify({ name: 'fixture', private: true, type: 'module' }),
    'tsconfig.json': JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext',
      strict: true, skipLibCheck: true, rootDir: 'src', outDir: 'dist' }, include: ['src/**/*.ts'] }),
    ...files })) {
    mkdirSync(join(root, path, '..'), { recursive: true });
    writeFileSync(join(root, path), text);
  }
  return root;
}

const FOLDER = {
  'src/app.ts': 'export const x = 1;\n',
  'src/natlang.d/note.nl': '---\nargs:\n  text: string\nreturns: string\n---\nSummarize text.\n',
};

test('check accepts entries, keys, values and matchAll loops in callable-folder TypeScript', () => {
  const root = project({ ...FOLDER, 'src/natlang.d/scan.ts':
    'export function scan(items: string[], text: string, table: Map<string, number>): number {\n' +
    '  let n = 0;\n' +
    '  for (const [i, item] of items.entries()) n += i + item.length;\n' +
    '  for (const key of table.keys()) n += key.length;\n' +
    '  for (const m of text.matchAll(/a/g)) n += m.index ?? 0;\n' +
    '  return n;\n}\n' });
  const result = checkProject(root, { runtimeModule: RUNTIME });
  assert.deepEqual(result.diagnostics, [], JSON.stringify(result.diagnostics));
});

test('check reports a loop source the run would refuse, with its location', () => {
  const root = project({ ...FOLDER, 'src/natlang.d/scan.ts':
    'export function scan(o: { a: number }): number {\n  let n = 0;\n  for (const x of o as any as Iterable<number>) n += x;\n  for (const y of new Set([1]).entries().map(e => e)) n += y[0];\n  return n;\n}\n' });
  const result = checkProject(root, { runtimeModule: RUNTIME });
  assert.equal(result.ok, false);
  assert.ok(result.diagnostics.some(item => /for \.\.\. of/.test(item.message)), JSON.stringify(result.diagnostics));
});

test('a callable-folder function loops over entries() and matchAll() when a turn runs', async () => {
  const { session: open, ts } = await import('./support/natlang.mjs');
  const { session } = open({ type: '(items: string[]) => number', instructions: 'Sum.', args: { items: ['ab', 'c'] },
    codebase: { scan: ts('scan',
      'export default function scan(items: string[]): number {\n  let n = 0;\n' +
      '  for (const [i, item] of items.entries()) n += i + item.length;\n' +
      '  for (const m of "a1b22".matchAll(/\\d+/g)) n += m[0].length;\n  return n;\n}\n') } });
  const outcome = await session.applyAsync('eval', { code: 'scan(items)' });
  assert.equal(outcome.kind, 'ok', outcome.text);
  assert.equal(outcome.value, 2 + 1 + 1 + 1 + 2);
});

test('a counted loop may join its bound to an early-exit condition or to a second counter comparison', async () => {
  const { checkConstrainedSource } = await import('../dist/compiler/policy.js');
  const ts = (await import('typescript')).default;
  const verdict = loop => checkConstrainedSource(ts.createSourceFile('x.ts',
    `let found = false; const n = 3, m = 2;\n${loop}`, ts.ScriptTarget.ES2022, true)).map(item => item.message);
  assert.deepEqual(verdict('for (let i = 0; i < n && !found; i++) {}'), []);
  assert.deepEqual(verdict('for (let i = 0; i < n && i < m; i++) {}'), []);
  assert.match(verdict('for (let i = 0; i < n || found; i++) {}')[0], /one bound.*optionally joined by &&.*`i < n && !found`/);
});

test('runtime: matchAll and Map/Set views iterate finitely', () => {
  assert.deepEqual([...finite('a1b22'.matchAll(/\d+/g))].map(m => m[0]), ['1', '22']);
  const map = new Map([['a', 1], ['b', 2]]);
  const keys = [];
  for (const key of finiteArrayIterator(map, 'keys')) { keys.push(key); map.set(key + 'x', 0); }
  assert.deepEqual(keys, ['a', 'b']);
  assert.deepEqual([...finiteArrayIterator(['x', 'y'], 'entries')], [[0, 'x'], [1, 'y']]);
  assert.throws(() => finite((function* () { yield 1; })()), /for/);
});
