import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, defineNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('a named function runs on the model its frontmatter names, and on the default model when none has that name', async () => {
  const judge = defineNatlang('---\nargs:\n  text: string\nreturns: string\nmodel: small\n---\nJudge text.\n', { name: 'judge' });
  const write = defineNatlang('---\nargs:\n  text: string\nreturns: string\n---\nWrite about text.\n', { name: 'write' });
  const big = scriptedModel(() => 'return "big";'), small = scriptedModel(() => 'return "small";');
  const runtime = createNatlangRuntime({ model: big.driver, models: { small: small.driver } });
  assert.equal(await runtime.run(() => judge('x')), 'small');
  assert.equal(await runtime.run(() => write('x')), 'big');
  assert.equal(await createNatlangRuntime({ model: big.driver }).run(() => judge('x')), 'big');
  assert.throws(() => defineNatlang('---\nreturns: string\nmodel: two words\n---\nx\n'), /model must name one of the runtime's models/);
});

test('uses: a function calls the package items it lists, by base name, besides its companion folder', async () => {
  const { mkdtempSync, mkdirSync, writeFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');
  const { loadNatlang } = await import('../dist/index.js');
  const root = mkdtempSync(join(tmpdir(), 'uses-'));
  const write = (path, text) => { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); };
  write('natlang.json', '{}');
  write('shared/double.ts', 'export default function double(n: number): number { return 2 * n; }\n');
  write('shared/twice.ts', "import double from './twice/inner.js';\nexport default function twice(n: number): number { return double(n); }\n");
  write('shared/twice/inner.ts', 'export default function inner(n: number): number { return 2 * n; }\n');
  write('main.nl', '---\nargs:\n  n: number\nreturns: number\n---\nAdd one to n with addOne.\n');
  write('main/addOne.nl', '---\nargs:\n  n: number\nreturns: number\nuses: [shared/double, shared/twice]\n---\nDouble n with double, then add one.\n');
  const model = scriptedModel(opening => opening.includes('Double n with double') ? 'return double(n) + twice(n) - double(n) + 1;' :
    'return await addOne(n);');
  const runtime = createNatlangRuntime({ model: model.driver });
  assert.equal(await runtime.run(() => loadNatlang(join(root, 'main.nl'))(5)), 11);
  write('bad.nl', '---\nreturns: number\nuses: [shared/missing]\n---\nx\n');
  assert.throws(() => loadNatlang(join(root, 'bad.nl')), /uses shared\/missing, but the package has no shared\/missing.nl or shared\/missing.ts/);
  write('loop/a.nl', '---\nreturns: number\nuses: [loop/b]\n---\nx\n');
  write('loop/b.nl', '---\nreturns: number\nuses: [loop/a]\n---\nx\n');
  assert.throws(() => loadNatlang(join(root, 'loop/a.nl')), /cannot reach itself/);
});

test('uses: a project whose functions share a used item checks, the item described once', async () => {
  const { mkdtempSync, mkdirSync, writeFileSync } = await import('node:fs');
  const { join } = await import('node:path');
  const { tmpdir } = await import('node:os');
  const { checkProject } = await import('../dist/index.js');
  const root = mkdtempSync(join(tmpdir(), 'uses-check-'));
  const write = (path, text) => { mkdirSync(join(root, path, '..'), { recursive: true }); writeFileSync(join(root, path), text); };
  write('natlang.json', JSON.stringify({ schema: 'natlang.package/v2', name: 'uses-check', version: '0.0.0', include: ['.'], targets: {} }));
  write('tsconfig.json', JSON.stringify({ compilerOptions: { target: 'ES2022', module: 'NodeNext', moduleResolution: 'NodeNext', strict: true, skipLibCheck: true } }));
  write('shared/twice.nl', '---\nargs:\n  n: number\nreturns: number\n---\nDouble n.\n');
  write('shared/half.ts', 'export default function half(n: number): number { return n / 2; }\n');
  write('a.nl', '---\nargs:\n  n: number\nreturns: number\nuses: [shared/twice, shared/half]\n---\nUse twice.\n');
  write('b.nl', '---\nargs:\n  n: number\nreturns: number\nuses: [shared/twice, shared/half]\n---\nUse twice too.\n');
  write('main.ts', "import a from './a.nl';\nimport b from './b.nl';\nexport const both = (n: number) => Promise.all([a(n), b(n)]);\n");
  const result = checkProject(root);
  assert.ok(result.ok, JSON.stringify(result.diagnostics));
});
