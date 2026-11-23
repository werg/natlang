// Tests for WS1 L6: aliased `uses:` map form, relative & package specifiers, cycle checking admitting `recursive:`,
// and relative callable-folder TypeScript module imports.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

test('WS1 L6: aliased uses: map form and relative specifiers bind under alias keys', async () => {
  const files = {
    'natlang.json': JSON.stringify({
      exports: {
        './shared/rubric': './shared/rubric.ts',
      },
    }),
    'shared/rubric.ts': 'export default function rubric(text: string): string { return `RUBRIC:${text.trim()}`; }\n',
    'tools/inspect.ts': 'export default function inspect(n: number): number { return n * 10; }\n',
    'driver.nl': [
      '---',
      'args:',
      '  x: string',
      '  n: number',
      'returns: string',
      'uses:',
      '  formatRubric: ./shared/rubric',
      '  mulTen: tools/inspect',
      '---',
      'Call formatRubric(x) and mulTen(n) and join them with a colon.',
    ].join('\n'),
  };

  const fn = loadVirtualNatlang(files, 'driver.nl');
  const model = scriptedModel(() => 'return `${await formatRubric(x)}:${await mulTen(n)}`;');
  const out = await createNatlangRuntime({ model: model.driver }).run(() => fn('  hello ', 4));
  assert.equal(out, 'RUBRIC:hello:40');
});

test('WS1 L6: uses: refuses cycles with the cycle path, recursive or not; a recursive function calls itself by name', async () => {
  assert.throws(
    () =>
      loadVirtualNatlang(
        {
          'natlang.json': '{}',
          'a.nl': '---\nargs: { n: number }\nreturns: number\nuses: [b]\n---\nCall b(n).\n',
          'b.nl': '---\nargs: { n: number }\nreturns: number\nuses: [a]\n---\nCall a(n).\n',
        },
        'a.nl',
      ),
    /leads back to this function \(a\.nl -> b\.nl -> a\.nl\)/i,
  );
  // Admitting the cycle once cut the second function's own helpers off one level down (a -> b -> a had no `next`).
  assert.throws(
    () =>
      loadVirtualNatlang(
        {
          'natlang.json': '{}',
          'a.nl': '---\nargs: { n: number }\nreturns: number\nrecursive: structural\nuses: { next: ./b }\n---\nCall next(n).\n',
          'b.nl': '---\nargs: { n: number }\nreturns: number\nrecursive: structural\nuses: { back: ./a }\n---\nCall back(n).\n',
        },
        'a.nl',
      ),
    /leads back to this function .*a recursive function calls itself by its own name/i,
  );
  const countdown = loadVirtualNatlang({
    'natlang.json': '{}',
    'count.nl': '---\nargs: { n: number }\nreturns: number\nrecursive: structural\n---\nCount down to 0.\n',
  }, 'count.nl');
  const model = scriptedModel(() => 'return n === 0 ? 0 : (await count(n - 1)) + 1;');
  assert.equal(await createNatlangRuntime({ model: model.driver }).run(() => countdown(3)), 3);
});

test('WS1 L6: callable-folder TypeScript module imports package-relative helper modules within project root', async () => {
  const files = {
    'natlang.json': '{}\n',
    'shared/math.ts': 'export function double(n: number): number { return n * 2; }\n',
    'calc.nl': '---\nargs: { n: number }\nreturns: number\n---\nCall step(n).\n',
    'calc/step.ts': 'import { double } from "../shared/math.js";\nexport default function step(n: number): number { return double(n) + 1; }\n',
  };
  const fn = loadVirtualNatlang(files, 'calc.nl');
  const model = scriptedModel(() => 'return step(n);');
  const result = await createNatlangRuntime({ model: model.driver }).run(() => fn(20));
  assert.equal(result, 41);
});

test('WS1 L6: a .d.nl.ts refers to a used function by `import type` of its own declaration, and the project type-checks', async () => {
  const { checkProject } = await import('../dist/index.js');
  const { fileURLToPath } = await import('node:url');
  const runtimeModule = { url: new URL('../dist/index.js', import.meta.url).href, path: fileURLToPath(new URL('../dist/index.js', import.meta.url)),
    types: fileURLToPath(new URL('../dist/index.d.ts', import.meta.url)), specifiers: ['@natlang/node'] };
  const { mkdtempSync, mkdirSync, readFileSync, rmSync, writeFileSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const dir = mkdtempSync(join(tmpdir(), 'natlang-uses-decl-'));
  try {
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'uses-decl', version: '0.1.0' }));
    mkdirSync(join(dir, 'shared'));
    writeFileSync(join(dir, 'shared/rubric.nl'),
      '---\nargs:\n  answer: Answer\nreturns: Grade\ntypes:\n  Answer: "{ text: string }"\n  Grade: "\'pass\' | \'fail\'"\n---\nGrade answer.\n');
    writeFileSync(join(dir, 'grade.nl'),
      '---\nargs:\n  answer: string\nreturns: string\nuses:\n  rubric: ./shared/rubric\n---\nCall rubric({ text: answer }) and return its grade.\n');
    // A TypeScript caller reaches the used function through the declaration, with its own types.
    writeFileSync(join(dir, 'main.ts'), [
      "import grade from './grade.nl';",
      "export async function check(text: string): Promise<'pass' | 'fail'> { return grade.rubric({ text }); }",
      '',
    ].join('\n'));
    const checked = checkProject(dir, { write: true, runtimeModule });
    assert.equal(checked.ok, true, JSON.stringify(checked.diagnostics));
    const declaration = readFileSync(join(dir, 'grade.d.nl.ts'), 'utf8');
    assert.match(declaration, /^import type __uses_rubric_0 from "\.\/shared\/rubric\.nl";$/m);
    assert.match(declaration, /readonly rubric: typeof __uses_rubric_0/);
    assert.doesNotMatch(declaration, /type Grade =/, 'the used function\'s aliases stay in its own declaration');

    // A wrong use is a type error at the TypeScript call site.
    writeFileSync(join(dir, 'main.ts'), "import grade from './grade.nl';\nexport async function check(text: string) { return grade.rubric({ words: text }); }\n");
    assert.equal(checkProject(dir, { write: false, runtimeModule }).ok, false);

    // The opt-out inlines the used function as before.
    assert.equal(checkProject(dir, { write: true, inlineUses: true, runtimeModule }).ok, false);
    assert.match(readFileSync(join(dir, 'grade.d.nl.ts'), 'utf8'), /type Grade = /);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
