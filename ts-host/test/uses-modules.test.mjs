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

test('WS1 L6: uses: refuses unguarded cycles with the cycle path, and admits cycles when recursive: structural is declared', () => {
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

  const recursiveLoaded = loadVirtualNatlang(
    {
      'natlang.json': '{}',
      'a.nl': '---\nargs: { n: number }\nreturns: number\nrecursive: structural\nuses: { next: ./b }\n---\nCall next(n).\n',
      'b.nl': '---\nargs: { n: number }\nreturns: number\nrecursive: structural\nuses: { back: ./a }\n---\nCall back(n).\n',
    },
    'a.nl',
  );
  assert.equal(typeof recursiveLoaded, 'function');
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
