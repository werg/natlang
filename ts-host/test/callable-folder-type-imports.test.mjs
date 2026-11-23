// An inline `nl.with<T>` built inside a callable-folder TypeScript factory showed its parameter as `unknown` when the
// factory's types came from an `import type` of the package's types.ts (`../types.js`): type-only imports were erased
// before the module was checked, so the type was `any`. They now reach the checker at their package paths.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { scriptedModel } from './support/natlang.mjs';

const FACTORY = "{\n  return nl.with<boolean>({ spec })`Decide whether lot meets spec.`;\n}\n";
const RESULT = '(lot: Lot) => Promise<boolean>';

async function childSignature(checks, extra = {}) {
  const files = {
    'natlang.json': '{}',
    'types.ts': 'export type Lot = { id: string, record: string };\n',
    'release.nl': '---\nargs:\n  spec: string\n  lot: Lot\nreturns: boolean\n---\nUse checks.checkFor.\n',
    'release/checks.ts': checks,
    ...extra,
  };
  const fn = loadVirtualNatlang(files, 'release.nl');
  const { driver, openings } = scriptedModel(opening => opening.includes('Decide whether lot meets spec') ? 'return true;' :
    'return await checks.checkFor(spec)(lot);');
  assert.equal(await createNatlangRuntime({ model: driver, calls: false }).run(() => fn('ph 7', { id: 'L1', record: 'ph 7.0' })), true);
  const child = openings.find(opening => opening.includes('Decide whether lot meets spec'));
  return child.split('\n')[0];
}

test('nl.with in a callable-folder factory typed from the package types.ts (import type ../types.js) shows its parameter type', async () => {
  for (const specifier of ['../types.js', '../types', '../types.ts']) {
    const line = await childSignature(`import type { Lot } from '${specifier}';\nexport function checkFor(spec: string): ${RESULT} ${FACTORY}`);
    assert.match(line, /\(lot: Lot\): boolean$/, `${specifier}: ${line}`);
  }
});

test('type-only imports follow the imported file\'s own type imports, and other package modules', async () => {
  const line = await childSignature(`import type { Lot } from '../model/lot.js';\nexport function checkFor(spec: string): ${RESULT} ${FACTORY}`,
    { 'model/lot.ts': "import type { Record2 } from './record.js';\nexport type Lot = { id: string, record: Record2 };\n",
      'model/record.ts': 'export type Record2 = string;\n' });
  assert.doesNotMatch(line, /unknown/, line);
  assert.match(line, /\(lot: /);
});

test('a factory that returns a saved nl function takes its parameters from the declared result type', async () => {
  const line = await childSignature(`import type { Lot } from '../types.js';\nexport function checkFor(spec: string): ${RESULT} {\n` +
    '  const judge = nl.with<boolean>({ spec })`Decide whether lot meets spec.`;\n  return judge;\n}\n');
  assert.match(line, /\(lot: Lot\): boolean$/, line);
});
