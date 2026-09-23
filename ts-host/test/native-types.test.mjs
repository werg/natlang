import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { checkHost } from '../dist/native/types.js';

test('native type parser handles the complete structural grammar', () => {
  for (const [source, canonical] of [
    ['string', 'string'], ['Folder', 'Folder'], ['FileHandle', 'FileHandle'],
    ['{x: number, note?: string}', '{ x: number, note?: string }'],
    ['(string | number)[]', '(string | number)[]'],
    ['Record<string, {key: string, size: number}>', 'Record<string, { key: string, size: number }>'],
    ['(item: number) => boolean', '(item: number) => boolean'],
    ['"open" | "closed"', '"open" | "closed"'],
    ['Map<string, number>', 'Map<string, number>'], ['Date', 'Date'],
    ['Live<"Ledger", "class", "Ledger">', 'Ledger'],
  ]) assert.equal(formatType(parseType(source)), canonical);
  assert.deepEqual(parseType('Map<string, number>').contract, { kind: 'tag', tag: 'Map' });
  assert.deepEqual(parseType('Live<"Ledger", "class", "Ledger">').contract, { kind: 'class', name: 'Ledger' });
  for (const removed of ['Fold<number, number>', 'Iterate<number>']) assert.throws(() => parseType(removed));
  assert.throws(() => parseType('{x: number, x: string}'), /duplicate field/);
  assert.throws(() => parseType('Lambda<number, string>'));
});

test('native fit mirrors promise, record, union and lambda variance', () => {
  const env = new TypeEnv({ Count: parseType('number') });
  const fits = (a, b) => fitsType(parseType(a), parseType(b), env);
  assert.equal(fits('"ok"', 'string'), true);
  assert.equal(fits('number', 'number | string'), true);
  assert.equal(fits('number | string', 'number'), false);
  assert.equal(fits('{x: number}', '{x: number, y?: string}'), true);
  assert.equal(fits('{x: number, y: string}', '{x: number}'), false);
  assert.equal(fits('(x: number) => number', '(x: number) => number'), true);
  assert.equal(fits('() => number', 'number'), false);
  assert.equal(fits('Count', 'number'), true);
  assert.equal(fits('Date', 'Date'), true);
  assert.equal(fits('Date', 'Map<string, number>'), false);
});

test('record types accept TS semicolon separators and nested aliases are not truncated', async () => {
  const { readTypeAliases } = await import('../dist/native/type-aliases.js');
  const aliases=readTypeAliases('// type Ignored = Bad;\nexport type A = { nested: { value: string; }; note?: "a;b"; };\ntype B = A[];');
  assert.deepEqual(Object.keys(aliases),['A','B']);
  assert.equal(formatType(parseType(aliases.A)), '{ nested: { value: string }, note?: "a;b" }');
  assert.throws(()=>readTypeAliases('type A = { value: string;'),/unterminated/);
  assert.throws(()=>readTypeAliases('type A = string; type A = number;'),/duplicate/);
});

test('host contracts check built-ins by tag across realms, classes by constructor, and shapes by members', async () => {
  const { runInNewContext } = await import('node:vm');
  const foreignDate = runInNewContext('new Date(0)');
  assert.equal(foreignDate instanceof Date, false);
  assert.equal(checkHost(foreignDate, { kind: 'tag', tag: 'Date' }), true);
  class Ledger { total = 0; }
  assert.equal(checkHost(new Ledger(), { kind: 'class', name: 'Ledger' }), true);
  assert.equal(checkHost(new Ledger(), { kind: 'class', name: 'Ledger' }, new Map([['Ledger', class Other {}]])), false);
  assert.equal(checkHost({ add() {}, total: 1 }, { kind: 'shape', members: ['add', 'total'] }), true);
  assert.equal(checkHost({ add() {} }, { kind: 'shape', members: ['add', 'total'] }), false);
  assert.equal(checkHost(() => 1, { kind: 'function' }), true);
});
