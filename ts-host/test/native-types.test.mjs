import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType, resultType } from '../dist/index.js';

test('native type parser handles the complete structural grammar', () => {
  for (const [source, canonical] of [
    ['string', 'string'], ['Folder', 'Folder'], ['FileHandle', 'FileHandle'],
    ['{x: number, note?: string}', '{ x: number, note?: string }'],
    ['(string | number)[]', '(string | number)[]'],
    ['Record<string, {key: string, size: number}>', 'Record<string, { key: string, size: number }>'],
    ['(item: number) => boolean', '(item: number) => boolean'],
    ['Map<number, string>', 'Map<number, string>'], ['Fold<number, number>', 'Fold<number, number>'],
    ['Iterate<number>', 'Iterate<number>'], ['"open" | "closed"', '"open" | "closed"'],
  ]) assert.equal(formatType(parseType(source)), canonical);
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
  assert.equal(fits('() => number', 'number'), true);
  assert.equal(fits('Count', 'number'), true);
  assert.equal(formatType(resultType(parseType('Map<number, boolean>'))), 'boolean[]');
});

test('record types accept TS semicolon separators and nested aliases are not truncated', async () => {
  const { readTypeAliases } = await import('../dist/native/type-aliases.js');
  const aliases=readTypeAliases('// type Ignored = Bad;\nexport type A = { nested: { value: string; }; note?: "a;b"; };\ntype B = A[];');
  assert.deepEqual(Object.keys(aliases),['A','B']);
  assert.equal(formatType(parseType(aliases.A)), '{ nested: { value: string }, note?: "a;b" }');
  assert.throws(()=>readTypeAliases('type A = { value: string;'),/unterminated/);
  assert.throws(()=>readTypeAliases('type A = string; type A = number;'),/duplicate/);
});
