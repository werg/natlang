import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType, resultType } from '../dist/index.js';

test('native type parser handles the complete structural grammar', () => {
  for (const [source, canonical] of [
    ['Text', 'Text'], ['Folder', 'Folder'], ['FileHandle', 'FileHandle'],
    ['{x: Num, note?: Text}', '{ x: Num, note?: Text }'],
    ['(Text | Num)[]', '(Text | Num)[]'],
    ['Dict<{key: Text, size: Num}>', 'Dict<{ key: Text, size: Num }>'],
    ['Lambda<{item: Num}, Bool>', 'Lambda<{ item: Num }, Bool>'],
    ['Map<Num, Text>', 'Map<Num, Text>'], ['Fold<Num, Num>', 'Fold<Num, Num>'],
    ['Iterate<Num>', 'Iterate<Num>'], ['"open" | "closed"', '"open" | "closed"'],
  ]) assert.equal(formatType(parseType(source)), canonical);
  assert.throws(() => parseType('{x: Num, x: Text}'), /duplicate field/);
  assert.throws(() => parseType('Lambda<Num, Text>'), /params must be a record/);
});

test('native fit mirrors promise, record, union and lambda variance', () => {
  const env = new TypeEnv({ Count: parseType('Num') });
  const fits = (a, b) => fitsType(parseType(a), parseType(b), env);
  assert.equal(fits('"ok"', 'Text'), true);
  assert.equal(fits('Num', 'Num | Text'), true);
  assert.equal(fits('Num | Text', 'Num'), false);
  assert.equal(fits('{x: Num}', '{x: Num, y?: Text}'), true);
  assert.equal(fits('{x: Num, y: Text}', '{x: Num}'), false);
  assert.equal(fits('Lambda<{x: Num}, Num>', 'Lambda<{x: Num}, Num>'), true);
  assert.equal(fits('Lambda<{}, Num>', 'Num'), true);
  assert.equal(fits('Count', 'Num'), true);
  assert.equal(formatType(resultType(parseType('Map<Num, Bool>'))), 'Bool[]');
});

test('record types accept TS semicolon separators and nested aliases are not truncated', async () => {
  const { readTypeAliases } = await import('../dist/native/type-aliases.js');
  const aliases=readTypeAliases('// type Ignored = Bad;\nexport type A = { nested: { value: Text; }; note?: "a;b"; };\ntype B = A[];');
  assert.deepEqual(Object.keys(aliases),['A','B']);
  assert.equal(formatType(parseType(aliases.A)), '{ nested: { value: Text }, note?: "a;b" }');
  assert.throws(()=>readTypeAliases('type A = { value: Text;'),/unterminated/);
  assert.throws(()=>readTypeAliases('type A = Text; type A = Num;'),/duplicate/);
});
