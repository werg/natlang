import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { parseTypeCached } from '../dist/native/types.js';
import { checkHost } from '../dist/native/types.js';

test('native type parser handles the complete structural grammar', () => {
  for (const [source, canonical] of [
    ['string', 'string'], ['Folder', 'Folder'], ['FileHandle', 'FileHandle'],
    ['{x: number, note?: string}', '{ x: number, note?: string }'],
    ['(string | number)[]', '(string | number)[]'],
    ['Record<string, {key: string, size: number}>', 'Record<string, { key: string, size: number }>'],
    ['(item: number) => boolean', '(item: number) => boolean'],
    ['"open" | "closed"', '"open" | "closed"'], ["'low' | 'high'", '"low" | "high"'], ["{ note: 'it\\'s' }", '{ note: "it\'s" }'],
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
  assert.throws(()=>readTypeAliases('type A = { value: string;'),/type alias syntax: '\}' expected/);
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

test("a program's own type shadows a platform global of the same name", async () => {
  const { coerce } = await import('../dist/native/values.js');
  const request = parseType('{ id: string, text: string }');
  const env = new TypeEnv({ Request: request, Item: parseType('{ request: Request }') });
  assert.deepEqual(env.resolve(parseType('Request')), request);
  assert.deepEqual(coerce({ id: 'R1', text: 'hi' }, parseType('Request'), env), { id: 'R1', text: 'hi' });
  assert.ok(fitsType(parseType('{ id: string, text: string }'), parseType('Request'), env));
  assert.deepEqual(coerce({ request: { id: 'R1', text: 'hi' } }, parseType('Item'), env), { request: { id: 'R1', text: 'hi' } });
  // Without a declaration the name still means the platform's Request; with type arguments it is never an alias.
  assert.equal(new TypeEnv().resolve(parseType('Request')).kind, 'host');
  assert.equal(new TypeEnv({ Map: request }).resolve(parseType('Map<string, number>')).kind, 'host');
  assert.throws(() => new TypeEnv({ Request: parseType('Request') }).resolve(parseType('Request')), /only in terms of itself/);
});

test('internal type parsing caches immutable syntax without changing the public parser', () => {
  const source = '{ item: { value: Item }[]; handle: Live<"Ledger", "shape", "id"> }';
  const internal = parseTypeCached(source);
  const copy = parseType(source);
  assert.notEqual(copy, internal);
  assert.deepEqual(copy, internal);
  copy.fields[0].type.element.fields[0].type.name = 'Changed';
  copy.fields[1].type.contract.members.push('other');
  assert.equal(formatType(parseType(source)), formatType(internal));
  assert.deepEqual(internal.fields[1].type.contract.members, ['id']);
  if (process.env.NATLANG_TYPE_PARSE_CACHE === 'off') {
    assert.notEqual(parseTypeCached(source), internal);
  } else {
    assert.equal(parseTypeCached(source), internal);
    assert.throws(() => { internal.fields[0].type.element.fields[0].type.name = 'Changed'; }, TypeError);
    assert.throws(() => internal.fields[1].type.contract.members.push('other'), TypeError);
  }
  const first = new TypeEnv({ Item: parseTypeCached('string') });
  const second = new TypeEnv({ Item: parseTypeCached('number') });
  const item = internal.fields[0].type.element.fields[0].type;
  assert.equal(formatType(first.resolve(item)), 'string');
  assert.equal(formatType(second.resolve(item)), 'number');
});

test('the parse cache evicts older declarations and skips oversized sources', () => {
  const before = parseTypeCached('{ cacheAnchor: string }');
  for (let index = 0; index < 4096; index++) parseTypeCached(JSON.stringify(`entry ${index}`));
  assert.notEqual(parseTypeCached('{ cacheAnchor: string }'), before);
  const byteAnchor = parseTypeCached('{ cacheByteAnchor: number }');
  for (let index = 0; index < 64; index++) parseTypeCached(JSON.stringify(`long ${index}: ${'x'.repeat(9 * 1024)}`));
  assert.notEqual(parseTypeCached('{ cacheByteAnchor: number }'), byteAnchor);
  const source = JSON.stringify('x'.repeat(17 * 1024));
  const large = parseTypeCached(source);
  assert.equal(formatType(large), source);
  assert.notEqual(parseTypeCached(source), large);
  assert.throws(() => parseTypeCached('{ missing: }'), /unexpected/);
});
