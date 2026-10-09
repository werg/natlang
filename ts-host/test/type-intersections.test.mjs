import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { typeScriptText } from '../dist/compiler/eval-check.js';

test('intersections of object types parse to one merged record', () => {
  assert.equal(formatType(parseType('{ a: number } & { b?: string }')), '{ a: number, b?: string }');
  assert.equal(formatType(parseType('{ a: number } & { a: number, c: boolean }')), '{ a: number, c: boolean }');
});

test('an alias in an intersection merges when it is resolved, and fits like the merged record', () => {
  const env = new TypeEnv({ Base: parseType('{ id: string }'), Dict: parseType('{ [k: string]: number }') });
  const type = parseType('Base & { note: string }');
  assert.equal(formatType(type), 'Base & { note: string }');
  assert.equal(formatType(env.resolve(type)), '{ id: string, note: string }');
  assert.equal(fitsType(parseType('{ id: string, note: string }'), type, env), true);
  assert.equal(fitsType(type, parseType('Base'), env), false, 'a record with extra fields does not fit the narrower record');
  assert.equal(fitsType(type, parseType('{ id: string, note: string }'), env), true);
  assert.throws(() => env.resolve(parseType('Dict & { a: number }')), /not all objects/);
});

test('string index signatures are dictionaries', () => {
  assert.equal(formatType(parseType('{ [k: string]: number }')), 'Record<string, number>');
  assert.deepEqual(parseType('{ [key: string]: { n: number }[] }').kind, 'dict');
  assert.throws(() => parseType('{ [k: number]: string }'), /string keys/);
  assert.throws(() => parseType('{ a: number, [k: string]: number }'), /bad|expected/);
  assert.throws(() => parseType('{ [k: string]: number, a: number }'), /stands alone/);
  assert.equal(fitsType(parseType('{ [k: string]: "a" }'), parseType('Record<string, string>')), true);
});

test('indexed access types resolve to the field type', () => {
  const env = new TypeEnv({ State: parseType('{ status: "open" | "closed", note?: string, tags: Record<string, number> }') });
  assert.equal(formatType(env.resolve(parseType("State['status']"))), '"open" | "closed"');
  assert.equal(formatType(env.resolve(parseType('State["note"]'))), 'string | null');
  assert.equal(formatType(env.resolve(parseType("State['tags']['x']"))), 'number');
  assert.equal(formatType(parseType("{ a: number }['a']")), 'number');
  assert.equal(fitsType(parseType('"open"'), parseType("State['status']"), env), true);
  assert.equal(fitsType(parseType('"stale"'), parseType("State['status']"), env), false);
  assert.throws(() => env.resolve(parseType("State['missing']")), /no field "missing"/);
  assert.equal(formatType(parseType("State['status']")), 'State["status"]');
});

test('the same types are valid TypeScript in declarations', () => {
  for (const text of ['A & { b: string }', '{ [k: string]: number }', "State['status']"])
    assert.equal(typeScriptText(text, new Set()), text);
});
