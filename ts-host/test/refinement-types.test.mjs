import assert from 'node:assert/strict';
import { test } from 'node:test';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { fitObligations, refinementChain, containsRefinement, TypeSyntaxError } from '../dist/native/types.js';
import { coerce } from '../dist/native/values.js';

const P = 'a reply that is polite';

test('Is<T, P> parses, formats canonically and normalizes whitespace', () => {
  const type = parseType(`Is<string,   "a reply\\n  that is polite" >`);
  assert.deepEqual(type, { kind: 'refined', base: { kind: 'prim', name: 'string' }, predicate: P });
  assert.equal(formatType(type), `Is<string, "${P}">`);
  assert.equal(formatType(parseType(formatType(type))), formatType(type));
  assert.equal(formatType(parseType(`{ quote: Is<string, 'copied verbatim'>, n: number }[]`)), '{ quote: Is<string, "copied verbatim">, n: number }[]');
  assert.equal(formatType(parseType('Is<"a" | "b", "short">')), 'Is<"a" | "b", "short">');
});

test('Is<T, P> needs a nonempty string literal predicate', () => {
  for (const bad of ['Is<string>', 'Is<string, "">', 'Is<string, "  ">', 'Is<string, 3>', 'Is<string, string>']) {
    assert.throws(() => parseType(bad), error => error instanceof TypeSyntaxError && /^refinement-predicate-invalid: /.test(error.message), bad);
  }
});

test('a refined type fits its base; its base fits it with an obligation; identical predicates fit without one', () => {
  const env = new TypeEnv();
  const refined = parseType(`Is<string, "${P}">`), other = parseType('Is<string, "another">'), plain = parseType('string');
  assert.ok(fitsType(refined, plain, env));
  assert.ok(fitsType(plain, refined, env));
  assert.ok(!fitsType(parseType('number'), refined, env));
  assert.ok(fitsType(refined, refined, env));
  assert.ok(fitsType(refined, other, env) === false, 'a different predicate is not established');
  assert.deepEqual(fitObligations(plain, refined, env), [{ path: 'value', predicate: P }]);
  assert.deepEqual(fitObligations(refined, refined, env), []);
  assert.equal(fitObligations(refined, other, env), null);
  assert.deepEqual(fitObligations(refined, plain, env), []);
  assert.deepEqual(fitObligations(parseType('"x"'), refined, env), [{ path: 'value', predicate: P }]);
});

test('nested refinements mean both predicates', () => {
  const env = new TypeEnv();
  const both = parseType('Is<Is<string, "a">, "b">'), a = parseType('Is<string, "a">'), b = parseType('Is<string, "b">');
  assert.deepEqual(refinementChain(both, env).predicates.sort(), ['a', 'b']);
  assert.ok(fitsType(both, a, env) && fitsType(both, b, env));
  assert.ok(!fitsType(a, both, env), 'a alone does not carry b');
  assert.deepEqual(fitObligations(parseType('string'), both, env).map(item => item.predicate).sort(), ['a', 'b']);
});

test('obligations are found inside records and lists; aliases are followed', () => {
  const env = new TypeEnv({ Subject: parseType('Is<string, "short">') });
  const target = parseType('{ subject: Subject, parts: Is<string, "polite">[], n: number }');
  const source = parseType('{ subject: string, parts: string[], n: number }');
  assert.deepEqual(fitObligations(source, target, env), [
    { path: 'value/subject', predicate: 'short' }, { path: 'value/parts/*', predicate: 'polite' }]);
  assert.ok(containsRefinement(target, env));
  assert.ok(!containsRefinement(parseType('{ a: string }[]'), env));
  assert.ok(fitsType(parseType('Is<string, "x">'), parseType('string | number'), env));
});

test('values coerce against the base of a refined type', () => {
  const env = new TypeEnv();
  assert.equal(coerce('hello', parseType('Is<string, "friendly">'), env), 'hello');
  assert.throws(() => coerce(3, parseType('Is<string, "friendly">'), env));
  assert.deepEqual(coerce({ q: 'x' }, parseType('{ q: Is<string, "verbatim"> }'), env), { q: 'x' });
});
