import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { TypeEnv, parseType, formatType, fitsType } from '../dist/index.js';
import { containsRefinement, fitObligations, TypeSyntaxError } from '../dist/native/types.js';
import { coerce, problems, MISSING } from '../dist/native/values.js';
import { schemaOf } from '../dist/native/agent.js';
import { genericResult, representationType } from '../dist/native/representation.js';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { describeTarget } from '../dist/compiler/targets.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { typeScriptText } from '../dist/compiler/eval-check.js';

function describeFromSource(typeExpr, declarations = '') {
  const source = `${declarations}\ndeclare const __target: ${typeExpr};\n`;
  const program = createVirtualProgram({ '/scope/main.ts': source });
  const file = program.getSourceFile('/scope/main.ts');
  const checker = program.getTypeChecker();
  let targetNode;
  const visit = node => {
    if (ts.isVariableDeclaration(node) && ts.isIdentifier(node.name) && node.name.text === '__target') targetNode = node;
    ts.forEachChild(node, visit);
  };
  visit(file);
  return describeTarget(program, checker, checker.getTypeAtLocation(targetNode), { location: targetNode });
}

test('tuple types parse, format, and reject required-after-optional', () => {
  const pair = parseType('[string, number]');
  assert.deepEqual(pair, {
    kind: 'tuple',
    elements: [{ kind: 'prim', name: 'string' }, { kind: 'prim', name: 'number' }],
  });
  assert.equal(formatType(pair), '[string, number]');

  assert.deepEqual(parseType('[]'), { kind: 'tuple', elements: [] });
  assert.deepEqual(parseType('[ ]'), { kind: 'tuple', elements: [] });
  assert.equal(formatType(parseType('[]')), '[]');

  assert.equal(formatType(parseType('[string, number,]')), '[string, number]');
  assert.equal(formatType(parseType('[[string, number], { ok: boolean }][]')), '[[string, number], { ok: boolean }][]');

  const opt = parseType('[string, number?]');
  assert.equal(opt.kind, 'tuple');
  assert.equal(opt.minLength, 1);
  assert.equal(formatType(opt), '[string, number?]');

  const optUnion = parseType('[string, (number | null)?]');
  assert.equal(formatType(optUnion), '[string, (number | null)?]');
  assert.deepEqual(parseType(formatType(optUnion)), optUnion);

  const labeled = parseType('[first: string, second?: number]');
  assert.equal(formatType(labeled), '[string, number?]');

  assert.throws(() => parseType('[string?, number]'), TypeSyntaxError);
  assert.throws(() => parseType('[string, number'), TypeSyntaxError);
});

test('tuple fitsType handles exact, literal, optional, and list subtyping', () => {
  const env = new TypeEnv();
  const fits = (a, b) => fitsType(parseType(a), parseType(b), env);
  assert.equal(fits('[string, number]', '[string, number]'), true);
  assert.equal(fits('["a", 1]', '[string, number]'), true);
  assert.equal(fits('[string, number]', '[number, string]'), false);
  assert.equal(fits('[string]', '[string, number?]'), true);
  assert.equal(fits('[string, number]', '[string, number?]'), true);
  assert.equal(fits('[string, number?]', '[string, number]'), false);
  assert.equal(fits('[]', '[string?]'), true);
  assert.equal(fits('[string, number]', '(string | number)[]'), true);
  assert.equal(fits('[]', 'string[]'), true);
  assert.equal(fits('string[]', '[string, string]'), false);
  assert.deepEqual(
    fitObligations(parseType('[string, number]'), parseType('[Is<string, "nonempty">, number]'), env),
    [{ path: 'value/0', predicate: 'nonempty' }],
  );
});

test('tuple coerce and problems validate lengths, element types, and optional tails', () => {
  const env = new TypeEnv();
  const pair = parseType('[string, number]');
  assert.deepEqual(coerce(['hi', 42], pair, env), ['hi', 42]);
  assert.throws(() => coerce(['hi'], pair, env), /2 elements/);
  assert.throws(() => coerce(['hi', 42, true], pair, env), /2 elements/);
  assert.throws(() => coerce(['hi', 'bad'], pair, env), /value\/1: type-mismatch/);
  assert.throws(() => coerce('not-tuple', pair, env), /type-mismatch/);

  const empty = parseType('[]');
  assert.deepEqual(coerce([], empty, env), []);
  assert.throws(() => coerce([1], empty, env), /0 elements/);

  const opt = parseType('[string, number?]');
  assert.deepEqual(coerce(['hi'], opt, env), ['hi']);
  assert.deepEqual(coerce(['hi', 7], opt, env), ['hi', 7]);
  assert.deepEqual(coerce(['hi', undefined], opt, env), ['hi']);
  assert.throws(() => coerce([], opt, env), /1\.\.2 elements/);
  assert.throws(() => coerce(['hi', 'no'], opt, env), /value\/1: type-mismatch/);
  assert.throws(() => coerce(['hi', 7, 8], opt, env), /1\.\.2 elements/);

  assert.deepEqual(problems(['hi', MISSING], pair, env, 'result').holes, [
    { path: 'result/1', code: 'hole', expected: 'number' },
  ]);
  assert.deepEqual(problems(['hi'], pair, env, 'result').holes, [
    { path: 'result/1', code: 'hole', expected: 'number' },
  ]);
});

test('schemaOf emits prefixItems, minItems, and maxItems for tuples', () => {
  const env = new TypeEnv();
  assert.deepEqual(schemaOf(parseType('[string, number]'), env), {
    type: 'array',
    prefixItems: [{ type: 'string' }, { type: 'number' }],
    minItems: 2,
    maxItems: 2,
  });
  assert.deepEqual(schemaOf(parseType('[string, number?]'), env), {
    type: 'array',
    prefixItems: [{ type: 'string' }, { type: 'number' }],
    minItems: 1,
    maxItems: 2,
  });
  assert.deepEqual(schemaOf(parseType('[]'), env), {
    type: 'array',
    prefixItems: [],
    minItems: 0,
    maxItems: 0,
  });
});

test('compiler/targets.ts lowers tuples faithfully and preserves Neuralese<[A, B]> for zip', () => {
  assert.equal(describeFromSource('[string, number]').natlang, '[string, number]');
  assert.equal(describeFromSource('[]').natlang, '[]');
  assert.equal(describeFromSource('[string, number?]').natlang, '[string, number?]');
  assert.equal(describeFromSource('[string, (number | null)?]').natlang, '[string, (null | number)?]');
  assert.equal(describeFromSource('[[string, number], boolean]').natlang, '[[string, number], boolean]');
  assert.equal(describeFromSource('Neuralese<[string, number]>').natlang, 'Neuralese<[string, number]>');

  // Round-trip inline nl returning zip-shaped Neuralese<[A, B]>
  const source = `
    type Entry = { id: string };
    async function run(a: Neuralese<Entry>, b: Neuralese<number>) {
      const zipped = await nl<Neuralese<[Entry, number]>>\`Pair a and b.\`(a, b);
      return zipped;
    }
  `;
  const program = createVirtualProgram({ '/scope/main.ts': source });
  const { plans, diagnostics } = analyzeInlineLambdas(program, [program.getSourceFile('/scope/main.ts')]);
  assert.deepEqual(diagnostics, []);
  assert.equal(plans[0].returns.natlang, 'Neuralese<[Entry, number]>');
  const parsed = parseType(plans[0].returns.natlang);
  const env = new TypeEnv({ Entry: parseType(plans[0].returns.aliases.Entry) });
  env.checkNames(parsed);
  assert.equal(formatType(parsed), 'Neuralese<[Entry, number]>');
});

test('typeScriptText and representation.ts handle tuples and utility types', () => {
  const known = new Set(['User']);
  assert.equal(typeScriptText('[string, number]', known), '[string, number]');
  assert.equal(typeScriptText('[(x: string) => number, boolean]', known), '[(x: string) => Promise<number>, boolean]');
  assert.equal(typeScriptText('Partial<User>', known), 'Partial<User>');

  const gen = genericResult({ R: '[string, number] | Neuralese<[string, number]>' }, 'R');
  assert.deepEqual(gen, {
    name: 'R',
    constraint: '[string, number] | Neuralese<[string, number]>',
    crisp: '[string, number]',
    neuralese: 'Neuralese<[string, number]>',
    dialect: 'DefaultDialect',
  });
  assert.equal(representationType('[string, number]', { kind: 'neuralese' }), 'Neuralese<[string, number]>');
});

test('Partial, Required, Pick, and Omit resolve over records and record unions', () => {
  const env = new TypeEnv({
    User: parseType('{ id: string, name: string, note?: string }'),
    Keys: parseType('"id" | "name"'),
    Event: parseType('{ kind: "a", id: string, x: number } | { kind: "b", id: string, y: string }'),
  });

  const partial = env.resolve(parseType('Partial<User>'));
  assert.equal(formatType(partial), '{ id?: string, name?: string, note?: string }');
  assert.deepEqual(coerce({ name: 'Ada' }, parseType('Partial<User>'), env), { name: 'Ada' });
  assert.deepEqual(coerce({}, parseType('Partial<User>'), env), {});
  assert.equal(fitsType(parseType('User'), parseType('Partial<User>'), env), true);
  assert.equal(fitsType(parseType('Partial<User>'), parseType('User'), env), false);

  const required = env.resolve(parseType('Required<User>'));
  assert.equal(formatType(required), '{ id: string, name: string, note: string }');
  assert.deepEqual(problems({ id: '1', name: 'Ada' }, parseType('Required<User>'), env, 'user').holes, [
    { path: 'user/note', code: 'hole', expected: 'note' },
  ]);
  assert.throws(() => coerce({ id: '1', name: 'Ada' }, parseType('Required<User>'), env, 'user', { preserveRecordExtras: true }), /missing/);
  assert.equal(fitsType(parseType('Required<User>'), parseType('User'), env), true);
  assert.equal(fitsType(parseType('User'), parseType('Required<User>'), env), false);

  const picked = env.resolve(parseType('Pick<User, Keys>'));
  assert.equal(formatType(picked), '{ id: string, name: string }');
  assert.deepEqual(coerce({ id: '1', name: 'Ada' }, parseType('Pick<User, Keys>'), env), { id: '1', name: 'Ada' });
  assert.throws(() => coerce({ id: '1', name: 'Ada', note: 'x' }, parseType('Pick<User, Keys>'), env), /unknown-field/);
  assert.throws(() => env.resolve(parseType('Pick<User, "missing">')), TypeSyntaxError);

  const omitted = env.resolve(parseType('Omit<User, "note">'));
  assert.equal(formatType(omitted), '{ id: string, name: string }');
  assert.equal(formatType(env.resolve(parseType('Partial<Omit<User, "id">>'))), '{ name?: string, note?: string }');

  // Union of records
  assert.equal(
    formatType(env.resolve(parseType('Partial<Event>'))),
    '{ kind?: "a", id?: string, x?: number } | { kind?: "b", id?: string, y?: string }',
  );
  assert.equal(
    formatType(env.resolve(parseType('Pick<Event, "kind" | "id">'))),
    '{ kind: "a", id: string } | { kind: "b", id: string }',
  );
  assert.equal(
    formatType(env.resolve(parseType('Omit<Event, "id">'))),
    '{ kind: "a", x: number } | { kind: "b", y: string }',
  );

  // Refinements stripped by Omit are not reported as required
  assert.equal(containsRefinement(parseType('Omit<{ a: string, b: Is<string, "short"> }, "b">'), env), false);
  assert.equal(containsRefinement(parseType('Pick<{ a: string, b: Is<string, "short"> }, "b">'), env), true);

  // Non-record target errors cleanly
  assert.throws(() => env.resolve(parseType('Partial<string>')), TypeSyntaxError);

  // targets.ts lowers TS utility types cleanly
  assert.equal(describeFromSource('Partial<User>', 'type User = { id: string; note?: string };').natlang, '{ id?: string, note?: string }');
  assert.equal(describeFromSource('Required<User>', 'type User = { id: string; note?: string };').natlang, '{ id: string, note: string }');
  assert.equal(describeFromSource('Pick<User, "id">', 'type User = { id: string; note?: string };').natlang, '{ id: string }');
  assert.equal(describeFromSource('Omit<User, "note">', 'type User = { id: string; note?: string };').natlang, '{ id: string }');
});

test('Record<LiteralUnion, V> resolves inline literals, aliases, and nested union aliases', () => {
  const env = new TypeEnv({
    Primary: parseType('"a" | "b"'),
    AllKeys: parseType('Primary | "c"'),
    NotString: parseType('1 | 2'),
  });

  // Inline literal union (including duplicate deduplication)
  assert.equal(formatType(parseType('Record<"a" | "b" | "a", number>')), '{ a: number, b: number }');
  assert.deepEqual(coerce({ a: 1, b: 2 }, parseType('Record<"a" | "b", number>'), env), { a: 1, b: 2 });

  // Aliased literal union and nested union alias
  assert.equal(formatType(env.resolve(parseType('Record<Primary, number>'))), '{ a: number, b: number }');
  assert.equal(formatType(env.resolve(parseType('Record<AllKeys, boolean>'))), '{ a: boolean, b: boolean, c: boolean }');
  assert.equal(formatType(env.resolve(parseType('Record<Primary | "d", string>'))), '{ a: string, b: string, d: string }');

  assert.deepEqual(schemaOf(parseType('Record<AllKeys, number>'), env), {
    type: 'object',
    properties: { a: { type: 'number' }, b: { type: 'number' }, c: { type: 'number' } },
    required: ['a', 'b', 'c'],
    additionalProperties: false,
  });

  assert.throws(() => env.resolve(parseType('Record<NotString, number>')), TypeSyntaxError);
  assert.equal(describeFromSource('Record<"a" | "b", number>').natlang, '{ a: number, b: number }');
});
