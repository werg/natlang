import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseType, formatType, fitsType, TypeEnv, TypeSyntaxError } from '../dist/native/types.js';
import { coerce, Reject } from '../dist/native/values.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder, neuraleseContentId } from '../dist/native/neuralese-store.js';
import { hexDigest } from '../dist/native/hash.js';
import { decodeTurnValue, encodeMessages, isNeuraleseRef, neuraleseRef, neuraleseSentinel, partsToText,
  sourceWithLiteralCalls, textToParts, writeLiterals } from '../dist/native/neuralese.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { compileScopeSnippet } from '../dist/scope-compiler.js';
import { compileModule } from '../dist/runtime/modules.js';
import { arrayToStringNeuralese, concatNeuralese, joinNeuralese, readNeuraleseIfReference } from '../dist/runtime/lowered.js';
import { readNeuraleseForCurrentTask } from '../dist/neuralese/combinators.js';
import { NativeToolAgent } from '../dist/native/agent.js';
import { createNatlangRuntime, iterateOn } from '../dist/runtime/node.js';
import { defineNatlang } from '../dist/runtime/callable.js';
import { IterationLimitError } from '../dist/runtime/iterate.js';
import { session as open } from './support/natlang.mjs';

const standIn = (width = 8) => {
  const store = new MemoryNeuraleseStore();
  return { store, port: new StandInNeuralesePort(store, hashingEmbedder(width), width) };
};

test('Neuralese<T> parses, formats and fits by element and dialect', () => {
  const plan = parseType('Neuralese<{ steps: string[] }>');
  assert.deepEqual(plan, { kind: 'neuralese', dialect: 'DefaultDialect',
    element: { kind: 'record', fields: [{ name: 'steps', optional: false, type: { kind: 'list', element: { kind: 'prim', name: 'string' } } }] } });
  assert.equal(formatType(plan), 'Neuralese<{ steps: string[] }>');
  const tagged = parseType('Neuralese<string, "nd:natlang@1">');
  assert.equal(tagged.dialect, 'nd:natlang@1');
  assert.equal(formatType(tagged), 'Neuralese<string, "nd:natlang@1">');
  assert.equal(formatType(parseType('Neuralese<(t: string) => number>')), 'Neuralese<(t: string) => number>');
  assert.ok(fitsType(parseType('Neuralese<"a">'), parseType('Neuralese<string>')));
  assert.ok(!fitsType(parseType('Neuralese<string>'), parseType('string')), 'a soft value is not a T');
  assert.ok(!fitsType(parseType('string'), parseType('Neuralese<string>')), 'a T is not a soft value');
  assert.ok(!fitsType(tagged, parseType('Neuralese<string>')), 'dialects do not unify');
});

test('nested Neuralese types and recursive function types are rejected; recursive data is not', () => {
  assert.throws(() => parseType('Neuralese<Neuralese<string>>'), /neuralese-nested/);
  const env = new TypeEnv({ Soft: parseType('Neuralese<string>') });
  assert.throws(() => env.checkNames(parseType('Neuralese<Soft>')), /neuralese-nested/);
  assert.throws(() => new TypeEnv({ F: parseType('(f: F) => string') }), error =>
    error instanceof TypeSyntaxError && /type-recursive-function/.test(error.message));
  assert.throws(() => new TypeEnv({ A: parseType('(b: B) => string'), B: parseType('{ f: A }') }), /type-recursive-function/);
  assert.doesNotThrow(() => new TypeEnv({ Tree: parseType('{ label: string, children: Tree[] }') }));
});

test('reference values check against Neuralese types without touching the payload', () => {
  const env = new TypeEnv({ Plan: parseType('{ steps: string[] }') });
  const id = neuraleseContentId({ dialect: 'd', length: 0, width: 4, dtype: 'f32', data: new Uint8Array() });
  const ref = neuraleseRef('Neuralese<Plan>', id);
  assert.equal(coerce(ref, parseType('Neuralese<Plan>'), env), ref);
  assert.throws(() => coerce(ref, parseType('Neuralese<string>'), env), Reject);
  assert.throws(() => coerce(ref, parseType('Plan'), env), Reject, 'a reference is not the value');
  assert.throws(() => coerce({ steps: [] }, parseType('Neuralese<Plan>'), env), Reject);
  const fromMarker = coerce(neuraleseSentinel(id), parseType('Neuralese<Plan>'), env);
  assert.ok(isNeuraleseRef(fromMarker));
  assert.deepEqual(fromMarker.$neuralese, { type: 'Neuralese<Plan>', id });
});

test('the store is content addressed, deduplicates, pins and collects; the stand-in port embeds the written text', async () => {
  const { store, port } = standIn(4);
  const a = await port.write('remember the plan');
  const b = await port.write('remember the plan');
  assert.match(a.id, /^nz1_[a-z2-7]{52}$/);
  assert.equal(a.id, b.id);
  assert.deepEqual([a.length, a.width, a.dtype, a.dialect], [3, 4, 'f32', 'nd:standin@0']);
  const block = await store.get(a.id);
  assert.equal(block.data.length, 3 * 4 * 4);
  const other = await port.write('something else');
  await store.pin(other.id);
  assert.deepEqual(await store.collect(new Set()), [a.id]);
  assert.ok(await store.has(other.id));
  await store.unpin(other.id);
  assert.deepEqual(await store.collect(new Set()), [other.id]);
  await assert.rejects(() => store.put({ dialect: 'd', length: 2, width: 4, dtype: 'f32', data: new Uint8Array(3) }), /bytes/);
});

test('literals move between marker text, conversation text, content parts and compiled source', async () => {
  const { port } = standIn();
  const written = await writeLiterals('const p: Neuralese<Plan> = <|neuralese|>keep it short<|/neuralese|>; return p;', port);
  assert.equal(written.blocks.length, 1);
  const id = written.blocks[0].id;
  assert.equal(written.text, `const p: Neuralese<Plan> = ${neuraleseSentinel(id)}; return p;`);
  const parts = textToParts(written.text);
  assert.deepEqual(parts.map(part => part.type), ['text', 'neuralese', 'text']);
  assert.equal(partsToText(parts), written.text);
  assert.equal(sourceWithLiteralCalls(written.text), `const p: Neuralese<Plan> = __neuralese(${JSON.stringify(id)}); return p;`);
  const unclosed = await writeLiterals('x = <|neuralese|>never closed', port);
  assert.equal(unclosed.blocks[0].truncated, true);
  await assert.rejects(() => writeLiterals('<|neuralese|>x<|/neuralese|>', undefined), /neuralese-unsupported-backend/);
  // Requests carry blocks as parts in content and in tool-call arguments; a reply's parts decode back to markers.
  const { messages, blocks } = encodeMessages([{ role: 'user', content: 'plain' }, { role: 'assistant', content: '',
    tool_calls: [{ id: '1', type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code: written.text }) } }] }]);
  assert.equal(blocks, 1);
  assert.equal(messages[0].content, 'plain');
  assert.deepEqual(messages[1].tool_calls[0].function.arguments.map(part => part.type), ['text', 'neuralese', 'text']);
  const decoded = await decodeTurnValue([['eval', { code: [{ type: 'text', text: 'const q: Neuralese<Plan> = ' }, { type: 'neuralese', id }] }]],
    port, {});
  assert.equal(decoded[0][1].code, `const q: Neuralese<Plan> = ${neuraleseSentinel(id)}`);
});

const SCOPE = { types: { Plan: '{ steps: string[] }' }, inputs: [{ name: 'plan', type: 'Neuralese<Plan>' }], locals: [], captures: [],
  imports: [], returns: 'Neuralese<Plan>' };
const codes = source => analyzeEvalSnippet(source, SCOPE).diagnostics.map(item => item.code);
const readouts = source => analyzeEvalSnippet(source, SCOPE).readouts.map(item => source.slice(item.start, item.end));
const READ_SCOPE = { ...SCOPE, locals: [{ name: 'read', type: '(value: Neuralese<Plan>) => Promise<Plan>', mutable: false }] };

const TEXT_SCOPE = { types: {}, inputs: [{ name: 'text', type: 'Neuralese<string>' },
  { name: 'numberText', type: 'Neuralese<number>' }, { name: 'nullText', type: 'Neuralese<null>' },
  { name: 'values', type: '(Neuralese<string> | Neuralese<null> | string | number | { toString(): string } | null | undefined)[]' }], locals: [], captures: [], imports: [], returns: 'string' };
const analyzeText = source => analyzeEvalSnippet(source, TEXT_SCOPE);
const textReadouts = source => analyzeText(source).readouts.map(item => source.slice(item.start, item.end));
const SOFT_STRING_UNION_SCOPE = { ...SCOPE, inputs: [{ name: 'value', type: 'Neuralese<string> | Neuralese<number> | string' }] };

test('eval code cannot inspect or branch on a soft value, while text conversions request typed readout', () => {
  assert.deepEqual(codes('plan.steps'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('plan["steps"]'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('if (plan) { console.log(1); }'), ['neuralese-condition']);
  assert.deepEqual(codes('const ok = plan ? 1 : 2;'), ['neuralese-condition']);
  assert.deepEqual(codes('const t = `plan: ${plan}`;'), []);
  assert.deepEqual(readouts('const t = `plan: ${plan}`;'), ['plan']);
  assert.deepEqual(codes('const t = "plan: " + plan;'), []);
  assert.deepEqual(readouts('const t = "plan: " + plan;'), ['plan']);
  assert.deepEqual(codes('const t = String(plan);'), []);
  assert.deepEqual(readouts('const t = String(plan);'), ['plan']);
  const allSoftUnion = analyzeEvalSnippet('return `value=${String(value)}`;', {
    ...SCOPE, inputs: [{ name: 'value', type: 'Neuralese<string> | Neuralese<number>' }], returns: 'string',
  });
  assert.deepEqual(allSoftUnion.diagnostics, []);
  assert.deepEqual(allSoftUnion.readouts.map(item => ['return `value=${String(value)}`;'.slice(item.start, item.end), item.conditional]),
    [['value', true]]);
  const mixedUnion = analyzeEvalSnippet('return `value=${String(value)}`;', { ...SOFT_STRING_UNION_SCOPE, returns: 'string' });
  assert.deepEqual(mixedUnion.diagnostics, []);
  assert.equal(mixedUnion.readouts.length, 1);
  assert.equal(mixedUnion.readouts[0].conditional, true);
  assert.deepEqual(analyzeEvalSnippet('return value.toString();', { ...SOFT_STRING_UNION_SCOPE, returns: 'string' }).diagnostics, []);
  assert.deepEqual(analyzeEvalSnippet('return "" + value;', { ...SOFT_STRING_UNION_SCOPE, returns: 'string' }).readouts.map(item => item.conditional), [true]);
  assert.deepEqual(codes('let text = ""; text += plan;'), []);
  assert.deepEqual(readouts('let text = ""; text += plan;'), ['plan']);
  assert.deepEqual(codes('function show() { return String(plan); }'), ['neuralese-readout-sync']);
  assert.deepEqual(codes('function show(String: (x: unknown) => string) { return String(plan); }'), []);
  assert.deepEqual(codes('const same = plan === plan;'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('JSON.stringify(plan)'), []);
  const typedJson = analyzeEvalSnippet('return JSON.stringify(value);', {
    ...SCOPE, inputs: [{ name: 'value', type: 'Neuralese<string> | Neuralese<number>' }], returns: 'string',
  });
  assert.deepEqual(typedJson.diagnostics, []);
  assert.deepEqual(typedJson.readouts.map(item => [item.kind, item.conditional]), [['json', true]]);
  const crispJson = analyzeEvalSnippet('return JSON.stringify(value);', {
    ...SOFT_STRING_UNION_SCOPE, returns: 'string',
  });
  assert.deepEqual(crispJson.diagnostics, []);
  assert.deepEqual(crispJson.readouts.map(item => [item.kind, item.conditional]), [['json', true]]);
  const directJson = analyzeEvalSnippet('return JSON.stringify(plan, (_key, value) => value, 2);', { ...SCOPE, returns: 'string' });
  assert.deepEqual(directJson.diagnostics, []);
  assert.deepEqual(directJson.readouts.map(item => [item.kind, item.conditional]), [['json', undefined]]);
  assert.deepEqual(directJson.readouts.map(item => 'return JSON.stringify(plan, (_key, value) => value, 2);'.slice(item.start, item.end)),
    ['JSON.stringify(plan, (_key, value) => value, 2)']);
  assert.deepEqual(codes('function show(JSON: any) { return JSON.stringify(plan); }'), []);
  assert.deepEqual(codes('const copy = { ...plan };'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('for (const key in plan) {}'), ['neuralese-opaque-access']);
  assert.deepEqual(analyzeEvalSnippet('for (const key in await read(plan)) {}', READ_SCOPE).diagnostics.map(item => item.code), []);
  assert.deepEqual(codes('const keep: Neuralese<Plan> = plan; return keep;'), []);
  assert.deepEqual(codes('let nested: Neuralese<Neuralese<Plan>>;'), ['neuralese-nested']);
  assert.deepEqual(codes('type F = (f: F) => string;'), ['type-recursive-function']);
  assert.deepEqual(codes('type Tree = { children: Tree[] };'), []);
});

test('computed member and object keys read soft values at the key position', async () => {
  const keyType = 'Neuralese<string> | string | symbol';
  const scope = { types: {}, inputs: [
    { name: 'record', type: 'Record<PropertyKey, string>' }, { name: 'key', type: keyType },
    { name: 'events', type: 'string[]' }], locals: [], captures: [], imports: [], returns: 'string' };
  const accessSource = 'const result = record[(events.push("key"), key)]; return result;';
  const access = analyzeEvalSnippet(accessSource, scope);
  assert.deepEqual(access.diagnostics, []);
  assert.deepEqual(access.readouts.map(item => [accessSource.slice(item.start, item.end), item.conditional]),
    [['(events.push("key"), key)', true]]);
  const objectSource = 'const result = { [(events.push("key"), key)]: (events.push("value"), "created") }; return result[key];';
  const object = analyzeEvalSnippet(objectSource, scope);
  assert.deepEqual(object.diagnostics, []);
  assert.deepEqual(object.readouts.map(item => [objectSource.slice(item.start, item.end), item.conditional]), [
    ['(events.push("key"), key)', true], ['key', true],
  ]);

  const compileScope = source => compileScopeSnippet(source, { inputBindings: ['record', 'key', 'events'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, scope) });
  const runScope = (compiled, live, values) => {
    assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
    const fn = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
      `${compiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
      output => output.result, live);
    return fn(values, {}, {});
  };
  const softKey = neuraleseRef('Neuralese<string>', 'nz1_ssssssssssssssssssss');
  const scopeEvents = [];
  const proxy = new Proxy({ alpha: 'found' }, { get(target, property, receiver) {
    scopeEvents.push(`lookup:${typeof property === 'symbol' ? 'symbol' : String(property)}`);
    return Reflect.get(target, property, receiver);
  } });
  const readKey = async value => {
    if (!isNeuraleseRef(value)) return value;
    scopeEvents.push('read:key');
    return 'alpha';
  };
  assert.equal(await runScope(compileScope(accessSource), { readNeuraleseIfReference: readKey },
    { record: proxy, key: softKey, events: scopeEvents }), 'found');
  assert.deepEqual(scopeEvents, ['key', 'read:key', 'lookup:alpha']);
  const readFailureEvents = [];
  await assert.rejects(runScope(compileScope(accessSource), { readNeuraleseIfReference: async () => {
    readFailureEvents.push('read:key'); throw new Error('key read failed');
  } }, { record: proxy, key: softKey, events: readFailureEvents }), /key read failed/);
  assert.deepEqual(readFailureEvents, ['key', 'read:key']);

  const objectScope = { ...scope, returns: 'Record<PropertyKey, string>' };
  const buildSource = 'return { [(events.push("key"), key)]: (events.push("value"), "created") };';
  const built = compileScopeSnippet(buildSource, { inputBindings: ['key', 'events'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, objectScope) });
  const buildEvents = [];
  const builtObject = await runScope(built, { readNeuraleseIfReference: async value => {
    if (!isNeuraleseRef(value)) return value;
    buildEvents.push('read:key'); return 'alpha';
  } }, { key: softKey, events: buildEvents });
  assert.deepEqual(builtObject, { alpha: 'created' });
  assert.deepEqual(buildEvents, ['key', 'read:key', 'value']);
  const symbol = Symbol('native key');
  const symbolObject = await runScope(built, { readNeuraleseIfReference: async value => value }, { key: symbol, events: [] });
  assert.equal(symbolObject[symbol], 'created');

  const keyModule = compileModule({ kind: 'module', id: 'computed-soft-keys', name: 'computedKeys', source: 'computedKeys.ts', revision: 'r1',
    text: `export async function lookup(getRecord: () => Record<PropertyKey, string>, key: Neuralese<string> | string | symbol,
      events: string[]) {
      return (events.push('base'), getRecord())[(events.push('key'), key)];
    }
    export async function make(key: Neuralese<string> | string | symbol, events: string[]) {
      return { [(events.push('key'), key)]: (events.push('value'), 'created') };
    }
    export async function nested(keys: (Neuralese<string> | string | symbol)[][]) {
      const output: Record<PropertyKey, number> = {};
      for (const row of keys) for (const key of row) output[key] = (output[key] ?? 0) + 1;
      return output;
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(keyModule, /readNeuraleseIfReference/);
  const keyModuleExports = {};
  const keyModuleFns = new Function('exports', '__natlang', `${keyModule}; return exports;`)(keyModuleExports, {
    guard: (_id, fn) => fn(), finite: iterable => iterable, readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      moduleEvents.push('read:key');
      return 'alpha';
    },
  });
  const moduleEvents = [];
  const moduleSoftKey = neuraleseRef('Neuralese<string>', 'nz1_tttttttttttttttttttt');
  const moduleRecord = new Proxy({ alpha: 'module found' }, { get(target, property, receiver) {
    moduleEvents.push(`lookup:${typeof property === 'symbol' ? 'symbol' : String(property)}`);
    return Reflect.get(target, property, receiver);
  } });
  assert.equal(await keyModuleFns.lookup(() => moduleRecord, moduleSoftKey, moduleEvents), 'module found');
  assert.deepEqual(moduleEvents, ['base', 'key', 'read:key', 'lookup:alpha']);
  moduleEvents.length = 0;
  assert.equal(await keyModuleFns.lookup(() => moduleRecord, 'alpha', moduleEvents), 'module found');
  assert.deepEqual(moduleEvents, ['base', 'key', 'lookup:alpha']);
  const moduleSymbol = Symbol('module key');
  const symbolRecord = { [moduleSymbol]: 'symbol found' };
  assert.equal(await keyModuleFns.lookup(() => symbolRecord, moduleSymbol, []), 'symbol found');

  moduleEvents.length = 0;
  const moduleBuilt = await keyModuleFns.make(moduleSoftKey, moduleEvents);
  assert.deepEqual(moduleBuilt, { alpha: 'created' });
  assert.deepEqual(moduleEvents, ['key', 'read:key', 'value']);
  const moduleSymbolObject = await keyModuleFns.make(moduleSymbol, []);
  assert.equal(moduleSymbolObject[moduleSymbol], 'created');

  const nestedCount = await keyModuleFns.nested([[moduleSoftKey, 'beta'], [moduleSoftKey, moduleSymbol]]);
  assert.deepEqual({ alpha: nestedCount.alpha, beta: nestedCount.beta, symbol: nestedCount[moduleSymbol] },
    { alpha: 2, beta: 1, symbol: 1 });
});

test('explicit Number and Boolean read only scalar Neuralese alternatives after native arguments evaluate', async () => {
  const scope = { types: {}, inputs: [
    { name: 'numberValue', type: 'Neuralese<number> | number' },
    { name: 'textValue', type: 'Neuralese<string> | string' },
    { name: 'events', type: 'string[]' },
  ], locals: [], captures: [], imports: [], returns: 'number' };
  const source = `return Number((events.push('number'), numberValue), (events.push('extra'), 9)) +
    (Boolean((events.push('boolean'), textValue)) ? 100 : 0);`;
  const analysis = analyzeEvalSnippet(source, scope);
  assert.deepEqual(analysis.diagnostics, []);
  assert.deepEqual(analysis.readouts.map(item => [item.kind, item.argument, item.conversion, item.conditional]), [
    ['scalar-conversion', 0, 'Number', true], ['scalar-conversion', 0, 'Boolean', true],
  ]);
  const compiled = compileScopeSnippet(source, { inputBindings: ['numberValue', 'textValue', 'events'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, scope) });
  const events = [];
  let failReads = false;
  const readNeuraleseIfReference = async value => {
    if (!isNeuraleseRef(value)) return value;
    events.push(`read:${value.$neuralese.id}`);
    if (failReads) throw new Error('scalar read failed');
    return value.$neuralese.id === 'nz1_nnnnnnnnnnnnnnnnnnnn' ? 17 : 'false';
  };
  const run = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${compiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuraleseIfReference });
  const softResult = await run({ numberValue: neuraleseRef('Neuralese<number>', 'nz1_nnnnnnnnnnnnnnnnnnnn'),
    textValue: neuraleseRef('Neuralese<string>', 'nz1_tttttttttttttttttttt'), events }, {}, {});
  assert.equal(softResult, 117, 'Boolean uses native truthiness of the exact read text');
  assert.deepEqual(events, ['number', 'extra', 'read:nz1_nnnnnnnnnnnnnnnnnnnn', 'boolean', 'read:nz1_tttttttttttttttttttt']);

  events.length = 0;
  const crispResult = await run({ numberValue: 3, textValue: '', events }, {}, {});
  assert.equal(crispResult, 3);
  assert.deepEqual(events, ['number', 'extra', 'boolean'], 'crisp alternatives go directly to native constructors');

  events.length = 0;
  failReads = true;
  await assert.rejects(() => run({ numberValue: neuraleseRef('Neuralese<number>', 'nz1_nnnnnnnnnnnnnnnnnnnn'),
    textValue: 'unused', events }, {}, {}), /scalar read failed/);
  assert.deepEqual(events, ['number', 'extra', 'read:nz1_nnnnnnnnnnnnnnnnnnnn'],
    'a failed trained read propagates after native argument evaluation and before later expressions');
  failReads = false;

  const module = compileModule({ kind: 'module', id: 'scalar-readout', name: 'scalarReadout', source: 'scalarReadout.ts', revision: 'r1',
    text: `export async function convert(numberValue: Neuralese<number> | number, textValue: Neuralese<string> | string) {
      return Number(numberValue) + (Boolean(textValue) ? 100 : 0);
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const exports = {};
  const moduleFns = new Function('exports', '__natlang', `${module}; return exports;`)(exports, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value =>
      isNeuraleseRef(value) ? (value.$neuralese.id === 'nz1_nnnnnnnnnnnnnnnnnnnn' ? 17 : 'false') : value,
  });
  assert.equal(await moduleFns.convert(neuraleseRef('Neuralese<number>', 'nz1_nnnnnnnnnnnnnnnnnnnn'),
    neuraleseRef('Neuralese<string>', 'nz1_tttttttttttttttttttt')), 117);

  const opaque = analyzeEvalSnippet('Number(value);', { ...scope,
    inputs: [{ name: 'value', type: 'Neuralese<{ amount: number }>' }], returns: 'number' });
  assert.ok(opaque.diagnostics.some(item => item.code === 'neuralese-opaque-access'));
  const condition = analyzeEvalSnippet('if (flag) return 1;', { ...scope,
    inputs: [{ name: 'flag', type: 'Neuralese<boolean> | boolean' }], returns: 'number' });
  assert.ok(condition.diagnostics.some(item => item.code === 'neuralese-condition'),
    'mixed soft/crisp conditionals cannot branch on wrapper truthiness');
  const shadowed = analyzeEvalSnippet('function f(Number: (x: unknown) => number) { return Number(value); }', {
    ...scope, inputs: [{ name: 'value', type: 'Neuralese<number>' }], returns: 'number' });
  assert.deepEqual(shadowed.readouts, [], 'shadowed constructors keep their declared call semantics');
});

test('string conversions read typed Neuralese values with native method ordering', async () => {
  const toString = analyzeText('return text.toString();');
  assert.deepEqual(toString.diagnostics, []);
  assert.deepEqual(textReadouts('return text.toString();'), ['text']);
  const numberToString = analyzeText('return numberText.toString();');
  assert.deepEqual(numberToString.diagnostics, []);
  assert.deepEqual(numberToString.readouts.map(item => 'numberText'), ['numberText']);
  const concatSource = `return ''.concat('prefix', text, 'suffix');`;
  const concat = analyzeText(concatSource);
  assert.deepEqual(concat.diagnostics, []);
  assert.deepEqual(textReadouts(concatSource), ["''.concat('prefix', text, 'suffix')"]);
  const numericConcat = analyzeText(`return ''.concat(numberText);`);
  assert.deepEqual(numericConcat.diagnostics, []);
  assert.deepEqual(numericConcat.readouts.map(item => item.kind), ['concat']);
  const join = analyzeText(`return values.join('|');`);
  assert.deepEqual(join.diagnostics, []);
  assert.deepEqual(join.readouts.map(item => [item.kind, "values.join('|')"]), [['join', "values.join('|')"]]);
  const arrayString = analyzeText('return values.toString();');
  assert.deepEqual(arrayString.diagnostics, []);
  assert.deepEqual(arrayString.readouts.map(item => [item.kind, 'values.toString()']), [['array-string', 'values.toString()']]);
  const analyzeJoinType = type => analyzeEvalSnippet('return values.join("|");', {
    ...TEXT_SCOPE, inputs: [{ name: 'values', type }],
  });
  for (const type of ['Neuralese<string>[] | string[]', 'readonly Neuralese<string>[] | readonly string[]',
    'readonly [Neuralese<string>] | readonly [string]']) {
    const unionJoin = analyzeJoinType(type);
    assert.deepEqual(unionJoin.diagnostics, [], type);
    assert.equal(unionJoin.readouts[0]?.kind, 'join', type);
  }
  const unionConcat = analyzeEvalSnippet('return "".concat(...values);', {
    ...TEXT_SCOPE, inputs: [{ name: 'values', type: 'Neuralese<string>[] | string[]' }],
  });
  assert.deepEqual(unionConcat.diagnostics, []);
  assert.equal(unionConcat.readouts[0]?.kind, 'concat');
  const plainUnionJoin = analyzeJoinType('string[] | number[]');
  assert.deepEqual(plainUnionJoin.diagnostics, []);
  assert.deepEqual(plainUnionJoin.readouts, []);
  const spreadConcat = analyzeText(`return ''.concat(...values);`);
  assert.deepEqual(spreadConcat.diagnostics, []);
  assert.deepEqual(spreadConcat.readouts.map(item => [item.kind, "''.concat(...values)"]), [['concat', "''.concat(...values)"]]);
  assert.deepEqual(analyzeText('function show() { return text.toString(); }').diagnostics.map(item => item.code),
    ['neuralese-readout-sync']);
  const planString = analyzeEvalSnippet('return plan.toString();', SCOPE);
  assert.deepEqual(planString.diagnostics, []);
  assert.deepEqual(planString.readouts.map(item => 'plan'), ['plan']);
  const mixedJoin = analyzeText("return ['label', numberText, nullText, 3, null].join('|');");
  assert.deepEqual(mixedJoin.diagnostics, []);
  assert.equal(mixedJoin.readouts[0].kind, 'join');

  const program = createVirtualProgram({ '/scope/main.ts': `async function show(text: Neuralese<number>) {
    return ''.concat('prefix', text, 'suffix');
  }` });
  assert.deepEqual(program.getSemanticDiagnostics(program.getSourceFile('/scope/main.ts')).map(item => item.code), []);

  const compiled = compileScopeSnippet(`return values.join('|');`, { inputBindings: ['values'], neuralese: true,
    analyze: source => analyzeText(source) });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.match(compiled.program, /joinNeuralese\(__natlang_join_receiver_[^,]+, __natlang_join_method_[^,]+, __natlang_join_args_[^,]+, async/);
  const arrayStringCompiled = compileScopeSnippet('return values.toString();', { inputBindings: ['values'], neuralese: true,
    analyze: source => analyzeText(source) });
  assert.equal(arrayStringCompiled.ok, true, JSON.stringify(arrayStringCompiled.diagnostics));
  assert.match(arrayStringCompiled.program, /arrayToStringNeuralese/);
  const scopeArrayStringEvents = [];
  const runArrayStringScope = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${arrayStringCompiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { arrayToStringNeuralese, readNeuralese: async ref => {
      scopeArrayStringEvents.push(`read:${ref.$neuralese.id}`); return 'scope soft';
    } });
  assert.equal(await runArrayStringScope({ values: [neuraleseRef('Neuralese<string>', 'nz1_qqqqqqqqqqqqqqqqqqqq'), 'crisp'] }, {}, {}),
    'scope soft,crisp');
  assert.deepEqual(scopeArrayStringEvents, ['read:nz1_qqqqqqqqqqqqqqqqqqqq']);
  const spreadCompiled = compileScopeSnippet(`return ''.concat(...values);`, { inputBindings: ['values'], neuralese: true,
    analyze: source => analyzeText(source) });
  assert.equal(spreadCompiled.ok, true, JSON.stringify(spreadCompiled.diagnostics));
  assert.match(spreadCompiled.program, /concatNeuralese/);
  assert.match(spreadCompiled.program, /__values = \[\.\.\.values\]/);
  const unionCompiled = compileScopeSnippet('return String(value);', { inputBindings: ['value'], neuralese: true,
    analyze: source => analyzeEvalSnippet(source, { ...SOFT_STRING_UNION_SCOPE, returns: 'string' }) });
  assert.equal(unionCompiled.ok, true, JSON.stringify(unionCompiled.diagnostics));
  assert.match(unionCompiled.program, /readNeuraleseIfReference/);
  assert.equal(await readNeuraleseIfReference('plain'), 'plain');

  const jsonScope = { ...SCOPE, inputs: [{ name: 'value', type: 'Neuralese<string> | Neuralese<number> | string' }], returns: 'string' };
  const jsonCompiled = compileScopeSnippet('return JSON.stringify(value);', { inputBindings: ['value'], neuralese: true,
    analyze: source => analyzeEvalSnippet(source, jsonScope) });
  assert.equal(jsonCompiled.ok, true, JSON.stringify(jsonCompiled.diagnostics));
  assert.match(jsonCompiled.program, /readNeuraleseIfReference/);
  assert.match(jsonCompiled.program, /\.stringify\], \[value\]\)/);
  let scopeJsonReads = 0;
  const runJsonScope = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${jsonCompiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      scopeJsonReads++;
      return value.$neuralese.type === 'Neuralese<number>' ? 17 : 'scope payload';
    } });
  assert.equal(await runJsonScope({ value: neuraleseRef('Neuralese<string>', 'nz1_qqqqqqqqqqqqqqqqqqqq') }, {}, {}), '"scope payload"');
  assert.equal(await runJsonScope({ value: 'crisp scope value' }, {}, {}), '"crisp scope value"');
  assert.equal(scopeJsonReads, 1);
  const runJsonScopeSource = async (source, value) => {
    const compiled = compileScopeSnippet(source, { inputBindings: ['value'], neuralese: true,
      analyze: text => analyzeEvalSnippet(text, jsonScope) });
    assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
    const scope = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
      `${compiled.program}; return __natlang_scope;`)(item => item, item => item, async item => item,
      output => output.result, { readNeuraleseIfReference: async item => isNeuraleseRef(item) ? 'scope payload' : item });
    return scope({ value }, {}, {});
  };
  const scopeSoftValue = neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr');
  assert.equal(await runJsonScopeSource('const serialized = JSON.stringify(value); return typeof serialized;', scopeSoftValue), 'string');
  assert.equal(await runJsonScopeSource(`return 'json=' + JSON.stringify(value);`, scopeSoftValue), 'json="scope payload"');
  assert.equal(await runJsonScopeSource('return JSON.stringify({ nested: JSON.stringify(value) });', scopeSoftValue),
    '{"nested":"\\"scope payload\\""}');
  assert.equal(await runJsonScopeSource('const serialized = JSON.stringify(value); return typeof serialized;', 'crisp'), 'string');
  assert.equal(await runJsonScopeSource(`return 'json=' + JSON.stringify(value);`, 'crisp'), 'json="crisp"');
  assert.equal(await runJsonScopeSource('return JSON.stringify({ nested: JSON.stringify(value) });', 'crisp'),
    '{"nested":"\\"crisp\\""}');

  const module = compileModule({ kind: 'module', id: 'module-join', name: 'joiner', source: 'joiner.ts', revision: 'r1',
    text: `export async function show(text: Neuralese<number>, values: (Neuralese<string> | string | number | { toString(): string } | null | undefined)[]) {
      return text.toString() + ''.concat(text) + values.join('|');
    }
    export async function arrayString(values: (Neuralese<string> | string)[]) { return values.toString(); }
    export async function arrayStringOrder(values: (Neuralese<string> | string)[], events: string[], empty: []) {
      return (events.push('receiver'), values).toString(...(events.push('argument'), empty));
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(module, /\(await __natlang\.readNeuralese\(text\)\)\.toString\(\)/);
  assert.match(module, /concatNeuralese/);
  assert.match(module, /joinNeuralese\(__natlang_join_receiver_[^,]+, __natlang_join_method_[^,]+, __natlang_join_args_[^,]+, async/);
  assert.match(module, /arrayToStringNeuralese\(__natlang_array_string_receiver_[^,]+, __natlang_array_string_method_[^,]+/);
  const moduleJoinEvents = [];
  const joinModuleExports = {};
  const moduleStringFns = new Function('exports', '__natlang', `${module}; return exports;`)(joinModuleExports, {
    guard: (_id, fn) => fn(), concatNeuralese, joinNeuralese, arrayToStringNeuralese,
    readNeuralese: async ref => { moduleJoinEvents.push(`read:${ref.$neuralese.id}`); return 'module soft'; },
  });
  const arrayTextRef = neuraleseRef('Neuralese<string>', 'nz1_eeeeeeeeeeeeeeeeeeee');
  assert.equal(await moduleStringFns.arrayString([arrayTextRef, 'ordinary']), 'module soft,ordinary');
  assert.equal(await moduleStringFns.arrayString(['crisp', 'only']), 'crisp,only');
  moduleJoinEvents.length = 0;
  const orderedArray = new Proxy([arrayTextRef, 'ordinary'], { get(target, property, receiver) {
    if (property === 'toString') moduleJoinEvents.push('method');
    if (property === 'join') moduleJoinEvents.push('join');
    if (property === 'length') moduleJoinEvents.push('length');
    if (property === '0' || property === '1') moduleJoinEvents.push(`item:${property}`);
    return Reflect.get(target, property, receiver);
  } });
  assert.equal(await moduleStringFns.arrayStringOrder(orderedArray, moduleJoinEvents, []), 'module soft,ordinary');
  assert.deepEqual(moduleJoinEvents, ['receiver', 'method', 'argument', 'join', 'length', 'item:0',
    'read:nz1_eeeeeeeeeeeeeeeeeeee', 'item:1']);
  const customToStringEvents = [];
  const customToStringValues = new Proxy([arrayTextRef], { get(target, property, receiver) {
    if (property === 'toString') {
      customToStringEvents.push('method');
      return function (...args) {
        customToStringEvents.push(`call:${this === customToStringValues}:${args.length}`);
        return 'custom text';
      };
    }
    customToStringEvents.push(`get:${String(property)}`);
    return Reflect.get(target, property, receiver);
  } });
  assert.equal(await moduleStringFns.arrayString(customToStringValues), 'custom text');
  assert.deepEqual(customToStringEvents, ['method', 'call:true:0']);

  const unionStringModule = compileModule({ kind: 'module', id: 'soft-union-string', name: 'unionString', source: 'unionString.ts', revision: 'r1',
    text: `export async function showAllSoft(value: Neuralese<string> | Neuralese<number>) {
      return \`value=\${value}\`;
    }
    export async function showMixed(value: Neuralese<string> | Neuralese<number> | string) {
      return \`value=\${String(value)}\`;
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(unionStringModule, /readNeuraleseIfReference/);
  const unionStringExports = {};
  const unionStringModuleExports = new Function('exports', '__natlang', `${unionStringModule}; return exports;`)(unionStringExports, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value => isNeuraleseRef(value) ? `read:${value.$neuralese.id}` : value,
  });
  const showAllSoft = unionStringModuleExports.showAllSoft;
  const showMixed = unionStringModuleExports.showMixed;
  assert.equal(await showAllSoft(neuraleseRef('Neuralese<string>', 'nz1_kkkkkkkkkkkkkkkkkkkk')),
    'value=read:nz1_kkkkkkkkkkkkkkkkkkkk');
  assert.equal(await showMixed('ordinary'), 'value=ordinary');
  assert.equal(await showMixed(neuraleseRef('Neuralese<string>', 'nz1_llllllllllllllllllll')),
    'value=read:nz1_llllllllllllllllllll');

  const jsonModule = compileModule({ kind: 'module', id: 'json-soft-union', name: 'jsonSoftUnion', source: 'jsonSoftUnion.ts', revision: 'r1',
    text: `export async function show(value: Neuralese<string> | Neuralese<number> | string) {
      return JSON.stringify(value);
    }
    export async function ordered(value: Neuralese<string> | string, events: string[]) {
      return JSON.stringify((events.push('value'), value),
        (events.push('replacer-argument'), (key, item) => (events.push('replacer:' + key), item)),
        (events.push('space'), 0));
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(jsonModule, /readNeuraleseIfReference/);
  const jsonEvents = [];
  const orderedEvents = [];
  const jsonExports = {};
  const jsonFns = new Function('exports', '__natlang', `${jsonModule}; return exports;`)(jsonExports, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      jsonEvents.push(`read:${value.$neuralese.type}`);
      orderedEvents.push(`read:${value.$neuralese.type}`);
      if (value.$neuralese.type === 'Neuralese<number>') return 17;
      return { answer: 'soft payload' };
    },
  });
  assert.equal(await jsonFns.show(neuraleseRef('Neuralese<string>', 'nz1_mmmmmmmmmmmmmmmmmmmm')), '{"answer":"soft payload"}');
  assert.equal(await jsonFns.show(neuraleseRef('Neuralese<number>', 'nz1_nnnnnnnnnnnnnnnnnnnn')), '17');
  assert.equal(await jsonFns.show('crisp'), '"crisp"');
  jsonEvents.length = 0;
  orderedEvents.length = 0;
  assert.equal(await jsonFns.ordered(neuraleseRef('Neuralese<string>', 'nz1_oooooooooooooooooooo'), orderedEvents),
    '{"answer":"soft payload"}');
  assert.deepEqual(orderedEvents, ['value', 'replacer-argument', 'space', 'read:Neuralese<string>', 'replacer:', 'replacer:answer']);
  assert.deepEqual(jsonEvents, ['read:Neuralese<string>']);
  const failedJson = new Function('exports', '__natlang', `${jsonModule}; return exports.show;`)({}, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async () => { throw new Error('typed JSON read failed'); },
  });
  await assert.rejects(() => failedJson(neuraleseRef('Neuralese<string>', 'nz1_pppppppppppppppppppp')),
    /typed JSON read failed/);

  const orderedModule = compileModule({ kind: 'module', id: 'concat-order', name: 'ordered', source: 'ordered.ts', revision: 'r1',
    text: `export async function show(text: Neuralese<number>, events: string[], receiver: () => string, later: () => Promise<string>) {
      return receiver().concat(text, (events.push('later-argument'), await later()), text);
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const moduleExports = {};
  const show = new Function('exports', '__natlang', `${orderedModule}; return exports.show;`)(moduleExports, {
    guard: (_id, fn) => fn(), concatNeuralese,
    readNeuralese: async ref => { events.push(`read:${ref.$neuralese.id}`); return 'value'; },
  });
  const events = [];
  const result = await show(neuraleseRef('Neuralese<number>', 'nz1_cccccccccccccccccccc'), events, () => {
    events.push('receiver'); return '';
  }, async () => 'later-value');
  assert.equal(result, 'valuelater-valuevalue');
  assert.deepEqual(events, ['receiver', 'later-argument', 'read:nz1_cccccccccccccccccccc', 'read:nz1_cccccccccccccccccccc']);

  const unionJoinModule = compileModule({ kind: 'module', id: 'union-join', name: 'unionJoin', source: 'unionJoin.ts', revision: 'r1',
    text: `export async function show(values: Neuralese<string>[] | string[]) { return values.join('|'); }
    export async function separator(values: Neuralese<string>[], delimiter: Neuralese<string> | string) { return values.join(delimiter as any); }
    export async function custom(values: Neuralese<string>[], args: string[], events: string[]) {
      return (events.push('receiver'), values).join(...(events.push('spread'), args));
    }`,
    types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(unionJoinModule, /joinNeuralese/);
  const unionJoinExports = {};
  const unionJoinFns = new Function('exports', '__natlang', `${unionJoinModule}; return exports;`)(unionJoinExports, {
    guard: (_id, fn) => fn(), joinNeuralese,
    readNeuralese: async ref => { events.push(`union-read:${ref.$neuralese.id}`);
      return ref.$neuralese.id === 'nz1_llllllllllllllllllll' ? ' / ' : 'read value'; },
  });
  const showUnionJoin = unionJoinFns.show;
  const unionJoinAnswer = await showUnionJoin([
    neuraleseRef('Neuralese<string>', 'nz1_jjjjjjjjjjjjjjjjjjjj'), 'ordinary',
  ]);
  assert.equal(unionJoinAnswer, 'read value|ordinary');
  assert.deepEqual(events.slice(-1), ['union-read:nz1_jjjjjjjjjjjjjjjjjjjj']);
  const softDelimiter = neuraleseRef('Neuralese<string>', 'nz1_llllllllllllllllllll');
  const joinedWithSoftDelimiter = await unionJoinFns.separator([
    neuraleseRef('Neuralese<string>', 'nz1_mmmmmmmmmmmmmmmmmmmm'),
    neuraleseRef('Neuralese<string>', 'nz1_nnnnnnnnnnnnnnnnnnnn'),
  ], softDelimiter);
  assert.equal(joinedWithSoftDelimiter, 'read value / read value');
  assert.deepEqual(events.slice(-3), [
    'union-read:nz1_llllllllllllllllllll',
    'union-read:nz1_mmmmmmmmmmmmmmmmmmmm',
    'union-read:nz1_nnnnnnnnnnnnnnnnnnnn',
  ]);
  assert.equal(await unionJoinFns.separator(['plain', 'strings'], '|'), 'plain|strings');
  const customJoinEvents = [];
  const customJoinValues = [neuraleseRef('Neuralese<string>', 'nz1_kkkllllllllllllllllll')];
  let customJoinProxy;
  customJoinValues.join = function (...args) {
    customJoinEvents.push(`custom:${this === customJoinProxy}:${args.join('|')}`);
    return 'custom join';
  };
  customJoinProxy = new Proxy(customJoinValues, { get(target, property, receiver) {
    if (property === 'join') customJoinEvents.push('method');
    return Reflect.get(target, property, receiver);
  } });
  assert.equal(await unionJoinFns.custom(customJoinProxy, ['first', 'second'], customJoinEvents), 'custom join');
  assert.deepEqual(customJoinEvents, ['receiver', 'method', 'spread', 'custom:true:first|second']);
  const customSoftSeparatorEvents = [];
  const customSoftSeparatorValues = [neuraleseRef('Neuralese<string>', 'nz1_oooooooooooooooooooo')];
  const customSoftSeparator = neuraleseRef('Neuralese<string>', 'nz1_pppppppppppppppppppp');
  customSoftSeparatorValues.join = function (delimiter) {
    customSoftSeparatorEvents.push(`custom:${this === customSoftSeparatorValues}:${delimiter === customSoftSeparator}`);
    return 'custom soft separator';
  };
  assert.equal(await unionJoinFns.separator(customSoftSeparatorValues, customSoftSeparator), 'custom soft separator');
  assert.deepEqual(customSoftSeparatorEvents, ['custom:true:true'], 'custom joins receive the original typed separator without implicit readout');

  const spreadModule = compileModule({ kind: 'module', id: 'concat-spread-order', name: 'spreadOrder', source: 'spreadOrder.ts', revision: 'r1',
    text: `export async function show(values: (Neuralese<string> | string)[], events: string[], receiver: () => string) {
      return receiver().concat(...(events.push('spread'), values), (events.push('tail'), 'end'));
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const spreadExports = {};
  const showSpread = new Function('exports', '__natlang', `${spreadModule}; return exports.show;`)(spreadExports, {
    guard: (_id, fn) => fn(), concatNeuralese,
    readNeuralese: async ref => { events.push(`read:${ref.$neuralese.id}`); return 'soft'; },
  });
  events.length = 0;
  const spreadAnswer = await showSpread([
    neuraleseRef('Neuralese<string>', 'nz1_hhhhhhhhhhhhhhhhhhhh'), 'middle',
    neuraleseRef('Neuralese<string>', 'nz1_iiiiiiiiiiiiiiiiiiii'),
  ], events, () => { events.push('receiver'); return 'start'; });
  assert.equal(spreadAnswer, 'startsoftmiddlesoftend');
  assert.deepEqual(events, ['receiver', 'spread', 'tail', 'read:nz1_hhhhhhhhhhhhhhhhhhhh', 'read:nz1_iiiiiiiiiiiiiiiiiiii']);

  const spreadOrder = [];
  const spreadValues = ['before', neuraleseRef('Neuralese<number>', 'nz1_gggggggggggggggggggg'), null,
    { toString() { spreadOrder.push('ordinary-coercion'); return 'after'; } }];
  const spreadResult = await concatNeuralese('', ''.concat, [...spreadValues], async ref => {
    spreadOrder.push(`typed-read:${ref.$neuralese.id}`); return 7;
  });
  assert.equal(spreadResult, 'before7nullafter');
  assert.deepEqual(spreadOrder, ['typed-read:nz1_gggggggggggggggggggg', 'ordinary-coercion']);

  const order = [];
  const items = ['label', 2, neuraleseRef('Neuralese<string>', 'nz1_aaaaaaaaaaaaaaaaaaaa'), , null,
    neuraleseRef('Neuralese<string>', 'nz1_bbbbbbbbbbbbbbbbbbbb'), neuraleseRef('Neuralese<null>', 'nz1_dddddddddddddddddddd')];
  const joined = (await joinNeuralese(items, Array.prototype.join, [':'], async ref => {
    order.push(ref.$neuralese.id);
    if (ref.$neuralese.type === 'Neuralese<null>') return null;
    return ref.$neuralese.id.includes('aaaa') ? 'first' : 'last';
  })).value;
  assert.equal(joined, 'label:2:first:::last:');
  assert.deepEqual(order, ['nz1_aaaaaaaaaaaaaaaaaaaa', 'nz1_bbbbbbbbbbbbbbbbbbbb', 'nz1_dddddddddddddddddddd']);
  const separatorOrder = [];
  const separatorItems = new Proxy([neuraleseRef('Neuralese<string>', 'nz1_cccccccccccccccccccc'), 'plain'], {
    get(target, property, receiver) {
      if (property === 'length') separatorOrder.push('length');
      if (property === '0' || property === '1') separatorOrder.push(`item:${property}`);
      return Reflect.get(target, property, receiver);
    },
  });
  const softSeparator = neuraleseRef('Neuralese<string>', 'nz1_dddddddddddddddddddd');
  const softSeparated = await joinNeuralese(separatorItems, Array.prototype.join, [softSeparator], async ref => {
    separatorOrder.push(`read:${ref.$neuralese.id}`);
    return ref.$neuralese.id === softSeparator.$neuralese.id ? ' / ' : 'resolved';
  });
  assert.equal(softSeparated.value, 'resolved / plain');
  assert.deepEqual(separatorOrder, ['length', 'read:nz1_dddddddddddddddddddd', 'item:0', 'read:nz1_cccccccccccccccccccc', 'item:1']);
  const crispSeparator = await joinNeuralese(['a', 'b'], Array.prototype.join, ['|'], async () => {
    assert.fail('crisp separators do not need typed readout');
  });
  assert.equal(crispSeparator.value, 'a|b');
  const chunkBoundaryValues = new Array(65537).fill('x');
  assert.equal((await joinNeuralese(chunkBoundaryValues, Array.prototype.join, ['|'], async () => 'unused')).value,
    chunkBoundaryValues.join('|'), 'chunked assembly preserves separators across the scratch boundary');
  assert.equal((await joinNeuralese(new Array(65536).fill(null), Array.prototype.join, [''], async () => 'unused')).value,
    '', 'empty chunks with an empty separator contribute no output');
  const failedSeparatorOrder = [];
  const failedSeparatorItems = new Proxy(['first', 'second'], {
    get(target, property, receiver) {
      if (property === 'length') failedSeparatorOrder.push('length');
      if (property === '0' || property === '1') failedSeparatorOrder.push(`item:${property}`);
      return Reflect.get(target, property, receiver);
    },
  });
  await assert.rejects(() => joinNeuralese(failedSeparatorItems, Array.prototype.join, [softSeparator], async () => {
    failedSeparatorOrder.push('separator-read'); throw new Error('separator unavailable');
  }), /separator unavailable/);
  assert.deepEqual(failedSeparatorOrder, ['length', 'separator-read'], 'a failed separator read precedes all indexed gets');

  for (const [reportedLength, expected, expectedIndices] of [
    [2.9, 'one|two', ['0', '1']], ['2', 'one|two', ['0', '1']],
    [undefined, '', []], [-3, '', []],
  ]) {
    const lengthEvents = [];
    const lengthProxy = new Proxy([
      neuraleseRef('Neuralese<string>', 'nz1_llllllllllllllllllll'),
      neuraleseRef('Neuralese<string>', 'nz1_mmmmmmmmmmmmmmmmmmmm'),
    ], {
      get(target, property, receiver) {
        if (property === 'length') { lengthEvents.push('length'); return reportedLength; }
        if (property === '0' || property === '1') lengthEvents.push(`item:${property}`);
        return Reflect.get(target, property, receiver);
      },
    });
    const joinedLength = await joinNeuralese(lengthProxy, Array.prototype.join, [softSeparator], async () => {
      lengthEvents.push('read');
      const id = lengthEvents.at(-2);
      return id === 'length' ? '|' : id === 'item:0' ? 'one' : 'two';
    });
    assert.equal(joinedLength.value, expected);
    const expectedEvents = ['length', 'read'];
    for (const index of expectedIndices) expectedEvents.push(`item:${index}`, 'read');
    assert.deepEqual(lengthEvents, expectedEvents);
  }

  for (const invalidLength of [1n, Symbol('length'), Object(1n), { valueOf() { return 1n; } }]) {
    const invalidLengthEvents = [];
    const invalidLengthProxy = new Proxy(['one'], {
      get(target, property, receiver) {
        if (property === 'length') { invalidLengthEvents.push('length'); return invalidLength; }
        if (property === '0') invalidLengthEvents.push('item:0');
        return Reflect.get(target, property, receiver);
      },
    });
    await assert.rejects(() => joinNeuralese(invalidLengthProxy, Array.prototype.join, [softSeparator], async () => {
      invalidLengthEvents.push('separator-read'); return '|';
    }), TypeError);
    assert.deepEqual(invalidLengthEvents, ['length'], 'ToLength errors happen before separator or indexed gets');
  }

  const coercionOrder = [];
  const customJoined = (await joinNeuralese([{ toString() { coercionOrder.push('object-before'); return 'before'; } },
    neuraleseRef('Neuralese<number>', 'nz1_eeeeeeeeeeeeeeeeeeee'),
    { toString() { coercionOrder.push('object-after'); return 'after'; } }], Array.prototype.join, ['|'], async () => {
    coercionOrder.push('typed-read'); return 7;
  })).value;
  assert.equal(customJoined, 'before|7|after');
  assert.deepEqual(coercionOrder, ['object-before', 'typed-read', 'object-after']);
  const nestedCycle = [];
  nestedCycle.push(nestedCycle);
  assert.equal((await joinNeuralese([neuraleseRef('Neuralese<string>', 'nz1_ffffffffffffffffffff'), ['nested', 'array'], nestedCycle],
    Array.prototype.join, ['|'], async () => 'soft')).value, 'soft|nested,array|');
  const customEvents = [];
  const customArray = [neuraleseRef('Neuralese<string>', 'nz1_gggggggggggggggggggg')];
  const customMethod = function (...args) {
    customEvents.push(`call:${this === customArray}:${args.join('|')}`);
    return Promise.resolve('custom result');
  };
  customArray.join = customMethod;
  const customValue = await joinNeuralese(customArray, customMethod, ['separator', 'extra'], async () => {
    customEvents.push('unexpected read'); return 'payload';
  });
  assert.ok(customValue.value instanceof Promise, 'a custom join thenable is not assimilated by async readout');
  assert.deepEqual(await customValue.value, 'custom result');
  assert.deepEqual(customEvents, ['call:true:separator|extra']);

  let hasCount = 0;
  let lengthReads = 0;
  const indexReads = [];
  const inherited = Object.create(Array.prototype);
  inherited[1] = 'inherited';
  const proxy = new Proxy(Object.setPrototypeOf(['first', , 'last'], inherited), {
    has(target, property) { hasCount++; return Reflect.has(target, property); },
    get(target, property, receiver) {
      if (property === 'length') lengthReads++;
      if (property === '0' || property === '1' || property === '2') indexReads.push(property);
      return Reflect.get(target, property, receiver);
    },
  });
  assert.equal((await joinNeuralese(proxy, Array.prototype.join, ['|'], async value => value)).value, 'first|inherited|last');
  assert.equal(hasCount, 0, 'native join reads indexed values without a separate has trap');
  assert.equal(lengthReads, 1);
  assert.deepEqual(indexReads, ['0', '1', '2']);
});

test('Array#toString preserves native and custom join dispatch while reading only direct soft elements', async () => {
  const inherited = Object.create(Array.prototype);
  inherited[1] = neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr');
  const events = [];
  const receiver = new Proxy(Object.setPrototypeOf(['first', , 'last'], inherited), {
    get(target, property, receiver) {
      if (property === 'join') events.push('join');
      if (property === 'length') events.push('length');
      if (property === '0' || property === '1' || property === '2') events.push(`item:${property}`);
      return Reflect.get(target, property, receiver);
    },
  });
  const output = await arrayToStringNeuralese(receiver, Array.prototype.toString, [], async ref => {
    events.push(`read:${ref.$neuralese.id}`); return 'inherited-soft';
  });
  assert.equal(output.value, 'first,inherited-soft,last');
  assert.deepEqual(events, ['join', 'length', 'item:0', 'item:1',
    'read:nz1_rrrrrrrrrrrrrrrrrrrr', 'item:2']);

  const thenable = Promise.resolve('custom join output');
  const customJoinEvents = [];
  const customJoinArray = [neuraleseRef('Neuralese<string>', 'nz1_ssssssssssssssssssss')];
  Object.defineProperty(customJoinArray, 'join', { get() {
    customJoinEvents.push('join-get');
    return function (...args) {
      customJoinEvents.push(`join-call:${this === customJoinArray}:${args.length}`);
      return thenable;
    };
  } });
  const customJoinResult = await arrayToStringNeuralese(customJoinArray, Array.prototype.toString, [], async () => {
    customJoinEvents.push('unexpected-read'); return 'payload';
  });
  assert.equal(customJoinResult.value, thenable, 'a custom join thenable is not assimilated by the helper');
  assert.deepEqual(customJoinEvents, ['join-get', 'join-call:true:0']);

  const nonCallableJoin = [neuraleseRef('Neuralese<string>', 'nz1_tttttttttttttttttttt')];
  nonCallableJoin.join = 3;
  assert.equal((await arrayToStringNeuralese(nonCallableJoin, Array.prototype.toString, [], async () => {
    assert.fail('fallback object tag does not read array elements');
  })).value, '[object Array]');

  const nestedReads = [];
  const nestedSoft = neuraleseRef('Neuralese<string>', 'nz1_vvvvvvvvvvvvvvvvvvvv');
  const nestedOutput = await arrayToStringNeuralese([[nestedSoft], nestedSoft], Array.prototype.toString, [], async ref => {
    nestedReads.push(ref.$neuralese.id); return 'direct soft';
  });
  assert.equal(nestedOutput.value, '[object Object],direct soft');
  assert.deepEqual(nestedReads, ['nz1_vvvvvvvvvvvvvvvvvvvv'], 'nested arrays retain native opaque stringification');

  const getterErrorEvents = [];
  const throwingJoinGetter = [1];
  Object.defineProperty(throwingJoinGetter, 'join', { get() {
    getterErrorEvents.push('join-get'); throw new Error('join getter failed');
  } });
  await assert.rejects(() => arrayToStringNeuralese(throwingJoinGetter, Array.prototype.toString, [], async () => 'unused'),
    /join getter failed/);
  assert.deepEqual(getterErrorEvents, ['join-get']);

  const customToStringPromise = Promise.resolve('custom toString');
  const customToStringReceiver = [neuraleseRef('Neuralese<string>', 'nz1_uuuuuuuuuuuuuuuuuuuu')];
  const customToString = function (value) {
    assert.equal(this, customToStringReceiver);
    assert.equal(value, 'argument');
    return customToStringPromise;
  };
  assert.equal((await arrayToStringNeuralese(customToStringReceiver, customToString, ['argument'], async () => {
    assert.fail('custom toString does not imply typed element readout');
  })).value, customToStringPromise);
  const throwingToString = () => { throw new Error('custom toString failed'); };
  await assert.rejects(() => arrayToStringNeuralese(customToStringReceiver, throwingToString, [], async () => {
    assert.fail('failed custom method does not imply typed element readout');
  }), /custom toString failed/);
  await assert.rejects(() => arrayToStringNeuralese(customToStringReceiver, Array.prototype.toString, [], async () => {
    throw new Error('typed element unavailable');
  }), /typed element unavailable/);
});

test('the default Error constructor reads a typed message after evaluating its arguments', async () => {
  const scope = { ...TEXT_SCOPE, inputs: [
    { name: 'text', type: 'Neuralese<string> | string' }, { name: 'events', type: 'string[]' }], returns: 'string' };
  for (const source of [
    'return new Error((events.push("message"), text), (events.push("options"), { cause: "cause" })).message;',
    'return Error((events.push("message"), text), (events.push("options"), { cause: "cause" })).message;',
  ]) {
    const analysis = analyzeEvalSnippet(source, scope);
    assert.deepEqual(analysis.diagnostics, []);
    assert.deepEqual(analysis.readouts.map(item => [item.kind, item.conditional]), [['error', true]]);
    const compiled = compileScopeSnippet(source, { inputBindings: ['text', 'events'], neuralese: true,
      analyze: text => analyzeEvalSnippet(text, scope) });
    const events = [];
    const run = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
      `${compiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
      output => output.result, { readNeuraleseIfReference: async value => {
        if (!isNeuraleseRef(value)) return value;
        events.push('read'); return 'resolved message';
      } });
    const message = await run({ text: neuraleseRef('Neuralese<string>', 'nz1_aaaaaaaaaaaaaaaaaaaa'), events }, {}, {});
    assert.equal(message, 'resolved message');
    assert.deepEqual(events, ['message', 'options', 'read']);
    events.length = 0;
    assert.equal(await run({ text: 'crisp message', events }, {}, {}), 'crisp message');
    assert.deepEqual(events, ['message', 'options']);
  }

  const module = compileModule({ kind: 'module', id: 'error-soft-message', name: 'errorMessage', source: 'errorMessage.ts', revision: 'r1',
    text: `export async function show(text: Neuralese<string> | string, events: string[]) {
      return new Error((events.push('message'), text), (events.push('options'), { cause: 'cause' })).message;
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const exports = {};
  const moduleFns = new Function('exports', '__natlang', `${module}; return exports;`)(exports, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      return 'module message';
    },
  });
  const events = [];
  assert.equal(await moduleFns.show(neuraleseRef('Neuralese<string>', 'nz1_bbbbbbbbbbbbbbbbbbbb'), events), 'module message');
  assert.deepEqual(events, ['message', 'options']);
});

test('standard String text arguments read soft values after the native call arguments', async () => {
  const scope = { types: {}, inputs: [
    { name: 'label', type: 'string' }, { name: 'query', type: 'Neuralese<string> | string' },
    { name: 'padding', type: 'Neuralese<string> | string' }, { name: 'events', type: 'string[]' },
    { name: 'service', type: '{ includes(value: Neuralese<string>): boolean }' },
  ], locals: [], captures: [], imports: [], returns: 'string' };
  const source = `return String((events.push('receiver'), label).includes((events.push('query'), query),
    (events.push('position'), 1)));`;
  const analysis = analyzeEvalSnippet(source, scope);
  assert.deepEqual(analysis.diagnostics, []);
  assert.deepEqual(analysis.readouts.map(item => [item.kind, item.argument, item.conditional]), [['string-argument', 0, true]]);
  const custom = analyzeEvalSnippet('return String(service.includes(query));', scope);
  assert.deepEqual(custom.diagnostics, []);
  assert.deepEqual(custom.readouts, [], 'a same-named user service method remains an ordinary function contract');
  const compiled = compileScopeSnippet(source, { inputBindings: ['label', 'query', 'events'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, scope) });
  const events = [];
  const run = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${compiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      events.push('read'); return 'ello';
    } });
  assert.equal(await run({ label: 'hello', query: neuraleseRef('Neuralese<string>', 'nz1_cccccccccccccccccccc'), events }, {}, {}), 'true');
  assert.deepEqual(events, ['receiver', 'query', 'position', 'read']);
  events.length = 0;
  assert.equal(await run({ label: 'hello', query: 'ello', events }, {}, {}), 'true');
  assert.deepEqual(events, ['receiver', 'query', 'position']);

  const module = compileModule({ kind: 'module', id: 'string-soft-args', name: 'stringArgs', source: 'stringArgs.ts', revision: 'r1',
    text: `export async function search(label: string, query: Neuralese<string> | string, events: string[]) {
      return (events.push('receiver'), label).includes((events.push('query'), query), (events.push('position'), 1));
    }
    export async function pad(padding: Neuralese<string> | string, events: string[]) {
      return 'x'.padStart((events.push('length'), 3), (events.push('padding'), padding));
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const moduleEvents = [];
  const moduleFns = new Function('exports', '__natlang', `${module}; return exports;`)({}, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      moduleEvents.push('read'); return value.$neuralese.id.endsWith('d') ? 'ello' : '*';
    },
  });
  assert.equal(await moduleFns.search('hello', neuraleseRef('Neuralese<string>', 'nz1_dddddddddddddddddddd'), moduleEvents), true);
  assert.deepEqual(moduleEvents, ['receiver', 'query', 'position', 'read']);
  moduleEvents.length = 0;
  assert.equal(await moduleFns.pad(neuraleseRef('Neuralese<string>', 'nz1_eeeeeeeeeeeeeeeeeeee'), moduleEvents), '**x');
  assert.deepEqual(moduleEvents, ['length', 'padding', 'read']);
});

test('soft string replace reads the receiver before native method lookup and string arguments', async () => {
  const scope = { types: {}, inputs: [
    { name: 'priorNotes', type: 'Neuralese<string> | string' }, { name: 'events', type: 'string[]' },
  ], locals: [], captures: [], imports: [], returns: 'string' };
  const source = `return ((events.push('receiver'), priorNotes)).replace(
    (events.push('search'), 'old'), (events.push('replacement'), 'new'));`;
  const analysis = analyzeEvalSnippet(source, scope);
  assert.deepEqual(analysis.diagnostics, []);
  assert.deepEqual(analysis.readouts.map(item => [item.kind, item.conditional]), [['string-replace', true]]);
  const compiled = compileScopeSnippet(source, { inputBindings: ['priorNotes', 'events'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, scope) });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));

  const events = [];
  let failRead = false;
  const run = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${compiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuraleseIfReference: async value => {
      if (!isNeuraleseRef(value)) return value;
      events.push('read');
      if (failRead) throw new Error('replace read failed');
      return 'old old';
    }, invokeWithReceiver: (method, receiver, args) => Reflect.apply(method, receiver, args) });

  const original = Object.getOwnPropertyDescriptor(String.prototype, 'replace');
  try {
    Object.defineProperty(String.prototype, 'replace', { configurable: true, get() {
      events.push('lookup');
      return function (...args) {
        events.push('invoke');
        return Reflect.apply(original.value, this, args);
      };
    } });
    const softResult = await run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr'), events }, {}, {});
    assert.equal(softResult, 'new old');
    assert.deepEqual(events, ['receiver', 'read', 'lookup', 'search', 'replacement', 'invoke']);

    events.length = 0;
    const crispResult = await run({ priorNotes: 'old old', events }, {}, {});
    assert.equal(crispResult, 'new old');
    assert.deepEqual(events, ['receiver', 'lookup', 'search', 'replacement', 'invoke']);

    events.length = 0;
    failRead = true;
    await assert.rejects(() => run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr'), events }, {}, {}),
      /replace read failed/);
    assert.deepEqual(events, ['receiver', 'read'], 'a failed read occurs before native lookup or argument evaluation');
    failRead = false;

    events.length = 0;
    Object.defineProperty(String.prototype, 'replace', { configurable: true, get() {
      events.push('lookup'); throw new Error('replace getter failed');
    } });
    await assert.rejects(() => run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr'), events }, {}, {}),
      /replace getter failed/);
    assert.deepEqual(events, ['receiver', 'read', 'lookup'], 'a throwing prototype getter prevents argument evaluation');

    events.length = 0;
    Object.defineProperty(String.prototype, 'replace', { configurable: true, get() {
      events.push('lookup'); return 17;
    } });
    await assert.rejects(() => run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr'), events }, {}, {}), TypeError);
    assert.deepEqual(events, ['receiver', 'read', 'lookup', 'search', 'replacement'],
      'native non-callable method failure occurs after arguments evaluate');

    events.length = 0;
    Object.defineProperty(String.prototype, 'replace', { configurable: true, get() {
      events.push('lookup'); return function () { events.push('invoke'); throw new Error('replace method failed'); };
    } });
    await assert.rejects(() => run({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr'), events }, {}, {}),
      /replace method failed/);
    assert.deepEqual(events, ['receiver', 'read', 'lookup', 'search', 'replacement', 'invoke']);
  } finally {
    Object.defineProperty(String.prototype, 'replace', original);
  }

  const pureScope = { ...scope, inputs: [{ name: 'priorNotes', type: 'Neuralese<string>' }] };
  const pureSource = `return priorNotes.replace('old', 'new');`;
  const pureAnalysis = analyzeEvalSnippet(pureSource, pureScope);
  assert.deepEqual(pureAnalysis.diagnostics, []);
  assert.deepEqual(pureAnalysis.readouts.map(item => [item.kind, item.conditional]), [['string-replace', undefined]]);
  const pureCompiled = compileScopeSnippet(pureSource, { inputBindings: ['priorNotes'], neuralese: true,
    analyze: text => analyzeEvalSnippet(text, pureScope) });
  const pureRun = new Function('__natlang_frozen', '__natlang_copy', '__natlang_settle', '__natlang_output', '__live',
    `${pureCompiled.program}; return __natlang_scope;`)(value => value, value => value, async value => value,
    output => output.result, { readNeuralese: async () => 'old notes',
      invokeWithReceiver: (method, receiver, args) => Reflect.apply(method, receiver, args) });
  assert.equal(await pureRun({ priorNotes: neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr') }, {}, {}), 'new notes');

  const module = compileModule({ kind: 'module', id: 'soft-string-replace', name: 'replaceNotes', source: 'replaceNotes.ts', revision: 'r1',
    text: `export async function replaceNotes(priorNotes: Neuralese<string> | string) {
      return priorNotes.replace('old', 'new');
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  const moduleFns = new Function('exports', '__natlang', `${module}; return exports;`)({}, {
    guard: (_id, fn) => fn(), readNeuraleseIfReference: async value => isNeuraleseRef(value) ? 'old notes' : value,
    invokeWithReceiver: (method, receiver, args) => Reflect.apply(method, receiver, args),
  });
  assert.equal(await moduleFns.replaceNotes(neuraleseRef('Neuralese<string>', 'nz1_rrrrrrrrrrrrrrrrrrrr')), 'new notes');

  const sync = analyzeEvalSnippet('function syncReplace(priorNotes: Neuralese<string>) { return priorNotes.replace("old", "new"); }',
    { ...scope, returns: 'string' });
  assert.ok(sync.diagnostics.some(item => item.code === 'neuralese-readout-sync'),
    'a synchronous function cannot receive an injected asynchronous read');
  const regex = analyzeEvalSnippet('priorNotes.replace(/old/, "new");', scope);
  assert.ok(regex.diagnostics.some(item => item.code === 'neuralese-opaque-access'),
    'RegExp search overloads remain outside the string-only path');
  const callback = analyzeEvalSnippet('priorNotes.replace("old", value => value);', scope);
  assert.ok(callback.diagnostics.some(item => item.code === 'neuralese-opaque-access'),
    'callback replacement overloads remain outside the string-only path');
});

test('scope lowering awaits the typed readout at the original coercion site', () => {
  const compiled = compileScopeSnippet('`plan=${plan};`', { inputBindings: ['plan'], neuralese: true,
    analyze: source => analyzeEvalSnippet(source, SCOPE) });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.match(compiled.program, /`plan=\$\{\(await __live\.readNeuralese\(\(plan\)\)\)\};`/);
  const textMethod = compileScopeSnippet('return text.toString();', { inputBindings: ['text'], neuralese: true,
    analyze: source => analyzeText(source) });
  assert.equal(textMethod.ok, true, JSON.stringify(textMethod.diagnostics));
  assert.match(textMethod.program, /\(await __live\.readNeuralese\(\(text\)\)\)\.toString\(\)/);
});

test('project module lowering uses the same typed async readout path', () => {
  const compiled = compileModule({ kind: 'module', id: 'module-1', name: 'view', source: 'view.ts', revision: 'r1',
    text: 'export async function show(plan: Neuralese<string>) { return `plan=${plan}`; }',
    types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(compiled, /`plan=\$\{await __natlang\.readNeuralese\(plan\)\}`/);
  const appended = compileModule({ kind: 'module', id: 'module-2', name: 'append', source: 'append.ts', revision: 'r1',
    text: 'export async function show(text: string, plan: Neuralese<string>) { text += plan; return text; }',
    types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(appended, /text \+= await __natlang\.readNeuralese\(plan\)/);
});

test('implicit readout without a task standard library raises a coded capability error', async () => {
  const ref = neuraleseRef('Neuralese<string>', `nz1_${'a'.repeat(52)}`);
  await assert.rejects(() => readNeuraleseForCurrentTask(ref), error =>
    error.code === 'neuralese-readout-unavailable' && error.capability === 'typed-readout' &&
    /services\.neuralese is missing or has no read body/.test(error.message));
});

test('a model-written literal takes its type from context, and an untyped one is rejected', () => {
  const id = `nz1_${'a'.repeat(52)}`;
  const typed = analyzeEvalSnippet(`const next: Neuralese<Plan> = __neuralese(${JSON.stringify(id)}); return next;`, SCOPE);
  assert.deepEqual(typed.diagnostics, []);
  assert.deepEqual(typed.neuralese.map(literal => [literal.id, literal.type]), [[id, 'Neuralese<Plan>']]);
  const returned = analyzeEvalSnippet(`return __neuralese(${JSON.stringify(id)});`, SCOPE);
  assert.deepEqual(returned.neuralese.map(literal => literal.type), ['Neuralese<Plan>']);
  const untyped = analyzeEvalSnippet(`const loose = __neuralese(${JSON.stringify(id)});`, SCOPE);
  assert.deepEqual(untyped.diagnostics.map(item => item.code), ['neuralese-untyped-literal']);
});

const neuraleseDriver = fn => Object.assign(fn, { neuralese: true });

test('a call writes a soft result through the port and the runtime keeps only a reference', async () => {
  const { store, port } = standIn();
  const { session } = open({ type: '() => Neuralese<string>', instructions: 'Write a note.' });
  const driver = neuraleseDriver(({ messages }) => messages.length <= 2 ?
    { calls: [['eval', { code: 'const note: Neuralese<string> = <|neuralese|>check the totals first<|/neuralese|>;\nreturn note;' }]] } :
    { calls: [['return_result', { status: 'success' }]] });
  await new NativeToolAgent(driver, { maxTurns: 4, neuralese: { store, port } }).run(session);
  assert.ok(isNeuraleseRef(session.lam.return), JSON.stringify(session.lam.return));
  assert.equal(session.lam.return.$neuralese.type, 'Neuralese<string>');
  const meta = await store.meta(session.lam.return.$neuralese.id);
  assert.equal(meta.length, 4);
});

test('eval can read code and built-in documentation through the same authorized lookup as read_code tool', async () => {
  const session = open({ type: '() => string', instructions: 'Inspect a built-in.' }).session;
  const result = await session.applyAsync('eval', { code: "const docs = read_code('iterateOn'); console.log(docs); docs;" });
  assert.equal(result.kind, 'ok', result.text);
  assert.match(result.text, /iterateOn/);
  assert.match(result.text, /stopping check/);
  const event = session.runtime.trace.events.find(item => item.kind === 'eval_code_read');
  assert.equal(event.name, 'iterateOn');
  assert.equal(event.outcome, 'ok');
  assert.match(event.text_sha256, /^[0-9a-f]{64}$/);

  const shadowed = await session.applyAsync('eval', { code: "const read_code = 'local value'; read_code;" });
  assert.equal(shadowed.kind, 'ok', shadowed.text);
  assert.match(shadowed.text, /local value/);
});

test('validated JSON soft results write the same typed body as an equivalent Neuralese marker', async () => {
  const type = 'Neuralese<{ count: number, flag: boolean }>';
  const body = '{"count":4,"flag":true}';
  const captureWriter = () => {
    const { store, port } = standIn();
    const writes = [], write = port.write.bind(port);
    port.write = async (text, options) => { writes.push({ text, options }); return write(text, options); };
    return { store, port, writes };
  };

  const direct = captureWriter();
  const directSession = open({ type: `() => ${type}`, instructions: 'Return the typed record.' },
    { neuralese: { store: direct.store, port: direct.port } }).session;
  const directResult = await directSession.applyAsync('return_result', { status: 'success', value: { flag: true, count: 4 } });
  assert.equal(directResult.kind, 'completed', directResult.text);
  assert.ok(isNeuraleseRef(directSession.lam.return));
  assert.equal(directSession.lam.return.$neuralese.type, type);
  assert.equal(direct.writes[0].text, body, 'typed field order gives a stable canonical JSON body');
  assert.equal(direct.writes[0].options.type, type);
  assert.equal(direct.writes[0].options.producer.source_kind, 'typed-json-result');
  assert.equal(direct.writes[0].options.producer.source, 'return_result');
  assert.equal(direct.writes[0].options.producer.marker_context, 'return-result');
  assert.equal(direct.writes[0].options.producer.result_type, type);
  assert.equal(direct.writes[0].options.producer.text_body_sha256, hexDigest(body));
  assert.equal(directSession.runtime.trace.events.find(event => event.kind === 'block_write').source_kind,
    'typed-json-result');

  const marker = captureWriter();
  const markerSession = open({ type: `() => ${type}`, instructions: 'Return the typed record.' },
    { neuralese: { store: marker.store, port: marker.port } }).session;
  const driver = neuraleseDriver(() => ({ calls: [['return_result', { status: 'success',
    value: `<|neuralese|>${body}<|/neuralese|>` }]] }));
  await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store: marker.store, port: marker.port } }).run(markerSession);
  assert.ok(isNeuraleseRef(markerSession.lam.return));
  assert.equal(markerSession.lam.return.$neuralese.type, type);
  assert.equal(marker.writes[0].text, direct.writes[0].text);
  assert.equal(markerSession.lam.return.$neuralese.id, directSession.lam.return.$neuralese.id,
    'the same port and body produce the same block irrespective of syntax');
  assert.equal(marker.writes[0].options.producer.result_type, type);
});

test('JSON soft-result promotion validates complete elements and preserves exact JSON number values', async () => {
  const { store, port } = standIn();
  const writes = [], write = port.write.bind(port);
  port.write = async (text, options) => { writes.push({ text, options }); return write(text, options); };
  const record = open({ type: '() => Neuralese<{ count: number; flag: boolean }>', instructions: 'Return the record.' },
    { neuralese: { store, port } }).session;
  const incomplete = await record.applyAsync('return_result', { status: 'success', value: { count: 4 } });
  assert.equal(incomplete.kind, 'rejected');
  assert.match(incomplete.text, /flag/);
  assert.equal(writes.length, 0, 'invalid data has no writer side effect');

  const opaque = open({ type: '() => Neuralese<unknown>', instructions: 'Return the soft value.' },
    { neuralese: { store, port } }).session;
  const refused = await opaque.applyAsync('return_result', { status: 'success', value: { count: 4 } });
  assert.equal(refused.kind, 'rejected');
  assert.equal(writes.length, 0, 'erased element types are not inferred from the supplied object');

  for (const [element, value, expectedText] of [['number', 42, '42'], ['boolean', true, 'true']]) {
    const scalar = open({ type: `() => Neuralese<${element}>`, instructions: 'Return the typed value.' },
      { neuralese: { store, port } }).session;
    const result = await scalar.applyAsync('return_result', { status: 'success', value });
    assert.equal(result.kind, 'completed', result.text);
    assert.equal(scalar.lam.return.$neuralese.type, `Neuralese<${element}>`);
    assert.equal(writes.at(-1).text, expectedText);
  }

  const jsonText = open({ type: '() => Neuralese<{ count: number, flag: boolean }>', instructions: 'Return the typed record.' },
    { neuralese: { store, port } }).session;
  const parsed = await jsonText.applyAsync('return_result', { status: 'success', value: '{"flag":true,"count":4}' });
  assert.equal(parsed.kind, 'completed', parsed.text);
  assert.equal(writes.at(-1).text, '{"count":4,"flag":true}', 'JSON fallback validates and canonicalizes the same declared shape');

  const nestedUnion = open({ type: '() => Neuralese<{ payload: { a: string, b: number } | { a: string } }>',
    instructions: 'Return the typed record.' }, { neuralese: { store, port } }).session;
  const unionResult = await nestedUnion.applyAsync('return_result', { status: 'success', value: { payload: { a: 'x' } } });
  assert.equal(unionResult.kind, 'completed', unionResult.text);
  assert.equal(writes.at(-1).text, '{"payload":{"a":"x"}}', 'a complete later union member remains available');

  const negativeZero = open({ type: '() => Neuralese<number>', instructions: 'Return a finite number.' },
    { neuralese: { store, port } }).session;
  const zeroResult = await negativeZero.applyAsync('return_result', { status: 'success', value: -0 });
  assert.equal(zeroResult.kind, 'completed', zeroResult.text);
  assert.equal(writes.at(-1).text, '-0');
  assert.ok(Object.is(JSON.parse(writes.at(-1).text), -0));

  const nestedNegativeZero = open({ type: '() => Neuralese<{ values: number[]; nested: { value: number } }>',
    instructions: 'Return the typed record.' }, { neuralese: { store, port } }).session;
  const nestedResult = await nestedNegativeZero.applyAsync('return_result', { status: 'success',
    value: { values: [-0, 2], nested: { value: -0 } } });
  assert.equal(nestedResult.kind, 'completed', nestedResult.text);
  assert.equal(writes.at(-1).text, '{"nested":{"value":-0},"values":[-0,2]}');
  const roundTrip = JSON.parse(writes.at(-1).text);
  assert.ok(Object.is(roundTrip.values[0], -0));
  assert.ok(Object.is(roundTrip.nested.value, -0));

  const nonFinite = open({ type: '() => Neuralese<number>', instructions: 'Return a finite number.' },
    { neuralese: { store, port } }).session;
  const beforeNonFinite = writes.length;
  const nonFiniteResult = await nonFinite.applyAsync('return_result', { status: 'success', value: Number.POSITIVE_INFINITY });
  assert.equal(nonFiniteResult.kind, 'rejected');
  assert.equal(writes.length, beforeNonFinite, 'non-finite numbers are rejected before writing');
});

test('structured final results write only validated plain text at declared Neuralese<string> leaves', async () => {
  const { store, port } = standIn();
  const writes = [], write = port.write.bind(port);
  port.write = async (text, options) => { writes.push({ text, options }); return write(text, options); };
  const existing = await port.write('keep this existing block', { type: 'Neuralese<string>' });
  const resultType = '() => { existing: Neuralese<string>; nested: { second: Neuralese<string>; marker: Neuralese<string> }; final: Neuralese<string>; count: number }';
  const value = { existing: neuraleseRef('Neuralese<string>', existing.id),
    nested: { second: 'second note', marker: neuraleseSentinel(existing.id) }, final: 'final note', count: 2 };
  const { session } = open({ type: resultType, instructions: 'Return the complete report.' }, { neuralese: { store, port } });
  const completed = await session.applyAsync('return_result', { status: 'success', value });
  assert.equal(completed.kind, 'completed', completed.text);
  assert.equal(writes.length, 3, 'the existing reference and marker do not invoke the writer again');
  assert.deepEqual(writes.slice(1).map(item => item.text), ['second note', 'final note']);
  assert.equal(session.lam.return.existing.$neuralese.id, existing.id);
  assert.equal(session.lam.return.nested.marker.$neuralese.id, existing.id);
  assert.equal(session.lam.return.nested.second.$neuralese.type, 'Neuralese<string>');
  assert.equal(session.lam.return.final.$neuralese.type, 'Neuralese<string>');
  assert.deepEqual(writes.slice(1).map(item => item.options.producer.result_path), [
    ['return', 'nested', 'second'], ['return', 'final'],
  ]);
  assert.deepEqual(session.runtime.trace.events.filter(item => item.kind === 'block_write' && item.source_kind === 'typed-text-result-field')
    .map(item => item.result_path), [['return', 'nested', 'second'], ['return', 'final']]);

  const beforeInvalid = writes.length;
  const invalid = open({ type: '() => { note: Neuralese<string>; count: number }', instructions: 'Return the report.' },
    { neuralese: { store, port } }).session;
  const rejected = await invalid.applyAsync('return_result', { status: 'success', value: { note: 'do not write yet', count: 'two' } });
  assert.equal(rejected.kind, 'rejected');
  assert.equal(writes.length, beforeInvalid, 'the entire declared shape is checked before the first leaf write');

  const collectionType = '() => { choice: Neuralese<string> | null; sequence: Neuralese<string>[]; byName: Record<string, Neuralese<string>> }';
  const collection = open({ type: collectionType, instructions: 'Return the report.' }, { neuralese: { store, port } });
  const collectionResult = await collection.session.applyAsync('return_result', { status: 'success', value: {
    choice: 'union note', sequence: ['first item', 'second item'], byName: { alpha: 'alpha note', beta: 'beta note' },
  } });
  assert.equal(collectionResult.kind, 'completed', collectionResult.text);
  assert.deepEqual(writes.slice(beforeInvalid).map(item => item.text),
    ['union note', 'first item', 'second item', 'alpha note', 'beta note']);
  assert.deepEqual(writes.slice(beforeInvalid).map(item => item.options.producer.result_path), [
    ['return', 'choice'], ['return', 'sequence', 0], ['return', 'sequence', 1],
    ['return', 'byName', 'alpha'], ['return', 'byName', 'beta'],
  ]);
  assert.ok(isNeuraleseRef(collection.session.lam.return.choice));
  assert.ok(collection.session.lam.return.sequence.every(isNeuraleseRef));
  assert.ok(Object.values(collection.session.lam.return.byName).every(isNeuraleseRef));
});

test('nested soft-text result normalization is shared by eval return, eval finish, and return_result in eval', async () => {
  const resultType = '() => { note: Neuralese<string> }';
  for (const [code, finish, expected, completed] of [
    ['return { note: "eval return" };', false, 'eval return', false],
    ['return { note: "eval finish" };', true, 'eval finish', true],
    ['return_result({ status: "success", value: { note: "eval tool" } });', false, 'eval tool', false],
  ]) {
    const { store, port } = standIn();
    const writes = [], write = port.write.bind(port);
    port.write = async (text, options) => { writes.push({ text, options }); return write(text, options); };
    const { session } = open({ type: resultType, instructions: 'Return the report.' }, { neuralese: { store, port } });
    const result = await session.applyAsync('eval', { code, finish });
    assert.equal(result.kind, completed ? 'completed' : 'ok', result.text);
    assert.equal(session.lam.return.note.$neuralese.type, 'Neuralese<string>');
    assert.equal(writes[0].text, expected);
  }
});

test('soft inputs are shown as blocks, and a backend without Neuralese support fails instead of falling back to text', async () => {
  const { store, port } = standIn();
  const block = await port.write('the customer prefers email');
  const input = neuraleseRef('Neuralese<string>', block.id);
  const body = { type: '(memo: Neuralese<string>) => string', instructions: 'Answer from memo.', args: { memo: input } };
  const seen = [];
  const driver = neuraleseDriver(({ messages }) => {
    seen.push(messages);
    return { calls: [['return_result', { status: 'success', value: 'email' }]] };
  });
  const { session } = open(body);
  await new NativeToolAgent(driver, { maxTurns: 2, neuralese: { store, port } }).run(session);
  assert.equal(session.lam.return, 'email');
  const parts = seen[0].flatMap(message => [message.content, ...(message.tool_calls ?? []).map(call => call.function.arguments)])
    .filter(Array.isArray).flat();
  assert.ok(parts.some(part => part.type === 'neuralese' && part.id === block.id), 'the opening carries the block as a part');
  assert.ok(!JSON.stringify(seen[0]).includes(''), 'no internal marker reaches the backend');
  const plain = open(body).session;
  await assert.rejects(() => new NativeToolAgent(() => ({ calls: [] }), { maxTurns: 2 }).run(plain), /neuralese-unsupported-backend/);
});

test('iteration needs a hard bound only for TypeScript predicates or with review off; nl predicates get the stopping prompt', async () => {
  const prompts = [];
  let checks = 0;
  const driver = ({ messages }) => {
    prompts.push(String(messages[0].content));
    checks++;
    return { calls: [['return_result', { status: 'success', value: checks >= 3 }]] };
  };
  const runtime = createNatlangRuntime({ model: driver });
  const atLeast = defineNatlang('---\nargs:\n  n: number\nreturns: boolean\n---\nIs n at least 2?\n', { name: 'atLeast' });
  assert.equal(await runtime.run(() => iterateOn(n => n + 1, 0).until(atLeast)), 2);
  assert.equal(prompts.length, 3);
  assert.match(prompts[0], /stopping condition of an iterative loop/);
  assert.match(prompts[0], /initial state, before any step/);
  assert.match(prompts[2], /completed 2 steps\. The last step changed the state/);
  await assert.rejects(() => runtime.run(() => iterateOn(n => n + 1, 0).until(n => n > 5)), error =>
    error instanceof IterationLimitError && /TypeScript stopping predicate requires withLimit/.test(error.message));
  assert.equal(await runtime.run(() => iterateOn(n => n + 1, 0).withLimit({ maxSteps: 10 }).until(n => n > 5)), 6);
  checks = -100;
  await assert.rejects(() => runtime.run(() => iterateOn(n => n + 1, 0).checkProgress('off').until(atLeast)),
    /progress review off requires withLimit/);
});
