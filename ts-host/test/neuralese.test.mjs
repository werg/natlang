import assert from 'node:assert/strict';
import { test } from 'node:test';
import { parseType, formatType, fitsType, TypeEnv, TypeSyntaxError } from '../dist/native/types.js';
import { coerce, Reject } from '../dist/native/values.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder, neuraleseContentId } from '../dist/native/neuralese-store.js';
import { decodeTurnValue, encodeMessages, isNeuraleseRef, neuraleseRef, neuraleseSentinel, partsToText,
  sourceWithLiteralCalls, textToParts, writeLiterals } from '../dist/native/neuralese.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { compileScopeSnippet } from '../dist/scope-compiler.js';
import { compileModule } from '../dist/runtime/modules.js';
import { concatNeuralese, joinNeuralese } from '../dist/runtime/lowered.js';
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

const TEXT_SCOPE = { types: {}, inputs: [{ name: 'text', type: 'Neuralese<string>' },
  { name: 'numberText', type: 'Neuralese<number>' }, { name: 'nullText', type: 'Neuralese<null>' },
  { name: 'values', type: '(Neuralese<string> | Neuralese<null> | string | number | { toString(): string } | null | undefined)[]' }], locals: [], captures: [], imports: [], returns: 'string' };
const analyzeText = source => analyzeEvalSnippet(source, TEXT_SCOPE);
const textReadouts = source => analyzeText(source).readouts.map(item => source.slice(item.start, item.end));

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
  assert.deepEqual(codes('let text = ""; text += plan;'), []);
  assert.deepEqual(readouts('let text = ""; text += plan;'), ['plan']);
  assert.deepEqual(codes('function show() { return String(plan); }'), ['neuralese-readout-sync']);
  assert.deepEqual(codes('function show(String: (x: unknown) => string) { return String(plan); }'), []);
  assert.deepEqual(codes('const same = plan === plan;'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('JSON.stringify(plan)'), []);
  assert.deepEqual(readouts('JSON.stringify(plan, (_key, value) => value, 2)'), ['plan']);
  assert.deepEqual(codes('function show(JSON: any) { return JSON.stringify(plan); }'), []);
  assert.deepEqual(codes('const copy = { ...plan };'), ['neuralese-opaque-access']);
  assert.deepEqual(codes('const keep: Neuralese<Plan> = plan; return keep;'), []);
  assert.deepEqual(codes('let nested: Neuralese<Neuralese<Plan>>;'), ['neuralese-nested']);
  assert.deepEqual(codes('type F = (f: F) => string;'), ['type-recursive-function']);
  assert.deepEqual(codes('type Tree = { children: Tree[] };'), []);
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
  assert.deepEqual(analyzeText(`return ''.concat(...values);`).diagnostics.map(item => item.code), ['neuralese-opaque-access']);
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
  assert.match(compiled.program, /await __live\.joinNeuralese\(\(values\), ['"]\|['"], async/);

  const module = compileModule({ kind: 'module', id: 'module-join', name: 'joiner', source: 'joiner.ts', revision: 'r1',
    text: `export async function show(text: Neuralese<number>, values: (Neuralese<string> | string | number | { toString(): string } | null | undefined)[]) {
      return text.toString() + ''.concat(text) + values.join('|');
    }`, types: {}, exports: {}, imports: [], codebase: {} }, {});
  assert.match(module, /\(await __natlang\.readNeuralese\(text\)\)\.toString\(\)/);
  assert.match(module, /concatNeuralese/);
  assert.match(module, /await __natlang\.joinNeuralese\(values, ['"]\|['"], async/);

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

  const order = [];
  const items = ['label', 2, neuraleseRef('Neuralese<string>', 'nz1_aaaaaaaaaaaaaaaaaaaa'), , null,
    neuraleseRef('Neuralese<string>', 'nz1_bbbbbbbbbbbbbbbbbbbb'), neuraleseRef('Neuralese<null>', 'nz1_dddddddddddddddddddd')];
  const joined = await joinNeuralese(items, ':', async ref => {
    order.push(ref.$neuralese.id);
    if (ref.$neuralese.type === 'Neuralese<null>') return null;
    return ref.$neuralese.id.includes('aaaa') ? 'first' : 'last';
  });
  assert.equal(joined, 'label:2:first:::last:');
  assert.deepEqual(order, ['nz1_aaaaaaaaaaaaaaaaaaaa', 'nz1_bbbbbbbbbbbbbbbbbbbb', 'nz1_dddddddddddddddddddd']);
  const coercionOrder = [];
  const customJoined = await joinNeuralese([{ toString() { coercionOrder.push('object-before'); return 'before'; } },
    neuraleseRef('Neuralese<number>', 'nz1_eeeeeeeeeeeeeeeeeeee'),
    { toString() { coercionOrder.push('object-after'); return 'after'; } }], '|', async () => {
    coercionOrder.push('typed-read'); return 7;
  });
  assert.equal(customJoined, 'before|7|after');
  assert.deepEqual(coercionOrder, ['object-before', 'typed-read', 'object-after']);
  const nestedCycle = [];
  nestedCycle.push(nestedCycle);
  assert.equal(await joinNeuralese([neuraleseRef('Neuralese<string>', 'nz1_ffffffffffffffffffff'), ['nested', 'array'], nestedCycle],
    '|', async () => 'soft'), 'soft|nested,array|');
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
