// Higher-order semantic functions (plans/semantic-fp/01 L4): natlang functions take and return functions with checked
// signatures, a crisp function at a function-typed slot is checked at each call, and `partial` binds leading parameters.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import * as natlang from '../dist/index.js';
const { createNatlangRuntime, loadVirtualNatlang } = natlang;
const compileVirtualProject = project => natlang.compileVirtualProject(project, natlang);
import { fitsType, parseType, signatureOf, TypeEnv } from '../dist/native/types.js';
import { coerce } from '../dist/native/values.js';
import { callNotes } from '../dist/native/agent.js';
import { describeArguments, parseTargetArguments } from '../dist/runtime/target-arguments.js';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { browserTest, scriptedModel } from './support/natlang.mjs';

const scope = { types: { Ticket: '{ id: string, text: string }' },
  inputs: [{ name: 'tickets', type: 'Ticket[]' }], locals: [], captures: [], imports: [], returns: 'unknown' };
const shapes = code => {
  const { plans, diagnostics } = analyzeEvalSnippet(code, scope);
  return { codes: diagnostics.map(item => item.code),
    signatures: plans.map(plan => `(${plan.parameters.map(item => `${item.name}: ${item.type.natlang}`).join(', ')}) => ${plan.returns.natlang}`) };
};

test('an uncalled nl may be returned, stored in a function-typed field, or passed where a function is expected', () => {
  assert.deepEqual(shapes('function judgeFor(rubric: string): (t: Ticket) => Promise<boolean> { return nl`Does t meet ${rubric}?`; }\n' +
    'const t = tickets[0]!;\nreturn await judgeFor("urgent")(t);'), { codes: [], signatures: ['(t: Ticket) => boolean'] });
  assert.deepEqual(shapes('const judges: { urgent: (t: Ticket) => Promise<boolean> } = { urgent: nl`Is t urgent?` };\nreturn await judges.urgent(tickets[0]!);'),
    { codes: [], signatures: ['(t: Ticket) => boolean'] });
  assert.deepEqual(shapes('async function per(make: (rubric: string) => (t: Ticket) => Promise<boolean>) { return make("r")(tickets[0]!); }\n' +
    'return await per(rubric => nl`Does t meet ${rubric}?`);'), { codes: [], signatures: ['(t: Ticket) => boolean'] });
});

test('a factory without a declared result takes its signature from how the functions it makes are called', () => {
  assert.deepEqual(shapes('function judgeFor(rubric: string) { return nl`Does t meet ${rubric}?`; }\nconst t = tickets[0]!;\nif (await judgeFor("urgent")(t)) {}'),
    { codes: [], signatures: ['(t: Ticket) => boolean'] });
  assert.deepEqual(shapes('const judgeFor = (rubric: string) => nl`Does t meet ${rubric}?`;\nconst urgent = judgeFor("urgent");\n' +
    'const hits: string[] = [];\nfor (const t of tickets) if (await urgent(t)) hits.push(t.id);'), { codes: [], signatures: ['(t: Ticket) => boolean'] });
});

test('an uncalled nl that nothing can call is still the missing-call error', () => {
  assert.ok(shapes('const xs = await Promise.all(tickets.map(t => nl<string>`Solve this ticket.`));').codes.includes('nl-not-called'));
});

test('an asynchronous function type is a natlang function type; parameters match by position', () => {
  assert.deepEqual(shapes('const pick = nl<(judge: (t: Ticket) => Promise<boolean>, all: Ticket[]) => Ticket[]>`Keep the tickets judge accepts.`;\n' +
    'return await pick(async t => t.text.length > 3, tickets);').signatures, ['(judge: (t: Ticket) => boolean, all: (Ticket)[]) => (Ticket)[]']);
  assert.deepEqual(parseType('(t: string) => Promise<boolean>'), parseType('(t: string) => boolean'));
  const env = new TypeEnv();
  assert.ok(fitsType(parseType('(ticket: string) => boolean'), parseType('(t: string, extra: number) => boolean'), env));
  assert.ok(fitsType(parseType('(t: string | number) => "yes"'), parseType('(t: string) => string'), env));
  assert.ok(!fitsType(parseType('(t: string) => string'), parseType('(t: string | number) => string'), env));
  assert.ok(!fitsType(parseType('(t: string, n: number) => string'), parseType('(t: string) => string'), env));
});

test('a crisp function at a function-typed slot is checked at each call and keeps one identity per slot', async () => {
  const env = new TypeEnv(), type = parseType('(n: number) => string');
  const label = n => n > 1 ? 'many' : 1;
  const checked = coerce(label, type, env, 'args/label');
  assert.equal(coerce(label, type, env, 'args/label'), checked);
  assert.equal(await checked(2), 'many');
  await assert.rejects(checked(1), /label\/return: type-mismatch, expected string/);
  await assert.rejects(checked('2'), /label\/n: type-mismatch, expected number/);
  assert.equal(signatureOf(checked).type.returns.kind, 'prim');
  assert.equal(coerce(checked, type, env), checked, 'a checked function fits its own slot as it is');
});

const ROUTE = '---\nargs: { judge: "(t: Ticket) => boolean", tickets: Ticket[] }\nreturns: string[]\n---\nKeep the ids of the tickets judge accepts.\n';
const TYPES = 'export type Ticket = { id: string; text: string };\n';
const TICKETS = [{ id: 'a', text: 'server down' }, { id: 'b', text: 'thanks' }];
const keepAccepted = 'const ids: string[] = []; for (const t of tickets) if (await judge(t)) ids.push(t.id); return ids';

test('a .nl function takes a function argument: a natlang callable whose signature fits, or a checked crisp function', async () => {
  const model = scriptedModel(opening => /Keep the ids/.test(opening) ? keepAccepted : 'return t.text.includes("down")');
  const runtime = createNatlangRuntime({ model: model.driver });
  const route = loadVirtualNatlang({ 'route.nl': ROUTE, 'types.ts': TYPES }, 'route.nl');
  const project = compileVirtualProject({ files: { 'main.ts': `import { nl } from '@natlang/node';
export type Ticket = { id: string; text: string };
export const outage = nl<(t: Ticket) => boolean>\`Does t report an outage?\`;
export const wrong = nl<(n: number) => boolean>\`Is n large?\`;
` } });
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const { outage, wrong } = project.require('main.ts');
  assert.deepEqual(await runtime.run(() => route(outage, TICKETS)), ['a']);
  assert.deepEqual(await runtime.run(() => route(async t => t.id === 'b', TICKETS)), ['b']);
  await assert.rejects(runtime.run(() => route(wrong, TICKETS)), /function-signature-mismatch: this slot takes a function \(t: Ticket\) => boolean, but the function given is \(n: number\) => boolean/);
});

test('a .nl function returns a function: its call note says how, and the result carries its signature', async () => {
  const notes = callNotes({ ownCode: false, allowAdHoc: true, functionResult: '(t: Ticket) => boolean' });
  assert.match(notes, /This call's result is a function of type \(t: Ticket\) => boolean\. Build it in eval and finish with it/);
  assert.doesNotMatch(callNotes({ ownCode: false, allowAdHoc: true }), /result is a function/);
  const make = '---\nargs: { rubric: string }\nreturns: "(t: Ticket) => boolean"\n---\nMake a judge of tickets against rubric.\n';
  const model = scriptedModel(opening => /Make a judge/.test(opening) ? 'return nl`Does t meet ${rubric}?`' : 'return t.text.includes("down")');
  const runtime = createNatlangRuntime({ model: model.driver });
  const judgeFor = loadVirtualNatlang({ 'judgeFor.nl': make, 'types.ts': TYPES }, 'judgeFor.nl');
  const verdicts = await runtime.run(async () => { const judge = await judgeFor('reports an outage');
    assert.equal(signatureOf(judge).type.params.fields[0].name, 't');
    return Promise.all(TICKETS.map(t => judge(t))); });
  assert.deepEqual(verdicts, [true, false]);
  assert.match(model.openings[0], /This call's result is a function of type \(t: Ticket\) => boolean/);
});

test('partial binds leading parameters, checks them now, and keeps on, in and iterateOn', async () => {
  const model = scriptedModel(opening => /Does text meet rubric/.test(opening) ? 'return text.length > rubric.length' : null);
  const runtime = createNatlangRuntime({ model: model.driver });
  const project = compileVirtualProject({ files: { 'main.ts': `import { nl } from '@natlang/node';
export const meets = nl<(rubric: string, text: string) => boolean>\`Does text meet rubric?\`;
export const short = meets.partial('abc');
` } });
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const { meets, short } = project.require('main.ts');
  assert.deepEqual(signatureOf(short).type.params.fields.map(field => field.name), ['text']);
  assert.equal(await runtime.run(() => short('abcdef')), true);
  assert.equal(await runtime.run(() => short('ab')), false);
  for (const method of ['on', 'in', 'iterateOn', 'partial']) assert.equal(typeof short[method], 'function', method);
  assert.throws(() => meets.partial(3), /\/rubric: type-mismatch, expected string, got 3/);
  assert.throws(() => meets.partial('a', 'b', 'c'), /takes 2 parameters \(rubric, text\), so it cannot bind 3/);
  const typed = compileVirtualProject({ files: { 'main.ts': `import { nl } from '@natlang/node';
const meets = nl<(rubric: string, text: string) => boolean>\`Does text meet rubric?\`;
export async function main(): Promise<boolean> { const short = meets.partial('abc'); return short(42); }
` } });
  assert.ok(typed.diagnostics.some(item => /number.*string|string.*number/.test(item.message)), JSON.stringify(typed.diagnostics));
});

test('Record keyed by an alias of string literals is the exact record', () => {
  const env = new TypeEnv({ Verdict: parseType('"refund" | "decline"'), Ledger: parseType('Record<Verdict, number>') });
  assert.deepEqual(env.resolve(parseType('Ledger')), parseType('{ refund: number, decline: number }'));
  assert.deepEqual(coerce({ refund: 1, decline: 0 }, parseType('Ledger'), env), { refund: 1, decline: 0 });
  assert.throws(() => coerce({ refund: 1 }, parseType('Ledger'), env, 'value', { preserveRecordExtras: true }), /decline/);
  assert.deepEqual(new TypeEnv({ K: parseType('string') }).resolve(parseType('Record<K, number>')), parseType('Record<string, number>'));
  assert.throws(() => new TypeEnv({ K: parseType('number') }).resolve(parseType('Record<K, number>')), /Record keys are strings: K is number/);
});

test('an alias the type grammar cannot read breaks only the signatures that use it, naming the alias', async () => {
  const types = 'export type Ticket = { id: string; text: string };\nexport type Keys = keyof Ticket;\n';
  const fine = loadVirtualNatlang({ 'route.nl': '---\nargs: { t: Ticket }\nreturns: string\n---\nName t.\n', 'types.ts': types }, 'route.nl');
  const model = scriptedModel(() => 'return t.id');
  assert.equal(await createNatlangRuntime({ model: model.driver }).run(() => fine({ id: 'x', text: 'y' })), 'x');
  assert.throws(() => loadVirtualNatlang({ 'pick.nl': '---\nargs: { k: Keys }\nreturns: string\n---\nName k.\n', 'types.ts': types }, 'pick.nl'),
    /type Keys is not in the natlang type grammar/);
});

test('a target argument may be a program type that contains Untrusted or shares a built-in name', () => {
  const root = mkdtempSync(join(tmpdir(), 'natlang-higher-order-'));
  try {
    writeFileSync(join(root, 'types.ts'), "import type { Untrusted } from '@natlang/node';\nexport type Request = { id: string; text: Untrusted<string> };\n");
    const declared = parseTargetArguments([{ name: 'requests', type: 'Request[]' }], 'main');
    assert.equal(describeArguments(declared, root)[0].problem, undefined);
  } finally { rmSync(root, { recursive: true, force: true }); }
});

browserTest('the browser runtime checks function signatures and binds partial arguments', async () => {
  const process = globalThis.process;
  let browser;
  try { globalThis.process = undefined; browser = await import('../dist/browser/natlang.js'); } finally { globalThis.process = process; }
  const model = scriptedModel(opening => /Keep the ids/.test(opening) ? keepAccepted : /Does text meet rubric/.test(opening) ?
    'return text.length > rubric.length' : 'return t.text.includes("down")');
  const project = browser.compileVirtualProject({ files: { 'main.ts': `import { nl } from '@natlang/browser';
export type Ticket = { id: string; text: string };
export const outage = nl<(t: Ticket) => boolean>\`Does t report an outage?\`;
export const wrong = nl<(n: number) => boolean>\`Is n large?\`;
export const meets = nl<(rubric: string, text: string) => boolean>\`Does text meet rubric?\`;
` } }, browser);
  assert.equal(project.ok, true, JSON.stringify(project.diagnostics));
  const { outage, wrong, meets } = project.require('main.ts');
  const route = browser.loadVirtualNatlang({ 'route.nl': ROUTE, 'types.ts': TYPES }, 'route.nl');
  const runtime = browser.createNatlangRuntime({ model: model.driver });
  assert.deepEqual(await runtime.run(() => route(outage, TICKETS)), ['a']);
  await assert.rejects(runtime.run(() => route(wrong, TICKETS)), /function-signature-mismatch/);
  assert.equal(await runtime.run(() => meets.partial('abc')('abcdef')), true);
});
