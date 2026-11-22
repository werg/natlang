import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { compileProject } from '../dist/compiler/project.js';
import { checkConstrainedSource } from '../dist/compiler/policy.js';
import { virtualProjectFiles } from '../dist/runtime/virtual-project.js';
import { parseType, TypeEnv } from '../dist/native/types.js';
import { APPROACH_PROMPT } from '../dist/native/prompt.js';
import { crispExamples } from '../playground/examples/crisp.mjs';
import { interfaceExamples } from '../playground/examples/interfaces.mjs';
import * as runtimeNamespace from '../dist/index.js';
import { Context, compileVirtualProject, createNatlangRuntime, loadVirtualNatlang } from '../dist/index.js';
import { callableMeta } from '../dist/runtime/callable.js';
import { TypeEnv as NativeTypeEnv } from '../dist/native/types.js';
import { collectObligations } from '../dist/native/refinement.js';

function analyze(source, declarations = '') {
  const files = { '/scope/decls.ts': declarations, '/scope/main.ts': source };
  const program = createVirtualProgram(files);
  return analyzeInlineLambdas(program, [program.getSourceFile('/scope/main.ts')], {
    scopeFiles: [program.getSourceFile('/scope/decls.ts')],
  });
}

test('inline nl supports Is<T, P>, nested Is, Untrusted<Is<T, P>>, and object intersections', () => {
  const decls = `
    type Base = { id: string };
    type Reply = Is<string, "polite and concise">;
    declare const input: Untrusted<string>;
  `;
  const source = `
    async function run() {
      const a: Is<string, "one line of at most 60 characters"> = await nl\`Summarize input.\`(input);
      const b = await nl<Is<Is<string, "polite">, "under two sentences">>\`Reply to input.\`(input);
      const c = await nl<Untrusted<Is<string, "factual">>>\`Extract fact from input.\`(input);
      const d = await nl<Is<Untrusted<string>, "factual">>\`Extract fact from input.\`(input);
      const e = await nl<Reply>\`Draft a reply to input.\`(input);
      const f = await nl<Base & { score: number }>\`Score input.\`(input);
      return { a, b, c, d, e, f };
    }
  `;
  const { plans, diagnostics } = analyze(source, decls);
  assert.deepEqual(diagnostics, []);
  assert.equal(plans.length, 6);
  assert.equal(plans[0].returns.natlang, 'Is<string, "one line of at most 60 characters">');
  assert.equal(plans[1].returns.natlang, 'Is<Is<string, "polite">, "under two sentences">');
  assert.equal(plans[2].returns.natlang, 'Untrusted<Is<string, "factual">>');
  assert.equal(plans[3].returns.natlang, 'Untrusted<Is<string, "factual">>');
  assert.equal(plans[4].returns.natlang, 'Is<string, "polite and concise">');
  assert.equal(plans[5].returns.natlang, 'Base & { score: number }');
  assert.deepEqual(plans[5].returns.aliases, { Base: '{ id: string }' });

  // Every emitted natlang type string parses and resolves cleanly in native/types.ts.
  for (const plan of plans) {
    const env = new TypeEnv(Object.fromEntries(Object.entries(plan.returns.aliases).map(([k, v]) => [k, parseType(v)])));
    const parsed = parseType(plan.returns.natlang);
    env.checkNames(parsed);
  }
});

test('inline nl still rejects non-object intersections', () => {
  const { diagnostics } = analyze('async function run(x: string) { return await nl<string & { extra: number }>`Do x.`(x); }');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /an intersection is portable only as object types joined with &/);
});

test('compileProject reports refinedSlots for direct and aliased inline Is<T, P> calls', () => {
  const files = virtualProjectFiles({
    'main.ts': `
      import { nl, type Is } from '@natlang/node';
      type Reply = Is<string, "polite and concise">;
      export async function respond(ticket: string): Promise<{ r1: Reply; r2: Is<string, "one line"> }> {
        const r1 = await nl<Reply>\`Answer ticket.\`(ticket);
        const r2 = await nl<Is<string, "one line">>\`Title ticket.\`(ticket);
        return { r1, r2 };
      }
    `,
  });
  const built = compileProject({ project: '/project', files, emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.equal(built.manifest.inline.length, 2);
  const predicates = (built.refinedSlots ?? []).map(slot => slot.predicate).sort();
  assert.deepEqual(predicates, ['one line', 'polite and concise']);
});

test('NatlangFunction.in(context), crisp Folder reducers, and Folder/FileHandle moveTo type-check cleanly', () => {
  const files = virtualProjectFiles({
    'greet.nl': '---\nargs:\n  name: string\nreturns: string\n---\nGreet name.\n',
    'main.ts': `
      import { type Context, type Folder } from '@natlang/node';
      import greet from './greet.nl';
      export async function run(ctx: Context, folder: Folder) {
        const bound = greet.in(ctx);
        const message: string = await bound('Ada');
        const crispStep = async (f: Folder, state: { remaining: number }) => {
          await f.file('a.txt').moveTo(f.dir('out'));
          return { remaining: state.remaining - 1 };
        };
        const out = await folder.apply(crispStep, { remaining: 1 });
        const prop = await folder.propose(crispStep, { remaining: 1 });
        const iter = await folder.iterateOn(crispStep, { remaining: 2 })
          .withMeasure(s => s.remaining)
          .until(s => s.remaining === 0);
        return { message, out, prop, iter };
      }
    `,
  });
  const built = compileProject({ project: '/project', files, emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  assert.equal(built.manifest.named.length, 1);
});

test('all playground crisp and crisp interface examples satisfy natlang finite-iteration source policy', () => {
  for (const example of [...crispExamples, ...interfaceExamples.filter(item => !item.modelRequired)]) {
    for (const [path, text] of Object.entries(example.files)) {
      if (!path.endsWith('.ts')) continue;
      const sf = ts.createSourceFile(path, text, ts.ScriptTarget.Latest, true);
      const diags = checkConstrainedSource(sf);
      assert.deepEqual(diags, [], `${example.id} (${path}) failed constrained policy: ${JSON.stringify(diags)}`);
    }
  }
});

test('APPROACH_PROMPT accurately reflects persistent top-level eval helpers', () => {
  assert.match(APPROACH_PROMPT, /Top-level helper functions and supported local values carry forward/);
  assert.doesNotMatch(APPROACH_PROMPT, /Ordinary JavaScript function declarations and function-valued locals are scoped to the eval/);
});

// --- Runtime: inline refinements are enforced and fed back like .nl frontmatter refinements ----------------------

const POLITE = 'a reply that is polite';
/**
 * A scripted runtime. Calls whose opening names `root(` run `rootTurns` in order; every other tool-loop call returns
 * the next of `answers`. The judge says a value is polite when it contains "sorry".
 */
function refinementScenario({ rootTurns = [], answers }) {
  const children = [], judged = [];
  let root = 0, answer = 0;
  const driver = Object.assign(async request => {
    const opening = String(request.messages[1]?.content ?? '');
    if (opening.includes('root(')) {
      const turn = rootTurns[Math.min(root++, rootTurns.length - 1)];
      return { calls: [typeof turn === 'function' ? turn(String(request.messages.at(-1).content)) : turn] };
    }
    children.push(String(request.messages.at(-1).content));
    return { calls: [['return_result', { status: 'success', value: answers[Math.min(answer++, answers.length - 1)] }]] };
  }, {
    decide: async request => {
      const value = /<<<value\n([^]*?)\nvalue>>>/.exec(String(request.messages.at(-1).content))[1];
      judged.push(value);
      const p = /sorry/i.test(value) ? 0.97 : 0.03;
      return { log_probs: [Math.log(p), Math.log(1 - p)] };
    },
  });
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'backend' }, calls: false });
  return { runtime, children, judged };
}

test('a refined .nl return and an inline nl<Is<T, P>> in eval get the same rejection feedback and retry', async () => {
  const feedbackOf = children => children.find(text => /refinement-unsatisfied/.test(text));
  // Reference: the frontmatter form.
  const named = refinementScenario({ answers: ['It is your fault.', 'We are sorry about the delay.'] });
  const reply = loadVirtualNatlang({ 'reply.nl': `---\nargs: { complaint: string }\nreturns: 'Is<string, "${POLITE}">'\n---\nAnswer the complaint.\n` }, 'reply.nl');
  assert.equal(await named.runtime.run(() => reply('late parcel')), 'We are sorry about the delay.');
  const reference = feedbackOf(named.children);
  assert.match(reference, new RegExp(`refinement-unsatisfied: Revise the value at return so that it is "${POLITE}"`));

  // The inline form, written by a model in eval.
  let seen = '';
  const inline = refinementScenario({
    rootTurns: [['eval', { code: `const r = await nl<Is<string, "${POLITE}">>\`Answer the complaint.\`(complaint);\nr` }],
      last => { seen = last; return ['return_result', { status: 'success', value: 'done' }]; }],
    answers: ['It is your fault.', 'We are sorry about the delay.'],
  });
  const root = loadVirtualNatlang({ 'root.nl': '---\nargs: { complaint: string }\nreturns: string\n---\nAnswer the complaint.\n' }, 'root.nl');
  await inline.runtime.run(() => root('late parcel'));
  assert.match(seen, /We are sorry about the delay\./, 'the eval saw the repaired child value');
  assert.deepEqual(inline.judged, ['It is your fault.', 'We are sorry about the delay.']);
  assert.equal(feedbackOf(inline.children), reference, 'inline and frontmatter refinements give the same feedback');
});

test('compiled project code enforces inline Is<T, P> on a direct, aliased, Untrusted and intersected result', async () => {
  const project = { files: {
    'main.ts': `
      import { nl, type Is, type Untrusted } from '@natlang/node';
      type Polite = Is<string, "${POLITE}">;
      type Base = { id: string };
      export const direct = (t: string) => nl<Is<string, "${POLITE}">>\`Answer t.\`(t);
      export const aliased = (t: string) => nl<Polite>\`Answer t.\`(t);
      export const outside = (t: string) => nl<Untrusted<Is<string, "${POLITE}">>>\`Answer t.\`(t);
      export const record = (t: string) => nl<Base & { reply: Polite }>\`Answer t.\`(t);
    `,
  } };
  const compiled = compileVirtualProject(project, runtimeNamespace, { target: 'node' });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.deepEqual(compiled.refinedSlots.map(slot => slot.slot + ':' + slot.predicate).sort(),
    ['return.reply:' + POLITE, 'return:' + POLITE, 'return:' + POLITE, 'return:' + POLITE]);
  const app = compiled.require('main.ts');
  for (const name of ['direct', 'aliased', 'outside']) {
    const s = refinementScenario({ answers: ['It is your fault.', 'We are sorry.'] });
    assert.equal(await s.runtime.run(() => app[name]('late parcel')), 'We are sorry.', name);
    assert.deepEqual(s.judged, ['It is your fault.', 'We are sorry.'], name);
  }
  const s = refinementScenario({ answers: [{ id: 'a', reply: 'It is your fault.' }, { id: 'a', reply: 'We are sorry.' }] });
  assert.deepEqual(await s.runtime.run(() => app.record('late parcel')), { id: 'a', reply: 'We are sorry.' });
  assert.deepEqual(s.judged, ['It is your fault.', 'We are sorry.']);
  assert.ok(s.children.some(text => /refinement-unsatisfied: Revise the value at return\/reply/.test(text)), s.children.join('\n---\n'));
});

test('Is<T, P> with a non-literal predicate names the predicate rule', () => {
  const { diagnostics } = analyze('async function run(x: string, p: string) { return await nl<Is<string, string>>`Do x.`(x); }');
  assert.equal(diagnostics.length, 1);
  assert.match(diagnostics[0].message, /refinement-predicate-invalid: Is<T, P> takes a nonempty string literal as P/);
});

test('a refinement on a field typed in both members of an intersection is checked, as refinedSlots reports it', () => {
  const env = new NativeTypeEnv();
  const type = parseType(`{ reply: Is<string, "${POLITE}"> } & { reply: string }`);
  assert.deepEqual(collectObligations({ reply: 'hi' }, type, env).map(item => `${item.path}:${item.predicate}`), [`return/reply:${POLITE}`]);
});

test('the Context type on the eval and module surfaces is the runtime class surface, and a program type named Context is its own', () => {
  const methods = name => Object.getOwnPropertyNames(Context.prototype).includes(name) ||
    Object.getOwnPropertyDescriptor(Context.prototype, name)?.get !== undefined;
  for (const name of ['with', 'without', 'edit', 'pick', 'union', 'contains', 'names', 'data']) assert.ok(methods(name), name);
  assert.ok(!methods('merge'));
  const files = virtualProjectFiles({
    'types.ts': 'export type Context = { topic: string };\n',
    'main.ts': `
      import { nl, type Context as Ctx } from '@natlang/node';
      import type { Context } from './types.js';
      export async function run(ctx: Ctx, c: Context) {
        const pickedNames: string[] = ctx.union(ctx.pick('a')).without('b').with({ c: 1 }).names;
        return { pickedNames, topic: c.topic, answer: await nl<string>\`Answer c.\`.in(ctx)(c) };
      }
    `,
  });
  const built = compileProject({ project: '/project', files, emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(built.ok, true, JSON.stringify(built.diagnostics));
  const merged = compileProject({ project: '/project', files: virtualProjectFiles({ 'main.ts':
    "import type { Context } from '@natlang/node';\nexport const f = (a: Context, b: Context) => a.merge(b);\n" }),
  emit: false, write: false, surfaceSpecifiers: ['@natlang/node'] });
  assert.equal(merged.ok, false, 'Context has union, not merge');
  // In eval, a call whose program declares its own Context type keeps it.
  const { plans, diagnostics } = analyze('async function run() { return await nl<Context>`Pick a topic.`(); }',
    'type Context = { topic: string };');
  assert.deepEqual(diagnostics, []);
  assert.equal(plans[0].returns.natlang, 'Context');
});

test('an inline nl rebound with .in(context) keeps that context through .with(captures)', async () => {
  const compiled = compileVirtualProject({ files: { 'main.ts': `
    import { nl } from '@natlang/node';
    export const make = (rubric: string) => nl<string>\`Grade the essay against rubric.\`;
  ` } }, runtimeNamespace, { target: 'node' });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  const fn = compiled.require('main.ts').make('be brief');
  const ctx = Context.empty().with({ note: 'graded essays are kept' });
  const rebound = fn.in(ctx);
  assert.equal(callableMeta(rebound).definition.contextId, ctx.id);
  assert.equal(typeof rebound.with, 'function');
  const again = rebound.with({ rubric: 'be thorough' });
  assert.equal(callableMeta(again).definition.contextId, ctx.id, '.with after .in keeps the context');
  assert.ok(Object.hasOwn(callableMeta(again).captures, 'note'), 'the context data stays in scope');
});

test('the prompt is true: top-level helpers carry forward, including one that calls another helper of the same eval', async () => {
  const { compileScopeSnippet } = await import('../dist/scope-compiler.js');
  const compiled = compileScopeSnippet('function f(x: number) { return x + 1; }\nconst g = (x: number) => f(x) * 2;\ng(1)');
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.deepEqual(compiled.persistentHelpers?.map(helper => helper.name), ['f', 'g']);
  assert.deepEqual(compiled.transientHelpers ?? [], []);
});

test('a helper that depends on a value that cannot persist stays in its eval, and so does a helper calling it', async () => {
  const { compileScopeSnippet } = await import('../dist/scope-compiler.js');
  const compiled = compileScopeSnippet('const reply = fetch("https://example.test/");\nconst step = () => reply;\n' +
    'function revise() { return step(); }\nconst plain = (n: number) => n + 1;\nrevise()', { allowNetwork: true });
  assert.equal(compiled.ok, true, JSON.stringify(compiled.diagnostics));
  assert.deepEqual(compiled.persistentHelpers?.map(helper => helper.name), ['plain']);
  assert.deepEqual(compiled.transientHelpers.map(helper => helper.name).sort(), ['revise', 'step']);
  assert.match(compiled.transientHelpers.find(helper => helper.name === 'revise').reason, /eval-local function step/);
});

test('a brand TypeScript distributes over a union is the brand of the union', () => {
  const { plans, diagnostics } = analyze(`
    async function run() {
      const a = await nl<Is<boolean, "the claim is supported">>\`Judge.\`();
      const b = await nl<Is<"yes" | "no", "a considered answer">>\`Answer.\`();
      const c = await nl<Untrusted<Is<string | number, "factual">>>\`Extract.\`();
      return { a, b, c };
    }
  `);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(plans.map(plan => plan.returns.natlang), ['Is<boolean, "the claim is supported">',
    'Is<"yes" | "no", "a considered answer">', 'Untrusted<Is<string | number, "factual">>']);
  for (const plan of plans) parseType(plan.returns.natlang);
});

test('an intersection of a refinement and its own base is that refinement', () => {
  const env = new NativeTypeEnv();
  const type = parseType(`{ reply: Is<string, "${POLITE}"> } & { reply: Is<string, "short"> } & { reply: string }`);
  assert.deepEqual(collectObligations({ reply: 'hi' }, type, env).map(item => `${item.path}:${item.predicate}`).sort(),
    [`return/reply:${POLITE}`, 'return/reply:short']);
});
