import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { natlangTransformer } from '../dist/compiler/lower.js';
import { checkConstrainedSource, authoredCallables } from '../dist/compiler/policy.js';

function analyze(source, declarations = '') {
  const files = { '/scope/decls.ts': declarations, '/scope/main.ts': source };
  const program = createVirtualProgram(files);
  return analyzeInlineLambdas(program, [program.getSourceFile('/scope/main.ts')], { scopeFiles: [program.getSourceFile('/scope/decls.ts')] });
}
const summary = plan => ({ params: plan.parameters.map(item => `${item.name}:${item.type.natlang ?? item.type.text}`),
  returns: plan.returns.natlang ?? plan.returns.text, captures: plan.captures.map(item => `${item.name}${item.mutable ? '*' : ''}`) });

const DECLS = `type Verdict = { ok: boolean, reason: string };
declare let note: string; declare const policy: string; declare let limit: number; declare let result: Verdict;
declare const reviewEach: (notes: string[], judge: (note: string) => Promise<Verdict>) => Promise<void>;`;

test('the enclosing result slot and call arguments give an unannotated nl its signature', () => {
  const { plans, diagnostics } = analyze('async function f() { result = await nl`Judge note against limit and policy.`(note); }', DECLS);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(summary(plans[0]), { params: ['note:string'], returns: 'Verdict', captures: ['limit*', 'policy'] });
  assert.deepEqual(plans[0].returns.aliases, { Verdict: '{ ok: boolean, reason: string }' });
});

test('contextual callback types, annotated locals and return-only annotations', () => {
  const { plans, diagnostics } = analyze(`async function f() {
    const judge: (note: string) => Promise<Verdict> = nl\`Judge note against policy.\`;
    await reviewEach([note], nl\`Judge note.\`);
    const flags: boolean[] = await Promise.all([note].map(n => nl<boolean>\`Is n urgent given limit?\`(n)));
    const full = nl<(item: string, count: number) => string>\`Repeat item count times.\`;
  }`, DECLS);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(plans.map(summary), [
    { params: ['note:string'], returns: 'Verdict', captures: ['policy'] },
    { params: ['note:string'], returns: 'Verdict', captures: [] },
    { params: ['n:string'], returns: 'boolean', captures: ['limit*'] },
    { params: ['item:string', 'count:number'], returns: 'string', captures: [] }]);
});

test('later uses of an unannotated local supply a unique return type', () => {
  const { plans, diagnostics } = analyze('async function f() { const later = await nl`Score note`(note); result = later; }', DECLS);
  assert.deepEqual(diagnostics, []);
  assert.equal(summary(plans[0]).returns, 'Verdict');
  const conflicting = analyze('async function f() { const x = await nl`Score note`(note); const a: string = x; const b: number = x; }', DECLS);
  assert.equal(conflicting.diagnostics[0].code, 'nl-ambiguous-signature');
});

test('missing, synchronous, any-typed and colliding signatures produce precise diagnostics', () => {
  // Nothing says what the result is: the call runs with an open result instead of being refused.
  const unknown = analyze('async function f() { const x = await nl`Check policy`(note); }', DECLS);
  assert.deepEqual(unknown.diagnostics, []);
  assert.equal(unknown.plans[0].returns.natlang, 'unknown');
  // A parameter with a part nothing types (a service's untyped rows) is open in that part.
  const partly = analyze('async function f(rows: any[]) { const facts = { n: rows.length, first: rows[0] }; const x: string = await nl`Judge facts`(facts); }', DECLS);
  assert.deepEqual(partly.diagnostics, []);
  assert.deepEqual(summary(partly.plans[0]).params, ['facts:{ n: number, first: unknown }']);
  const sync = analyze('function f() { [note].filter(nl`Keep?`); }', DECLS);
  assert.equal(sync.diagnostics[0].code, 'nl-sync-callback');
  const anyTarget = analyze('async function f() { const x: any = await nl`Anything`(); }', DECLS);
  assert.deepEqual(anyTarget.diagnostics, []);
  assert.equal(anyTarget.plans[0].returns.natlang, 'unknown');
  const collision = analyze('async function f(a: { note: string }, note2: string) { const note = "x"; result = await nl`Judge`(note, note); }', DECLS);
  assert.equal(collision.diagnostics[0].code, 'nl-parameter-collision');
  const spread = analyze('async function f(items: string[]) { result = await nl`Judge`(...items); }', DECLS);
  assert.equal(spread.diagnostics[0].code, 'nl-spread');
});

test('explicit nl.with captures cannot collide with inferred or annotated input names', () => {
  const inferred = analyze(`async function f(currentDraft: string) {
    const result: boolean = await nl.with({ currentDraft })
      \`Judge the supplied currentDraft.\`(currentDraft);
  }`);
  assert.equal(inferred.plans.length, 0);
  assert.equal(inferred.diagnostics[0].code, 'nl-capture-parameter-collision');
  assert.match(inferred.diagnostics[0].message, /Rename the capture or the call argument/);

  const annotated = analyze(`async function f(currentDraft: string) {
    const judge = nl.with<(currentDraft: string) => Promise<boolean>>({ currentDraft })
      \`Judge the supplied currentDraft.\`;
    await judge(currentDraft);
  }`);
  assert.equal(annotated.plans.length, 0);
  assert.equal(annotated.diagnostics[0].code, 'nl-capture-parameter-collision');

  const disjoint = analyze(`async function f(currentDraft: string) {
    const policy = 'Use the current policy.';
    const inferred: boolean = await nl.with({ policy })
      \`Judge currentDraft using policy.\`(currentDraft);
    const annotated = nl.with<(draft: string) => Promise<boolean>>({ policy })
      \`Judge draft using policy.\`;
    await annotated(currentDraft);
  }`);
  assert.deepEqual(disjoint.diagnostics, []);
  assert.deepEqual(disjoint.plans.map(plan => plan.parameters.map(parameter => parameter.name)), [['currentDraft'], ['draft']]);
  assert.deepEqual(disjoint.plans.map(plan => plan.captures.map(capture => capture.name)), [['policy'], ['policy']]);
});

test('a suffix .with binds captures to an nl template through the existing explicit capture plan', () => {
  const source = `async function f(currentDraft: string) {
    const policy = 'Use current policy.';
    const next = await nl<boolean>\`Judge currentDraft under policy.\`.with({ policy })(currentDraft);
    return next;
  }`;
  const result = analyze(source);
  assert.deepEqual(result.diagnostics, []);
  assert.deepEqual(result.plans.map(summary), [{ params: ['currentDraft:string'], returns: 'boolean', captures: ['policy'] }]);
  const file = createVirtualProgram({ '/scope/main.ts': source }).getSourceFile('/scope/main.ts');
  const transformed = ts.transform(file, [natlangTransformer({ plans: new Map(result.plans.map(plan =>
    [`${plan.sourceSpan.start}:${plan.sourceSpan.end}`, plan])), runtime: '__natlang', constrained: false,
    guardPrefix: '__guard', modulePath: 'main.ts' })]);
  const printed = ts.createPrinter().printFile(transformed.transformed[0]);
  assert.match(printed, /nl\.__inline/);
  assert.doesNotMatch(printed, /\.with\(\{ policy \}\)/, 'suffix capture is compiled away rather than called on the function');
  transformed.dispose();
});

test('captures follow lexical scope, shadowing, and exact mentions; interpolations capture their bindings', () => {
  const { plans } = analyze(`async function f(item: string) {
    let tally = 0;
    { const inner = 1; }
    const later = await nl<number>\`Add item to tally, not inner or later or itemized \${policy}\`(item);
    let afterwards = 2;
  }`, DECLS);
  assert.deepEqual(summary(plans[0]).captures, ['tally*', 'policy']);
  const backtick = analyze('async function f() { const x: string = await nl`Use \\`missing\\` here`(); }', DECLS);
  assert.equal(backtick.diagnostics[0].code, 'nl-unknown-name');
});

test('host objects become live targets; Folder is a portable handle type', () => {
  const { plans, diagnostics } = analyze(`class Ledger { total = 0; add(x: number) { this.total += x; } }
    async function f(ledger: Ledger, when: Date, index: Map<string, number>) {
      const next: Date = await nl\`Advance when by a day using index and ledger.\`(when);
    }`, DECLS);
  assert.deepEqual(diagnostics, []);
  assert.deepEqual(plans[0].returns.host, { kind: 'tag', tag: 'Date' });
  const kinds = Object.fromEntries(plans[0].captures.map(capture => [capture.name, capture.type.host]));
  assert.deepEqual(kinds.index, { kind: 'tag', tag: 'Map' });
  assert.deepEqual(kinds.ledger, { kind: 'class', name: 'Ledger' });
});

test('eval snippets are analyzed against the declared scope, with snippet-relative spans', () => {
  const { plans, diagnostics } = analyzeEvalSnippet('const threshold = 3;\nreturn await nl`Is note longer than threshold?`(note);',
    { types: {}, inputs: [{ name: 'note', type: 'string' }], locals: [], captures: [], imports: [], returns: 'boolean' });
  assert.deepEqual(diagnostics, []);
  assert.equal(plans[0].sourceSpan.line, 2);
  assert.deepEqual(summary(plans[0]), { params: ['note:string'], returns: 'boolean', captures: ['threshold'] });
  assert.equal(plans[0].captures[0].source, 'local');
});

test('the nl intrinsic is recognized by its declaration, not its spelling', () => {
  const { plans } = analyze('async function f() { const nl = (s: TemplateStringsArray) => async () => 1; result = await nl`not natlang`() as never; }', DECLS);
  assert.equal(plans.length, 0);
});

const policy = source => checkConstrainedSource(ts.createSourceFile('m.ts', source, ts.ScriptTarget.ES2022, true)).map(item => item.code);

test('constrained source accepts finite iteration and rejects open-ended forms', () => {
  assert.deepEqual(policy(`for (const x of [1, 2]) {}
    for (let i = 0; i < 10; i++) {}
    for (let i = 10; i >= 0; i -= 2) {}
    [1].map(x => x).filter(Boolean);
    for await (const event of step.iterateOn(1).streamUntil(done)) {}
    for await (const chunk of response.body) {}
    scheduler.setInterval(job, 5);`), []);
  for (const source of ['while (x) {}', 'do {} while (x)', 'for (;;) {}', 'for (const k in o) {}', 'setInterval(tick, 5)', 'window.setInterval(tick, 5)',
    'const o = { [Symbol.iterator]: () => it }', 'Iterator.from(source)', 'class Forever extends Iterator {}',
    'function* g() {}', 'for (let i = 0; i < n; i--) {}', 'for (let i = 0; i < n; i++) { i++; }', 'for (let i = 0; i < xs.length; i++) { xs.push(1); }',
    'for (let i = 0; i < count(); i++) {}', 'eval("1")', 'new Function("")', 'import("x")'])
    assert.ok(policy(source).length, source);
});

test('inline instruction provenance retains checked interpolation spans and types', () => {
  const source = 'const limit = 17;\nreturn await nl<boolean>`Check ${note} against ${limit}.`(note);';
  const scope = { types: {}, inputs: [{ name: 'note', type: 'string' }], locals: [], captures: [], imports: [], returns: 'boolean' };
  const { plans, diagnostics } = analyzeEvalSnippet(source, scope);
  assert.deepEqual(diagnostics, []);
  const plan = plans[0];
  assert.equal(source.slice(plan.templateSpan.start, plan.templateSpan.end), '`Check ${note} against ${limit}.`');
  assert.deepEqual(plan.strings, ['Check ', ' against ', '.']);
  assert.deepEqual(plan.interpolations.map(item => item.expression), ['note', 'limit']);
  assert.deepEqual(plan.interpolations.map(item => source.slice(item.sourceSpan.start, item.sourceSpan.end)), ['note', 'limit']);
  assert.equal(plan.interpolations[0].type.natlang, 'string');
  assert.equal(plan.interpolations[1].type.text, '17');
  assert.ok(plan.interpolations.every(item => item.sourceSpan.line === 2));
  assert.equal(JSON.parse(JSON.stringify(plan)).interpolations[1].expression, 'limit');
});

test('inline provenance keeps distinct identical sites and empty static bindings', () => {
  const source = 'const first = nl<boolean>`Check note.`; const second = nl<boolean>`Check note.`; return await first(note);';
  const { plans, diagnostics } = analyzeEvalSnippet(source,
    { types: {}, inputs: [{ name: 'note', type: 'string' }], locals: [], captures: [], imports: [], returns: 'boolean' });
  assert.deepEqual(diagnostics, []);
  assert.equal(plans.length, 2);
  assert.notEqual(plans[0].definitionId, plans[1].definitionId);
  for (const plan of plans) {
    assert.deepEqual(plan.interpolations, []);
    assert.equal(source.slice(plan.templateSpan.start, plan.templateSpan.end), '`Check note.`');
  }
});


test('invalid multiple nl type arguments are rejected before any child schema is lowered', () => {
  const declarations = DECLS + '\ntype Context = { policy: string };';
  for (const expression of [
    'nl.with<Context, Verdict>({ policy })`Judge note.`(note)',
    'nl<Context, Verdict>`Judge note.`(note)',
    'nl.with({ policy })<Context, Verdict>`Judge note.`(note)',
  ]) {
    const result = analyze(`async function f() { const value = await ${expression}; }`, declarations);
    assert.equal(result.plans.length, 0);
    assert.equal(result.diagnostics.length, 1);
    assert.equal(result.diagnostics[0].code, 'nl-type-arguments');
    assert.match(result.diagnostics[0].message, /child result type or full callable signature/);
  }
  const valid = analyze('async function f() { const value = await nl.with<Verdict>({ policy })`Judge note.`(note); }', declarations);
  assert.deepEqual(valid.diagnostics, []);
  assert.equal(valid.plans.length, 1);
  assert.equal(valid.plans[0].returns.natlang, 'Verdict');
});
