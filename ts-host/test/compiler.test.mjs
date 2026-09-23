import assert from 'node:assert/strict';
import { test } from 'node:test';
import ts from 'typescript';
import { createVirtualProgram } from '../dist/compiler/host.js';
import { analyzeInlineLambdas } from '../dist/compiler/inline.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { checkConstrainedSource, authoredCallables, findRecursion, lexicalResolver } from '../dist/compiler/policy.js';

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
  const unknown = analyze('async function f() { const x = await nl`Check policy`(note); }', DECLS);
  assert.equal(unknown.diagnostics[0].code, 'nl-unknown-return');
  assert.match(unknown.diagnostics[0].message, /write `nl<Verdict>`/);
  const sync = analyze('function f() { [note].filter(nl`Keep?`); }', DECLS);
  assert.equal(sync.diagnostics[0].code, 'nl-sync-callback');
  const anyTarget = analyze('async function f() { const x: any = await nl`Anything`(); }', DECLS);
  assert.equal(anyTarget.diagnostics[0].code, 'nl-unknown-return');
  const collision = analyze('async function f(a: { note: string }, note2: string) { const note = "x"; result = await nl`Judge`(note, note); }', DECLS);
  assert.equal(collision.diagnostics[0].code, 'nl-parameter-collision');
  const spread = analyze('async function f(items: string[]) { result = await nl`Judge`(...items); }', DECLS);
  assert.equal(spread.diagnostics[0].code, 'nl-spread');
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
    for await (const event of step.iterateOn(1).streamUntil(done)) {}`), []);
  for (const source of ['while (x) {}', 'do {} while (x)', 'for (;;) {}', 'for (const k in o) {}', 'for await (const x of stream) {}',
    'function* g() {}', 'for (let i = 0; i < n; i--) {}', 'for (let i = 0; i < n; i++) { i++; }', 'for (let i = 0; i < xs.length; i++) { xs.push(1); }',
    'for (let i = 0; i < count(); i++) {}', 'eval("1")', 'new Function("")', 'import("x")'])
    assert.ok(policy(source).length, source);
});

test('the authored-call graph reports direct and mutual recursion with its path', () => {
  const file = ts.createSourceFile('m.ts', `function a() { return b(); }
    function b() { return c(); }
    function c() { return a(); }
    function self() { return self(); }
    function ok() { return [1].map(x => x + 1); }`, ts.ScriptTarget.ES2022, true);
  const callables = authoredCallables(file, 'm');
  const diagnostics = findRecursion(callables, lexicalResolver(callables));
  assert.equal(diagnostics.length, 2);
  assert.ok(diagnostics.some(item => /a → b → c → a|b → c → a → b|c → a → b → c/.test(item.message)));
  assert.ok(diagnostics.some(item => /`self` calls itself/.test(item.message)));
});
