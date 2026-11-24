/**
 * Targeted tests for the Semantic-FP review & improvement pass fixes:
 * - Harness messages & behaviour (10 defects from HELD_FAILURES_20261123 §3.8)
 * - Decision-readout tie-break (mean log-prob + deterministic seed + opt-out)
 * - Compiler, runtime, stdlib, CLI, and memetic guidance improvement fixes
 */
import assert from 'node:assert/strict';
import { existsSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { test } from 'node:test';

import {
  CallStore,
  Folder,
  createNatlangRuntime,
  decide,
  loadVirtualNatlang,
  parseType,
  rollbackSource,
  TypeEnv,
} from '../dist/index.js';
import { CALLS_VERSION } from '../dist/calls/types.js';
import { hexDigest } from '../dist/native/hash.js';
import { compileScopeSnippet } from '../dist/scope-compiler.js';
import { literalText } from '../dist/native/pseudo-call.js';
import { NL_DEPTH_LIMIT_SEARCH_BUILT_INS_LINE, schemaOf } from '../dist/native/agent.js';
import { analyzeEvalSnippet } from '../dist/compiler/eval-check.js';
import { checkNlIdioms, checkNlIdiomsIn } from '../dist/compiler/idiom.js';
import { compileModule } from '../dist/runtime/modules.js';
import { loadNamedFunction, readNatlangFrontmatter } from '../dist/runtime/loader.js';
import { virtualSourceFiles } from '../dist/runtime/virtual-project.js';
import { writesFields, writesProblem } from '../dist/runtime/writes.js';
import { escalate, semanticReduce, stepsOf } from '../dist/std/index.js';
import { evaluateDirectorySplitWithModel, improveDirectory } from '../dist/improvement/program.js';
import { proposeFromRuns } from '../dist/improvement/propose-from-runs.js';
import { parseArgs } from '../dist/cli/args.js';

const CLI = fileURLToPath(new URL('../dist/cli/main.js', import.meta.url));

// ---------------------------------------------------------------------------------------------------------------------
// 1. Harness message & behaviour fixes
// ---------------------------------------------------------------------------------------------------------------------

test('1.1: no <<cut off>> footer on Stored local when console.log already printed the full string', async () => {
  const fullText = 'A'.repeat(400);
  const toolOutputs = [];
  let turn = 0;
  const model = async req => {
    if (turn > 0) toolOutputs.push(String(req.messages.at(-1)?.content ?? ''));
    turn++;
    if (turn === 1) return { calls: [['eval', { code: `const doc = ${JSON.stringify(fullText)}; console.log(doc);` }]] };
    if (turn === 2) return { calls: [['eval', { code: `const hidden = ${JSON.stringify('B'.repeat(400))};` }]] };
    return { calls: [['return_result', { status: 'success', value: 'done' }]] };
  };
  const fn = loadVirtualNatlang({ 'f.nl': '---\nargs: {}\nreturns: string\n---\nWork.\n' }, 'f.nl');
  assert.equal(await createNatlangRuntime({ model, seed: { mode: 'backend' }, calls: false }).run(() => fn()), 'done');
  assert.match(toolOutputs[0], /Stored local doc = \(full string printed above, 400 chars\)/);
  assert.doesNotMatch(toolOutputs[0], /<<cut off/);
  assert.match(toolOutputs[1], /Stored local hidden = .*<<cut off/);
});

test('1.2: rejected service write is NOT reported as Already performed before the failure', async () => {
  const toolOutputs = [];
  let turn = 0;
  const model = async req => {
    if (turn > 0) toolOutputs.push(String(req.messages.at(-1)?.content ?? ''));
    turn++;
    if (turn === 1) return { calls: [['eval', { code: 'await board.commit_move("bad");' }]] };
    if (turn === 2) return { calls: [['eval', { code: 'await board.commit_move("ok"); throw new Error("after");' }]] };
    return { calls: [['return_result', { status: 'success', value: 'done' }]] };
  };
  const board = {
    async commit_move(move) {
      if (move === 'bad') throw new Error('illegal move');
      return { accepted: true };
    },
  };
  const fn = loadVirtualNatlang({ 'f.nl': '---\nargs: {}\nreturns: string\n---\nPlay.\n' }, 'f.nl');
  assert.equal(
    await createNatlangRuntime({ model, seed: { mode: 'backend' }, calls: false, services: { board } }).run(() => fn()),
    'done',
  );
  assert.match(toolOutputs[0], /illegal move/);
  assert.doesNotMatch(toolOutputs[0], /Already performed before the failure/);
  assert.match(toolOutputs[1], /Already performed before the failure \(not undone\): board\.commit_move/);
});

test('1.4: literalText unwraps markdown code fences, return_result omits nl.with<T> hint, tuple reports an array', async () => {
  assert.deepEqual(literalText('```json\n{"a": 1, "b": "hi"}\n```'), { value: { a: 1, b: 'hi' } });
  assert.deepEqual(literalText('```\n[1, 2]\n```'), { value: [1, 2] });

  const replies = [];
  let turn = 0;
  const model = async req => {
    if (turn > 0) replies.push(String(req.messages.at(-1)?.content ?? ''));
    turn++;
    if (turn === 1) return { calls: [['return_result', { status: 'success', value: 'not valid json at all' }]] };
    if (turn === 2) return { calls: [['return_result', { status: 'success', value: '```json\n["x", 7]\n```' }]] };
    return { calls: [['return_result', { status: 'failed', reason: 'stop' }]] };
  };
  const fn = loadVirtualNatlang({ 'f.nl': '---\nargs: {}\nreturns: "[string, number]"\n---\nReturn pair.\n' }, 'f.nl');
  const out = await createNatlangRuntime({ model, seed: { mode: 'backend' }, calls: false }).run(() => fn());
  assert.deepEqual(out, ['x', 7]);
  assert.match(replies[0], /written as an array/);
  assert.doesNotMatch(replies[0], /nl\.with<T>/);
});

test('1.5: scoped-fs rejects [object Object], non-string handle args, and self-nesting moveTo', async () => {
  const folder = Folder.fromFiles({ 'inbox/s01.md': 'hello\n' });
  const file = folder.file('inbox/s01.md');
  assert.throws(() => folder.file('[object Object]s01.md'), /pass a string path or handle\.path/);
  assert.throws(() => folder.file(file), /pass handle\.path/);
  assert.throws(() => folder.root().dir(folder.dir('inbox')), /pass handle\.path/);
  await assert.rejects(
    file.moveTo(folder.dir('traced/s01.md')),
    /instead of nesting s01\.md inside a folder with the same name/,
  );
  // Valid move into folder.dir('traced') still works cleanly.
  await file.moveTo(folder.dir('traced'));
  assert.equal(await folder.readText('traced/s01.md'), 'hello\n');
});

test('1.7-1.10: built-in visibility, const immutability wording, TDZ self-shadowing, and leaked markup detection', async () => {
  // 1.7: saved helper referencing refine/assume/search_code/NatlangContext/read compiles cleanly
  const withBuiltins = compileScopeSnippet(
    'async function helper(v: string) { const c = NatlangContext.empty(); const s = search_code("q"); const r = await refine(v, "ok"); return [c, s, r, assume(v, "ok"), read]; }\n1',
  );
  assert.equal(withBuiltins.ok, true, JSON.stringify(withBuiltins.diagnostics));
  assert.equal(withBuiltins.persistentHelpers?.length, 1);

  // 1.7: read_code("refine"), notATool("refine"), scopeGuide()
  const replies = [];
  let turn = 0;
  const model = async req => {
    if (turn > 0) replies.push(String(req.messages.at(-1)?.content ?? ''));
    turn++;
    if (turn === 1) return { calls: [['read_code', { name: 'refine' }]] };
    if (turn === 2) return { calls: [['refine', { value: 'hi' }]] };
    if (turn === 3) return { calls: [['eval', { code: 'nonexistentSymbol()' }]] };
    return { calls: [['return_result', { status: 'success', value: 'ok' }]] };
  };
  const fn = loadVirtualNatlang({
    'f.nl': '---\nargs: {}\nreturns: string\n---\nTest.\n',
    'f/h.ts': 'export default function h(): number { return 1; }\n',
  }, 'f.nl');
  assert.equal(await createNatlangRuntime({ model, seed: { mode: 'backend' }, calls: false }).run(() => fn()), 'ok');
  assert.match(replies[0], /refine\(value, predicate/);
  assert.match(replies[1], /refine is not a tool: refine is in this call's eval scope/);
  assert.match(replies[2], /decide, refine, assume, transcript, NatlangContext and search_code/);
  assert.match(NL_DEPTH_LIMIT_SEARCH_BUILT_INS_LINE, /search_code/);

  // 1.8: const immutability error names `const`
  const constReassign = compileScopeSnippet('const x = 1;\nx = 2;');
  assert.equal(constReassign.ok, false);
  assert.match(constReassign.diagnostics[0].message, /"x" was declared with `const` and cannot be reassigned; use `let` or a new variable name/);
  const priorConst = compileScopeSnippet('x = 2;', { localBindings: [{ name: 'x', mutable: false }] });
  assert.equal(priorConst.ok, false);
  assert.match(priorConst.diagnostics[0].message, /"x" was declared with `const` and cannot be reassigned/);

  // 1.9: TDZ self-shadowing inside blocks, loops, callbacks, and shadowed locals
  for (const snippet of [
    'if (true) { const intent = await intent(x); }',
    'for (const item of items) { const intent = await intent(item); }',
    'await Promise.all(items.map(async item => { const intent = await intent(item); return intent; }));',
  ]) {
    const res = compileScopeSnippet(snippet, { inputBindings: ['x', 'items'], helperBindings: ['intent'] });
    assert.equal(res.ok, false, snippet);
    assert.ok(res.diagnostics.some(d => d.code === 'invalid-binding' && /temporal dead zone.*const intentResult/.test(d.message)), JSON.stringify(res.diagnostics));
  }
  const shadowedLocalTdz = compileScopeSnippet('const count = count + 1; count', { localBindings: [{ name: 'count', mutable: false }] });
  assert.equal(shadowedLocalTdz.ok, false);
  assert.ok(shadowedLocalTdz.diagnostics.some(d => d.code === 'invalid-binding' && /temporal dead zone.*const countResult/.test(d.message)));
  // Self-reference inside function body is NOT a TDZ error
  const selfRefFn = compileScopeSnippet('const f = (n: number): number => n <= 0 ? 0 : n; f(1)');
  assert.equal(selfRefFn.ok, true, JSON.stringify(selfRefFn.diagnostics));

  // 1.10: leaked </think> / <tool_call> produces a single concise error
  const leaked = compileScopeSnippet('const x = 1;\n</think><tool_call><function=return_result></tool_call>');
  assert.equal(leaked.ok, false);
  assert.equal(leaked.diagnostics.length, 1);
  assert.equal(leaked.diagnostics[0].code, 'typescript-syntax');
  assert.match(leaked.diagnostics[0].message, /Tool-call or <think> markup leaked into eval\.code/);
  // Valid string literal containing </think> still compiles cleanly
  assert.equal(compileScopeSnippet('const tag = "</think>"; tag').ok, true);
});

// ---------------------------------------------------------------------------------------------------------------------
// 2. Decision-readout tie-break
// ---------------------------------------------------------------------------------------------------------------------

test('2: decision-readout breaks 2-vs-2 saturated ties by mean log-prob (with NATLANG_DECISION_TIEBREAK=first opt-out)', async () => {
  const fn = loadVirtualNatlang({
    'gate.nl': '---\nargs: { brief: string }\nreturns: \'"needs-review" | "pass"\'\nreadout: decision\n---\nJudge brief.\n',
  }, 'gate.nl');
  // Every conditional saturates in float64 (gaps of 800+ nats), so the mean probabilities tie exactly at [0.5, 0.5].
  // 2 samples prefer index 0 by ~800 nats, 2 samples prefer index 1 by ~1000 nats: mean log-prob favors index 1 ("pass").
  const logProbsByReasoning = {
    'think 0': [-0.01, -800],
    'think 1': [-0.01, -800],
    'think 2': [-1000, -0.01],
    'think 3': [-1000, -0.01],
  };
  const driver = Object.assign(
    async req => ({ text: '"pass"', reasoning: `think ${req.decision_reasoning?.index}`, completion_tokens: 4 }),
    { decide: async req => ({ log_probs: logProbsByReasoning[req.reasoning], tokens: [2, 2] }) },
  );
  const runtime = createNatlangRuntime({ model: { driver }, seed: { mode: 'derived', root: 7 }, calls: false });
  const chosen = await runtime.run(() => decide(fn, 'brief-1'));
  assert.equal(chosen.value, 'pass', 'index 1 wins 2-2 tie via stronger mean log-prob margin');

  // Opt-out restores first-index tie-break
  process.env.NATLANG_DECISION_TIEBREAK = 'first';
  try {
    const legacy = await runtime.run(() => decide(fn, 'brief-2'));
    assert.equal(legacy.value, 'needs-review', 'NATLANG_DECISION_TIEBREAK=first picks index 0');
  } finally {
    delete process.env.NATLANG_DECISION_TIEBREAK;
  }
});

// ---------------------------------------------------------------------------------------------------------------------
// 3. Compiler, runtime, stdlib & CLI fixes
// ---------------------------------------------------------------------------------------------------------------------

test('3.1: callable-folder import type resolves transitive/inline/export/package types, invalidates cache, rejects unresolved', () => {
  // Transitive + inline `import { type ... }` + `export type { ... } from` + package export
  const okFiles = virtualSourceFiles({
    'natlang.json': JSON.stringify({ name: '@acme/demo', exports: { './shared-types': './shared/reexport.ts' } }),
    'shared/leaf.ts': 'export type Leaf = { label: "ok" | "bad" };\n',
    'shared/reexport.ts': 'export type { Leaf } from "./leaf.js";\n',
    'root.nl': '---\nargs: { x: string }\nreturns: string\n---\nCall helper.\n',
    'root/helper.ts': 'import { type Leaf } from "@acme/demo/shared-types";\nexport default function helper(v: Leaf): string { return v.label; }\n',
  }, '/project');
  const loaded = loadNamedFunction('/project/root.nl', okFiles);
  assert.doesNotThrow(() => compileModule(loaded.codebase.helper, loaded.codebase));

  // Unresolved relative `import type` fails at load instead of collapsing to `any`
  assert.throws(
    () => loadNamedFunction('/project/root.nl', virtualSourceFiles({
      'natlang.json': '{}',
      'root.nl': '---\nargs: {}\nreturns: string\n---\nCall helper.\n',
      'root/helper.ts': 'import type { Ghost } from "../does-not-exist.js";\nexport default function helper(v: Ghost): string { return String(v); }\n',
    }, '/project')),
    /cannot import type "\.\.\/does-not-exist\.js"/,
  );

  // Missing exported member in an existing type file surfaces TS2305 in compileModule
  const badMember = loadNamedFunction('/project/root.nl', virtualSourceFiles({
    'natlang.json': '{}',
    'shared/leaf.ts': 'export type Leaf = { label: string };\n',
    'root.nl': '---\nargs: {}\nreturns: string\n---\nCall helper.\n',
    'root/helper.ts': 'import type { MissingExport } from "../shared/leaf.js";\nexport default function helper(v: MissingExport): string { return String(v); }\n',
  }, '/project'));
  assert.throws(() => compileModule(badMember.codebase.helper, badMember.codebase), /TS2305/);
});

test('3.2: untrusted-instruction propagates through string[], destructuring (all elements), for..of, and records', () => {
  const scope = {
    types: {},
    inputs: [{ name: 'untrustedText', type: 'Untrusted<string>' }],
    locals: [], captures: [], imports: [], returns: 'string',
  };
  const check = code => analyzeEvalSnippet(code, scope).diagnostics.filter(d => d.code === 'untrusted-instruction');
  assert.equal(check('const lines = untrustedText.split("\\n"); return await nl<string>`Read: ${lines[0]}`();').length, 1);
  assert.equal(check('const [first, second] = untrustedText.split("\\n"); return await nl<string>`Read: ${second}`();').length, 1);
  assert.equal(check('for (const line of untrustedText.split("\\n")) { await nl<string>`Check: ${line}`(); } return "ok";').length, 1);
  assert.equal(check('const box = { line: untrustedText }; return await nl<string>`Check: ${box.line}`();').length, 1);
  // Non-text destructured property (length: number) is NOT tainted
  assert.equal(check('const { length } = untrustedText; return await nl<string>`Count: ${length}`();').length, 0);
});

test('3.3: writes: multi-line newline join, all matching patterns checked, Is<string, P> refined file enforcement, and contract protection', async () => {
  // Multi-line writes: joined with '\n'
  const fm = readNatlangFrontmatter('kind: directory-reducer\nargs: {}\nreturns: string\nwrites:\n  {\n    "a.md": string\n  }\n');
  assert.ok(String(fm.writes).includes('\n'));

  // Broad glob does NOT shadow later specific literal rule
  const env = new TypeEnv();
  const fields = writesFields('{ "*.md": string, "status.md": "open\\n" | "closed\\n" }', env);
  const enc = s => new TextEncoder().encode(s);
  assert.equal(writesProblem(fields, env, 'fn', [{ path: 'notes.md', kind: 'added', after: enc('anything') }]), undefined);
  assert.match(
    writesProblem(fields, env, 'fn', [{ path: 'status.md', kind: 'added', after: enc('draft\n') }]) ?? '',
    /status\.md holds "draft\\n", which is not a "open\\n" \| "closed\\n"/,
  );

  // Is<string, P> file type in writes: checked asynchronously via crisp checker on return
  const reducer = loadVirtualNatlang({
    'r.nl': '---\nkind: directory-reducer\nargs: {}\nreturns: string\nwrites: { "summary.md": Is<string, "starts-with-heading"> }\n---\nWrite summary.md.\n',
  }, 'r.nl');
  const replies = [];
  let turn = 0;
  const model = async req => {
    if (turn > 0) replies.push(String(req.messages.at(-1)?.content ?? ''));
    turn++;
    if (turn === 1) return { calls: [['eval', { code: 'await folder.file("summary.md").writeText("no heading\\n"); return "done";', finish: true }]] };
    return { calls: [['eval', { code: 'await folder.file("summary.md").writeText("# Title\\nbody\\n"); return "done";', finish: true }]] };
  };
  const runtime = createNatlangRuntime({
    model, seed: { mode: 'backend' }, calls: false,
    refinements: { crisp: { 'starts-with-heading': v => typeof v === 'string' && v.startsWith('# ') } },
  });
  const folder = Folder.fromFiles({}, 'write');
  assert.equal(await runtime.run(() => folder.apply(reducer)), 'done');
  assert.match(replies[0], /writes-unsatisfied: summary\.md failed summary\.md/);
  assert.equal(await folder.readText('summary.md'), '# Title\nbody\n');
});

test('3.4: idiom checker respects map-form usesMap', () => {
  const rec = loadNamedFunction('/project/a.nl', virtualSourceFiles({
    'natlang.json': '{}',
    'shared/helper.ts': 'export default function helper(): string { return "ok"; }\n',
    'a.nl': '---\nargs: { x: string }\nreturns: "\'yes\' | \'no\'"\nuses:\n  myHelper: ./shared/helper\n---\nUse myHelper.\n',
  }, '/project'));
  // Has usesMap -> not a leaf -> no idiom-decision-readout warning
  assert.deepEqual(checkNlIdioms(rec, 'a.nl').filter(d => d.code === 'idiom-decision-readout'), []);
  // checkNlIdiomsIn excludes used alias from companion folder recursion
  assert.deepEqual(checkNlIdiomsIn([{ a: rec }], s => s), []);
});

test('3.5: CallStore.annotationsOf(functionId), escalate WeakMap isolation & per-field risk, semanticReduce structured carry, stepsOf fences', async () => {
  // stepsOf ignores numbered lines inside both ``` and ~~~ fences
  const body = 'Intro paragraph.\n\n```md\n1. Not a step\n2. Still fenced\n```\n\n~~~txt\n1. Also fenced\n~~~\n\nSecond paragraph.\n';
  assert.deepEqual(stepsOf(body), [
    'Intro paragraph.',
    '```md\n1. Not a step\n2. Still fenced\n```',
    '~~~txt\n1. Also fenced\n~~~',
    'Second paragraph.',
  ]);

  // semanticReduce empty items with carry: 'neuralese' returns initial directly; structured carry parses JSON
  const runtimeEmpty = createNatlangRuntime({ model: async () => ({ text: '""', calls: [] }), calls: false });
  const initObj = { count: 0, tags: ['a'] };
  const emptyOut = await runtimeEmpty.run(() => semanticReduce([], initObj, s => s, { carry: 'neuralese' }));
  assert.strictEqual(emptyOut, initObj);

  // Per-field risk floors on R3 record decisions
  const recFn = loadVirtualNatlang({
    'route.nl': '---\nargs: { text: string }\nreturns: \'{ action: "allow" | "block", tier: "low" | "high" }\'\nreadout: decision\n---\nRoute text.\n',
  }, 'route.nl');
  const localDriver = Object.assign(
    async () => ({ text: '{"action":"allow","tier":"low"}', reasoning: 'r', completion_tokens: 2 }),
    {
      decide: async req => ({
        // Four options: allow/low (0.72), block/low (0.26), allow/high (0.01), block/high (0.01)
        // Marginal P(action="block") = 0.27 >= 0.25, while overall confidence 0.72 >= floor 0.70!
        log_probs: req.options.map(opt => {
          const v = JSON.parse(opt);
          const p = v.action === 'allow' && v.tier === 'low' ? 0.72
            : v.action === 'block' && v.tier === 'low' ? 0.26 : 0.01;
          return Math.log(p);
        }),
      }),
    },
  );
  let fallbackRan = 0;
  const teacherDriver = Object.assign(
    async () => { fallbackRan++; return { text: '{"action":"block","tier":"high"}', reasoning: 't', completion_tokens: 2 }; },
    { decide: async req => { fallbackRan++; return { log_probs: req.options.map(opt => Math.log(opt.includes('"block"') && opt.includes('"high"') ? 0.97 : 0.01)) }; } },
  );
  const rt = createNatlangRuntime({ model: { driver: localDriver, decisionSamples: 0 }, models: { teacher: { driver: teacherDriver, decisionSamples: 0 } }, calls: false });
  const dotted = escalate(recFn, { to: 'teacher', floor: 0.7, risk: { 'action.block': 0.25 } });
  const rDot = await rt.run(() => dotted('suspect'));
  assert.equal(rDot.escalated, true);
  assert.equal(rDot.reason, 'risk:action.block');
  assert.ok(fallbackRan >= 1, 'the teacher answered');
  const afterDotted = fallbackRan;

  const nested = escalate(recFn, { to: 'teacher', floor: 0.7, risk: { action: { block: 0.25 } } });
  const rNest = await rt.run(() => nested('suspect'));
  assert.equal(rNest.escalated, true);
  assert.equal(rNest.reason, 'risk:action.block');
  assert.ok(fallbackRan > afterDotted, 'the teacher answered again');

  // CallStore.annotationsOf filters by functionId so global limit doesn't starve per-function history
  const storeDir = mkdtempSync(join(tmpdir(), 'natlang-store-fnid-'));
  const store = CallStore.open(storeDir);
  store.writeSettings({ minFreeBytes: 0 });
  try {
    store.annotate('call-target', 'escalation', { function: 'nl:target.nl', escalated: true, agreed: false, local_ms: 10, fallback_ms: 50 }, 'std', false);
    for (let i = 0; i < 10; i++) {
      store.annotate(`call-noise-${i}`, 'escalation', { function: 'nl:other.nl', escalated: false, local_ms: 5 }, 'std', false);
    }
    const filtered = store.annotationsOf('escalation', { limit: 3, functionId: 'nl:target.nl' });
    assert.equal(filtered.length, 1);
    assert.equal(filtered[0].call_id, 'call-target');
  } finally {
    store.close();
    rmSync(storeDir, { recursive: true, force: true });
  }
});

test('3.6: schemaOf omits prefixItems for 0-length tuple (valid JSON Schema 2020-12)', () => {
  const env = new TypeEnv();
  const emptyTuple = schemaOf(parseType('[]'), env);
  assert.deepEqual(emptyTuple, { type: 'array', minItems: 0, maxItems: 0 });
  assert.equal('prefixItems' in emptyTuple, false);

  const optTuple = schemaOf(parseType('[string?]'), env);
  assert.ok(Array.isArray(optTuple.anyOf));
  assert.deepEqual(optTuple.anyOf[0], { type: 'array', minItems: 0, maxItems: 0 });
  assert.equal('prefixItems' in optTuple.anyOf[0], false);
  assert.deepEqual(optTuple.anyOf[1], { type: 'array', prefixItems: [{ type: 'string' }], minItems: 1, maxItems: 1 });
});

test('3.7: natlang check --idiom=strict alias and robust JSON / non-numeric confidence handling', () => {
  const parsed = parseArgs(['check', '.', '--idiom=strict', '--json']);
  assert.equal(parsed.options.get('--idiom'), true);
  assert.equal(parsed.options.get('--idiom-strict'), true);

  const dir = mkdtempSync(join(tmpdir(), 'natlang-idiom-strict-'));
  const fake = mkdtempSync(join(tmpdir(), 'natlang-idiom-fake-'));
  try {
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'idiom-t', version: '0.1.0' }));
    writeFileSync(join(dir, 'a.nl'), '---\nargs: { x: string }\nreturns: string\n---\nSplit x.\n');
    writeFileSync(join(fake, 'natlang.json'), JSON.stringify({
      schema: 'natlang.package/v2', name: '@natlang/idiom-reviewer', version: '0.1.0',
      include: ['main.ts'], targets: { 'idiom-reviewer': { entry: 'main.ts', authority: ['filesystem:workspace'] } },
    }));
    writeFileSync(join(fake, 'main.ts'), `export async function main(ctx: { io: { output: { write(t: string): void } } }): Promise<number> {
      ctx.io.output.write('starting review {noise}\\n');
      ctx.io.output.write(JSON.stringify({ issues: [
        { file: 'a.nl', kind: 'mechanism-in-prose', excerpt: 'Split x.', confidence: 0.9, escalated: false, suggestion: 'Use TS helper.' },
        { file: 'a.nl', kind: 'cue-word', excerpt: 'Split x.', confidence: null, escalated: false, suggestion: 'Remove cue.' }
      ] }) + '\\n');
      return 1;
    }\n`);
    const ran = spawnSync(process.execPath, [CLI, 'check', dir, '--idiom=strict', '--json'], {
      encoding: 'utf8', env: { ...process.env, NO_COLOR: '1', NATLANG_IDIOM_REVIEWER: fake, NATLANG_SERVER: 'http://127.0.0.1:9' },
    });
    assert.equal(ran.status, 1, ran.stderr);
    const report = JSON.parse(ran.stdout);
    assert.ok(report.diagnostics.some(d => d.code === 'idiom-review-mechanism-in-prose' && d.severity === undefined));
    assert.ok(report.diagnostics.some(d => d.code === 'idiom-review-cue-word' && d.severity === 'warning' && /p=0\.00/.test(d.message)));
  } finally {
    rmSync(dir, { recursive: true, force: true });
    rmSync(fake, { recursive: true, force: true });
  }
});

test('3.8 & 3.3d: improveDirectory protects writes:/ensures:, supports guidance.json evaluation, proposal, adoption, rollback', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-improve-guidance-'));
  try {
    writeFileSync(join(dir, 'natlang.json'), JSON.stringify({ schema: 'natlang.package/v2', name: 'g-demo', version: '0.1.0' }));
    writeFileSync(join(dir, 'types.ts'), "export type Label = 'billing' | 'tech';\n");
    writeFileSync(join(dir, 'classify.nl'), [
      '---',
      'args: { ticket: string }',
      'returns: Label',
      'ensures:',
      '  - { check: "result === \'billing\' || result === \'tech\'", message: "a declared label" }',
      '---',
      'Classify ticket.',
      '',
    ].join('\n'));
    writeFileSync(join(dir, 'examples.json'), JSON.stringify([
      { name: 'e1', function: 'classify.nl', inputs: { ticket: 'invoice 1' }, expected: 'billing' },
      { name: 'e2', function: 'classify.nl', inputs: { ticket: 'invoice 2' }, expected: 'billing' },
      { name: 'e3', function: 'classify.nl', inputs: { ticket: 'invoice 3' }, expected: 'billing', protected: true },
    ]));

    // 1. Modifying ensures: is rejected
    const weakensContract = await improveDirectory({
      directory: dir,
      proposal: { files: { 'classify.nl': '---\nargs: { ticket: string }\nreturns: Label\n---\nClassify ticket.\n' } },
      evaluateSplit: async (_d, cases) => ({ passed: cases.length, total: cases.length, passRate: 1, results: [] }),
    });
    assert.equal(weakensContract.status, 'rejected');
    assert.ok(weakensContract.reasons.some(r => /writes: or ensures: contract/.test(r)));

    // 2. Invalid guidance.json schema is rejected
    const badGuidance = await improveDirectory({
      directory: dir,
      proposal: { files: { 'guidance.json': JSON.stringify({ program: 123 }) } },
      evaluateSplit: async (_d, cases) => ({ passed: cases.length, total: cases.length, passRate: 1, results: [] }),
    });
    assert.equal(badGuidance.status, 'rejected');
    assert.ok(badGuidance.reasons.some(r => /guidance\.json "program" must be a non-empty string/.test(r)));

    // 3. Model that succeeds ONLY when <natlang_program_guidance> contains "INVOICE_RULE"
    const model = async req => {
      const prompt = req.messages.map(m => String(m.content ?? '')).join('\n');
      const hasGuidance = prompt.includes('INVOICE_RULE');
      return { calls: [['return_result', { status: 'success', value: hasGuidance ? 'billing' : 'tech' }]] };
    };
    // Verify evaluateDirectorySplitWithModel loads guidance.json and wires programGuidance
    const baseEval = await evaluateDirectorySplitWithModel(dir, [
      { id: '1', source: 'examples', function: 'classify.nl', inputs: { ticket: 'invoice 1' }, expected: 'billing', group: 'g1', split: 'dev' },
    ], model);
    assert.equal(baseEval.passRate, 0);

    // 4. Run-driven proposeFromRuns in guidance mode proposes guidance.json, gates, adopts, and rolls back
    const adopted = await improveDirectory({
      directory: dir,
      seed: 0,
      model,
      requireImprovement: true,
      propose: ctx => createNatlangRuntime({ model, calls: false }).run(() => proposeFromRuns(ctx, {
        mode: 'guidance',
        evaluateSupport: cases => evaluateDirectorySplitWithModel(dir, cases, model),
        stages: {
          diagnose: async () => 'INVOICE_RULE: map invoices to billing.',
          shouldRevise: async () => true,
          revise: async () => { throw new Error('should not edit .nl in guidance mode'); },
        },
      })),
    });
    assert.equal(adopted.status, 'accepted', JSON.stringify(adopted.reasons));
    assert.deepEqual(adopted.changed, ['guidance.json']);
    assert.equal(adopted.baseline?.confirmation.passRate, 0);
    assert.equal(adopted.proposed?.confirmation.passRate, 1);
    assert.ok(existsSync(join(dir, 'guidance.json')));
    assert.match(readFileSync(join(dir, 'guidance.json'), 'utf8'), /INVOICE_RULE/);

    // Byte-identical rollback removes guidance.json
    rollbackSource(adopted.record);
    assert.equal(existsSync(join(dir, 'guidance.json')), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
