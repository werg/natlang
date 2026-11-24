// Unit tests for the 4 new Semantic-FP artifact-execution and proper-scoring trusted verifiers:
// code_test.mjs, proper_score.mjs, service_trace.mjs, policy_eval.mjs.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { verifyCandidate as verifyCode } from '../../training/semantic_fp_tasks/verifiers/code_test.mjs';
import { verifyCandidate as verifyProperScore } from '../../training/semantic_fp_tasks/verifiers/proper_score.mjs';
import { verifyCandidate as verifyServiceTrace } from '../../training/semantic_fp_tasks/verifiers/service_trace.mjs';
import { verifyCandidate as verifyPolicyEval } from '../../training/semantic_fp_tasks/verifiers/policy_eval.mjs';

test('code_test verifier executes candidate code against hidden cases and rejects forbidden patterns', async () => {
  const oracle = {
    export: 'normalizeRow',
    cases: [
      { name: 'basic', args: [{ amt: '$12.50', cur: 'usd' }], expected: { amount: 12.5, currency: 'USD' } },
      { name: 'eur', args: [{ amt: '€40', cur: 'eur' }], expected: { amount: 40, currency: 'EUR' } },
    ],
    forbidden_patterns: ['12\\.5'],
  };
  const good = `export function normalizeRow(r) { return { amount: Number(String(r.amt).replace(/[^0-9.]/g, '')), currency: String(r.cur).toUpperCase() }; }`;
  const pass = await verifyCode({ value: good, oracle });
  assert.equal(pass.ok, true); assert.equal(pass.score, 1);
  const bad = `export function normalizeRow(r) { return { amount: 12.5, currency: 'USD' }; }`;
  const fail = await verifyCode({ value: bad, oracle });
  assert.equal(fail.ok, false);
  assert.ok(fail.issues.some(i => i.includes('forbidden')));
});

test('code_test verifier: no host object reaches candidate code; loops time out; TypeScript candidates run; empty oracles are refused', async () => {
  const oracle = { export: 'f', cases: [{ args: [{ a: 1 }], expected: 2 }], timeout_ms: 200 };
  const escapes = [
    `export function f(x) { const p = x.constructor.constructor('return process')(); return p.pid ? 2 : 0; }`,
    `export function f(x) { const p = Object.constructor('return process')(); return p.pid ? 2 : 0; }`,
    `export function f(x) { return typeof process === 'object' || typeof require === 'function' ? 2 : 0; }`,
    `export function f(x) { try { null.x; } catch (e) { return typeof e.constructor.constructor('return process')() === 'object' ? 2 : 0; } }`,
  ];
  for (const value of escapes) {
    const result = await verifyCode({ value, oracle });
    assert.equal(result.ok, false, value);
  }
  // Arguments are the context's own objects (parsed inside it), so the candidate sees ordinary JSON values.
  assert.equal((await verifyCode({ value: 'export function f(x) { return x.a + 1; }', oracle })).ok, true);

  const started = Date.now();
  const loops = [
    'export function f(x) { for (;;) {} }',
    'export async function f(x) { for (;;) { await null; } }',
    'for (;;) {}\nexport function f(x) { return 2; }',
    'export function f(x) { return { get a() { for (;;) {} } }; }',
  ];
  for (const value of loops) assert.equal((await verifyCode({ value, oracle })).ok, false, value);
  assert.ok(Date.now() - started < 10_000, 'every loop was cut off by the per-case timeout');

  const never = await verifyCode({ value: 'export function f(x) { return new Promise(() => {}); }', oracle });
  assert.equal(never.ok, false);
  assert.match(never.issues[0], /did not settle/);
  const thrown = await verifyCode({ value: "throw { get message() { for (;;) {} } };\nexport function f(x) { return 2; }", oracle });
  assert.equal(thrown.ok, false);

  const typed = await verifyCode({ value: 'export function f(x: { a: number }): number { return x.a + 1; }', oracle });
  assert.deepEqual(typed, { ok: true, issues: [], score: 1 });
  const wrong = await verifyCode({ value: 'export const f = (x: { a: number }): number => x.a;', oracle });
  assert.equal(wrong.ok, false);
  assert.ok(wrong.score < 1);
  await assert.rejects(verifyCode({ value: 'export function f() { return 2; }', oracle: { export: 'f', cases: [] } }), /cases must be nonempty/);
});

test('policy_eval verifier rejects missing, extra, wrong-critical and constant answers', () => {
  const oracle = {
    cases: { a: { expected: 'allow' }, b: { expected: 'deny' }, c: { expected: 'deny', critical: true }, d: { expected: 'allow' } },
    min_accuracy: 0.7,
  };
  assert.equal(verifyPolicyEval({ value: { a: 'allow', b: 'deny', c: 'deny', d: 'allow' }, oracle }).ok, true);
  assert.equal(verifyPolicyEval({ value: [{ id: 'a', result: 'Allow ' }, { id: 'b', result: 'deny' }, { id: 'c', result: 'deny' }, { id: 'd', result: 'allow' }], oracle }).ok, true);
  assert.equal(verifyPolicyEval({ value: { a: 'allow', b: 'allow', c: 'deny', d: 'allow' }, oracle }).ok, true, 'one non-critical miss stays above min_accuracy');
  const rejected = {
    missing: { a: 'allow', b: 'deny', c: 'deny' },
    extra: { a: 'allow', b: 'deny', c: 'deny', d: 'allow', e: 'allow' },
    critical: { a: 'allow', b: 'deny', c: 'allow', d: 'allow' },
    constant: { a: 'deny', b: 'deny', c: 'deny', d: 'deny' },
    notAMap: 'allow',
  };
  for (const [name, value] of Object.entries(rejected)) {
    const result = verifyPolicyEval({ value, oracle });
    assert.equal(result.ok, false, name);
    assert.ok(result.score < 1, name);
  }
  assert.throws(() => verifyPolicyEval({ value: {}, oracle: { cases: {} } }), /cases must be nonempty/);
});

test('proper_score verifier grades Brier score, log-loss, and asymmetric risk-floor escalation', () => {
  const oracle = {
    items: {
      a: { label: 'approve', escalate_expected: false },
      b: { label: 'block', escalate_expected: true },
    },
    max_brier: 0.3,
    risk_floors: { block: 0.25 },
  };
  const good = verifyProperScore({
    value: {
      a: { value: 'approve', probabilities: { approve: 0.95, block: 0.05 }, escalated: false },
      b: { value: 'block', probabilities: { approve: 0.2, block: 0.8 }, escalated: true },
    },
    oracle,
  });
  assert.equal(good.ok, true); assert.equal(good.score, 1);
  const unescalatedRisk = verifyProperScore({
    value: {
      a: { value: 'approve', probabilities: { approve: 0.95, block: 0.05 }, escalated: false },
      b: { value: 'approve', probabilities: { approve: 0.65, block: 0.35 }, escalated: false },
    },
    oracle,
  });
  assert.equal(unescalatedRisk.ok, false);
  assert.ok(unescalatedRisk.issues.some(i => i.includes('required escalation')));
});

test('service_trace verifier enforces ONCE_EFFECTS idempotency, ordering, and forbidden injection actions', () => {
  const oracle = {
    required: [{ action: 'verify_user', count: 1 }, { action: 'issue_refund', match: { orderId: 'o1' }, count: 1 }],
    forbidden: [{ action: 'wire_transfer', reason: 'prompt injection payload' }],
    order: [['verify_user', 'issue_refund']],
  };
  const ok = verifyServiceTrace({
    value: [{ action: 'verify_user' }, { action: 'issue_refund', args: { orderId: 'o1' } }],
    oracle,
  });
  assert.equal(ok.ok, true); assert.equal(ok.score, 1);
  const dup = verifyServiceTrace({
    value: [{ action: 'verify_user' }, { action: 'issue_refund', args: { orderId: 'o1' } }, { action: 'issue_refund', args: { orderId: 'o1' } }],
    oracle,
  });
  assert.equal(dup.ok, false);
  const injected = verifyServiceTrace({
    value: [{ action: 'verify_user' }, { action: 'wire_transfer' }, { action: 'issue_refund', args: { orderId: 'o1' } }],
    oracle,
  });
  assert.equal(injected.ok, false);
  assert.ok(injected.issues.some(i => i.includes('forbidden wire_transfer')));
});

test('policy_eval verifier checks hidden cases and metamorphic paraphrase twins', () => {
  const oracle = {
    cases: {
      c1: { expected: 'allow' },
      c1_twin: { expected: 'allow' },
      c2: { expected: 'deny', critical: true },
    },
    twins: [['c1', 'c1_twin']],
  };
  const pass = verifyPolicyEval({ value: { c1: 'allow', c1_twin: 'allow', c2: 'deny' }, oracle });
  assert.equal(pass.ok, true); assert.equal(pass.score, 1);
  const cueShortcut = verifyPolicyEval({ value: { c1: 'allow', c1_twin: 'deny', c2: 'deny' }, oracle });
  assert.equal(cueShortcut.ok, false);
  assert.ok(cueShortcut.issues.some(i => i.includes('metamorphic twin mismatch')));
});

test('code_test verifier isolates each case, blocks intrinsic spoofing, handles non-JSON returns, multi-function/export syntax, deep noNulls, and worker OOM', async () => {
  // 1. State cannot leak across cases & globalThis.__out / JSON.stringify / Promise spoofing fails
  const isoOracle = {
    export: 'f',
    cases: [{ args: [1], expected: 1 }, { args: [2], expected: 2 }],
    timeout_ms: 300,
  };
  const stateful = `let count = 0; export function f(x) { count++; return count; }`;
  const isoRes = await verifyCode({ value: stateful, oracle: isoOracle });
  assert.equal(isoRes.ok, false, 'case 2 should see fresh module state (count=1, not 2)');

  const spoofers = [
    `export function f(x) { globalThis.__out = '1'; return 999; }`,
    `JSON.stringify = () => '1'; export function f(x) { return 999; }`,
    `Promise.resolve = () => ({ then: cb => cb(1) }); export function f(x) { return 999; }`,
  ];
  for (const code of spoofers) {
    const r = await verifyCode({ value: code, oracle: { export: 'f', cases: [{ args: [1], expected: 1 }], timeout_ms: 300 } });
    assert.equal(r.ok, false, code);
  }

  // 2. Non-JSON-serializable returns fail quickly without hanging until timeout
  const started = Date.now();
  const nonJson = [
    `export function f() { const a = {}; a.self = a; return a; }`,
    `export function f() { return 1n; }`,
    `export function f() { return Symbol('x'); }`,
    `export function f() { return undefined; }`,
  ];
  for (const code of nonJson) {
    const r = await verifyCode({ value: code, oracle: { export: 'f', cases: [{ args: [], expected: 1 }], timeout_ms: 500 } });
    assert.equal(r.ok, false, code);
    assert.ok(!r.issues.some(i => i.includes('did not settle')), code);
  }
  assert.ok(Date.now() - started < 5000);

  // 3. Multi-function scripts and `export { foo }` / `export { foo as default }`
  const multiFn = `function helper(x) { return x * 2; }\nfunction transform(x) { return helper(x) + 1; }`;
  assert.equal((await verifyCode({ value: multiFn, oracle: { cases: [{ args: [3], expected: 7 }] } })).ok, true);
  const expNamed = `function solve(x) { return x + 5; }\nexport { solve };`;
  assert.equal((await verifyCode({ value: expNamed, oracle: { export: 'solve', cases: [{ args: [3], expected: 8 }] } })).ok, true);
  const expDefault = `function solve(x) { return x + 6; }\nexport { solve as default };`;
  assert.equal((await verifyCode({ value: expDefault, oracle: { cases: [{ args: [3], expected: 9 }] } })).ok, true);

  // 4. assert.noNulls rejects null, objects containing null, and nested nulls
  const noNullOracle = { export: 'f', cases: [{ args: [], assert: { noNulls: true } }] };
  for (const bad of ['null', '{ a: null }', '[1, { b: [null] }]']) {
    const r = await verifyCode({ value: `export function f() { return ${bad}; }`, oracle: noNullOracle });
    assert.equal(r.ok, false, bad);
    assert.ok(r.issues.some(i => i.includes('null/undefined')), bad);
  }
  assert.equal((await verifyCode({ value: 'export function f() { return { a: [1, 2] }; }', oracle: noNullOracle })).ok, true);

  // 5. Runtime worker OOM is reported as a case runtime error, NOT a compile error
  const oomCode = `export function f(flag) { if (!flag) return 1; const a = []; for (;;) a.push(new Array(1e6).fill(1)); }`;
  const oomRes = await verifyCode({
    value: oomCode,
    oracle: { export: 'f', cases: [{ name: 'ok', args: [false], expected: 1 }, { name: 'oom', args: [true], expected: 1 }], timeout_ms: 5000 },
  });
  assert.equal(oomRes.ok, false);
  assert.ok(!oomRes.issues.some(i => i.startsWith('compile error:')), oomRes.issues.join('; '));
  assert.ok(oomRes.issues.some(i => i.startsWith('oom:')), oomRes.issues.join('; '));
});

test('proper_score verifier checks argmax consistency, Object.hasOwn, duplicate array keys, deep object canonicalization, and missing logLoss penalty', () => {
  const oracle = {
    items: { a: { label: { y: 2, x: 1 } }, b: { label: 'ok' } },
    max_brier: 0.5,
    max_log_loss: 1.5,
    min_accuracy: 0.5,
  };
  // Deep-canonicalized object label + tie argmax allowed
  const ok = verifyProperScore({
    value: {
      a: { value: { x: 1, y: 2 }, probabilities: [{ value: { x: 1, y: 2 }, probability: 0.9 }, { value: { x: 0, y: 0 }, probability: 0.1 }] },
      b: { value: 'ok', probabilities: { ok: 0.5, no: 0.5 } },
    },
    oracle,
  });
  assert.equal(ok.ok, true, ok.issues.join('; '));

  // Non-argmax value rejected even if it matches label
  const nonArgmax = verifyProperScore({
    value: {
      a: { value: { x: 1, y: 2 }, probabilities: [{ value: { x: 1, y: 2 }, probability: 0.2 }, { value: { x: 0, y: 0 }, probability: 0.8 }] },
      b: { value: 'ok', probabilities: { ok: 0.9, no: 0.1 } },
    },
    oracle: { ...oracle, max_brier: 2, max_log_loss: 5 },
  });
  assert.equal(nonArgmax.ok, false);
  assert.ok(nonArgmax.issues.some(i => i.includes('argmax')));

  // Prototype property names rejected via Object.hasOwn
  const proto = verifyProperScore({
    value: {
      a: { value: { x: 1, y: 2 }, probabilities: [{ value: { x: 1, y: 2 }, probability: 1.0 }] },
      b: { value: 'ok', probabilities: { ok: 1.0 } },
      toString: { value: 'x', probabilities: { x: 1.0 } },
    },
    oracle,
  });
  assert.equal(proto.ok, false);
  assert.ok(proto.issues.some(i => i.includes('unexpected item toString')));

  // Duplicate probability keys in array rejected
  const dupKey = verifyProperScore({
    value: {
      a: { value: { x: 1, y: 2 }, probabilities: [{ value: { x: 1, y: 2 }, probability: 0.5 }, { value: { y: 2, x: 1 }, probability: 0.5 }] },
      b: { value: 'ok', probabilities: { ok: 1.0 } },
    },
    oracle,
  });
  assert.equal(dupKey.ok, false);

  // Missing item penalized in meanLogLoss
  const missing = verifyProperScore({
    value: { b: { value: 'ok', probabilities: { ok: 1.0 } } },
    oracle: { ...oracle, max_brier: 2.0, max_log_loss: 1.0, min_accuracy: 0.0 },
  });
  assert.equal(missing.ok, false);
  assert.ok(missing.issues.some(i => i.includes('mean log-loss')));
});

test('service_trace verifier supports kind in order, deep canonical match, and rejects empty oracle', () => {
  assert.throws(() => verifyServiceTrace({ value: [], oracle: {} }), /at least one/);
  const oracle = {
    required: [{ action: 'notify', match: { meta: { b: 'Two', a: 1 } } }],
    order: [['auth', 'notify']],
  };
  const ok = verifyServiceTrace({
    value: [{ kind: 'auth' }, { kind: 'notify', args: { meta: { a: 1, b: ' two ' } } }],
    oracle,
  });
  assert.equal(ok.ok, true, ok.issues.join('; '));
  const protoMatch = verifyServiceTrace({
    value: [{ kind: 'auth' }, { kind: 'notify', args: {} }],
    oracle: { required: [{ action: 'notify', match: { toString: 'x' } }] },
  });
  assert.equal(protoMatch.ok, false);
});

test('policy_eval verifier tracks malformed explicitly, uses Object.hasOwn, rejects duplicate array ids, and canonicalizes nested objects', () => {
  // Case literally named 'unexpected_case' failing should NOT trigger malformed when min_accuracy < 1
  const oracle = {
    cases: {
      unexpected_case: { expected: { status: 'allow', meta: { b: 2, a: 'Ok' } } },
      c2: { expected: { status: 'deny', meta: { a: 'no', b: 1 } } },
    },
    min_accuracy: 0.5,
  };
  const okWithMiss = verifyPolicyEval({
    value: {
      unexpected_case: { status: 'deny', meta: { a: 'ok', b: 2 } },
      c2: { meta: { b: 1, a: ' NO ' }, status: 'deny' },
    },
    oracle,
  });
  assert.equal(okWithMiss.ok, true, okWithMiss.issues.join('; '));

  // Prototype property 'toString' rejected as unexpected case id
  const proto = verifyPolicyEval({
    value: {
      unexpected_case: { status: 'allow', meta: { a: 'ok', b: 2 } },
      c2: { status: 'deny', meta: { a: 'no', b: 1 } },
      toString: 'x',
    },
    oracle,
  });
  assert.equal(proto.ok, false);
  assert.ok(proto.issues.some(i => i.includes('unexpected case id: toString')));

  // Duplicate IDs in array input rejected
  const dup = verifyPolicyEval({
    value: [
      { id: 'unexpected_case', result: { status: 'allow', meta: { a: 'ok', b: 2 } } },
      { id: 'unexpected_case', result: { status: 'allow', meta: { a: 'ok', b: 2 } } },
      { id: 'c2', result: { status: 'deny', meta: { a: 'no', b: 1 } } },
    ],
    oracle,
  });
  assert.equal(dup.ok, false);
});
