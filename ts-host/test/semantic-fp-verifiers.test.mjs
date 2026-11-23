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
