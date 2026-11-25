import test from 'node:test';
import assert from 'node:assert/strict';

import {
  CompiledSiteCache,
  chooseTierCrisp,
  compileOnCreateClosure,
  compiledCacheKey,
  createChooseTierPolicy,
} from '../dist/runtime/jit.js';
import {
  COMPILED_DECISION_SCHEMA,
  calibrateTemperature,
  scoreCompiledDecision,
} from '../dist/tune/decision-compile.js';
import { verifyCrispCandidate, wrapCrispFastPath } from '../dist/tune/crisp-promote.js';
import {
  createWhichScopesPolicy,
  createWorthRememberingPolicy,
  formatStageADemonstrations,
  worthRememberingCrisp,
} from '../dist/memory/experience-kb.js';
import { hoistReturnContract } from '../dist/native/contract-hoist.js';
import { rankBranchCandidates, runPrefixSharedBranchSearch } from '../dist/native/branch-search.js';
import { TypeEnv, parseType } from '../dist/native/types.js';

test('JIT chooseTier policy, provenance cache, and compile-on-creation closures', async () => {
  const baseStats = {
    siteId: 'triage::route.nl',
    callCount: 1,
    verifiedExampleCount: 0,
    refinementFailureRate: 0,
    meanDecisionConfidence: 0.92,
    repairRate: 0,
    isSynthesizedLiteral: false,
    hasCrispFastPath: false,
    hasCompiledWeights: false,
    hasTier15Weights: false,
    hasTier2Weights: false,
    compilerAvailable: true,
  };

  assert.equal(chooseTierCrisp(baseStats), '0');
  assert.equal(chooseTierCrisp({ ...baseStats, callCount: 2 }), '0.5');
  assert.equal(chooseTierCrisp({ ...baseStats, isSynthesizedLiteral: true }), '1');
  assert.equal(chooseTierCrisp({ ...baseStats, verifiedExampleCount: 8, refinementFailureRate: 0.25 }), '1.5');
  assert.equal(chooseTierCrisp({ ...baseStats, verifiedExampleCount: 120, repairRate: 0.2 }), '2');
  assert.equal(chooseTierCrisp({ ...baseStats, hasCrispFastPath: true }), 'crisp');

  const policy = createChooseTierPolicy({ mode: 'crisp' });
  assert.equal(await policy({ ...baseStats, callCount: 5 }), '1');

  const cache = new CompiledSiteCache(2);
  let compileCalls = 0;
  const compileFn = async () => {
    compileCalls += 1;
    return { adapterId: `ad-${compileCalls}`, nativeStateId: `ns-${compileCalls}` };
  };

  const e1 = await compileOnCreateClosure(
    cache,
    {
      siteId: 'screening::meets',
      resolvedHeaderText: 'Does candidate meet requirement?',
      partialArgsJson: JSON.stringify({ minYears: 5 }),
      compilerSha: 'comp-1',
      policySha: 'pol-1',
    },
    compileFn,
  );
  // Identical partial args hit cache
  const e2 = await compileOnCreateClosure(
    cache,
    {
      siteId: 'screening::meets',
      resolvedHeaderText: 'Does candidate meet requirement?',
      partialArgsJson: JSON.stringify({ minYears: 5 }),
      compilerSha: 'comp-1',
      policySha: 'pol-1',
    },
    compileFn,
  );
  assert.equal(compileCalls, 1);
  assert.equal(e2.cacheKey, e1.cacheKey);
  assert.equal(e2.hits, 1);

  // Different partial args produce distinct cache key & compile separately
  const e3 = await compileOnCreateClosure(
    cache,
    {
      siteId: 'screening::meets',
      resolvedHeaderText: 'Does candidate meet requirement?',
      partialArgsJson: JSON.stringify({ minYears: 10 }),
      compilerSha: 'comp-1',
      policySha: 'pol-1',
    },
    compileFn,
  );
  assert.equal(compileCalls, 2);
  assert.notEqual(e3.cacheKey, e1.cacheKey);
  assert.equal(cache.invalidatePolicy('pol-1'), 2);
  assert.equal(cache.size, 0);
  assert.notEqual(
    compiledCacheKey({ siteId: 's', headerSha: 'h', compilerSha: 'c', policySha: 'p1' }),
    compiledCacheKey({ siteId: 's', headerSha: 'h', compilerSha: 'c', policySha: 'p2' }),
  );
});

test('Compiled decision temperature calibration and asymmetric risk settlement', () => {
  const samples = [
    { logits: [5, 0, 0], targetIndex: 0 },
    { logits: [5, 0, 0], targetIndex: 0 },
    { logits: [5, 0, 0], targetIndex: 0 },
    { logits: [5, 0, 0], targetIndex: 1 },
    { logits: [0, 5, 0], targetIndex: 1 },
    { logits: [0, 5, 0], targetIndex: 1 },
    { logits: [0, 5, 0], targetIndex: 1 },
    { logits: [0, 5, 0], targetIndex: 2 },
  ];
  const cal = calibrateTemperature(samples);
  assert.ok(cal.temperature > 1.2, `expected temperature > 1.2, got ${cal.temperature}`);
  assert.ok(cal.nllAfter < cal.nllBefore);
  assert.ok(cal.eceAfter < cal.eceBefore);

  const artifact = {
    schema: COMPILED_DECISION_SCHEMA,
    siteId: 'pr::review.nl',
    labels: ['approve', 'block'],
    temperature: 1.0,
    fastThreshold: 0.70,
    riskThresholds: { block: 0.25 },
  };
  // approve ~0.731, block ~0.269 >= 0.25 risk floor -> must NOT settle
  const risky = scoreCompiledDecision(artifact, [1.0, 0.0]);
  assert.equal(risky.settled, false);
  assert.equal(risky.reason, 'risk-floor');

  // approve ~0.953, block ~0.047 < 0.25 -> fast-path settles
  const safe = scoreCompiledDecision(artifact, [3.0, 0.0]);
  assert.equal(safe.settled, true);
  assert.equal(safe.label, 'approve');
  assert.equal(safe.reason, 'fast-path');
});

test('Crisp fast-path verification enforces 0 FP on dev & confirmation and falls back cleanly', async () => {
  const makeCases = (prefix) => [
    { id: `${prefix}-1`, args: ['[SPAM] buy now'], expected: 'spam' },
    { id: `${prefix}-2`, args: ['invoice #123 failed'], expected: 'billing' },
    { id: `${prefix}-3`, args: ['app crashed on login'], expected: 'technical' },
    { id: `${prefix}-4`, args: ['[SPAM] free prize'], expected: 'spam' },
  ];

  const goodCrisp = (text) => (text.startsWith('[SPAM]') ? { hit: true, value: 'spam' } : { hit: false });
  const okReport = await verifyCrispCandidate(goodCrisp, { dev: makeCases('d'), confirmation: makeCases('c') });
  assert.equal(okReport.promoted, true);
  assert.equal(okReport.dev.falsePositives, 0);
  assert.equal(okReport.confirmation.coverage, 0.5);

  const badCrisp = (text) => (text.includes('failed') ? { hit: true, value: 'technical' } : { hit: false });
  const badReport = await verifyCrispCandidate(badCrisp, { dev: makeCases('d'), confirmation: makeCases('c') });
  assert.equal(badReport.promoted, false);
  assert.ok(badReport.dev.falsePositives > 0);

  const wrapped = wrapCrispFastPath(goodCrisp, async (t) => (t.includes('invoice') ? 'billing' : 'technical'));
  assert.equal(await wrapped.call('[SPAM] hi'), 'spam');
  assert.equal(await wrapped.call('invoice #9'), 'billing');
  assert.equal(wrapped.metrics.crispHits, 1);
  assert.equal(wrapped.metrics.neuralFallbacks, 1);
});

test('experience-KB policies are pluggable parts; Stage-A formatting of retrieved hits', async () => {
  const scopes = await createWhichScopesPolicy()({ taskGroup: 'sql', repoSha: 'abc', siteId: 'q.nl' });
  assert.deepEqual(scopes, ['natlang:global', 'group:sql', 'repo:abc', 'fn:q.nl']);
  assert.equal(worthRememberingCrisp({ taskId: 'h1', verifiedSuccess: true, wasRepairWin: false, heldExcluded: true, textLength: 50 }), false);
  assert.equal(worthRememberingCrisp({ taskId: 't1', verifiedSuccess: true, wasRepairWin: false, heldExcluded: false, textLength: 50 }), true);

  // nl and shadow need the natural-language policy; the crisp rule never stands in for it
  assert.throws(() => createWhichScopesPolicy({ mode: 'nl' }), /kb:whichScopes.*pass it as `nl`/);
  assert.throws(() => createWorthRememberingPolicy({ mode: 'shadow' }), /kb:worthRemembering/);
  const nlScopes = createWhichScopesPolicy({ mode: 'nl', nl: () => ['natlang:global'] });
  assert.deepEqual(await nlScopes({ taskGroup: 'sql' }), ['natlang:global']);

  const formatted = formatStageADemonstrations([
    { recordId: 'r-peer', scope: 'group:sql', kind: 'repair', text: 'Always use COALESCE on nullable window sums.',
      taskIds: ['task-B'], similarity: 0.9, weight: 0.71 },
  ]);
  assert.match(formatted, /\[group:sql \| repair \| w=0\.71\] Always use COALESCE/);
  assert.equal(formatStageADemonstrations([]), '');
});

test('Contract hoisting extracts union literals/schemas and fast-checks return values', () => {
  const env = new TypeEnv({});
  const unionContract = hoistReturnContract(parseType('"approve" | "block" | "escalate"'), env);
  assert.deepEqual(unionContract.allowedLiterals, ['"approve"', '"block"', '"escalate"']);
  assert.equal(unionContract.fastCheck('approve').ok, true);
  assert.equal(unionContract.fastCheck('unknown').ok, false);

  const recContract = hoistReturnContract(parseType('{ id: number; status: "ok" | "err" }'), env);
  assert.equal(recContract.jsonSchema.type, 'object');
  assert.deepEqual(recContract.jsonSchema.required, ['id', 'status']);
  assert.equal(recContract.fastCheck({ id: 1, status: 'ok' }).ok, true);
  assert.equal(recContract.fastCheck({ id: 'bad', status: 'ok' }).ok, false);
});

test('Prefix-shared branch@4 ranks crisp-first, verifier-second, and fires onBranchWin', async () => {
  const wins = [];
  const baseRequest = { messages: [{ role: 'user', content: 'solve' }], tools: [], seed: 42, temperature: 0 };
  const sampleTurn = async (req, idx) => {
    if (idx === 0) return { calls: [['return_result', { status: 'success', value: 'bad-label' }]], text: '' };
    if (idx === 1) return { calls: [], text: '', malformed_call: 'bad_json' };
    if (idx === 2) return { calls: [['return_result', { status: 'success', value: 'ok' }]], text: '' };
    return { calls: [['return_result', { status: 'success', value: 'ok' }]], text: '' };
  };
  const scores = [0.99, 0.95, 0.62, 0.88];
  const res = await runPrefixSharedBranchSearch(
    baseRequest,
    sampleTurn,
    {
      branches: 4,
      scoreBranch: (_turn, idx) => scores[idx],
      validateReturnValue: (v) => (v === 'ok' ? { ok: true } : { ok: false, reason: 'invalid label' }),
      onBranchWin: (w) => wins.push(w),
    },
    { callId: 'call-1', turnNumber: 1 },
  );
  // Branch 0 & 1 fail crisp checks despite higher verifier scores; branch 3 beats branch 2 on verifier score!
  assert.equal(res.selected.index, 3);
  assert.equal(res.branchWin, true);
  assert.equal(wins.length, 1);
  assert.equal(wins[0].pass1CrispOk, false);
  assert.equal(wins[0].selectedCrispOk, true);
  assert.equal(wins[0].selectedScore, 0.88);
  assert.deepEqual(rankBranchCandidates(res.ranked).map(c => c.index), [3, 2, 0, 1]);
});
