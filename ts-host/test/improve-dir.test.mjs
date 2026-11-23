/**
 * WS8 N4: `natlang improve DIR` directory-form case construction, deterministic split separation,
 * accept/reject verification transaction, and byte-identical rollback via `improvement/adoption.ts`.
 */
import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { CallStore, rollbackSource } from '../dist/index.js';
import { CALLS_VERSION } from '../dist/calls/types.js';
import { hexDigest } from '../dist/native/hash.js';
import { buildDirectoryCasePack, improveDirectory } from '../dist/improvement/program.js';
import { gatherRunEvidence, trustedCallOutput } from '../dist/improvement/run-evidence.js';
import { proposeFromRuns } from '../dist/improvement/propose-from-runs.js';
import { builtin } from '../dist/runtime/builtin.js';
import { createNatlangRuntime } from '../dist/index.js';
import { scriptedModel, withGeneratedDecisions } from './support/natlang.mjs';
import { improvementCommand } from '../dist/cli/improvement.js';

function createFixtureProgram() {
  const dir = mkdtempSync(join(tmpdir(), 'natlang-improve-dir-fixture-'));
  writeFileSync(join(dir, 'natlang.json'), JSON.stringify({
    schema: 'natlang.package/v2',
    name: 'triage-mini',
    version: '0.1.0',
  }, null, 2) + '\n');
  writeFileSync(join(dir, 'types.ts'), "export type Label = 'billing' | 'technical' | 'spam';\n");
  writeFileSync(join(dir, 'classify.nl'), `---
description: Classify a support ticket.
args: { ticket: string }
returns: Label
readout: decision
---
Pick the label for ticket.
`);
  writeFileSync(join(dir, 'examples.json'), JSON.stringify([
    { name: 'ex1', function: 'classify.nl', inputs: { ticket: 'Charged twice on invoice 101' }, expected: 'billing' },
    { name: 'ex2', function: 'classify.nl', inputs: { ticket: 'Need PDF copy of receipt' }, expected: 'billing' },
    { name: 'ex3', function: 'classify.nl', inputs: { ticket: 'Checkout returns 500 error' }, expected: 'technical' },
    { name: 'ex4', function: 'classify.nl', inputs: { ticket: 'App crashes opening settings' }, expected: 'technical' },
    { name: 'ex5', function: 'classify.nl', inputs: { ticket: 'WIN FREE WATCHES NOW!!!' }, expected: 'spam' },
    { name: 'ex6', function: 'classify.nl', inputs: { ticket: 'Protected outage check' }, expected: 'technical', protected: true },
  ], null, 2) + '\n');
  return dir;
}

function freshStore() {
  const root = mkdtempSync(join(tmpdir(), 'natlang-improve-dir-store-'));
  const store = CallStore.open(root);
  store.writeSettings({ minFreeBytes: 0, specialization: 'off' });
  return {
    root,
    store,
    dispose() {
      try { store.close(); } catch {} finally { rmSync(root, { recursive: true, force: true }); }
    },
  };
}

function recordCall(store, { callId, programRoot, source, ticket, output }) {
  const instrText = JSON.stringify('Pick the label for ticket.');
  const inputText = JSON.stringify(ticket);
  const outputText = JSON.stringify(output);
  const instrHash = hexDigest(instrText);
  const inputHash = hexDigest(inputText);
  const outputHash = hexDigest(outputText);
  const blobs = new Map([
    [instrHash, instrText],
    [inputHash, inputText],
    [outputHash, outputText],
  ]);
  store.record({
    version: CALLS_VERSION,
    call_id: callId,
    parent_call_id: null,
    parent_action_index: null,
    task_id: 't1',
    program_id: 'triage-mini',
    build_hash: null,
    program_root: programRoot,
    definition: {
      id: 'classify.nl',
      name: 'classify',
      source,
      key: 'k1',
      interface: 'i1',
      site: 'named',
      subtype: 'function',
      params: [{ name: 'ticket', type: 'string' }],
      returns: 'Label',
      instructions: { complete: true, hash: instrHash, bytes: instrText.length },
      types: { Label: "'billing' | 'technical' | 'spam'" },
      readout: 'decision',
    },
    executor: { kind: 'agent', model_id: 'test-model', model_revision: null },
    inputs: { ticket: { complete: true, hash: inputHash, bytes: inputText.length } },
    captures: {},
    capture_writes: [],
    output: { complete: true, hash: outputHash, bytes: outputText.length },
    outcome: 'done',
    detail: '',
    started_at: '2026-11-23T10:00:00.000Z',
    ended_at: '2026-11-23T10:00:01.000Z',
    effects: [],
    folder: null,
    approach: { evals: [], hash: null },
    cost: { model_requests: 1, tokens_in: 100, tokens_out: 10, wall_ms: 50, turns: 1, evals: 0 },
    features: {},
    audit_of: null,
    events: null,
  }, blobs);
}

test('buildDirectoryCasePack merges examples.json and CallStore with deterministic disjoint splits', () => {
  const dir = createFixtureProgram();
  const { store, dispose } = freshStore();
  try {
    // One duplicate of ex1 (must be deduplicated) and two fresh recorded calls.
    recordCall(store, { callId: 'c-dup', programRoot: dir, source: 'classify.nl', ticket: 'Charged twice on invoice 101', output: 'spam' });
    recordCall(store, { callId: 'c-new-1', programRoot: dir, source: 'classify.nl', ticket: 'Refund for cancelled seat', output: 'billing' });
    recordCall(store, { callId: 'c-new-2', programRoot: dir, source: 'classify.nl', ticket: 'API returns 502 bad gateway', output: 'technical' });

    const packA = buildDirectoryCasePack(dir, { store, seed: 42 });
    const packB = buildDirectoryCasePack(dir, { store, seed: 42 });
    const packC = buildDirectoryCasePack(dir, { store, seed: 99 });

    assert.equal(packA.sources.examples, 6);
    assert.equal(packA.sources.callStore, 2, 'duplicate input from store is deduplicated');
    assert.equal(packA.cases.length, 8);
    assert.ok(packA.splits.support.length >= 1);
    assert.ok(packA.splits.dev.length >= 1);
    assert.ok(packA.splits.confirmation.length >= 1);

    // Deterministic for the same seed.
    assert.deepEqual(
      packA.cases.map(c => [c.id, c.split]),
      packB.cases.map(c => [c.id, c.split]),
    );
    // Protected example always lands in confirmation regardless of seed.
    assert.ok(packA.splits.confirmation.some(c => c.name === 'ex6'));
    assert.ok(packC.splits.confirmation.some(c => c.name === 'ex6'));

    // Disjoint groups across support, dev, confirmation.
    const sGroups = new Set(packA.splits.support.map(c => c.group));
    const dGroups = new Set(packA.splits.dev.map(c => c.group));
    const cGroups = new Set(packA.splits.confirmation.map(c => c.group));
    for (const g of sGroups) { assert.ok(!dGroups.has(g)); assert.ok(!cGroups.has(g)); }
    for (const g of dGroups) { assert.ok(!cGroups.has(g)); }

    // Different seed changes regular-case split assignment.
    const splitMap = p => Object.fromEntries(p.cases.filter(c => !c.protected).map(c => [c.id, c.split]));
    assert.notDeepEqual(splitMap(packA), splitMap(packC));
  } finally {
    dispose();
    rmSync(dir, { recursive: true, force: true });
  }
});

test('improveDirectory accepts a verified proposal and rolls back byte-identically', async () => {
  const dir = createFixtureProgram();
  const originalClassify = readFileSync(join(dir, 'classify.nl'), 'utf8');
  const improvedClassify = originalClassify + 'Prefer technical for outages and billing for invoices.\n';
  const guidanceText = 'Always distinguish payment failures (billing) from server 500 errors (technical).\n';
  try {
    const evaluateSplit = async (targetDir, cases) => {
      const text = readFileSync(join(targetDir, 'classify.nl'), 'utf8');
      const improved = text.includes('Prefer technical for outages');
      // Baseline passes half; improved passes all.
      const results = cases.map((c, index) => ({
        id: c.id,
        function: c.function,
        ok: improved || index % 2 === 0,
        expected: c.expected,
        received: improved || index % 2 === 0 ? c.expected : 'spam',
      }));
      const passed = results.filter(r => r.ok).length;
      return { passed, total: results.length, passRate: results.length ? passed / results.length : 1, results };
    };

    // Dry run verifies without touching files.
    const dry = await improveDirectory({
      directory: dir,
      seed: 1,
      proposal: { files: { 'classify.nl': improvedClassify, 'guidance.md': guidanceText }, summary: 'Add triage guidance' },
      evaluateSplit,
      dryRun: true,
    });
    assert.equal(dry.status, 'checked');
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), originalClassify);
    assert.equal(existsSync(join(dir, 'guidance.md')), false);

    // Real adoption installs both edits and writes an adoption record.
    const adopted = await improveDirectory({
      directory: dir,
      seed: 1,
      propose: ({ support }) => {
        assert.ok(support.length >= 1, 'proposer receives support split');
        return { files: { 'classify.nl': improvedClassify, 'guidance.md': guidanceText }, summary: 'Add triage guidance' };
      },
      evaluateSplit,
    });
    assert.equal(adopted.status, 'accepted');
    assert.deepEqual(adopted.changed, ['classify.nl', 'guidance.md']);
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), improvedClassify);
    assert.equal(readFileSync(join(dir, 'guidance.md'), 'utf8'), guidanceText);
    assert.ok(adopted.record && adopted.recordPath && existsSync(adopted.recordPath));

    // Byte-identical rollback restores original classify.nl and removes added guidance.md.
    rollbackSource(adopted.record);
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), originalClassify);
    assert.equal(existsSync(join(dir, 'guidance.md')), false);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('improveDirectory rejects confirmation pass-rate drop, check failure, and protected types.ts edits', async () => {
  const dir = createFixtureProgram();
  const originalClassify = readFileSync(join(dir, 'classify.nl'), 'utf8');
  const originalTypes = readFileSync(join(dir, 'types.ts'), 'utf8');
  try {
    // 1. Confirmation pass rate drops -> rejected, files untouched.
    const regressed = await improveDirectory({
      directory: dir,
      seed: 0,
      proposal: { files: { 'classify.nl': originalClassify + 'Classify everything as spam.\n' } },
      evaluateSplit: async (targetDir, cases) => {
        const broken = readFileSync(join(targetDir, 'classify.nl'), 'utf8').includes('everything as spam');
        const results = cases.map(c => ({
          id: c.id,
          function: c.function,
          ok: !broken,
          expected: c.expected,
          received: broken ? 'spam' : c.expected,
        }));
        const passed = results.filter(r => r.ok).length;
        return { passed, total: results.length, passRate: results.length ? passed / results.length : 1, results };
      },
    });
    assert.equal(regressed.status, 'rejected');
    assert.ok(regressed.reasons.some(r => /confirmation pass rate dropped/i.test(r)), JSON.stringify(regressed.reasons));
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), originalClassify);

    // 2. Broken .nl syntax fails natlang check -> rejected, files untouched.
    const invalid = await improveDirectory({
      directory: dir,
      seed: 0,
      proposal: { files: { 'classify.nl': '---\nargs: { ticket: UnknownType }\nreturns: Label\n---\nBroken.\n' } },
      evaluateSplit: async (_dir, cases) => ({ passed: cases.length, total: cases.length, passRate: 1, results: [] }),
    });
    assert.equal(invalid.status, 'rejected');
    assert.ok(invalid.reasons.some(r => /check failed/i.test(r)), JSON.stringify(invalid.reasons));
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), originalClassify);

    // 3. Protected types.ts edit -> rejected before running checks.
    const protectedEdit = await improveDirectory({
      directory: dir,
      seed: 0,
      proposal: { files: { 'types.ts': "export type Label = 'spam';\n" } },
      evaluateSplit: async (_dir, cases) => ({ passed: cases.length, total: cases.length, passRate: 1, results: [] }),
    });
    assert.equal(protectedEdit.status, 'rejected');
    assert.ok(protectedEdit.reasons.some(r => /types\.ts/i.test(r)), JSON.stringify(protectedEdit.reasons));
    assert.equal(readFileSync(join(dir, 'types.ts'), 'utf8'), originalTypes);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('natlang improve DIR --dry-run CLI detects directory form and reports splits', async () => {
  const dir = createFixtureProgram();
  const { root: storeRoot, dispose } = freshStore();
  try {
    const code = await improvementCommand([dir, '--dry-run', '--store', storeRoot, '--seed', '7', '--json']);
    assert.equal(code, 0);
  } finally {
    dispose();
    rmSync(dir, { recursive: true, force: true });
  }
});

test('natlang improve DIR without a model or a change set says what to pass', async () => {
  const dir = createFixtureProgram();
  const { root: storeRoot, dispose } = freshStore();
  try {
    await assert.rejects(improvementCommand([dir, '--store', storeRoot]),
      /proposes a change from the program's recorded runs on a model: pass --model MODEL/);
  } finally {
    dispose();
    rmSync(dir, { recursive: true, force: true });
  }
});

/** An in-memory call store of the shape the evidence reader uses: rows, records, values, events, children, annotations. */
function memoryStore(dir) {
  const rows = [], records = new Map(), values = new Map(), events = new Map(), notes = new Map();
  let clock = 0;
  const ref = value => { const hash = hexDigest(JSON.stringify(value)); values.set(hash, value); return { complete: true, hash }; };
  const add = ({ id, parent = null, outcome = 'done', ticket, output, detail = '', evts, feedback, escalation, source = 'classify.nl' }) => {
    const record = { call_id: id, parent_call_id: parent, outcome, detail, definition: { id: 'classify', source },
      inputs: { ticket: ref(ticket) }, output: output === undefined ? null : ref(output), started_at: String(++clock).padStart(4, '0') };
    records.set(id, record);
    rows.push({ call_id: id, parent_call_id: parent, definition_id: 'classify', definition_source: source, outcome, started_at: record.started_at });
    if (evts) events.set(id, evts);
    if (feedback) notes.set(id, [{ kind: 'feedback', value: feedback }]);
    if (escalation) notes.set(id, [{ kind: 'escalation', value: { function: 'classify', local_ms: 5, ...escalation } }]);
  };
  return {
    add,
    calls: ({ outcome, limit = 1000 } = {}) => rows.filter(row => !outcome || row.outcome === outcome).slice().reverse().slice(0, limit),
    call: id => records.get(id),
    value: r => r?.complete ? values.get(r.hash) : undefined,
    events: id => events.get(id),
    children: id => rows.filter(row => row.parent_call_id === id),
    annotations: id => notes.get(id) ?? [],
    root: dir,
  };
}

test('run evidence: failures, escalations with the fallback answer, repairs and feedback; untrusted outputs leave the case pack', () => {
  const dir = createFixtureProgram();
  try {
    const store = memoryStore(dir);
    store.add({ id: 'f1', outcome: 'failed', ticket: 'refund for order 7?', detail: 'no label fits' });
    store.add({ id: 'b1', ticket: 'my card was declined at checkout', output: 'billing' });
    store.add({ id: 'l1', ticket: 'my card was declined at checkout', output: 'technical',
      escalation: { escalated: true, reason: 'confidence', agreed: false, fallback_call: 'b1' } });
    store.add({ id: 'r1', ticket: 'site down', output: 'technical', evts: [{ kind: 'ensures_check', check: 'result !== "spam"', outcome: 'fail' }] });
    store.add({ id: 'bad', ticket: 'invoice wrong', output: 'spam', feedback: { verdict: 'bad' } });
    store.add({ id: 'fix', ticket: 'pay later?', output: 'spam', feedback: { verdict: 'fix', expected: 'billing' } });
    store.add({ id: 'ok', ticket: 'login fails', output: 'technical' });

    const evidence = gatherRunEvidence(store, dir)['classify.nl'];
    assert.equal(evidence.calls, 7);
    assert.deepEqual(evidence.failures, [{ inputs: { ticket: 'refund for order 7?' }, outcome: 'failed', reason: 'no label fits' }]);
    assert.deepEqual(evidence.escalations, [{ inputs: { ticket: 'my card was declined at checkout' }, local: 'technical', fallback: 'billing',
      agreed: false, reason: 'confidence' }]);
    assert.deepEqual(evidence.repairs.map(item => item.problems), [['postcondition result !== "spam" failed']]);
    assert.deepEqual(evidence.feedback.map(item => [item.verdict, item.expected]).sort(), [['bad', undefined], ['fix', 'billing']]);

    assert.equal(trustedCallOutput(store, store.call('l1')).trusted, false, 'the local half of an escalation is not an expected result');
    assert.equal(trustedCallOutput(store, store.call('b1')).trusted, true);
    assert.equal(trustedCallOutput(store, store.call('bad')).trusted, false);
    assert.deepEqual(trustedCallOutput(store, store.call('fix')), { trusted: true, expected: 'billing' });

    const pack = buildDirectoryCasePack(dir, { store, seed: 0 });
    const fromStore = new Map(pack.cases.filter(item => item.source === 'call-store').map(item => [item.inputs.ticket, item.expected]));
    assert.equal(fromStore.get('my card was declined at checkout'), 'billing', 'the fallback answer is the expected result');
    assert.equal(fromStore.has('invoice wrong'), false);
    assert.equal(fromStore.get('pay later?'), 'billing', 'a fixed call carries the tester\'s expected result');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('proposeFromRuns: the improver reads the runs and the support misses, revises only the body, and the gate adopts it', async () => {
  const dir = createFixtureProgram();
  const original = readFileSync(join(dir, 'classify.nl'), 'utf8');
  try {
    const store = memoryStore(dir);
    store.add({ id: 'b1', ticket: 'card declined', output: 'billing' });
    store.add({ id: 'l1', ticket: 'card declined', output: 'technical', escalation: { escalated: true, reason: 'confidence', fallback_call: 'b1' } });
    const seen = {};
    const stages = {
      diagnose: async (target, evidence) => { seen.target = target; seen.evidence = evidence; return 'Payment failures are read as technical.'; },
      shouldRevise: async (_target, _evidence, diagnosis) => { seen.diagnosis = diagnosis; return true; },
      revise: async () => 'Pick the label for ticket. A failed or declined payment is billing.',
    };
    const evaluateSplit = async (targetDir, cases) => {
      const good = readFileSync(join(targetDir, 'classify.nl'), 'utf8').includes('declined payment is billing');
      const results = cases.map((c, index) => ({ id: c.id, function: c.function, ok: good || index % 2 === 1, expected: c.expected, received: 'spam' }));
      const passed = results.filter(r => r.ok).length;
      return { passed, total: results.length, passRate: results.length ? passed / results.length : 1, results };
    };
    const result = await improveDirectory({ directory: dir, seed: 3, store, evaluateSplit,
      propose: context => proposeFromRuns(context, { store, stages, evaluateSupport: cases => evaluateSplit(dir, cases) }) });
    assert.equal(result.status, 'accepted', JSON.stringify(result.reasons));
    assert.deepEqual(result.changed, ['classify.nl']);
    assert.equal(seen.target.path, 'classify.nl');
    assert.match(seen.target.declaration, /^description: Classify a support ticket\.\nargs: \{ ticket: string \}/);
    assert.equal(seen.target.instructions, 'Pick the label for ticket.');
    assert.equal(seen.evidence.escalations[0].fallback, 'billing');
    assert.ok(seen.evidence.wrong.length >= 1, 'support examples the program answers wrongly are evidence');
    const front = original.slice(0, original.indexOf('---', 3) + 4);
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), `${front}Pick the label for ticket. A failed or declined payment is billing.\n`,
      'front matter is byte-identical; only the instructions change');
    rollbackSource(result.record);
    assert.equal(readFileSync(join(dir, 'classify.nl'), 'utf8'), original);

    // A revision that declines, repeats the instructions or carries front matter changes nothing.
    for (const [shouldRevise, revised, why] of [[false, 'x', undefined], [true, 'Pick the label for ticket.', 'repeats'], [true, '---\nargs: {}\n---\nx', 'front matter']]) {
      const proposal = await proposeFromRuns({ directory: dir, files: { 'classify.nl': original }, support: [] },
        { store, stages: { ...stages, shouldRevise: async () => shouldRevise, revise: async () => revised } });
      assert.deepEqual(proposal.files, {});
      assert.equal(proposal.decisions[0].changed, false);
      if (why) assert.match(proposal.decisions[0].problem, new RegExp(why));
    }
    // Protected functions and a program whose runs went well get no proposal.
    assert.deepEqual((await proposeFromRuns({ directory: dir, files: { 'classify.nl': original }, support: [] },
      { store, stages, protectedPaths: new Set(['classify.nl']) })), null);
    assert.equal(await proposeFromRuns({ directory: dir, files: { 'classify.nl': original }, support: [] }, { stages }), null);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('the improver stages are built-in natlang programs that run on the model (diagnose, decide, revise)', async () => {
  for (const name of ['improveDiagnose', 'improveShouldRevise', 'improveRevise']) assert.equal(typeof builtin(name), 'function');
  const { driver, openings } = scriptedModel(opening => /Return target's new instructions/.test(opening) ?
    'return "Pick the label for ticket. A declined payment is billing.";' : 'return "Declined payments are labelled technical.";');
  const model = withGeneratedDecisions(driver, () => true);
  const runtime = createNatlangRuntime({ model, calls: false });
  const dir = createFixtureProgram();
  try {
    const store = memoryStore(dir);
    store.add({ id: 'x', ticket: 'card declined', output: 'technical', feedback: { verdict: 'fix', expected: 'billing' } });
    const original = readFileSync(join(dir, 'classify.nl'), 'utf8');
    const proposal = await runtime.run(() => proposeFromRuns({ directory: dir, files: { 'classify.nl': original }, support: [] }, { store }));
    assert.equal(proposal.decisions[0].diagnosis, 'Declined payments are labelled technical.');
    assert.match(proposal.files['classify.nl'], /A declined payment is billing\.\n$/);
    assert.ok(openings.some(opening => opening.includes('card declined')), 'the diagnosis call sees the run evidence');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
