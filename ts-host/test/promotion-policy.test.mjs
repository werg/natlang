import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime, loadVirtualNatlang, CallStore, caseHashes, crispPolicy, emptyEvidence, ruleOf, DEFAULT_SETTINGS,
  promotionPolicy, reviewPromotions, sanitizeDecision, TierLedger, TierEngine } from '../dist/index.js';

const freshStore = () => CallStore.open(mkdtempSync(join(tmpdir(), 'natlang-promotion-')));
const done = store => { const root = store.root; store.close(); rmSync(root, { recursive: true, force: true }); };
const SHOUT = '---\nargs: { text: string }\nreturns: string\n---\nReturn the text in upper case.\n';
const CASES = `export const cases = [
  { when: (args: { text: string }) => args.text.length < 10, run: (args: { text: string }) => args.text.toUpperCase() },
];
`;
const rule = ruleOf(DEFAULT_SETTINGS);
const summary = (state, evidence) => ({ subject: 'case', id: 'x', state, rule, evidence: { ...emptyEvidence(), ...evidence } });

async function withCase(store, tier) {
  const model = Object.assign(async () => ({ prompt_tokens: 1, calls: [['return_result', { status: 'success', value: 'SEED' }]] }), { model: 'seed' });
  await createNatlangRuntime({ model, calls: store, specialization: 'off' }).run(() => loadVirtualNatlang({ 'shout.nl': SHOUT }, 'shout.nl')('seed'));
  const seed = store.call(store.calls()[0].call_id);
  const [hash] = caseHashes(CASES);
  store.saveCompilation({ definitionKey: seed.definition.key, definitionId: seed.definition.id, definitionName: 'shout', definitionSource: seed.definition.source,
    interfaceHash: seed.definition.interface, programRoot: null, files: { 'cases.ts': CASES }, caseHashes: [hash],
    links: [{ caseHash: hash, callId: seed.call_id, role: 'training' }] });
  store.setTier(hash, tier);
  return { hash, callId: seed.call_id };
}

test('the crisp policy is the case rule: promote on clean evidence, demote past the bound, keep otherwise', () => {
  assert.equal(crispPolicy(summary('shadow', { compared: 10, live_compared: 3 })).decision, 'promote');
  assert.equal(crispPolicy(summary('shadow', { compared: 10, worse: 1, live_compared: 3 })).decision, 'keep');
  assert.equal(crispPolicy(summary('shadow', { compared: 9, live_compared: 9 })).decision, 'keep');
  assert.equal(crispPolicy(summary('active', { audited: 5, audit_worse: 1 })).decision, 'demote');
  assert.equal(crispPolicy(summary('active', { served: 8, handed_off: 2 })).decision, 'demote');
  assert.equal(crispPolicy(summary('active', { served: 40, handed_off: 1 })).decision, 'keep');
  assert.equal(crispPolicy(summary('active', { served: 40, guard_misses: 100, infrastructure: 100 })).decision, 'keep');
});

test('a case and a tier with the same evidence get the same decision', () => {
  const ledger = new TierLedger();
  for (let index = 0; index < 10; index++) ledger.apply({ fn: 'f', tier: 'tier1', event: 'shadow_equal' }, rule, 'shadow', false);
  const row = ledger.row('f', 'tier1');
  const tier = { subject: 'tier', id: 'f/tier1', state: 'shadow', evidence: row.evidence, rule };
  assert.equal(crispPolicy(tier).decision, crispPolicy({ ...tier, subject: 'case' }).decision);
  assert.equal(crispPolicy(tier).decision, 'promote');
});

test('the store applies the crisp rule inline under the default policy, and leaves cases alone under nl', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionComparisons: 1, promotionLiveComparisons: 0 });
    const { hash, callId } = await withCase(store, 'shadow');
    store.caseVerdict(hash, callId, 'shadow', 'equal');
    assert.equal(store.caseStats(hash).tier, 'active');
    assert.match(store.caseStats(hash).note, /promoted by evidence/);
    store.setTier(hash, 'shadow');
    store.writeSettings({ promotionPolicy: 'nl' });
    store.caseVerdict(hash, callId, 'shadow', 'equal');
    assert.equal(store.caseStats(hash).tier, 'shadow', 'nothing is decided per call under nl');
  } finally { done(store); }
});

test('nl policy: a scripted model decides in the review, off the hot path', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionPolicy: 'nl' });
    const { hash, callId } = await withCase(store, 'shadow');
    store.caseVerdict(hash, callId, 'shadow', 'equal');
    const asked = [];
    const nl = s => { asked.push(s); return { decision: 'promote', reason: 'looks good' }; };
    const reviewed = await reviewPromotions(store, nl);
    assert.equal(asked.length, 1);
    assert.equal(asked[0].subject, 'case');
    assert.equal(asked[0].evidence.compared, 1);
    assert.equal(store.caseStats(hash).tier, 'active');
    assert.match(store.caseStats(hash).note, /promoted by nl policy: looks good/);
    assert.deepEqual(reviewed.map(item => [item.subject, item.decision.decision, item.applied]), [['case', 'promote', true]]);
    const again = await reviewPromotions(store, () => ({ decision: 'demote', reason: 'recent trend' }));
    assert.equal(store.caseStats(hash).tier, 'demoted');
    assert.equal(again[0].applied, true);
    await reviewPromotions(store, () => 'garbage');
    assert.equal(store.caseStats(hash).tier, 'demoted', 'an invalid answer is a keep');
  } finally { done(store); }
});

test('shadow policy: the crisp decision serves and the nl one is only compared', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionPolicy: 'shadow', promotionComparisons: 1, promotionLiveComparisons: 0 });
    const { hash, callId } = await withCase(store, 'shadow');
    store.caseVerdict(hash, callId, 'shadow', 'equal');
    assert.equal(store.caseStats(hash).tier, 'active', 'crisp applies inline in shadow mode');
    const reviewed = await reviewPromotions(store, () => ({ decision: 'demote', reason: 'nl disagrees' }));
    assert.equal(store.caseStats(hash).tier, 'active');
    assert.equal(reviewed[0].decision.decision, 'keep', 'the crisp answer is served');
    assert.equal(reviewed[0].applied, false);
    const policy = promotionPolicy('shadow', () => ({ decision: 'promote', reason: 'x' }));
    assert.equal((await policy(summary('active', {}))).decision, 'keep');
  } finally { done(store); }
});

test('nl policy decides tiers too: a promoted event is kept with the last call and read back by an engine', async () => {
  const store = freshStore();
  try {
    store.writeSettings({ promotionPolicy: 'nl', promotionComparisons: 2, promotionLiveComparisons: 0 });
    const { callId } = await withCase(store, 'active');
    for (let index = 0; index < 2; index++)
      store.annotate(callId, 'tier', { fn: 'shout', tier: 'tier1', event: 'shadow_equal', state: 'shadow', call_id: callId }, 'tiers', false);
    await reviewPromotions(store, () => ({ decision: 'promote', reason: 'agrees' }));
    const events = store.tierRows().map(row => row.value);
    assert.equal(events.at(-1).event, 'promoted');
    assert.equal(events.at(-1).evidence.by, 'nl');
    const ledger = new TierLedger();
    ledger.hydrate(store, 'shout', () => 'shadow', ruleOf(store.settings()));
    assert.equal(ledger.state('shout', 'tier1', 'shadow'), 'active');
    void TierEngine;
  } finally { done(store); }
});

test('sanitizeDecision never moves anything on a malformed answer', () => {
  assert.equal(sanitizeDecision(null).decision, 'keep');
  assert.equal(sanitizeDecision({ decision: 'maybe' }).decision, 'keep');
  assert.deepEqual(sanitizeDecision({ decision: 'demote', reason: 'r' }), { decision: 'demote', reason: 'r' });
});
