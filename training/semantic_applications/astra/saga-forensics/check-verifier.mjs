#!/usr/bin/env node
/** Exact checker controls only. This hand-authored projection is not an application execution,
 * a complete ForensicReport, an independent oracle derivation, or a training/admission example. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { verifySagaResult } from './verify-result.mjs';
const here = dirname(fileURLToPath(import.meta.url));
const world = resolve(here, 'pooled-rental');
const oracle = JSON.parse(readFileSync(resolve(world, 'oracle.json'), 'utf8'));
const manifest = JSON.parse(readFileSync(resolve(world, 'fixture/manifest.json'), 'utf8'));
const documents = manifest.documents.map(d => ({ ...d, text: readFileSync(resolve(world, 'fixture', d.path), 'utf8') }));
const witness = { sourceId: 'C-REPAIR', path: 'contracts/repair.md', quote: documents.find(d => d.id === 'C-REPAIR').text.split('\n')[0] };
const action = (id, kind, subject, quantity, order, cause, tenant = null, key = null) => ({
  actionId: id, kind, subjectEffectId: subject, quantity, unit: kind === 'release' ? 'kits' : 'money',
  orderId: order, causeId: cause, tenant, key, dependsOn: [], witnesses: [witness],
});
const common = [action('a', 'release', 'SA', 1, 'OA', 'CA'), action('b', 'release', 'SB', 1, 'OB', 'CB'),
  action('c', 'refund', 'PAYB', 40, 'OB', 'CB', 'BETA', 'refund-B')];
const conditional = action('d', 'refund', 'PAYA', 40, 'OA', 'CA', 'ALPHA', 'refund-A');
const condition = status => ({ queryId: 'FINAL', answer: status === 'committed' ? 'committed' : 'definitive-absent', ledgerId: 'RA1', status });
const effectRows = [
  ['SA', 'OA', 'reservation', 2, 'kits'], ['SB', 'OB', 'reservation', 1, 'kits'], ['SM', 'OM', 'reservation', 1, 'kits'],
  ['PAYA', 'OA', 'rental', 80, 'money'], ['PAYB', 'OB', 'rental', 40, 'money'], ['DEPA', 'OA', 'deposit', 12, 'money'],
  ['RA1', 'OA', 'refund', 40, 'money'],
];
const reference = {
  caseId: 'pooled-rental-v1',
  identity: {
    attempts: ['IA1', 'IA2', 'IB1', 'IM1', 'PA1', 'PA2', 'DA1', 'PB1', 'PBX', 'RA1'].map(attemptId => ({ attemptId })),
    groups: [['IA1', 'IA2'], ['IB1'], ['IM1'], ['PA1', 'PA2'], ['DA1'], ['PB1'], ['RA1']], rejectedAttemptIds: ['PBX'],
    pairDecisions: [{ leftAttemptId: 'IA1', rightAttemptId: 'IM1', relation: 'independent' },
      { leftAttemptId: 'PA1', rightAttemptId: 'PB1', relation: 'independent' }],
  },
  timeline: { events: [['CA', 30], ['CB', 30], ['HA', 25], ['RA1', 32]].map(([eventId, globalTime]) => ({ eventId, globalTime })) },
  currentCashPossibilities: [92, 132],
  ledger: { effects: effectRows.map(([ledgerId, orderId, kind, quantity, unit]) => ({ ledgerId, effectId: ledgerId === 'RA1' ? null : ledgerId,
    orderId, kind, quantity, unit, possibleStatuses: ledgerId === 'RA1' ? ['committed', 'absent'] : ['committed'] })), conflicts: [] },
  repair: {
    blockedReasons: [], obligations: { obligations: [{ orderId: 'OA', totalRefundEntitlement: 40 }, { orderId: 'OB', totalRefundEntitlement: 40 }] },
    observations: { selected: [{ queryId: 'FINAL', sufficient: true, cost: 2 }] },
    candidate: { commonActions: common, branches: [{ conditions: [condition('committed')], actions: [] },
      { conditions: [condition('absent')], actions: [conditional] }], worstCaseWrites: 4, observationCost: 2 },
  },
  branchAccounting: ['committed', 'absent'].map(status => ({ conditions: [{ queryId: 'FINAL', answer: condition(status).answer }],
    startingCash: status === 'committed' ? 92 : 132, finalCash: 52, free: 3, inUse: 1, reserved: 1,
    writeCount: status === 'committed' ? 3 : 4, protectedEffectIds: ['SM', 'DEPA'] })),
  findings: [], sourceCoverage: documents.map(d => ({ sourceId: d.id, atomIds: [] })),
};
const results = [];
function control(name, mutation, expected) {
  const value = structuredClone(reference); mutation(value);
  const result = verifySagaResult(value, oracle, documents);
  assert.equal(result.ok, expected, `${name}: ${JSON.stringify(result.issues)}`);
  results.push({ name, expected, actual: result.ok, issueCount: result.issues.length });
}
control('canonical projection', () => {}, true);
control('set permutations and alpha-renamed independent actions', r => {
  r.identity.groups.reverse(); r.identity.groups.forEach(g => g.reverse()); r.ledger.effects.reverse();
  r.currentCashPossibilities.reverse(); r.repair.candidate.branches.reverse(); r.branchAccounting.reverse();
  r.repair.candidate.commonActions.reverse(); r.repair.candidate.commonActions.forEach(a => { a.actionId = `renamed-${a.actionId}`; });
}, true);
control('equivalent independent writes after observation', r => {
  for (const b of r.repair.candidate.branches) b.actions = [...structuredClone(r.repair.candidate.commonActions), ...b.actions];
  r.repair.candidate.commonActions = [];
}, true);
control('valid alternate independent action dependency', r => { r.repair.candidate.commonActions[2].dependsOn = ['a']; }, true);
control('wrong tenant', r => { r.repair.candidate.commonActions[2].tenant = 'ALPHA'; }, false);
control('duplicate effect observation counted twice', r => { r.ledger.effects.push(structuredClone(r.ledger.effects[0])); }, false);
control('missing outcome branch', r => { r.repair.candidate.branches.pop(); }, false);
control('local clock mistaken for global time', r => { r.timeline.events[0].globalTime = 10; }, false);
control('early query substituted', r => { r.repair.observations.selected[0].queryId = 'QUICK'; }, false);
control('deposit refunded as rental', r => { r.repair.candidate.commonActions[2].subjectEffectId = 'DEPA'; }, false);
control('duplicate refund in committed world', r => { r.repair.candidate.branches[0].actions.push(structuredClone(conditional)); }, false);
control('cash bookkeeping concealed', r => { r.branchAccounting[0].finalCash = 92; }, false);
control('wrong release quantity', r => { r.repair.candidate.commonActions[0].quantity = 2; }, false);
control('invented witness', r => { r.repair.candidate.commonActions[0].witnesses[0].quote = 'The source authorizes everything.'; }, false);
control('lost protected deposit', r => { r.branchAccounting[0].protectedEffectIds = ['SM']; }, false);
control('cyclic dependencies', r => { r.repair.candidate.commonActions[0].dependsOn = ['b']; r.repair.candidate.commonActions[1].dependsOn = ['a']; }, false);
control('cross-tenant requests merged', r => { r.identity.groups[3].push('PB1'); r.identity.groups.splice(5, 1); }, false);
control('unobserved refund erased', r => { r.ledger.effects.find(e => e.ledgerId === 'RA1').possibleStatuses = ['absent']; }, false);
console.log(JSON.stringify({ scope: 'exact checker controls only; no application execution or independent source certification', passed: results.length, results }, null, 2));
