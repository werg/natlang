import assert from 'node:assert/strict';
import { test } from 'node:test';
import { caseFrom } from '../scripts/inline-curriculum/semantic-iterate-reducers-v18-novel-data.mjs';

const makeCase = (authorizationRule, requestId = 'FISH-730', authorized = true) => caseFrom({
  slug: 'authority-scope-test', sourceGroup: 'authority-scope-test:world', domain: 'fixture review',
  owner: 'Review desk', requestId,
  requestText: 'Select one eligible item.',
  ruleText: `An item qualifies when the review is complete; ${authorizationRule.requirement} is required.`,
  selectionInstruction: 'Select exactly one eligible item when any qualifies; otherwise return none.',
  authorizationRule,
  candidates: [{ id: 'ITEM-A', metric: 10 }, { id: 'ITEM-B', metric: 20 }],
  compare: (a, b) => b.metric - a.metric,
  provisional: rows => rows.slice(0, 1),
  eligible: item => item.id === 'ITEM-A',
  select: rows => rows.slice(0, 1),
  format: rows => rows.length ? rows.map(row => row.id).join('; ') : 'none',
  measure: rows => rows.length ? rows.map(row => String(row.metric)).join('; ') : 'none',
  registerLine: row => `${row.id}: ${row.metric}.`,
  auditLine: row => `${row.id}: review complete=${row.id === 'ITEM-A'}.`,
  exceptionText: 'No exception.',
  authorityText: 'Authority status is intentionally not part of this wording test.',
  authorized, selectionFormat: 'one item', measureFormat: 'digits',
  finalDecisions: ['assign', 'hold', 'no_action'], approvedAction: 'assign', noAction: 'no_action',
});

test('request-scoped current permit and release rules are stated without inventing per-item grants', () => {
  const permit = makeCase({ requirement: 'a current fisheries permit', scope: 'request' });
  const permitGate = 'a current fisheries permit for request FISH-730';
  assert.ok(permit.decision_rule.includes(permitGate));
  assert.ok(permit.fields.decision.includes(permitGate));
  assert.ok(permit.passes[3].constraint.includes(permitGate));
  assert.ok(permit.decision_rule.includes('do not add a separate item-specific authorization requirement'));
  assert.equal(permit.source_summary.selectedItems, 'ITEM-A');
  assert.equal(permit.source_summary.decision, 'assign');
  const pendingPermit = makeCase({ requirement: 'a current fisheries permit', scope: 'request' }, 'FISH-731', false);
  assert.equal(pendingPermit.source_summary.decision, 'hold', 'authorization status remains source-authored gold input');
  assert.equal(pendingPermit.fields.decision.replace('FISH-731', 'FISH-730'), permit.fields.decision,
    'the template does not reveal authority status');

  const release = makeCase({ requirement: 'the parks duty officer release for dispatch', scope: 'request' }, 'TREE-710');
  const releaseGate = 'the parks duty officer release for dispatch for request TREE-710';
  assert.ok(release.decision_rule.includes(releaseGate));
  assert.ok(release.fields.decision.includes(releaseGate));
  assert.ok(release.passes[3].constraint.includes('do not add a separate item-specific authorization requirement'));
  assert.equal(release.source_summary.decision, 'assign');
});

test('item-scoped authority metadata requires coverage for each selected item', () => {
  const row = makeCase({ requirement: 'a signed review approval', scope: 'each_selected_item' }, 'CASE-41');
  assert.ok(row.decision_rule.includes('a signed review approval for each selected item in request CASE-41'));
  assert.ok(row.fields.decision.includes('a signed review approval for each selected item in request CASE-41'));
  assert.ok(row.passes[3].constraint.includes('a request-level record alone does not establish item-specific authorization'));
  assert.ok(!row.decision_rule.includes('do not add a separate item-specific authorization requirement'));
});
