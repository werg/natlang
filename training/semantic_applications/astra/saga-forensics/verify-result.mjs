#!/usr/bin/env node
/** External exact checker. Never included in model-visible project or fixture.
 * This checks a report against independently reviewed fixture invariants. It does not interpret policy text,
 * construct effect identities, choose a repair or execute the application. Semantic witness support remains
 * an independent review obligation; exact quotation alone cannot establish entailment.
 */
import { readFileSync } from 'node:fs';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
const here = dirname(fileURLToPath(import.meta.url));
const canonical = value => JSON.stringify(value);
const sorted = xs => xs.map(canonical).sort();
const sameSet = (a, b) => canonical(sorted(a)) === canonical(sorted(b));

export function verifySagaResult(report, oracle, documents) {
  const issues = [];
  const demand = (condition, message) => { if (!condition) issues.push(message); };
  const visit = value => {
    if (!value || typeof value !== 'object') return;
    if ('sourceId' in value && 'path' in value && 'quote' in value) {
      const document = documents.find(d => d.id === value.sourceId && d.path === value.path);
      demand(Boolean(document), `Unknown witness identity ${value.sourceId}/${value.path}`);
      demand(typeof value.quote === 'string' && value.quote.length > 0 && Boolean(document?.text.includes(value.quote)),
        `Witness quote does not occur in ${value.sourceId}/${value.path}`);
    }
    for (const child of Object.values(value)) if (typeof child === 'object') {
      if (Array.isArray(child)) child.forEach(visit); else visit(child);
    }
  };
  visit(report);
  demand(report.caseId === 'pooled-rental-v1', 'Wrong caseId');
  const groups = report.identity?.groups ?? [];
  demand(sameSet((report.identity?.attempts ?? []).map(a => a.attemptId), oracle.attempt_ids), 'Missing or extra source attempts');
  demand(sameSet(groups.map(g => [...g].sort()), oracle.identity_groups.map(g => [...g].sort())), 'Wrong identity partition');
  for (const pair of oracle.duplicate_attempt_groups) {
    demand(groups.some(group => pair.every(id => group.includes(id))), `Missing duplicate join ${pair.join('/')}`);
  }
  for (const pair of oracle.independent_same_key_groups) {
    demand(!groups.some(group => pair.every(id => group.includes(id))), `Merged independent identities ${pair.join('/')}`);
    demand(report.identity?.pairDecisions?.some(d => d.relation === 'independent' &&
      sameSet([d.leftAttemptId, d.rightAttemptId], pair)), `Missing witnessed independent decision ${pair.join('/')}`);
  }
  demand(sameSet(report.identity?.rejectedAttemptIds ?? [], oracle.rejected_attempts), 'Wrong rejected attempt set');
  const times = report.timeline?.events ?? [];
  for (const [id, expected] of Object.entries(oracle.global_event_times)) {
    const matches = times.filter(t => t.eventId === id);
    demand(matches.length === 1 && matches[0].globalTime === expected, `Wrong or ambiguous normalized event time ${id}`);
  }
  demand(sameSet(report.currentCashPossibilities ?? [], oracle.current_cash_possibilities), 'Wrong current cash possibilities');
  const effects = report.ledger?.effects ?? [];
  const expectedEffects = oracle.effect_invariants;
  demand(effects.length === expectedEffects.length, 'Missing or extra ledger effects');
  for (const expected of expectedEffects) {
    const matches = effects.filter(e => e.ledgerId === expected.ledgerId);
    demand(matches.length === 1, `Ledger identity multiplicity ${expected.ledgerId}`);
    const actual = matches[0];
    if (!actual) continue;
    for (const [field, value] of Object.entries(expected)) {
      demand(Array.isArray(value) ? sameSet(actual[field] ?? [], value) : actual[field] === value,
        `Wrong ledger ${expected.ledgerId}.${field}`);
    }
  }
  demand((report.ledger?.conflicts ?? []).length === 0, 'Unresolved ledger conflict in a source-resolved fixture');
  const repair = report.repair;
  demand((repair?.blockedReasons ?? []).length === 0, 'Repair remains blocked');
  const obligations = repair?.obligations?.obligations ?? [];
  for (const [order, amount] of Object.entries(oracle.refund_entitlements)) {
    const match = obligations.filter(o => o.orderId === order);
    demand(match.length === 1 && match[0].totalRefundEntitlement === amount, `Wrong entitlement ${order}`);
  }
  const selected = repair?.observations?.selected ?? [];
  demand(selected.length === 1 && selected[0]?.queryId === oracle.observation.queryId && selected[0]?.sufficient === true,
    'Selected observation does not provide the required minimal sufficient partition');
  demand(selected[0]?.cost === oracle.observation.cost, 'Wrong observation cost');
  const candidate = repair?.candidate;
  demand(Boolean(candidate), 'No repair candidate');
  if (!candidate) return { ok: false, issues, semantic_review_required: true };
  demand(candidate.worstCaseWrites === oracle.minimum_worst_case_writes, 'Wrong minimum worst-case write count');
  demand(candidate.observationCost === oracle.observation.cost, 'Wrong total observation cost');
  const branches = candidate.branches ?? [];
  demand(branches.length === 2, 'Each supported refund world must have one outcome branch');
  const common = candidate.commonActions ?? [];
  const counts = [];
  demand((report.branchAccounting ?? []).length === 2, 'Missing or extra branch accounting rows');
  for (const status of ['committed', 'absent']) {
    const matching = branches.filter(b => (b.conditions ?? []).some(c =>
      c.queryId === oracle.observation.queryId && c.ledgerId === oracle.uncertain_refund_ledger_id && c.status === status));
    demand(matching.length === 1, `Missing or duplicate ${status} branch`);
    if (matching.length !== 1) continue;
    const branch = matching[0];
    demand(branch.conditions.length === 1, `Unsupported extra condition in ${status} branch`);
    const condition = branch.conditions[0];
    demand(condition.answer === (status === 'committed' ? 'committed' : 'definitive-absent'), `Wrong source answer label in ${status} branch`);
    const actions = [...common, ...(branch.actions ?? [])];
    const ids = actions.map(a => a.actionId);
    demand(new Set(ids).size === ids.length, `Duplicate action identifiers in ${status} branch`);
    const expected = [...oracle.branch_independent_action_requirements];
    if (status === 'absent') expected.push(oracle.absent_branch_action_requirement);
    demand(actions.length === expected.length, `Wrong write multiplicity in ${status} branch`);
    for (const action of expected) {
      const matches = actions.filter(a => Object.entries(action).every(([key, value]) => a[key] === value));
      demand(matches.length === 1, `Missing, duplicate or wrong business action ${canonical(action)} in ${status}`);
    }
    for (const action of actions) {
      demand(action.quantity > 0 && Number.isInteger(action.quantity), 'Action quantity must be a positive integer');
      demand((action.dependsOn ?? []).every(id => ids.includes(id) && id !== action.actionId), `Invalid prerequisite for ${action.actionId}`);
      demand((action.witnesses ?? []).length > 0, `Unwitnessed action ${action.actionId}`);
    }
    // Topological replay checks declared dependencies without imposing an arbitrary independent action order.
    const pending = [...actions], done = new Set();
    for (let n = 0; n < actions.length; n++) {
      const i = pending.findIndex(a => (a.dependsOn ?? []).every(id => done.has(id)));
      if (i < 0) break;
      const [action] = pending.splice(i, 1); done.add(action.actionId);
    }
    demand(pending.length === 0, `Cyclic action prerequisites in ${status} branch`);
    const startCash = oracle.confirmed_cash - (status === 'committed' ? oracle.uncertain_refund_amount : 0);
    const finalCash = startCash - actions.filter(a => a.kind === 'refund').reduce((sum, a) => sum + a.quantity, 0);
    const finalFree = oracle.current_stock.free + actions.filter(a => a.kind === 'release').reduce((sum, a) => sum + a.quantity, 0);
    demand(finalCash === oracle.common_final_cash, `Cash does not conserve in ${status} branch`);
    demand(finalFree === oracle.final_stock.free, `Stock does not conserve in ${status} branch`);
    const accounting = (report.branchAccounting ?? []).filter(a => a.conditions?.some(c => c.queryId === condition.queryId && c.answer === condition.answer));
    demand(accounting.length === 1, `Missing or duplicate accounting for ${status}`);
    if (accounting.length === 1) {
      const a = accounting[0];
      demand(a.startingCash === startCash && a.finalCash === finalCash && a.free === finalFree &&
        a.inUse === oracle.final_stock.inUse && a.reserved === oracle.final_stock.reserved && a.writeCount === actions.length,
      `False branch accounting in ${status}`);
      demand(sameSet(a.protectedEffectIds ?? [], oracle.protected_effect_ids), `Wrong protected effects in ${status}`);
    }
    counts.push(actions.length);
  }
  demand(counts.length === 2 && Math.max(...counts) === candidate.worstCaseWrites, 'Declared write bound differs from branch replay');
  demand((report.findings ?? []).every(f => f.severity !== 'defect' && f.severity !== 'unresolved'), 'Audit still reports a defect or unresolved premise');
  demand(sameSet((report.sourceCoverage ?? []).map(c => c.sourceId), documents.map(d => d.id)), 'Incomplete source document coverage');
  return { ok: issues.length === 0, issues, semantic_review_required: true };
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  if (!process.argv[2]) throw new Error('Usage: node verify-result.mjs REPORT.json');
  const world = resolve(here, 'pooled-rental');
  const manifest = JSON.parse(readFileSync(resolve(world, 'fixture/manifest.json'), 'utf8'));
  const documents = manifest.documents.map(d => ({ ...d, text: readFileSync(resolve(world, 'fixture', d.path), 'utf8') }));
  const oracle = JSON.parse(readFileSync(resolve(world, 'oracle.json'), 'utf8'));
  const result = verifySagaResult(JSON.parse(readFileSync(resolve(process.argv[2]), 'utf8')), oracle, documents);
  console.log(JSON.stringify(result, null, 2));
  if (!result.ok) process.exitCode = 1;
}
