import assert from 'node:assert/strict';
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import * as explainer from '../../applications/dist/failure-explainer/index.js';
import { refinements } from '../../applications/dist/failure-explainer/refinements.js';
import { classifyAdmissionReason, classifyAdmissionReasons, dpoHoldReasons, RULES } from '../scripts/admission-dispositions.mjs';
import { advisoryFile, explainerIdentity, writeAdvisory } from '../scripts/advisory-file.mjs';
import { main as explainMain, ledgerReasons } from '../scripts/explain-advisory.mjs';
import { scriptedModel } from './support/natlang.mjs';

const card = { family: 'folder_find', program_id: 'p1', error: '', checks: { answer: true, files: null }, outcome_detail: 'done',
  actions: ['["eval",{"code":"return read(\\"a.txt\\")"}]'], feedback: ['file not found: a.txt'], turn_count: 2, delegation_count: 0 };
const TAGS = ['api_or_type_contract', 'state_recovery'];

/** A model that answers each of the three functions with the code the test gives it. */
function runtimeFor(answers) {
  const { driver, openings } = scriptedModel(opening => {
    if (opening.includes('card is one failed attempt')) return `return ${JSON.stringify(answers.failure)};`;
    if (opening.includes('bucket lists admission reasons')) return `return ${JSON.stringify(answers.triage)};`;
    if (opening.includes('facts are computed from a gate')) return `return ${JSON.stringify(answers.gate)};`;
    return null;
  });
  return { runtime: createNatlangRuntime({ model: driver }), openings };
}

test('explainFailure: a checked explanation names a tag and quotes the card; bad quotes and tags are reported', async () => {
  const good = { tag: 'state_recovery', proposed_tag: null, why: 'The program read a file that is not there.', confidence: 'high',
    evidence: [{ quote: 'file not found: a.txt' }] };
  const { runtime, openings } = runtimeFor({ failure: good });
  const ok = await explainer.explainFailures(runtime, [{ id: 'c1', card }], TAGS);
  assert.deepEqual(ok.c1.problems, []);
  assert.equal(ok.c1.value.tag, 'state_recovery');
  assert.match(openings[0], /file not found: a\.txt/, 'the card reaches the model');
  const bad = runtimeFor({ failure: { ...good, tag: 'made_up', evidence: [{ quote: 'invented text' }] } });
  const flagged = await explainer.explainFailures(bad.runtime, [{ id: 'c1', card }], TAGS);
  assert.equal(flagged.c1.problems.length, 2);
  // A new tag must be a name that is not already a tag.
  assert.deepEqual(explainer.checkFailureExplanation(card, TAGS, { ...good, tag: 'new', proposed_tag: 'state_recovery' }),
    ['proposed_tag state_recovery is already a tag; use it as the tag']);
});

test('refinement checkers: proposed_tag exactly when new, sentence counts, prefix, gate_unchanged', () => {
  const [explanation, proposals, gate] = Object.keys(refinements).map(key => refinements[key]);
  const base = { tag: 'x', proposed_tag: null, why: 'One. Two.', evidence: [{ quote: 'q' }], confidence: 'low' };
  assert.equal(explanation({ ...base, tag: 'new', proposed_tag: 'empty_listing' }), true);
  assert.equal(explanation({ ...base, tag: 'new', proposed_tag: null }), false);
  assert.equal(explanation({ ...base, proposed_tag: 'empty_listing' }), false);
  assert.equal(explanation({ ...base, tag: 'new', proposed_tag: 'Not Snake' }), false);
  assert.equal(explanation({ ...base, why: 'A. B. C. D.' }), false);
  assert.equal(explanation({ ...base, evidence: [] }), false);
  assert.equal(proposals({ proposals: [{ reason: 'defect: x', match_prefix: 'defect' }] }), true);
  assert.equal(proposals({ proposals: [{ reason: 'defect: x', match_prefix: 'other' }] }), false);
  assert.equal(proposals({ proposals: [{ reason: 'defect: x', match_prefix: '' }] }), false);
  const explained = { pattern: 'It fails in the long band.', candidate_causes: [{ cause: 'c', support: 's', confidence: 'low' }], next_checks: ['n'], gate_unchanged: true };
  assert.equal(gate(explained), true);
  assert.equal(gate({ ...explained, gate_unchanged: false }), false);
});

test('rejection triage: bucket, category examples, and the crisp verifier of proposed rules', async () => {
  const rows = [{ reason: 'wrong_return', id: 'r1', family: 'f' }, { reason: 'defect_unrepaired: a', id: 'r2', family: 'f' },
    { reason: 'defect_unrepaired: b', id: 'r3', family: 'g' }, { reason: 'defect_unrepaired: a', id: 'r4', family: 'f' }, { reason: 'replay failed', id: 'r5', family: 'f' }];
  const category = reason => classifyAdmissionReason(reason).category;
  const bucket = explainer.unclassifiedBucket(rows, category);
  assert.deepEqual(bucket.map(entry => [entry.reason, entry.count]), [['defect_unrepaired: a', 2], ['defect_unrepaired: b', 1]]);
  const classified = [...new Set(rows.map(row => row.reason))].map(reason => ({ reason, category: category(reason) }));
  const categories = explainer.ruleCategories(RULES, classified);
  assert.ok(categories.find(item => item.category === 'candidate_failure').example_reasons.includes('wrong_return'));
  const proposal = (reason, prefix, category = 'candidate_failure') => ({ reason, category, match_prefix: prefix, rationale: 'r',
    next_action: categories.find(item => item.category === category).action });
  const good = [proposal('defect_unrepaired: a', 'defect_unrepaired'), proposal('defect_unrepaired: b', 'defect_unrepaired')];
  const { runtime } = runtimeFor({ triage: { proposals: good } });
  const result = await explainer.triage(runtime, bucket, categories, classified);
  assert.deepEqual(result.proposals.map(item => item.problems), [[], []]);
  assert.deepEqual(result.missing, []);
  const context = { bucket, classified, categories, proposals: good };
  // A prefix that also covers an already classified reason is refused.
  const wrong = { reason: 'wrong_thing: a', count: 1, examples: [] };
  assert.match(explainer.checkProposedRule(proposal('wrong_thing: a', 'wrong'), { ...context, bucket: [wrong], proposals: [proposal('wrong_thing: a', 'wrong')] }).join(),
    /would also cover "wrong_return", already classified as candidate_failure/);
  assert.match(explainer.checkProposedRule(proposal('defect_unrepaired: a', 'wrong'), context).join(), /does not match its reason/);
  // A prefix that covers a bucket reason proposed for another category is refused.
  const split = [proposal('defect_unrepaired: a', 'defect_unrepaired'), proposal('defect_unrepaired: b', 'defect_unrepaired', 'oracle_or_source_review')];
  assert.match(explainer.checkProposedRule(split[0], { ...context, proposals: split }).join(), /not proposed for candidate_failure/);
  assert.match(explainer.checkProposedRule({ ...good[0], category: 'nope' }, context).join(), /not one of the categories/);
  assert.match(explainer.checkProposedRule({ ...good[0], next_action: 'something_else' }, context).join(), /not the action of candidate_failure/);
});

const facts = { schema: 'natlang.self-feedback-evaluation', passed: false,
  failed: [{ stratum: 'length_band:long', metric: 'kl_plain_to_projected_nats', observed: 0.05, limit: 0.02, direction: 'max', margin: 0.03 }],
  passed_strata: ['first', 'last'], worst_windows: [{ id: 'abc@0', metric: 'kl_plain_to_projected_nats', observed: 0.09 }],
  context: { run: 'r', checkpoint: null, step: 10, earlier_reports: [] } };

test('explainGateFailure: causes must cite the facts; inconsistent facts are refused before any model call', async () => {
  const good = { pattern: 'Only the long band exceeds the KL limit.', gate_unchanged: true, next_checks: ['KL by position band on the long windows'],
    candidate_causes: [{ cause: 'drift late in long spans', support: 'length_band:long kl_plain_to_projected_nats margin 0.03', confidence: 'medium' }] };
  const { runtime, openings } = runtimeFor({ gate: good });
  const result = await explainer.explainGate(runtime, facts, '{"report": true}');
  assert.deepEqual(result.problems, []);
  assert.match(openings[0], /length_band:long/);
  assert.deepEqual(explainer.checkGateExplanation(facts, { ...good, candidate_causes: [{ cause: 'x', support: 'vibes', confidence: 'low' }] }).length, 1);
  assert.deepEqual(explainer.checkGateFacts({ ...facts, failed: [{ ...facts.failed[0], observed: 0.01 }] }).length, 1);
  await assert.rejects(explainer.explainGate(runtime, { ...facts, failed: [] }, '{}'), /gate facts are inconsistent/);
});

test('advisory files are labelled with the explainer hash, immutable, and the classifiers are untouched', () => {
  const sources = { a: 'text a', b: 'text b' };
  const identity = explainerIdentity(sources, 'model-1');
  assert.match(identity.explainer, /^natlang@[0-9a-f]{16}$/);
  assert.notEqual(identity.explainer, explainerIdentity(sources, 'model-2').explainer);
  assert.notEqual(identity.explainer, explainerIdentity({ ...sources, a: 'changed' }, 'model-1').explainer);
  const file = advisoryFile('explainFailure', identity, { x: 1 }, { c1: 1 });
  assert.equal(file.advisory, true);
  assert.equal(file.acted_on_by_pipeline, false);
  assert.equal(file.explainer, identity.explainer);
  assert.match(file.inputs_sha256, /^[0-9a-f]{64}$/);
  const path = join(mkdtempSync(join(tmpdir(), 'advisory-')), 'out.advisory.json');
  writeAdvisory(path, file);
  assert.throws(() => writeAdvisory(path, file), /exists/);
  assert.deepEqual(JSON.parse(readFileSync(path, 'utf8')).items, { c1: 1 });
  // The crisp classification of reasons is exactly what it was.
  assert.equal(classifyAdmissionReason('defect_unrepaired: a').category, 'unclassified_review_pending');
  assert.equal(classifyAdmissionReason('wrong_return').category, 'candidate_failure');
  assert.deepEqual(dpoHoldReasons(['defect_unrepaired: a', 'wrong_return']).map(item => item.category), ['unclassified_review_pending']);
  assert.equal(classifyAdmissionReasons(['replay failed']).length, 1);
});

test('the rejections command writes a labelled proposal file beside the ledger and changes no classification', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'rejections-'));
  const ledger = join(dir, 'ledger.jsonl');
  writeFileSync(ledger, [{ id: 'a', family: 'f', reasons: ['wrong_return', 'defect_unrepaired: x'] }, { id: 'b', reasons: ['defect_unrepaired: y'] }].map(item => JSON.stringify(item)).join('\n') + '\n');
  assert.deepEqual(ledgerReasons(readFileSync(ledger, 'utf8')).map(row => row.reason), ['wrong_return', 'defect_unrepaired: x', 'defect_unrepaired: y']);
  const before = ['wrong_return', 'defect_unrepaired: x'].map(reason => classifyAdmissionReason(reason));
  const action = RULES.find(rule => rule.category === 'candidate_failure').action;
  const proposals = ['defect_unrepaired: x', 'defect_unrepaired: y'].map(reason => ({ reason, category: 'candidate_failure', next_action: action,
    match_prefix: 'defect_unrepaired', rationale: 'Like the defect failures.' }));
  const { runtime } = runtimeFor({ triage: { proposals } });
  assert.equal(await explainMain(['rejections', '--ledger', ledger, '--model', 'scripted'], { createRuntime: () => runtime }), 0);
  const out = JSON.parse(readFileSync(`${ledger}.rejection-proposals.json`, 'utf8'));
  assert.equal(out.advisory, true);
  assert.equal(out.function, 'triageRejections');
  assert.match(out.explainer, /^natlang@/);
  assert.deepEqual(out.items.proposals.map(item => item.problems), [[], []]);
  assert.deepEqual(['wrong_return', 'defect_unrepaired: x'].map(reason => classifyAdmissionReason(reason)), before);
  // Written once, and refused before any model call.
  await assert.rejects(explainMain(['rejections', '--ledger', ledger, '--model', 'scripted'], { createRuntime: () => assert.fail('no model call') }), /exists/);
  assert.ok(existsSync(ledger));
});
