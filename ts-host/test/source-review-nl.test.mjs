import assert from 'node:assert/strict';
import { test } from 'node:test';
import { createHash } from 'node:crypto';
import { cpSync, mkdtempSync, readFileSync, writeFileSync, readdirSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { spawnSync } from 'node:child_process';
import { createNatlangRuntime } from '../dist/runtime/node.js';
import { builtinNames } from '../dist/builtin/index.js';
import { SOURCE_REVIEWS } from '../dist/teacher/source-review.js';
import { canonicalJson, checkItemRecommendation, checkRowVerdict, chooseSourcePrecedents, recommendationsAgree, reviewSourceItem,
  reviewSourceRow, reviewerHash, reviewerIdentity, sourceRowPrecheck, SourceReviewContractError } from '../dist/teacher/source-review-nl.js';

const repo = new URL('../../', import.meta.url).pathname;

/** A model whose tool loop answers `answers` in turn and whose judge accepts every refined value. */
function scriptedModel(answers) {
  const turns = [];
  let next = 0;
  const driver = Object.assign(async ({ messages }) => {
    turns.push(messages.at(-1));
    return { calls: [['return_result', { status: 'success', value: answers[Math.min(next++, answers.length - 1)] }]] };
  }, { decide: async () => ({ log_probs: [Math.log(0.97), Math.log(0.03)] }) });
  return { driver, turns };
}
const run = (model, work) => createNatlangRuntime({ model: { driver: model.driver }, seed: { mode: 'backend' }, calls: false }).run(work);

const item = { dataset: 'sst2', id: 'new-item-1', visible: 'The film is fine. Nothing else is said.', annotated_label: 'negative',
  contract: 'Label the sentiment of the sentence.', answer_format: null };
const hold = { recommendation: 'hold', concern: 'label-disagrees-with-source', reason: 'The text calls the film fine, which supports a positive label.',
  evidence: [{ quote: 'The film is fine.' }], proposed_entry: { reason: 'The text calls the film fine, which supports a positive label.' }, confidence: 'high' };
const admit = { recommendation: 'admit', concern: 'none', reason: 'The text supports the label.', evidence: [], proposed_entry: null, confidence: 'high' };
const row = { id: 'row-1', dataset: 'qa', split: 'test', source_groups: ['g1'], question: 'What is the mean?', answer_format: 'number',
  evidence: 'The mean is 3977.75 across four values.', gold: '3977.75', actual: '3977.7500000000005' };
const verdict = { status: 'normalization-candidate', rationale: 'The evidence states 3977.75; the produced value adds floating-point noise.',
  evidence: [{ quote: 'The mean is 3977.75' }], confidence: 'high' };

test('both reviewers ship as runtime-owned built-ins', () => {
  assert.ok(builtinNames().includes('reviewSourceItem') && builtinNames().includes('reviewSourceRow'));
  const identity = reviewerIdentity('reviewSourceItem', 'model-a');
  assert.equal(identity.kind, 'natlang');
  assert.match(identity.reviewer_hash, /^natlang@[0-9a-f]{16}$/);
  assert.notEqual(identity.reviewer_hash, reviewerIdentity('reviewSourceItem', 'model-b').reviewer_hash, 'the executor is part of the hash');
  assert.notEqual(identity.reviewer_hash, reviewerIdentity('reviewSourceRow', 'model-a').reviewer_hash, 'so is the definition');
  assert.equal(identity.hash_inputs.definition_source_sha256,
    createHash('sha256').update(readFileSync(new URL('../src/builtin/reviewSourceItem.nl', import.meta.url))).digest('hex'));
});

test('the hash and the canonical JSON agree with scripts/source_review.py', () => {
  assert.equal(reviewerHash('a'.repeat(64), 'natlang-ts-host/0.1.0', 'scripted-model'), 'natlang@3dbe1b19e71bf333');
  const value = { b: 'é', a: ['x', null, true], c: { z: '1', y: '2' } };
  assert.equal(createHash('sha256').update(canonicalJson(value)).digest('hex'), 'aea9636472c15a19eedd4c2d943dd40069d4b88452faaaac66d069a39e5b8b3a');
});

test('the default mode is crisp: no recommendation and no model call', async () => {
  const model = scriptedModel([hold]);
  assert.equal(await run(model, () => reviewSourceItem(item, { primary: { executor: 'scripted' } })), null);
  assert.equal(await run(model, () => reviewSourceRow(row, { primary: { executor: 'scripted' } })), null);
  assert.equal(model.turns.length, 0);
});

test('nl mode asks the reviewer and returns the recommendation with its reviewer, never a registry change', async () => {
  const before = readFileSync(join(repo, 'training/source-reviews/holds.jsonl'));
  const model = scriptedModel([hold]);
  const output = await run(model, () => reviewSourceItem(item, { mode: 'nl', primary: { executor: 'scripted' } }));
  assert.equal(output.kind, 'item');
  assert.deepEqual(output.recommendation, hold);
  assert.equal(output.reviewer.reviewer_hash, reviewerIdentity('reviewSourceItem', 'scripted').reviewer_hash);
  assert.deepEqual(output.input.item, item);
  assert.ok(model.turns.length >= 1);
  assert.ok(readFileSync(join(repo, 'training/source-reviews/holds.jsonl')).equals(before));
  assert.equal(SOURCE_REVIEWS.some(review => review.id === item.id), false);
});

test('an admit recommendation carries concern none and no proposed entry', async () => {
  const output = await run(scriptedModel([admit]), () => reviewSourceItem(item, { mode: 'nl', primary: { executor: 'scripted' } }));
  assert.equal(output.recommendation.recommendation, 'admit');
  assert.equal(output.recommendation.proposed_entry, null);
});

test('a recommendation that breaks the typed contract is refused with a message that says what to return', () => {
  assert.throws(() => checkItemRecommendation(item, { ...hold, proposed_entry: null }), /proposed_entry \{ reason \} is present exactly when/);
  assert.throws(() => checkItemRecommendation(item, { ...admit, proposed_entry: { reason: 'x' } }), /exactly when recommendation is "hold"/);
  assert.throws(() => checkItemRecommendation(item, { ...hold, concern: 'none' }), /concern is "none" exactly when/);
  assert.throws(() => checkItemRecommendation(item, { ...hold, evidence: [{ quote: 'The film is wonderful.' }] }), error =>
    error instanceof SourceReviewContractError && /copied exactly from item\.visible/.test(error.message));
  assert.equal(checkItemRecommendation(item, hold), hold);
  const precheck = sourceRowPrecheck(row);
  assert.throws(() => checkRowVerdict(row, { ...precheck, exact_match: true }, verdict), /equivalent" whenever precheck\.exact_match/);
  assert.throws(() => checkRowVerdict(row, precheck, { ...verdict, evidence: [{ quote: 'The mean is 12' }] }), /row\.evidence/);
  assert.throws(() => checkRowVerdict(row, precheck, { ...verdict, status: 'maybe' }), /status is/);
});

test('row prechecks are exact: equality, source span and numeric representation', () => {
  assert.deepEqual(sourceRowPrecheck(row), { exact_match: false, actual_is_exact_source_span: false, numerically_equal: true });
  assert.deepEqual(sourceRowPrecheck({ ...row, gold: 'a copper statue', actual: 'a copper statue', evidence: 'It is a copper statue here.' }),
    { exact_match: true, actual_is_exact_source_span: true, numerically_equal: null });
  assert.equal(sourceRowPrecheck({ ...row, gold: '1,400', actual: '1400' }).numerically_equal, true);
  assert.equal(sourceRowPrecheck({ ...row, gold: '1400', actual: '1401' }).numerically_equal, false);
  assert.equal(sourceRowPrecheck({ ...row, gold: 'about 3', actual: '3' }).numerically_equal, null);
  assert.equal(sourceRowPrecheck({ ...row, actual: '' }).actual_is_exact_source_span, false);
});

test('an exact match is equivalent without a call; other rows go to the reviewer', async () => {
  const model = scriptedModel([verdict]);
  const exact = await run(model, () => reviewSourceRow({ ...row, actual: row.gold }, { mode: 'nl', primary: { executor: 'scripted' } }));
  assert.equal(exact.reviewer.kind, 'crisp');
  assert.equal(exact.recommendation.status, 'equivalent');
  assert.equal(model.turns.length, 0);
  const reviewed = await run(model, () => reviewSourceRow(row, { mode: 'nl', primary: { executor: 'scripted' } }));
  assert.deepEqual(reviewed.recommendation, verdict);
  assert.equal(reviewed.input.precheck.numerically_equal, true);
  assert.equal(reviewed.reviewer.function, 'reviewSourceRow');
});

test('a second executor gives an independent recommendation and agreement is exact equality of the enums', async () => {
  const calls = [];
  const second = { executor: 'other-model', call: async (name, args) => { calls.push([name, args.length]); return { ...admit }; } };
  const output = await run(scriptedModel([hold]), () => reviewSourceItem(item, { mode: 'nl', primary: { executor: 'scripted' }, second }));
  assert.deepEqual(calls, [['reviewSourceItem', 2]]);
  assert.equal(output.second_reviewer.hash_inputs.executor, 'other-model');
  assert.equal(output.second_recommendation.recommendation, 'admit');
  assert.equal(recommendationsAgree('item', output.recommendation, output.second_recommendation), false);
  assert.equal(recommendationsAgree('row', verdict, { ...verdict, rationale: 'other words' }), true);
});

test('shadow mode serves the reviewer and records that the crisp side recommended nothing', async () => {
  const traces = [];
  const runtime = createNatlangRuntime({ model: { driver: scriptedModel([hold]).driver }, seed: { mode: 'backend' }, calls: false, trace: trace => traces.push(trace) });
  const output = await runtime.run(() => reviewSourceItem(item, { mode: 'shadow', primary: { executor: 'scripted' } }));
  assert.deepEqual(output.recommendation, hold);
  const [event] = traces.flatMap(trace => trace.events).filter(entry => entry.kind === 'pluggable_shadow');
  assert.equal(event.name, 'reviewSourceItem');
  assert.equal(event.agree, false);
});

test('precedents are the most similar held items of the same dataset', () => {
  const held = SOURCE_REVIEWS.find(review => review.dataset === 'banking77');
  const [first, ...rest] = chooseSourcePrecedents('banking77', held.text, 3);
  assert.equal(first.id, held.id);
  assert.ok(rest.length <= 2);
  assert.deepEqual(chooseSourcePrecedents('no-such-dataset', 'text'), []);
});

test('a reviewer output becomes a receipt through scripts/source_review.py, with no admission and an open decision', async () => {
  const work = mkdtempSync(join(tmpdir(), 'source-review-'));
  const registry = join(work, 'registry');
  cpSync(join(repo, 'training/source-reviews'), registry, { recursive: true });
  const outputs = [
    await run(scriptedModel([hold]), () => reviewSourceItem(item, { mode: 'nl', primary: { executor: 'scripted' } })),
    await run(scriptedModel([verdict]), () => reviewSourceRow(row, { mode: 'nl', primary: { executor: 'scripted' } })),
    await run(scriptedModel([]), () => reviewSourceRow({ ...row, id: 'row-2', actual: row.gold }, { mode: 'nl', primary: { executor: 'scripted' } })),
  ];
  const file = join(work, 'outputs.jsonl');
  writeFileSync(file, outputs.map(output => JSON.stringify(output)).join('\n') + '\n');
  const result = spawnSync('python3', [join(repo, 'scripts/source_review.py'), '--registry', registry, '--manifests', join(work, 'm'), 'receipt', file],
    { encoding: 'utf8' });
  assert.equal(result.status, 0, result.stderr);
  const printed = result.stdout.trim().split('\n').map(line => JSON.parse(line));
  assert.equal(printed.length, 3);
  assert.ok(printed.every(entry => entry.decision === null && entry.training_admission === false));
  const receipts = readdirSync(join(registry, 'receipts')).map(name => JSON.parse(readFileSync(join(registry, 'receipts', name), 'utf8')));
  const itemReceipt = receipts.find(receipt => receipt.kind === 'item');
  assert.equal(itemReceipt.training_admission, false);
  assert.equal(itemReceipt.reviewer.reviewer_hash, reviewerIdentity('reviewSourceItem', 'scripted').reviewer_hash);
  assert.equal(itemReceipt.subject.subject_sha256, createHash('sha256').update(canonicalJson(outputs[0].input)).digest('hex'));
  assert.ok(readFileSync(join(registry, 'holds.jsonl')).equals(readFileSync(join(repo, 'training/source-reviews/holds.jsonl'))), 'a receipt changes no registry');
  const decide = spawnSync('python3', [join(repo, 'scripts/source_review.py'), '--registry', registry, '--manifests', join(work, 'm'), 'decide',
    itemReceipt.id, '--decision', 'hold', '--by', 'agent:test'], { encoding: 'utf8' });
  assert.equal(decide.status, 0, decide.stderr);
  assert.equal(readFileSync(join(registry, 'holds.jsonl'), 'utf8').trim().split('\n').length, SOURCE_REVIEWS.length + 1);
});
