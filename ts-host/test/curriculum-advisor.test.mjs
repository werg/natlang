import assert from 'node:assert/strict';
import { mkdtempSync, readFileSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { test } from 'node:test';
import { createNatlangRuntime } from '../dist/index.js';
import * as advisor from '../../applications/dist/curriculum-advisor/index.js';
import { refinements } from '../../applications/dist/curriculum-advisor/refinements.js';
import { runtimeFailureReason } from '../dist/teacher/curriculum-policy.js';
import { SLICE_TARGETS, DOMAIN_TARGETS } from '../dist/teacher/curriculum.js';
import { main } from '../scripts/curriculum-advisory.mjs';
import { scriptedModel } from './support/natlang.mjs';

function runtimeFor(answers) {
  const { driver } = scriptedModel(opening => {
    if (opening.includes('card holds a question')) return `return ${JSON.stringify(answers.equivalence)};`;
    if (opening.includes('coverage lists, for the admitted cases')) return `return ${JSON.stringify(answers.batch)};`;
    return null;
  });
  return createNatlangRuntime({ model: driver });
}

const heldRow = (id, source = 'qasper') => ({ task: { program_ir: { version: 'natlang.program/2', id, kind: 'program', source, family: 'qa',
    semantics: { root: 'main.nl', files: {}, inputs: { question: 'Who wrote it?' }, expected: ['Ada Lovelace'] } } },
  provenance: { runtime_contract_version: 20 },
  outcome: { accepted: false, status: 'done', rejection_reasons: ['answer'], value: 'Lovelace', checks: {} }, trajectory: [] });

test('judgeAnswerEquivalence releases a row to human review only; the hold and the admission are unchanged', async () => {
  const row = heldRow('q1');
  assert.equal(runtimeFailureReason(row), 'unreviewed_extractive_answer_equivalence');
  const card = advisor.equivalenceCard(row);
  assert.deepEqual([card.id, card.gold, card.answer], ['q1', ['Ada Lovelace'], 'Lovelace']);
  const equivalent = { verdict: 'equivalent', reason: 'The answer is the surname of the annotated answer.', evidence: [{ quote: 'Lovelace' }] };
  const runtime = runtimeFor({ equivalence: equivalent });
  const [release] = await advisor.judgeEquivalences(runtime, [card]);
  assert.equal(release.release, 'human_review');
  assert.equal(release.admit, false);
  assert.equal(release.training_admission, 'unchanged');
  // The policy still holds the row whatever the judgment says.
  assert.equal(runtimeFailureReason(row), 'unreviewed_extractive_answer_equivalence');
  // A different, unsure or unquoted judgment keeps the row held.
  for (const judgment of [{ ...equivalent, verdict: 'different' }, { verdict: 'unsure', reason: 'Not clear.', evidence: [] },
    { ...equivalent, evidence: [{ quote: 'not in the card' }] }]) {
    const kept = advisor.releaseToReview(card, judgment);
    assert.equal(kept.release, 'keep_held');
    assert.equal(kept.admit, false);
  }
  assert.equal(advisor.releaseToReview(card, undefined).release, 'keep_held');
  assert.equal(advisor.equivalenceCard(heldRow('x', 'commitpack')), undefined, 'only extractive sources');
  assert.equal(advisor.equivalenceCard({ ...heldRow('y'), outcome: { accepted: true } }), undefined, 'only rejected rows');
});

test('curriculum advisor refinement checkers', () => {
  const [judgment, proposal] = Object.keys(refinements).map(key => refinements[key]);
  assert.equal(judgment({ verdict: 'equivalent', reason: 'One sentence.', evidence: [{ quote: 'q' }] }), true);
  assert.equal(judgment({ verdict: 'equivalent', reason: 'One sentence.', evidence: [] }), false);
  assert.equal(judgment({ verdict: 'unsure', reason: 'One sentence.', evidence: [] }), true);
  assert.equal(judgment({ verdict: 'equivalent', reason: 'Two. Sentences.', evidence: [{ quote: 'q' }] }), false);
  assert.equal(judgment({ verdict: 'maybe', reason: 'One.', evidence: [] }), false);
  assert.equal(proposal({ note: 'n', batch: [{ slice: 's', domain: 'd', count: 3, rationale: 'r' }] }), true);
  assert.equal(proposal({ note: 'n', batch: [{ slice: 's', domain: 'd', count: 0, rationale: 'r' }] }), false);
  assert.equal(proposal({ note: 'n', batch: [{ slice: 's', domain: 'd', count: 1.5, rationale: 'r' }] }), false);
});

test('coverage facts and the batch proposal check', async () => {
  const cases = [...Array(6).fill({ slice: 'inline_placement', domain: 'logic' }), { slice: 'iterate', domain: 'actor' }];
  const coverage = advisor.coverageFacts(cases, { slices: SLICE_TARGETS, domains: DOMAIN_TARGETS }, 10);
  const slice = name => coverage.slices.find(item => item.name === name);
  assert.equal(coverage.total, 7);
  assert.equal(slice('inline_placement').count, 6);
  assert.ok(Math.abs(slice('inline_placement').share - 6 / 7) < 1e-12);
  assert.equal(slice('writing').target, 0);
  assert.equal(coverage.cells.find(item => item.slice === 'iterate' && item.domain === 'actor').count, 1);
  assert.equal(coverage.cells.length, coverage.slices.length * coverage.domains.length);
  const good = { note: 'Folder failures stay uncovered.', batch: [{ slice: 'observation_followup', domain: 'relational', count: 6, rationale: 'It has a target share of 0.30 and no cases.' }] };
  const runtime = runtimeFor({ batch: good });
  const result = await advisor.proposeBatch(runtime, coverage);
  assert.deepEqual(result.problems, []);
  const bad = { note: 'n', batch: [{ slice: 'writing', domain: 'logic', count: 8, rationale: 'r' }, { slice: 'nope', domain: 'logic', count: 5, rationale: 'r' }] };
  assert.equal(advisor.checkBatchProposal(coverage, bad).length, 3, 'zero-target slice, unknown slice, total above the batch size');
});

test('the commands write labelled advisory files and read-only inputs', async () => {
  const dir = mkdtempSync(join(tmpdir(), 'curriculum-advisory-'));
  const rows = join(dir, 'rows.jsonl');
  writeFileSync(rows, [heldRow('q1'), heldRow('q2', 'commitpack')].map(row => JSON.stringify(row)).join('\n') + '\n');
  const runtime = runtimeFor({ equivalence: { verdict: 'equivalent', reason: 'Same person.', evidence: [{ quote: 'Lovelace' }] } });
  assert.equal(await main(['equivalence', '--rows', rows, '--model', 'scripted'], { createRuntime: () => runtime }), 0);
  const out = JSON.parse(readFileSync(`${rows}.equivalence-review.json`, 'utf8'));
  assert.equal(out.advisory, true);
  assert.equal(out.acted_on_by_pipeline, false);
  assert.deepEqual(out.items.map(item => [item.id, item.release, item.admit]), [['q1', 'human_review', false]]);
  const cases = join(dir, 'cases.jsonl');
  writeFileSync(cases, JSON.stringify({ curriculum: { slice: 'iterate', domain: 'actor' } }) + '\n');
  const batch = runtimeFor({ batch: { note: 'n', batch: [{ slice: 'iterate', domain: 'logic', count: 5, rationale: 'The slice is short of its share.' }] } });
  assert.equal(await main(['next-batch', '--cases', cases, '--batch-size', '10', '--model', 'scripted'], { createRuntime: () => batch }), 0);
  const proposed = JSON.parse(readFileSync(`${cases}.next-batch.json`, 'utf8'));
  assert.equal(proposed.function, 'nextCollectionBatch');
  assert.deepEqual(proposed.items.problems, []);
});
