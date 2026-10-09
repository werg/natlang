import assert from 'node:assert/strict';
import { test } from 'node:test';
import { join, resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';

const dist = process.env.NATLANG_TEST_DIST ? resolve(process.env.NATLANG_TEST_DIST) : resolve('dist');
const [collector, corrections, curriculum, materializer, prompt] = await Promise.all([
  import(pathToFileURL(join(dist, 'teacher/collector.js'))),
  import(pathToFileURL(join(dist, 'teacher/corrections.js'))),
  import(pathToFileURL(join(dist, 'teacher/curriculum.js'))),
  import(pathToFileURL(join(dist, 'teacher/native-materializer.js'))),
  import(pathToFileURL(join(dist, 'native/prompt.js'))),
]);
const { programRow } = collector;
const { correctedVariant, reasoningFitsFix } = corrections;
const { replayReference } = curriculum;
const { materializeNativeRows } = materializer;
const { TOOLS_PROMPT } = prompt;

const options = { systemPrompt: TOOLS_PROMPT, contextTokens: 16384, rootSeed: 909 };

/** A reference row whose first action is a refused `while` loop, fixed by the reference's own first action. */
async function recoveredRow(reasoning) {
  const [record] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  record.curriculum.reference.root.unshift(['eval', { code: 'let n = 0;\nwhile (n < 3) n++;\nn' }]);
  const { run, trajectory } = await replayReference(record, TOOLS_PROMPT);
  trajectory[0].assistant.reasoning = reasoning;
  trajectory[1].assistant.reasoning = 'Loops are refused, so I read the events directly.';
  return programRow(record, 'fixture-teacher', 'run-1', { model: 'fixture-teacher', context_tokens: 16384 }, run, trajectory);
}

test('a corrected variant makes the fix first, with the reasoning from before the failure, and trains only the fix', async () => {
  const row = await recoveredRow('I read the snapshot and the events for the team.');
  assert.match(JSON.stringify(row.trajectory[1].context.at(-1)), /while/);
  const result = await correctedVariant(row, { failed: [0], fixed: 1 }, options);
  assert.ok('row' in result, result.rejected);
  const { row: variant, decision } = result;
  assert.equal(decision, 0);
  assert.equal(variant.outcome.accepted, true);
  assert.equal(variant.trajectory.length, row.trajectory.length - 1);
  assert.equal(variant.trajectory[0].assistant.reasoning, 'I read the snapshot and the events for the team.');
  assert.deepEqual(variant.trajectory[0].assistant.calls.map(call => call.arguments), row.trajectory[1].assistant.calls.map(call => call.arguments));
  assert.doesNotMatch(JSON.stringify(variant.trajectory), /while \(/);
  const { run_id: runId, ...variantOf } = variant.provenance.variant;
  assert.deepEqual(variantOf, { version: 'corrected-first-attempt/1', parent: row.id, left_out: [0], fixed: 1, decision: 0 });
  assert.equal(typeof runId, 'string');
  const turns = materializeNativeRows([variant]).turns;
  assert.deepEqual(turns.filter(turn => turn.training_admission.approved).map(turn => turn.decision.index), [0]);
  assert.ok(turns.slice(1).every(turn => turn.training_admission.reason === 'context of a corrected variant'));
});

test('reasoning that planned what failed is not given to the fix', async () => {
  const row = await recoveredRow('I count the events with a `while` loop.');
  assert.deepEqual(await correctedVariant(row, { failed: [0], fixed: 1 }, options),
    { rejected: 'the reasoning before the failed attempt names what failed' });
  // Words the error only suggests are not what failed.
  const error = 'forbidden-loop: `while` loops are not allowed here. Use `for (const item of array)` or `step.iterateOn(initial).until(done)`.';
  assert.equal(reasoningFitsFix('Keep edges where until is null, for each item of the list.', error, 'while (x) {}', 'for (const e of edges) {}'), true);
  assert.equal(reasoningFitsFix('I page through with while.', error, 'while (x) {}', 'for (const e of edges) {}'), false);
});

test('programRow forwards explicit source collection guidance and preserves collector review precedence', () => {
  const [record] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  record.collection_guidance = { training_admission: false,
    review_scope: 'sampled-actions-require-independent-semantic-review' };
  const run = { outcome: { accepted: true, status: 'accepted', oracle: { accepted: true } }, trace: [] };
  const ordinary = programRow(record, 'fixture-teacher', 'run-guided', {}, run, []);
  assert.deepEqual(ordinary.collection_guidance, record.collection_guidance);

  const authoredReview = { training_admission: false, root_action: { sampled: false },
    child_actions: { source: 'provider', sampled: true } };
  const reviewed = programRow(record, 'fixture-teacher', 'run-guided-authored', {}, run, [],
    { collection_guidance: authoredReview });
  assert.deepEqual(reviewed.collection_guidance, authoredReview,
    'collector-supplied authored-root review details take precedence');
  assert.equal(reviewed.outcome.oracle.accepted, true, 'guidance must not rewrite oracle eligibility evidence');
});
