// A hinted twin's request states its technique as required (H3, 2026-11-03): iterateOn, and inline natural-language calls.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { checkHintedTechnique, hintedTechnique, techniqueAccepted } from '../dist/teacher/required-technique.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';
import { hintFor, hinted } from '../scripts/inline-curriculum/lib.mjs';
import { missingDataError } from './support/datasets.mjs';

const SEEDS = [[7, 0], [7, 1], [11, 0], [29, 2]];

test('hintedTechnique derives the requirement from the twin, never from an unhinted case', () => {
  const hint = 'Required: solve this with iterateOn, not with evals. Required: make each judgment about an item with a natural-language function called on that item.';
  assert.deepEqual(hintedTechnique({ hinted_of: 'a', hint, iterate: 'required', inline: 'required' }), { iterate: true, inline: true });
  assert.deepEqual(hintedTechnique({ hinted_of: 'a', hint: 'Required: solve this with iterateOn.', iterate: 'required', inline: 'required' }), { iterate: true, inline: false });
  assert.equal(hintedTechnique({ iterate: 'required', inline: 'required' }), undefined);
  assert.equal(hintedTechnique({ hinted_of: 'a', hint: 'Sketch only', iterate: 'required' }), undefined);
  const iteration = [{ kind: 'iteration_step', phase: 'step' }, { kind: 'iteration_step', phase: 'done' }];
  const inline = [[{ kind: 'manifest', inline_instruction_site: { returns: {} } }]];
  assert.equal(techniqueAccepted(checkHintedTechnique({ iterate: true, inline: true }, iteration, inline)), true);
  assert.equal(techniqueAccepted(checkHintedTechnique({ iterate: true, inline: true }, [], inline)), false);
  assert.equal(techniqueAccepted(checkHintedTechnique({ iterate: true, inline: true }, iteration, [[{ kind: 'manifest' }]])), false);
  assert.equal(techniqueAccepted(checkHintedTechnique({ iterate: true, inline: true }, [iteration[0]], inline)), false, 'an unfinished iteration is not enough');
});

test('edit_stream :hinted twins enforce iterateOn with inline calls; the unhinted twin does not', async t => {
  let enforced = 0, accepted = 0;
  for (const [seed, index] of SEEDS) {
    let record;
    try { [record] = FAMILIES.edit_stream.build(seed, index, 'train'); }
    catch (error) { if (missingDataError(error)) { t.skip(error.message); return; } throw error; }
    const twin = hinted(record, hintFor(record.curriculum));
    assert.deepEqual(hintedTechnique(twin.curriculum), { iterate: true, inline: true });
    // The reference follows the hint: accepted.
    const reference = (await replayReference(twin, TOOLS_PROMPT)).run.outcome;
    assert.equal(reference.accepted, true, `${twin.id}: reference rejected: ${reference.rejection_reasons}`);
    assert.equal(reference.checks.hinted_technique, true);
    accepted++;
    // The exploit the review recorded: the final text typed by hand into return_result, no iterateOn.
    const final = record.curriculum.reference.root.at(-1)[1].value;
    const typed = structuredClone(twin);
    typed.curriculum.reference.root = [['return_result', { status: 'success', value: final }]];
    const handTyped = (await replayReference(typed, TOOLS_PROMPT)).run.outcome;
    assert.equal(handTyped.accepted, false, `${twin.id}: hand-typed answer accepted`);
    assert.deepEqual(handTyped.rejection_reasons, ['hinted_technique']);
    enforced++;
    // The same answer to the unhinted twin (never told the technique) is still graded on the result.
    const plain = structuredClone(record);
    plain.curriculum.reference.root = [['return_result', { status: 'success', value: final }]];
    assert.equal((await replayReference(plain, TOOLS_PROMPT)).run.outcome.accepted, true);
  }
  t.diagnostic(`hinted twins: ${accepted} references accepted, ${enforced} hand-typed answers rejected`);
});
