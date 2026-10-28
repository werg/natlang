// Adversarial qualification of the edit_stream edit-fidelity oracle and the contract_diagnosis function-effect oracle
// (2026-10-28). AGENTS.md "Task contracts and reward validity": reference replay alone does not qualify an oracle, so
// each oracle must reject meaningful wrong answers (the two exploits the 2026-10-26 audit recorded, keyword stuffing,
// partial edits, rename misses, hard-coded answers, repaired-but-wrong functions) and accept the reference and
// legitimate variants. Cases come from training seeds only; no held or development packet content is used here.
import assert from 'node:assert/strict';
import { test } from 'node:test';
import { checkConstraints, editFidelityProblems } from '../dist/evaluation/constraints.js';
import { replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';
import { editFidelityFromRecord } from '../scripts/inline-curriculum/edit-stream.mjs';
import { contractDiagnosisNegatives, contractDiagnosisPositives, editStreamNegatives, editStreamPositives,
  editStreamStructure } from '../scripts/inline-curriculum/oracle-adversaries.mjs';
import { missingDataError } from './support/datasets.mjs';

const SEEDS = [[7, 0], [7, 1], [7, 2], [11, 0], [11, 1], [11, 3], [29, 0], [29, 2], [2027, 1], [2027, 4]];

test('edit_stream edit-fidelity oracle rejects wrong edits and accepts the edited document', async t => {
  let rejected = 0, accepted = 0;
  for (const [seed, index] of SEEDS) {
    let record;
    try { [record] = FAMILIES.edit_stream.build(seed, index, 'train'); }
    catch (error) { if (missingDataError(error)) { t.skip(error.message); return; } throw error; }
    const id = `${seed}:${index}`, { fidelity } = editStreamStructure(record);
    assert.ok(fidelity, `${id} carries an edit_fidelity constraint`);
    assert.ok(!record.semantics.expected.some(c => ['include_words', 'word_count'].includes(c.kind)), `${id}: superseded constraints are gone`);
    assert.deepEqual(editFidelityFromRecord(record), fidelity, `${id}: the stored-case derivation equals the generated constraint`);
    for (const [name, text] of editStreamNegatives(record)) {
      assert.equal(checkConstraints(text, record.semantics.expected).passed, false, `${id}: accepted negative "${name}"`);
      assert.ok(editFidelityProblems(text, fidelity).length > 0, `${id}: edit_fidelity alone must reject "${name}"`);
      rejected++;
    }
    for (const [name, text] of editStreamPositives(record)) {
      const verdict = checkConstraints(text, record.semantics.expected);
      assert.equal(verdict.passed, true, `${id}: rejected positive "${name}": ${verdict.failed.join('; ')} ${editFidelityProblems(text, fidelity).join('; ')}`);
      accepted++;
    }
  }
  t.diagnostic(`edit_stream: ${rejected} negatives rejected, ${accepted} positives accepted over ${SEEDS.length} cases`);
  assert.ok(rejected >= 20 * SEEDS.length && accepted >= 5 * SEEDS.length);
});

test('edit_fidelity: renames are counted everywhere and nowhere else', () => {
  const c = { kind: 'edit_fidelity', version: 'edit-fidelity/1', tolerance: 0.025, deleted: ['Old text about Zeta corp results and more.'],
    paragraphs: ['Alder met Brennan in Paris on Monday and Alder later spoke to reporters about the long talks.', 'The end.'],
    renames: [{ from: 'Smith', to: 'Alder' }] };
  const reference = 'Alder met Brennan in Paris on Monday and Alder later spoke to reporters about the long talks.\n\nThe end.';
  assert.deepEqual(editFidelityProblems(reference, c), []);
  assert.ok(editFidelityProblems(reference.replace('and Alder', 'and Smith'), c).length);
  assert.ok(editFidelityProblems(reference.replace('Brennan', 'Alder'), c).length);
  assert.ok(editFidelityProblems(`${reference}\n\nOld text about Zeta corp results.`, c).length);
});

test('contract_diagnosis function-effect oracle rejects unrepaired, hard-coded and wrongly repaired runs', async t => {
  let rejected = 0, accepted = 0;
  const run = async (record, root) => {
    const probe = structuredClone(record);
    probe.curriculum.reference.root = root;
    return (await replayReference(probe, TOOLS_PROMPT)).run.outcome;
  };
  for (const [seed, index] of [[7, 0], [11, 2], [29, 5]]) for (const record of FAMILIES.contract_diagnosis.build(seed, index)) {
    const id = `${seed}:${index}:${record.curriculum.variant}`;
    for (const [name, root] of contractDiagnosisNegatives(record)) {
      const outcome = await run(record, root);
      assert.equal(outcome.accepted, false, `${id} accepted negative "${name}"`);
      assert.equal(outcome.checks.function_effects, false, `${id}: the function check rejects "${name}"`);
      rejected++;
    }
    for (const [name, root] of contractDiagnosisPositives(record)) {
      const outcome = await run(record, root);
      assert.equal(outcome.accepted, true, `${id} rejected positive "${name}": ${JSON.stringify(outcome.function_effects ?? outcome.checks)}`);
      accepted++;
    }
    if (record.curriculum.variant === 'defect') {
      // The recorded exploit is a correct answer: only the function check rejects it.
      const exploit = await run(record, contractDiagnosisNegatives(record)[0][1]);
      assert.equal(exploit.checks.answer, true);
      assert.match(exploit.function_effects.problems.join('\n'), /line_total was not repaired/);
    }
  }
  t.diagnostic(`contract_diagnosis: ${rejected} negatives rejected, ${accepted} positives accepted`);
});
