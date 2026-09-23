import assert from 'node:assert/strict';
import { test } from 'node:test';
import { admitRow, coverage, renderOpening, replayReference, verifyCases } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';

const synthetic = Object.entries(FAMILIES).filter(([, family]) => !family.source);

test('every synthetic curriculum family builds cases that verify', async () => {
  const records = synthetic.flatMap(([, family]) => family.build(7, 0));
  const results = await verifyCases(records, TOOLS_PROMPT);
  assert.deepEqual(results.filter(item => !item.ok).map(item => `${item.id}: ${item.problems.join('; ')}`), []);
  assert.equal(new Set(records.map(record => record.id)).size, records.length);
});

test('a decisive observation in the opening and a pair group with one answer are rejected', async () => {
  const [first, second] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  const leaked = structuredClone(first);
  leaked.curriculum.decisive = [{ marker: leaked.semantics.inputs.team, source: 'eval', note: 'visible in the opening' }];
  const same = structuredClone(second);
  same.semantics.expected = first.semantics.expected;
  same.curriculum.reference.root.at(-1)[1].value = first.semantics.expected;
  const results = await verifyCases([leaked, same], TOOLS_PROMPT);
  assert.match(results[0].problems.join(' '), /visible in the opening/);
  assert.match(results[1].problems.join(' '), /one expected result for every variant/);
});

test('admission requires the decisive observation before the first result decision', async () => {
  const [record] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  const { run, trajectory } = await replayReference(record, TOOLS_PROMPT);
  const row = { task: { program_ir: record }, outcome: run.outcome, trajectory };
  assert.equal(admitRow(row).admitted, true);
  // The same answer returned before reading the events: correct value, premature choice.
  const hasty = structuredClone(record);
  hasty.curriculum.reference.root = [hasty.curriculum.reference.root.at(-1)];
  const quick = await replayReference(hasty, TOOLS_PROMPT);
  const verdict = admitRow({ task: { program_ir: hasty }, outcome: quick.run.outcome, trajectory: quick.trajectory });
  assert.equal(quick.run.outcome.accepted, true);
  assert.deepEqual(verdict.reasons.map(reason => reason.split(':')[0]), ['missing_observation']);
  // Observing only after a staged return is premature.
  const late = structuredClone(record);
  late.curriculum.reference.root = [['eval', { code: `return ${JSON.stringify(record.semantics.expected)};` }],
    ...record.curriculum.reference.root];
  const staged = await replayReference(late, TOOLS_PROMPT);
  const early = admitRow({ task: { program_ir: late }, outcome: staged.run.outcome, trajectory: staged.trajectory });
  assert.deepEqual(early.reasons.map(reason => reason.split(':')[0]), ['premature_choice']);
});

test('inline and edit expectations are enforced by admission', async () => {
  const [semantic, crisp] = FAMILIES.inline_review_each.build(7, 0);
  // Answering a per-item semantic filter directly, without inline children.
  const direct = structuredClone(semantic);
  direct.curriculum.reference.root = [direct.curriculum.reference.root.at(-1)];
  const plain = await replayReference(direct, TOOLS_PROMPT);
  assert.deepEqual(admitRow({ task: { program_ir: direct }, outcome: plain.run.outcome, trajectory: plain.trajectory }).reasons, ['inline_missing']);
  // A gratuitous inline child for a field test.
  const eager = structuredClone(crisp);
  eager.curriculum.reference.root = [['eval', { code: 'const kept = await review_each(inbox(), nl`Is priority of ticket at least 3?`);\nkept' }],
    eager.curriculum.reference.root.at(-1)];
  eager.curriculum.reference.children = inboxAnswers(eager);
  const extra = await replayReference(eager, TOOLS_PROMPT);
  assert.deepEqual(admitRow({ task: { program_ir: eager }, outcome: extra.run.outcome, trajectory: extra.trajectory }).reasons, ['gratuitous_inline']);
  // Editing a helper that already meets its contract.
  const [, sound] = FAMILIES.contract_diagnosis.build(7, 0);
  const meddling = structuredClone(sound);
  meddling.curriculum.reference.root.splice(1, 0, ['edit_function', { name: 'line_total', find: 'Math.max(0,', replace_with: 'Math.max(0, 0 +' }]);
  const edited = await replayReference(meddling, TOOLS_PROMPT);
  assert.deepEqual(admitRow({ task: { program_ir: meddling }, outcome: edited.run.outcome, trajectory: edited.trajectory }).reasons, ['unwarranted_edit']);
  const summary = coverage([admitRow({ task: { program_ir: direct }, outcome: plain.run.outcome, trajectory: plain.trajectory })]);
  assert.equal(summary.rejections.inline_missing, 1);
});

function inboxAnswers(record) {
  const source = record.semantics.files['urgent_queue/inbox.ts'];
  const tickets = JSON.parse(source.slice(source.indexOf('= ') + 2, source.indexOf(';\n')));
  return tickets.map(ticket => ({ match: JSON.stringify(ticket.id), value: ticket.priority >= 3 }));
}

test('the function listing shows TypeScript and natlang doc comments', async () => {
  const [record] = FAMILIES.child_sufficiency.build(7, 0);
  const opening = await renderOpening(record, TOOLS_PROMPT);
  assert.match(opening, /\/\*\* The first-line records for a refund claim\. \*\/\\n\s*function basic\(/);
  assert.match(opening, /\/\*\* Judge whether evidence settles a refund claim, and which way\. \*\/\\ndeclare function assess\(/);
});
