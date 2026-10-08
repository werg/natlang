import assert from 'node:assert/strict';
import { test } from 'node:test';
import { admitRow, coverage, renderOpening, replayReference, verifyCases } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { FAMILIES } from '../scripts/inline-curriculum/families.mjs';
import { missingDataError } from './support/datasets.mjs';

const synthetic = Object.entries(FAMILIES).filter(([, family]) => !family.source && !family.externalData);

test('every synthetic curriculum family builds cases that verify', async t => {
  // A family drawn from data this machine lacks is named, not verified; every other family still is.
  const records = [], missing = [];
  for (const [name, family] of synthetic) {
    try { records.push(...family.build(7, 0)); }
    catch (error) { if (!missingDataError(error)) throw error; missing.push(`${name}: ${error.message}`); }
  }
  if (missing.length) t.diagnostic(`not verified, data missing: ${missing.join('; ')}`);
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

test('admission preserves the declared evidence level of a case oracle', async () => {
  const [base] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  const { run, trajectory } = await replayReference(base, TOOLS_PROMPT);
  for (const level of ['exact', 'normalized', 'span', 'judged']) {
    const record = structuredClone(base);
    record.semantics.oracle = level;
    const verdict = admitRow({ task: { program_ir: record }, outcome: run.outcome, trajectory });
    assert.equal(verdict.oracle_level, level);
  }
  const described = structuredClone(base);
  described.semantics.oracle = { level: 'span', source: 'checked-excerpt' };
  assert.equal(admitRow({ task: { program_ir: described }, outcome: run.outcome, trajectory }).oracle_level, 'span');
});

test('reference evidence markers do not impose a hidden contract on correct children', async () => {
  const [, record] = FAMILIES.inline_review_each.build(7, 0);
  record.curriculum.reference.root = [['eval',
    { code: 'const kept = await review_each(inbox(), nl`Is priority of ticket at least 3?`);\nkept' }],
    record.curriculum.reference.root.at(-1)];
  record.curriculum.inline = 'optional';
  record.curriculum.reference.children = inboxAnswers(record);
  const { run, trajectory } = await replayReference(record, TOOLS_PROMPT);
  const observed = structuredClone(record);
  observed.curriculum.reference.children[0].evidence = [observed.curriculum.reference.children[0].match];
  assert.equal(admitRow({ task: { program_ir: observed }, outcome: run.outcome, trajectory }).admitted, true);
  observed.curriculum.reference.children[0].evidence = ['marker-never-shown'];
  const different = admitRow({ task: { program_ir: observed }, outcome: run.outcome, trajectory });
  assert.equal(different.admitted, true);
  assert.ok(different.notes.includes('reference_evidence_differs'));
});

test('a row showing an outcome the runtime no longer produces is not admitted', async () => {
  const [record] = FAMILIES.relational_dynamic_snapshot.build(7, 0);
  const { run, trajectory } = await replayReference(record, TOOLS_PROMPT);
  const withTool = (content, code = 'const x = 1') => trajectory.map(turn => ({ ...turn, context: [...turn.context,
    { role: 'assistant', content: '', tool_calls: [{ id: 'c', type: 'function', function: { name: 'eval', arguments: JSON.stringify({ code }) } }] },
    { role: 'tool', tool_call_id: 'c', content }] }));
  const reasons = trajectory => admitRow({ task: { program_ir: record }, outcome: run.outcome, trajectory }).reasons;
  assert.deepEqual(reasons(withTool('Stored local results = [[Promise #2; live value, use it in eval]]')), ['obsolete_outcome:unawaited_promise']);
  assert.deepEqual(reasons(withTool('unknown type name State', 'type State = { n: number };\nlet s: State = { n: 1 };')),
    ['obsolete_outcome:eval_declared_type']);
  // A type nothing declared is still unknown, and the row stands.
  assert.deepEqual(reasons(withTool('unknown type name State')), []);
  assert.deepEqual(reasons(withTool('This call is itself a judgment handed over with nl, so make it here instead of handing it on')),
    ['obsolete_outcome:inline_delegation_ban']);
  assert.deepEqual(reasons(withTool('1:1 invalid-binding: Injected binding "facts" is duplicated.')),
    ['obsolete_outcome:duplicate_injected_binding']);
});

test('successful direct and delegated judgments are admitted; unwarranted edits are rejected', async () => {
  const [semantic, crisp] = FAMILIES.inline_review_each.build(7, 0);
  // Answering a per-item semantic filter directly, without inline children.
  const direct = structuredClone(semantic);
  direct.curriculum.reference.root = [direct.curriculum.reference.root.at(-1)];
  const plain = await replayReference(direct, TOOLS_PROMPT);
  const directVerdict = admitRow({ task: { program_ir: direct }, outcome: plain.run.outcome, trajectory: plain.trajectory });
  assert.deepEqual([directVerdict.reasons, directVerdict.notes], [[], ['judged_directly']]);
  // A successful keyword-assisted route is recorded without declaring it an incorrect answer.
  const keywords = structuredClone(semantic);
  keywords.curriculum.reference.root = [['eval', { code: 'inbox().filter(t => /down|failing|broken/i.test(t.text)).map(t => t.id)' }],
    keywords.curriculum.reference.root.at(-1)];
  const regex = await replayReference(keywords, TOOLS_PROMPT);
  assert.deepEqual(admitRow({ task: { program_ir: keywords }, outcome: regex.run.outcome, trajectory: regex.trajectory }).reasons, []);
  assert.ok(admitRow({ task: { program_ir: keywords }, outcome: regex.run.outcome, trajectory: regex.trajectory }).notes.includes('regex_used'));
  // Delegating a field test is valid even when the seed suggests a direct answer.
  const eager = structuredClone(crisp);
  eager.curriculum.reference.root = [['eval', { code: 'const kept = await review_each(inbox(), nl`Is priority of ticket at least 3?`);\nkept' }],
    eager.curriculum.reference.root.at(-1)];
  eager.curriculum.reference.children = inboxAnswers(eager);
  const extra = await replayReference(eager, TOOLS_PROMPT);
  const delegatedVerdict = admitRow({ task: { program_ir: eager }, outcome: extra.run.outcome, trajectory: extra.trajectory });
  assert.deepEqual([delegatedVerdict.reasons, delegatedVerdict.notes], [[], ['delegated_optional']]);
  // Editing a helper that already meets its contract.
  const [, sound] = FAMILIES.contract_diagnosis.build(7, 0);
  const meddling = structuredClone(sound);
  meddling.curriculum.reference.root.splice(1, 0, ['edit_code', { name: 'line_total', find: 'Math.max(0,', replace_with: 'Math.max(0, 0 +' }]);
  const edited = await replayReference(meddling, TOOLS_PROMPT);
  assert.deepEqual(admitRow({ task: { program_ir: meddling }, outcome: edited.run.outcome, trajectory: edited.trajectory }).reasons, ['unwarranted_edit']);
  const summary = coverage([admitRow({ task: { program_ir: direct }, outcome: plain.run.outcome, trajectory: plain.trajectory })]);
  assert.equal(summary.rejections.inline_missing, undefined);
});

function inboxAnswers(record) {
  const source = record.semantics.files['urgent_queue/inbox.ts'];
  const tickets = JSON.parse(source.slice(source.indexOf('= ') + 2, source.indexOf(';\n')));
  return tickets.map(ticket => ({ match: JSON.stringify(ticket.id), value: ticket.priority >= 3 }));
}

test('the function listing shows TypeScript and natlang doc comments, and an external service by its declaration', async () => {
  const [record] = FAMILIES.child_sufficiency.build(7, 0);
  assert.deepEqual(Object.keys(record.semantics.services), ['records'], 'the records store is external, not a file');
  const opening = await renderOpening(record, TOOLS_PROMPT);
  assert.match(opening, /declare namespace records \{\\n {2}\/\*\* The first-line records for a refund claim\. \*\/\\n {2}export function basic\(/);
  assert.match(opening, /\/\*\* Judge whether evidence settles a refund claim, and which way\. \*\/\\ndeclare function assess\(/);
});

test('inline children below the depth limit have delegation available in their opening', async () => {
  const [, record] = FAMILIES.inline_review_each.build(7, 0);
  record.curriculum.reference.root = [['eval',
    { code: 'const kept = await review_each(inbox(), nl`Is priority of ticket at least 3?`);\nkept' }],
    record.curriculum.reference.root.at(-1)];
  record.curriculum.reference.children = inboxAnswers(record);
  const { trajectory } = await replayReference(record, TOOLS_PROMPT);
  const openings = trajectory.map(turn => String(turn.context[1].content));
  const child = openings.find(text => text.includes('Is priority of ticket at least 3?'));
  assert.match(child, /built-ins nl, iterateOn, transcript and decide/);
  assert.doesNotMatch(child, /nl is not available/);
  assert.match(openings[0], /Eval also has the built-ins nl, iterateOn, transcript and decide/);
});

test('child evidence is attributed by arguments, not another item in captured collections', () => {
  const [record] = FAMILIES.inline_review_each.build(7, 0);
  record.curriculum.decisive = [];
  record.curriculum.inline = 'optional';
  record.curriculum.reference.children = [{ match: 'item-b.txt', evidence: ['evidence for item B'] }];
  const child = (argument, capture) => ({ context: [
    { role: 'system', content: '' },
    { role: 'user', content: 'You are inside this call: nl@eval:1(item: string): string\nInstructions: Read item.' },
    { role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', function: { name: 'eval', arguments: JSON.stringify({
      code: `const inputs = read_inputs();\nconst item = inputs.item;\n// Variables of the calling code, captured by this call:\nconst collection = ${JSON.stringify(capture)};`,
    }) } }] },
    { role: 'tool', tool_call_id: 'scope_0', content: argument },
  ], assistant: { calls: [{ tool: 'return_result', arguments: { status: 'success', value: 'ok' } }] } });
  const a = child('item-a.txt: evidence for item A', 'item-b.txt: evidence for item B');
  const b = child('item-b.txt: evidence for item B', 'item-a.txt: evidence for item A');
  const verdict = trajectory => admitRow({ task: { program_ir: record }, outcome: { status: 'done', accepted: true }, trajectory });
  assert.deepEqual(verdict([a, b]).reasons, []);
  assert.deepEqual(verdict([a]).reasons, [], 'not every item needs its own delegated child');
  b.context[3].content = 'item-b.txt: unread';
  assert.equal(verdict([a, b]).admitted, true);
  assert.ok(verdict([a, b]).notes.includes('reference_evidence_differs'));
});


test('partial benchmark agreement is held for review rather than teaching incorrect per-item decisions', () => {
  const [record] = FAMILIES.inline_review_each.build(7, 0);
  record.curriculum.decisive = [];
  record.curriculum.reference.children = [];
  const row = { task: { program_ir: record }, outcome: { status: 'done', accepted: true,
    oracle: { level: 'agreement', accepted: true, score: 0.95 } }, trajectory: [] };
  assert.ok(admitRow(row).reasons.includes('quality_pending_partial_agreement'));
  row.outcome.oracle = { level: 'exact', accepted: true };
  row.outcome.files_check = { failed: ['r.csv:item'] };
  assert.ok(admitRow(row).reasons.includes('quality_pending_partial_files'));
});
