import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { existsSync } from 'node:fs';
import { test } from 'node:test';
import { session as open, ts } from './support/natlang.mjs';
import { executeProgram } from '../dist/teacher/collector.js';
import { admitRow, replayReference } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { idempotentRetry } from '../scripts/inline-curriculum/followup.mjs';
import { commaqaArithmetic, commaqaNumeric, commaqaQuestion, numericNationalityFact, explicitRelationFact, explicitStoreSchema } from '../scripts/inline-curriculum/sources-ai2.mjs';
import { factStore } from '../scripts/inline-curriculum/logic.mjs';
import { externalModule } from '../dist/native/external.js';
import { auditEvidence } from '../scripts/inline-curriculum/audit-commaqa-evidence.mjs';
import { recursionRewrite } from '../scripts/inline-curriculum/failure.mjs';
import { modernizeContracts } from '../scripts/inline-curriculum/modernize-contracts.mjs';
import { eventRetry } from '../scripts/inline-curriculum/failure.mjs';

test('movie facts expose every source relation with consistent labels and entity order', () => {
  const labels = { table_directed: ['movie', 'director'], table_maward: ['movie', 'movie award'],
    table_writer: ['movie', 'writer'], text_writer: ['movie', 'writer'], table_year: ['movie', 'release year'], text_actor: ['movie', 'actor'],
    text_dob: ['person', 'birth year'], text_nation: ['person', 'country'],
    text_paward: ['person', 'personal award'], table_paward: ['person', 'personal award'], text_produced: ['movie', 'producer'] };
  for (const [predicate, [left, right]] of Object.entries(labels))
    assert.equal(explicitRelationFact(`${predicate}(One, Two)`), `${left}: One ; ${right}: Two.`);
  assert.throws(() => explicitRelationFact('unknown(One, Two)'), /unsupported/);
});

test('movie schema derives specialist ownership from the actual world', () => {
  assert.equal(explicitStoreSchema(['table_writer', 'text_paward']),
    'table_expert holds movie/writer credits. text_expert holds personal awards.');
  assert.equal(explicitStoreSchema(['text_writer', 'table_paward']),
    'table_expert holds personal awards. text_expert holds movie/writer credits.');
});

test('board migration restarts exactly-once retries with irreversible repeated-write prefixes', () => {
  const original = { id: 'board', source_revisions: [], curriculum: { family: 'event_retry', family_version: 2 },
    semantics: { files: {}, services: { board: 'const REJECT = false;' } },
    handoff: { prefix: [[{ calls: [['eval', { code: 'await board.commit_move({});' }]] },
      { calls: [['eval', { code: 'await board.commit_move({});' }]] }]] } };
  const { record, changes } = modernizeContracts(original);
  assert.equal(record.handoff, undefined);
  assert.match(changes[0], /irreversible repeated writes/);
  assert.ok(original.handoff, 'historical input remains intact');
});

test('numeric evidence audit detects source mismatches independently of arithmetic', () => {
  const world = {
    kb: { table_nationj: ['table_nationj(Ada, Vexa)'], text_jthrow: ['text_jthrow(Ada, 88.0)'] },
    pred_lang_config: { table: ['d', 'j'].map(s => ({ predicate: `nation${s}_p($1, ?)`,
      questions: [`Who are the ${s === 'd' ? 'discus' : 'javelin'} throwers?`], steps: [{ question: `table_nation${s}(?, $1)` }] })) },
    per_fact_context: { 'table_nationj(Ada, Vexa)': 'Ada, Vexa, Discus',
      'text_jthrow(Ada, 88.0)': 'Ada threw javelin 88.0 meters.' },
    qa_pairs: [{ facts_used: ['table_nationj(Ada, Vexa)'] }],
  };
  assert.equal(auditEvidence([world], 'fixture').raw_inversions, 1);
  assert.deepEqual(auditEvidence([world], 'fixture').failures, []);
  world.per_fact_context['text_jthrow(Ada, 88.0)'] = 'Ada threw discus 88.0 meters.';
  assert.equal(auditEvidence([world], 'fixture').failures[0].kind, 'throw_rendering');
});

test('failed eval distinguishes discarded bindings from mutations to callable module state', async () => {
  const { session, lam } = open({ type: '() => number', instructions: 'Count.', codebase: {
    counter: ts('counter', 'let count = 0; export function next(): number { count += 1; return count; }'),
  } });
  const failed = await session.applyAsync('eval', { code: 'const count = counter.next(); throw new Error("view failed");' });
  assert.equal(failed.kind, 'error');
  assert.match(failed.text, /state changes.*callable modules.*not undone/);
  assert.equal(Object.hasOwn(lam.let, 'count'), false);
  const inspected = await session.applyAsync('eval', { code: 'counter.next()' });
  assert.equal(inspected.value, 2);
});

test('supplied highlighter consumes matchAll results through a finite array', async () => {
  const source = await readFile(new URL('../../codebases/highlighter/highlight/highlight_file/split_source.ts', import.meta.url), 'utf8');
  const codebase = { split_source: ts('split_source', source, {}, {
    SourceFile: '{ path: string, text: string }',
    Parts: '{ frontmatter: string, lines: string[], functions: string[], is_code: boolean }',
  }) };
  const { session } = open({ type: '() => string[]', instructions: 'List functions.', codebase });
  const result = await session.applyAsync('eval', { code:
    'return split_source({ path: "sample.nl", text: "import helper from \\\"helper\\\"\\n\\n---\\nargs: {}\\n---\\nhelper()\\nother()" }).functions;' });
  assert.equal(result.kind, 'ok', result.text);
  assert.deepEqual(result.value, ['helper', 'other']);
});

test('lost-ack service is readable and rejects changed commands without additional delivery', async () => {
  const [record] = idempotentRetry(7, 0);
  const key = record.semantics.effects['mail.send'].fail_key;
  let turn = 0;
  const trajectory = [];
  const driver = async request => {
    trajectory.push(request.messages.at(-1).content);
    if (turn++ === 0) return { calls: [['read_code', { name: 'mail' }]] };
    if (turn === 2) return { calls: [['eval', { code: `
      const r = reminders.find(item => item.id === ${JSON.stringify(key)});
      try { mail.send({ key: r.id, to: r.to, text: r.text }); } catch (error) {}
      let changedRejected = false;
      try { mail.send({ key: r.id, to: r.to, text: r.text + " changed" }); } catch (error) { changedRejected = true; }
      if (!changedRejected) throw new Error("changed retry accepted");
      for (const reminder of reminders) mail.send({ key: reminder.id, to: reminder.to, text: reminder.text });
      return reminders.map(item => item.id);` }]] };
    return { text: 'done' };
  };
  const run = await executeProgram(record, driver, { rootSeed: 7, systemPrompt: TOOLS_PROMPT, contextTokens: 16384, maxTurns: 4 });
  assert.equal(run.outcome.accepted, true, JSON.stringify(run.outcome.checks));
  assert.ok(trajectory.some(text => /normal return \(null\) confirms delivery/.test(String(text))));
  assert.equal(run.outcome.effects['mail.send'].length, 4);
});

test('numeric reference computes its intermediates and unknown arithmetic fails closed', () => {
  assert.equal(commaqaArithmetic({ op: 'select', q: 'What is the difference between #3 and #4?' }), 'Math.abs(step3 - step4)');
  assert.match(commaqaArithmetic({ op: 'projectValues', q: 'What is the smallest value among #2?' }), /Math.min/);
  assert.match(commaqaArithmetic({ op: 'filterValues_keys', q: 'Is #3 greater than 42.2?' }), /value > 42.2/);
  assert.throws(() => commaqaArithmetic({ op: 'unknown', q: 'Invent a value' }), /unsupported/);
});

test('numeric table evidence uses original sport predicates and rejects unknown facts', () => {
  assert.equal(numericNationalityFact('table_nationd(Ada, Vexa)'), 'athlete: Ada ; country: Vexa ; sport: Discus Throw.');
  assert.equal(numericNationalityFact('table_nationj(Bob, Vexa)'), 'athlete: Bob ; country: Vexa ; sport: Javelin Throw.');
  assert.throws(() => numericNationalityFact('table_other(Ada, Vexa)'), /unsupported/);
});

test('fact-store search retrieves literal matches and leaves semantic decisions to the caller', () => {
  const store = externalModule('facts', factStore([{ id: '1', text: 'Ada is from Vexa.' },
    { id: '2', text: 'Ada threw 42.1.' }, { id: '3', text: 'Bob is from Vexa.' }], 'Facts.', { search: true }));
  assert.deepEqual(store.exports.search(' ada ').map(row => row.id), ['1', '2']);
  assert.deepEqual(store.exports.search('['), []);
  assert.throws(() => store.exports.search('  '), /nonempty/);
  assert.equal(store.exports.page(1).length, 3);
});

test('admission permits preventing a planted compile failure but does not waive runtime evidence', async () => {
  const [record] = recursionRewrite(7, 0);
  const { run, trajectory } = await replayReference(record, TOOLS_PROMPT);
  const continued = trajectory.slice(1).map(turn => ({ ...turn, context: turn.context.filter(message =>
    !(message.role === 'tool' && String(message.content).includes('recursion is not allowed'))) }));
  record.handoff = { kind: 'failed_action', call: 0, prefix: [[]],
    rejected: { calls: [['eval', { code: record.semantics.failure_seed.code }]] } };
  const outcome = { ...run.outcome, seeded_failure: { observed: false, replaced_by_handoff: true } };
  const verdict = admitRow({ task: { program_ir: record }, outcome, trajectory: continued });
  assert.equal(verdict.admitted, true, JSON.stringify(verdict.reasons));
  assert.ok(verdict.notes.includes('planted_compile_failure_prevented'));
  record.handoff.rejected.calls[0][1].code = 'a different action';
  assert.ok(admitRow({ task: { program_ir: record }, outcome, trajectory: continued }).reasons.some(reason => reason.startsWith('missing_observation')));
  record.handoff.rejected.calls[0][1].code = record.semantics.failure_seed.code;
  record.semantics.failure_seed.kind = 'runtime';
  assert.ok(admitRow({ task: { program_ir: record }, outcome, trajectory: continued }).reasons.some(reason => reason.startsWith('missing_observation')));
});

test('legacy editable board migration preserves gold and source identity and drops incompatible prefixes', async () => {
  const [current] = eventRetry(7, 0);
  const old = structuredClone(current);
  old.semantics.files['move_card/board.ts'] = old.semantics.services.board;
  delete old.semantics.services.board;
  old.handoff = { prefix: [[{ calls: [['edit_code', { name: 'board', find: 'REJECT', replace_with: 'false' }]] }]] };
  const migrated = modernizeContracts(old);
  assert.ok(migrated.changes.length);
  assert.deepEqual(migrated.record.semantics.expected, old.semantics.expected);
  assert.deepEqual(migrated.record.source_groups, old.source_groups);
  assert.equal(migrated.record.handoff, undefined);
  assert.ok(old.handoff, 'original evidence must not be mutated');
  const replayed = await replayReference(migrated.record, TOOLS_PROMPT);
  assert.equal(replayed.run.outcome.accepted, true, JSON.stringify(replayed.run.outcome));
});

for (const [variant, build] of [['numeric', commaqaNumeric], ['explicit', commaqaQuestion]]) {
  const available = existsSync(new URL(`../../vendor/datasets/commaqa/v1/commaqa_${variant}`, import.meta.url));
  for (const index of [0, 1, 2, 3, 4, 5]) test(`CommaQA ${variant} ${index}: source reference uses real arithmetic and scoped specialists`, { skip: !available }, async () => {
    const [record] = build(7, index);
    assert.equal(record.curriculum.family_version, variant === 'numeric' ? 3 : 4);
    assert.match(record.semantics.files['answer_question/text_expert.nl'], /measurements as strings/);
    const result = await replayReference(record, TOOLS_PROMPT);
    assert.equal(result.run.outcome.accepted, true, JSON.stringify(result.run.outcome));
  });
}

test('TextWorld navigation migration preserves goals and exposes only current exits', async () => {
  const { textworldIterate } = await import('../scripts/inline-curriculum/textworld.mjs');
  const [fresh] = textworldIterate(7, 0);
  assert.equal(fresh.curriculum.family_version, 2);
  const old = structuredClone(fresh);
  old.curriculum.family_version = 1;
  const worldPath = Object.keys(old.semantics.files).find(path => path.endsWith('/world.ts'));
  const worldSource = old.semantics.services?.world ?? old.semantics.files[worldPath];
  const legacySource = worldSource.replace(/  const exits = commands\(\).filter[^\n]+\n  if \(exits.length\) lines.push[^\n]+\n/, '');
  if (old.semantics.services?.world) old.semantics.services.world = legacySource;
  else old.semantics.files[worldPath] = legacySource;
  old.handoff = { prefix: [[{ calls: [['eval', { code: 'world.look()' }]] }]] };
  const migrated = modernizeContracts(old);
  assert.equal(migrated.record.handoff, undefined);
  assert.deepEqual(migrated.record.semantics.expected, old.semantics.expected);
  assert.deepEqual(migrated.record.source_groups, old.source_groups);
  assert.match(migrated.record.semantics.services?.world ?? migrated.record.semantics.files[worldPath], /Available exits/);
  assert.match(migrated.record.semantics.files[migrated.record.semantics.root], /Keep exploration notes/);
  const replay = await replayReference(migrated.record, TOOLS_PROMPT);
  assert.equal(replay.run.outcome.accepted, true, JSON.stringify(replay.run.outcome));
});
