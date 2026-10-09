import assert from 'node:assert/strict';
import test from 'node:test';
import { quarantineReason, generationHoldReason, runtimeFailureReason, retiredFamily, RETIRED_FAMILIES,
  legacyMarkdownTerminalNewlineFailure } from '../dist/teacher/curriculum-policy.js';
import { QUARANTINE_RULES, RUNTIME_FAILURE_RULES, GENERATION_HOLD_RULES, firstReason, holds } from '../dist/teacher/curriculum-rules.js';
import { hasExistingTreeValueContract } from '../dist/teacher/tree-contract.js';
import { sourceReviewReason } from '../dist/teacher/source-review.js';
import { runtimeFailureRulesFor } from '../dist/benchmarks/registry.js';

/*
 * The differential test: the rule ladders as they were written before they became a data registry (plans/NATLANG_NATIVE_REVIEW.md,
 * P6), copied verbatim, against the registry-driven policy on a large seeded sample of records and rows that reaches every rung.
 */
function legacyQuarantineReason(record) {
  if (record.source === 'treedst' && !hasExistingTreeValueContract(record.semantics?.inputs?.state, record.semantics?.expected))
    return 'unverified_tree_transition_contract';
  const sourceReview = sourceReviewReason(record);
  if (sourceReview) return sourceReview;
  if (record.family === 'cb_highlighter' && record.generation?.highlighter_quality_version !== 2)
    return 'legacy_highlighter_oracle';
  const curriculum = record.curriculum;
  if (curriculum?.family === 'commaqa_numeric' && (curriculum.family_version ?? 1) < 3)
    return 'legacy_numeric_reference_contract';
  if (curriculum?.family === 'commaqa_question' && curriculum.family_version === 3)
    return 'unverified_movie_schema_contract';
  if (curriculum?.family === 'entailment_premises' && curriculum.variant === 'premise_removed')
    return 'unverified_counterfactual';
  if (curriculum?.family === 'folder_extract' && (!record.semantics.files_oracle?.quote_sources ||
      !record.semantics.files_oracle.return_count)) return 'legacy_extraction_contract';
  if (curriculum?.family === 'folder_edit' && (!record.semantics.files_oracle?.rubric ||
      !record.semantics.files_oracle.return_count)) return 'legacy_rewrite_contract';
  if (curriculum?.family === 'folder_index' && (!record.semantics.files_oracle?.return_count ||
      record.semantics.files_oracle.total === undefined)) return 'legacy_counts_contract';
  if (curriculum?.family === 'folder_triage' && record.semantics.files_oracle?.compare !== 'moves')
    return 'legacy_move_contract';
  if (curriculum?.family === 'folder_find' && !curriculum.answer_evidence?.length)
    return 'legacy_article_evidence_policy';
  if (curriculum?.family === 'folder_mixed' && record.dataset === 'banking77' && curriculum.payment_scope_version !== 2)
    return 'legacy_payment_scope';
}
const legacyGenerationHold = record => record.source === 'qasper' ? 'awaiting_extractive_equivalence_oracle' : undefined;
function legacyRuntimeFailureReason(row) {
  if (row.outcome?.accepted !== false) return;
  const record = row.task.program_ir;
  if (legacyMarkdownTerminalNewlineFailure(row)) return 'legacy_markdown_terminal_newline_oracle';
  if (record.source === 'treedst' && (typeof record.semantics.oracle !== 'object' ||
      record.semantics.oracle.normalization !== 'named-tree')) return 'obsolete_named_tree_oracle';
  for (const rule of runtimeFailureRulesFor(String(record.source))) {
    const reason = rule(row, record);
    if (reason) return reason;
  }
  if ((record.source === 'qasper' || record.source === 'musique') && row.outcome?.rejection_reasons?.includes('answer'))
    return 'unreviewed_extractive_answer_equivalence';
  if (Number(row.provenance?.runtime_contract_version ?? 0) >= 17) return;
  const family = record.curriculum?.family ?? record.family;
  if (family === 'inline_late_binding' || (record.family === 'cb_reconciliation' &&
      JSON.stringify(record.semantics.expected).includes('__proto__')) ||
      (family === 'logic_proof_verifier' && JSON.stringify(row.trajectory).includes('bad character at')))
    return 'obsolete_runtime_contract';
}

// A small seeded generator: the same sample on every run.
function rng(seed) { let state = seed >>> 0; return () => (state = (state * 1664525 + 1013904223) >>> 0) / 2 ** 32; }
const pick = (next, list) => list[Math.floor(next() * list.length)];

const FAMILIES = ['commaqa_numeric', 'commaqa_question', 'entailment_premises', 'folder_extract', 'folder_edit', 'folder_index', 'folder_triage',
  'folder_find', 'folder_mixed', 'inline_late_binding', 'logic_proof_verifier', 'inline_type_repair', 'other', undefined];
const SOURCES = ['treedst', 'qasper', 'musique', 'commitpack', 'banking', 'tatqa', 'other', undefined];
const ORACLES = [{ normalization: 'named-tree' }, { normalization: 'other' }, {}, 'text', undefined];
const flag = [undefined, 0, 1, '', 'x', [], ['a'], true, false];
function oracleFiles(next) {
  const files = {};
  for (const key of ['quote_sources', 'return_count', 'rubric', 'total']) if (next() < 0.7) files[key] = pick(next, flag.concat([0, 3]));
  if (next() < 0.7) files.compare = pick(next, ['moves', 'markdown-terminal-newline', 'other']);
  return next() < 0.1 ? undefined : files;
}
function sampleRecord(next, index) {
  const family = pick(next, FAMILIES);
  const curriculum = next() < 0.1 ? undefined : { family, family_version: pick(next, [undefined, 1, 2, 3, 4, '2']), variant: pick(next, [undefined, 'premise_removed', 'other']),
    answer_evidence: pick(next, [undefined, [], ['a'], 'x']), payment_scope_version: pick(next, [undefined, 1, 2]) };
  return { version: 'natlang.program/2', id: `record-${index}`, kind: 'program', source: pick(next, SOURCES),
    family: pick(next, ['cb_highlighter', 'cb_reconciliation', 'other', undefined]), dataset: pick(next, ['banking77', 'other', undefined]),
    generation: next() < 0.2 ? undefined : { highlighter_quality_version: pick(next, [undefined, 1, 2]) }, curriculum,
    semantics: { root: 'main.nl', files: {}, inputs: { state: pick(next, [undefined, {}, { a: 1 }]) },
      expected: pick(next, [undefined, 1, 'x', { v: '__proto__' }, ['__proto__']]), oracle: pick(next, ORACLES), files_oracle: oracleFiles(next) } };
}
function sampleRow(next, record) {
  const expected = record.semantics.expected;
  // The legacy ladder reads JSON.stringify(expected).includes(...) for cb_reconciliation: keep it defined, as real rows do.
  if (record.family === 'cb_reconciliation' && expected === undefined) record.semantics.expected = 'x';
  return { task: { program_ir: record }, provenance: next() < 0.2 ? undefined : { runtime_contract_version: pick(next, [undefined, null, 0, 16, 17, 18, '17', 'x']) },
    outcome: { accepted: pick(next, [false, false, false, true, undefined]), status: pick(next, ['done', 'failed']),
      rejection_reasons: pick(next, [undefined, ['answer'], ['files'], ['answer', 'files'], []]), checks: {} },
    trajectory: pick(next, [[], [{ text: 'bad character at 3' }], [{ text: 'fine' }]]) };
}

test('quarantine, generation-hold and runtime-failure ladders: registry policy equals the original ladders on a seeded sample', () => {
  const next = rng(20261009), seen = { quarantine: new Set(), hold: new Set(), runtime: new Set() };
  for (let index = 0; index < 40000; index++) {
    const record = sampleRecord(next, index), row = sampleRow(next, record);
    const quarantine = legacyQuarantineReason(record);
    assert.equal(quarantineReason(record), quarantine, `quarantine ${JSON.stringify(record)}`);
    const hold = legacyGenerationHold(record);
    assert.equal(generationHoldReason(record), hold);
    const runtime = legacyRuntimeFailureReason(row);
    assert.equal(runtimeFailureReason(row), runtime, `runtime ${JSON.stringify(row)}`);
    seen.quarantine.add(quarantine); seen.hold.add(hold); seen.runtime.add(runtime);
  }
  // The sample reaches every rung of the registry (the benchmark and source-review steps give their own reasons).
  for (const entry of QUARANTINE_RULES) if (entry.reason) assert.ok(seen.quarantine.has(entry.reason), `quarantine rung ${entry.id} never fired`);
  for (const entry of GENERATION_HOLD_RULES) assert.ok(seen.hold.has(entry.reason));
  for (const entry of RUNTIME_FAILURE_RULES) if (entry.reason && entry.id !== 'markdown-terminal-newline')
    assert.ok(seen.runtime.has(entry.reason), `runtime rung ${entry.id} never fired`);
});

test('the legacy CommitPack markdown rung still fires through its named predicate', () => {
  const record = { version: 'natlang.program/2', id: 'cp-1', kind: 'program', source: 'commitpack',
    semantics: { root: 'main.nl', files: {}, expected: 'ok',
      folder_files: { 'change-request.json': JSON.stringify({ path: 'a.md', find: 'old', replace_with: 'new' }), 'a.md': 'old\n' },
      expected_files: { 'change-request.json': JSON.stringify({ path: 'a.md', find: 'old', replace_with: 'new' }), 'a.md': 'new\n' } } };
  const row = { task: { program_ir: record }, provenance: {},
    outcome: { accepted: false, status: 'done', rejection_reasons: ['files'], value: 'ok', files: { 'change-request.json': JSON.stringify({ path: 'a.md', find: 'old', replace_with: 'new' }), 'a.md': 'new' },
      checks: { answer: true, file_return_consistency: true, files: false }, files_check: { failed: ['a.md'] } },
    trajectory: [] };
  assert.equal(legacyMarkdownTerminalNewlineFailure(row), true);
  assert.equal(runtimeFailureReason(row), 'legacy_markdown_terminal_newline_oracle');
  assert.equal(legacyRuntimeFailureReason(row), 'legacy_markdown_terminal_newline_oracle');
});

test('retired families are data and the set is unchanged', () => {
  assert.deepEqual([...RETIRED_FAMILIES], ['inline_type_repair']);
  assert.equal(retiredFamily({ curriculum: { family: 'inline_type_repair' } }), 'inline_type_repair');
  assert.equal(retiredFamily({ curriculum: { family: 'folder_find' } }), undefined);
});

test('the evaluator: conditions, order, steps and unknown names', () => {
  const context = { record: { a: { b: 2 }, list: ['x'], text: 'abc' }, curriculum: undefined };
  const at = (path, op, value, extra = {}) => holds({ path, op, value, ...extra }, context);
  assert.equal(at('record.a.b', 'eq', 2), true);
  assert.equal(at('record.a.b', 'ne', 2), false);
  assert.equal(at('record.a.c', 'undefined'), true);
  assert.equal(at('record.a.c', 'falsy'), true);
  assert.equal(at('record.a.b', 'in', [1, 2]), true);
  assert.equal(at('record.a.c', 'lt', 3, { default: 1 }), true);
  assert.equal(at('record.a.b', 'below-as-number', 3, { default: 0 }), true);
  assert.equal(at('record.text', 'below-as-number', 3, { default: 0 }), true, 'not a number: below');
  assert.equal(at('record.list', 'includes', 'x'), true);
  assert.equal(at('record.text', 'includes', 'b'), true);
  assert.equal(at('record.a', 'json-includes', '"b":2'), true);
  assert.equal(holds({ any: [] }, context), false);
  assert.equal(holds({ all: [] }, context), true);
  assert.throws(() => holds({ predicate: 'nope' }, context), /no predicate named "nope"/);
  const registry = [{ step: 'first' }, { id: 'r', reason: 'rung', when: [], violates: { all: [] } }];
  assert.equal(firstReason(registry, context, { steps: { first: () => 'from-step' } }), 'from-step');
  assert.equal(firstReason(registry, context, { steps: { first: () => undefined } }), 'rung');
  assert.throws(() => firstReason(registry, context), /no step named "first"/);
});
