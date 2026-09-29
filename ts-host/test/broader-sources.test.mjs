import test from 'node:test';
import assert from 'node:assert/strict';
import { treeEdits, buildBroaderSources } from '../scripts/inline-curriculum/broader-sources.mjs';
import { sourceCase } from '../scripts/inline-curriculum/directory-sources.mjs';
import { referenceRow } from '../scripts/inline-curriculum/references.mjs';
import { admitRow } from '../dist/teacher/curriculum.js';
import { TOOLS_PROMPT } from '../dist/native/prompt.js';
import { defaultToolSurfaceHash } from '../dist/teacher/collector.js';
import { checkOracle, namedTreeCanonical } from '../dist/teacher/oracle.js';
const node = (name, children = []) => ({ name, children });
const info = { license: 'CC-BY-SA-3.0', original_split: 'train', revision: 'fixture', sha256: 'a'.repeat(64), files: [] };
const before = node('Root', [node('user', [node('flight', [node('book', [node('object', [node('equals', [node('destination', [node('Paris')])])])])])])]);
const after = structuredClone(before); after.children[0].children[0].children[0].children[0].children[0].children[0].children[0].name = 'Berlin';

test('named-child tree comparison accepts sibling permutations and rejects value loss or duplicates', async () => {
  const expected = node('Root', [node('one'), node('two')]);
  const spec = { level: 'normalized', normalization: 'named-tree' };
  assert.equal((await checkOracle(node('Root', [node('two'), node('one')]), expected, spec)).accepted, true);
  for (const bad of [node('Root', [node('one')]), node('Root', [node('one'), node('one')]),
    node('Root', [node('one'), node('THREE')]), { ...expected, extra: true }, null]) {
    assert.equal((await checkOracle(bad, expected, spec)).accepted, false);
  }
  assert.equal(namedTreeCanonical(node('Root', [node('same'), node('same')])), null);
});

test('tree patch preserves unaffected nodes and rejects reordered ambiguous or changed operators', () => {
  const changes = treeEdits(before, after);
  const state = structuredClone(before);
  Function('state', changes.map(edit => edit.code).join('\n'))(state);
  assert.deepEqual(state, after);
  assert.deepEqual(changes[0].values, ['Berlin']);
  assert.throws(() => treeEdits(node('a', [node('b')]), node('c', [node('b')])), /operator_change/);
  assert.throws(() => treeEdits(node('a', [node('b'), node('b')]), node('a', [node('b'), node('c')])), /ambiguous/);
  assert.throws(() => treeEdits(node('a', [node('b'), node('c')]), node('a', [node('c'), node('b')])), /reordering/);
});

test('non-file tree edit keeps conversation context, rejects implicit values, and replays natively', async () => {
  const turns = [{ turn_id: 'first', utterance: 'Paris', input_dialog_state: null, target_dialog_state: before },
    { turn_id: 'edit', utterance: 'Change it to Berlin', input_dialog_state: before, target_dialog_state: after, input_system_acts: [] },
    { turn_id: 'implicit', utterance: 'Change it there', input_dialog_state: before, target_dialog_state: after }];
  const built = buildBroaderSources({ treedst: { info, rows: [{ session_id: 's1', turns }] } }, 12, sourceCase);
  assert.equal(built.records.length, 1);
  const record = built.records[0];
  assert.equal(record.task_modality, 'tree-edit');
  assert.equal(record.semantics.folder_files, undefined);
  assert.equal(record.semantics.inputs.history.length, 1);
  assert.deepEqual(built.rejected.map(r => r.reason), ['not_an_existing_tree_edit', 'implicit_tree_value_requires_review']);
  const row = await referenceRow(record, 0, { modelId: 'source-static-reference', rootSeed: 929,
    systemPrompt: TOOLS_PROMPT, contextTokens: 16384, maxTurns: 60,
    toolSurfaceSha256: await defaultToolSurfaceHash(), collectionRole: 'reference' });
  assert.equal(admitRow(row).admitted, true);
});

test('scientific QA rejects conflicting answers, missing evidence and held-out documents', () => {
  const a = { unanswerable: false, extractive_spans: ['answer'], evidence: ['answer is here'] };
  const paper = { id: 'p', title: 'Paper', abstract: '', full_text: [{ section_name: 'Section', paragraphs: ['answer is here'] }],
    qas: [{ question_id: 'q', question: 'What is here?', answers: [{ answer: a }, { answer: { ...a, extractive_spans: ['other'] } }] }] };
  const built = buildBroaderSources({ qasper: { info, rows: [paper] } }, 12, sourceCase);
  assert.equal(built.records.length, 0); assert.equal(built.rejected[0].reason, 'answer_annotation_disagreement');
  paper.qas[0].answers.pop();
  assert.equal(buildBroaderSources({ qasper: { info, rows: [paper] } }, 12, sourceCase).records.length, 1);
  assert.equal(buildBroaderSources({ qasper: { info: { ...info, held_out_ids: ['p'] }, rows: [paper] } }, 12, sourceCase).rejected[0].reason, 'held_out_paper');
  const claim = { id: 1, claim: 'Claim', cited_doc_ids: [3], evidence: { 3: [{ label: 'SUPPORT', sentences: [0] }] } };
  const data = { claims: [claim], corpus: [{ doc_id: 3, title: 'Doc', abstract: ['Evidence.'] }] };
  assert.equal(buildBroaderSources({ scifact: { info: { ...info, held_out_ids: ['3'] }, rows: [data] } }, 12, sourceCase).records.length, 0);
  claim.evidence[3][0].sentences = [5];
  assert.equal(buildBroaderSources({ scifact: { info, rows: [data] } }, 12, sourceCase).rejected[0].reason, 'missing_claim_evidence');
});

test('obsolete formatting/tree failures and unreviewed scientific span equivalence stay out of negatives', async () => {
  const {runtimeFailureReason} = await import('../dist/teacher/curriculum-policy.js');
  const row = (source, oracle, accepted = false, rejection_reasons = ['answer']) => ({
    task: {program_ir: {source, semantics: {oracle}}}, provenance: {runtime_contract_version: 17},
    outcome: {accepted, rejection_reasons},
  });
  assert.equal(runtimeFailureReason(row('qasper', 'exact')), 'unreviewed_extractive_answer_equivalence');
  assert.equal(runtimeFailureReason(row('qasper', 'exact', true)), undefined);
  assert.equal(runtimeFailureReason(row('qasper', 'exact', false, ['files'])), undefined);
  assert.equal(runtimeFailureReason(row('tatqa', 'exact')), 'obsolete_json_format_oracle');
  assert.equal(runtimeFailureReason(row('tatqa', {level: 'normalized', normalization: 'json-string-record'})), undefined);
  assert.equal(runtimeFailureReason(row('treedst', 'exact')), 'obsolete_named_tree_oracle');
  assert.equal(runtimeFailureReason(row('treedst', {level: 'normalized', normalization: 'named-tree'})), undefined);
});
