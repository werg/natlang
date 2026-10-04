import test from 'node:test';
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { NLLB_CANDIDATE, NLLB_REFERENCE_POLICY, nllbProgramBinding, nllbReferenceVisible,
  verifyNllbBundle } from '../scripts/inline-curriculum/nllb-reference-policy.mjs';

const sha = value => createHash('sha256').update(value, 'utf8').digest('hex');
function fixture() {
  const sourceText = 'Original sentence in a source language.';
  const targetText = 'A faithful translation in the target language.';
  const taskId = 'nllb-seed:ace_Latn->eng_Latn:line-1', group = 'nllb-seed/text-component/test';
  const reference = { task_id: taskId, source_group: group, role: 'train_support', split: 'train', source_pair: 'ace_Latn-eng_Latn',
    source_language: 'Acehnese (Latin script)', target_language: 'English', target_text: targetText,
    source_line_content_sha256: sha(sourceText), target_line_content_sha256: sha(targetText),
    visibility: 'host-only', license: NLLB_CANDIDATE.license };
  const task = { version: 'natlang.program/2', id: `nllb-reference:${taskId}`, split: 'train',
    source_ids: [taskId], source_groups: [group],
    generation: { generator: 'natlang.nllb_seed_translation_static_adapter/1', derived_role: 'train_support' },
    semantics: { root: 'translate.nl', files: { 'translate.nl': 'source program' },
      inputs: { source_language: reference.source_language, source_text: sourceText, target_language: reference.target_language },
      expected: targetText },
    external_source: { original_split: 'train', source_pair_folder: reference.source_pair, license: NLLB_CANDIDATE.license,
      source_line_content_sha256: reference.source_line_content_sha256, target_line_content_sha256: reference.target_line_content_sha256,
      archive_sha256: NLLB_CANDIDATE.archive,
      nllb_reference: { policy: NLLB_REFERENCE_POLICY, candidate_manifest_sha256: NLLB_CANDIDATE.manifest,
        candidate_ir_sha256: NLLB_CANDIDATE.ir, host_references_sha256: NLLB_CANDIDATE.references,
        archive_sha256: NLLB_CANDIDATE.archive, task_id: taskId, source_group: group,
        source_language: reference.source_language, target_language: reference.target_language } } };
  const row = { trajectory: [{ context: [{ role: 'user', content: `Translate ${sourceText} from ${reference.source_language} to ${reference.target_language}.` }] }] };
  return { task, row, reference };
}

test('NLLB static answer binds to exact source, host reference, language pair, and support role', () => {
  const { task, row, reference } = fixture();
  assert.equal(nllbProgramBinding(task, reference), true);
  assert.equal(nllbReferenceVisible(task, row), true);
  assert.equal(nllbProgramBinding(task, { ...reference, target_text: 'substituted target' }), false);
  assert.equal(nllbProgramBinding({ ...task, semantics: { ...task.semantics, expected: 'other answer' } }, reference), false);
  assert.equal(nllbProgramBinding({ ...task, semantics: { ...task.semantics,
    inputs: { ...task.semantics.inputs, target_language: 'Another language' } } }, reference), false);
  assert.equal(nllbProgramBinding({ ...task, source_groups: ['different-group'] }, reference), false);
  assert.equal(nllbProgramBinding({ ...task, generation: { ...task.generation, derived_role: 'train_query' } }, reference), false);
  assert.equal(nllbProgramBinding({ ...task, split: 'test' }, reference), false);
  assert.equal(nllbProgramBinding({ ...task, external_source: { ...task.external_source,
    source_line_content_sha256: sha('substituted source') } }, reference), false);
});

test('NLLB direct answer policy rejects a missing visible source input', () => {
  const { task, row } = fixture();
  row.trajectory[0].context[0].content = 'Translate this sentence from Acehnese to English.';
  assert.equal(nllbReferenceVisible(task, row), false);
});

test('mutated reference-source policy fails closed before resolving candidate files', async () => {
  const policy = { version: NLLB_REFERENCE_POLICY, candidate_manifest_sha256: NLLB_CANDIDATE.manifest,
    candidate_ir_sha256: NLLB_CANDIDATE.ir, host_references_sha256: NLLB_CANDIDATE.references,
    archive_sha256: NLLB_CANDIDATE.archive, license: NLLB_CANDIDATE.license, support_only: true,
    reference_derived: true, bindings: { path: 'bindings.jsonl', sha256: '0'.repeat(64), rows: 1 } };
  await assert.rejects(verifyNllbBundle('/tmp/not-created/static.manifest.json', {
    cases: 1, source_answer_policy: NLLB_REFERENCE_POLICY, source_proof: { sha256: '0'.repeat(64) },
    nllb_reference_policy: { ...policy, support_only: false },
  }), /nllb_reference_policy_mismatch/);
});
