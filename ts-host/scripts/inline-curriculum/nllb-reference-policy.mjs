/** Narrow source policy for NLLB-Seed human-reference static SFT. */
import { createHash } from 'node:crypto';
import { dirname, resolve } from 'node:path';
import { readFile } from 'node:fs/promises';
import { jsonlRows, fileDigest } from '../jsonl-stream.mjs';

export const NLLB_REFERENCE_POLICY = 'natlang.nllb_seed_human_reference_sft/1';
export const NLLB_CANDIDATE = Object.freeze({
  manifest: '555cc96076afe98e91536c1b98ada4bf61ebf4ee7b4b556ccdb7025e51c53b72',
  ir: '413c7d648ba4eef126ae3cf1750cecfdb7b1470cd489e3999d50a3757803e260',
  references: 'd9dd200a7cab13de7e53bc68c63700803c71c5fe233ba52c036d0ddf87c6640d',
  archive: 'd269fa2bebba88c85de8912a5c1e5ddd9fd2086f8432b29f868627918b131c6d',
  license: 'CC-BY-SA-4.0',
});

const sha = value => createHash('sha256').update(value).digest('hex');
const canonical = value => JSON.stringify(value, Object.keys(value ?? {}).sort());
const exact = (a, b) => JSON.stringify(a) === JSON.stringify(b);

export function nllbProgramBinding(program, reference) {
  const source = program?.external_source ?? {};
  const input = program?.semantics?.inputs ?? {};
  const external = source.nllb_reference ?? {};
  return program?.version === 'natlang.program/2' &&
    program?.generation?.generator === 'natlang.nllb_seed_translation_static_adapter/1' &&
    program?.split === 'train' && program?.generation?.derived_role === 'train_support' &&
    Array.isArray(program.source_ids) && program.source_ids.length === 1 &&
    program.source_ids[0] === reference?.task_id &&
    Array.isArray(program.source_groups) && program.source_groups.length === 1 &&
    program.source_groups[0] === reference?.source_group &&
    reference?.role === 'train_support' && reference?.split === 'train' &&
    reference?.visibility === 'host-only' && reference?.license === NLLB_CANDIDATE.license &&
    reference?.target_text === program?.semantics?.expected &&
    sha(Buffer.from(String(input.source_text ?? ''), 'utf8')) === source?.source_line_content_sha256 &&
    (!reference?.source_line_content_sha256 || reference.source_line_content_sha256 === source.source_line_content_sha256) &&
    sha(Buffer.from(String(reference?.target_text ?? ''), 'utf8')) === reference?.target_line_content_sha256 &&
    input.source_language === reference.source_language && input.target_language === reference.target_language &&
    (!reference.source_line_content_sha256 || source.source_line_content_sha256 === reference.source_line_content_sha256) &&
    source.target_line_content_sha256 === reference.target_line_content_sha256 &&
    (!reference.source_pair || source.source_pair_folder === reference.source_pair) && source.original_split === 'train' &&
    source.license === NLLB_CANDIDATE.license && source.archive_sha256 === NLLB_CANDIDATE.archive &&
    external.policy === NLLB_REFERENCE_POLICY && external.candidate_manifest_sha256 === NLLB_CANDIDATE.manifest &&
    external.candidate_ir_sha256 === NLLB_CANDIDATE.ir && external.host_references_sha256 === NLLB_CANDIDATE.references &&
    external.task_id === reference.task_id && external.source_group === reference.source_group;
}

/** Validate source archive artifacts and exact selected source/reference joins before static conversion. */
export async function verifyNllbBundle(manifestPath, bundleManifest) {
  const policy = bundleManifest.nllb_reference_policy;
  if (!policy) {
    if (bundleManifest.source_answer_policy === NLLB_REFERENCE_POLICY) throw new Error('nllb_reference_policy_missing');
    return null;
  }
  if (policy.version !== NLLB_REFERENCE_POLICY || policy.candidate_manifest_sha256 !== NLLB_CANDIDATE.manifest ||
      policy.candidate_ir_sha256 !== NLLB_CANDIDATE.ir || policy.host_references_sha256 !== NLLB_CANDIDATE.references ||
      policy.archive_sha256 !== NLLB_CANDIDATE.archive || policy.license !== NLLB_CANDIDATE.license ||
      policy.support_only !== true || policy.reference_derived !== true || !policy.bindings ||
      typeof policy.bindings.path !== 'string' || typeof policy.bindings.sha256 !== 'string' ||
      policy.bindings.sha256 !== bundleManifest.source_proof?.sha256 ||
      policy.bindings.rows !== bundleManifest.cases || bundleManifest.source_answer_policy !== NLLB_REFERENCE_POLICY)
    throw new Error('nllb_reference_policy_mismatch');
  const root = dirname(resolve(manifestPath));
  if (policy.candidate_dir !== '../nllb-seed-static-full-v2') throw new Error('nllb_candidate_path_not_allowlisted');
  const candidateDir = resolve(root, policy.candidate_dir);
  const candidateManifestPath = resolve(candidateDir, 'manifest.json');
  const candidateManifestRaw = await readFile(candidateManifestPath);
  if (sha(candidateManifestRaw) !== NLLB_CANDIDATE.manifest) throw new Error('nllb_candidate_manifest_hash_mismatch');
  const candidateManifest = JSON.parse(candidateManifestRaw.toString('utf8'));
  if (candidateManifest.source?.archive_sha256 !== NLLB_CANDIDATE.archive || candidateManifest.source?.license !== NLLB_CANDIDATE.license)
    throw new Error('nllb_candidate_source_pin_mismatch');
  const sourceIrPath = resolve(candidateDir, 'translation-source-ir.jsonl');
  const refsPath = resolve(candidateDir, 'translation-references.host-only.jsonl');
  if (await fileDigest(sourceIrPath) !== NLLB_CANDIDATE.ir || await fileDigest(refsPath) !== NLLB_CANDIDATE.references)
    throw new Error('nllb_candidate_artifact_hash_mismatch');
  const bindingPath = resolve(root, policy.bindings.path);
  if (!bindingPath.startsWith(`${root}/`) || await fileDigest(bindingPath) !== policy.bindings.sha256)
    throw new Error('nllb_source_bindings_hash_mismatch');
  const wanted = new Map(), sourceIds = new Set();
  for await (const pair of jsonlRows(bindingPath)) {
    if (pair.schema !== 'natlang.nllb_seed_reference_binding/1' || !pair.source_ir || !pair.reference)
      throw new Error('nllb_source_binding_invalid');
    const taskId = pair.reference.task_id;
    if (sourceIds.has(pair.source_ir.id) || wanted.has(taskId) || pair.source_ir.source_ids?.[0] !== taskId)
      throw new Error('nllb_source_binding_duplicate');
    if (pair.source_ir.generation?.derived_role !== 'train_support' || pair.source_ir.split !== 'train' ||
        pair.reference.role !== 'train_support' || pair.reference.split !== 'train')
      throw new Error('nllb_source_binding_not_support_only');
    sourceIds.add(pair.source_ir.id);
    wanted.set(taskId, pair);
  }
  if (wanted.size !== bundleManifest.cases || policy.bindings.rows !== wanted.size)
    throw new Error('nllb_source_binding_count_mismatch');
  const sourceRows = jsonlRows(sourceIrPath), referenceRows = jsonlRows(refsPath);
  const matched = new Set();
  try {
    for await (const sourceIr of sourceRows) {
      const ref = await referenceRows.next();
      if (ref.done) throw new Error('nllb_candidate_ref_count_mismatch');
      const binding = wanted.get(sourceIr.source_ids?.[0]);
      if (!binding) continue;
      if (!exact(sourceIr, binding.source_ir) || !exact(ref.value, binding.reference))
        throw new Error(`nllb_source_reference_join_mismatch:${sourceIr.id}`);
      matched.add(binding.reference.task_id);
    }
    if (!(await referenceRows.next()).done || matched.size !== wanted.size)
      throw new Error('nllb_source_reference_join_incomplete');
  } finally { await sourceRows.return(); await referenceRows.return(); }
  return { directAnswers: true, sourceRows: matched.size, bindingPath, bindings: wanted };
}

export function nllbReferenceVisible(program, row) {
  const expected = program?.semantics?.inputs ?? {};
  const beforeAnswer = row?.trajectory?.at(-1)?.context;
  if (!Array.isArray(beforeAnswer)) return false;
  const visible = beforeAnswer.filter(message => message?.role === 'user' || message?.role === 'tool')
    .map(message => String(message.content ?? '')).join('\n');
  return [expected.source_language, expected.source_text, expected.target_language].every(value =>
    typeof value === 'string' && value.length > 0 && visible.includes(value));
}
