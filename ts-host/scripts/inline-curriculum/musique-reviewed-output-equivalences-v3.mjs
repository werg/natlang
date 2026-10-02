/** Exact source-pinned equivalences reviewed from Qwen generation traces. */
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { sourceConversionDigest } from '../../dist/teacher/source-conversion.js';

const REGISTRY = 'musique-reviewed-output-equivalence/2026-10-02-v3';
const REGISTRY_SHA256 = 'f716d6f6914ef3972fe6a754b18bef221326a2e7b6b97bf7c4635ee61576820b';
const SUFFIX = ':reviewed-output-equivalence-v3';
const bytes = readFileSync(new URL('./musique-reviewed-output-equivalences-v3.json', import.meta.url));
const sha = value => createHash('sha256').update(value).digest('hex');
if (sha(bytes) !== REGISTRY_SHA256) throw new Error('musique_equivalence_v3_registry_hash_mismatch');
const registry = JSON.parse(bytes);
if (registry.version !== REGISTRY || registry.entries.length !== 5 || registry.pending_review.length !== 1 ||
    registry.entries.some(entry => sha(entry.accepted) !== entry.accepted_answer_sha256 ||
      sha(entry.captured_teacher_output) !== entry.captured_teacher_output_sha256))
  throw new Error('musique_equivalence_v3_registry_shape_mismatch');
const entries = new Map(registry.entries.map(entry => [entry.source_id, entry]));
const canonicalDigest = value => sourceConversionDigest(value);
const same = (left, right) => canonicalDigest(left) === canonicalDigest(right);

function verifyBase(record, entry) {
  const hashes = record?.semantics?.folder_files && Object.fromEntries(Object.entries(record.semantics.folder_files)
    .sort(([a], [b]) => a.localeCompare(b)).map(([name, text]) => [name, typeof text === 'string' ? sha(text) : null]));
  return record?.id === entry.base_ir_id &&
    canonicalDigest(record) === entry.base_ir_sha256 &&
    record.source === 'musique' &&
    same(record.source_ids, entry.source_pin.source_ids) &&
    same(record.source_groups, entry.source_pin.source_groups) &&
    same(record.source_revisions, entry.source_pin.source_revisions) &&
    record.external_source?.snapshot_sha256 === entry.source_pin.external_source.snapshot_sha256 &&
    record.semantics?.expected === entry.primary &&
    sha(record.semantics?.files?.[record.semantics?.root] ?? '') === entry.prompt_sha256 &&
    canonicalDigest(record.curriculum?.reference) === entry.reference_sha256 &&
    same(hashes, entry.folder_file_sha256) &&
    record.generation?.reviewed_output_equivalence_v3 === undefined;
}

function auditFor(entry) {
  return {
    registry: REGISTRY,
    registry_sha256: REGISTRY_SHA256,
    source_id: entry.source_id,
    base_ir_id: entry.base_ir_id,
    base_ir_sha256: entry.base_ir_sha256,
    source_revision: entry.source_pin.source_revisions,
    source_snapshot_sha256: entry.source_pin.external_source.snapshot_sha256,
    prompt_sha256: entry.prompt_sha256,
    reference_sha256: entry.reference_sha256,
    source_file_sha256: entry.folder_file_sha256,
    original_primary_gold: entry.primary,
    exact_accepted_answer_string: entry.accepted,
    accepted_answer_sha256: entry.accepted_answer_sha256,
    captured_teacher_output_sha256: entry.captured_teacher_output_sha256,
    captured_teacher_output_is_not_automatically_accepted: true,
    teacher_artifact_sha256: entry.teacher_artifact_sha256,
    exact_string_only: true,
    global_fuzzy_or_substring_matching: false,
    evidence: entry.evidence,
    review_conclusion: entry.review_conclusion,
  };
}

/** Adds one exact reviewed standalone answer form only to its fully pinned base source record. */
export function applyReviewedMusiqueOutputEquivalenceV3(record) {
  if (record?.source !== 'musique') return record;
  const sourceId = record.external_source?.source_id ?? record.source_ids?.[0];
  const entry = entries.get(sourceId);
  if (!entry) return record;
  const suffixId = `${entry.base_ir_id}${SUFFIX}`;
  if (record.id === suffixId || record.generation?.reviewed_output_equivalence_v3 !== undefined) {
    const base = structuredClone(record);
    base.id = entry.base_ir_id;
    base.semantics.oracle = structuredClone(entry.base_oracle);
    delete base.generation.reviewed_output_equivalence_v3;
    if (!verifyBase(base, entry) || !same(record.generation.reviewed_output_equivalence_v3, auditFor(entry)))
      throw new Error(`musique_equivalence_v3_variant_drift:${sourceId}`);
    return structuredClone(record);
  }
  if (!verifyBase(record, entry)) throw new Error(`musique_equivalence_v3_source_proof_mismatch:${sourceId}`);
  const result = structuredClone(record);
  result.id = suffixId;
  result.semantics.oracle = { level: 'normalized', alternates: [...new Set([
    ...(entry.base_oracle.alternates ?? []), entry.accepted,
  ])] };
  result.generation.reviewed_output_equivalence_v3 = auditFor(entry);
  return result;
}
