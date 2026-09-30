/**
 * Fail-closed registry for ten source-pinned output equivalences reviewed from
 * the 2026-09-30 MuSiQue revision-4 trajectories. This module is deliberately
 * applied by the source builder only after full source-group assembly. It creates a reviewed IR
 * when the exact program, prompt, gold, evidence files, groups, and reference
 * match the pinned registry and the raw teacher string matches byte-for-byte.
 */
import { createHash } from 'node:crypto';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const DATA_PATH = path.join(HERE, 'musique-reviewed-output-equivalences-v1.json');
const REGISTRY_BYTES_SHA256 = 'e4683434011df1a36954b5ac81ea5ff185d16eecddf72a8929d47840f19b0d8a';
export const REVIEWED_OUTPUT_EQUIVALENCE_REGISTRY = 'musique-reviewed-output-equivalence/2026-09-30-v1';
export const REVIEWED_OUTPUT_EQUIVALENCE_SUFFIX = ':reviewed-output-equivalence-v1';

const sha = value => createHash('sha256').update(value).digest('hex');
const canonical = value => Array.isArray(value) ? `[${value.map(canonical).join(',')}]` :
  value && typeof value === 'object' ? `{${Object.entries(value).sort(([a], [b]) => a.localeCompare(b))
    .map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}` : JSON.stringify(value);
const digest = value => sha(canonical(value));
const same = (left, right) => canonical(left) === canonical(right);
const registryBytes = fs.readFileSync(DATA_PATH);
if (sha(registryBytes) !== REGISTRY_BYTES_SHA256) throw new Error('reviewed_output_equivalence_registry_data_hash_mismatch');
const registry = JSON.parse(registryBytes.toString('utf8'));
if (registry.version !== REVIEWED_OUTPUT_EQUIVALENCE_REGISTRY || registry.entries.length !== 10)
  throw new Error('reviewed_output_equivalence_registry_shape_mismatch');
const bySourceId = new Map(registry.entries.map(entry => [entry.source_id, entry]));

function identityFor(record, entry) {
  if (!record || typeof record !== 'object' || !entry) return null;
  const prompt = record.semantics?.files?.[record.semantics?.root];
  const placement = record.id === entry.canonical_base_ir_id ? 'canonical-original' :
    record.id === entry.title_base_ir_id ? 'reviewed-title-variant' : null;
  if (!placement) return null;
  const expectedDigest = placement === 'canonical-original' ? entry.canonical_base_ir_sha256 : entry.title_base_ir_sha256;
  const expectedRef = placement === 'canonical-original' ? entry.canonical_reference_sha256 : entry.title_reference_sha256;
  const expectedFiles = placement === 'canonical-original' ? entry.canonical_folder_file_sha256 : entry.title_folder_file_sha256;
  const expectedOracle = placement === 'canonical-original' ? entry.canonical_base_oracle : entry.title_base_oracle;
  const fileHashes = record.semantics?.folder_files && Object.fromEntries(Object.entries(record.semantics.folder_files)
    .map(([name, text]) => [name, typeof text === 'string' ? sha(text) : null]).sort(([a], [b]) => a.localeCompare(b)));
  const expectedSourceIds = [entry.source_id];
  const refDigest = digest(record.curriculum?.reference);
  const sourceId = record.external_source?.source_id ?? record.source_ids?.[0];
  if (digest(record) !== expectedDigest || sourceId !== entry.source_id || record.source !== 'musique' ||
      !same(record.source_ids, expectedSourceIds) || !same(record.source_groups, entry.source_groups) ||
      !same(record.source_revisions, entry.source_revisions) || record.semantics?.expected !== entry.primary ||
      prompt !== entry.question_prompt || refDigest !== expectedRef || !same(fileHashes, expectedFiles) ||
      record.external_source?.snapshot_sha256 !== entry.snapshot_sha256 || !same(record.semantics?.oracle, expectedOracle) ||
      record.generation?.reviewed_answer_equivalence !== undefined) return null;
  return { placement, expectedDigest, expectedRef, expectedFiles };
}

function auditFor(entry, placement, recordDigest, baseId) {
  const variantId = `${baseId}${REVIEWED_OUTPUT_EQUIVALENCE_SUFFIX}`;
  return {
    registry: REVIEWED_OUTPUT_EQUIVALENCE_REGISTRY,
    review: 'root-approved-source-pinned-output-equivalence',
    approval_scope: 'exact source_id, primary gold, and registered teacher string only; artifact publication remains pending',
    source_id: entry.source_id,
    placement,
    canonical_base_ir_id: entry.canonical_base_ir_id,
    canonical_base_ir_sha256: entry.canonical_base_ir_sha256,
    title_base_ir_id: entry.title_base_ir_id,
    title_base_ir_sha256: entry.title_base_ir_sha256,
    source_row_sha256: entry.source_row_sha256,
    source_snapshot_sha256: entry.snapshot_sha256,
    question: entry.question,
    primary_gold: entry.primary,
    exact_accepted_teacher_string: entry.accepted,
    teacher_input_match: 'exact-string-only',
    oracle_comparison: 'existing-normalized-oracle-no-fuzzy-or-value-tolerance-added',
    source_groups: entry.source_groups,
    source_revisions: entry.source_revisions,
    reference_sha256: placement === 'canonical-original' ? entry.canonical_reference_sha256 : entry.title_reference_sha256,
    source_files_sha256: placement === 'canonical-original' ? entry.canonical_folder_file_sha256 : entry.title_folder_file_sha256,
    original_oracle: structuredClone(placement === 'canonical-original' ? entry.canonical_base_oracle : entry.title_base_oracle),
    candidate_oracle: { level: 'normalized', alternates: [
      ...(placement === 'canonical-original' ? entry.canonical_base_oracle : entry.title_base_oracle).alternates,
      entry.accepted,
    ] },
    base_record_sha256: recordDigest,
    base_ir_id: baseId,
    variant_ir_id: variantId,
    accepted_output_sha256: sha(entry.accepted),
    rationale: entry.rationale,
    disposition: entry.disposition,
    builder_integration: 'not-approved-not-wired',
  };
}

function validateReviewedVariant(record, entry) {
  const audit = record?.generation?.reviewed_answer_equivalence;
  if (!audit || audit.registry !== REVIEWED_OUTPUT_EQUIVALENCE_REGISTRY ||
      audit.source_id !== entry.source_id || audit.exact_accepted_teacher_string !== entry.accepted ||
      record.id !== `${audit.base_ir_id}${REVIEWED_OUTPUT_EQUIVALENCE_SUFFIX}` ||
      !same(record.semantics?.oracle, audit.candidate_oracle)) return false;
  const placement = audit.placement;
  const baseId = placement === 'canonical-original' ? entry.canonical_base_ir_id :
    placement === 'reviewed-title-variant' ? entry.title_base_ir_id : null;
  if (!baseId || audit.base_ir_id !== baseId) return false;
  const reverted = structuredClone(record);
  reverted.id = baseId;
  reverted.semantics.oracle = structuredClone(placement === 'canonical-original' ? entry.canonical_base_oracle : entry.title_base_oracle);
  delete reverted.generation.reviewed_answer_equivalence;
  const identity = identityFor(reverted, entry);
  if (!identity || !same(audit, auditFor(entry, placement, identity.expectedDigest, baseId))) return false;
  return true;
}

/**
 * Accepts only the exact raw teacher output and exact immutable original or
 * titled source record. No normalization, substring matching, aliases inferred
 * from the answer, or numeric coercion is performed.
 */
export function applyReviewedMusiqueOutputEquivalence(record, teacherAnswer) {
  const sourceId = record?.external_source?.source_id ?? record?.source_ids?.[0];
  const entry = bySourceId.get(sourceId);
  if (!entry) return null;
  if (teacherAnswer !== entry.accepted) throw new Error(`reviewed_output_equivalence_teacher_string_mismatch:${sourceId}`);
  if (record?.generation?.reviewed_answer_equivalence) {
    if (!validateReviewedVariant(record, entry)) throw new Error(`reviewed_output_equivalence_variant_drift:${sourceId}`);
    return structuredClone(record);
  }
  const identity = identityFor(record, entry);
  if (!identity) throw new Error(`reviewed_output_equivalence_source_proof_mismatch:${sourceId}`);
  const variant = structuredClone(record);
  const baseId = record.id;
  const review = auditFor(entry, identity.placement, identity.expectedDigest, baseId);
  variant.id = `${baseId}${REVIEWED_OUTPUT_EQUIVALENCE_SUFFIX}`;
  const priorOracle = identity.placement === 'canonical-original' ? entry.canonical_base_oracle : entry.title_base_oracle;
  variant.semantics.oracle = { level: 'normalized', alternates: [...priorOracle.alternates, entry.accepted] };
  variant.generation.reviewed_answer_equivalence = review;
  if (!validateReviewedVariant(variant, entry)) throw new Error(`reviewed_output_equivalence_internal_validation_failed:${sourceId}`);
  return variant;
}

/** Returns an immutable public view for review tools; callers cannot mutate registry state. */
export function reviewedOutputEquivalenceEntry(sourceId) {
  const entry = bySourceId.get(sourceId);
  return entry ? structuredClone(entry) : null;
}
