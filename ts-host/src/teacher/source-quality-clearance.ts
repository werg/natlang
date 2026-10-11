/**
 * A narrowly scoped review receipt may clear a held source-quality annotation for conversion.
 * It never changes the source IR or grants training/trace admission.
 */
import { canonical } from '../adaptation/identity.js';
import { hexDigest } from '../native/hash.js';

type Dict = Record<string, unknown>;

export type SourceQualityClearance = {
  schema: 'natlang.source-quality-clearance/1';
  id: string;
  scope: 'source-conversion-only';
  decision: 'clear-for-conversion';
  training_admission: false;
  trace_admission: false;
  evidence: { role: string; path: string; sha256: string }[];
  source_policy: { path: string; sha256: string; held_source_ids: string[]; held_source_groups: string[] };
  programs: {
    program_id: string;
    program_ir_sha256: string;
    source_quality_sha256: string;
    source_path: string;
    source_snapshot_sha256: string;
    source_manifest_path: string;
    source_manifest_sha256: string;
    source_ids: string[];
    source_groups: string[];
    source_rows: { id: string; row_index: number; row_sha256: string; group: string; split: string }[];
  }[];
  root_approval: {
    schema: 'natlang.source-quality-clearance-approval/1';
    decision: 'approved';
    authority: 'root';
    scope: 'source-conversion-only';
    payload_sha256: string;
    training_admission: false;
    trace_admission: false;
  };
};

/** The CLI creates this wrapper only after verifying all referenced bytes on disk. */
export type VerifiedSourceQualityClearance = {
  schema: 'natlang.verified-source-quality-clearance/1';
  review_sha256: string;
  review_canonical_sha256: string;
  evidence_files_verified: true;
  review: SourceQualityClearance;
};

const hash = (value: unknown): string => hexDigest(canonical(value));
const isSha256 = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const isRecord = (value: unknown): value is Dict => !!value && typeof value === 'object' && !Array.isArray(value);
const stringArray = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string');

function payloadWithoutApproval(review: Dict): Dict {
  const { root_approval: _approval, ...payload } = review;
  return payload;
}

/** Validate the receipt's own root approval binding and all program pins using only in-memory JSON values. */
export function sourceQualityClearanceProblems(value: unknown): string[] {
  const wrapper = isRecord(value) && value.schema === 'natlang.verified-source-quality-clearance/1' ? value : undefined;
  const review = wrapper ? wrapper.review : value;
  if (!isRecord(review)) return ['source_quality_clearance_missing'];
  const problems: string[] = [];
  if (review.schema !== 'natlang.source-quality-clearance/1' || typeof review.id !== 'string' || !review.id.trim() ||
      review.scope !== 'source-conversion-only' || review.decision !== 'clear-for-conversion' ||
      review.training_admission !== false || review.trace_admission !== false)
    problems.push('source_quality_clearance_schema_or_scope_invalid');
  if (!Array.isArray(review.evidence) || !review.evidence.length || review.evidence.some(item => !isRecord(item) ||
      typeof item.role !== 'string' || !item.role.trim() || typeof item.path !== 'string' || !item.path.trim() ||
      !isSha256(item.sha256)))
    problems.push('source_quality_clearance_evidence_invalid');
  else {
    const roles = new Set((review.evidence as Dict[]).map(item => item.role));
    if (!roles.has('source_quality_review') || !roles.has('source_closure'))
      problems.push('source_quality_clearance_review_or_closure_evidence_missing');
  }
  if (!isRecord(review.source_policy) || typeof review.source_policy.path !== 'string' || !review.source_policy.path.trim() ||
      !isSha256(review.source_policy.sha256) || !stringArray(review.source_policy.held_source_ids) ||
      !stringArray(review.source_policy.held_source_groups) ||
      new Set(review.source_policy.held_source_ids).size !== review.source_policy.held_source_ids.length ||
      new Set(review.source_policy.held_source_groups).size !== review.source_policy.held_source_groups.length)
    problems.push('source_quality_clearance_source_policy_invalid');
  if (!Array.isArray(review.programs) || !review.programs.length) problems.push('source_quality_clearance_programs_missing');
  else {
    const ids = new Set<string>();
    for (const entry of review.programs) {
      if (!isRecord(entry) || typeof entry.program_id !== 'string' || !entry.program_id.trim() || ids.has(entry.program_id) ||
          !isSha256(entry.program_ir_sha256) || !isSha256(entry.source_quality_sha256) ||
          typeof entry.source_path !== 'string' || !entry.source_path.trim() || !isSha256(entry.source_snapshot_sha256) ||
          typeof entry.source_manifest_path !== 'string' || !entry.source_manifest_path.trim() ||
          !isSha256(entry.source_manifest_sha256) || !stringArray(entry.source_ids) || !stringArray(entry.source_groups) ||
          !Array.isArray(entry.source_rows) || !entry.source_rows.length || entry.source_rows.some(row => !isRecord(row) ||
            typeof row.id !== 'string' || !row.id || !Number.isSafeInteger(row.row_index) || Number(row.row_index) < 0 ||
            !isSha256(row.row_sha256) || typeof row.group !== 'string' || !row.group || row.split !== 'train')) {
        problems.push('source_quality_clearance_program_binding_invalid');
        continue;
      }
      ids.add(entry.program_id);
      const rowIds = (entry.source_rows as Dict[]).map(row => row.id);
      const rowGroups = [...new Set((entry.source_rows as Dict[]).map(row => row.group))];
      if (new Set(rowIds).size !== rowIds.length ||
          canonical(entry.source_ids) !== canonical(rowIds) || canonical(entry.source_groups) !== canonical(rowGroups))
        problems.push('source_quality_clearance_source_set_mismatch');
    }
  }
  const approval = review.root_approval;
  if (!isRecord(approval) || approval.schema !== 'natlang.source-quality-clearance-approval/1' ||
      approval.decision !== 'approved' || approval.authority !== 'root' || approval.scope !== 'source-conversion-only' ||
      approval.training_admission !== false || approval.trace_admission !== false || !isSha256(approval.payload_sha256) ||
      approval.payload_sha256 !== hash(payloadWithoutApproval(review)))
    problems.push('source_quality_clearance_root_approval_invalid');
  if (wrapper && (!isSha256(wrapper.review_sha256) || !isSha256(wrapper.review_canonical_sha256) ||
      wrapper.review_canonical_sha256 !== hash(review) || wrapper.evidence_files_verified !== true))
    problems.push('source_quality_clearance_file_verification_missing');
  return problems;
}

/** Return the exact review binding only when it matches this unchanged, train-split program IR. */
export function sourceQualityClearanceForProgram(value: unknown, programValue: unknown): Dict | undefined {
  if (!isRecord(programValue)) return;
  const wrapper = isRecord(value) && value.schema === 'natlang.verified-source-quality-clearance/1' ? value : undefined;
  if (!wrapper || sourceQualityClearanceProblems(wrapper).length) return;
  const review = wrapper.review as Dict;
  const external = programValue.external_source as Dict | undefined;
  const quality = external?.quality as Dict | undefined;
  if (!external || !quality || programValue.version !== 'natlang.program/2' || quality.version !== 'natlang.source_quality/1' || quality.status !== 'held' ||
      programValue.split !== 'train' || external.original_split !== 'train') return;
  const entry = (review.programs as Dict[]).find(item => item.program_id === programValue.id);
  if (!entry || entry.program_ir_sha256 !== hash(programValue) || entry.source_quality_sha256 !== hash(quality) ||
      entry.source_snapshot_sha256 !== external.snapshot_sha256 || entry.source_path !== external.source_path ||
      entry.source_manifest_sha256 !== external.source_manifest_sha256 ||
      !Array.isArray(external.source_rows) || canonical(entry.source_rows) !== canonical(external.source_rows) ||
      canonical(entry.source_ids) !== canonical(programValue.source_ids ?? []) ||
      canonical(entry.source_groups) !== canonical(programValue.source_groups ?? [])) return;
  const policy = review.source_policy as Dict;
  const sourceIds = entry.source_ids as string[], sourceGroups = entry.source_groups as string[];
  if (sourceIds.some(id => (policy.held_source_ids as string[]).includes(id)) ||
      sourceGroups.some(group => (policy.held_source_groups as string[]).includes(group))) return;
  return { schema: 'natlang.source-quality-clearance-binding/1', review_id: review.id,
    review_sha256: wrapper.review_sha256, review_canonical_sha256: wrapper.review_canonical_sha256, program_id: programValue.id,
    program_ir_sha256: entry.program_ir_sha256, source_quality_sha256: entry.source_quality_sha256,
    source_snapshot_sha256: entry.source_snapshot_sha256, source_manifest_sha256: entry.source_manifest_sha256,
    source_ids: entry.source_ids, source_groups: entry.source_groups,
    policy_held_source_ids: (policy.held_source_ids as string[]), policy_held_source_groups: (policy.held_source_groups as string[]),
    scope: 'source-conversion-only', training_admission: false, trace_admission: false };
}
