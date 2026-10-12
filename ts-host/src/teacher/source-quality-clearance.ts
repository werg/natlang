/**
 * A narrowly scoped review receipt may clear a held source-quality annotation for conversion.
 * It never changes the source IR or grants training/trace admission.
 */
import { canonical } from '../native/canonical-data.js';
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
  /** When set to exact-selected-actions-only, conversion applies only to bound child actions. */
  target_scope?: 'program' | 'exact-selected-actions-only';
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
    selected_targets?: SelectedSourceQualityTarget[];
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

export type SelectedSourceQualityTarget = {
  candidate_id: string;
  item_id: string;
  item_sha256: string;
  program_source_row_sha256: string;
  trajectory_id: string;
  trajectory_sha256: string;
  decision_index: number;
  invocation_id: string;
  target_sha256: string;
};

export type SourceQualityActionIdentity = Pick<SelectedSourceQualityTarget,
  'trajectory_id' | 'trajectory_sha256' | 'decision_index' | 'invocation_id' | 'target_sha256'>;

/** The CLI creates this wrapper only after verifying all referenced bytes on disk. */
export type VerifiedSourceQualityClearance = {
  schema: 'natlang.verified-source-quality-clearance/1';
  review_sha256: string;
  review_canonical_sha256: string;
  evidence_files_verified: true;
  review: SourceQualityClearance;
};

export type SourceQualityClearanceRef = {
  schema: 'natlang.source-quality-clearance-ref/1';
  sidecar_path: string;
  file_sha256: string;
  review_canonical_sha256: string;
  review_id: string;
  program_id: string;
  program_ir_sha256: string;
  source_quality_sha256: string;
  source_snapshot_sha256: string;
  source_manifest_sha256: string;
  source_ids: string[];
  source_groups: string[];
  selected_target?: SelectedSourceQualityTarget;
  scope: 'source-conversion-only';
  training_admission: false;
  trace_admission: false;
};

/** Parse one packet sidecar after verifying its exact bytes and every signed evidence digest. */
export function verifySourceQualityClearanceSidecar(bytes: Uint8Array, expectedFileSha256: string,
  evidenceFilesBySha256: ReadonlyMap<string, Uint8Array>): VerifiedSourceQualityClearance {
  if (!isSha256(expectedFileSha256) || hexDigest(bytes) !== expectedFileSha256)
    throw new Error('source_quality_clearance_sidecar_file_hash_mismatch');
  let review: unknown;
  try { review = JSON.parse(new TextDecoder().decode(bytes)); }
  catch { throw new Error('source_quality_clearance_sidecar_invalid_json'); }
  const reviewProblems = sourceQualityClearanceProblems(review);
  if (reviewProblems.length) throw new Error(`source_quality_clearance_sidecar_invalid:${reviewProblems.join(',')}`);
  if (!isRecord(review) || !Array.isArray(review.evidence))
    throw new Error('source_quality_clearance_sidecar_evidence_invalid');
  for (const evidence of review.evidence) {
    if (!isRecord(evidence) || !isSha256(evidence.sha256))
      throw new Error('source_quality_clearance_sidecar_evidence_invalid');
    const evidenceBytes = evidenceFilesBySha256.get(evidence.sha256);
    if (!evidenceBytes || hexDigest(evidenceBytes) !== evidence.sha256)
      throw new Error(`source_quality_clearance_evidence_hash_mismatch:${evidence.sha256}`);
  }
  const verified: VerifiedSourceQualityClearance = { schema: 'natlang.verified-source-quality-clearance/1',
    review_sha256: hexDigest(bytes), review_canonical_sha256: hash(review), evidence_files_verified: true,
    review: review as SourceQualityClearance };
  const problems = sourceQualityClearanceProblems(verified);
  if (problems.length) throw new Error(`source_quality_clearance_sidecar_invalid:${problems.join(',')}`);
  return verified;
}

const hash = (value: unknown): string => hexDigest(canonical(value));
const isSha256 = (value: unknown): value is string => typeof value === 'string' && /^[a-f0-9]{64}$/.test(value);
const isRecord = (value: unknown): value is Dict => !!value && typeof value === 'object' && !Array.isArray(value);
const stringArray = (value: unknown): value is string[] => Array.isArray(value) && value.every(item => typeof item === 'string');

function isSelectedTarget(value: unknown): value is SelectedSourceQualityTarget {
  return isRecord(value) && typeof value.candidate_id === 'string' && !!value.candidate_id.trim() &&
    typeof value.item_id === 'string' && !!value.item_id.trim() && isSha256(value.item_sha256) &&
    isSha256(value.program_source_row_sha256) &&
    typeof value.trajectory_id === 'string' && !!value.trajectory_id.trim() && isSha256(value.trajectory_sha256) &&
    Number.isSafeInteger(value.decision_index) && Number(value.decision_index) >= 0 &&
    typeof value.invocation_id === 'string' && !!value.invocation_id.trim() && isSha256(value.target_sha256);
}

function isActionIdentity(value: unknown): value is SourceQualityActionIdentity {
  return isRecord(value) && typeof value.trajectory_id === 'string' && !!value.trajectory_id.trim() &&
    isSha256(value.trajectory_sha256) && Number.isSafeInteger(value.decision_index) && Number(value.decision_index) >= 0 &&
    typeof value.invocation_id === 'string' && !!value.invocation_id.trim() && isSha256(value.target_sha256);
}

function selectedTargetMatches(target: SelectedSourceQualityTarget, action: SourceQualityActionIdentity): boolean {
  return target.trajectory_id === action.trajectory_id && target.trajectory_sha256 === action.trajectory_sha256 &&
    target.decision_index === action.decision_index && target.invocation_id === action.invocation_id &&
    target.target_sha256 === action.target_sha256;
}

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
  if (review.target_scope !== undefined && review.target_scope !== 'program' &&
      review.target_scope !== 'exact-selected-actions-only')
    problems.push('source_quality_clearance_target_scope_invalid');
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
      if (review.target_scope === 'exact-selected-actions-only') {
        if (!Array.isArray(entry.selected_targets) || !entry.selected_targets.length) {
          problems.push('source_quality_clearance_selected_targets_missing');
          continue;
        }
        const candidateIds = new Set<string>();
        const actionKeys = new Set<string>();
        for (const target of entry.selected_targets as unknown[]) {
          if (!isSelectedTarget(target) || !rowIds.includes(target.item_id) ||
              (entry.source_rows as Dict[]).find(row => row.id === target.item_id)?.row_sha256 !== target.program_source_row_sha256) {
            problems.push('source_quality_clearance_selected_target_invalid');
            continue;
          }
          const actionKey = canonical({ trajectory_id: target.trajectory_id,
            trajectory_sha256: target.trajectory_sha256, decision_index: target.decision_index,
            invocation_id: target.invocation_id, target_sha256: target.target_sha256 });
          if (candidateIds.has(target.candidate_id) || actionKeys.has(actionKey))
            problems.push('source_quality_clearance_selected_target_duplicate');
          candidateIds.add(target.candidate_id);
          actionKeys.add(actionKey);
        }
      }
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
export function sourceQualityClearanceForProgram(value: unknown, programValue: unknown,
  action?: SourceQualityActionIdentity): Dict | undefined {
  const binding = sourceQualityClearanceProgramBinding(value, programValue);
  if (!binding) return;
  const review = isRecord(value) && value.schema === 'natlang.verified-source-quality-clearance/1'
    ? value.review as Dict : undefined;
  if (review?.target_scope === 'exact-selected-actions-only') {
    if (!action || !isActionIdentity(action)) return;
    const entry = (review.programs as Dict[]).find(item => item.program_id === String(binding.program_id));
    const selectedTargets = Array.isArray(entry?.selected_targets) ? entry.selected_targets as SelectedSourceQualityTarget[] : [];
    const target = selectedTargets.find(item => selectedTargetMatches(item, action));
    return target ? { ...binding, selected_target: target } : undefined;
  }
  return binding;
}

/** Internal identity check used to construct an exact action reference, never an admission by itself. */
function sourceQualityClearanceProgramBinding(value: unknown, programValue: unknown): Dict | undefined {
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

/** Build a small packet-relative reference while retaining the signed clearance in one sidecar. */
export function sourceQualityClearanceRefForProgram(value: VerifiedSourceQualityClearance, programValue: unknown,
  sidecarPath: string, fileSha256: string, action?: SourceQualityActionIdentity): SourceQualityClearanceRef | undefined {
  const binding = sourceQualityClearanceProgramBinding(value, programValue);
  if (!binding || !isSha256(fileSha256) || fileSha256 !== value.review_sha256 ||
      !isSha256(value.review_canonical_sha256) || value.review_canonical_sha256 !== hash(value.review) ||
      typeof sidecarPath !== 'string' || !sidecarPath.trim() || sidecarPath.startsWith('/') ||
      sidecarPath.includes('\\') || sidecarPath.split('/').some(part => !part || part === '.' || part === '..')) return;
  const scoped = value.review.target_scope === 'exact-selected-actions-only';
  let selectedTarget: SelectedSourceQualityTarget | undefined;
  if (scoped) {
    if (!action || !isActionIdentity(action)) return;
    const entry = value.review.programs.find(item => item.program_id === String(binding.program_id));
    selectedTarget = entry?.selected_targets?.find(target => selectedTargetMatches(target, action));
    if (!selectedTarget) return;
  }
  return { schema: 'natlang.source-quality-clearance-ref/1', sidecar_path: sidecarPath,
    file_sha256: value.review_sha256, review_canonical_sha256: value.review_canonical_sha256,
    review_id: value.review.id, program_id: String(binding.program_id),
    program_ir_sha256: String(binding.program_ir_sha256), source_quality_sha256: String(binding.source_quality_sha256),
    source_snapshot_sha256: String(binding.source_snapshot_sha256), source_manifest_sha256: String(binding.source_manifest_sha256),
    source_ids: [...(binding.source_ids as string[])], source_groups: [...(binding.source_groups as string[])],
    ...(scoped ? { selected_target: structuredClone(selectedTarget) } : {}),
    scope: 'source-conversion-only', training_admission: false, trace_admission: false };
}

/** A compact reference is trusted only alongside the verified sidecar and the exact pinned program. */
export function sourceQualityClearanceRefProblems(value: unknown, clearance: VerifiedSourceQualityClearance | undefined,
  programValue: unknown, actualTarget?: SourceQualityActionIdentity): string[] {
  if (!isRecord(value) || value.schema !== 'natlang.source-quality-clearance-ref/1')
    return ['source_quality_clearance_reference_invalid'];
  const ref = value as unknown as SourceQualityClearanceRef;
  if (typeof ref.sidecar_path !== 'string' || !ref.sidecar_path.trim() || ref.sidecar_path.startsWith('/') ||
      ref.sidecar_path.includes('\\') || ref.sidecar_path.split('/').some(part => !part || part === '.' || part === '..') ||
      !isSha256(ref.file_sha256) || !isSha256(ref.review_canonical_sha256) || typeof ref.review_id !== 'string' ||
      !ref.review_id.trim() || typeof ref.program_id !== 'string' || !ref.program_id.trim() ||
      !isSha256(ref.program_ir_sha256) || !isSha256(ref.source_quality_sha256) ||
      !isSha256(ref.source_snapshot_sha256) || !isSha256(ref.source_manifest_sha256) ||
      !stringArray(ref.source_ids) || !stringArray(ref.source_groups) ||
      ref.scope !== 'source-conversion-only' || ref.training_admission !== false || ref.trace_admission !== false)
    return ['source_quality_clearance_reference_invalid'];
  if (!clearance || sourceQualityClearanceProblems(clearance).length)
    return ['source_quality_clearance_sidecar_unresolved'];
  const scoped = clearance.review.target_scope === 'exact-selected-actions-only';
  if (scoped && (!isActionIdentity(actualTarget) || !isSelectedTarget(ref.selected_target) ||
      !selectedTargetMatches(ref.selected_target, actualTarget)))
    return ['source_quality_clearance_selected_target_mismatch'];
  if (!scoped && ref.selected_target !== undefined)
    return ['source_quality_clearance_selected_target_unexpected'];
  const expected = sourceQualityClearanceRefForProgram(clearance, programValue, ref.sidecar_path,
    clearance.review_sha256, scoped ? actualTarget : undefined);
  if (!expected || canonical(ref) !== canonical(expected)) return ['source_quality_clearance_reference_mismatch'];
  return [];
}
