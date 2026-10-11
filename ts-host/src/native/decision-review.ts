import { canonical } from '../adaptation/identity.js';
import { hexDigest } from './hash.js';

/** Explicit quality review of one exact authored target, independent of its parent's final grade. */
export type NativeDecisionApproval = {
  schema: 'natlang.native-decision-approval/1';
  trajectory_id: string;
  source_row_sha256: string;
  decision_index: number;
  target_sha256: string;
  review_sha256: string;
  reason: string;
  evidence: string[];
  source_label_correction?: {
    evidence_sha256: string;
    item_id: string;
    item_sha256: string;
    input_sha256: string;
    original_annotation: boolean;
    corrected_label: boolean;
    actual_generated_value: boolean;
  };
};

export function nativeDecisionTargetDigest(target: unknown): string {
  return hexDigest(canonical(JSON.parse(JSON.stringify(target))));
}

export function validNativeDecisionApproval(approval: unknown, expected: {
  trajectory_id: string; source_row_sha256: string; decision_index: number; target_sha256: string;
}): approval is NativeDecisionApproval {
  if (!approval || typeof approval !== 'object' || Array.isArray(approval)) return false;
  const value = approval as Record<string, unknown>;
  const correction = value.source_label_correction;
  const correctionValid = correction === undefined || (correction !== null && typeof correction === 'object' &&
    !Array.isArray(correction) && typeof (correction as Record<string, unknown>).evidence_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test((correction as Record<string, unknown>).evidence_sha256 as string) &&
    typeof (correction as Record<string, unknown>).item_id === 'string' &&
    typeof (correction as Record<string, unknown>).item_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test((correction as Record<string, unknown>).item_sha256 as string) &&
    typeof (correction as Record<string, unknown>).input_sha256 === 'string' &&
    /^[a-f0-9]{64}$/.test((correction as Record<string, unknown>).input_sha256 as string) &&
    typeof (correction as Record<string, unknown>).original_annotation === 'boolean' &&
    typeof (correction as Record<string, unknown>).corrected_label === 'boolean' &&
    typeof (correction as Record<string, unknown>).actual_generated_value === 'boolean' &&
    (correction as Record<string, unknown>).corrected_label ===
      (correction as Record<string, unknown>).actual_generated_value &&
    (correction as Record<string, unknown>).corrected_label !==
      (correction as Record<string, unknown>).original_annotation);
  return correctionValid && value.schema === 'natlang.native-decision-approval/1' &&
    value.trajectory_id === expected.trajectory_id && value.source_row_sha256 === expected.source_row_sha256 &&
    value.decision_index === expected.decision_index && value.target_sha256 === expected.target_sha256 &&
    [value.source_row_sha256, value.target_sha256, value.review_sha256].every(hash =>
      typeof hash === 'string' && /^[a-f0-9]{64}$/.test(hash)) &&
    Number.isSafeInteger(value.decision_index) && Number(value.decision_index) >= 0 &&
    typeof value.reason === 'string' && !!value.reason.trim() && Array.isArray(value.evidence) &&
    value.evidence.length > 0 && value.evidence.every(item => typeof item === 'string' && !!item.trim());
}
