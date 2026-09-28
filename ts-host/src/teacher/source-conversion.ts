/** Offline source replay evidence is distinct from the original agent's task-success label. */
import { hexDigest } from '../native/hash.js';
type Dict = Record<string, unknown>;
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Dict).sort(([a], [b]) =>
    a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value) ?? 'null';
}
export const sourceConversionDigest = (value: unknown): string => hexDigest(canonical(value));

export function sourceConversionProblems(row: { task?: Dict; provenance?: Dict; outcome?: Dict; trajectory?: unknown[] }): string[] {
  const provenance = row.provenance ?? {}, evidence = provenance.source_conversion as Dict | undefined;
  if (!evidence && provenance.collection_role !== 'external_replay') return [];
  if (!evidence) return ['missing_source_conversion'];
  const task = row.task?.program_ir as Dict | undefined;
  const source = task?.external_source as Dict | undefined;
  const reasons: string[] = [];
  if (evidence.version !== 'natlang.source_static_conversion/1') reasons.push('unsupported_source_conversion');
  if (!task || task.split !== 'train' || !source || ['test', 'validation', 'dev'].includes(String(source.original_split)))
    reasons.push('held_out_source_conversion');
  if (!task || evidence.program_ir_sha256 !== sourceConversionDigest(task) ||
      evidence.native_outcome_sha256 !== sourceConversionDigest(row.outcome) ||
      evidence.native_trajectory_sha256 !== sourceConversionDigest(row.trajectory)) reasons.push('source_conversion_digest_mismatch');
  if (!evidence.native_replay_accepted || !row.outcome?.accepted) reasons.push('source_conversion_not_validated');
  if (source?.snapshot_sha256 !== evidence.source_snapshot_sha256) reasons.push('source_conversion_snapshot_mismatch');
  if (provenance.collection_role === 'external_replay' &&
      (evidence.source_success !== true || !['independent_editor_slice', 'independent_file_creation'].includes(String(evidence.conversion_scope)) ||
       evidence.whole_issue_replayed !== false)) reasons.push('invalid_external_replay_scope');
  return reasons;
}
