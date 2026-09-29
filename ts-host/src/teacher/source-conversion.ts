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

/** Explicit user release of these pinned retired evaluations, never a general test-split bypass. */
export function retiredWorkflowEvaluationReleased(program: Dict): boolean {
  const source = program.external_source as Dict | undefined;
  const release = source?.retired_evaluation_release as Dict | undefined;
  const revisions: Record<string,string> = {
    'typesafe/evalsafe-invoice-processing':'6beeb2d2acd65c086c835022f5f4d7434114cafc',
    'typesafe/evalsafe-customer-service':'b1342f5a704587dbc465867c38c2694348ff86e4',
    'typesafe/evalsafe-security-incidents':'fbe1ea5c69cf494157fd23f2002a0d9d9a418443',
    'typesafe/evalsafe-agent-trace-observability':'8635540973910a92465fe2bc53e195375aa6e1a8',
  };
  return program.split === 'train' && source?.original_split === 'test' &&
    (program.generation as Dict | undefined)?.generator === 'natlang.workflowevals_adapter/1' &&
    typeof source.repository === 'string' && !!revisions[source.repository] &&
    source.revision === revisions[source.repository] && program.source === `workflowevals:${source.repository.replace('typesafe/evalsafe-','')}` &&
    Array.isArray(program.source_revisions) && program.source_revisions.length === 1 && program.source_revisions[0] === source.revision &&
    program.license === 'Apache-2.0' && source.license === 'Apache-2.0' &&
    release?.status === 'retired_evaluation_released_for_training' && release.authorization === 'user_request_2026-09-29' &&
    release.scope === 'these four pinned revisions only' && release.original_split === 'test' && release.training_split === 'train';
}

export function sourceConversionProblems(row: { task?: Dict; provenance?: Dict; outcome?: Dict; trajectory?: unknown[] }): string[] {
  const provenance = row.provenance ?? {}, evidence = provenance.source_conversion as Dict | undefined;
  const program = row.task?.program_ir as Dict | undefined;
  const quality = (program?.external_source as Dict | undefined)?.quality as Dict | undefined;
  const generator = (program?.generation as Dict | undefined)?.generator;
  const qualityRequired = ['natlang.recovered_source_adapter/1','natlang.workflowevals_adapter/1'].includes(String(generator));
  const qualityProblems = (!quality && qualityRequired) || quality &&
    (quality.version !== 'natlang.source_quality/1' || quality.status !== 'eligible' ||
     !Array.isArray(quality.checks) || !quality.checks.length || quality.checks.some(item => typeof item !== 'string' || !item)) ? ['source_quality_held'] : [];
  if (qualityRequired && (program?.split !== 'train' || (program?.external_source as Dict | undefined)?.original_split !== 'train') && (!program || !retiredWorkflowEvaluationReleased(program)))
    qualityProblems.push('held_out_source_quality');
  if (!evidence && provenance.collection_role !== 'external_replay') return qualityProblems;
  if (!evidence) return ['missing_source_conversion'];
  const task = row.task?.program_ir as Dict | undefined;
  const source = task?.external_source as Dict | undefined;
  const reasons: string[] = [...qualityProblems];
  if (evidence.version !== 'natlang.source_static_conversion/1') reasons.push('unsupported_source_conversion');
  if (!task || task.split !== 'train' || !source || (['test', 'validation', 'dev'].includes(String(source.original_split)) && !retiredWorkflowEvaluationReleased(task)))
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
