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

/** Reconstruct only tool output the scripted answer's context actually contains. */
export function workflowReferenceVisibility(row: { task?: Dict; trajectory?: unknown[] }): Dict | undefined {
  const program = row.task?.program_ir as Dict | undefined;
  const semantics = program?.semantics as Dict | undefined;
  const turns = (row.trajectory ?? []) as Dict[];
  const final = turns.at(-1), context = final?.context as Dict[] | undefined;
  if (!semantics || !context?.length) return;
  const finalCalls = (final?.assistant as Dict | undefined)?.calls as Dict[] | undefined;
  if (finalCalls?.[0]?.tool !== 'return_result' || (finalCalls[0].arguments as Dict | undefined)?.status !== 'success') return;
  const sourceFile = (semantics.files as Dict | undefined)?.[String(semantics.root)];
  const instructions = typeof sourceFile === 'string' ? sourceFile.split('\n---\n').at(-1)?.trim() : undefined;
  if (!instructions || !context.some(message => message.role === 'user' && String(message.content).includes(instructions))) return;
  const visible = new Map<string,string>();
  let active: string | undefined;
  for (let index = 0; index + 1 < turns.length; index++) {
    const calls = (turns[index]?.assistant as Dict | undefined)?.calls as Dict[] | undefined;
    if (calls?.length !== 1) continue;
    const call = calls[0]!, args = call.arguments as Dict | undefined;
    const messages = turns[index + 1]?.context as Dict[] | undefined;
    const output = messages?.at(-1);
    if (output?.role !== 'tool') continue;
    const text = String(output.content ?? '');
    if (call.tool === 'eval' && args?.code === 'console.log(JSON.stringify(state));') {
      active = 'state';
      if (!text.startsWith('console:\n')) return;
      visible.set(active, text.slice('console:\n'.length).split('\n<<cut off:')[0]!.split('\n')[0]!);
    } else if (call.tool === 'read_file' && typeof args?.path === 'string') {
      active = args.path;
      visible.set(active, text.split('\n<<cut off:')[0]!);
    } else if (call.tool === 'read_page' && active) {
      visible.set(active, (visible.get(active) ?? '') + text.replace(/\n<<page \d+ of \d+(?: shown;[^\n]*|, the last)>>[\s\S]*$/, ''));
    } else continue;
    // A prior page that was compacted away is not evidence for this scripted answer.
    if (!context.some(message => message.role === 'tool' && message.tool_call_id === output.tool_call_id &&
        String(message.content) === text)) return;
  }
  const expected = program?.task_modality === 'primitive' ?
    {state: JSON.stringify((semantics.inputs as Dict | undefined)?.state)} : semantics.folder_files as Dict | undefined;
  if (!expected || !Object.keys(expected).length) return;
  const reconstructed: Dict = {};
  for (const [name, value] of Object.entries(expected)) {
    const actual = visible.get(name);
    if (typeof value !== 'string' || actual?.trimEnd() !== value.trimEnd()) return;
    reconstructed[name] = actual.trimEnd();
  }
  return {version:'natlang.visible_source_inputs/1', complete_before_answer:true,
    inputs_sha256:sourceConversionDigest(Object.fromEntries(Object.entries(expected).map(([key,value]) => [key,String(value).trimEnd()]))),
    visible_inputs_sha256:sourceConversionDigest(reconstructed), files:Object.keys(expected).length,
    question_instructions_sha256:sourceConversionDigest(instructions)};
}

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
  // These native curriculum pools are reserved in references.mjs. Handoffs
  // retain their source groups even when their own split is mislabeled train.
  const reservedEvaluation = Array.isArray(program?.source_groups) &&
    program.source_groups.some(group => typeof group === 'string' && /^s(?:102|900):/.test(group));
  const qualityRequired = ['natlang.recovered_source_adapter/1','natlang.workflowevals_adapter/1'].includes(String(generator));
  const qualityProblems = (!quality && qualityRequired) || quality &&
    (quality.version !== 'natlang.source_quality/1' || quality.status !== 'eligible' ||
     !Array.isArray(quality.checks) || !quality.checks.length || quality.checks.some(item => typeof item !== 'string' || !item)) ? ['source_quality_held'] : [];
  const sourceIdentity = program?.external_source as Dict | undefined;
  if ((program?.generation as Dict | undefined)?.operation_derivative &&
      (typeof sourceIdentity?.source_id !== 'string' ||
       !Array.isArray(program?.source_ids) || !program.source_ids.includes(sourceIdentity.source_id) ||
       sourceIdentity.source_id === sourceIdentity.derived_from_source_id))
    qualityProblems.push('operation_derivative_source_identity_mismatch');
  if (reservedEvaluation) qualityProblems.push('held_out_reserved_curriculum');
  // Offline history replays are review artifacts until a migration admission
  // contract verifies both teacher provenance and the changed runtime context.
  if (provenance.history_migration || provenance.collection_role === 'migration_candidate')
    qualityProblems.push('history_migration_review_pending');
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
  if (task && retiredWorkflowEvaluationReleased(task) && provenance.synthetic_reasoning === 'action-notes/1') {
    const visibility = workflowReferenceVisibility(row);
    if (!visibility || !evidence.visible_source_inputs ||
        sourceConversionDigest(visibility) !== sourceConversionDigest(evidence.visible_source_inputs))
      reasons.push('source_input_visibility_unverified');
  }
  if (source?.snapshot_sha256 !== evidence.source_snapshot_sha256) reasons.push('source_conversion_snapshot_mismatch');
  if (provenance.collection_role === 'external_replay' &&
      (evidence.source_success !== true || !['independent_editor_slice', 'independent_file_creation'].includes(String(evidence.conversion_scope)) ||
       evidence.whole_issue_replayed !== false)) reasons.push('invalid_external_replay_scope');
  return reasons;
}
