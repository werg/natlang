/**
 * Stable, machine-readable interpretation of raw admission/compatibility reasons.
 * Keep `reason` intact: disposition is an index for review and routing, not a replacement label.
 */
const RULES = [
  { category: 'migration_or_replay_pending', action: 'review_migration_provenance_and_runtime_context_before_promotion', matches: r =>
    r === 'history_migration_review_pending' },
  { category: 'duplicate_or_superseded', action: 'retain_original_and_link_selected_replacement', matches: r =>
    /^(?:superseded_eligible_trajectory_id|duplicate_input_trajectory_id_not_written)$/.test(r) },
  { category: 'evaluation_or_unsupported', action: 'preserve_evaluation_scope_do_not_migrate_into_training', matches: r =>
    /^(?:held_out_source_conversion|held_out_source_quality|held_out_reserved_curriculum|not_training_split)$/.test(r) },
  { category: 'migration_or_replay_pending', action: 'replay_with_current_runtime_or_refresh_evidence', matches: r =>
    /^(?:obsolete_(?:outcome|program_ir|runtime_contract|named_tree_oracle|json_format_oracle)|requires_non_curriculum_adapter|unsupported_source_conversion|source_conversion_(?:digest_mismatch|not_validated|snapshot_mismatch)|source_input_visibility_unverified|incomplete_trajectory|the replay saw other results than the run did|replay failed)$/.test(r) },
  { category: 'oracle_or_source_review', action: 'retain_raw_evidence_for_source_or_oracle_review', matches: r =>
    /(?:source_review|unverified_|legacy_|extractive_answer_equivalence|quality_pending|trajectory_review_pending|unreviewed_.*oracle|missing_source_conversion|source_quality_held|invalid_external_replay_scope)/.test(r) },
  { category: 'evaluation_or_unsupported', action: 'exclude_from_training_and_review_scope', matches: r =>
    /^(?:retired_family|not_explicit_teacher_collection|missing_trajectory_id|unsupported_.*)$/.test(r) },
  { category: 'candidate_failure', action: 'retain_as_failure_candidate_only_after_in_place_replay', matches: r =>
    /^(?:wrong_return|fabricated_result|missing_observation|premature_choice|missing_answer_evidence|defect_not_repaired|unwarranted_edit|missing_observation:.*|premature_choice:.*)$/.test(r) },
];

export function classifyAdmissionReason(rawReason) {
  const reason = String(rawReason ?? 'unknown_reason');
  const base = reason.split(':', 1)[0];
  const rule = RULES.find(candidate => candidate.matches(reason) || candidate.matches(base));
  return { reason, category: rule?.category ?? 'unclassified_review_pending',
    next_action: rule?.action ?? 'retain_raw_evidence_and_review_before_training' };
}

export function classifyAdmissionReasons(reasons = []) {
  return [...new Set(reasons.map(String))].map(classifyAdmissionReason);
}

/** Compatibility, quality and source-review holds must never be DPO negatives. */
export function dpoHoldReasons(reasons = []) {
  return classifyAdmissionReasons(reasons).filter(item =>
    ['migration_or_replay_pending', 'oracle_or_source_review', 'evaluation_or_unsupported',
     'duplicate_or_superseded', 'unclassified_review_pending'].includes(item.category));
}
