import { hasExistingTreeValueContract } from './tree-contract.js';
/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';
import { sourceReviewReason } from './source-review.js';
import { tatqaAnswerRecordCanonical, tatqaAnswerRecordsEqual } from '../evaluation/oracles.js';

// Untyped nl results now run open, so inline_type_repair no longer triggers its required compiler refusal.
export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(['inline_type_repair']);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}

/** Source counterfactuals whose labels cannot be established by deleting an annotated proof leaf. */
export function quarantineReason(record: ProgramRecord): string | undefined {
  if (record.source === 'treedst' && !hasExistingTreeValueContract(record.semantics?.inputs?.state, record.semantics?.expected))
    return 'unverified_tree_transition_contract';
  const sourceReview = sourceReviewReason(record);
  if (sourceReview) return sourceReview;
  if (record.family === 'cb_highlighter' &&
    (record.generation as { highlighter_quality_version?: number } | undefined)?.highlighter_quality_version !== 2)
    return 'legacy_highlighter_oracle';
  const curriculum = record.curriculum as { family?: string; family_version?: number; variant?: string; answer_evidence?: string[];
    payment_scope_version?: number } | undefined;
  if (curriculum?.family === 'commaqa_numeric' && (curriculum.family_version ?? 1) < 3)
    return 'legacy_numeric_reference_contract';
  if (curriculum?.family === 'commaqa_question' && curriculum.family_version === 3)
    return 'unverified_movie_schema_contract';
  if (curriculum?.family === 'entailment_premises' && curriculum.variant === 'premise_removed')
    return 'unverified_counterfactual';
  if (curriculum?.family === 'folder_extract' && (!record.semantics.files_oracle?.quote_sources ||
      !record.semantics.files_oracle.return_count)) return 'legacy_extraction_contract';
  if (curriculum?.family === 'folder_edit' && (!record.semantics.files_oracle?.rubric ||
      !record.semantics.files_oracle.return_count)) return 'legacy_rewrite_contract';
  if (curriculum?.family === 'folder_index' && (!record.semantics.files_oracle?.return_count ||
      record.semantics.files_oracle.total === undefined)) return 'legacy_counts_contract';
  if (curriculum?.family === 'folder_triage' && record.semantics.files_oracle?.compare !== 'moves')
    return 'legacy_move_contract';
  if (curriculum?.family === 'folder_find' && !curriculum.answer_evidence?.length)
    return 'legacy_article_evidence_policy';
  if (curriculum?.family === 'folder_mixed' && record.dataset === 'banking77' && curriculum.payment_scope_version !== 2)
    return 'legacy_payment_scope';
}


/** Unfair answer-span contracts must not consume live generation requests. Verified static reads remain useful. */
export function generationHoldReason(record: ProgramRecord): string | undefined {
  return record.source === 'qasper' ? 'awaiting_extractive_equivalence_oracle' : undefined;
}

/** Old infrastructure failures must not teach models that correct actions are bad decisions. */
export function runtimeFailureReason(row: { task: Record<string, unknown>; provenance?: Record<string, unknown>;
  outcome?: Record<string, unknown>; trajectory?: unknown[] }): string | undefined {
  if (row.outcome?.accepted !== false) return;
  const record = row.task.program_ir as ProgramRecord;
  if (record.source === 'treedst' && (typeof record.semantics.oracle !== 'object' ||
      record.semantics.oracle.normalization !== 'named-tree')) return 'obsolete_named_tree_oracle';
  if (record.source === 'tatqa' && (typeof record.semantics.oracle !== 'object' ||
      !['json-string-record', 'tatqa-answer-record'].includes(record.semantics.oracle.normalization ?? ''))) return 'obsolete_json_format_oracle';
  if (record.source === 'tatqa' && typeof record.semantics.oracle === 'object' &&
      record.semantics.oracle.normalization === 'json-string-record' &&
      JSON.parse(tatqaAnswerRecordCanonical(record.semantics.expected) ?? '{}').answer?.numeric === true &&
      tatqaAnswerRecordsEqual(row.outcome?.value, record.semantics.expected) &&
      tatqaAnswerRecordsEqual((row.outcome?.files as Record<string, string> | undefined)?.['answer.json'], row.outcome?.value))
    return 'legacy_tatqa_numeric_display_oracle';
  // Extractive annotations do not enumerate every semantically equivalent span boundary.
  // Preserve wrong-answer traces for review without teaching valid paraphrases as negatives.
  if ((record.source === 'qasper' || record.source === 'musique') &&
      (row.outcome?.rejection_reasons as string[] | undefined)?.includes('answer'))
    return 'unreviewed_extractive_answer_equivalence';
  if (Number(row.provenance?.runtime_contract_version ?? 0) >= 17) return;
  const family = (record.curriculum as { family?: string } | undefined)?.family ?? record.family;
  if (family === 'inline_late_binding' || (record.family === 'cb_reconciliation' &&
      JSON.stringify(record.semantics.expected).includes('__proto__')) ||
      (family === 'logic_proof_verifier' && JSON.stringify(row.trajectory).includes('bad character at')))
    return 'obsolete_runtime_contract';
}


/** Reviewed episode holds are narrower than source holds: the task can still produce good trajectories. */
export function trajectoryReviewReason(row: { provenance?: Record<string, unknown> }): string | undefined {
  // Game33 recovered from placing the wrong object on the target mantle. Keep the original trace,
  // but do not approve its intermediate decisions until independently curated.
  const original = row.provenance?.reused_from as { provenance?: Record<string, unknown> } | undefined;
  if ([row.provenance?.trace_sha256, original?.provenance?.trace_sha256]
    .includes('63975b4cef1a6e5cf1d1d1ee02928875cd136617c97670857f2a231bbb125b75'))
    return 'trajectory_review_pending';
}


/** Evaluation tolerances are useful for scoring; partial correctness is not a positive training target. */
export function trainingQualityReason(row: { provenance?: Record<string, unknown>; outcome?: Record<string, unknown> }): string | undefined {
  const reviewed = trajectoryReviewReason(row);
  if (reviewed) return reviewed;
  const outcome = row.outcome ?? {};
  if (Array.isArray(outcome.quality_pending) && outcome.quality_pending.length) return 'quality_pending';
  const oracle = outcome.oracle as { level?: string; score?: number; needs_review?: boolean } | undefined;
  const files = outcome.files_check as { failed?: unknown[]; pending?: unknown[]; errors?: unknown[]; score?: number } | undefined;
  if (oracle?.needs_review || files?.pending?.length || files?.errors?.length) return 'quality_pending';
  if (outcome.accepted === true) {
    if (['agreement', 'span'].includes(oracle?.level ?? '') && (oracle?.score ?? 1) < 1)
      return 'quality_pending_partial_agreement';
    if (files?.failed?.length || (files?.score ?? 1) < 1) return 'quality_pending_partial_files';
  }
}
