/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';
import { sourceReviewReason } from './source-review.js';

// Untyped nl results now run open, so inline_type_repair no longer triggers its required compiler refusal.
export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(['inline_type_repair']);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}

/** Source counterfactuals whose labels cannot be established by deleting an annotated proof leaf. */
export function quarantineReason(record: ProgramRecord): string | undefined {
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


/** Old infrastructure failures must not teach models that correct actions are bad decisions. */
export function runtimeFailureReason(row: { task: Record<string, unknown>; provenance?: Record<string, unknown>;
  outcome?: Record<string, unknown>; trajectory?: unknown[] }): string | undefined {
  if (row.outcome?.accepted !== false || Number(row.provenance?.runtime_contract_version ?? 0) >= 17) return;
  const record = row.task.program_ir as ProgramRecord;
  const family = (record.curriculum as { family?: string } | undefined)?.family ?? record.family;
  if (family === 'inline_late_binding' || (record.family === 'cb_reconciliation' &&
      JSON.stringify(record.semantics.expected).includes('__proto__')) ||
      (family === 'logic_proof_verifier' && JSON.stringify(row.trajectory).includes('bad character at')))
    return 'obsolete_runtime_contract';
}
