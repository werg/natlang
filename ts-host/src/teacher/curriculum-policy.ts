/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';

// Untyped nl results now run open, so inline_type_repair no longer triggers its required compiler refusal.
export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(['inline_type_repair']);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}

/** Source counterfactuals whose labels cannot be established by deleting an annotated proof leaf. */
export function quarantineReason(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string; variant?: string; answer_evidence?: string[] } | undefined;
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
}
