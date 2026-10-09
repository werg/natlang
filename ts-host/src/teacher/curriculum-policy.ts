import { hasExistingTreeValueContract } from './tree-contract.js';
/** Shared collection/admission policy for exercises whose premise no longer exists in the runtime. */
import type { ProgramRecord } from './program.js';
import { sourceReviewReason } from './source-review.js';
import { markdownTerminalNewlineEqual } from '../evaluation/oracles.js';
import '../benchmarks/builtin.js';
import { runtimeFailureRulesFor } from '../benchmarks/registry.js';
import { GENERATION_HOLD_RULES, QUARANTINE_RULES, RETIRED_FAMILY_NAMES, RUNTIME_FAILURE_RULES, firstReason, type RuleContext } from './curriculum-rules.js';

// Untyped nl results now run open, so inline_type_repair no longer triggers its required compiler refusal. The names are data
// in curriculum-rules.ts (RETIRED_FAMILY_NAMES), with the rule ladders below.
export const RETIRED_FAMILIES: ReadonlySet<string> = new Set(RETIRED_FAMILY_NAMES);
export function retiredFamily(record: ProgramRecord): string | undefined {
  const curriculum = record.curriculum as { family?: string } | undefined;
  return curriculum?.family && RETIRED_FAMILIES.has(curriculum.family) ? curriculum.family : undefined;
}

/** What the ladders' named predicates and steps read; the rungs themselves are data (curriculum-rules.ts). */
const contextOf = (record: ProgramRecord, row?: unknown): RuleContext => ({ record, curriculum: record.curriculum, row });

/** Source counterfactuals whose labels cannot be established by deleting an annotated proof leaf. The ladder is QUARANTINE_RULES. */
export function quarantineReason(record: ProgramRecord): string | undefined {
  return firstReason(QUARANTINE_RULES, contextOf(record), {
    predicates: { 'no-tree-value-contract': () => !hasExistingTreeValueContract(record.semantics?.inputs?.state, record.semantics?.expected) },
    steps: { 'source-review': () => sourceReviewReason(record) },
  });
}


/** Unfair answer-span contracts must not consume live generation requests. Verified static reads remain useful. */
export function generationHoldReason(record: ProgramRecord): string | undefined {
  return firstReason(GENERATION_HOLD_RULES, contextOf(record));
}

/** Exact old CommitPack false-negative shape: every non-target file and causal check still passes. */
export function legacyMarkdownTerminalNewlineFailure(row: { task?: Record<string, unknown>; outcome?: Record<string, unknown> }): boolean {
  const record = row.task?.program_ir as ProgramRecord | undefined, outcome = row.outcome;
  const generation = record?.generation as Record<string, unknown> | undefined;
  if (!record || record.source !== 'commitpack' || !outcome || outcome.accepted !== false ||
      record.id.endsWith(':markdown-terminal-newline-v1') ||
      generation?.markdown_edit_contract_revision === 'commitpack-markdown-terminal-newline-v1' ||
      record.semantics.files_oracle?.compare === 'markdown-terminal-newline' ||
      JSON.stringify(outcome.rejection_reasons) !== JSON.stringify(['files']) || outcome.status !== 'done') return false;
  const input = record.semantics.folder_files as Record<string, string> | undefined;
  const expected = record.semantics.expected_files as Record<string, string> | undefined;
  const actual = outcome.files as Record<string, string> | undefined;
  if (!input || !expected || !actual || typeof input['change-request.json'] !== 'string') return false;
  let request: { path?: unknown; find?: unknown; replace_with?: unknown };
  try { request = JSON.parse(input['change-request.json']); } catch { return false; }
  const path = request.path;
  if (typeof path !== 'string' || path.startsWith('/') || path.includes('\\') || path.split('/').some(part => !part || part === '.' || part === '..') ||
      !/\.md$/i.test(path) || typeof request.find !== 'string' || !request.find.trim() ||
      typeof request.replace_with !== 'string' || !request.replace_with.trim()) return false;
  const before = input[path], gold = expected[path], got = actual[path];
  if (typeof before !== 'string' || typeof gold !== 'string' || typeof got !== 'string' || got === gold ||
      before.split(request.find).length !== 2 || before.replace(request.find, request.replace_with) !== gold ||
      markdownTerminalNewlineEqual(path, before, gold) || !markdownTerminalNewlineEqual(path, got, gold)) return false;
  const expectedPaths = Object.keys(expected).sort(), inputPaths = Object.keys(input).sort(), actualPaths = Object.keys(actual).sort();
  if (JSON.stringify(expectedPaths) !== JSON.stringify(inputPaths) || JSON.stringify(actualPaths) !== JSON.stringify(expectedPaths) ||
      inputPaths.some(name => name !== path && (input[name] !== expected[name] || actual[name] !== expected[name]))) return false;
  const checks = outcome.checks as Record<string, unknown> | undefined;
  const filesCheck = outcome.files_check as { failed?: unknown[]; errors?: unknown[]; pending?: unknown[] } | undefined;
  return outcome.value === record.semantics.expected && checks?.answer === true && checks?.file_return_consistency === true &&
    checks.files === false && Object.entries(checks).filter(([key, value]) => key !== 'files' && value !== true).length === 0 &&
    JSON.stringify(filesCheck?.failed) === JSON.stringify([path]) &&
    !(filesCheck?.errors?.length) && !(filesCheck?.pending?.length);
}

/** Old infrastructure failures must not teach models that correct actions are bad decisions. The ladder is RUNTIME_FAILURE_RULES. */
export function runtimeFailureReason(row: { task: Record<string, unknown>; provenance?: Record<string, unknown>;
  outcome?: Record<string, unknown>; trajectory?: unknown[] }): string | undefined {
  if (row.outcome?.accepted !== false) return;
  const record = row.task.program_ir as ProgramRecord;
  const effectiveFamily = () => (record.curriculum as { family?: string } | undefined)?.family ?? record.family;
  return firstReason(RUNTIME_FAILURE_RULES, contextOf(record, row), {
    predicates: {
      'legacy-markdown-terminal-newline': () => legacyMarkdownTerminalNewlineFailure(row),
      'treedst-oracle-not-named-tree': () => typeof record.semantics.oracle !== 'object' || record.semantics.oracle.normalization !== 'named-tree',
      'effective-family-inline-late-binding': () => effectiveFamily() === 'inline_late_binding',
      'effective-family-logic-proof-verifier': () => effectiveFamily() === 'logic_proof_verifier',
    },
    steps: {
      // Dataset-specific rules (e.g. benchmarks/tatqa) register by source name; sources are disjoint, so order is moot.
      'benchmark-rules': () => {
        for (const rule of runtimeFailureRulesFor(String(record.source))) {
          const reason = rule(row as Parameters<typeof rule>[0], record as unknown as Record<string, any>);
          if (reason) return reason;
        }
      },
    },
  });
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


/** Evaluation tolerances are useful for scoring; partial label agreement is not a positive training target. Free-text
 * (span) answers above their threshold are admitted even when imperfect (owner 2026-10-06: not overly picky). */
export function trainingQualityReason(row: { provenance?: Record<string, unknown>; outcome?: Record<string, unknown> }): string | undefined {
  const reviewed = trajectoryReviewReason(row);
  if (reviewed) return reviewed;
  const outcome = row.outcome ?? {};
  if (Array.isArray(outcome.quality_pending) && outcome.quality_pending.length) return 'quality_pending';
  const oracle = outcome.oracle as { level?: string; score?: number; needs_review?: boolean } | undefined;
  const files = outcome.files_check as { failed?: unknown[]; pending?: unknown[]; errors?: unknown[]; score?: number } | undefined;
  if (oracle?.needs_review || files?.pending?.length || files?.errors?.length) return 'quality_pending';
  if (outcome.accepted === true) {
    if (oracle?.level === 'agreement' && (oracle?.score ?? 1) < 1)
      return 'quality_pending_partial_agreement';
    if (files?.failed?.length || (files?.score ?? 1) < 1) return 'quality_pending_partial_files';
  }
}
