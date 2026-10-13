import type { SourceSnapshot, Witness, ProposedEdit, ExactCheck } from './types.js';
/** Exact source and patch mechanics; these methods contain no story policy. */
export type StoryEvidence = {
  /** Checks path/id identity and exact quotation containment in source text. */
  checkWitnesses(snapshot: SourceSnapshot, witnesses: Witness[]): ExactCheck[];
  /** Checks listed paths, matching original spans, and absence of overlapping edits. */
  checkEdits(snapshot: SourceSnapshot, edits: ProposedEdit[], allowedPaths: string[]): ExactCheck[];
  /** Applies exact listed replacements to a new in-memory snapshot; verifies spans first. */
  applySnapshot(snapshot: SourceSnapshot, edits: ProposedEdit[], allowedPaths: string[]): SourceSnapshot;
};
declare module 'natlang:services' { export const storyEvidence: StoryEvidence; }
