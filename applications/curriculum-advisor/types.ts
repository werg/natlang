import type { Untrusted } from '@natlang/node';

/** A held row whose wrong-looking answer may be a valid span of the same meaning (qasper, musique). Text fields come from datasets and model output. */
export type EquivalenceCard = {
  id: string, source: string, family: string,
  question: string,
  /** The annotated answers. Extractive annotations do not enumerate every valid span boundary. */
  gold: string[],
  /** What the model answered. */
  answer: string,
};
/** The card as the function reads it: quoted data, not instructions. */
export type UntrustedEquivalenceCard = {
  id: string, source: string, family: string,
  question: Untrusted<string>, gold: Untrusted<string>[], answer: Untrusted<string> };
export type Equivalence = "equivalent" | "different" | "unsure";
export type EquivalenceJudgment = {
  verdict: Equivalence,
  /** One sentence comparing the answer with the gold answers. */
  reason: string,
  /** Quotes copied from gold or answer that support the verdict. */
  evidence: { quote: string }[],
};

export type CountRow = { name: string, target: number, count: number, share: number };
/** Where the admitted cases stand against the target shares, computed by crisp code. */
export type CoverageFacts = {
  total: number, batch_size: number,
  slices: CountRow[], domains: CountRow[],
  /** Joint counts of slice and domain, so empty cells are visible. */
  cells: { slice: string, domain: string, count: number }[],
};
export type BatchLine = { slice: string, domain: string, count: number, rationale: string };
export type BatchProposal = { batch: BatchLine[], note: string };

/** The result of judgeAnswerEquivalence: an equivalent or different verdict quotes at least one text. */
export type CheckedEquivalenceJudgment = Is<EquivalenceJudgment, "a judgment whose verdict is equivalent, different or unsure, whose reason is one sentence, and whose evidence holds at least one non-empty quote unless the verdict is unsure">;
/** The result of nextCollectionBatch: positive whole counts and one rationale sentence per line. */
export type CheckedBatchProposal = Is<BatchProposal, "a proposal whose batch lines each have a positive whole count and a rationale, and whose note is a string">;
