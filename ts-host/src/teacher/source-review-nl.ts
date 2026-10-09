/**
 * The natlang reviewers of source items and produced rows (plans/SOURCE_REVIEW_PROGRAM.md, review items P2 and P10).
 *
 * `reviewSourceItem` recommends at intake whether an item deserves a hold; `reviewSourceRow` recommends on a produced
 * row whether the target is semantically equivalent to the source answer. Both are runtime-owned natlang functions
 * (src/builtin/*.nl) and both are advice: nothing here changes the registry (source-review.ts) or any admission. A
 * reviewer output carries what scripts/source_review.py turns into a receipt (the reviewer by hash, the recommendation,
 * `training_admission` false), and a person or an agent session decides.
 *
 * The exact parts stay crisp: the row prechecks, the choice of precedents, the checks of the typed contract (quotes
 * occur in the text, `proposed_entry` exactly for a hold) and the reviewer hash. The part is pluggable
 * (`crisp | nl | shadow`); the default `crisp` recommends nothing, which is today's behaviour.
 */
import { createHash } from 'node:crypto';
import { builtin, builtinDefinitionKey } from '../runtime/builtin.js';
import { builtinSource } from '../builtin/index.js';
import { pluggable, type PluggableSetting } from '../runtime/pluggable.js';
import { untrusted } from '../runtime/surface.js';
import { SOURCE_REVIEWS } from './source-review.js';

/** Part of every reviewer hash: a change of the review machinery that should make old receipts distinguishable. */
export const SOURCE_REVIEW_COMPILER_VERSION = 'natlang-ts-host/0.1.0';

export type Confidence = 'high' | 'medium' | 'low';
export type ItemConcern = 'none' | 'label-disagrees-with-source' | 'unstated-premise' | 'ambiguous-question' | 'contract-mismatch' | 'needs-context';
export type SourceItem = { dataset: string; id: string; visible: string; annotated_label: string; contract: string; answer_format: string | null };
export type Precedent = { id: string; concern: string; reason: string };
export type ItemRecommendation = {
  recommendation: 'admit' | 'hold'; concern: ItemConcern; reason: string; evidence: { quote: string }[];
  proposed_entry: { reason: string } | null; confidence: Confidence;
};
export type SourceRow = { id: string; dataset: string; split: string; source_groups: string[]; question: string; answer_format: string;
  evidence: string; gold: string; actual: string };
export type RowPrecheck = { exact_match: boolean; actual_is_exact_source_span: boolean; numerically_equal: boolean | null };
export type RowStatus = 'equivalent' | 'normalization-candidate' | 'mismatch' | 'ambiguous';
export type RowVerdict = { status: RowStatus; rationale: string; evidence: { quote: string }[]; confidence: Confidence };

/** Who said it: a natlang function by its exact definition, compiler and executor, or an exact precheck. */
export type ReviewerIdentity =
  | { kind: 'crisp'; function: string }
  | { kind: 'natlang'; function: string; definition_key: string; call_id: string | null;
      hash_inputs: { definition_source_sha256: string; compiler_version: string; executor: string }; reviewer_hash: string };

/** What scripts/source_review.py receipt reads. */
export type ReviewOutput<Input, Recommendation> = {
  kind: 'item' | 'row'; input: Input; reviewer: ReviewerIdentity; recommendation: Recommendation | null;
  second_reviewer?: ReviewerIdentity; second_recommendation?: Recommendation; excluded?: 'transport_error';
};

/** An executor of the reviewer functions: its identity (model or prompt variant) and how to call a built-in by name. */
export type ReviewExecutor = { executor: string; call?: (name: string, args: unknown[]) => Promise<unknown> };
export type ReviewOptions = {
  /** `crisp` (default) recommends nothing; `nl` asks the reviewer; `shadow` asks and compares with the crisp side. */
  mode?: PluggableSetting;
  primary: ReviewExecutor;
  /** A second executor (another model or prompt variant) for an independent agreement signal. */
  second?: ReviewExecutor;
};

const sha256 = (text: string): string => createHash('sha256').update(text).digest('hex');

/** The hash a receipt carries; scripts/source_review.py recomputes it the same way. */
export function reviewerHash(definitionSourceSha256: string, compilerVersion: string, executor: string): string {
  return `natlang@${sha256(`${definitionSourceSha256}\0${compilerVersion}\0${executor}`).slice(0, 16)}`;
}

/** Canonical JSON with sorted keys and compact separators, as scripts/source_review.py hashes it (strings, booleans, null, arrays). */
export function canonicalJson(value: unknown): string {
  const walk = (item: unknown): unknown => Array.isArray(item) ? item.map(walk) : item && typeof item === 'object' ?
    Object.fromEntries(Object.entries(item as Record<string, unknown>).filter(([, entry]) => entry !== undefined).sort(([a], [b]) => a < b ? -1 : a > b ? 1 : 0)
      .map(([key, entry]) => [key, walk(entry)])) : item;
  return JSON.stringify(walk(value));
}

export function reviewerIdentity(name: 'reviewSourceItem' | 'reviewSourceRow', executor: string): ReviewerIdentity {
  const definitionSourceSha256 = sha256(builtinSource(name));
  return { kind: 'natlang', function: name, definition_key: builtinDefinitionKey(name), call_id: null,
    hash_inputs: { definition_source_sha256: definitionSourceSha256, compiler_version: SOURCE_REVIEW_COMPILER_VERSION, executor },
    reviewer_hash: reviewerHash(definitionSourceSha256, SOURCE_REVIEW_COMPILER_VERSION, executor) };
}

/** A failure of the typed contract of a reviewer; the message says what to return instead. */
export class SourceReviewContractError extends Error {
  constructor(message: string) { super(message); this.name = 'SourceReviewContractError'; }
}

const asNumber = (text: string): number | undefined => {
  const cleaned = text.trim().replace(/(?<=\d),(?=\d{3}(\D|$))/g, '');
  return /^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$/.test(cleaned) ? Number(cleaned) : undefined;
};

/** The exact checks that settle a row without a model: exact match, exact source span, numeric equality. */
export function sourceRowPrecheck(row: Pick<SourceRow, 'gold' | 'actual' | 'evidence'>): RowPrecheck {
  const gold = asNumber(row.gold), actual = asNumber(row.actual);
  const numericallyEqual = gold === undefined || actual === undefined ? null :
    gold === actual || Math.abs(gold - actual) <= 1e-12 * Math.max(Math.abs(gold), Math.abs(actual));
  return { exact_match: row.gold === row.actual, actual_is_exact_source_span: row.actual.length > 0 && row.evidence.includes(row.actual),
    numerically_equal: numericallyEqual };
}

const words = (text: string): Set<string> => new Set(text.toLowerCase().match(/[\p{L}\p{N}]+/gu) ?? []);

/** Up to `count` held items of the same dataset whose text is most similar to `visible` (Jaccard over words); the model reads their wording. */
export function chooseSourcePrecedents(dataset: string, visible: string, count = 3): Precedent[] {
  const target = words(visible);
  return SOURCE_REVIEWS.filter(review => review.dataset === dataset && review.status === 'pending').map(review => {
    const other = words(review.text);
    let shared = 0;
    for (const word of target) if (other.has(word)) shared++;
    return { review, score: shared / (target.size + other.size - shared || 1) };
  }).sort((a, b) => b.score - a.score || (a.review.id < b.review.id ? -1 : 1)).slice(0, count)
    .map(({ review }) => ({ id: review.id, concern: 'held', reason: review.reason }));
}

function checkQuotes(evidence: unknown, source: string, where: string): void {
  if (!Array.isArray(evidence)) throw new SourceReviewContractError('evidence is a list of { quote } entries; return [] when no sentence applies');
  evidence.forEach((entry, index) => {
    const quote = (entry as { quote?: unknown } | null)?.quote;
    if (typeof quote !== 'string' || !source.includes(quote))
      throw new SourceReviewContractError(`evidence[${index}].quote is copied exactly from ${where}; copy a sentence that occurs there`);
  });
}

/** The crisp checks of the typed contract of `reviewSourceItem`. */
export function checkItemRecommendation(item: Pick<SourceItem, 'visible'>, value: ItemRecommendation): ItemRecommendation {
  if (value.recommendation !== 'admit' && value.recommendation !== 'hold')
    throw new SourceReviewContractError('recommendation is "admit" or "hold"');
  if ((value.concern === 'none') !== (value.recommendation === 'admit'))
    throw new SourceReviewContractError('concern is "none" exactly when recommendation is "admit"; name the concern for a hold');
  const proposed = value.proposed_entry;
  if ((value.recommendation === 'hold') !== (proposed !== null && typeof proposed?.reason === 'string' && proposed.reason.trim() !== ''))
    throw new SourceReviewContractError('proposed_entry { reason } is present exactly when recommendation is "hold", and null for "admit"');
  checkQuotes(value.evidence, item.visible, 'item.visible');
  return value;
}

/** The crisp checks of the typed contract of `reviewSourceRow`. */
export function checkRowVerdict(row: Pick<SourceRow, 'evidence'>, precheck: RowPrecheck, value: RowVerdict): RowVerdict {
  if (!['equivalent', 'normalization-candidate', 'mismatch', 'ambiguous'].includes(value.status))
    throw new SourceReviewContractError('status is "equivalent", "normalization-candidate", "mismatch" or "ambiguous"');
  if (precheck.exact_match && value.status !== 'equivalent')
    throw new SourceReviewContractError('status is "equivalent" whenever precheck.exact_match is true');
  checkQuotes(value.evidence, row.evidence, 'row.evidence');
  return value;
}

const callBuiltin = async (name: string, args: unknown[]): Promise<unknown> => (builtin(name) as (...positional: unknown[]) => Promise<unknown>)(...args);

/** Run one reviewer on one function and check its answer. */
async function ask<Result>(executor: ReviewExecutor, name: 'reviewSourceItem' | 'reviewSourceRow', args: unknown[],
    check: (value: Result) => Result): Promise<{ reviewer: ReviewerIdentity; recommendation: Result }> {
  const value = await (executor.call ?? callBuiltin)(name, args) as Result;
  return { reviewer: reviewerIdentity(name, executor.executor), recommendation: check(value) };
}

/**
 * Recommend at intake whether `item` deserves a hold. Items that already have a registry entry are not reviewed again;
 * their entry is the standing decision (the caller checks `pendingSourceReview` first).
 */
export async function reviewSourceItem(item: SourceItem, options: ReviewOptions,
    precedents: Precedent[] = chooseSourcePrecedents(item.dataset, item.visible)):
    Promise<ReviewOutput<{ item: SourceItem; precedents: Precedent[] }, ItemRecommendation> | null> {
  const input = { item, precedents };
  const args = [{ ...item, visible: untrusted(item.visible, 'source item'), annotated_label: untrusted(item.annotated_label, 'source item') }, precedents];
  const run = pluggable<[ReviewExecutor], { reviewer: ReviewerIdentity; recommendation: ItemRecommendation } | null>({
    crisp: () => null,
    nl: executor => ask<ItemRecommendation>(executor, 'reviewSourceItem', args, value => checkItemRecommendation(item, value)),
  }, options.mode, { default: 'crisp', name: 'reviewSourceItem', serve: 'nl', same: (crisp, nl) => crisp === null && nl === null });
  const first = await run(options.primary);
  if (!first) return null;
  const output: ReviewOutput<typeof input, ItemRecommendation> = { kind: 'item', input, reviewer: first.reviewer, recommendation: first.recommendation };
  if (options.second) {
    const second = await ask<ItemRecommendation>(options.second, 'reviewSourceItem', args, value => checkItemRecommendation(item, value));
    output.second_reviewer = second.reviewer; output.second_recommendation = second.recommendation;
  }
  return output;
}

/**
 * Recommend on a produced row whether `actual` says what `gold` says. An exact match is `equivalent` without a call
 * (a crisp reviewer); the packets still need an explicit decision before they read it.
 */
export async function reviewSourceRow(row: SourceRow, options: ReviewOptions,
    precheck: RowPrecheck = sourceRowPrecheck(row)):
    Promise<ReviewOutput<{ row: SourceRow; precheck: RowPrecheck }, RowVerdict> | null> {
  const input = { row, precheck };
  if (precheck.exact_match) {
    return { kind: 'row', input, reviewer: { kind: 'crisp', function: 'precheck' },
      recommendation: { status: 'equivalent', rationale: 'The produced answer equals the source answer exactly.', evidence: [], confidence: 'high' } };
  }
  const args = [{ ...row, question: untrusted(row.question, 'source row'), evidence: untrusted(row.evidence, 'source row'),
    gold: untrusted(row.gold, 'source row'), actual: untrusted(row.actual, 'source row') }, precheck];
  const run = pluggable<[ReviewExecutor], { reviewer: ReviewerIdentity; recommendation: RowVerdict } | null>({
    crisp: () => null,
    nl: executor => ask<RowVerdict>(executor, 'reviewSourceRow', args, value => checkRowVerdict(row, precheck, value)),
  }, options.mode, { default: 'crisp', name: 'reviewSourceRow', serve: 'nl', same: (crisp, nl) => crisp === null && nl === null });
  const first = await run(options.primary);
  if (!first) return null;
  const output: ReviewOutput<typeof input, RowVerdict> = { kind: 'row', input, reviewer: first.reviewer, recommendation: first.recommendation };
  if (options.second) {
    const second = await ask<RowVerdict>(options.second, 'reviewSourceRow', args, value => checkRowVerdict(row, precheck, value));
    output.second_reviewer = second.reviewer; output.second_recommendation = second.recommendation;
  }
  return output;
}

/** Whether two recommendations agree: the `recommendation` enum for items, the `status` enum for rows. */
export const recommendationsAgree = (kind: 'item' | 'row', a: ItemRecommendation | RowVerdict, b: ItemRecommendation | RowVerdict): boolean =>
  kind === 'item' ? (a as ItemRecommendation).recommendation === (b as ItemRecommendation).recommendation : (a as RowVerdict).status === (b as RowVerdict).status;
