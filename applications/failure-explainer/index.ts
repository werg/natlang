/**
 * Advisory failure explanation (plans/FAILURE_EXPLANATION_PROGRAM.md, review items P5 and P11). Three natural-language
 * functions (explainFailure.nl, triageRejections.nl, explainGateFailure.nl) read crisp facts: the card of a failed
 * student-projection attempt, the bucket of admission reasons no rule classifies, the facts of a failed gate. This file
 * is the crisp side: the checks of every answer against its inputs. The label and writer of the advisory files are
 * ts-host/scripts/advisory-file.mjs, shared with the curriculum advisories.
 *
 * Nothing here decides anything the pipeline acts on. The audit's counts and tags, `classifyAdmissionReason`,
 * `dpoHoldReasons`, the gates and the training admission never read these files.
 */
import { untrusted, type NatlangRuntime } from '@natlang/node';
import explainFailure from './explainFailure.nl';
import triageRejections from './triageRejections.nl';
import explainGateFailure from './explainGateFailure.nl';
import type { BucketEntry, FailureCard, UntrustedBucketEntry, UntrustedFailureCard, FailureExplanation, GateExplanation, GateFacts, RuleCategory, RuleProposal, RuleProposals } from './types.js';

export type * from './types.js';

export const UNCLASSIFIED_CATEGORY = 'unclassified_review_pending';
export const UNCLASSIFIED_ACTION = 'retain_raw_evidence_and_review_before_training';

// ---------------------------------------------------------------- explainFailure
/** Every text the explanation may quote: the card's error, detail, actions and feedback. */
function cardTexts(card: FailureCard): string[] {
  return [String(card.error ?? ''), String(card.outcome_detail ?? ''), ...card.actions.map(String), ...card.feedback.map(String)];
}
/** Problems of an explanation against the card and the tag vocabulary; empty when it is acceptable. */
export function checkFailureExplanation(card: FailureCard, tags: string[], out: FailureExplanation): string[] {
  const problems: string[] = [];
  if (out.tag !== 'new' && !tags.includes(out.tag)) problems.push(`tag ${JSON.stringify(out.tag)} is neither one of the tags nor new`);
  if (out.tag === 'new' && out.proposed_tag && tags.includes(out.proposed_tag)) problems.push(`proposed_tag ${out.proposed_tag} is already a tag; use it as the tag`);
  const texts = cardTexts(card);
  for (const { quote } of out.evidence) if (!texts.some(text => text.includes(quote))) problems.push(`the quote ${JSON.stringify(quote.slice(0, 80))} does not occur in the card`);
  return problems;
}
export type Checked<T> = { value: T, problems: string[] };

export async function explainFailures(runtime: NatlangRuntime, cards: { id: string, card: FailureCard }[], tags: string[]): Promise<Record<string, Checked<FailureExplanation>>> {
  const out: Record<string, Checked<FailureExplanation>> = {};
  for (const { id, card } of cards) {
    const wrapped: UntrustedFailureCard = { ...card, error: untrusted(card.error, 'receipt.error'), outcome_detail: untrusted(card.outcome_detail, 'receipt.outcome'),
      actions: card.actions.map(text => untrusted(text, 'model action')), feedback: card.feedback.map(text => untrusted(text, 'tool feedback')) };
    const value = await runtime.run(() => explainFailure(wrapped, tags), { name: 'explain-failure' }) as FailureExplanation;
    out[id] = { value, problems: checkFailureExplanation(card, tags, value) };
  }
  return out;
}

// ---------------------------------------------------------------- triageRejections
/** A rule candidate as the admission ladder would apply it: the reason, or its prefix before the first colon, starts with the prefix. */
export const prefixCovers = (prefix: string, reason: string): boolean => prefix.length > 0 && (reason.startsWith(prefix) || reason.split(':', 1)[0]!.startsWith(prefix));

export type ClassifiedReason = { reason: string, category: string };
/** The reasons that fall to the unclassified category, grouped with their counts and example rows. */
export function unclassifiedBucket(rows: { reason: string, id?: string, family?: string }[], classify: (reason: string) => string, maxExamples = 5): BucketEntry[] {
  const groups = new Map<string, BucketEntry>();
  for (const row of rows) {
    if (classify(row.reason) !== UNCLASSIFIED_CATEGORY) continue;
    const entry = groups.get(row.reason) ?? { reason: row.reason, count: 0, examples: [] };
    entry.count++;
    if (entry.examples.length < maxExamples && row.id !== undefined) entry.examples.push({ id: row.id, family: row.family ?? 'unknown' });
    groups.set(row.reason, entry);
  }
  return [...groups.values()].sort((a, b) => b.count - a.count || (a.reason < b.reason ? -1 : 1));
}
/** The categories and example reasons of an existing ladder, as data for the function. */
export function ruleCategories(rules: { category: string, action: string }[], classified: ClassifiedReason[], perCategory = 8): RuleCategory[] {
  const byCategory = new Map<string, RuleCategory>();
  for (const rule of rules) if (!byCategory.has(rule.category)) byCategory.set(rule.category, { category: rule.category, action: rule.action, example_reasons: [] });
  for (const { reason, category } of classified) {
    const entry = byCategory.get(category);
    if (entry && entry.example_reasons.length < perCategory && !entry.example_reasons.includes(reason)) entry.example_reasons.push(reason);
  }
  return [...byCategory.values()];
}
/**
 * Exact verifier of one proposed rule. It holds when the prefix matches the proposal's own reason, the category is a known
 * one with that category's action, the prefix covers no reason that is already classified into another category (so adding
 * the rule changes no existing non-unclassified result), and every reason of the bucket it covers is proposed for the same category.
 */
export function checkProposedRule(proposal: RuleProposal, context: { bucket: BucketEntry[], classified: ClassifiedReason[], categories: RuleCategory[], proposals: RuleProposal[] }): string[] {
  const problems: string[] = [];
  const category = context.categories.find(item => item.category === proposal.category);
  if (!context.bucket.some(entry => entry.reason === proposal.reason)) problems.push('reason is not in the bucket');
  if (!category) problems.push(`category ${JSON.stringify(proposal.category)} is not one of the categories`);
  else if (proposal.category !== UNCLASSIFIED_CATEGORY && proposal.next_action !== category.action) problems.push(`next_action is not the action of ${proposal.category}`);
  if (!prefixCovers(proposal.match_prefix, proposal.reason)) problems.push('match_prefix does not match its reason');
  for (const { reason, category: existing } of context.classified)
    if (existing !== UNCLASSIFIED_CATEGORY && prefixCovers(proposal.match_prefix, reason)) {
      problems.push(`match_prefix would also cover ${JSON.stringify(reason)}, already classified as ${existing}`);
      break;
    }
  for (const entry of context.bucket) {
    if (entry.reason === proposal.reason || !prefixCovers(proposal.match_prefix, entry.reason)) continue;
    const other = context.proposals.find(item => item.reason === entry.reason);
    if (!other || other.category !== proposal.category) problems.push(`match_prefix also covers ${JSON.stringify(entry.reason)}, which is not proposed for ${proposal.category}`);
  }
  return problems;
}
export type CheckedProposal = { proposal: RuleProposal, problems: string[] };

export async function triage(runtime: NatlangRuntime, bucket: BucketEntry[], categories: RuleCategory[], classified: ClassifiedReason[]): Promise<{ proposals: CheckedProposal[], missing: string[] }> {
  if (!bucket.length) return { proposals: [], missing: [] };
  const wrapped: UntrustedBucketEntry[] = bucket.map(entry => ({ ...entry, reason: untrusted(entry.reason, 'admission reason') }));
  const result = await runtime.run(() => triageRejections(wrapped, categories), { name: 'triage-rejections' }) as RuleProposals;
  const context = { bucket, classified, categories, proposals: result.proposals };
  return { proposals: result.proposals.map(proposal => ({ proposal, problems: checkProposedRule(proposal, context) })),
    missing: bucket.filter(entry => !result.proposals.some(item => item.reason === entry.reason)).map(entry => entry.reason) };
}

// ---------------------------------------------------------------- explainGateFailure
/** The names a cause may cite as its support: fields of the facts and strata, metrics and windows named in them. */
export function citableNames(facts: GateFacts): string[] {
  return [...Object.keys(facts), 'margin', 'observed', 'limit', 'worst_windows', 'earlier_reports', 'checkpoint', 'step',
    ...facts.failed.flatMap(item => [item.stratum, item.metric]), ...facts.passed_strata, ...facts.worst_windows.flatMap(item => [item.id, item.metric])];
}
/** Problems of a gate explanation against its facts; empty when acceptable. */
export function checkGateExplanation(facts: GateFacts, out: GateExplanation): string[] {
  const problems: string[] = [];
  if (out.gate_unchanged !== true) problems.push('gate_unchanged must be true');
  if (facts.passed) problems.push('the facts describe a gate that passed; there is nothing to explain');
  const names = citableNames(facts);
  for (const cause of out.candidate_causes)
    if (!names.some(name => name && cause.support.includes(name))) problems.push(`support ${JSON.stringify(cause.support.slice(0, 80))} cites no field, stratum, metric or window of the facts`);
  return problems;
}
/** Facts must be consistent with themselves: the failed entries have a margin of the right sign, and a failed gate lists at least one. */
export function checkGateFacts(facts: GateFacts): string[] {
  const problems: string[] = [];
  if (!facts.passed && !facts.failed.length) problems.push('the gate failed but no failed entry is listed');
  if (facts.passed && facts.failed.length) problems.push('the gate passed but failed entries are listed');
  for (const item of facts.failed) {
    const failing = item.direction === 'min' ? item.observed < item.limit : item.observed > item.limit;
    if (!failing) problems.push(`${item.stratum}/${item.metric} is listed as failed but observed ${item.observed} meets the limit ${item.limit}`);
  }
  return problems;
}
export async function explainGate(runtime: NatlangRuntime, facts: GateFacts, report: string): Promise<Checked<GateExplanation>> {
  const factProblems = checkGateFacts(facts);
  if (factProblems.length) throw new Error(`gate facts are inconsistent: ${factProblems.join('; ')}`);
  const value = await runtime.run(() => explainGateFailure(facts, untrusted(report, 'gate report')), { name: 'explain-gate-failure' }) as GateExplanation;
  return { value, problems: checkGateExplanation(facts, value) };
}
