/**
 * Promotion policy selection and the off-hot-path review (plans/TIERED_ENGINE.md "Evidence and promotion").
 *
 * The mechanism is crisp and lives in calls/evidence.ts (counters, summaries, applying a decision). The policy, deciding
 * promote / keep / demote from a summary, has two implementations: the fixed rule `crispPolicy`, and the natural-language
 * function applications/specializer/promote.nl. `promotionPolicy` selects between them with `pluggable()` under the store
 * setting `promotionPolicy` (`crisp` default, `nl`, `shadow`). The natural-language policy never runs per call:
 * `reviewPromotions` runs from the specializer loop, over the evidence the store has by then.
 */
import type { CallStore } from '../calls/store.js';
import { TierLedger } from '../calls/tiers.js';
import { crispPolicy, ruleOf, summaryOfTier, type EvidenceSummary, type PromotionDecision } from '../calls/evidence.js';
import { pluggable, type PluggableMode } from './pluggable.js';

export type PolicyFunction = (summary: EvidenceSummary) => PromotionDecision | Promise<PromotionDecision>;

/** What a policy may answer; anything else is read as `keep`, so a confused model never moves a case or tier. */
export function sanitizeDecision(value: unknown): PromotionDecision {
  const answer = value as Partial<PromotionDecision> | null;
  const decision = answer?.decision;
  if (decision === 'promote' || decision === 'keep' || decision === 'demote')
    return { decision, reason: typeof answer?.reason === 'string' && answer.reason ? answer.reason : '(no reason given)' };
  return { decision: 'keep', reason: 'the policy gave no valid decision' };
}

/** The policy as one function: `crisp`, `nl`, or `shadow` (both run, the crisp decision serves, agreement is traced). */
export function promotionPolicy(mode: PluggableMode, nl: PolicyFunction): (summary: EvidenceSummary) => Promise<PromotionDecision> {
  return pluggable<[EvidenceSummary], PromotionDecision>({ crisp: summary => crispPolicy(summary),
    nl: async summary => sanitizeDecision(await nl(summary)) }, mode,
  { name: 'promotion', serve: 'crisp', same: (crisp, other) => crisp.decision === other.decision });
}

export type ReviewedSubject = { subject: 'case' | 'tier'; id: string; state: string; decision: PromotionDecision; applied: boolean };

/**
 * Judge every case of a current compilation and every tier with evidence by the store's policy and apply the decisions.
 * Under `crisp` the store has already applied the rule as evidence arrived, so nothing is left to do. Under `shadow`
 * both policies are asked and the crisp answer is the one applied (it already was); the natural-language answer is
 * only compared. Under `nl` its answers are applied: a case through `store.decideTier`, a tier by a `promoted` /
 * `demoted` event kept with its last call, which running engines read back.
 */
export async function reviewPromotions(store: CallStore, nl: PolicyFunction, options: { mode?: PluggableMode } = {}): Promise<ReviewedSubject[]> {
  const mode = options.mode ?? store.settings().promotionPolicy;
  if (mode === 'crisp') return [];
  const policy = promotionPolicy(mode, nl);
  const reviewed: ReviewedSubject[] = [];
  const apply = mode === 'nl';
  for (const compilation of store.compilations({ status: 'current', limit: 1000 })) {
    for (const item of store.cases(compilation.id)) {
      if (item.tier === 'disabled' || item.tier === 'demoted') continue;
      const summary = store.caseSummary(item.hash);
      if (!summary) continue;
      const decision = await policy(summary);
      const tier = apply ? store.decideTier(summary, decision, 'nl') : summary.state;
      reviewed.push({ subject: 'case', id: item.hash, state: summary.state, decision, applied: tier !== summary.state });
    }
  }
  const rule = ruleOf(store.settings());
  const ledger = new TierLedger();
  ledger.hydrate(store, undefined, tier => tier === 'tier2' || tier.startsWith('tier3:') ? 'shadow' : 'active', rule);
  for (const { fn, tier, row, call } of ledger.entries()) {
    if (tier === 'tier0' || tier === 'tier3') continue;
    const state = row.state;
    const decision = await policy(summaryOfTier(`${fn}/${tier}`, row, rule));
    const event = apply ? ledger.decide(fn, tier, decision, 'nl') : undefined;
    if (event && call) store.annotate(call, 'tier', { ...event, call_id: call, state }, 'policy', false);
    reviewed.push({ subject: 'tier', id: `${fn}/${tier}`, state, decision, applied: !!event });
  }
  return reviewed;
}
