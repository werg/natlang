/**
 * Evidence and promotion, one module for compilation cases and for tiers (plans/TIERED_ENGINE.md "Evidence and promotion").
 *
 * Mechanism (crisp, here): the evidence counters, how events fold into them, the summary a policy judges, and applying a
 * decision. Policy: given an `EvidenceSummary`, answer promote / keep / demote with a reason. The fixed rule is
 * `crispPolicy`; the natural-language policy is applications/specializer/promote.nl, and `runtime/promotion.ts` selects
 * between them with `pluggable()` under the store setting `promotionPolicy`. A case and a tier are judged by the same
 * policy over the same counters, so one meaning of compared, worse, live, audited and handed off applies to both.
 */
import type { CallStoreSettings } from './types.js';

/** What a case or a tier has shown. Counts are since the subject was created, or since it was last demoted (tiers). */
export type Evidence = {
  /** Calls the subject served. */
  served: number;
  /** Calls it started and then had to hand to something below (the answer was wrong or its code failed): a failure of the subject. */
  handed_off: number;
  /** Tiers only: attempts that stopped at the subject's guard. Not a failure: the subject was never asked to answer. */
  guard_misses: number;
  /** Tiers only: attempts lost to the infrastructure (a timeout, an unreachable model). Bad luck, not a break. */
  infrastructure: number;
  /** Comparisons against the reference executor: held-out replays, shadow checks and audits. */
  compared: number; worse: number; better: number;
  /** Of those, comparisons on calls made after the subject existed (shadow and audit). Replays share their context with the training calls. */
  live_compared: number; live_worse: number;
  /** Of those, audits of calls the subject served. */
  audited: number; audit_worse: number;
  /** The most recent comparisons (up to `RECENT`). */
  recent_compared: number; recent_worse: number;
};
export const RECENT = 10;
export const emptyEvidence = (): Evidence => ({ served: 0, handed_off: 0, guard_misses: 0, infrastructure: 0, compared: 0, worse: 0, better: 0,
  live_compared: 0, live_worse: 0, audited: 0, audit_worse: 0, recent_compared: 0, recent_worse: 0 });

/** The thresholds, from the store's settings (`acceptanceBound`, `promotionComparisons`, `promotionLiveComparisons`). */
export type EvidenceRule = { bound: number; comparisons: number; liveComparisons: number; auditMinimum: number; callMinimum: number };
/** Audits before a worse share demotes, and served plus handed-off calls before the hand-off share demotes (§6.2). */
export const AUDIT_MINIMUM = 5, CALL_MINIMUM = 10;
export const ruleOf = (settings: Pick<CallStoreSettings, 'acceptanceBound' | 'promotionComparisons'> & Partial<Pick<CallStoreSettings, 'promotionLiveComparisons'>>): EvidenceRule =>
  ({ bound: settings.acceptanceBound, comparisons: settings.promotionComparisons, liveComparisons: settings.promotionLiveComparisons ?? 0,
    auditMinimum: AUDIT_MINIMUM, callMinimum: CALL_MINIMUM });

export type EvidenceState = 'shadow' | 'active' | 'demoted' | 'disabled';
export type EvidenceSummary = {
  subject: 'case' | 'tier';
  /** Case hash, or `function/tier`. */
  id: string;
  state: EvidenceState;
  evidence: Evidence;
  rule: EvidenceRule;
};
export type PromotionDecision = { decision: 'promote' | 'keep' | 'demote'; reason: string };

/**
 * The fixed rule (§6.2). A shadow subject is promoted after `comparisons` comparisons with at most `bound` of them worse,
 * and, of those, `liveComparisons` on live calls within the same bound. An active one is demoted when its audits, its
 * hand-offs or its comparisons exceed the bound. Anything else keeps its state. Guard misses and infrastructure failures
 * are not evidence against a subject.
 */
export function crispPolicy(summary: EvidenceSummary): PromotionDecision {
  const { evidence: e, rule } = summary;
  const share = (worse: number, of: number) => `${worse}/${of}`;
  if (summary.state === 'shadow') {
    if (e.compared >= rule.comparisons && e.worse <= rule.bound * e.compared && e.live_compared >= rule.liveComparisons &&
        e.live_worse <= rule.bound * e.live_compared)
      return { decision: 'promote', reason: `${e.compared} comparisons (${share(e.worse, e.compared)} worse), ${e.live_compared} live (${share(e.live_worse, e.live_compared)} worse) within ${rule.bound}` };
    return { decision: 'keep', reason: 'not enough clean comparisons yet' };
  }
  if (summary.state === 'active') {
    const calls = e.served + e.handed_off;
    if (e.audited >= rule.auditMinimum && e.audit_worse > rule.bound * e.audited)
      return { decision: 'demote', reason: `${share(e.audit_worse, e.audited)} audits worse, over ${rule.bound}` };
    if (calls >= rule.callMinimum && e.handed_off > rule.bound * calls)
      return { decision: 'demote', reason: `${share(e.handed_off, calls)} calls handed off, over ${rule.bound}` };
    if (e.compared >= rule.comparisons && e.worse > rule.bound * e.compared)
      return { decision: 'demote', reason: `${share(e.worse, e.compared)} comparisons worse, over ${rule.bound}` };
  }
  return { decision: 'keep', reason: 'within bounds' };
}

/** The state a decision leads to. Only a shadow subject is promoted and only an active one demoted; others stay. */
export function applyDecision(state: EvidenceState, decision: PromotionDecision['decision']): EvidenceState {
  if (decision === 'promote' && state === 'shadow') return 'active';
  if (decision === 'demote' && state === 'active') return 'demoted';
  return state;
}

// --- Folding tier events into evidence ------------------------------------------------------------------------------

export type DeoptKind = 'guard' | 'verify' | 'error' | 'infrastructure';

/** `infrastructure` for a failure that says nothing about the subject (the model or the network), else `error`. */
export function failureKind(error: unknown): 'error' | 'infrastructure' {
  const text = error instanceof Error ? `${error.name} ${error.message}` : String(error);
  return /ETIMEDOUT|ECONNRESET|ECONNREFUSED|ENOTFOUND|EAI_AGAIN|timed? ?out|rate.?limit|\b429\b|\b50[234]\b|unavailable|socket hang up|fetch failed|network|abort/i.test(text) ? 'infrastructure' : 'error';
}

/** A tier's running evidence: the counters, and the window behind `recent_*`. */
export type TierEvidence = { state: EvidenceState; evidence: Evidence; window: boolean[] };
export const newTierEvidence = (state: EvidenceState): TierEvidence => ({ state, evidence: emptyEvidence(), window: [] });

/** Fold one event into the counters. `served`, `deopt` (by kind) and shadow comparisons count; `promoted`/`demoted` are decisions. */
export function foldEvent(row: TierEvidence, event: { event: string; kind?: string }): void {
  const e = row.evidence;
  if (event.event === 'served') e.served++;
  else if (event.event === 'deopt') {
    if (event.kind === 'guard') e.guard_misses++;
    else if (event.kind === 'infrastructure') e.infrastructure++;
    else e.handed_off++;
  } else if (event.event === 'shadow_equal' || event.event === 'shadow_worse') {
    const worse = event.event === 'shadow_worse';
    e.compared++; e.live_compared++;
    if (worse) { e.worse++; e.live_worse++; }
    row.window.push(worse);
    if (row.window.length > RECENT) row.window.shift();
    e.recent_compared = row.window.length; e.recent_worse = row.window.filter(Boolean).length;
  }
}

/** A tier judged by the shared policy: a demoted tier is judged as a shadow one, over the evidence gathered since it was demoted. */
export function summaryOfTier(id: string, row: TierEvidence, rule: EvidenceRule): EvidenceSummary {
  return { subject: 'tier', id, state: row.state === 'demoted' ? 'shadow' : row.state, evidence: row.evidence, rule };
}

/** Move a tier to the state a decision leads to; a demoted tier starts a fresh evidence window. Returns the new state if it moved. */
export function moveTier(row: TierEvidence, decision: PromotionDecision['decision']): EvidenceState | undefined {
  const base: EvidenceState = row.state === 'demoted' ? 'shadow' : row.state;
  const next = applyDecision(base, decision);
  if (next === base) return;
  setTierState(row, next);
  return next;
}
export function setTierState(row: TierEvidence, state: EvidenceState): void {
  row.state = state;
  if (state === 'demoted') { row.evidence = emptyEvidence(); row.window = []; }
}
