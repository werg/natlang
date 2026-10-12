/**
 * Storage retention as a pluggable policy (DECOMPOSITION.md). scripts/storage_retention.py computes the facts of each
 * large file and its crisp verdict; classifyRetention.nl is the natural-language implementation of the same decision.
 * `classifyItems` selects one by the mode (`crisp`, `nl`, `shadow`) through ts-host's `pluggable`, bounds every
 * natural-language verdict exactly (`refusal`) and returns the records of disagreements for the shadow log.
 *
 * Deleting stays in storage_retention.py and acts only on the served tiers.
 */
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableSetting } from '@natlang/node';
import classifyRetention from './classifyRetention.nl';
import type { RetentionFacts, RetentionItem, RetentionThresholds, RetentionTier, RetentionVerdict } from './types.js';

export type * from './types.js';

export const TIERS: readonly RetentionTier[] = ['KEEP', 'ACTIVE', 'PRUNE-AUTO', 'PRUNE-ASK'];
const DELETING: ReadonlySet<RetentionTier> = new Set(['PRUNE-AUTO', 'PRUNE-ASK']);

/**
 * The exact bound on a natural-language verdict: why it may not serve, or null. A verdict outside the four tiers is
 * refused, and so is one that lets a file go which the facts protect (registered, named, a record, a keep path, in use,
 * or in a run that changed within `finished_after_days`).
 */
export function refusal(facts: RetentionFacts, thresholds: RetentionThresholds, verdict: RetentionVerdict): string | null {
  if (!verdict || !TIERS.includes(verdict.tier)) return `tier ${JSON.stringify(verdict?.tier)} is not one of ${TIERS.join(', ')}`;
  if (!DELETING.has(verdict.tier)) return null;
  if (facts.registered) return 'a manifest pins this file; it is KEEP';
  if (facts.named_by_reference) return 'a script, recipe, plan or certificate names this file; it is KEEP';
  if (facts.record_file) return 'a log, receipt, certificate or labels file is KEEP';
  if (facts.policy_keep_path) return 'the file is under a policy keep path; it is KEEP';
  if (facts.in_use !== null) return `the file is in use (${facts.in_use}); it is ACTIVE`;
  if (facts.run_age_days < thresholds.finished_after_days) return `its run changed ${facts.run_age_days} d ago, within ${thresholds.finished_after_days} d; it is ACTIVE`;
  return null;
}

/** One record of the shadow log: a disagreement in tier, a refused verdict or a failed natural-language call. */
export type ShadowRecord = {
  time: string, mode: 'nl' | 'shadow', id: string, path: string, logical: string, bytes: number,
  crisp_tier: RetentionTier, crisp_reason: string, nl_tier: RetentionTier | null, nl_reason: string | null,
  refused: string | null, error: string | null, served: 'crisp' | 'nl',
};
export type Classified = { id: string, tier: RetentionTier, reason: string, served: 'crisp' | 'nl' };
export type Summary = { mode: 'crisp' | 'nl' | 'shadow', items: number, agree: number, disagree: number, refused: number, errors: number };

/** Run the natural-language function on one file, in a task of its own. */
export type NlClassify = (facts: RetentionFacts, thresholds: RetentionThresholds) => Promise<RetentionVerdict>;
export const nlClassify = (runtime: NatlangRuntime): NlClassify => (facts, thresholds) =>
  runtime.run(() => classifyRetention(facts, thresholds), { name: 'ops-retention.classify' }) as Promise<RetentionVerdict>;

/**
 * Classify every item under the mode. `crisp` returns the crisp verdicts and calls no model. `shadow` serves crisp and
 * records each disagreement; `nl` serves the natural-language verdict unless `refusal` refuses it (then crisp serves).
 * `concurrency` bounds how many calls are in flight at once.
 */
export async function classifyItems(items: RetentionItem[], thresholds: RetentionThresholds, setting: PluggableSetting,
    classify: NlClassify, options: { concurrency?: number, now?: () => Date } = {}): Promise<{ classified: Classified[], records: ShadowRecord[], summary: Summary }> {
  const mode = pluggableMode(setting, 'crisp');
  const now = options.now ?? (() => new Date());
  const classified: Classified[] = new Array(items.length);
  const records: ShadowRecord[] = [];
  const summary: Summary = { mode, items: items.length, agree: 0, disagree: 0, refused: 0, errors: 0 };
  const one = async (item: RetentionItem, index: number) => {
    if (mode === 'crisp') { classified[index] = { id: item.id, ...item.crisp, served: 'crisp' }; return; }
    const seen: { verdict?: RetentionVerdict, error?: string } = {};
    const decide = pluggable<[], RetentionVerdict>({
      crisp: () => item.crisp,
      nl: async () => {
        try { seen.verdict = await classify(item.facts, thresholds); return seen.verdict; }
        catch (error) { seen.error = (error as Error)?.message ?? String(error); throw error; }
      },
    }, mode, { name: 'ops.retention.classify', serve: mode === 'shadow' ? 'crisp' : 'nl', same: (a, b) => a.tier === b.tier });
    let served: 'crisp' | 'nl' = mode === 'shadow' ? 'crisp' : 'nl';
    let verdict: RetentionVerdict;
    try { verdict = await decide(); } catch { verdict = item.crisp; served = 'crisp'; }
    const refused = seen.verdict ? refusal(item.facts, thresholds, seen.verdict) : null;
    if (served === 'nl' && refused) { verdict = item.crisp; served = 'crisp'; }
    classified[index] = { id: item.id, tier: verdict.tier, reason: verdict.reason, served };
    const agree = !seen.error && seen.verdict?.tier === item.crisp.tier;
    if (seen.error) summary.errors++;
    if (refused) summary.refused++;
    if (agree) summary.agree++; else summary.disagree++;
    if (!agree || refused) records.push({ time: now().toISOString(), mode, id: item.id, path: item.path, logical: item.logical, bytes: item.bytes,
      crisp_tier: item.crisp.tier, crisp_reason: item.crisp.reason, nl_tier: seen.verdict?.tier ?? null, nl_reason: seen.verdict?.reason ?? null,
      refused, error: seen.error ?? null, served });
  };
  const width = Math.max(1, options.concurrency ?? 4);
  let next = 0;
  await Promise.all(Array.from({ length: Math.min(width, items.length) }, async () => {
    while (next < items.length) { const index = next++; await one(items[index]!, index); }
  }));
  return { classified, records, summary };
}
