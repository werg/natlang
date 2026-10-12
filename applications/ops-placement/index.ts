/**
 * Job placement as a pluggable policy (DECOMPOSITION.md). scripts/coreweave/placement.py gathers each job's needs and
 * the fleet's facts and computes the crisp decision; placeJob.nl is the natural-language implementation of the same
 * decision. `placeItems` selects one by the mode (`crisp`, `nl`, `shadow`) through ts-host's `pluggable`, bounds
 * every natural-language decision exactly (`refusal`) and returns the records of disagreements for the shadow log.
 *
 * Launching stays in scripts/coreweave/cwctl.py, which only launches a job whose served decision is "coreweave".
 */
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableSetting } from '@natlang/node';
import placeJob from './placeJob.nl';
import type { FleetFacts, JobNeeds, PlacementDecision, PlacementItem, PlacementTarget } from './types.js';

export type * from './types.js';

export const TARGETS: readonly PlacementTarget[] = ['dgx', 'pop', 'coreweave', 'queue-dgx', 'ask'];

/**
 * The exact bound on a natural-language decision: why it may not serve, or null. CoreWeave must be ready, the job
 * must fit one GPU with the margin, stay within the parallel limit and the budget, and not need the teacher; DGX
 * must have the headroom; Pop only for Pop's lineages; moving an active lineage is always the owner's decision.
 */
export function refusal(job: JobNeeds, fleet: FleetFacts, decision: PlacementDecision): string | null {
  if (!decision || !TARGETS.includes(decision.where)) return `where ${JSON.stringify(decision?.where)} is not one of ${TARGETS.join(', ')}`;
  if (job.moves_active_lineage && decision.where !== 'ask' && job.lineage_owner !== 'pop') return 'moving an active lineage is the owner\'s decision; where is "ask"';
  if (decision.where === 'pop' && job.lineage_owner !== 'pop') return 'Pop runs only the lineages it owns';
  if (decision.where === 'dgx' && job.gpu_memory_gb > fleet.dgx_headroom_gb) return `needs ${job.gpu_memory_gb} GB, DGX headroom is ${fleet.dgx_headroom_gb} GB; it is "queue-dgx"`;
  if (decision.where !== 'coreweave') return null;
  if (!fleet.coreweave_ready) return 'CoreWeave is not ready (credential, bucket or GPU smoke missing)';
  if (job.needs_teacher) return 'the teacher on DGX is unreachable from CoreWeave';
  if (job.lineage_owner === 'pop') return 'Pop owns this lineage';
  if (job.gpu_memory_gb > fleet.coreweave_gpu_memory_gb - fleet.coreweave_fit_margin_gb) return `needs ${job.gpu_memory_gb} GB, more than one sandbox GPU allows (${fleet.coreweave_gpu_memory_gb - fleet.coreweave_fit_margin_gb} GB)`;
  if (fleet.coreweave_active >= fleet.coreweave_max_parallel) return `${fleet.coreweave_active} sandboxes already run (limit ${fleet.coreweave_max_parallel})`;
  if (fleet.coreweave_budget_left_usd !== null && (job.coreweave_cost_usd === null || job.coreweave_cost_usd > fleet.coreweave_budget_left_usd))
    return `cost ${job.coreweave_cost_usd ?? 'unknown'} USD against ${fleet.coreweave_budget_left_usd} USD left this week`;
  return null;
}

export type ShadowRecord = {
  time: string, mode: 'nl' | 'shadow', id: string, crisp_where: PlacementTarget, crisp_reason: string,
  nl_where: PlacementTarget | null, nl_reason: string | null, refused: string | null, error: string | null, served: 'crisp' | 'nl',
};
export type Placed = { id: string, where: PlacementTarget, reason: string, served: 'crisp' | 'nl' };
export type Summary = { mode: 'crisp' | 'nl' | 'shadow', items: number, agree: number, disagree: number, refused: number, errors: number };

export type NlPlace = (job: JobNeeds, fleet: FleetFacts) => Promise<PlacementDecision>;
export const nlPlace = (runtime: NatlangRuntime): NlPlace => (job, fleet) =>
  runtime.run(() => placeJob(job, fleet), { name: 'ops-placement.place' }) as Promise<PlacementDecision>;

/** Decide every item under the mode, as ops-retention's classifyItems does (same shadow and refusal semantics). */
export async function placeItems(items: PlacementItem[], setting: PluggableSetting, place: NlPlace,
    options: { now?: () => Date } = {}): Promise<{ placed: Placed[], records: ShadowRecord[], summary: Summary }> {
  const mode = pluggableMode(setting, 'crisp');
  const now = options.now ?? (() => new Date());
  const placed: Placed[] = [];
  const records: ShadowRecord[] = [];
  const summary: Summary = { mode, items: items.length, agree: 0, disagree: 0, refused: 0, errors: 0 };
  for (const item of items) {
    if (mode === 'crisp') { placed.push({ id: item.id, ...item.crisp, served: 'crisp' }); continue; }
    const seen: { decision?: PlacementDecision, error?: string } = {};
    const decide = pluggable<[], PlacementDecision>({
      crisp: () => item.crisp,
      nl: async () => {
        try { seen.decision = await place(item.job, item.fleet); return seen.decision; }
        catch (error) { seen.error = (error as Error)?.message ?? String(error); throw error; }
      },
    }, mode, { name: 'ops.placement.place', serve: mode === 'shadow' ? 'crisp' : 'nl', same: (a, b) => a.where === b.where });
    let served: 'crisp' | 'nl' = mode === 'shadow' ? 'crisp' : 'nl';
    let decision: PlacementDecision;
    try { decision = await decide(); } catch { decision = item.crisp; served = 'crisp'; }
    const refused = seen.decision ? refusal(item.job, item.fleet, seen.decision) : null;
    if (served === 'nl' && refused) { decision = item.crisp; served = 'crisp'; }
    placed.push({ id: item.id, where: decision.where, reason: decision.reason, served });
    const agree = !seen.error && seen.decision?.where === item.crisp.where;
    if (seen.error) summary.errors++;
    if (refused) summary.refused++;
    if (agree) summary.agree++; else summary.disagree++;
    if (!agree || refused) records.push({ time: now().toISOString(), mode, id: item.id, crisp_where: item.crisp.where,
      crisp_reason: item.crisp.reason, nl_where: seen.decision?.where ?? null, nl_reason: seen.decision?.reason ?? null,
      refused, error: seen.error ?? null, served });
  }
  return { placed, records, summary };
}
