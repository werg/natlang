import type { Candidate } from '../../adaptation/types.js';
import { canonical } from '../../adaptation/identity.js';
import type { SearchCandidate } from '../types.js';
import type { EvaluationSuite } from '../../evaluation/types.js';
import { better, frontier, type Member } from '../../gepa/index.js';
/**
 * The GEPA selection math (frontier, seeded draw, better, prune) is one module, src/gepa, which the program-improver
 * application also imports as `natlang:gepa`. This file adapts component-search candidates to it.
 */
export function memberOf(candidate: SearchCandidate): Member {
  return { id: candidate.id, quality: candidate.validation.quality ?? 0,
    scores: candidate.validation.results.map(result => ({ caseId: result.caseId, quality: result.quality ?? 0 })) };
}
/** The per-case winners of a population, sorted (a candidate once per case it wins). */
export function frontierParents(population: readonly SearchCandidate[]): string[] { return frontier(population.map(memberOf)); }
/** Conservative three-way composition: conflicting instruction changes are never combined. */
export function mergeCandidates(baseline: Candidate, left: Candidate, right: Candidate): Candidate | null {
  const merged: Record<string, Candidate[string]> = {};
  for (const key of Object.keys(baseline)) {
    const a = left[key]!, b = right[key]!, base = canonical(baseline[key]);
    if (canonical(a) !== base && canonical(b) !== base && canonical(a) !== canonical(b)) return null;
    merged[key] = canonical(a) !== base ? a : b;
  }
  return merged;
}
function measures(candidate: SearchCandidate) {
  const results = candidate.validation.results;
  return { modelCalls: results.reduce((sum, result) => sum + result.usage.modelCalls, 0),
    latency: results.reduce((sum, result) => sum + result.latencyMs, 0) / results.length,
    cost: results.some(result => result.usage.cost === null) ? null : results.reduce((sum, result) => sum + result.usage.cost!, 0) };
}
export function selectionEligible(candidate: SearchCandidate, policy: EvaluationSuite['selection'] = {}): boolean {
  if (!candidate.validation.gatesPassed || candidate.validation.quality === null) return false;
  const use = measures(candidate);
  return (policy.maxMeanLatencyMs === undefined || use.latency <= policy.maxMeanLatencyMs) &&
    (policy.maxModelCalls === undefined || use.modelCalls <= policy.maxModelCalls) &&
    (policy.maxCost === undefined || use.cost !== null && use.cost <= policy.maxCost);
}
export function meanBetter(left: SearchCandidate, right: SearchCandidate, policy: EvaluationSuite['selection'] = {}): boolean {
  const tie = policy.tieBreak && policy.tieBreak !== 'baseline' ? policy.tieBreak : undefined;
  const view = (candidate: SearchCandidate) => ({ candidate, quality: candidate.validation.quality });
  return better(view(left), view(right), { eligible: item => selectionEligible(item.candidate, policy),
    measure: tie ? item => measures(item.candidate)[tie] : undefined });
}
