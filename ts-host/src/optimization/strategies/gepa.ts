import type { Candidate } from '../../adaptation/types.js';
import { canonical } from '../../adaptation/identity.js';
import type { SearchCandidate } from '../types.js';
import type { EvaluationSuite } from '../../evaluation/types.js';
/** Per-example frontier parent selection, derived from Ax GEPA's validation winners archive. */
export function frontierParents(population: readonly SearchCandidate[]): string[] {
  const cases = population[0]?.validation.results.map(result => result.caseId) ?? [];
  const winners: string[] = [];
  for (const caseId of cases) {
    const best = Math.max(...population.map(candidate => candidate.validation.results.find(result => result.caseId === caseId)?.quality ?? -1));
    for (const candidate of population) if (candidate.validation.results.find(result => result.caseId === caseId)?.quality === best) winners.push(candidate.id);
  }
  return winners.sort();
}
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
  if (!selectionEligible(left, policy)) return false;
  if (!selectionEligible(right, policy)) return true;
  if (left.validation.quality !== right.validation.quality) return left.validation.quality! > right.validation.quality!;
  if (!policy.tieBreak || policy.tieBreak === 'baseline') return false;
  const a = measures(left)[policy.tieBreak], b = measures(right)[policy.tieBreak];
  return a !== null && (b === null || a < b);
}
