import type { EvaluationBatch, EvaluationCase } from './types.js';
/** Paired changes refer to the same case, not an assumed causal attribution to a nested call. */
export function pairedChanges(baseline: EvaluationBatch, selected: EvaluationBatch) {
  return baseline.results.map(before => {
    const after = selected.results.find(result => result.caseId === before.caseId);
    return { caseId: before.caseId, baseline: before.quality, selected: after?.quality ?? null,
      delta: before.quality !== null && after?.quality !== null && after?.quality !== undefined ? after.quality - before.quality : null,
      baselineOutcome: before.outcome.kind, selectedOutcome: after?.outcome.kind ?? null };
  });
}
export function evaluationSummary(batch: EvaluationBatch, cases: readonly EvaluationCase[]) {
  const outcomes: Record<string, number> = {}, gates: Record<string, { passed: number; failed: number }> = {};
  const slices: Record<string, { count: number; quality: number | null }> = {};
  for (const result of batch.results) {
    const outcome = result.outcome.error?.outcome ?? result.outcome.kind; outcomes[outcome] = (outcomes[outcome] ?? 0) + 1;
    for (const [key, passed] of Object.entries(result.gates)) { gates[key] ??= { passed: 0, failed: 0 }; gates[key][passed ? 'passed' : 'failed']++; }
    for (const slice of cases.find(testCase => testCase.id === result.caseId)?.slices ?? []) {
      slices[slice] ??= { count: 0, quality: 0 }; slices[slice].count++;
      if (result.quality === null) slices[slice].quality = null; else if (slices[slice].quality !== null) slices[slice].quality! += result.quality;
    }
  }
  for (const slice of Object.values(slices)) if (slice.quality !== null) slice.quality /= slice.count;
  return { quality: batch.quality, cases: batch.results.length, gatesPassed: batch.gatesPassed, outcomes, gates, slices,
    meanLatencyMs: batch.results.length ? batch.results.reduce((sum, result) => sum + result.latencyMs, 0) / batch.results.length : null,
    modelCalls: batch.results.reduce((sum, result) => sum + result.usage.modelCalls, 0),
    coverage: [...new Set(batch.results.flatMap(result => [...result.coverage]))].sort() };
}
export function replicateUncertainty(batches: readonly EvaluationBatch[]) {
  const values = batches.map(batch => batch.quality).filter((value): value is number => value !== null);
  const mean = values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
  const variance = values.length > 1 ? values.reduce((sum, value) => sum + (value - mean!) ** 2, 0) / (values.length - 1) : null;
  return { replicates: batches.length, completeReplicates: values.length, mean, standardDeviation: variance === null ? null : Math.sqrt(variance),
    standardError: variance === null ? null : Math.sqrt(variance / values.length) };
}
