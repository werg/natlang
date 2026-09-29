import type { MetricResult, EvaluationBatch, EvaluationResult } from './types.js';
import { fingerprint } from '../adaptation/identity.js';
export function validateMetrics(input: MetricResult, required: readonly string[] = []): MetricResult {
  if (!input || !Number.isFinite(input.quality) || input.quality < 0 || input.quality > 1) throw new Error('quality must be finite and in [0,1]');
  if (Object.values(input.gates ?? {}).some(value => typeof value !== 'boolean')) throw new Error('gates must be boolean');
  if (Object.values(input.metrics ?? {}).some(value => !Number.isFinite(value))) throw new Error('metrics must be finite');
  if (required.some(key => !Object.hasOwn(input.gates ?? {}, key))) throw new Error('required regression gate missing');
  if (input.feedback !== undefined && typeof input.feedback !== 'string') throw new Error('feedback must be text');
  if (input.judge && !['accepted', 'rejected', 'uncertain', 'unavailable'].includes(input.judge)) throw new Error('malformed judge verdict');
  return input;
}
export function summarize(results: readonly EvaluationResult[]): EvaluationBatch {
  const sorted = [...results].sort((a, b) => a.id.localeCompare(b.id));
  const valid = sorted.length > 0 && sorted.every(result => result.status === 'scored' && result.quality !== null);
  return { results: sorted, quality: valid ? sorted.reduce((sum, result) => sum + result.quality!, 0) / sorted.length : null,
    gatesPassed: valid && sorted.every(result => Object.values(result.gates).every(Boolean)), digest: fingerprint(sorted) };
}
