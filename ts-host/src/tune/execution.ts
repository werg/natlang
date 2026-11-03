/** Shared invocation limits for tune collection, paired evaluation, repairs and whole-graph rollouts. */
import { createHash } from 'node:crypto';
import type { ModelConfig, ModelDriver } from '../runtime/runtime.js';
import { canonicalJson } from '../package/archive.js';

export type TuneExecution = Pick<ModelConfig, 'maxTurns' | 'maxTokens' | 'maxSeconds' | 'maxFailureRepairs'>;
export function tuneModel(driver: ModelDriver, execution?: TuneExecution): ModelConfig {
  for (const [key, value] of Object.entries(execution ?? {})) {
    if (!['maxTurns', 'maxTokens', 'maxSeconds', 'maxFailureRepairs'].includes(key) || typeof value !== 'number' ||
        !Number.isFinite(value) || value < (key === 'maxFailureRepairs' ? 0 : Number.MIN_VALUE) ||
        (key !== 'maxSeconds' && !Number.isInteger(value))) throw new RangeError(`invalid tune execution limit: ${key}`);
  }
  return { driver, ...execution };
}
export const executionKey = (execution?: TuneExecution): string => createHash('sha256').update(canonicalJson(execution ?? {})).digest('hex').slice(0, 16);

/** Bind the same native artifact store used by transport, including provenance reads and typed writes. */
export async function tuneRuntime(driver: ModelDriver, execution?: TuneExecution) {
  const neuralese = await driver.neuraleseRuntime?.();
  return { model: tuneModel(driver, execution), ...(neuralese ? { neuralese } : {}) };
}
/** Roots in flight per tune arm or rollout round when `tune.concurrency` is unset: the native server batches
 * concurrent decodes, so independent roots share each decode step instead of running one at a time. */
export const DEFAULT_TUNE_CONCURRENCY = 4;
export function tuneConcurrency(value: unknown): number {
  if (value === undefined) return DEFAULT_TUNE_CONCURRENCY;
  if (typeof value !== 'number' || !Number.isInteger(value) || value < 1) throw new RangeError('tune.concurrency is a positive integer');
  return value;
}

/** `fn` over `items` with at most `limit` in flight; results in input order. The first failure stops new work and is
 * rethrown once the work already started has settled. */
export async function mapBounded<T, R>(items: readonly T[], limit: number, fn: (item: T, index: number) => Promise<R>): Promise<R[]> {
  const results = new Array<R>(items.length);
  let next = 0, failure: { error: unknown } | undefined;
  const worker = async () => {
    while (!failure && next < items.length) {
      const index = next++;
      try { results[index] = await fn(items[index]!, index); }
      catch (error) { failure ??= { error }; }
    }
  };
  await Promise.all(Array.from({ length: Math.min(Math.max(1, limit), items.length) }, worker));
  if (failure) throw failure.error;
  return results;
}

export function rethrowInfrastructureFailure(error: unknown): void {
  if (error && typeof error === 'object' && ['neuralese-unsupported-backend', 'natlang-platform-unavailable']
    .includes(String((error as { code?: unknown }).code))) throw error;
}
