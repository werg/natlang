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
