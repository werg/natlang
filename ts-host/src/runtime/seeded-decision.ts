import { createHash } from 'node:crypto';
const hash = (...parts: unknown[]) => createHash('sha256').update(JSON.stringify(parts)).digest('hex');
/** A 31-bit seed derived from its named parts. */
/** Seeded categorical draw from softmax(logProbs / temperature) (the decision readout's exploration law). */
export function sampleOption(logProbs: readonly number[], temperature: number, seed: number): number {
  if (!(temperature > 0) || !Number.isFinite(temperature)) throw new Error('decision sampling needs a positive temperature');
  const top = Math.max(...logProbs.map(value => value / temperature));
  const weights = logProbs.map(value => Math.exp(value / temperature - top));
  const total = weights.reduce((sum, value) => sum + value, 0);
  // Mulberry32 from the derived seed: deterministic per (rollout, call, turn).
  let state = seed >>> 0;
  state = (state + 0x6d2b79f5) >>> 0;
  let mixed = Math.imul(state ^ (state >>> 15), state | 1);
  mixed ^= mixed + Math.imul(mixed ^ (mixed >>> 7), mixed | 61);
  let draw = (((mixed ^ (mixed >>> 14)) >>> 0) / 4294967296) * total;
  for (let index = 0; index < weights.length; index++) { draw -= weights[index]!; if (draw < 0) return index; }
  return weights.length - 1;
}

export const derivedSeed = (...parts: unknown[]) => Number.parseInt(hash('natlang.tune-rollout-seed/1', ...parts).slice(0, 8), 16) & 0x7fffffff;
