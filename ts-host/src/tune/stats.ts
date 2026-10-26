/**
 * Paired arm comparisons clustered by source/world component (plans/HIERARCHICAL_LORA_TUNING.md §9). Roots pair by ID;
 * the resampling unit is the component, never a turn, call or correlated teacher sample. Fixed-seed cluster bootstrap.
 */
import { rootSuccess, type ArmResult } from './evaluate.js';

export type PairedComparison = {
  readonly baseline: string; readonly candidate: string;
  readonly pairs: number; readonly clusters: number; readonly unpaired: number;
  readonly baselineSuccess: number; readonly candidateSuccess: number;
  /** candidate − baseline success, averaged over paired roots. */
  readonly effect: number; readonly ci: readonly [number, number]; readonly level: number;
  /** Too few independent components for the interval to support a decision. */
  readonly underpowered: boolean;
};

/** mulberry32: small deterministic PRNG so reported intervals are reproducible. */
function random(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) >>> 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

const quantile = (sorted: readonly number[], q: number) => {
  if (!sorted.length) return NaN;
  const position = (sorted.length - 1) * q, low = Math.floor(position), high = Math.ceil(position);
  return sorted[low]! + (sorted[high]! - sorted[low]!) * (position - low);
};

export const MIN_CLUSTERS = 8;

export function pairedComparison(baseline: ArmResult, candidate: ArmResult,
  options: { seed?: number; resamples?: number; level?: number; minClusters?: number } = {}): PairedComparison {
  const byId = new Map(baseline.roots.map(root => [root.id, root]));
  const clusters = new Map<string, { baseline: number[]; candidate: number[] }>();
  let unpaired = 0;
  for (const root of candidate.roots) {
    const other = byId.get(root.id);
    if (!other) { unpaired++; continue; }
    const cluster = clusters.get(root.component) ?? { baseline: [], candidate: [] };
    cluster.baseline.push(rootSuccess(other)); cluster.candidate.push(rootSuccess(root));
    clusters.set(root.component, cluster);
  }
  unpaired += baseline.roots.filter(root => !candidate.roots.some(item => item.id === root.id)).length;
  const groups = [...clusters.values()];
  const sums = (selected: typeof groups) => {
    let b = 0, c = 0, n = 0;
    for (const group of selected) { n += group.baseline.length; b += group.baseline.reduce((s, v) => s + v, 0); c += group.candidate.reduce((s, v) => s + v, 0); }
    return { b: n ? b / n : 0, c: n ? c / n : 0, n };
  };
  const total = sums(groups), level = options.level ?? 0.95, resamples = options.resamples ?? 2000;
  const draw = random(options.seed ?? 20261026), effects: number[] = [];
  if (groups.length) for (let i = 0; i < resamples; i++) {
    const sample = Array.from({ length: groups.length }, () => groups[Math.floor(draw() * groups.length)]!);
    const value = sums(sample); effects.push(value.c - value.b);
  }
  effects.sort((a, b) => a - b);
  return { baseline: baseline.arm, candidate: candidate.arm, pairs: total.n, clusters: groups.length, unpaired,
    baselineSuccess: total.b, candidateSuccess: total.c, effect: total.c - total.b,
    ci: [quantile(effects, (1 - level) / 2), quantile(effects, 1 - (1 - level) / 2)], level,
    underpowered: groups.length < (options.minClusters ?? MIN_CLUSTERS) };
}
