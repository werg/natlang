/**
 * Measured placement and subtree discovery (plans/HIERARCHICAL_LORA_TUNING.md §5.2).
 *
 * Every candidate is a refit, not a zeroed copy: removing nodes sets their manifest scale to 0 and the remaining
 * adapters are retrained on the same data receipts. Candidates are measured on DEVELOPMENT roots only, paired by root
 * against the unpruned stack, with component-clustered intervals. The accepted candidate is the cheapest whose point
 * success loss is within the declared envelope; the whole success/cost frontier is reported. Deepest-first is a search
 * order, not an attribution theorem; ambiguous single removals are retried together with their siblings.
 *
 * Which nodes to try next is policy: `propose` is the pluggable seam (crisp default: deepest first). The arithmetic,
 * refits and measurements are crisp.
 */
import { existsSync, mkdirSync, statSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { sealManifest, type WeightManifest } from '../adaptation/weights/manifest.js';
import { releaseFile, tuneDirectory } from '../adaptation/weights/release.js';
import { armSummary, type ArmResult } from './evaluate.js';
import { pairedComparison, type PairedComparison } from './stats.js';
import { scaffoldManifest } from './collect.js';
import { evaluateRelease, saveArm, trainRelease, type TuneData, type TuneOptions } from './pipeline.js';
import { DEFAULT_PLACEMENT, type PlacementSettings } from './settings.js';

export type ServingCost = { readonly activeRanks: number; readonly exportedBytes: number; readonly distinctStacks: number;
  readonly meanMs: number; readonly total: number };
export type Candidate = { readonly kind: 'unpruned' | 'removal' | 'tree-roots'; readonly removed: readonly string[];
  readonly treeRoots: readonly string[]; readonly release: string; readonly success: number; readonly cost: ServingCost;
  readonly comparison: PairedComparison | null; readonly withinEnvelope: boolean;
  /** A stack identity some measured call really ran with the tuned stack (activation readiness evidence). */
  readonly exercised: string | null; readonly meanMs: number };
export type PlacementResult = { readonly schema: 'natlang.tune-placement/v1'; readonly reference: string;
  readonly settings: PlacementSettings; readonly frontier: readonly Candidate[]; readonly selected: Candidate;
  readonly ambiguous: readonly string[] };

/** Serving cost of a release under its measured arm: summed active ranks over bindings' stacks, exported bytes of
 * served nodes, distinct non-empty stacks, measured mean root latency. */
export function servingCost(root: string, manifest: WeightManifest, arm: ArmResult, weights: PlacementSettings['costWeights']): ServingCost {
  const artifacts = new Map(manifest.artifacts.map(item => [item.node, item]));
  const live = (node: string) => artifacts.has(node) && (manifest.scales[node] ?? 1) !== 0;
  let activeRanks = 0;
  const stacks = new Set<string>();
  for (const binding of manifest.bindings) {
    const nodes = binding.ancestry.filter(live);
    activeRanks += nodes.reduce((sum, node) => sum + artifacts.get(node)!.rank, 0);
    if (nodes.length) stacks.add(nodes.join('|'));
  }
  const directory = dirname(releaseFile(root, manifest.digest));
  let exportedBytes = 0;
  for (const item of manifest.artifacts) if (live(item.node) && item.exports?.gguf) {
    const path = join(directory, item.exports.gguf.file);
    if (existsSync(path)) exportedBytes += statSync(path).size;
  }
  const meanMs = armSummary(arm).meanMs;
  return { activeRanks, exportedBytes, distinctStacks: stacks.size, meanMs,
    total: activeRanks * weights.rank + exportedBytes * weights.bytes + stacks.size * weights.stack + meanMs * weights.latencyMs };
}

const depth = (manifest: WeightManifest, node: string): number => {
  const parents = new Map(manifest.tree.map(item => [item.id, item.parent]));
  let count = 0;
  for (let cursor = parents.get(node); cursor; cursor = parents.get(cursor)) count++;
  return count;
};

/** The crisp default proposal order: trained nodes deepest first, then by ID. */
export function deepestFirst(manifest: WeightManifest, removed: ReadonlySet<string>): string[] {
  return manifest.artifacts.map(item => item.node).filter(node => !removed.has(node))
    .sort((a, b) => depth(manifest, b) - depth(manifest, a) || a.localeCompare(b));
}

function withScales(scaffold: WeightManifest, removed: readonly string[]): WeightManifest {
  const { digest: _digest, ...rest } = scaffold;
  return sealManifest({ ...rest, scales: { ...scaffold.scales, ...Object.fromEntries(removed.map(node => [node, 0])) } });
}

export async function runPlacement(options: TuneOptions, data: TuneData, reference: WeightManifest, config: {
  settings?: PlacementSettings; propose?: (manifest: WeightManifest, removed: ReadonlySet<string>) => string[];
  /** Author or proposer tree-root sets evaluated as additional candidates (subtree promotion). */
  treeRoots?: readonly (readonly string[])[]; log?: (line: string) => void } = {}): Promise<PlacementResult> {
  const settings = config.settings ?? options.settings.placement ?? DEFAULT_PLACEMENT, log = config.log ?? (() => {});
  const development = data.examples.filter(example => example.partition === 'development');
  if (!development.length) throw new Error('placement needs development roots; add more examples or groups so the split has some');
  const measure = async (manifest: WeightManifest, arm: string) => {
    const result = await evaluateRelease(options, data, manifest, development, arm, 'development');
    saveArm(options.root, result);
    return result;
  };
  const referenceArm = await measure(reference, 'hierarchy');
  const candidate = (kind: Candidate['kind'], removed: readonly string[], treeRoots: readonly string[], manifest: WeightManifest, arm: ArmResult): Candidate => {
    const comparison = kind === 'unpruned' ? null : pairedComparison(referenceArm, arm);
    const success = armSummary(arm).success;
    return { kind, removed, treeRoots, release: manifest.digest, success, cost: servingCost(options.root, manifest, arm, settings.costWeights),
      comparison, withinEnvelope: comparison ? -comparison.effect <= settings.maxSuccessLoss : true,
      exercised: arm.roots.flatMap(item => item.stacks).find(stack => stack.effective > 0 && stack.status === 'ready')?.identity ?? null,
      meanMs: armSummary(arm).meanMs };
  };
  const frontier: Candidate[] = [candidate('unpruned', [], [], reference, referenceArm)];
  let current = frontier[0]!;
  const removed = new Set<string>(), ambiguous: string[] = [];
  const tryRemoval = async (nodes: readonly string[]) => {
    const set = [...removed, ...nodes].sort();
    log(`natlang tune: placement candidate without ${nodes.join(', ')}`);
    const { manifest } = await trainRelease(options, data, { manifest: withScales(data.scaffold, set), label: `placement:-${set.join(',')}` }, [], log);
    const entry = candidate('removal', set, [], manifest, await measure(manifest, `placement-${frontier.length}`));
    frontier.push(entry);
    return entry;
  };
  for (const node of (config.propose ?? deepestFirst)(reference, removed)) {
    const entry = await tryRemoval([node]);
    if (entry.withinEnvelope && entry.cost.total < current.cost.total) { removed.add(node); current = entry; continue; }
    const [low, high] = entry.comparison!.ci;
    if (-high <= settings.maxSuccessLoss && settings.maxSuccessLoss <= -low || entry.comparison!.underpowered) ambiguous.push(node);
  }
  // Interacting groups: ambiguous siblings removed together.
  const parent = new Map(reference.tree.map(item => [item.id, item.parent]));
  const groups = new Map<string, string[]>();
  for (const node of ambiguous) if (!removed.has(node)) groups.set(parent.get(node) ?? '', [...groups.get(parent.get(node) ?? '') ?? [], node]);
  for (const group of groups.values()) if (group.length > 1) {
    const entry = await tryRemoval(group);
    if (entry.withinEnvelope && entry.cost.total < current.cost.total) { group.forEach(node => removed.add(node)); current = entry; }
  }
  for (const roots of config.treeRoots ?? []) {
    log(`natlang tune: placement candidate with tree roots ${roots.join(', ')}`);
    const scaffold = scaffoldManifest(options.program, options.student.base, { treeRoots: roots });
    const { manifest } = await trainRelease(options, data, { manifest: scaffold, label: `tree-roots:${roots.join(',')}` }, [], log);
    frontier.push(candidate('tree-roots', [], roots, manifest, await measure(manifest, `tree-roots-${frontier.length}`)));
  }
  const selected = frontier.filter(item => item.withinEnvelope).sort((a, b) => a.cost.total - b.cost.total)[0] ?? frontier[0]!;
  const result: PlacementResult = { schema: 'natlang.tune-placement/v1', reference: reference.digest, settings, frontier, selected, ambiguous };
  const directory = join(tuneDirectory(options.root), 'placement');
  mkdirSync(directory, { recursive: true });
  writeFileSync(join(directory, `${reference.digest}.json`), JSON.stringify(result, null, 2));
  return result;
}
