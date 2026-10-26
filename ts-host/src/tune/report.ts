/**
 * The tune report (plans/HIERARCHICAL_LORA_TUNING.md §9): every evaluated arm of a partition, paired against the
 * untuned base with component-clustered intervals, with strata, macro and traffic-weighted success, costs and stack
 * fidelity. It reads saved evaluations, placement results and sweep journals; it runs nothing.
 */
import { existsSync, readdirSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { tuneDirectory } from '../adaptation/weights/release.js';
import { armSummary, rootSuccess, type ArmResult, type RootOutcome } from './evaluate.js';
import { pairedComparison, type PairedComparison } from './stats.js';
import { readJournal, type SweepCell } from './sweep.js';
import type { PlacementResult } from './placement.js';
import type { TuneReport } from './pipeline.js';

/** §9 arm names; an arm saved under another name is listed as is. */
export const ARM_ORDER = ['untuned', 'program-single', 'flat-lambda', 'hierarchy', 'placed', 'families', 'distilled-base', 'teacher'];

const rate = (roots: readonly RootOutcome[]) => roots.length ? roots.reduce((sum, root) => sum + rootSuccess(root), 0) / roots.length : null;

export type ArmRow = ReturnType<typeof armSummary> & {
  readonly macro: number | null; readonly real: number | null; readonly synthetic: number | null;
  readonly oracleBacked: number | null; readonly weak: number | null; readonly versusUntuned: PairedComparison | null };

export type TuneOverview = { readonly partition: string; readonly arms: readonly ArmRow[]; readonly placement: PlacementResult | null;
  readonly sweeps: Readonly<Record<string, readonly SweepCell[]>>; readonly latest: TuneReport | null; readonly manualActions: 0;
  /** Function names whose roots are too few for a reliable per-function estimate. */
  readonly lowData: readonly string[] };

export function armRow(arm: ArmResult, untuned?: ArmResult): ArmRow {
  const byFunction = new Map<string, RootOutcome[]>();
  for (const root of arm.roots) byFunction.set(root.function, [...byFunction.get(root.function) ?? [], root]);
  const macro = byFunction.size ? [...byFunction.values()].reduce((sum, roots) => sum + rate(roots)!, 0) / byFunction.size : null;
  return { ...armSummary(arm), macro, real: rate(arm.roots.filter(root => root.source === 'real')),
    synthetic: rate(arm.roots.filter(root => root.source === 'synthetic')),
    oracleBacked: rate(arm.roots.filter(root => root.oracle)), weak: rate(arm.roots.filter(root => !root.oracle)),
    versusUntuned: untuned && arm !== untuned ? pairedComparison(untuned, arm) : null };
}

export function loadOverview(root: string, partition = 'development'): TuneOverview {
  const base = tuneDirectory(root), directory = join(base, 'evaluations', partition);
  const arms = existsSync(directory) ? readdirSync(directory).filter(file => file.endsWith('.json'))
    .map(file => JSON.parse(readFileSync(join(directory, file), 'utf8')) as ArmResult) : [];
  const rank = (name: string) => { const index = ARM_ORDER.indexOf(name); return index < 0 ? ARM_ORDER.length : index; };
  arms.sort((a, b) => rank(a.arm) - rank(b.arm) || a.arm.localeCompare(b.arm));
  const untuned = arms.find(arm => arm.arm === 'untuned');
  const placementDir = join(base, 'placement');
  const placements = existsSync(placementDir) ? readdirSync(placementDir).sort().map(file => JSON.parse(readFileSync(join(placementDir, file), 'utf8')) as PlacementResult) : [];
  const sweepsDir = join(base, 'sweeps');
  const sweeps = existsSync(sweepsDir) ? Object.fromEntries(readdirSync(sweepsDir).map(name => [name, readJournal(join(sweepsDir, name, 'journal.jsonl'))])) : {};
  const reports = join(base, 'reports');
  const latest = existsSync(reports) ? readdirSync(reports).sort().at(-1) : undefined;
  const counts = new Map<string, number>();
  for (const arm of arms.slice(0, 1)) for (const item of arm.roots) counts.set(item.function, (counts.get(item.function) ?? 0) + 1);
  return { partition, arms: arms.map(arm => armRow(arm, untuned)), placement: placements.at(-1) ?? null, sweeps,
    latest: latest ? JSON.parse(readFileSync(join(reports, latest), 'utf8')) as TuneReport : null, manualActions: 0,
    lowData: [...counts].filter(([, n]) => n < 5).map(([name]) => name).sort() };
}

const pct = (value: number | null) => value === null ? '-' : `${(value * 100).toFixed(1)}%`;

export function formatOverview(overview: TuneOverview): string {
  const lines = [`arms on ${overview.partition} roots (success: oracle where declared, else completion):`];
  lines.push('  arm | roots | success | macro | real | synthetic | oracle | weak | vs untuned [CI] | mean ms | tokens | degraded/fallback calls | distinct stacks');
  for (const arm of overview.arms) {
    const versus = arm.versusUntuned ? `${(arm.versusUntuned.effect * 100).toFixed(1)}pt [${(arm.versusUntuned.ci[0] * 100).toFixed(1)}, ${(arm.versusUntuned.ci[1] * 100).toFixed(1)}]` +
      `${arm.versusUntuned.underpowered ? ` underpowered (${arm.versusUntuned.clusters} components)` : ''}` : '-';
    lines.push(`  ${arm.arm} | ${arm.roots} | ${pct(arm.success)} | ${pct(arm.macro)} | ${pct(arm.real)} | ${pct(arm.synthetic)} | ${pct(arm.oracleBacked)} | ${pct(arm.weak)} | ` +
      `${versus} | ${arm.meanMs.toFixed(0)} | ${arm.tokens} | ${arm.degradedCalls}/${arm.fallbackCalls} of ${arm.calls} | ${arm.distinctStacks}`);
  }
  if (!overview.arms.length) lines.push('  (no evaluations yet: natlang tune <program> --evaluate development)');
  if (overview.placement) {
    const { selected, frontier } = overview.placement;
    lines.push(`placement: selected ${selected.kind}${selected.removed.length ? ` without ${selected.removed.join(', ')}` : ''} ` +
      `(success ${pct(selected.success)}, cost ${selected.cost.total.toFixed(2)}) among ${frontier.length} candidates`);
    for (const item of frontier) lines.push(`  ${item.kind} ${item.removed.join(',') || item.treeRoots.join(',') || '-'}: success ${pct(item.success)}, ` +
      `cost ${item.cost.total.toFixed(2)} (ranks ${item.cost.activeRanks}, bytes ${item.cost.exportedBytes}, stacks ${item.cost.distinctStacks})${item.withinEnvelope ? '' : ' outside envelope'}`);
  }
  for (const [name, cells] of Object.entries(overview.sweeps)) {
    const done = cells.filter(cell => cell.status === 'done');
    lines.push(`sweep ${name}: ${done.length} cells done, ${cells.filter(cell => cell.status === 'failed').length} failed, ` +
      `${cells.filter(cell => cell.status === 'skipped' || cell.status === 'unsupported').length} skipped/unsupported`);
  }
  if (overview.latest) lines.push(`time to first tuned call: ${(overview.latest.totalMs / 1000).toFixed(1)}s (latest tune, ${overview.latest.activated ? 'activated' : 'not activated'})`);
  if (overview.lowData.length) lines.push(`too few roots for per-function estimates: ${overview.lowData.join(', ')}`);
  lines.push('manual author actions: 0');
  return lines.join('\n');
}
