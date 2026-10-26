/**
 * Incremental builds (plans/HIERARCHICAL_LORA_TUNING.md §8 "Incremental execution").
 *
 * Against the active release, an edit dirties some scope nodes (keys.ts `invalidation`). Instead of refitting the whole
 * tree:
 *
 * - a **candidate** trains only the dirty lambda leaves and nodes the release does not have yet, against every other
 *   node pinned at its exact released artifact. Dirty shared ancestors stay pinned at their older versions: the
 *   receipt records which older binding semantics each such parent was trained on, so the candidate is never
 *   advertised as a refreshed joint build;
 * - a **shared refit** trains the dirty nodes and every residual fit against them (their dependents), the rest pinned;
 * - a **full** joint refit runs after `tune.incremental.fullRefitEvery` incremental builds, on `--full`, or when the
 *   recipe changed (pinned residuals of another rank or parameterization cannot compose).
 *
 * Trained nodes the release already had warm-start from their released artifact; their optimizer moments continue
 * only where the trainer's exact continuation contract holds. Only rows whose binding stack contains a trained node
 * are fitted. Families with their learned gates and the soft states of untouched bindings carry over unchanged.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { currentRelease, readRelease, tuneDirectory } from '../adaptation/weights/release.js';
import type { WeightManifest } from '../adaptation/weights/manifest.js';
import { canonical } from '../adaptation/identity.js';
import { invalidation } from './keys.js';
import { trainRelease, type ReleaseSelector, type TrainVariant, type TuneOptions, type TuneReport } from './pipeline.js';

export type BuildKind = 'full' | 'candidate' | 'shared-refit';
export type IncrementalSettings = {
  /** Run a full joint refit after this many incremental builds (default 5). */
  readonly fullRefitEvery?: number;
  /** A dirty shared ancestor: build a candidate first and then the shared refit (default), or only the shared refit. */
  readonly sharedEdits?: 'candidate-then-refit' | 'refit';
};
export type IncrementalPlan = {
  readonly kind: BuildKind;
  /** Nodes this build trains (warm-started where the release had them). */
  readonly trained: readonly string[];
  readonly pinned: Readonly<Record<string, string>>;
  readonly warm: Readonly<Record<string, { readonly artifact: string; readonly optimizer?: string }>>;
  /** Pinned dirty nodes and the older binding semantics their pinned version was trained on. */
  readonly stalePins: Readonly<Record<string, Readonly<Record<string, string>>>>;
  /** Bindings whose rows are fitted: those whose stack contains a trained node. */
  readonly bindings: readonly string[];
};
export type Previous = { readonly manifest: WeightManifest; readonly directory: string };
export type BuildStep = { readonly kind: BuildKind; readonly previous: Previous;
  /** A candidate this shared refit completes: its trained leaves seed the refit. */
  readonly seed?: Previous };

/** Which nodes a candidate or a shared refit trains and pins, from the active release and the current scaffold. */
export function planIncremental(previous: WeightManifest, current: WeightManifest, kind: 'candidate' | 'shared-refit'): IncrementalPlan {
  const change = invalidation(previous, current);
  const released = new Map(previous.artifacts.map(item => [item.node, item.artifact]));
  const training = previous.receipts.training as { optimizer_states?: Record<string, string | null> } | undefined;
  const level = new Map(current.tree.map(node => [node.id, node.level]));
  // Nodes the release has no residual for were either untrained (rank 0 at their level: the recipe is unchanged) or
  // are new; only new ones and released residuals are candidates for training.
  const known = new Set(previous.tree.map(node => node.id));
  const isNew = (node: string) => !known.has(node);
  const trained = new Set(current.tree.map(node => node.id).filter(isNew));
  for (const node of change.dirty) if (released.has(node) && (kind === 'shared-refit' || level.get(node) === 'lambda')) trained.add(node);
  if (kind === 'shared-refit') {
    // Residuals fit against a dirty residual-bearing parent depend on it; a dirty node without a residual (rank 0 at
    // its level) shaped nothing below it.
    const children = new Map<string, string[]>();
    for (const node of current.tree) if (node.parent) children.set(node.parent, [...children.get(node.parent) ?? [], node.id]);
    const visit = (node: string): void => { for (const child of children.get(node) ?? []) { if (released.has(child)) trained.add(child); visit(child); } };
    for (const node of change.dirty) if (released.has(node)) visit(node);
  }
  const pinned: Record<string, string> = {}, warm: Record<string, { artifact: string; optimizer?: string }> = {};
  for (const node of current.tree) {
    const artifact = released.get(node.id);
    if (!artifact) continue;
    if (trained.has(node.id)) {
      const optimizer = training?.optimizer_states?.[node.id];
      warm[node.id] = { artifact, ...(optimizer ? { optimizer } : {}) };
    } else pinned[node.id] = artifact;
  }
  const stalePins: Record<string, Record<string, string>> = {};
  for (const node of change.dirty) if (pinned[node])
    stalePins[node] = Object.fromEntries(previous.bindings.filter(entry => entry.ancestry.includes(node)).map(entry => [entry.id, entry.semanticHash]));
  const bindings = current.bindings.filter(entry => entry.ancestry.some(node => trained.has(node))).map(entry => entry.id).sort();
  return { kind, trained: [...trained].sort(), pinned, warm, stalePins, bindings };
}

/** The variant an incremental build trains: the current scaffold with the release's families and learned gates. */
export function carried(previous: Previous, current: WeightManifest): WeightManifest {
  const bindings = new Set(current.bindings.map(entry => entry.id));
  return { ...current, families: previous.manifest.families, memberships: previous.manifest.memberships.filter(item => bindings.has(item.binding)) };
}

const releaseOf = (root: string, digest: string): Previous =>
  ({ manifest: readRelease(root, digest), directory: join(tuneDirectory(root), 'releases', digest) });

/**
 * The builds an `--incremental` run makes next. A candidate whose pinned shared parents are stale is completed by its
 * shared refit, planned against the release the candidate pinned (the candidate's own bindings already carry the new
 * semantics, so comparing against it would find nothing to refit).
 */
export function nextBuilds(root: string, active: Previous, current: WeightManifest, options: { full?: boolean;
  settings?: IncrementalSettings; recipe: unknown }): { steps: BuildStep[]; reason: string } | null {
  const receipts = active.manifest.receipts as { build?: string; pinnedFrom?: string; stalePins?: Record<string, unknown> };
  if (!options.full && receipts.build === 'candidate' && receipts.pinnedFrom && Object.keys(receipts.stalePins ?? {}).length)
    return { steps: [{ kind: 'shared-refit', previous: releaseOf(root, receipts.pinnedFrom), seed: active }],
      reason: 'the active candidate pinned stale shared parents' };
  if (!options.full && invalidation(active.manifest, current).fresh) return null;
  const { kinds, reason } = incrementalKinds(active.manifest, current, { ...options, sinceFull: readCounter(root).sinceFull });
  return { steps: kinds.map(kind => ({ kind, previous: active })), reason };
}

/** The build kinds for a change against the active release, from the counter and the recipe. */
export function incrementalKinds(previous: WeightManifest, current: WeightManifest, options: { full?: boolean;
  sinceFull: number; settings?: IncrementalSettings; recipe: unknown }): { kinds: BuildKind[]; reason: string } {
  const every = options.settings?.fullRefitEvery ?? 5;
  if (options.full) return { kinds: ['full'], reason: '--full' };
  if (canonical(previous.receipts.recipe ?? null) !== canonical(options.recipe ?? null))
    return { kinds: ['full'], reason: 'the recipe changed since the active release' };
  if (options.sinceFull + 1 >= every) return { kinds: ['full'], reason: `${options.sinceFull + 1} incremental builds reach tune.incremental.fullRefitEvery (${every})` };
  const candidate = planIncremental(previous, current, 'candidate'), shared = planIncremental(previous, current, 'shared-refit');
  if (!shared.trained.length) return { kinds: ['full'], reason: 'no node changed semantics; the full build reuses its training receipt' };
  const sharedDirty = shared.trained.some(node => !candidate.trained.includes(node));
  if (!sharedDirty) return { kinds: ['candidate'], reason: 'only lambda leaves changed' };
  const kinds: BuildKind[] = options.settings?.sharedEdits === 'refit' || !candidate.bindings.length ? ['shared-refit'] : ['candidate', 'shared-refit'];
  return { kinds, reason: 'a shared ancestor changed' };
}

/** A release selector for one incremental build kind against the active release. */
export function incrementalSelector(options: TuneOptions, step: BuildStep): ReleaseSelector {
  const { kind, previous, seed } = step;
  return async (data, timings, log) => {
    if (kind === 'full') {
      const variant: TrainVariant = { manifest: carried(previous, data.scaffold), receipts: { build: 'full' } };
      return (await trainRelease(options, data, variant, timings, log)).manifest;
    }
    const plan = planIncremental(previous.manifest, data.scaffold, kind);
    // A completed candidate's freshly trained leaves seed its shared refit.
    const seeded = new Map((seed?.manifest.artifacts ?? []).map(item => [item.node, item.artifact]));
    const seedTraining = seed?.manifest.receipts.training as { optimizer_states?: Record<string, string | null> } | undefined;
    const candidateTrained = new Set((seed?.manifest.receipts.trained as string[] | undefined) ?? []);
    const warm = Object.fromEntries(plan.trained.flatMap(node => {
      if (candidateTrained.has(node) && seeded.has(node)) {
        const optimizer = seedTraining?.optimizer_states?.[node];
        return [[node, { artifact: seeded.get(node)!, ...(optimizer ? { optimizer } : {}) }]];
      }
      return plan.warm[node] ? [[node, plan.warm[node]!]] : [];
    }));
    const rows = data.trainRows.filter(row => row.binding && plan.bindings.includes(row.binding));
    if (!rows.length) throw new Error(`the ${kind} build has no training rows for its changed bindings; collect data for them or run --full`);
    log(`natlang tune: ${kind} build trains ${plan.trained.length} node(s), pins ${Object.keys(plan.pinned).length}, ` +
      `warm-starts ${Object.keys(warm).length}; ${rows.length} row(s) of ${plan.bindings.length} binding(s)`);
    const trainedBindings = new Set(plan.bindings);
    const variant: TrainVariant = { manifest: carried(previous, data.scaffold), rows, label: kind, pinned: plan.pinned, warm,
      artifactDirs: [join(previous.directory, 'artifacts'), ...(seed ? [join(seed.directory, 'artifacts')] : [])],
      carrySoftStates: { states: previous.manifest.softStates.filter(item => !trainedBindings.has(item.binding)), directory: previous.directory },
      receipts: { build: kind, pinnedFrom: previous.manifest.digest, trained: plan.trained, pinned: plan.pinned, stalePins: plan.stalePins,
        ...(seed ? { completes: seed.manifest.digest } : {}) } };
    return (await trainRelease(options, data, variant, timings, log)).manifest;
  };
}

/**
 * Run the next incremental builds (`natlang tune --incremental`, watch mode). `build` validates and activates one
 * step's release (`tuneProgram` with `incrementalSelector`). Undefined when there is no active release; null when the
 * active release matches the program.
 */
export async function runIncrementalBuilds(root: string, current: WeightManifest, programBuildHash: string,
  options: { full?: boolean; settings?: IncrementalSettings; recipe: unknown; log?: (line: string) => void },
  build: (step: BuildStep) => Promise<TuneReport>): Promise<TuneReport[] | null | undefined> {
  const active = currentRelease(root);
  if (!active?.active) return undefined;
  const previous = { manifest: active.manifest, directory: active.directory };
  let next = nextBuilds(root, previous, current, options);
  // Non-instruction sources (helpers, services, types) change trajectories without changing any binding's semantics.
  if (!next && active.manifest.program.buildHash !== programBuildHash) next = { steps: [{ kind: 'full', previous }], reason: 'non-instruction sources changed' };
  if (!next) return null;
  options.log?.(`natlang tune: ${next.steps.map(step => step.kind).join(' then ')} build (${next.reason})`);
  const reports: TuneReport[] = [];
  for (const step of next.steps) {
    // A shared refit right after its candidate is seeded by the candidate just activated.
    const latest = currentRelease(root);
    const report = await build(step.kind === 'shared-refit' && reports.length && latest
      ? { ...step, seed: { manifest: latest.manifest, directory: latest.directory } } : step);
    reports.push(report);
    if (!report.activated || !report.release) break;
    recordBuild(root, step.kind, report.release);
  }
  return reports;
}

/** Incremental builds since the last full refit (advanced only by activated builds). */
export type IncrementalCounter = { readonly schema: 'natlang.tune-incremental/v1'; readonly sinceFull: number; readonly last: string | null };
const counterPath = (root: string) => join(tuneDirectory(root), 'incremental.json');
export function readCounter(root: string): IncrementalCounter {
  const path = counterPath(root);
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) : { schema: 'natlang.tune-incremental/v1', sinceFull: 0, last: null };
}
export function recordBuild(root: string, kind: BuildKind, release: string): IncrementalCounter {
  const previous = readCounter(root);
  // A candidate followed by its shared refit counts as one incremental build.
  const counter: IncrementalCounter = { schema: 'natlang.tune-incremental/v1', last: kind,
    sinceFull: kind === 'full' ? 0 : kind === 'shared-refit' && previous.last === 'candidate' ? previous.sinceFull : previous.sinceFull + 1 };
  mkdirSync(tuneDirectory(root), { recursive: true });
  writeFileSync(counterPath(root) + '.tmp', JSON.stringify({ ...counter, release }, null, 1) + '\n');
  renameSync(counterPath(root) + '.tmp', counterPath(root));
  return counter;
}
