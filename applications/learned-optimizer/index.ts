/**
 * The joint learned optimizer as a Natlang program (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). adapt.nl revises a
 * program's native instruction states, skill states and LoRA mixtures together through `iterateOn`; train.nl
 * schedules the outer training. This file is the outside world those stages act on: the `optimizer` and `trainer`
 * services, backed by the exact Python mechanism (`python -m natlang_neuralese.tune.learned service|train|...`).
 * Bundles stay host-side; the program sees their identities and exact support losses.
 */
import { spawn } from 'node:child_process';
import { mkdtemp, readFile, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import type { NatlangRuntime } from '@natlang/node';
import adaptFlow from './adapt.nl';
import trainFlow from './train.nl';
import type { AdaptationState, BundleSummary, EvidenceView, TrainingMetrics, TrainingState } from './types.js';

export type * from './types.js';

/** How to run the Python entry: an argv ending in `-m` (e.g. `[".venv/bin/python", "-m"]`) and its environment. */
export type PythonEntry = { readonly argv: readonly string[]; readonly env?: Readonly<Record<string, string>>; readonly cwd?: string };

async function runPython(entry: PythonEntry, command: string, spec: Record<string, unknown>): Promise<unknown> {
  const directory = await mkdtemp(join(tmpdir(), 'natlang-lopt-'));
  const specPath = join(directory, 'spec.json'), resultPath = join(directory, 'result.json');
  await writeFile(specPath, JSON.stringify({ ...spec, result: resultPath }));
  let tail = '';
  await new Promise<void>((resolveRun, reject) => {
    const child = spawn(entry.argv[0]!, [...entry.argv.slice(1), 'natlang_neuralese.tune.learned', command, '--spec', specPath],
      { cwd: entry.cwd ?? process.cwd(), env: { ...process.env, ...entry.env }, stdio: ['ignore', 'pipe', 'pipe'] });
    const keep = (chunk: Buffer) => { tail = (tail + chunk.toString()).slice(-4000); };
    child.stdout.on('data', keep); child.stderr.on('data', keep);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`learned optimizer ${command} exited with ${code}\n${tail.trim()}`)));
  });
  if (command !== 'service') return JSON.parse(tail.trim().split('\n').pop()!).result;
  return JSON.parse(await readFile(resultPath, 'utf8'));
}

type Bundle = { id: string; parent: string | null; provenance?: { actor?: string } };
const neuraleseId = (value: unknown): string => {
  const ref = (value as { $neuralese?: { id?: unknown } })?.$neuralese;
  if (!ref || typeof ref.id !== 'string') throw new TypeError('a digest must be a native Neuralese value');
  return ref.id;
};

/** The adaptation services over one prepared program context (`apply.Context`: a tune plan and its output dir). */
export class AdaptationServices {
  private readonly bundles = new Map<string, Bundle>();
  constructor(private readonly python: PythonEntry, private readonly plan: string, private readonly out: string) {}
  bundle(id: string): Bundle { const found = this.bundles.get(id); if (!found) throw new Error(`unknown bundle ${id}`); return found; }
  private async call(op: string, extra: Record<string, unknown>) {
    return await runPython(this.python, 'service', { plan: this.plan, out: this.out, op, ...extra }) as Record<string, any>;
  }
  private summary(result: Record<string, any>, actor: string): BundleSummary {
    const bundle = result.bundle as Bundle;
    this.bundles.set(bundle.id, bundle);
    return { id: bundle.id, supportLoss: result.supportLoss as number, parent: bundle.parent, actor };
  }
  private parent(id: string | null) { return id === null ? null : this.bundle(id); }
  readonly optimizer = {
    evidence: async (): Promise<{ view: EvidenceView; scopes: string[]; incomingLoss: number }> => {
      const result = await this.call('evidence', {});
      return { view: result.view, scopes: result.scopes, incomingLoss: result.incomingLoss };
    },
    propose: async (request: { parent: string | null; sample: boolean; seed: number; focus: string[]; digests: { binding: string; digest: unknown }[] }) =>
      this.summary(await this.call('propose', { parent: this.parent(request.parent), sample: request.sample, seed: request.seed,
        focus: request.focus, digests: request.digests.map(item => [item.binding, neuraleseId(item.digest)]) }), 'learned-optimizer'),
    refine: async (request: { parent: string | null; steps: number }) =>
      this.summary(await this.call('refine', { parent: this.parent(request.parent), steps: request.steps }), 'in-bank-refinement'),
    retrieve: async (request: { parent: string | null }) =>
      this.summary(await this.call('retrieve', { parent: this.parent(request.parent) }), 'gradient-free-mixing'),
  };
  /** The release fragment (composed scope adapters and writer states) of the selected bundle. */
  async release(id: string | null) { return this.call('release', { parent: this.parent(id) }); }
}

export const optimizerDeclaration = `/** Exact adaptation services: they score, write and compose; they never decide what to revise. */
export function evidence(): Promise<{ view: EvidenceView; scopes: string[]; incomingLoss: number }>;
/** The learned writer's complete bundle from parent (null: the incoming bundle); sample false decodes deterministically. */
export function propose(request: { parent: string | null; sample: boolean; seed: number; focus: string[]; digests: { binding: string; digest: Neuralese<EvidenceDigest> }[] }): Promise<BundleSummary>;
/** Local in-bank refinement from parent by gradient on the support evidence. */
export function refine(request: { parent: string | null; steps: number }): Promise<BundleSummary>;
/** Gradient-free mixing over the bank from parent (the retrieval baseline). */
export function retrieve(request: { parent: string | null }): Promise<BundleSummary>;`;

/** The outer-training services over one operation (bank, targets, optimizer state, store), each a resumable spec. */
export class TrainingServices {
  constructor(private readonly python: PythonEntry, private readonly operation: {
    readonly common: Record<string, unknown>; readonly bank: string; readonly state: string; readonly targets: string;
    readonly targetsSpec: Record<string, unknown>; readonly records: string; readonly receipt: string; readonly ladder?: Record<string, unknown>;
    readonly recipe?: Record<string, unknown> }) {}
  private collected = { records: 0, contrastGroups: 0, noContrastGroups: 0, snr: 0 };
  private latestLadder: Record<string, number | null> = {};
  private targetCount = 0;
  private async receipt(): Promise<Record<string, any> | null> {
    try { return JSON.parse(await readFile(this.operation.receipt, 'utf8')); } catch { return null; }
  }
  private train(extra: Record<string, unknown>) {
    const o = this.operation;
    return runPython(this.python, 'train', { ...o.common, bank: o.bank, targets: o.targets, state: o.state, receipt: o.receipt,
      ...(o.recipe ? { recipe: o.recipe } : {}), ...extra });
  }
  readonly trainer = {
    metrics: async (): Promise<TrainingMetrics> => {
      const receipt = await this.receipt();
      const targets = this.targetCount || (await readFile(this.operation.targets, 'utf8').catch(() => '')).split('\n').filter(Boolean).length;
      return { step: receipt?.step ?? 0, labelsStale: !!receipt?.labels_stale, targetsAdmitted: targets,
        proposalsRecorded: this.collected.records, contrastGroups: this.collected.contrastGroups,
        noContrastGroups: this.collected.noContrastGroups, advantageSnr: this.collected.snr, ladder: this.latestLadder,
        programs: new Set(((await readFile(this.operation.targets, 'utf8').catch(() => '')).split('\n').filter(Boolean)
          .map(line => (JSON.parse(line) as { manifest: { program: { id: string } } }).manifest.program.id))).size };
    },
    refitLabels: async () => {
      this.targetCount = await runPython(this.python, 'targets', { ...this.operation.common, ...this.operation.targetsSpec,
        bank: this.operation.bank, out: this.operation.targets }) as number;
      return { targets: this.targetCount };
    },
    supervised: async (request: { steps: number }) => ({ step: await this.train({ steps: request.steps }) as number }),
    collectProposals: async (request: { perContext: number }) => {
      const summary = await runPython(this.python, 'collect', { ...this.operation.common, bank: this.operation.bank, state: this.operation.state,
        targets: this.operation.targets, perContext: request.perContext, out: this.operation.records }) as Record<string, number>;
      this.collected = { records: summary.records!, contrastGroups: summary.contrastGroups!, noContrastGroups: summary.noContrastGroups!,
        snr: Number.isFinite(summary.snr) ? summary.snr! : 1e9 };
      return { recorded: summary.records! };
    },
    policy: async () => ({ step: await this.train({ steps: 0, policy_records: [this.operation.records] }) as number, estimator: 'joint-ratio-v1' }),
    publishBank: async () => {
      const receipt = await this.receipt();
      const published = (receipt?.published ?? []) as string[];
      return { snapshot: published[published.length - 1] ?? 'unchanged' };
    },
    ladder: async () => {
      if (!this.operation.ladder) return this.latestLadder;
      await runPython(this.python, 'ladder', this.operation.ladder);
      const report = JSON.parse(await readFile(this.operation.ladder.out as string, 'utf8')) as { aggregate: { medianGaps: Record<string, number | null> } };
      return this.latestLadder = report.aggregate.medianGaps;
    },
  };
}

export const trainerDeclaration = `/** Exact outer-training services: gradients, updates, refits, publication and measurement. */
export function metrics(): Promise<TrainingMetrics>;
export function refitLabels(): Promise<{ targets: number }>;
export function supervised(request: { steps: number }): Promise<{ step: number }>;
export function collectProposals(request: { perContext: number }): Promise<{ recorded: number }>;
export function policy(): Promise<{ step: number; estimator: string }>;
export function publishBank(): Promise<{ snapshot: string }>;
export function ladder(): Promise<Record<string, number | null>>;`;

/** Run one adaptation episode of at most `maxSteps` revisions. */
export async function adapt(runtime: NatlangRuntime, services: AdaptationServices, maxSteps: number): Promise<AdaptationState> {
  return runtime.run(() => adaptFlow(maxSteps), { services: { optimizer: services.optimizer },
    serviceDeclarations: { optimizer: optimizerDeclaration } });
}

/** Run at most `cycles` scheduled training cycles. */
export async function train(runtime: NatlangRuntime, services: TrainingServices, cycles: number): Promise<TrainingState> {
  return runtime.run(() => trainFlow(cycles), { services: { trainer: services.trainer }, serviceDeclarations: { trainer: trainerDeclaration } });
}
