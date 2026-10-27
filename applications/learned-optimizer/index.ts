/**
 * The joint learned optimizer as a Natlang program (plans/neuralese/JOINT_LEARNED_OPTIMIZER.md). adapt.nl revises a
 * program's native instruction states, skill states and LoRA mixtures together through `iterateOn`; train.nl
 * schedules the outer training. This file is the outside world those stages act on: the `optimizer` and `trainer`
 * services, backed by the exact Python mechanism (`python -m natlang_neuralese.tune.learned service|train|...`).
 * Bundles stay host-side; the program sees their identities and exact support losses.
 */
import { spawn } from 'node:child_process';
import { readFile } from 'node:fs/promises';
import type { NatlangRuntime } from '@natlang/node';
import adaptFlow from './adapt.nl';
import trainFlow from './train.nl';
import { optimizerDeclaration, type AdaptationServices, type PythonService } from './python.mjs';
import type { AdaptationState, TrainingMetrics, TrainingState } from './types.js';

export type * from './types.js';

export { AdaptationServices, PythonService, optimizerDeclaration, type PythonEntry } from './python.mjs';

function runCommand(argv: readonly string[], args: readonly string[], cwd?: string): Promise<void> {
  return new Promise((resolveRun, reject) => {
    let tail = '';
    const child = spawn(argv[0]!, [...argv.slice(1), ...args], { cwd: cwd ?? process.cwd(), stdio: ['ignore', 'pipe', 'pipe'] });
    const keep = (chunk: Buffer) => { tail = (tail + chunk.toString()).slice(-4000); };
    child.stdout.on('data', keep); child.stderr.on('data', keep);
    child.once('error', reject);
    child.once('exit', code => code === 0 ? resolveRun() : reject(new Error(`${[...argv, ...args].join(' ')} exited with ${code}\n${tail.trim()}`)));
  });
}

/** The outer-training services over one operation (bank, targets, optimizer state, store), each a resumable spec. */
export class TrainingServices {
  constructor(private readonly python: PythonService, private readonly operation: {
    readonly common: Record<string, unknown>; readonly bank: string; readonly state: string; readonly targets: string;
    readonly targetsSpec: Record<string, unknown>; readonly records: string; readonly receipt: string; readonly ladder?: Record<string, unknown>;
    readonly recipe?: Record<string, unknown>;
    /** Score proposals by real task outcomes: the natlang CLI argv (e.g. `["node", "ts-host/bin/natlang.mjs"]`) and the
     * directory for proposal releases. Without it the outer reward is the shaping reward alone. */
    readonly outcomes?: { readonly natlang: readonly string[]; readonly releases: string; readonly cwd?: string } }) {}
  private collected = { records: 0, contrastGroups: 0, noContrastGroups: 0, snr: 0 };
  private latestLadder: Record<string, number | null> = {};
  private targetCount = 0;
  private async receipt(): Promise<Record<string, any> | null> {
    try { return JSON.parse(await readFile(this.operation.receipt, 'utf8')); } catch { return null; }
  }
  private train(extra: Record<string, unknown>) {
    const o = this.operation;
    return this.python.command('train', { ...o.common, bank: o.bank, targets: o.targets, state: o.state, receipt: o.receipt,
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
      this.targetCount = await this.python.command('targets', { ...this.operation.common, ...this.operation.targetsSpec,
        bank: this.operation.bank, out: this.operation.targets }) as number;
      return { targets: this.targetCount };
    },
    supervised: async (request: { steps: number }) => ({ step: await this.train({ steps: request.steps }) as number }),
    collectProposals: async (request: { perContext: number }) => {
      const o = this.operation, sampling = { ...o.common, bank: o.bank, state: o.state, targets: o.targets, perContext: request.perContext };
      let scored: Record<string, unknown> = {};
      if (o.outcomes) {
        // Real task outcomes: proposals become releases, each program scores its own on query roots.
        const phase = await this.python.command('proposals', { ...sampling, out: `${o.records}.proposals.jsonl`,
          releases: o.outcomes.releases }) as { requests: Record<string, string> };
        const outcomes: Record<string, unknown> = {};
        for (const [root, path] of Object.entries(phase.requests)) {
          await runCommand(o.outcomes.natlang, ['tune', root, '--score-proposals', path], o.outcomes.cwd);
          Object.assign(outcomes, JSON.parse(await readFile(`${path}.outcomes.json`, 'utf8')));
        }
        scored = { proposals: `${o.records}.proposals.jsonl`, outcomes };
      }
      const summary = await this.python.command('collect', { ...sampling, ...scored, out: o.records }) as Record<string, number>;
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
      await this.python.command('ladder', this.operation.ladder);
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
