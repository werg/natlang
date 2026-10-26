import { executionKey } from './execution.js';
/**
 * `natlang tune --rounds`: the SFT/RL round schedule (rounds.ts) over the ordinary pipeline (plan §5.1a).
 *
 * Every round trains through `trainRelease` (an SFT round continues the parent's build on the teacher rows, an RL
 * round continues it on fresh student rollouts plus those rows, a repair round on the verified on-policy corrections of
 * the parent release plus those rows, repairs.ts), rollouts sample training roots from the parent
 * release on the backend with exact sampling receipts, and rounds are compared on development roots only. The best
 * development release is what `tuneProgram` then validates and activates, as for a single joint fit.
 */
import { join } from 'node:path';
import { readRelease, tuneDirectory } from '../adaptation/weights/release.js';
import { WeightBinding } from '../adaptation/weights/binding.js';
import { rootSuccess } from './evaluate.js';
import { evaluateRelease, trainRelease, type ReleaseSelector, type TuneOptions } from './pipeline.js';
import { runRollouts } from './rollouts.js';
import { readRepairRows, runRepairs } from './repairs.js';
import { DEFAULT_ROUNDS, runRounds, type RoundKind } from './rounds.js';

export type RoundsSettings = {
  readonly schedule?: readonly RoundKind[]; readonly patience?: number; readonly minDelta?: number;
  /** Rollouts per training root and round (group-relative rewards need at least two). */
  readonly samples?: number; readonly temperature?: number;
  /** Recipe fields for RL rounds; default alternates SFT and RL steps so the teacher rows keep anchoring the fit. */
  readonly rlRecipe?: Readonly<Record<string, unknown>>;
  /** Continue an interrupted initial fit without inventing an evaluated parent release. Later rounds use their actual parent. */
  readonly initialFitParent?: string;
  /** Repair rounds: verified corrections kept per wrong training root (default 1, from the latest boundary back) and
   * boundaries tried per root (default all). */
  readonly repairsPerRoot?: number; readonly repairBoundaries?: number;
};

export function roundsSelector(options: TuneOptions, settings: RoundsSettings = {}): ReleaseSelector {
  const schedule = settings.schedule ?? DEFAULT_ROUNDS;
  if (schedule.includes('rl') && options.settings.backend !== 'native')
    throw new Error('RL rounds sample the student with exact sampling receipts: set tune.backend to "native" (a neuraleseServer profile), or use an SFT-only schedule');
  return async (data, _timings, log) => {
    const directory = join(tuneDirectory(options.root), 'rounds', `native-runtime-v2-${options.settings.execution ? `${data.dataKey}-execution-${executionKey(options.settings.execution)}` : data.dataKey}`);
    const development = data.examples.filter(example => example.partition === 'development');
    if (!development.length) throw new Error('rounds are selected on development roots; this program has none');
    const journal = await runRounds({ journal: join(directory, 'journal.json'), schedule, patience: settings.patience, minDelta: settings.minDelta, log,
      train: async ({ index, kind, parent, rollouts, repairs }) => {
        const { manifest, build } = await trainRelease(options, data, { manifest: data.scaffold, label: `round-${index}-${kind}`,
          // Corrections join the earlier usable rows; their prompts (the student's prefix) carry no target weight.
          ...(repairs ? { rows: [...data.trainRows, ...readRepairRows(repairs)] } : {}),
          ...(kind === 'rl' ? { recipe: { objectives: ['sft', 'rl'], ...settings.rlRecipe } } : {}),
          ...(parent ? { parent: parent.build } : index === 0 && settings.initialFitParent ? { parent: settings.initialFitParent } : {}), ...(rollouts ? { rollouts } : {}) }, [], log);
        return { release: manifest.digest, build };
      },
      rollouts: async ({ index, parent }) => {
        const release = readRelease(options.root, parent.release);
        const served = await options.serve({ manifest: release, directory: join(tuneDirectory(options.root), 'releases', release.digest), capture: true });
        try {
          const summary = await runRollouts({ root: options.root, program: options.program, release, driver: served.driver,
            weights: [new WeightBinding(release, { program: options.program, available: served.loaded, nativeState: !!served.nativeState, strict: true })],
            examples: data.examples, samples: settings.samples ?? 4, temperature: settings.temperature ?? 0.8, round: index,
            out: join(directory, `round-${index}.rollouts.jsonl`), policyCheckpointSha256: options.student.base.checkpoint_sha256, log, services: options.services, execution: options.settings.execution });
          return { path: summary.path, written: summary.written, reused: summary.reused, successes: summary.successes, skipped: summary.skipped.length };
        } finally { await served.close(); }
      },
      repairs: async ({ index, parent }) => {
        const release = readRelease(options.root, parent.release);
        const served = await options.serve({ manifest: release, directory: join(tuneDirectory(options.root), 'releases', release.digest) });
        try {
          const summary = await runRepairs({ root: options.root, program: options.program, scaffold: data.scaffold, release,
            weights: new WeightBinding(release, { program: options.program, available: served.loaded, nativeState: !!served.nativeState, strict: true }),
            student: served.driver, teacher: options.teacher, teacherId: options.teacherId, examples: data.examples, round: index,
            records: join(directory, `round-${index}.repairs.jsonl`), rows: join(directory, `round-${index}.repair-rows.jsonl`),
            checkpointSha256: options.student.base.checkpoint_sha256, maxPerRoot: settings.repairsPerRoot, maxBoundaries: settings.repairBoundaries, log,
            services: options.services, execution: options.settings.execution });
          return { records: summary.records, rows: summary.rows, admitted: summary.admitted, rejected: summary.rejected,
            studentCorrect: summary.studentCorrect, roots: summary.roots };
        } finally { await served.close(); }
      },
      evaluate: async digest => {
        const arm = await evaluateRelease(options, data, readRelease(options.root, digest), development, `round:${digest.slice(0, 12)}`, 'development');
        return { success: arm.roots.reduce((sum, root) => sum + rootSuccess(root), 0), roots: arm.roots.length };
      } });
    if (!journal.best) throw new Error(`no round produced a release${journal.stopped ? ` (${journal.stopped})` : ''}`);
    log(`natlang tune: rounds selected round ${journal.best.round} (${journal.best.success} development successes)`);
    return readRelease(options.root, journal.best.release);
  };
}
