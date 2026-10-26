/**
 * SFT/RL tuning rounds (plans/HIERARCHICAL_LORA_TUNING.md §5.1a).
 *
 * Round 0 fits the program on the teacher's training rows (SFT). Each later round continues the BEST release so far:
 * an RL round first samples rollouts of the training roots from that release (`runRollouts`), then fits a
 * continuation of its build on those rollouts (the trainer's `parent` plan field, so the sampled node and soft-state
 * versions are recorded lineage); an SFT round continues it on the training rows. Every round is measured on
 * DEVELOPMENT roots only, through the injected evaluator. The best release by development success is kept; after
 * `patience` consecutive rounds without an improvement above `minDelta`, the schedule stops early. Confirmation and
 * protected roots are never used here.
 *
 * How SFT and RL mix: they never mix within one optimizer step. An RL round's fit alternates objectives step by step
 * when its recipe says `objectives: ["sft", "rl"]` (the recommended default for the injected trainer: the teacher's
 * rows keep anchoring the specialization while the policy objective moves it toward the oracle), or trains the policy
 * objective alone with `["rl"]`.
 *
 * The journal (`natlang.tune-rounds/v1`) is rewritten after every stage, so a rerun resumes inside a round: sampled
 * rollouts, a trained release and an evaluation are each reused, and the train stage itself resumes from its build.
 */
import { existsSync, mkdirSync, readFileSync, renameSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';

export type RoundKind = 'sft' | 'rl';
export type RoundParent = { readonly round: number; readonly release: string; readonly build: string };
export type RoundRollouts = { readonly path: string; readonly written: number; readonly reused: number; readonly successes: number;
  readonly skipped: number };
export type RoundRecord = {
  readonly index: number; readonly kind: RoundKind; readonly parent: RoundParent | null;
  readonly status: 'started' | 'trained' | 'evaluated' | 'skipped';
  readonly rollouts?: RoundRollouts; readonly release?: string; readonly build?: string;
  readonly development?: { readonly success: number; readonly roots: number }; readonly improved?: boolean; readonly reason?: string;
};
export type RoundsJournal = { readonly schema: 'natlang.tune-rounds/v1'; readonly schedule: readonly RoundKind[];
  readonly rounds: readonly RoundRecord[];
  readonly best: { readonly round: number; readonly release: string; readonly build: string; readonly success: number } | null;
  readonly stopped: string | null };

export type RoundsOptions = {
  readonly journal: string;
  /** Round kinds in order; the first must be "sft" (RL samples from an existing release). */
  readonly schedule?: readonly RoundKind[];
  /** Stop after this many consecutive rounds without a development improvement. */
  readonly patience?: number;
  readonly minDelta?: number;
  /** Fit one round: from scratch (parent null) or as a continuation of the parent's build; returns its stored release. */
  readonly train: (round: { index: number; kind: RoundKind; parent: RoundParent | null; rollouts: string | null }) =>
    Promise<{ release: string; build: string }>;
  /** Sample training-root rollouts from the parent release (`runRollouts`). */
  readonly rollouts: (round: { index: number; parent: RoundParent }) => Promise<RoundRollouts>;
  /** Development-root success of a release (never confirmation or protected roots). */
  readonly evaluate: (release: string) => Promise<{ success: number; roots: number }>;
  readonly log?: (line: string) => void;
};

export const DEFAULT_ROUNDS: readonly RoundKind[] = ['sft', 'rl', 'rl', 'rl'];

export function readRounds(path: string): RoundsJournal | null {
  return existsSync(path) ? JSON.parse(readFileSync(path, 'utf8')) as RoundsJournal : null;
}

function write(path: string, journal: RoundsJournal): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path + '.tmp', JSON.stringify(journal, null, 2) + '\n');
  renameSync(path + '.tmp', path);
}

export async function runRounds(options: RoundsOptions): Promise<RoundsJournal> {
  const schedule = options.schedule ?? DEFAULT_ROUNDS, patience = options.patience ?? 2, minDelta = options.minDelta ?? 0;
  if (!schedule.length || schedule[0] !== 'sft') throw new Error('a round schedule starts with an sft round');
  if (schedule.some(kind => kind !== 'sft' && kind !== 'rl')) throw new Error('round kinds are sft or rl');
  if (!Number.isInteger(patience) || patience < 1) throw new Error('patience is a positive integer');
  const previous = readRounds(options.journal);
  if (previous && JSON.stringify(previous.schedule.slice(0, schedule.length)) !== JSON.stringify(schedule.slice(0, previous.schedule.length)))
    throw new Error(`the round journal ${options.journal} was written for another schedule; use a new journal`);
  const rounds: RoundRecord[] = [...(previous?.rounds ?? [])];
  let best = previous?.best ?? null;
  const save = (stopped: string | null = null) => write(options.journal, { schema: 'natlang.tune-rounds/v1', schedule, rounds, best, stopped });
  const stale = () => {
    let count = 0;
    for (const round of [...rounds].reverse()) { if (round.status !== 'evaluated' && round.status !== 'skipped' || round.improved) break; count++; }
    return count;
  };
  for (let index = 0; index < schedule.length; index++) {
    const kind = schedule[index]!;
    let record = rounds[index];
    if (record?.status === 'evaluated' || record?.status === 'skipped') continue;
    if (stale() >= patience) {
      const reason = `no development improvement in ${patience} rounds`;
      options.log?.(`rounds: stopped (${reason})`);
      save(reason);
      return readRounds(options.journal)!;
    }
    const parent: RoundParent | null = index === 0 ? null : best && { round: best.round, release: best.release, build: best.build };
    if (index > 0 && !parent) throw new Error('no evaluated release to continue');
    const put = (next: RoundRecord) => { rounds[index] = record = next; save(); };
    if (!record) put({ index, kind, parent, status: 'started' });
    let rollouts: RoundRollouts | undefined = record!.rollouts;
    if (kind === 'rl' && !rollouts) {
      rollouts = await options.rollouts({ index, parent: parent! });
      put({ ...record!, rollouts });
      options.log?.(`round ${index}: ${rollouts.written + rollouts.reused} rollouts, ${rollouts.successes} new successes`);
    }
    if (kind === 'rl' && rollouts!.written + rollouts!.reused === 0) {
      put({ ...record!, status: 'skipped', improved: false, reason: 'no scorable rollouts' });
      continue;
    }
    if (record!.status === 'started') {
      const trained = await options.train({ index, kind, parent: record!.parent, rollouts: rollouts?.path ?? null });
      put({ ...record!, status: 'trained', release: trained.release, build: trained.build });
    }
    const development = await options.evaluate(record!.release!);
    const improved = best === null || development.success > best.success + minDelta;
    if (improved) best = { round: index, release: record!.release!, build: record!.build!, success: development.success };
    put({ ...record!, status: 'evaluated', development, improved });
    options.log?.(`round ${index} (${kind}): development success ${development.success.toFixed(3)} over ${development.roots} roots`
      + (improved ? ' (best)' : ''));
  }
  save();
  return readRounds(options.journal)!;
}
