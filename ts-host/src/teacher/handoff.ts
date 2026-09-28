/**
 * Handoffs and preference pairs. A handoff gives a teacher a run where the model that made it failed: the run's earlier
 * turns are replayed call by call (replay.ts) and the teacher answers the failed turn and every turn after it. The
 * teacher's decision at that point and the failed one are a preference pair on the failing model's own state.
 *
 * A site is a decision that failed (an action the runtime rejected or refused, one that raised, an attempt the task's
 * checker refused) and was the first of its call's failures in a row, or the result a run returned and the task did not
 * accept. Every pair is checked by replay under the current runtime (failsInPlace): the failed response, put back in
 * its place, sees what it saw and still fails.
 */
import type { ModelTurn } from '../contracts.js';
import { executeProgram, programRow, type ProgramRecord, type TeacherRow } from './collector.js';
import { retiredFamily } from './curriculum-policy.js';
import { materializeNativeRows } from './native-materializer.js';
import { callNumbers, indexOf, observed, openingsOf, placeOf, recorded, scriptedDriver, scriptOf, type Turn } from './replay.js';

export const HANDOFF_VERSION = 'natlang.handoff/2';
export const PREFERENCE_VERSION = 'natlang.preference_pair/2';

export type SiteKind = 'failed_action' | 'wrong_result';
export type Site = { index: number; kind: SiteKind };

/**
 * What a program record carries to be run as a handoff: the replayed turns per recorded call, the calls' openings (to
 * match the replay's calls to them, replay.ts callMatcher), and which call is handed over.
 */
export type Handoff = { version: typeof HANDOFF_VERSION; kind: SiteKind;
  source: { trajectory_id: string; model: string; index: number; program_id: string };
  prefix: ModelTurn[][]; openings: string[]; call: number; rejected: ModelTurn };

export type ReplayOptions = { systemPrompt: string; contextTokens: number; maxTurns?: number; rootSeed: number };

type Dict = Record<string, unknown>;
const program = (row: TeacherRow) => {
  const { handoff: _, ...record } = (row.task as { program_ir: ProgramRecord }).program_ir;
  return record as ProgramRecord;
};

/** The sites of a run that was not accepted. */
export function handoffSites(row: TeacherRow): Site[] {
  if (retiredFamily(row.task.program_ir) || (row.outcome as Dict | undefined)?.accepted !== false) return [];
  if ((row.outcome?.checks as Dict | undefined)?.seeded_failure_requirement === false) return [];
  const result = materializeNativeRows([row], { failedRuns: true });
  if (result.unlinked.length) return [];
  const trajectory = row.trajectory as unknown as Turn[], calls = callNumbers(trajectory);
  const decisions = new Map(result.turns.map(turn => [(turn.decision as Dict).index as number, turn.decision as Dict]));
  const sites: Site[] = [];
  const lastFailed = new Map<number, boolean>();
  trajectory.forEach((_, index) => {
    const failed = decisions.get(index)?.failed_action === true;
    if (failed && !lastFailed.get(calls[index]!)) sites.push({ index, kind: 'failed_action' });
    lastFailed.set(calls[index]!, failed);
  });
  // The root's last turn returned the result the task did not accept.
  const outcome = row.outcome as Dict, last = calls.lastIndexOf(0);
  if (outcome.status === 'done' && last >= 0 &&
      (trajectory[last]!.assistant.calls ?? []).some(call => call.tool === 'return_result'))
    sites.push({ index: last, kind: 'wrong_result' });
  return sites;
}

/** The handoff of a run at a site: the turns before it, per call, and the failed turn's response. */
export function handoffAt(row: TeacherRow, site: Site): Handoff {
  const trajectory = row.trajectory as unknown as Turn[], calls = callNumbers(trajectory);
  const later = new Set(trajectory.map((_, index) => index).filter(index => index >= site.index));
  return { version: HANDOFF_VERSION, kind: site.kind,
    source: { trajectory_id: row.id as string, model: String(row.provenance.model), index: site.index,
      program_id: program(row).id },
    prefix: scriptOf(trajectory, later), openings: openingsOf(trajectory), call: calls[site.index]!,
    rejected: recorded(trajectory[site.index]!) };
}

/** The program record that runs a handoff, as its own task. */
export function handoffRecord(row: TeacherRow, handoff: Handoff): ProgramRecord {
  const record = program(row);
  return { ...record, id: `${record.id}:handoff:${handoff.source.trajectory_id.split(':').pop()}:${handoff.source.index}`, handoff };
}

/**
 * Whether a response, put in the place of a row's turn `index` after the row's earlier turns, sees what that turn saw
 * and fails as it did there: a failed action fails again, a wrong result is again not accepted. Null if so, else why not.
 */
export async function failsInPlace(row: TeacherRow, index: number, response: ModelTurn, kind: SiteKind,
    options: ReplayOptions, runId: string): Promise<string | null> {
  if (retiredFamily(program(row))) return 'the exercise belongs to a retired curriculum family';
  const trajectory = row.trajectory as unknown as Turn[], calls = callNumbers(trajectory);
  const place = placeOf(calls, index);
  const script = scriptOf(trajectory, new Set(trajectory.map((_, at) => at).filter(at => at >= index)));
  (script[place.call] ??= []).push(response);
  const { driver, turns, places } = scriptedDriver(script, openingsOf(trajectory));
  const record = program(row);
  const run = await executeProgram(record, driver, { ...options, runId });
  const at = places.findIndex(item => item.call === place.call && item.nth === place.nth);
  if (at < 0) return 'the replay did not reach the failed turn';
  for (const [replayed, item] of places.entries()) {
    const source = item.scripted ? indexOf(trajectory, item) : -1;
    if (source >= 0 && source <= index &&
        JSON.stringify(observed(turns[replayed]!.context)) !== JSON.stringify(observed(trajectory[source]!.context)))
      return 'the replay saw other results than the run did';
  }
  if ((run.outcome.checks as Dict | undefined)?.seeded_failure_requirement === false)
    return 'the exercise did not exhibit its required seeded failure';
  if (kind === 'wrong_result') return run.outcome.accepted ? 'the result is accepted now' : null;
  const replay = programRow(record, 'replay', runId, row.provenance, run, turns as unknown as Dict[]);
  const decision = materializeNativeRows([replay], { failedRuns: true }).turns
    .find(turn => (turn.decision as Dict).index === at)?.decision as Dict | undefined;
  return decision?.failed_action === true ? null : 'the action no longer fails';
}

/** A response as a training target, as the materializer renders a decision's. */
export function responseTarget(response: ModelTurn, index: number): Dict {
  const target: Dict = { role: 'assistant', content: response.text ?? '' };
  if (response.calls?.length) target.tool_calls = response.calls.map(([tool, args], offset) => ({
    id: `teacher_${index}_${offset}`, type: 'function', function: { name: tool, arguments: JSON.stringify(args ?? {}) } }));
  return target;
}

const reasoningOf = (response: ModelTurn) =>
  (Object.hasOwn(response, 'execution_plan') ? response.execution_plan : response.reasoning) ?? null;

/**
 * The preference pair of a row whose turn `index` is preferred to `rejected` made in its place: the chosen decision
 * must be approved for training, and the rejected response must fail in place (failsInPlace). A pair or why not.
 */
export async function preferencePair(row: TeacherRow, index: number, rejected: ModelTurn, kind: SiteKind,
    evidence: Dict, options: ReplayOptions, runId: string): Promise<Dict | { rejected: string }> {
  const turns = materializeNativeRows([row]).turns;
  const chosen = turns.find(turn => (turn.decision as Dict).index === index);
  if (!chosen) return { rejected: 'the run is not accepted or its outcomes cannot be linked' };
  if (!(chosen.training_admission as Dict).approved)
    return { rejected: `the chosen decision is not approved: ${(chosen.training_admission as Dict).reason}` };
  const failure = await failsInPlace(row, index, rejected, kind, options, runId);
  if (failure) return { rejected: failure };
  const reasoning = reasoningOf(rejected);
  // A handoff task is its source program's: it must fall on the same side of a train/held-out split.
  const programId = (row.task.program_ir.handoff as Handoff | undefined)?.source.program_id ?? chosen.program_id;
  return { version: PREFERENCE_VERSION, id: `${row.id}:preference:${index}`, kind,
    program_id: programId, source_groups: [programId], messages: chosen.messages, tools: chosen.tools,
    chosen: { target: chosen.target, reasoning: chosen.teacher_reasoning,
      ...(chosen.teacher_reasoning_trained === false ? { reasoning_trained: false } : {}) },
    rejected: { target: responseTarget(rejected, index), reasoning,
      // The same reasoning as the chosen side is context shared by both, not what the pair is about.
      ...(reasoning === chosen.teacher_reasoning && chosen.teacher_reasoning_trained === false ? { reasoning_trained: false } : {}) },
    evidence };
}
