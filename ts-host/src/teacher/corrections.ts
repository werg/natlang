/**
 * Corrected variants: a trajectory in which a call made a failing attempt and then fixed it, rewritten so that the
 * call makes the fix first. The fix is given the reasoning that preceded the first failed attempt, which does not
 * react to an error that no longer happens.
 *
 * Checked by replay under the current runtime, against a control replay of the unchanged row (recorded outputs can
 * predate runtime changes): the control must end accepted; the variant must end accepted too, and its fix must show
 * exactly what the control's fix showed. A fix that relied on what the failure changed or taught fails there.
 */
import { executeProgram, programRow, programRunId, type ProgramRecord, type TeacherRow } from './collector.js';
import { callNumbers, openingsOf, placeOf, scriptedDriver, scriptOf, type Turn } from './replay.js';
import { admitRow, type CurriculumRecord } from './curriculum.js';
import { text } from './opening.js';

export const CORRECTION_VERSION = 'corrected-first-attempt/1';

/** Failed attempts of one call (flat trajectory indices) and the successful turn of that call that followed them. */
export type CorrectionSite = { failed: number[]; fixed: number };

export type CorrectionResult = { row: TeacherRow; decision: number } | { rejected: string };

export type ReplayOptions = { systemPrompt: string; contextTokens: number; maxTurns?: number; rootSeed: number };

/** The strings in a value (code in call arguments, not its JSON escapes). */
function strings(value: unknown): string[] {
  if (typeof value === 'string') return [value];
  if (value && typeof value === 'object') return Object.values(value).flatMap(strings);
  return [];
}

/** Code-like words: identifiers, keywords and dotted names. */
function words(value: unknown): Set<string> {
  return new Set(strings(value).flatMap(item => item.match(/[A-Za-z_$][\w$]*(?:\.[A-Za-z_$][\w$]*)*/g) ?? []));
}

/**
 * Whether reasoning fits a fix: it must not name what went wrong, which is what the error names, the failed attempt
 * used and the fix left behind: the `while` of a loop refusal, the step.iterateOn of a TypeError, a move the failure
 * already performed. Words the error only suggests (its advice) are not what went wrong. Names are code-like: dotted,
 * with `_` or `$`, camelCase, or a keyword the runtime refuses.
 */
export function reasoningFitsFix(reasoning: string, error: string, failed: unknown, fix: unknown): boolean {
  const used = words(failed), kept = words(fix), said = words(reasoning);
  return ![...words(error)].some(word => /[.$_]|[a-z][A-Z]|^(?:while|do|recursion)$/.test(word) &&
    used.has(word) && !kept.has(word) && said.has(word));
}

/** What a call's turn showed: the tool results in the context of that call's next turn, or the run's end. */
function outputOf(trajectory: Turn[], calls: number[], call: number, nth: number, ending: string): string {
  const turns = trajectory.filter((_, index) => calls[index] === call);
  const turn = turns[nth], next = turns[nth + 1];
  if (!turn) return '(missing)';
  if (!next) return ending;
  return next.context.slice(turn.context.length + 1).filter(message => message.role === 'tool')
    .map(message => text(message.content)).join('\n');
}

/** Replay a row's recorded responses, call by call in order, leaving out some turns and changing the reasoning of one. */
async function replay(row: TeacherRow, options: ReplayOptions, runId: string, leftOut: Set<number>,
    reasoned?: { index: number; reasoning: string; planned: boolean }) {
  const trajectory = row.trajectory as unknown as Turn[];
  const changed = trajectory.map(turn => structuredClone(turn));
  if (reasoned) {
    const assistant = changed[reasoned.index]!.assistant;
    if (reasoned.planned) assistant.execution_plan = reasoned.reasoning; else assistant.reasoning = reasoned.reasoning;
  }
  const { driver, turns, places } = scriptedDriver(scriptOf(changed, leftOut), openingsOf(trajectory));
  const record = (row.task as { program_ir: ProgramRecord }).program_ir;
  const run = await executeProgram(record, driver, { ...options, runId });
  return { run, turns, calls: places.map(place => place.call), ending: JSON.stringify([run.outcome.status, run.outcome.value]) };
}

/** The corrected variant of a row at one site, or why there is none. */
export async function correctedVariant(row: TeacherRow, site: CorrectionSite, options: ReplayOptions): Promise<CorrectionResult> {
  const trajectory = row.trajectory as unknown as Turn[], calls = callNumbers(trajectory);
  const first = trajectory[site.failed[0]!]!, fix = trajectory[site.fixed]!;
  // A failed attempt that started calls of its own would change which calls the variant starts, and in what order.
  const known = new Set(calls.slice(0, site.failed[0]!));
  if (calls.slice(site.failed[0]!, site.fixed).some(call => call !== calls[site.fixed] && !known.has(call)))
    return { rejected: 'a failed attempt started calls of its own' };
  const planned = Object.hasOwn(first.assistant, 'execution_plan');
  const reasoning = (planned ? first.assistant.execution_plan : first.assistant.reasoning) ?? '';
  const place = placeOf(calls, site.fixed), failedPlace = placeOf(calls, site.failed[0]!);
  const error = outputOf(trajectory, calls, failedPlace.call, failedPlace.nth, '');
  if (!reasoningFitsFix(reasoning, error, first.assistant.calls, fix.assistant.calls))
    return { rejected: 'the reasoning before the failed attempt names what failed' };
  const provenance = row.provenance as Record<string, unknown>;
  const runId = programRunId(0, provenance);
  const control = await replay(row, options, runId, new Set());
  if (!control.run.outcome.accepted) return { rejected: 'the row does not replay to an accepted end under the current runtime' };
  const variant = await replay(row, options, runId, new Set(site.failed), { index: site.fixed, reasoning, planned });
  if (!variant.run.outcome.accepted) return { rejected: 'the program no longer ends accepted without the failed attempts' };
  const nthInVariant = place.nth - site.failed.length;
  if (outputOf(variant.turns, variant.calls, place.call, nthInVariant, variant.ending) !==
      outputOf(control.turns, control.calls, place.call, place.nth, control.ending))
    return { rejected: 'the fix shows something other than it showed after the failure' };
  const decision = variant.calls.findIndex((call, index) => call === place.call &&
    variant.calls.slice(0, index).filter(other => other === call).length === nthInVariant);
  const record = (row.task as { program_ir: ProgramRecord }).program_ir;
  const variantRow = programRow(record, String(provenance.model), runId, { ...provenance,
    variant: { version: CORRECTION_VERSION, parent: row.id, left_out: site.failed, fixed: site.fixed, decision, run_id: runId } },
    variant.run, variant.turns as unknown as Record<string, unknown>[]);
  // A curriculum case's checks hold for the variant too: a failed attempt may have made the observation it needs.
  if ((record as CurriculumRecord).curriculum) {
    const admission = admitRow(variantRow as Parameters<typeof admitRow>[0]);
    if (!admission.admitted || admission.notes?.includes('judged_directly'))
      return { rejected: `the variant is not admitted: ${[...admission.reasons, ...admission.notes ?? []].join(', ')}` };
  }
  return { row: { ...variantRow, id: `${row.id}:corrected:${site.failed[0]}` }, decision };
}
