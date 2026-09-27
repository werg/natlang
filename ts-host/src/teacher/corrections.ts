/**
 * Corrected variants: a trajectory in which a call made a failing attempt and then fixed it, rewritten so that the
 * call makes the fix first. The fix is given the reasoning that preceded the first failed attempt, which does not
 * react to an error that no longer happens.
 *
 * Checked by replay under the current runtime, against a control replay of the unchanged row (recorded outputs can
 * predate runtime changes): the control must end accepted; the variant must end accepted too, and its fix must show
 * exactly what the control's fix showed. A fix that relied on what the failure changed or taught fails there.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { executeProgram, programRow, programRunId, trajectoryTurn, type ProgramRecord, type TeacherRow } from './collector.js';
import { openingText, text, type Message } from './opening.js';

export const CORRECTION_VERSION = 'corrected-first-attempt/1';

type Turn = { context: Message[]; assistant: { reasoning?: string | null; execution_plan?: string | null; content?: string;
  calls?: { tool: string; arguments: Record<string, unknown> }[] } };

/** Failed attempts of one call (flat trajectory indices) and the successful turn of that call that followed them. */
export type CorrectionSite = { failed: number[]; fixed: number };

export type CorrectionResult = { row: TeacherRow; decision: number } | { rejected: string };

export type ReplayOptions = { systemPrompt: string; contextTokens: number; maxTurns?: number; rootSeed: number };

/** The response a recorded turn gave, as the reply to replay. */
function recorded(turn: Turn): ModelTurn {
  const assistant = turn.assistant;
  return { calls: (assistant.calls ?? []).map(call => [call.tool, call.arguments] as [string, Record<string, unknown>]),
    text: assistant.content ?? '', reasoning: assistant.reasoning ?? undefined,
    ...(Object.hasOwn(assistant, 'execution_plan') ? { execution_plan: assistant.execution_plan ?? null } : {}) };
}

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

/**
 * Calls in the order they start: a call is identified by its number, not its opening text, which changes with the
 * runtime (a recorded row's openings can predate the current wording). The same code starts the same calls in the
 * same order; when it does not, the replay diverges and is not accepted.
 */
function callNumbers(trajectory: Turn[]): number[] {
  const seen = new Map<string, number>();
  return trajectory.map(turn => { const key = openingText(turn.context);
    if (!seen.has(key)) seen.set(key, seen.size);
    return seen.get(key)!; });
}

/** A turn's place: its call's number and its number among that call's turns. */
function placeOf(calls: number[], index: number) {
  return { call: calls[index]!, nth: calls.slice(0, index).filter(call => call === calls[index]).length };
}

/** What a call's turn showed: the tool results in the context of that call's next turn, or the run's end. */
function outputOf(trajectory: Turn[], call: number, nth: number, ending: string): string {
  const calls = callNumbers(trajectory), turns = trajectory.filter((_, index) => calls[index] === call);
  const turn = turns[nth], next = turns[nth + 1];
  if (!turn) return '(missing)';
  if (!next) return ending;
  return next.context.slice(turn.context.length + 1).filter(message => message.role === 'tool')
    .map(message => text(message.content)).join('\n');
}

/** Replay a row's recorded responses, call by call in order, leaving out some turns and changing the reasoning of one. */
async function replay(row: TeacherRow, options: ReplayOptions, runId: string, leftOut: Set<number>,
    reasoned?: { index: number; reasoning: string; planned: boolean }) {
  const trajectory = row.trajectory as unknown as Turn[], recordedCalls = callNumbers(trajectory);
  const script = new Map<number, ModelTurn[]>();
  trajectory.forEach((turn, index) => {
    if (leftOut.has(index)) return;
    const response = recorded(turn);
    if (reasoned?.index === index) {
      if (reasoned.planned) response.execution_plan = reasoned.reasoning; else response.reasoning = reasoned.reasoning;
    }
    script.set(recordedCalls[index]!, [...script.get(recordedCalls[index]!) ?? [], response]);
  });
  const started = new Map<string, number>(), used = new Map<number, number>(), turns: Record<string, unknown>[] = [];
  const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
    const key = openingText(request.messages as Message[]);
    if (!started.has(key)) started.set(key, started.size);
    const call = started.get(key)!, at = used.get(call) ?? 0;
    used.set(call, at + 1);
    const response = script.get(call)?.[at] ??
      { calls: [['return_result', { status: 'failed', reason: 'The replay ran past the recorded trajectory.' }]] as [string, Record<string, unknown>][] };
    turns.push(trajectoryTurn(request, response));
    return response;
  };
  const record = (row.task as { program_ir: ProgramRecord }).program_ir;
  const run = await executeProgram(record, driver, { ...options, runId });
  return { run, turns: turns as unknown as Turn[], ending: JSON.stringify([run.outcome.status, run.outcome.value]) };
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
  const error = outputOf(trajectory, failedPlace.call, failedPlace.nth, '');
  if (!reasoningFitsFix(reasoning, error, first.assistant.calls, fix.assistant.calls))
    return { rejected: 'the reasoning before the failed attempt names what failed' };
  const provenance = row.provenance as Record<string, unknown>;
  const runId = programRunId(0, provenance);
  const control = await replay(row, options, runId, new Set());
  if (!control.run.outcome.accepted) return { rejected: 'the row does not replay to an accepted end under the current runtime' };
  const variant = await replay(row, options, runId, new Set(site.failed), { index: site.fixed, reasoning, planned });
  if (!variant.run.outcome.accepted) return { rejected: 'the program no longer ends accepted without the failed attempts' };
  const nthInVariant = place.nth - site.failed.length;
  if (outputOf(variant.turns, place.call, nthInVariant, variant.ending) !== outputOf(control.turns, place.call, place.nth, control.ending))
    return { rejected: 'the fix shows something other than it showed after the failure' };
  const variantCalls = callNumbers(variant.turns);
  const decision = variantCalls.findIndex((call, index) => call === place.call &&
    variantCalls.slice(0, index).filter(other => other === call).length === nthInVariant);
  const record = (row.task as { program_ir: ProgramRecord }).program_ir;
  const variantRow = programRow(record, String(provenance.model), runId, { ...provenance,
    variant: { version: CORRECTION_VERSION, parent: row.id, left_out: site.failed, fixed: site.fixed, decision } },
    variant.run, variant.turns as unknown as Record<string, unknown>[]);
  return { row: { ...variantRow, id: `${row.id}:corrected:${site.failed[0]}` }, decision };
}
