import { callMatcher, observed, scriptedDriver } from '../runtime/trajectory-replay.js';
export { callMatcher, observed, scriptedDriver, REPLAY_END } from '../runtime/trajectory-replay.js';
/**
 * Replaying recorded model turns under the current runtime. A recorded call is numbered by the order it first asked
 * the model; a replayed call is matched to the recorded call with its opening text, or else to the most similar one
 * (openings change wording with the runtime, and concurrent calls ask in no fixed order). A replay that goes wrong
 * anyway is caught by comparing what it shows.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { openingLength, openingText, text, type Message } from '../runtime/conversation.js';

export type { Turn } from '../runtime/trajectory-replay.js';
import type { Turn } from '../runtime/trajectory-replay.js';

/** A turn's place: its call's number and its number among that call's turns. */
export type Place = { call: number; nth: number };

/** The response a recorded turn gave, as the reply to replay. */
export function recorded(turn: Turn): ModelTurn {
  const assistant = turn.assistant;
  return { calls: (assistant.calls ?? []).map(call => [call.tool, call.arguments] as [string, Record<string, unknown>]),
    // Usage calibrates the agent's context estimate and determines forced compaction. Preserve it in replay.
    ...(turn.model_response?.prompt_tokens === undefined ? {} : { prompt_tokens: turn.model_response.prompt_tokens }),
    ...(turn.model_response?.completion_tokens === undefined ? {} : { completion_tokens: turn.model_response.completion_tokens }),
    text: assistant.content ?? '', reasoning: assistant.reasoning ?? undefined,
    ...(Object.hasOwn(assistant, 'execution_plan') ? { execution_plan: assistant.execution_plan ?? null } : {}) };
}

/** Each turn's call number, calls numbered in the order they first ask the model. */
export function callNumbers(trajectory: Turn[]): number[] {
  const seen = new Map<string, number>();
  return trajectory.map(turn => { const key = openingText(turn.context);
    if (!seen.has(key)) seen.set(key, seen.size);
    return seen.get(key)!; });
}

export function placeOf(calls: number[], index: number): Place {
  return { call: calls[index]!, nth: calls.slice(0, index).filter(call => call === calls[index]).length };
}

/** The flat index of a place in a trajectory, or -1. */
export function indexOf(trajectory: Turn[], place: Place): number {
  const calls = callNumbers(trajectory);
  return calls.findIndex((call, index) => call === place.call &&
    calls.slice(0, index).filter(other => other === call).length === place.nth);
}

/** The opening text of each recorded call, by call number. */
export function openingsOf(trajectory: Turn[]): string[] {
  const openings: string[] = [];
  for (const turn of trajectory) { const text = openingText(turn.context); if (!openings.includes(text)) openings.push(text); }
  return openings;
}

/** A script per call: the recorded responses of each call, in order, leaving out some turns. */
export function scriptOf(trajectory: Turn[], leftOut = new Set<number>()): ModelTurn[][] {
  const calls = callNumbers(trajectory), script: ModelTurn[][] = [];
  trajectory.forEach((turn, index) => { if (!leftOut.has(index)) (script[calls[index]!] ??= []).push(recorded(turn)); });
  return script;
}

/**
 * The turns of a handoff run (teacher/handoff.ts) that replayed the handoff's prefix, and the turn handed over (-1 when
 * the run never reached it). Derived from the run's own turns: a collector sees requests arrive concurrently, before
 * their turns take their places in the trajectory.
 */
export function handoffTurns(trajectory: Turn[], handoff: { prefix: ModelTurn[][]; openings: string[]; call: number }):
    { prefix: number[]; at: number } {
  const place = callMatcher(handoff.openings), prefix: number[] = [];
  let at = -1;
  trajectory.forEach((turn, index) => {
    const { call, nth } = place({ messages: turn.context } as unknown as ModelTurnRequest);
    const replayed = handoff.prefix[call]?.length ?? 0;
    if (nth < replayed) prefix.push(index);
    else if (call === handoff.call && nth === replayed) at = index;
  });
  return { prefix, at };
}
