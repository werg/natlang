/**
 * Replaying recorded model turns under the current runtime. A recorded call is numbered by the order it first asked
 * the model; a replayed call is matched to the recorded call with its opening text, or else to the most similar one
 * (openings change wording with the runtime, and concurrent calls ask in no fixed order). A replay that goes wrong
 * anyway is caught by comparing what it shows.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { trajectoryTurn } from './collector.js';
import { openingLength, openingText, text, type Message } from './opening.js';

export type Turn = { context: Message[]; assistant: { reasoning?: string | null; execution_plan?: string | null;
  content?: string; calls?: { tool: string; arguments: Record<string, unknown> }[] } };

/** A turn's place: its call's number and its number among that call's turns. */
export type Place = { call: number; nth: number };

/** The response a recorded turn gave, as the reply to replay. */
export function recorded(turn: Turn): ModelTurn {
  const assistant = turn.assistant;
  return { calls: (assistant.calls ?? []).map(call => [call.tool, call.arguments] as [string, Record<string, unknown>]),
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

/**
 * What a turn saw beyond its call's opening: the tool results of the call's earlier turns, without what differs from
 * run to run and says nothing about the task: a live value's per-process number and time taken in milliseconds.
 */
export function observed(context: Message[]): string[] {
  return context.slice(openingLength(context)).filter(message => message.role === 'tool').map(message => text(message.content)
    .replace(/ #\d+; live value/g, ' #; live value').replace(/\b(\w*Ms): \d+/g, '$1: _'));
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

export const REPLAY_END: ModelTurn = {
  calls: [['return_result', { status: 'failed', reason: 'The replay ran past the recorded trajectory.' }]] };

const wordsOf = (text: string) => new Set(text.split(/\W+/).filter(Boolean));

/**
 * Places requests as they arrive, as turns of the recorded calls whose openings are given: a call with the same opening
 * text as a recorded call not yet matched is that call; otherwise the unmatched one sharing most of its words, if it
 * shares at least half; otherwise a call the recording does not have, numbered after the recorded ones.
 */
export function callMatcher(openings: string[]): (request: ModelTurnRequest) => Place {
  const matched = new Map<string, number>(), taken = new Set<number>(), used = new Map<number, number>();
  const recordedWords = openings.map(wordsOf);
  let extra = openings.length;
  const match = (text: string): number => {
    const exact = openings.findIndex((opening, call) => !taken.has(call) && opening === text);
    if (exact >= 0) return exact;
    const words = wordsOf(text);
    let best = -1, score = 0.5;
    recordedWords.forEach((other, call) => {
      if (taken.has(call)) return;
      const shared = [...words].filter(word => other.has(word)).length;
      const similarity = shared / Math.max(1, new Set([...words, ...other]).size);
      if (similarity >= score) { best = call; score = similarity; }
    });
    return best >= 0 ? best : extra++;
  };
  return request => {
    const text = openingText(request.messages as Message[]);
    if (!matched.has(text)) { const call = match(text); matched.set(text, call); taken.add(call); }
    const call = matched.get(text)!, nth = used.get(call) ?? 0;
    used.set(call, nth + 1);
    return { call, nth };
  };
}

/**
 * A driver that answers each call's turns from its script, in order, and every other turn with `rest`. It records the
 * turns it answered as trajectory turns, with their places and whether the script answered them.
 */
export function scriptedDriver(script: ModelTurn[][], openings: string[],
    rest: (request: ModelTurnRequest, place: Place) => Promise<ModelTurn> = async () => REPLAY_END) {
  const place = callMatcher(openings), turns: Record<string, unknown>[] = [], places: (Place & { scripted: boolean })[] = [];
  const driver = async (request: ModelTurnRequest): Promise<ModelTurn> => {
    const at = place(request), scripted = script[at.call]?.[at.nth];
    const response = scripted ? structuredClone(scripted) : await rest(request, at);
    turns.push(trajectoryTurn(request, response));
    places.push({ ...at, scripted: !!scripted });
    return response;
  };
  return { driver, turns: turns as unknown as Turn[], places };
}
