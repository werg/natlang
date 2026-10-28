import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { trajectoryTurn } from './trajectory.js';
import { openingLength, openingText, text, type Message } from './conversation.js';
export type Place = { call: number; nth: number };
export type Turn = { context: Message[]; model_response?: { prompt_tokens?: number; completion_tokens?: number }; assistant: { reasoning?: string | null; execution_plan?: string | null;
  content?: string; calls?: { tool: string; arguments: Record<string, unknown> }[] } };

/**
 * What a turn saw beyond its call's opening: the tool results of the call's earlier turns, without what differs from
 * run to run and says nothing about the task: a live value's per-process number and time taken in milliseconds.
 */
export function observed(context: Message[]): string[] {
  return context.slice(openingLength(context)).filter(message => message.role === 'tool').map(message => text(message.content)
    .replace(/ #\d+; live value/g, ' #; live value').replace(/\b(\w*Ms): \d+/g, '$1: _'));
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
