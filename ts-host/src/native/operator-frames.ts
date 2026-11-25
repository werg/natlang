/**
 * Learned Neuralese operators: the sites the runtime compacts with and how a consumer is shown their state
 * (plans/neuralese/OPERATORS_RESTORE_20261117.md; registry in training/neuralese/natlang_neuralese/model/learned_operators.py).
 * An operator write is an ordinary model call at its site plus the in-context capture of its reply (`NativeCapture`);
 * compact-history captures the model's own `compact_history` note in place.
 *
 * Whether the runtime compacts with a learned operator or keeps the crisp text cut is the `compaction` setting of the
 * native agent (a pluggable part: `crisp` keeps the cut, `nl` uses the learned operator).
 */
import { VIEW_PROMPT } from '../builtin/index.js';
import { shownCharsFor } from './cutoff.js';
import { neuraleseRefSentinel, type NeuraleseRef } from './neuralese.js';

export type ValueOperator = 'compact-value' | 'compact-params' | 'extract';
export type CompactSite = { operator: ValueOperator; instructions: string; value: string; note?: string };
/** One reply of a model call to capture in its own context: its request and, when the server reported them, its exact
 * prompt and sampled token ids (else its text and reasoning). */
export type CapturedCall = { messages: unknown[]; tools: unknown[]; prompt?: number[]; reply?: number[];
  /** Instead of `prompt`: the exact prompt with the native states it imported where they stood (an operator call under
   * its soft prompt state), as the call's RL capture recorded it. */
  promptSegments?: ({ kind: 'tokens'; token_ids: number[] } | { kind: 'native_state'; id: string })[];
  text?: string; reasoning?: string;
  /** Capture the tokens of this text inside the reply (a `compact_history` note in its tool call) instead of the
   * reply's content. */
  select?: string;
  /** Write this text as the span after the whole reply (a Neuralese text write in its producing call,
   * `NeuralesePort.write`'s `context`), instead of capturing the reply's own tokens. */
  write?: string;
  /** The chat-template switches the call was rendered with (`{enable_thinking: false}` for an operator's call): a
   * capture that renders or forces the call again renders it the same way. */
  chat_template_kwargs?: Record<string, unknown> };

/**
 * The producing context of one model call (`NativeWriteContext`): the request's messages and tools and, when the
 * server reported its RL capture, the exact prompt (token ids, or segments with the states it imported) and the
 * sampled reply ids. Shared by the operator capture (agent.ts `capturedRef`) and Neuralese text writes.
 */
export function callContext(request: { messages: unknown[]; tools?: unknown[]; thinking?: false },
    response: { raw_response?: Record<string, unknown> }): import('./neuralese-store.js').NativeWriteContext {
  const sampled = response.raw_response?.x_natlang_rl_capture as { prompt?: { kind?: string; token_ids?: number[]; id?: string }[];
    generated_inputs?: { token_id?: number }[] } | undefined;
  const tokens = (segment: { kind?: string; token_ids?: number[] }) => segment.kind === 'tokens' && Array.isArray(segment.token_ids);
  const base = { messages: request.messages, tools: request.tools ?? [],
    ...(request.thinking === false ? { chat_template_kwargs: { enable_thinking: false } } : {}) };
  const exact = !!sampled?.prompt?.length && !!sampled.generated_inputs?.length &&
    sampled.prompt.every(segment => tokens(segment) || (segment.kind === 'native_state' && typeof segment.id === 'string')) &&
    sampled.generated_inputs.every(input => Number.isSafeInteger(input?.token_id));
  if (!exact) return base;
  // A prompt that imported state (an operator's soft prompt state) keeps those imports in place.
  const plain = sampled!.prompt!.every(tokens);
  return { ...base, ...(plain ? { prompt: sampled!.prompt!.flatMap(segment => segment.token_ids!) } :
    { promptSegments: sampled!.prompt!.map(segment => tokens(segment) ? { kind: 'tokens' as const, token_ids: segment.token_ids! } :
      { kind: 'native_state' as const, id: segment.id! }) }),
    reply: sampled!.generated_inputs!.map(input => input.token_id!) };
}

/** The latest producing call of each call id (bounded): a Neuralese text write of that call's value (its typed result, a
 * literal it wrote, an argument it passed to a child call) is written at the end of it. */
const PRODUCING_CALLS = new Map<string, import('./neuralese-store.js').NativeWriteContext>();
export function noteProducingCall(callId: string, context: import('./neuralese-store.js').NativeWriteContext): void {
  PRODUCING_CALLS.delete(callId);
  PRODUCING_CALLS.set(callId, context);
  while (PRODUCING_CALLS.size > 4096) PRODUCING_CALLS.delete(PRODUCING_CALLS.keys().next().value!);
}
export function producingCall(callId: string | null | undefined): import('./neuralese-store.js').NativeWriteContext | undefined {
  return callId ? PRODUCING_CALLS.get(callId) : undefined;
}
export type NativeCapture = (call: CapturedCall, options?: { type?: string }) =>
  Promise<import('./neuralese-store.js').NeuraleseBlockMeta>;

/**
 * How a consumer is shown an operator's state: open + state + close. Must equal the Python registry's
 * `consumer_open`/`consumer_close` (tests/neuralese/test_learned_operators.py checks this file).
 */
export const OPERATOR_FRAMES: Record<CompactSite['operator'] | 'compact-history', { open: string; close: string }> = {
  'compact-value': { open: 'Compacted value: ', close: '\n\n' },
  'compact-params': { open: 'Compacted argument: ', close: '\n\n' },
  'compact-history': { open: 'Earlier steps of this session, compacted: ', close: '\n\nContinue from the most recent steps.' },
  'extract': { open: 'Extracts from the documents, one per document:\n', close: '\n\n' },
};

/** The framed sentinel of an operator's state, as text the driver turns into a native-state part. */
export function framedState(operator: CompactSite['operator'] | 'compact-history', ref: NeuraleseRef): string {
  const frame = OPERATOR_FRAMES[operator];
  return frame.open + neuraleseRefSentinel(ref) + frame.close;
}

/** The operators' reply caps in tokens (learned_operators.py `seed_cap`). */
export const OPERATOR_SPAN_CAPS: Record<CompactSite['operator'] | 'compact-history', number> = {
  'compact-value': 512, 'compact-params': 384, 'compact-history': 1024, 'extract': 256 };

/**
 * With learned compaction selected, a tool output or argument longer than this many characters (~2k tokens) is cut and
 * compacted, whatever the call's context budget: learned compaction keeps large values out of the context, so it does
 * not wait until a value would overflow the budget's page (OPERATORS_RESTORE_20261117.md §10). The agent option
 * `compactionChars` and env NATLANG_NEURALESE_COMPACTION_CHARS set it.
 */
export const OPERATOR_COMPACT_CHARS = 8000;

/** The characters a message shows of one output: the budget's page (`shownCharsFor`), and with learned compaction
 * selected at most `compactionChars`. */
export function compactionPageChars(contextTokens: number | null | undefined, compactionChars?: number): number {
  const page = shownCharsFor(contextTokens);
  return compactionChars === undefined ? page : Math.min(page, compactionChars);
}

/** Values shorter than this keep the text cut only: a structural cut of a small value does not spend an operator call
 * (the case floor, COMPACTION_OPERATOR_CONVERTERS.md §3b). */
export const OPERATOR_MIN_CHARS = 800;

/**
 * The write site of compact-value and compact-params: the builtin `view(value, instructions)` (one summarizer family,
 * DECISIONS.md 2026-10-09). Must equal training/neuralese/natlang_neuralese/view.py `view_site`
 * (tests/neuralese/test_learned_operators.py checks this file).
 */
export function viewSiteMessages(value: string, instructions?: string): { role: 'system' | 'user'; content: string }[] {
  return [{ role: 'system', content: VIEW_PROMPT },
    { role: 'user', content: `${instructions ? `instructions:\n${instructions}\n\n` : ''}value:\n${value}` }];
}

/** The purpose of a compact-value site: a cut-off tool or service output read for the calling function's purpose. */
export function valueInstructions(tool: string, purpose: string): string {
  return `Read this ${tool} result for: ${purpose}`;
}

/** The purpose of a compact-params site: a cut-off argument of the call that receives it. */
export function argumentInstructions(name: string, type: string, purpose: string): string {
  return `The call that receives this argument as ${name} (${type}) answers: ${purpose}`;
}

/** A call's instructions as an operator's purpose text (bounded). */
export function operatorPurpose(body: string): string {
  const text = body.replace(/\n+$/, '');
  return text.length > 2000 ? text.slice(0, 2000) : text;
}

