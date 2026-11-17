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
import { neuraleseRefSentinel, type NeuraleseRef } from './neuralese.js';

export type ValueOperator = 'compact-value' | 'compact-params' | 'extract';
export type CompactSite = { operator: ValueOperator; instructions: string; value: string; note?: string };
/** One reply of a model call to capture in its own context: its request and, when the server reported them, its exact
 * prompt and sampled token ids (else its text and reasoning). */
export type CapturedCall = { messages: unknown[]; tools: unknown[]; prompt?: number[]; reply?: number[];
  text?: string; reasoning?: string;
  /** Capture the tokens of this text inside the reply (a `compact_history` note in its tool call) instead of the
   * reply's content. */
  select?: string };
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
