/**
 * Runtime-owned prompt pieces and their soft forms (plans/neuralese/DECISIONS.md 40, 41).
 *
 * Every text the runtime itself puts in front of the model (the central interpreter prompt and its extensions, the
 * decision readout's prompt, the fixed paragraphs of the predicate prompt, compaction frames, turn notices) is a piece
 * with a stable ID. A piece's soft form is a `Neuralese<SystemPrompt>` value, initialised from the piece's text and then
 * trained with the base model or adapted by self-improvement. A bank maps piece IDs to soft forms.
 *
 * Softening is one substitution: under a Neuralese driver, each banked piece's exact text in an outgoing message
 * becomes its block. Pieces are matched longest first, so a variant (the depth-limit interpreter) wins over a piece it
 * contains. Text that is not a banked piece stays text. The training-data converter uses the same registry to find
 * pieces in recorded conversations.
 */
import { DECISION_SYSTEM_PROMPT } from './decision.js';
import { isNeuraleseRef, neuraleseSentinel, type NeuraleseRef } from './neuralese.js';
import { APPROACH_PROMPT, AUTOMATIC_NOTE, COMPACTED_RESULT, COMPACTION_NOTICE, directoryReducerPrompt, FILE_TOOL_SURFACES, FUNCTION_TOOLS_PROMPT,
  GENERATION_GUIDANCE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN, LAST_TURN_NOTICE, NL_DEPTH_LIMIT_NOTICE, promptAtNlDepthLimit,
  TOOLS_PROMPT } from './prompt.js';
import { predicatePrompt } from '../runtime/iterate.js';

export const SYSTEM_PROMPT_TYPE = 'Neuralese<SystemPrompt>';

export type PromptPiece = { readonly id: string; readonly text: string };

/** The fixed paragraphs of the predicate prompt: those that do not depend on the step counts. */
function predicatePieces(): PromptPiece[] {
  const variants = [predicatePrompt(0, 0), predicatePrompt(1, 0), predicatePrompt(3, 2)].map(text => text.split('\n'));
  return variants[0]!.filter(paragraph => variants.every(variant => variant.includes(paragraph)))
    .map((text, index) => ({ id: `predicate/${index}`, text }));
}

/** The current runtime's prompt pieces. IDs are stable across runtime versions; texts are this version's. */
export function promptPieces(): PromptPiece[] {
  const pieces: PromptPiece[] = [
    { id: 'interpreter', text: TOOLS_PROMPT },
    { id: 'interpreter/depth-limit', text: promptAtNlDepthLimit(TOOLS_PROMPT) },
    { id: 'depth-limit-notice', text: NL_DEPTH_LIMIT_NOTICE },
    { id: 'function-tools', text: FUNCTION_TOOLS_PROMPT },
    ...FILE_TOOL_SURFACES.flatMap(surface => [
      { id: `directory-reducer/${surface}`, text: directoryReducerPrompt(surface, true) },
      { id: `directory-reducer/${surface}/depth-limit`, text: directoryReducerPrompt(surface, false) }]),
    { id: 'generation-guidance', text: GENERATION_GUIDANCE },
    { id: 'approach', text: APPROACH_PROMPT },
    { id: 'decision', text: DECISION_SYSTEM_PROMPT },
    ...predicatePieces(),
    { id: 'compaction-notice', text: COMPACTION_NOTICE },
    { id: 'automatic-note', text: AUTOMATIC_NOTE },
    { id: 'handover/open', text: HANDOVER_NOTE_OPEN },
    { id: 'handover/close', text: HANDOVER_NOTE_CLOSE },
    { id: 'last-turn-notice', text: LAST_TURN_NOTICE },
    { id: 'compacted-result', text: COMPACTED_RESULT },
  ];
  const seen = new Set<string>();
  return pieces.filter(piece => piece.text.trim() && !seen.has(piece.text) && seen.add(piece.text));
}

/** Soft forms by piece ID, with the text each was registered for (the text that is replaced). */
export type SystemPromptBank = ReadonlyMap<string, { readonly text: string; readonly value: NeuraleseRef }>;

/** A bank from soft forms by piece ID; each piece's text comes from `pieces` (default: this runtime's pieces). */
export function systemPromptBank(values: Readonly<Record<string, unknown>>, pieces: readonly PromptPiece[] = promptPieces()): SystemPromptBank {
  const texts = new Map(pieces.map(piece => [piece.id, piece.text]));
  const bank = new Map<string, { text: string; value: NeuraleseRef }>();
  for (const [id, value] of Object.entries(values)) {
    const text = texts.get(id);
    if (text === undefined) throw new Error(`system-prompt-unknown-piece: ${id} is not a prompt piece of this runtime`);
    if (!isNeuraleseRef(value)) throw new TypeError(`system prompt piece ${id}: not a Neuralese value`);
    bank.set(id, { text, value });
  }
  return bank;
}

/** Where pieces occur in a text, longest pieces first and without overlaps: [start, end, piece]. */
export function findPieces(text: string, pieces: readonly PromptPiece[]): [number, number, PromptPiece][] {
  const found: [number, number, PromptPiece][] = [];
  const taken = (start: number, end: number) => found.some(([s, e]) => start < e && s < end);
  for (const piece of [...pieces].sort((a, b) => b.text.length - a.text.length)) {
    if (!piece.text) continue;
    for (let at = text.indexOf(piece.text); at >= 0; at = text.indexOf(piece.text, at + piece.text.length))
      if (!taken(at, at + piece.text.length)) found.push([at, at + piece.text.length, piece]);
  }
  return found.sort((a, b) => a[0] - b[0]);
}

/** A text with each banked piece replaced by its block's sentinel. */
export function softenText(text: string, bank: SystemPromptBank): string {
  if (!bank.size) return text;
  const pieces = [...bank].map(([id, entry]) => ({ id, text: entry.text }));
  let out = '', last = 0;
  for (const [start, end, piece] of findPieces(text, pieces)) {
    out += text.slice(last, start) + neuraleseSentinel(bank.get(piece.id)!.value.$neuralese.id);
    last = end;
  }
  return out + text.slice(last);
}

let scopedSource: () => SystemPromptBank | undefined = () => undefined;

/** Install where scoped soft prompts come from (the learning module's `withSystemPrompts`). */
export function setSystemPromptSource(next: () => SystemPromptBank | undefined): void { scopedSource = next; }

/** The bank in force: the runtime's bank with the pieces of the current scope (`withSystemPrompts`) over it. */
export function activeSystemPrompts(base: SystemPromptBank | undefined): SystemPromptBank | undefined {
  const scoped = scopedSource();
  if (!scoped?.size) return base;
  return base?.size ? new Map([...base, ...scoped]) : scoped;
}

/** Messages with the banked pieces of their text content softened. Model replies (assistant turns) are left alone. */
export function softenMessages<M extends Record<string, unknown>>(messages: readonly M[], bank: SystemPromptBank | undefined): M[] {
  if (!bank?.size) return [...messages];
  return messages.map(message => message.role !== 'assistant' && typeof message.content === 'string' ?
    { ...message, content: softenText(message.content, bank) } : message);
}
