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
import { APPROACH_PROMPT, AUTOMATIC_NOTE, COMPACTED_RESULT, COMPACTION_NOTICE, directoryReducerPrompt, DIRECTORY_REDUCER_DECLARATION, FILE_TOOL_SURFACES, FUNCTION_TOOLS_PROMPT,
  GENERATION_GUIDANCE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN, LAST_TURN_NOTICE, NL_DEPTH_LIMIT_NOTICE, promptAtNlDepthLimit,
  SEMANTIC_RESULT_PROMPT, TEXT_NEURALESE_EMULATION_PROMPT, TOOLS_PROMPT } from './prompt.js';
import { VIEW_PROMPT } from '../builtin/index.js';
import { DEFAULT_JUDGE_INSTRUCTIONS, predicatePrompt } from '../runtime/iterate.js';

export const SYSTEM_PROMPT_TYPE = 'Neuralese<SystemPrompt>';

export type PromptRendering = 'full' | 'program-header';
export type PromptPiece = { readonly id: string; readonly text: string; readonly scope: 'generic' | 'program' };

/** The fixed paragraphs of the predicate prompt: those that do not depend on the step counts. */
function predicatePieces(): PromptPiece[] {
  const variants = [predicatePrompt(0, 0), predicatePrompt(1, 0), predicatePrompt(3, 2)].map(text => text.split('\n'));
  return variants[0]!.filter(paragraph => variants.every(variant => variant.includes(paragraph)))
    .map((text, index) => ({ id: `predicate/${index}`, text, scope: 'generic' as const }));
}

/** The current runtime's prompt pieces. IDs are stable across runtime versions; texts are this version's. */
export function promptPieces(): PromptPiece[] {
  const pieces: PromptPiece[] = [
    { id: 'interpreter', text: TOOLS_PROMPT, scope: 'generic' },
    { id: 'interpreter/depth-limit', text: promptAtNlDepthLimit(TOOLS_PROMPT), scope: 'generic' },
    { id: 'depth-limit-notice', text: NL_DEPTH_LIMIT_NOTICE, scope: 'program' },
    { id: 'directory-reducer/declaration', text: DIRECTORY_REDUCER_DECLARATION, scope: 'program' },
    { id: 'neuralese/text-emulation', text: TEXT_NEURALESE_EMULATION_PROMPT, scope: 'program' },
    { id: 'neuralese/string-boundaries', text: SEMANTIC_RESULT_PROMPT.split('\n\n')[1]!, scope: 'program' },
    { id: 'function-tools/edit-capability', text: FUNCTION_TOOLS_PROMPT.split('\n')[1]!, scope: 'program' },
    { id: 'function-tools', text: FUNCTION_TOOLS_PROMPT, scope: 'generic' },
    ...FILE_TOOL_SURFACES.flatMap<PromptPiece>(surface => [
      { id: `directory-reducer/${surface}`, text: directoryReducerPrompt(surface, true), scope: 'generic' },
      { id: `directory-reducer/${surface}/depth-limit`, text: directoryReducerPrompt(surface, false), scope: 'generic' },
      { id: `directory-reducer/${surface}/read-only`, text: directoryReducerPrompt(surface, true, false), scope: 'generic' },
      { id: `directory-reducer/${surface}/read-only/depth-limit`, text: directoryReducerPrompt(surface, false, false), scope: 'generic' }]),
    { id: 'generation-guidance', text: GENERATION_GUIDANCE, scope: 'generic' },
    { id: 'approach', text: APPROACH_PROMPT, scope: 'generic' },
    { id: 'decision', text: DECISION_SYSTEM_PROMPT, scope: 'generic' },
    ...predicatePieces(),
    { id: 'compaction-notice', text: COMPACTION_NOTICE, scope: 'generic' },
    { id: 'automatic-note', text: AUTOMATIC_NOTE, scope: 'generic' },
    { id: 'handover/open', text: HANDOVER_NOTE_OPEN, scope: 'generic' },
    { id: 'handover/close', text: HANDOVER_NOTE_CLOSE, scope: 'generic' },
    { id: 'last-turn-notice', text: LAST_TURN_NOTICE, scope: 'generic' },
    { id: 'compacted-result', text: COMPACTED_RESULT, scope: 'generic' },
    { id: 'progress-judge', text: DEFAULT_JUDGE_INSTRUCTIONS, scope: 'generic' },
    { id: 'view', text: VIEW_PROMPT, scope: 'generic' },
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
export function findPieces<P extends Pick<PromptPiece, 'id' | 'text'>>(text: string, pieces: readonly P[]): [number, number, P][] {
  const found: [number, number, P][] = [];
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

/** Remove runtime-owned generic prose, preserving program declarations and caller-supplied additions verbatim.
 * Program pieces can be nested in a larger generic piece (also used by soft prompt banks).
 * Matching the two scopes independently makes those declarations survive the encompassing replacement.
 */
export function renderSystemPrompt(text: string, mode: PromptRendering = 'full', pieces = promptPieces()): string {
  if (mode === 'full') return text;
  const generic = findPieces(text, pieces.filter(piece => piece.scope === 'generic'));
  const program = findPieces(text, pieces.filter(piece => piece.scope === 'program'));
  let out = '', cursor = 0;
  for (const [start, end] of generic) {
    out += text.slice(cursor, start);
    const retained = program.filter(([a, b]) => a >= start && b <= end).map(([a, b]) => text.slice(a, b));
    if (retained.length) out += '\n\n' + retained.join('\n\n') + '\n\n';
    cursor = end;
  }
  return (out + text.slice(cursor)).trim();
}

/** Shared outbound rendering; an empty header has no system turn. History and user content are untouched. */
export function renderPromptMessages<M extends Record<string, unknown>>(messages: readonly M[], mode: PromptRendering = 'full'): M[] {
  if (mode === 'full') return [...messages];
  return messages.flatMap(message => {
    if (message.role !== 'system' || typeof message.content !== 'string') return [message];
    const content = renderSystemPrompt(message.content, mode);
    return content ? [{ ...message, content }] : [];
  });
}
