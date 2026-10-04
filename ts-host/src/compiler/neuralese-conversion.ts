/**
 * Conversion of natlang trajectory records to Neuralese form (plans/neuralese/DECISIONS.md 41, 42; S5 §2.2).
 *
 * What becomes Neuralese, and why (decision 42): a value is worth a soft form when it is reused (one encoding serves
 * many reads), when one agent produces it and another consumes it (the model writes it and the next reads it, with no
 * text in between), or when it is large and a short digest saves context. A value read once by the call that produced
 * it stays text: turning it into Neuralese costs a pass and saves nothing. Every site is counted, converted or kept
 * exact with its reason.
 *
 * - **Prompt sites** (reused by every call). The system message and runtime-written frames are split into registered
 *   prompt pieces (native/system-prompts.ts); each becomes a soft parameter part `{ type: 'soft', name: 'prompt:<id>' }`.
 *   Text that matches no registered piece (an older runtime's wording) becomes a versioned piece `prompt:system@<sha12>`.
 *   Program guidance (`guidance@<sha12>`) is a soft parameter of its own, adapted by self-improvement.
 * - **Instructions** (stored function bodies). Instructions used by at least `instructionsReuse` distinct calls of the
 *   corpus become a shared soft parameter `instructions@<sha12>` (one parameter for all their calls). A deterministic
 *   `instructionsShare` of single-use instructions converts too, so the position has Neuralese samples; the rest stay
 *   text (`single-use`).
 * - **Handover sites** (written by one turn, read by every later one). A `compact_history` note becomes
 *   `{ $write: { name, type: 'Neuralese<HandoverNote>', source } }` in the call's arguments, and the pinned note
 *   message reads the same block (`{ type: 'read', name, source }` between the soft handover frames). `source` is the
 *   crisp note: the teacher's view; the read carries it too, since the producing call may lie outside the record.
 * - **Digest sites** (large values in the opening listing). A value the listing cuts off becomes
 *   `{ type: 'digest', name, source, preview }` when the record has the full value (the root call's inputs): a short
 *   block the digest operator writes from the full value, shown in place of the cut-off preview while the value itself
 *   stays in scope for exact access. Without the full value the preview stays (`full-value-unavailable`).
 *
 * Kept exact, counted: tool outputs (`single-use`), results of child `nl` calls printed by eval (`needs-graph-record`:
 * a handoff, but these per-call records do not hold the producing call), `nl` literals in eval code
 * (`later-curriculum-step`), turn-count notices (`dynamic-text`).
 *
 * Initialisation: soft parameters start from their text encoded in one forward pass through the port (`encode`), not
 * from a summarising call. Their texts are collected once in `pieces`.
 */
import { createHash } from 'node:crypto';
import { promptPieces, findPieces, type PromptPiece } from '../native/system-prompts.js';
import { AUTOMATIC_NOTE, DIGEST_PROMPT, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN } from '../native/prompt.js';

export const NEURALESE_CONVERSION_VERSION = 'natlang.neuralese-conversion/2';
export const HANDOVER_TYPE = 'Neuralese<HandoverNote>';

export type ConvertedPart = { type: 'text'; text: string } | { type: 'soft'; name: string } | { type: 'read'; name: string; source: string } |
  { type: 'digest'; name: string; holder: string; value_type: string; source: string; preview: string };
type Message = Record<string, unknown> & { role: string; content?: unknown; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] };
export type SoftPiece = { name: string; kind: 'system-prompt' | 'program-guidance' | 'function-body'; text: string };
export type SiteCounts = Record<string, { converted: number; exact: Record<string, number> }>;
export type ConversionOptions = {
  pieces?: readonly PromptPiece[];
  /** Distinct calls per instructions digest over the corpus (`instructionsDigest`); without it every instructions
   * site counts as single-use. */
  instructionCalls?: ReadonlyMap<string, number>;
  instructionsReuse?: number;  // default 2
  instructionsShare?: number;  // default 0.1
};

/** The digest under which instructions are counted for reuse. */
export const instructionsDigest = (text: string) => sha12(text);
/** The instructions of a record's opening, if it has them. */
export function openingInstructions(record: { messages: readonly Message[] }): string | undefined {
  const opening = record.messages.find(message => message.role === 'user');
  return typeof opening?.content === 'string' ? INSTRUCTIONS.exec(opening.content)?.[2] : undefined;
}
/** The call a record belongs to: every turn of one call shares it. */
export const callOf = (record: Record<string, unknown>) =>
  String((record.source_ref as { trajectory_id?: unknown } | undefined)?.trajectory_id ?? record.id ?? '');

const sha12 = (text: string) => createHash('sha256').update(text).digest('hex').slice(0, 12);
const GUIDANCE = /\n\n<natlang_program_guidance>\n([\s\S]*?)\n<\/natlang_program_guidance>\n/;
const INSTRUCTIONS = /(Instructions:\n)([\s\S]*?)(\n\n(?:In eval|Eval also|$))/;
const NL_LITERAL = /\bnl(?:\.with\([^)]*\))?(?:<[^`]*?>)?`/g;
const DYNAMIC_NOTICE = /\n\n\[\d+ turns left in this call\.[^\]]*\]$/;
const LISTING_LINE = /^(\w+): ([^=\n]*?) = (.*<<cut off: [^\n]*)$/gm;
const OPENING_CALL = /^You are inside this call: (\w+)\(/;

/** The record's parts and soft pieces, with site counts. */
export function convertTrajectory<R extends { messages: Message[]; target?: Message }>(record: R, options: ConversionOptions = {}):
    { record: R & { neuralese_conversion: { version: string; sites: SiteCounts } }; pieces: SoftPiece[] } {
  const registered = options.pieces ?? promptPieces();
  const pieces = new Map<string, SoftPiece>();
  const sites: SiteCounts = {};
  const count = (kind: string, reason?: string, n = 1) => {
    const site = sites[kind] ??= { converted: 0, exact: {} };
    if (reason) site.exact[reason] = (site.exact[reason] ?? 0) + n; else site.converted += n;
  };
  const soft = (name: string, kind: SoftPiece['kind'], text: string): ConvertedPart => {
    if (!pieces.has(name)) pieces.set(name, { name, kind, text });
    return { type: 'soft', name };
  };
  /** Registered pieces as soft parts; the rest as text, or (`unmatched`) as a versioned piece. */
  const promptParts = (text: string, unmatched: 'text' | 'versioned'): ConvertedPart[] => {
    const parts: ConvertedPart[] = [];
    const rest = (chunk: string) => {
      if (!chunk) return;
      if (unmatched === 'versioned' && chunk.trim()) {
        parts.push(soft(`prompt:system@${sha12(chunk)}`, 'system-prompt', chunk));
        count('prompt', undefined);
      } else parts.push({ type: 'text', text: chunk });
    };
    let last = 0;
    for (const [start, end, piece] of findPieces(text, registered)) {
      rest(text.slice(last, start));
      parts.push(soft(`prompt:${piece.id}`, 'system-prompt', piece.text));
      count('prompt');
      last = end;
    }
    rest(text.slice(last));
    return parts;
  };
  const handoverName = (note: string) => `handover:${sha12(note.trim())}`;
  // Eval calls that run child natural-language calls: their printed results are another call's output.
  const childCalls = new Set<string>();
  for (const message of record.messages) for (const call of message.tool_calls ?? []) {
    const args = call.function.name === 'eval' ? parseArguments(call.function.arguments) : undefined;
    if (typeof args?.code === 'string' && NL_LITERAL.test(args.code)) childCalls.add(String(call.id));
    NL_LITERAL.lastIndex = 0;
  }
  // The root call's full inputs, when this record is the root call: the sources of its listing digests.
  const semantics = ((record as Record<string, unknown>).task as { program_ir?: { semantics?: { root?: string; inputs?: Record<string, unknown> } } } | undefined)
    ?.program_ir?.semantics;
  const opening = record.messages.find(message => message.role === 'user')?.content;
  const callName = typeof opening === 'string' ? OPENING_CALL.exec(opening)?.[1] : undefined;
  const inputs = semantics?.inputs && callName && semantics.root === `${callName}.nl` ? semantics.inputs : undefined;
  /** The opening listing with each cut-off value whose full value is known as a digest site. */
  const listingParts = (text: string): ConvertedPart[] => {
    const parts: ConvertedPart[] = [];
    let last = 0;
    for (const match of text.matchAll(LISTING_LINE)) {
      const [line, name, valueType, preview] = match;
      if (!inputs || !(name! in inputs)) { count('digest', 'full-value-unavailable'); continue; }
      const source = JSON.stringify(inputs[name!]);
      const at = match.index! + line!.length - preview!.length;
      parts.push({ type: 'text', text: text.slice(last, at) },
        { type: 'digest', name: `digest:${sha12(source)}`, holder: name!, value_type: valueType!, source, preview: preview! });
      // The digest operator's instructions are a prompt piece: soft and trained with the rest.
      soft('prompt:digest', 'system-prompt', DIGEST_PROMPT);
      count('digest');
      last = at + preview!.length;
    }
    if (last < text.length) parts.push({ type: 'text', text: text.slice(last) });
    return parts.length ? parts : [{ type: 'text', text }];
  };

  const convertMessage = (message: Message, index: number): Message => {
    if (message.role === 'system' && typeof message.content === 'string') {
      const guidance = GUIDANCE.exec(message.content);
      if (!guidance) return { ...message, content: promptParts(message.content, 'versioned') };
      count('program-guidance');
      return { ...message, content: [...promptParts(message.content.slice(0, guidance.index), 'versioned'),
        { type: 'text', text: '\n\n<natlang_program_guidance>\n' },
        soft(`guidance@${sha12(guidance[1]!)}`, 'program-guidance', guidance[1]!),
        { type: 'text', text: '\n</natlang_program_guidance>\n' },
        ...promptParts(message.content.slice(guidance.index + guidance[0].length), 'versioned')] };
    }
    if (message.role === 'user' && typeof message.content === 'string') {
      const text = message.content;
      if (text.startsWith(HANDOVER_NOTE_OPEN) && text.endsWith(HANDOVER_NOTE_CLOSE)) {
        const note = text.slice(HANDOVER_NOTE_OPEN.length, text.length - HANDOVER_NOTE_CLOSE.length);
        count('handover-read');
        return { ...message, content: [soft('prompt:handover/open', 'system-prompt', HANDOVER_NOTE_OPEN),
          { type: 'read', name: handoverName(note), source: note }, soft('prompt:handover/close', 'system-prompt', HANDOVER_NOTE_CLOSE)] };
      }
      if (text === AUTOMATIC_NOTE) return { ...message, content: promptParts(text, 'text') };
      const instructions = INSTRUCTIONS.exec(text);
      if (index <= 1 && instructions) {
        const digest = sha12(instructions[2]!);
        const calls = options.instructionCalls?.get(digest) ?? 1;
        // A deterministic share of single-use instructions, by digest, so every turn of a call agrees.
        const sampled = parseInt(digest.slice(0, 8), 16) / 0xffffffff < (options.instructionsShare ?? 0.1);
        if (calls >= (options.instructionsReuse ?? 2) || sampled) {
          count('instructions', undefined);
          count(calls >= (options.instructionsReuse ?? 2) ? 'instructions-reused' : 'instructions-coverage');
          const at = instructions.index + instructions[1]!.length;
          return { ...message, content: [{ type: 'text', text: text.slice(0, at) },
            soft(`instructions@${digest}`, 'function-body', instructions[2]!),
            { type: 'text', text: text.slice(at + instructions[2]!.length) }] };
        }
        count('instructions', 'single-use');
      }
      return message;
    }
    if (message.role === 'tool' && typeof message.content === 'string') {
      if (DYNAMIC_NOTICE.test(message.content)) count('notice', 'dynamic-text');
      let parts = promptParts(message.content, 'text');
      if (message.tool_call_id === 'scope_0') parts = parts.flatMap(part => part.type === 'text' ? listingParts(part.text) : [part]);
      else if (childCalls.has(String(message.tool_call_id))) count('child-result', 'needs-graph-record');
      else count('tool-output', 'single-use');
      return parts.some(part => part.type !== 'text') ? { ...message, content: parts } : message;
    }
    if (message.role === 'assistant' && message.tool_calls?.length) {
      let changed = false;
      const calls = message.tool_calls.map(call => {
        const args = parseArguments(call.function.arguments);
        if (call.function.name === 'eval' && typeof args?.code === 'string') {
          const literals = args.code.match(NL_LITERAL)?.length ?? 0;
          if (literals) count('nl-literal', 'later-curriculum-step', literals);
        }
        if (call.function.name !== 'compact_history' || typeof args?.note !== 'string') return call;
        changed = true;
        count('handover-write');
        const note = args.note;
        return { ...call, function: { ...call.function, arguments: JSON.stringify({ ...args,
          note: { $write: { name: handoverName(note), type: HANDOVER_TYPE, source: note } } }) } };
      });
      return changed ? { ...message, tool_calls: calls } : message;
    }
    return message;
  };

  const messages = record.messages.map(convertMessage);
  const target = record.target ? convertMessage(record.target, record.messages.length) : undefined;
  return { record: { ...record, messages, ...(target ? { target } : {}),
    neuralese_conversion: { version: NEURALESE_CONVERSION_VERSION, sites } }, pieces: [...pieces.values()] };
}

function parseArguments(text: string): Record<string, unknown> | undefined {
  try { const value = JSON.parse(text); return value && typeof value === 'object' ? value : undefined; } catch { return undefined; }
}
