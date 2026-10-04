/**
 * Conversion of natlang trajectory records to Neuralese form (plans/neuralese/DECISIONS.md 41, S5 §2.2).
 *
 * Everything that can be Neuralese becomes Neuralese; every other site is counted with the reason it stays exact.
 *
 * Always converted:
 *
 * - **Prompt sites.** The system message and runtime-written frames are split into registered prompt pieces
 *   (native/system-prompts.ts). Each becomes a soft parameter part `{ type: 'soft', name: 'prompt:<id>' }`. Text that
 *   matches no registered piece (an older runtime's wording) becomes a versioned piece `prompt:system@<sha12>`, so its
 *   wording is the initialisation; whitespace between pieces stays text. Program guidance is a soft parameter of its
 *   own (`guidance@<sha12>`): it belongs to the program and is adapted by self-improvement.
 * - **Handover sites.** A `compact_history` note is a model write: in the call's arguments the note becomes
 *   `{ $write: { name, type: 'Neuralese<HandoverNote>', source } }`, and the pinned note message reads the same block
 *   (`{ type: 'read', name }` between the soft handover frames). The name is the note's digest, so the producing and
 *   consuming records of one trajectory agree. `source` is the crisp note: the write's teacher view.
 *
 * Converted when asked (curriculum steps of S5 §6): call instructions as soft function bodies (`instructions`), and
 * tool outputs whose only consumer is the model (`tool-outputs`): an output becomes `{ type: 'encode', name, source }`,
 * a block the model's writer produces from the source (the teacher sees the source). An output is model-only when no
 * later turn of the record copies an exact value out of it (an identifier, number, path or quoted string): a text-level
 * stand-in for the consumer trace, which these records do not carry. Outputs whose values are copied stay exact.
 *
 * Counted, kept exact: tool outputs with copied exact values, `nl` literals in eval code (model-written soft code is a
 * later step), dynamic notices (turn counts).
 *
 * Soft parameter texts are not repeated in every record: `pieces` collects each name's initial text once.
 */
import { createHash } from 'node:crypto';
import { promptPieces, findPieces, type PromptPiece } from '../native/system-prompts.js';
import { AUTOMATIC_NOTE, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN } from '../native/prompt.js';

export const NEURALESE_CONVERSION_VERSION = 'natlang.neuralese-conversion/1';
export const HANDOVER_TYPE = 'Neuralese<HandoverNote>';

export type ConvertedPart = { type: 'text'; text: string } | { type: 'soft'; name: string } | { type: 'read'; name: string } |
  { type: 'encode'; name: string; source: string };
type Message = Record<string, unknown> & { role: string; content?: unknown; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] };
export type SoftPiece = { name: string; kind: 'system-prompt' | 'program-guidance' | 'function-body'; text: string };
export type SiteCounts = Record<string, { converted: number; exact: Record<string, number> }>;
export type ConversionOptions = { convert?: readonly ('instructions' | 'tool-outputs')[]; pieces?: readonly PromptPiece[] };

const sha12 = (text: string) => createHash('sha256').update(text).digest('hex').slice(0, 12);
const GUIDANCE = /\n\n<natlang_program_guidance>\n([\s\S]*?)\n<\/natlang_program_guidance>\n/;
const INSTRUCTIONS = /(Instructions:\n)([\s\S]*?)(\n\n(?:In eval|Eval also|$))/;
const NL_LITERAL = /\bnl(?:\.with\([^)]*\))?(?:<[^`]*?>)?`/g;
const DYNAMIC_NOTICE = /\n\n\[\d+ turns left in this call\.[^\]]*\]$/;
const EXACT_VALUES = [
  /"[^"\n]{4,80}"/g,                                  // quoted strings
  /(?<![\w.])-?\d+(?:\.\d+)?(?![\w.])/g,                // numbers
  /\b[A-Za-z][\w-]*[\d_-][\w-]*\b/g,                  // identifiers with digits, underscores or dashes
  /(?:\.{0,2}\/)?[\w.-]+(?:\/[\w.-]+)+/g,              // paths
];

/** The exact values of a text worth tracking: long enough that a later occurrence is a copy, not a coincidence. */
export function exactValues(text: string): Set<string> {
  const values = new Set<string>();
  for (const pattern of EXACT_VALUES) for (const match of text.matchAll(pattern)) {
    const value = match[0];
    if (value.replace(/^"|"$/g, '').length >= 4 || /\d{3,}|\d\.\d/.test(value)) values.add(value.replace(/^"|"$/g, ''));
  }
  return values;
}

/** Whether a later text copies one of a tool output's exact values. */
const copies = (values: Set<string>, later: string) => [...values].some(value => later.includes(value));
/** The text of a message the model wrote: its content, reasoning and tool-call arguments. */
const writtenText = (message: Message | undefined) => !message || message.role !== 'assistant' ? '' :
  [typeof message.content === 'string' ? message.content : '', String(message.reasoning_content ?? ''),
    ...(message.tool_calls ?? []).map(call => call.function.arguments)].join('\n');

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
          { type: 'read', name: handoverName(note) }, soft('prompt:handover/close', 'system-prompt', HANDOVER_NOTE_CLOSE)] };
      }
      if (text === AUTOMATIC_NOTE) return { ...message, content: promptParts(text, 'text') };
      const instructions = INSTRUCTIONS.exec(text);
      if (index <= 1 && instructions) {
        if (options.convert?.includes('instructions')) {
          count('instructions');
          const at = instructions.index + instructions[1]!.length;
          return { ...message, content: [{ type: 'text', text: text.slice(0, at) },
            soft(`instructions@${sha12(instructions[2]!)}`, 'function-body', instructions[2]!),
            { type: 'text', text: text.slice(at + instructions[2]!.length) }] };
        }
        count('instructions', 'later-curriculum-step');
      }
      return message;
    }
    if (message.role === 'tool' && typeof message.content === 'string') {
      if (DYNAMIC_NOTICE.test(message.content)) count('notice', 'dynamic-text');
      const parts = promptParts(message.content, 'text');
      // The output itself: the text parts around runtime pieces (notices).
      const output = parts.filter(part => part.type === 'text').map(part => (part as { text: string }).text).join('');
      const later = [...record.messages.slice(index + 1), ...(record.target ? [record.target] : [])].map(writtenText).join('\n');
      if (copies(exactValues(output), later)) count('tool-output', 'copied-exact-values');
      else if (!options.convert?.includes('tool-outputs') || !output.trim()) count('tool-output', 'later-curriculum-step');
      else {
        count('tool-output');
        const encoded = parts.map(part => part.type === 'text' && part.text.trim() ?
          { type: 'encode' as const, name: `value:${sha12(part.text)}`, source: part.text } : part);
        return { ...message, content: encoded };
      }
      return parts.some(part => part.type === 'soft') ? { ...message, content: parts } : message;
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
