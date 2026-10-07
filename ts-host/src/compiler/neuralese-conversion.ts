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
 * - **Child results** (one call produces a value, its caller reads it: the recurrence of calling a function, retrieving
 *   its value and splicing it into the caller's trajectory). A child `nl` call's `return_result` value that the
 *   caller's eval output prints, within one collected run (`childResults`, built over the corpus by the converter
 *   script), becomes `{ $write: { name: 'result:<producer-identity-sha12>', type, source } }` in the child's final call (the template
 *   readout's write site) and a `{ type: 'read', name, source }` part where the caller's eval output shows it. A
 *   trainer writes the block from the child's record and trains it by the caller's loss. Kept exact, counted:
 *   `crisp-value` (a boolean, number or short text: its exact form is the value), `producer-missing` (the child's
 *   final turn is not in the corpus), `value-not-printed` (the output does not show the returned value as returned).
 *
 * - **Argument reads** (field-level flow). A structured child result's long text field that the caller passes into
 *   another call (`notes.push(found.facts)`, then `answer(notes)`) is written field by field in the producer's
 *   `return_result` (`{ facts: { $write: ... } }`, the other fields exact) and read where the consumer's argument
 *   listing (`scope_0`) shows it. Producers are identified as for child results; ambiguous ones stay text.
 *
 * Kept exact, counted: tool outputs (`single-use`), `nl` literals in eval code (`later-curriculum-step`), turn-count
 * notices (`dynamic-text`).
 *
 * Initialisation: soft parameters start from their text encoded in one forward pass through the port (`encode`), not
 * from a summarising call. Their texts are collected once in `pieces`.
 */
import { createHash } from 'node:crypto';
import type { InlineInstructionIndex } from './inline-instruction-index.js';
import { promptPieces, findPieces, type PromptPiece } from '../native/system-prompts.js';
import { AUTOMATIC_NOTE, DIGEST_PROMPT, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN } from '../native/prompt.js';

export const NEURALESE_CONVERSION_VERSION = 'natlang.neuralese-conversion/8';
export const HANDOVER_TYPE = 'Neuralese<HandoverNote>';

export type ConvertedPart = { type: 'text'; text: string } | { type: 'soft'; name: string } | { type: 'read'; name: string; source: string } |
  { type: 'digest'; name: string; holder: string; value_type: string; source: string; preview: string };
type Message = Record<string, unknown> & { role: string; content?: unknown; tool_calls?: { id?: string; function: { name: string; arguments: string } }[] };
export type SoftPiece = { name: string; kind: 'system-prompt' | 'program-guidance' | 'function-body'; text: string };
export type SiteCounts = Record<string, { converted: number; exact: Record<string, number> }>;
export type ConversionOptions = {
  inlineInstructions?: InlineInstructionIndex;
  pieces?: readonly PromptPiece[];
  /** Distinct calls per instructions digest over the corpus (`instructionsDigest`); without it every instructions
   * site counts as single-use. */
  instructionCalls?: ReadonlyMap<string, number>;
  instructionsReuse?: number;  // default 2
  instructionsShare?: number;  // default 0.1
  /** Per collected run (`callOf`): child calls' returned values (`childValueText`), and those a caller's eval output
   * prints. The corpus pass indexes unique producer records/invocations; repeated equal values
   * remain exact unless their producer can be identified. Without an index results stay exact. */
  childResults?: ReadonlyMap<string, { returned: readonly string[]; read: ReadonlySet<string>;
    producers?: readonly { id: string; invocation: string; value: string; field?: string; parent?: string; renderings?: readonly string[] }[];
    readers?: readonly { invocation: string; tool_call_id?: string; value: string; producer_id: string }[] }>;
};

type ChildResultProducer = { id: string; invocation: string; value: string; field?: string; parent?: string;
  renderings: string[]; order?: number };
type ChildResultOutput = { invocation: string; tool_call_id: string; text: string; argument: boolean; order?: number };
type ChildResultRun = { returned: Set<string>; producers: ChildResultProducer[];
  outputs: Map<string, ChildResultOutput> };

/** One corpus-pass index for child results. It records only explicit successful return_result targets;
 * host captures and finish:true actions are observations, never synthesized model targets. */
export class ChildResultIndexBuilder {
  private readonly runs = new Map<string, ChildResultRun>();

  add(record: Record<string, unknown> & { messages?: Message[] }, render: (value: unknown) => string = renderValueForChild): void {
    if (!Array.isArray(record.messages)) return;
    const runId = callOf(record);
    let run = this.runs.get(runId);
    if (!run) this.runs.set(runId, run = { returned: new Set(), producers: [], outputs: new Map() });
    const decision = record.decision as { index?: unknown } | undefined;
    const decisionId = /:decision:(\d+)$/.exec(String(record.id ?? ''));
    const idOrder = decisionId ? Number(decisionId[1]) : undefined;
    const order = typeof decision?.index === 'number' && Number.isSafeInteger(decision.index) && decision.index >= 0 ? decision.index :
      Number.isSafeInteger(idOrder) ? idOrder : undefined;
    const returned = childReturn(record as { messages: Message[]; target?: Message } & Record<string, unknown>);
    if (returned !== undefined) {
      const target = record.target as Message | undefined;
      const call = target?.tool_calls?.find(item => item.function.name === 'return_result');
      const args = call ? parseArguments(call.function.arguments) : undefined;
      const value = args?.value;
      run.returned.add(returned);
      const invocation = invocationOf(record), parent = (record.source_ref as { parent_invocation_id?: unknown } | undefined)?.parent_invocation_id;
      const id = String(record.id);
      run.producers.push({ id, invocation, value: returned, ...(typeof parent === 'string' ? { parent } : {}),
        renderings: value === undefined ? [returned] : [...new Set([render(value), JSON.stringify(value)])], order });
      if (value && typeof value === 'object' && !Array.isArray(value)) for (const [field, text] of Object.entries(value))
        if (typeof text === 'string' && text.length >= MIN_CHILD_RESULT_CHARS && text !== returned) {
          run.returned.add(text);
          run.producers.push({ id: `${id}#${field}`, invocation, field, value: text,
            ...(typeof parent === 'string' ? { parent } : {}), renderings: [JSON.stringify(text).slice(1, -1)], order });
        }
    }
    const children = childCallIds(record.messages, childFunctionNames(record));
    const invocation = invocationOf(record);
    for (const message of record.messages) if (message.role === 'tool' &&
        (children.has(String(message.tool_call_id)) || message.tool_call_id === 'scope_0') &&
        typeof message.content === 'string' && typeof message.tool_call_id === 'string') {
      const key = `${invocation}:${message.tool_call_id}`;
      const existing = run.outputs.get(key);
      if (!existing || (order !== undefined && (existing.order === undefined || order < existing.order))) run.outputs.set(key, { invocation,
        tool_call_id: message.tool_call_id, text: message.content, argument: message.tool_call_id === 'scope_0', order });
    }
  }

  finish(): ReadonlyMap<string, { returned: readonly string[]; read: ReadonlySet<string>;
    producers: readonly Omit<ChildResultProducer, 'order'>[];
    readers: readonly { invocation: string; tool_call_id: string; value: string; producer_id: string }[] }> {
    const result = new Map();
    for (const [runId, run] of this.runs) {
      const readers: { invocation: string; tool_call_id: string; value: string; producer_id: string }[] = [];
      for (const output of run.outputs.values()) {
        const matches: { producer: ChildResultProducer; form: string }[] = [];
        for (const producer of run.producers) {
          // Ingestion order is not evidence. Without authoritative per-decision order, keep exact text.
          if (producer.order === undefined || output.order === undefined || producer.order >= output.order) continue;
          if (producer.invocation === output.invocation) continue;
          const candidates = run.producers.filter(p => p.value === producer.value);
          if (candidates.some(p => p.parent !== undefined) && producer.parent !== output.invocation) continue;
          for (const form of new Set([producer.value, ...producer.renderings])) {
            const shown = output.argument ? output.text.includes(`"${form}"`) : output.text.includes(form);
            if (producer.value.length >= MIN_CHILD_RESULT_CHARS && shown) matches.push({ producer, form });
          }
        }
        // If a whole structured value is shown, its own field text is part of that same observation.
        // Keep the whole-object edge so its writer and every reader use one coherent block.
        const wholeProducerIds = new Set(matches.filter(({ producer }) => !producer.field).map(({ producer }) => producer.id));
        const candidates = matches.filter(({ producer }) => !producer.field ||
          ![...wholeProducerIds].some(id => producer.id.startsWith(`${id}#`)));
        const selected = new Map<string, ChildResultProducer>();
        for (const { producer } of candidates) selected.set(producer.value, producer);
        for (const [value, producer] of selected) {
          const peers = run.producers.filter(p => p.value === value && p.order !== undefined && output.order !== undefined && p.order < output.order &&
            (run.producers.some(candidate => candidate.value === value && candidate.parent !== undefined) ?
              p.parent === output.invocation : true));
          if (peers.length === 1 && !readers.some(reader => reader.invocation === output.invocation &&
              reader.tool_call_id === output.tool_call_id && reader.value === value))
            readers.push({ invocation: output.invocation, tool_call_id: output.tool_call_id, value, producer_id: producer.id });
        }
      }
      // A producer cannot write both its complete object and overlapping field blocks. Whole-object reads win.
      const wholeReads = new Set(readers.filter(reader => !reader.producer_id.includes('#')).map(reader => reader.producer_id));
      const coherentReaders = readers.filter(reader => ![...wholeReads].some(id => reader.producer_id.startsWith(`${id}#`)));
      result.set(runId, { returned: [...run.returned], read: new Set(coherentReaders.map(reader => reader.value)),
        producers: run.producers.map(({ order: _order, ...producer }) => producer), readers: coherentReaders });
    }
    return result;
  }
}

function renderValueForChild(value: unknown): string {
  // Keep rendering local to this module's shared child index without a dependency on CLI code.
  if (typeof value === 'string') return value;
  try { return JSON.stringify(value); } catch { return String(value); }
}

/** Shortest returned text that becomes a written value: shorter values are their exact form. */
export const MIN_CHILD_RESULT_CHARS = 16;
/** A `return_result` value as text, if it is one a child result can carry (text or structured; not a primitive). */
export function childValueText(value: unknown): string | undefined {
  if (typeof value === 'string') return value;
  return value !== null && typeof value === 'object' ? JSON.stringify(value) : undefined;
}
/** The calls of eval code that run child `nl` calls, by tool call ID. */
export function childCallIds(messages: readonly Message[], functionNames: readonly string[] = []): Set<string> {
  const ids = new Set<string>();
  for (const message of messages) for (const call of message.tool_calls ?? []) {
    const args = call.function.name === 'eval' ? parseArguments(call.function.arguments) : undefined;
    if (typeof args?.code === 'string' && (NL_LITERAL.test(args.code) || functionNames.some(name =>
      new RegExp(`\\b${name.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\s*\\(`).test(args.code as string)))) ids.add(String(call.id));
    NL_LITERAL.lastIndex = 0;
  }
  return ids;
}
/** Named natural-language functions carried by the pinned program, including scoped file functions. */
export function childFunctionNames(record: Record<string, unknown>): string[] {
  const files = (record.task as { program_ir?: { semantics?: { files?: Record<string, unknown> } } } | undefined)
    ?.program_ir?.semantics?.files ?? {};
  return [...new Set(Object.keys(files).filter(path => path.endsWith('.nl'))
    .map(path => path.split('/').pop()!.slice(0, -3)).filter(name => /^[A-Za-z_$][\w$]*$/.test(name)))];
}
/** A child call's successful `return_result` value text in a record's target (a child call: not the run's root). */
export function childReturn(record: { messages: readonly Message[]; target?: Message } & Record<string, unknown>): string | undefined {
  const call = record.target?.tool_calls?.find(c => c.function.name === 'return_result');
  const args = call ? parseArguments(call.function.arguments) : undefined;
  if (!args || args.status !== 'success' || !('value' in args)) return undefined;
  const opening = record.messages.find(message => message.role === 'user')?.content;
  // Any call's name, anonymous `nl` literals' (`nl@eval:6`) included.
  const callName = typeof opening === 'string' ? /^You are inside this call: ([^\s(]+)\(/.exec(opening)?.[1] : undefined;
  const root = ((record.task as { program_ir?: { semantics?: { root?: string } } } | undefined)?.program_ir?.semantics)?.root;
  if (!callName || root?.split('/').pop() === `${callName}.nl`) return undefined;
  return childValueText(args.value);
}
/** The returned values (long enough to write) that `text`, a caller's eval output, prints. */
export function printedResults(text: string, returned: readonly string[]): string[] {
  return returned.filter(value => value.length >= MIN_CHILD_RESULT_CHARS && text.includes(value));
}

/** The digest under which instructions are counted for reuse. */
export const instructionsDigest = (text: string) => sha12(text);
/** The instructions of a record's opening, if it has them. */
export function openingInstructions(record: { messages: readonly Message[] }): string | undefined {
  const opening = record.messages.find(message => message.role === 'user');
  return typeof opening?.content === 'string' ? INSTRUCTIONS.exec(opening.content)?.[2] : undefined;
}
/** The invocation a record's turn belongs to: the recorded invocation ID, else the call's opening (its messages
 * through the argument listing), which every turn of one call shares and sibling calls of one lambda do not. */
export function invocationOf(record: Record<string, unknown>): string {
  const recorded = (record.source_ref as { invocation_id?: unknown } | undefined)?.invocation_id;
  if (recorded !== undefined && recorded !== null) return String(recorded);
  const messages = Array.isArray(record.messages) ? record.messages as Message[] : [];
  const scope = messages.findIndex(message => message.role === 'tool' && message.tool_call_id === 'scope_0');
  return `call:${sha12(JSON.stringify([callOf(record), messages.slice(0, scope >= 0 ? scope + 1 : 2)]))}`;
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
  const invocation = invocationOf(record as Record<string, unknown>);
  const inlineWriters = options.inlineInstructions?.writers.filter(writer => writer.decision_id === (record as Record<string, unknown>).id) ?? [];
  const inlineRead = options.inlineInstructions?.reads.find(read => read.trajectory_id === callOf(record as Record<string, unknown>) && read.invocation_id === invocation);
  const inlineWriter = inlineRead && options.inlineInstructions?.writers.find(writer => writer.writer_id === inlineRead.writer_id);
  const inlineBody = inlineWriter?.template_source.slice(1, -1);
  const plainInline = inlineWriter && inlineBody === inlineWriter.template_segments[0];
  for (const hold of options.inlineInstructions?.held ?? []) if (hold.decision_id === (record as Record<string, unknown>).id)
    count('inline-instruction', hold.reason);

  // Eval calls that run child natural-language calls: their printed results are another call's output.
  const childCalls = childCallIds(record.messages, childFunctionNames(record as Record<string, unknown>));
  const run = options.childResults?.get(callOf(record as Record<string, unknown>));
  const resultName = (value: string, producerId?: string) => {
    const producer = run?.producers?.find(p => producerId ? p.id === producerId : p.value === value);
    return `result:${sha12(JSON.stringify([callOf(record as Record<string, unknown>), producer?.invocation, producer?.id, value]))}`;
  };
  /** A caller's eval output with each printed child result as a read of the child's written value; with `kind`
   * 'argument-read', a call's argument listing with each value another call produced (counted only when found). */
  const childResultParts = (parts: ConvertedPart[], kind: 'child-result' | 'argument-read' = 'child-result',
      toolCallId?: string): ConvertedPart[] => {
    const text = parts.map(part => part.type === 'text' ? part.text : '').join('');
    const links = run?.readers?.filter(r => r.invocation === invocation &&
      (r.tool_call_id === undefined || r.tool_call_id === toolCallId)) ?? [];
    const eligible = run?.readers ? links.map(r => r.value) : [...run?.read ?? []];
    // An argument listing shows a passed value whole, as a quoted string: a match inside longer text (an answer that is
    // also a phrase of some article) is not that value.
    const whole = (form: string) => kind === 'child-result' ? text.includes(form) : text.includes(`"${form}"`);
    const forms = eligible.flatMap(value => [...new Set([value, ...(run?.producers?.find(p => p.value === value)?.renderings ?? [])])]
      .filter(whole).map(form => ({form,value})));
    const shown = forms.sort((a,b) => b.form.length-a.form.length);
    if (kind === 'argument-read' && (!shown.length || parts.some(part => part.type !== 'text'))) return parts;
    if (!shown.length || parts.some(part => part.type !== 'text')) {
      count('child-result', run?.producers?.some(p => p.value.length >= MIN_CHILD_RESULT_CHARS &&
        text.includes(p.value) && run.producers!.filter(other => other.value === p.value).length > 1) ? 'ambiguous-producer' :
        !run?.returned.length ? 'producer-missing' :
        run.returned.every(value => value.length < MIN_CHILD_RESULT_CHARS) ? 'crisp-value' : 'value-not-printed');
      return parts;
    }
    const out: ConvertedPart[] = [];
    let at = 0;
    while (at < text.length) {
      // The earliest (then longest) printed value from here on.
      let best: [number, {form: string; value: string}] | undefined;
      for (const match of shown) {
        const quoted = kind === 'argument-read' ? text.indexOf(`"${match.form}"`, at) : -1;
        const index = kind === 'argument-read' ? (quoted < 0 ? -1 : quoted + 1) : text.indexOf(match.form, at);
        if (index >= 0 && (!best || index < best[0])) best = [index, match];
      }
      if (!best) break;
      if (best[0] > at) out.push({ type: 'text', text: text.slice(at, best[0]) });
      out.push({ type: 'read', name: resultName(best[1].value, links.find(r => r.value === best![1].value)?.producer_id), source: best[1].value });
      count(kind);
      at = best[0] + best[1].form.length;
    }
    if (at < text.length) out.push({ type: 'text', text: text.slice(at) });
    return out;
  };
  // The root call's full inputs, when this record is the root call: the sources of its listing digests.
  const semantics = ((record as Record<string, unknown>).task as { program_ir?: { semantics?: { root?: string; inputs?: Record<string, unknown> } } } | undefined)
    ?.program_ir?.semantics;
  const opening = record.messages.find(message => message.role === 'user')?.content;
  const callName = typeof opening === 'string' ? OPENING_CALL.exec(opening)?.[1] : undefined;
  const inputs = semantics?.inputs && callName && semantics.root?.split('/').pop() === `${callName}.nl` ? semantics.inputs : undefined;
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
        if (inlineRead && plainInline && inlineBody !== undefined) {
          const at = instructions.index + instructions[1]!.length;
          if (text.slice(at, at + inlineBody.length) === inlineBody &&
              (instructions[2] === inlineBody || instructions[2] === inlineRead.realized_instruction)) {
            count('inline-instruction-read');
            return { ...message, content: [{ type: 'text', text: text.slice(0, at) },
              { type: 'read', name: inlineRead.writer_id, source: inlineBody },
              { type: 'text', text: text.slice(at + inlineBody.length) }] };
          }
          count('inline-instruction', 'opening-source-mismatch');
          return message; // never claim a detached shared instruction for a failed causal link
        }
        if (inlineRead) { count('inline-instruction', 'escaped-template-body'); return message; }
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
      if (message.tool_call_id === 'scope_0') parts = childResultParts(parts, 'argument-read', String(message.tool_call_id))
        .flatMap(part => part.type === 'text' ? listingParts(part.text) : [part]);
      else if (childCalls.has(String(message.tool_call_id))) parts = childResultParts(parts, 'child-result', String(message.tool_call_id));
      else count('tool-output', 'single-use');
      return parts.some(part => part.type !== 'text') ? { ...message, content: parts } : message;
    }
    if (message.role === 'assistant' && message.tool_calls?.length) {
      let changed = false;
      const calls = message.tool_calls.map(call => {
        const args = parseArguments(call.function.arguments);
        if (call.function.name === 'eval' && typeof args?.code === 'string') {
          const literals = args.code.match(NL_LITERAL)?.length ?? 0;
          const linked = index === record.messages.length ? inlineWriters.filter(writer => writer.target_tool_call_id === call.id) : [];
          const eligible = linked.filter(writer => writer.template_source.slice(1,-1) === writer.template_segments[0]);
          if (linked.length !== eligible.length) count('inline-instruction', 'escaped-template-body', linked.length - eligible.length);
          if (eligible.length) {
            const sorted = eligible.slice().sort((a,b) => a.code_span.start - b.code_span.start);
            const valid = sorted.every((writer, i) => writer.code === args.code &&
              (!i || sorted[i-1]!.code_span.end <= writer.code_span.start));
            if (valid) {
              const parts: unknown[] = []; let cursor = 0;
              for (const writer of sorted) {
                const start = writer.code_span.start + 1, end = writer.code_span.end - 1;
                parts.push({type:'text',text:args.code.slice(cursor,start)});
                parts.push({$write:{name:writer.writer_id,type:'Neuralese<string>',source:args.code.slice(start,end)}});
                cursor=end; count('inline-instruction-write'); count('nl-literal');
              }
              parts.push({type:'text',text:args.code.slice(cursor)});
              if (literals > eligible.length) count('nl-literal','later-curriculum-step',literals-eligible.length);
              changed=true;
              return {...call,neuralese_code:{schema:'natlang.inline-instruction-code/1',code_sha256:createHash('sha256').update(args.code).digest('hex'),parts,
                sites:sorted.map(writer=>({name:writer.writer_id,actual_tool_call_id:writer.tool_call_id,definition_id:writer.definition_id,code_span:writer.code_span,plan:writer.plan}))}};
            }
            count('inline-instruction','overlapping-or-mismatched-source',eligible.length);
          }
          if (literals) count('nl-literal', 'later-curriculum-step', literals);
        }
        if (call.function.name === 'return_result' && args?.status === 'success' && 'value' in args) {
          // A child call's value that its caller reads: written at the template readout's site.
          const value = childValueText(args.value);
          // Field by field: a structured result's text fields that another call reads are written on their own.
          if (value !== undefined && !run?.read.has(value) && args.value && typeof args.value === 'object' && !Array.isArray(args.value)) {
            const fields = Object.entries(args.value as Record<string, unknown>).map(([field, text]) => {
              const producer = typeof text === 'string' ? run?.producers?.find(p => p.field === field && p.value === text && p.invocation === invocation) : undefined;
              if (!producer || !run?.readers?.some(r => r.producer_id === producer.id)) return [field, text];
              count('child-result-write');
              return [field, { $write: { name: resultName(text as string, producer.id), type: 'Neuralese<string>', source: text } }];
            });
            if (fields.some(([, text]) => text && typeof text === 'object' && '$write' in (text as object))) {
              changed = true;
              return { ...call, function: { ...call.function, arguments: JSON.stringify({ ...args, value: Object.fromEntries(fields) }) } };
            }
          }
          if (value === undefined || !run?.read.has(value)) return call;
          // Only the child's own return is a producer. A root returning the same value or
          // another call's historical return must never claim that child's block.
          const ownProducer = run?.producers?.find(p => p.value === value && p.invocation === invocation);
          if (run.producers && (!ownProducer || (run.readers && !run.readers.some(r => r.producer_id === ownProducer.id)))) return call;
          changed = true;
          count('child-result-write');
          return { ...call, function: { ...call.function, arguments: JSON.stringify({ ...args,
            value: { $write: { name: resultName(value, ownProducer?.id), type: typeof args.value === 'string' ? 'Neuralese<string>' : 'Neuralese<unknown>', source: value } } }) } };
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
