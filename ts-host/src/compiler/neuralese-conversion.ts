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
 *   prompt pieces (native/system-prompts.ts); each becomes a content-versioned soft parameter part
 *   `{ type: 'soft', name: 'prompt:<id>@<sha12>' }`, so the same stable runtime ID can safely carry old/new wording.
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

export const NEURALESE_CONVERSION_VERSION = 'natlang.neuralese-conversion/13';
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
    readers?: readonly { invocation: string; tool_call_id?: string; value: string; producer_id: string }[];
    observed_host_contexts?: readonly { invocation: string; tool_call_id: string; value: string; producer_id: string;
      body_sha256: string; capture_sha256: string; result_type: string }[] }>;
  /** Exact runtime graph edges independently validated against the collected result. These permit typed
   * Neuralese argument reads that are not printed by an eval caller. */
  softStateEdges?: ValidatedSoftStateEdgeSet;
};

export type ValidatedSoftStateEdge = {
  block_id: string;
  writer_call_id: string;
  writer_node: string;
  writer_record_id: string;
  writer_decision_index: number;
  reader_call_id: string;
  reader_node: string;
  reader_record_id: string;
  reader_decision_index: number;
  consumer_argument: string;
  consumer_signature: string;
  expected_type: 'Neuralese<string>';
  body_sha256: string;
  body_source: string;
};
export type ValidatedSoftStateEdgeSet = {
  schema: 'natlang.validated-runtime-soft-state-edges/1';
  status: 'passed';
  validation: { validator: 'validateSoftStateEdge'; review_sha256: string; result_sha256: string };
  source: { trajectory_id: string; source_row_sha256: string; split: string; source_groups: string[];
    transport_mode: 'text-marker-standin/2'; learned_vectors: false; qualification_certificate: false; training_admission: false };
  edges: ValidatedSoftStateEdge[];
};

function exactPathGet(value: unknown, path: readonly (string | number)[]): unknown {
  let current = value;
  for (const key of path) {
    if (typeof key === 'number') {
      if (!Array.isArray(current) || !Number.isSafeInteger(key) || key < 0 || key >= current.length) return undefined;
      current = current[key];
    } else {
      if (!current || typeof current !== 'object' || Array.isArray(current) || !Object.hasOwn(current, key)) return undefined;
      current = (current as Record<string, unknown>)[key];
    }
  }
  return current;
}

function exactPathSet<T>(value: T, path: readonly (string | number)[], replacement: unknown): T {
  if (!path.length) return replacement as T;
  const [head, ...tail] = path;
  if (head === undefined) throw new Error('typed result path is absent');
  if (typeof head === 'number') {
    if (!Array.isArray(value) || head < 0 || head >= value.length) throw new Error('typed result path is absent');
    const copy = value.slice();
    copy[head] = exactPathSet(copy[head], tail, replacement);
    return copy as T;
  }
  if (!value || typeof value !== 'object' || Array.isArray(value) || !Object.hasOwn(value, head))
    throw new Error('typed result path is absent');
  return { ...(value as Record<string, unknown>), [head]: exactPathSet((value as Record<string, unknown>)[head], tail, replacement) } as T;
}

function stableJson(value: unknown): string | undefined {
  if (value === null || typeof value === 'boolean' || typeof value === 'string') return JSON.stringify(value);
  if (typeof value === 'number') return Number.isFinite(value) ? (Object.is(value, -0) ? '-0' : JSON.stringify(value)) : undefined;
  if (Array.isArray(value)) {
    const items = value.map(stableJson);
    return items.some(item => item === undefined) ? undefined : `[${items.join(',')}]`;
  }
  if (!value || typeof value !== 'object' || Object.getPrototypeOf(value) !== Object.prototype) return undefined;
  const fields = Object.keys(value as Record<string, unknown>).sort().map(key => {
    const child = stableJson((value as Record<string, unknown>)[key]);
    return child === undefined ? undefined : `${JSON.stringify(key)}:${child}`;
  });
  return fields.some(item => item === undefined) ? undefined : `{${fields.join(',')}}`;
}

type ChildResultProducer = { id: string; invocation: string; value: string; field?: string; parent?: string;
  renderings: string[]; order?: number };
type ChildResultOutput = { invocation: string; tool_call_id: string; text: string; argument: boolean; order?: number };
type ObservedHostResult = { id: string; invocation: string; parent?: string; value: string; body_sha256: string;
  capture_sha256: string; result_type: string; order?: number };
type ChildResultRun = { returned: Set<string>; producers: ChildResultProducer[];
  outputs: Map<string, ChildResultOutput>; observed_host_results: ObservedHostResult[] };

function sha256Text(value: string): string { return createHash('sha256').update(value).digest('hex'); }

/** Authenticate a completed host return against the same decision's successful typed result.
 * This is crisp context evidence, never a model-written Neuralese producer. */
function observedHostReturn(record: Record<string, unknown>, returned: string, value: unknown):
  Omit<ObservedHostResult, 'order'> | undefined {
  const source = record.source_ref as Record<string, unknown> | undefined;
  const receipt = source?.host_result_capture as Record<string, unknown> | undefined;
  const capture = receipt?.capture as Record<string, unknown> | undefined;
  if (!capture || capture.version !== 'reduction-trace/1' || capture.kind !== 'host_capture' ||
      capture.capture_kind !== 'invocation_output' || capture.complete !== true ||
      capture.origin !== 'observed-host-result; not a model-generated writer target' ||
      typeof source?.invocation_id !== 'string' || capture.call_id !== source.invocation_id ||
      typeof capture.result_type !== 'string' || !capture.result_type ||
      typeof capture.value_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(capture.value_sha256) ||
      stableJson(capture.value) !== stableJson(value)) return undefined;
  const encoded = JSON.stringify(capture.value);
  if (encoded === undefined || sha256Text(encoded) !== capture.value_sha256 || childValueText(capture.value) !== returned)
    return undefined;
  const parent = typeof source.parent_invocation_id === 'string' ? source.parent_invocation_id : undefined;
  const id = typeof record.id === 'string' ? record.id : '';
  if (!id) return undefined;
  return { id, invocation: String(source.invocation_id), ...(parent ? { parent } : {}), value: returned,
    body_sha256: sha256Text(returned), capture_sha256: sha256Text(JSON.stringify(capture)), result_type: capture.result_type };
}

/** One corpus-pass index for child results. It records only explicit successful return_result targets;
 * host captures and finish:true actions are observations, never synthesized model targets. */
export class ChildResultIndexBuilder {
  private readonly runs = new Map<string, ChildResultRun>();

  add(record: Record<string, unknown> & { messages?: Message[] }, render: (value: unknown) => string = renderValueForChild): void {
    if (!Array.isArray(record.messages)) return;
    const runId = callOf(record);
    let run = this.runs.get(runId);
    if (!run) this.runs.set(runId, run = { returned: new Set(), producers: [], outputs: new Map(), observed_host_results: [] });
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
      const hostResult = observedHostReturn(record, returned, value);
      if (hostResult) run.observed_host_results.push({ ...hostResult, ...(order !== undefined ? { order } : {}) });
      if (!hostResult) {
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
    }
    const children = childCallIds(record.messages, childFunctionNames(record));
    const evalCalls = new Set(record.messages.flatMap(message => message.role === 'assistant' ?
      (message.tool_calls ?? []).filter(call => call.function?.name === 'eval' && typeof call.id === 'string').map(call => call.id!) : []));
    const invocation = invocationOf(record);
    for (const message of record.messages) if (message.role === 'tool' &&
        (children.has(String(message.tool_call_id)) || message.tool_call_id === 'scope_0' || evalCalls.has(String(message.tool_call_id))) &&
        typeof message.content === 'string' && typeof message.tool_call_id === 'string') {
      const key = `${invocation}:${message.tool_call_id}`;
      const existing = run.outputs.get(key);
      if (!existing || (order !== undefined && (existing.order === undefined || order < existing.order))) run.outputs.set(key, { invocation,
        tool_call_id: message.tool_call_id, text: message.content, argument: message.tool_call_id === 'scope_0', order });
    }
  }

  finish(): ReadonlyMap<string, { returned: readonly string[]; read: ReadonlySet<string>;
    producers: readonly Omit<ChildResultProducer, 'order'>[];
    readers: readonly { invocation: string; tool_call_id: string; value: string; producer_id: string }[];
    observed_host_contexts: readonly { invocation: string; tool_call_id: string; value: string; producer_id: string;
      body_sha256: string; capture_sha256: string; result_type: string }[] }> {
    const result = new Map();
    for (const [runId, run] of this.runs) {
      const readers: { invocation: string; tool_call_id: string; value: string; producer_id: string }[] = [];
      const observed_host_contexts: { invocation: string; tool_call_id: string; value: string; producer_id: string;
        body_sha256: string; capture_sha256: string; result_type: string }[] = [];
      for (const output of run.outputs.values()) {
        for (const host of run.observed_host_results) {
          if (host.parent !== output.invocation || host.order === undefined || output.order === undefined || host.order >= output.order ||
              !output.text.includes(host.value)) continue;
          if (!observed_host_contexts.some(item => item.invocation === output.invocation &&
              item.tool_call_id === output.tool_call_id && item.producer_id === host.id))
            observed_host_contexts.push({ invocation: output.invocation, tool_call_id: output.tool_call_id, value: host.value,
              producer_id: host.id, body_sha256: host.body_sha256, capture_sha256: host.capture_sha256, result_type: host.result_type });
        }
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
        producers: run.producers.map(({ order: _order, ...producer }) => producer), readers: coherentReaders, observed_host_contexts });
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
  const openingText = typeof opening === 'string' ? opening : Array.isArray(opening) ? opening.map(part =>
    typeof part?.text === 'string' ? part.text : '').join('') : undefined;
  // Any call's name, anonymous `nl` literals' (`nl@eval:6`) included.
  const callName = openingText === undefined ? undefined : /^You are inside this call: ([^\s(]+)\(/.exec(openingText)?.[1];
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
    { record: R & { neuralese_conversion: { version: string; sites: SiteCounts; soft_state_edges?: unknown[] } }; pieces: SoftPiece[] } {
  const registered = options.pieces ?? promptPieces();
  const pieces = new Map<string, SoftPiece>();
  const sites: SiteCounts = {};
  const observedHostMetadata: { schema: string; origin: string; invocation_id: string; tool_call_id: string;
    producer_record_id: string; result_type: string; body_sha256: string; capture_sha256: string;
    visible_as_crisp_context: true; model_writer_target: false; recurrence_edge: false }[] = [];
  const count = (kind: string, reason?: string, n = 1) => {
    const site = sites[kind] ??= { converted: 0, exact: {} };
    if (reason) site.exact[reason] = (site.exact[reason] ?? 0) + n; else site.converted += n;
  };
  const sourceRef = (record as Record<string, unknown>).source_ref as Record<string, unknown> | undefined;
  const decision = (record as Record<string, unknown>).decision as Record<string, unknown> | undefined;
  const decisionIndex = decision?.index;
  const rowId = String((record as Record<string, unknown>).id ?? '');
  const trajectoryId = typeof sourceRef?.trajectory_id === 'string' ? sourceRef.trajectory_id : '';
  const externalReadContexts = sourceRef?.provider_expanded_read_contexts;
  const externalBodies = new Map<string, { body: string; type: string; receipt: Record<string, unknown> }>();
  if (externalReadContexts !== undefined) {
    if (!Array.isArray(externalReadContexts)) throw new Error(`invalid provider-expanded context receipt for ${rowId}`);
    for (const candidate of externalReadContexts) {
      if (!candidate || typeof candidate !== 'object') throw new Error(`invalid provider-expanded context receipt for ${rowId}`);
      const receipt = candidate as Record<string, unknown>;
      const block = receipt.block as Record<string, unknown> | undefined;
      const invocation = invocationOf(record as Record<string, unknown>);
      if (!['natlang.provider-expanded-read-context/1', 'natlang.provider-expanded-read-context/2'].includes(String(receipt.schema)) ||
          receipt.invocation_id !== invocation ||
          receipt.source_row_sha256 !== sourceRef?.source_row_sha256 ||
          receipt.trace_sha256 !== ((record as Record<string, unknown>).provenance as Record<string, unknown> | undefined)?.trace_sha256 ||
          receipt.learned_vectors !== false || receipt.qualification_certificate !== false || receipt.training_admission !== false ||
          !block || typeof block.id !== 'string' || typeof block.type !== 'string' || !block.type.startsWith('Neuralese<') ||
          typeof block.body !== 'string' || !/^[0-9a-f]{64}$/.test(String(block.body_sha256)) ||
          createHash('sha256').update(block.body).digest('hex') !== block.body_sha256 ||
          receipt.block_read === null || typeof receipt.block_read !== 'object' ||
          (receipt.block_read as Record<string, unknown>).block !== block.id ||
          receipt.model_turn === null || typeof receipt.model_turn !== 'object' ||
          !Array.isArray((receipt.model_turn as Record<string, unknown>).inputs) ||
          !((receipt.model_turn as Record<string, unknown>).inputs as Record<string, unknown>[]).some(input =>
            input.node === (receipt.block_read as Record<string, unknown>).node && input.port === 'read' && input.block === block.id) ||
          !['configured-function-definition', 'same-run-producer'].includes(String(receipt.origin)))
        throw new Error(`provider-expanded context provenance mismatch for ${rowId}`);
      const id = block.id as string;
      if (externalBodies.has(id)) throw new Error(`duplicate provider-expanded context block ${id}`);
      const read = receipt.block_read as Record<string, unknown>;
      const turn = receipt.model_turn as Record<string, unknown>;
      if (read.call_id !== invocation || turn.call_id !== invocation || read.node === undefined ||
          !/^[0-9a-f]{64}$/.test(String(receipt.transport_provenance_sha256)) ||
          !/^[0-9a-f]{64}$/.test(String(receipt.raw_request_sha256)) ||
          !/^[0-9a-f]{64}$/.test(String(receipt.rendered_request_sha256)))
        throw new Error(`provider-expanded context graph or transport binding mismatch for ${id}`);
      if (receipt.schema === 'natlang.provider-expanded-read-context/2' &&
          (receipt.origin !== 'same-run-producer' || receipt.writer_target_selected !== false ||
           (receipt.parent_invocation_id !== null && typeof receipt.parent_invocation_id !== 'string')))
        throw new Error(`provider-expanded context-only writer receipt is incomplete for ${id}`);
      if (receipt.origin === 'configured-function-definition') {
        const readout = receipt.readout as Record<string, unknown> | undefined;
        const definition = receipt.definition as Record<string, unknown> | undefined;
        if (!readout || readout.schema !== 'natlang.text-template-readout/1' || readout.call !== 'return_result' ||
            readout.value !== 'decode' || readout.value_type !== 'string' || readout.read_body_id !== id ||
            readout.read_source_sha256 !== block.body_sha256 || readout.learned_vectors !== false ||
            readout.qualification_certificate !== false || readout.training_admission !== false ||
            !definition || definition.id !== `nz-fn:${id}`)
          throw new Error(`configured function context lacks exact readout and definition binding: ${id}`);
      } else {
        const writer = receipt.producer_write as Record<string, unknown> | null;
        const readInputs = Array.isArray(read.inputs) ? read.inputs as Record<string, unknown>[] : [];
        // Trace seq is invocation-local. The explicit writer node on this read's input is the cross-call causal link.
        const writerSourceValid = receipt.schema === 'natlang.provider-expanded-read-context/2' ?
          writer?.producer === 'text-marker-emulation' && writer.source_kind === 'typed-text-result' :
          writer?.learned_vectors === false;
        if (!writer || writer.kind !== 'block_write' || writer.block !== id ||
            writer.call_id === invocation || typeof writer.node !== 'string' ||
            !readInputs.some(input => input.node === writer.node && input.block === id) ||
            writer.truncated !== false || !writerSourceValid || writer.result_type !== block.type ||
            writer.text_body_sha256 !== block.body_sha256)
          throw new Error(`provider-expanded producer context lacks an earlier writer: ${id}`);
        if (receipt.schema === 'natlang.provider-expanded-read-context/2') {
          const pairs = receipt.additional_read_turn_pairs ?? [];
          if (!Array.isArray(pairs) || pairs.some(value => {
            if (!value || typeof value !== 'object') return true;
            const pair = value as Record<string, unknown>;
            const pairRead = pair.block_read as Record<string, unknown> | undefined;
            const pairTurn = pair.model_turn as Record<string, unknown> | undefined;
            const pairInputs = Array.isArray(pairRead?.inputs) ? pairRead.inputs as Record<string, unknown>[] : [];
            const turnInputs = Array.isArray(pairTurn?.inputs) ? pairTurn.inputs as Record<string, unknown>[] : [];
            return !pairRead || pairRead.kind !== 'block_read' || pairRead.call_id !== invocation ||
              pairRead.block !== id || typeof pairRead.node !== 'string' ||
              !pairInputs.some(input => input.node === writer.node && input.block === id) ||
              !pairTurn || pairTurn.kind !== 'model_turn' || pairTurn.call_id !== invocation ||
              !turnInputs.some(input => input.node === pairRead.node && input.port === 'read' && input.block === id);
          })) throw new Error(`provider-expanded context has an invalid repeated read/turn binding: ${id}`);
        }
      }
    externalBodies.set(id, { body: block.body, type: block.type, receipt });
  }
  }
  const typedResultReceipts = (((decision?.assistant as Record<string, unknown> | undefined)?.calls as Record<string, unknown>[] | undefined) ?? [])
    .flatMap(action => {
      const outcome = action.outcome as Record<string, unknown> | undefined;
      return Array.isArray(outcome?.typed_result_writes) ?
        (outcome.typed_result_writes as Record<string, unknown>[]).map(receipt => ({ action, outcome, receipt })) : [];
    });
  const preservedEvalResultCount = typedResultReceipts.filter(({ receipt }) =>
    ['eval-return', 'eval-finish'].includes(String(receipt.source)) && receipt.body_source === undefined).length;
  if (preservedEvalResultCount) count('typed-result-write', 'eval-result-kept-without-source-span', preservedEvalResultCount);
  const edges = options.softStateEdges?.edges ?? [];
  const matchingEdge = (role: 'writer' | 'reader') => edges.filter(edge =>
    role === 'writer' ? edge.writer_call_id === invocationOf(record as Record<string, unknown>) &&
      edge.writer_record_id === rowId && edge.writer_decision_index === decisionIndex :
    edge.reader_call_id === invocationOf(record as Record<string, unknown>) &&
      edge.reader_record_id === rowId && edge.reader_decision_index === decisionIndex);
  const writerEdges = matchingEdge('writer');
  const readerEdges = matchingEdge('reader');
  if (writerEdges.length > 1 || readerEdges.length > 1) throw new Error(`ambiguous validated soft-state edge for ${rowId}`);
  const softEdgeForRecord = writerEdges[0] ?? readerEdges[0];
  if (softEdgeForRecord) {
    const proof = options.softStateEdges;
    const ir = ((record as Record<string, unknown>).task as { program_ir?: { source_groups?: unknown[]; split?: unknown } } | undefined)?.program_ir;
    if (!proof || proof.schema !== 'natlang.validated-runtime-soft-state-edges/1' || proof.status !== 'passed' ||
        proof.validation.validator !== 'validateSoftStateEdge' || proof.source.trajectory_id !== trajectoryId ||
        proof.source.source_row_sha256 !== sourceRef?.source_row_sha256 || proof.source.split !== ir?.split ||
        JSON.stringify(proof.source.source_groups) !== JSON.stringify(ir?.source_groups) ||
        proof.source.transport_mode !== 'text-marker-standin/2' || proof.source.learned_vectors !== false ||
        proof.source.qualification_certificate !== false || proof.source.training_admission !== false)
      throw new Error(`validated soft-state provenance does not bind ${rowId}`);
    const edge = softEdgeForRecord;
    if (edge.expected_type !== 'Neuralese<string>' || !/^[a-z2-7]{20,}$/.test(edge.block_id.replace(/^nz1_/, '')) ||
        !/^[0-9a-f]{64}$/.test(edge.body_sha256)) throw new Error(`invalid soft-state edge descriptor for ${rowId}`);
    if (typeof softEdgeForRecord.body_source !== 'string' ||
        createHash('sha256').update(softEdgeForRecord.body_source).digest('hex') !== softEdgeForRecord.body_sha256)
      throw new Error(`soft-state proof body digest mismatch for ${rowId}`);
  }
  const soft = (name: string, kind: SoftPiece['kind'], text: string): ConvertedPart => {
    // Runtime prompt IDs are stable labels, but their wording can change between
    // collected runs. Bind each converted prompt parameter to its actual text so
    // a corpus merge never expands an old call with a newer prompt under one ID.
    const identity = kind === 'system-prompt' && name.startsWith('prompt:') && !name.startsWith('prompt:system@') ?
      `${name}@${sha12(text)}` : name;
    if (!pieces.has(identity)) pieces.set(identity, { name: identity, kind, text });
    else if (pieces.get(identity)!.text !== text) throw new Error(`soft-piece-identity-collision: ${identity}`);
    return { type: 'soft', name: identity };
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
  const blockName = (edge: ValidatedSoftStateEdge) => `soft-state:${edge.block_id}`;
  const markerBody = (edge: ValidatedSoftStateEdge) => {
    if (edge.writer_record_id !== rowId) return undefined;
    const calls = record.target?.tool_calls ?? [];
    let call = calls.find(item => item.function?.name === 'return_result');
    let args = call ? parseArguments(call.function.arguments) : undefined;
    let value = args?.value;
    let body: string | undefined, rawMarker: string | undefined, markerStart: number | undefined, markerEnd: number | undefined;
    if (call && args?.status === 'success' && typeof value === 'string' &&
        value.startsWith('<|neuralese|>') && value.endsWith('<|/neuralese|>')) {
      body = value.slice('<|neuralese|>'.length, -'<|/neuralese|>'.length);
      rawMarker = value;
    } else {
      const matches = calls.flatMap(candidate => {
        if (candidate.function?.name !== 'eval') return [];
        const candidateArgs = parseArguments(candidate.function.arguments);
        if (candidateArgs?.finish !== true || typeof candidateArgs.code !== 'string') return [];
        const markers = [...candidateArgs.code.matchAll(/<\|neuralese\|>([\s\S]*?)<\|\/neuralese\|>/g)];
        if (markers.length !== 1) return [];
        const match = markers[0]!;
        return [{ call: candidate, args: candidateArgs, body: match[1]!, rawMarker: match[0],
          start: match.index!, end: match.index! + match[0].length }];
      });
      if (matches.length === 1) {
        ({ call, args, body, rawMarker, start: markerStart, end: markerEnd } = matches[0]!);
        value = rawMarker;
      }
    }
    if (!call || !args || typeof body !== 'string' || typeof rawMarker !== 'string')
      throw new Error(`soft-state producer ${rowId} lacks one complete typed Neuralese writer action`);
    if (body !== edge.body_source || createHash('sha256').update(body).digest('hex') !== edge.body_sha256)
      throw new Error(`soft-state body digest mismatch for ${rowId}`);
    return { call, args, rawMarker, body, ...(markerStart === undefined ? {} : { markerStart, markerEnd }) };
  };
  const handoverName = (note: string) => `handover:${sha12(note.trim())}`;
  const invocation = invocationOf(record as Record<string, unknown>);
  const inlineWriters = options.inlineInstructions?.writers.filter(writer => writer.decision_id === (record as Record<string, unknown>).id) ?? [];
  const inlineRead = options.inlineInstructions?.reads.find(read => read.trajectory_id === callOf(record as Record<string, unknown>) && read.invocation_id === invocation);
  const inlineWriter = inlineRead && options.inlineInstructions?.writers.find(writer => writer.writer_id === inlineRead.writer_id);
  const inlineBody = inlineWriter?.body_source ?? inlineWriter?.template_source.slice(1, -1);
  const plainInline = inlineWriter && (inlineWriter.body_source !== undefined ?
    inlineWriter.body_code_source === inlineWriter.template_source.slice(1, -1) && !!inlineWriter.plan.capture_binding_plan :
    inlineBody === inlineWriter.template_segments[0]);
  for (const hold of options.inlineInstructions?.held ?? []) if (hold.decision_id === (record as Record<string, unknown>).id)
    count('inline-instruction', hold.reason);

  // Eval calls that run child natural-language calls: their printed results are another call's output.
  const childCalls = childCallIds(record.messages, childFunctionNames(record as Record<string, unknown>));
  const evalCalls = new Set(record.messages.flatMap(message => message.role === 'assistant' ?
    (message.tool_calls ?? []).filter(call => call.function?.name === 'eval' && typeof call.id === 'string').map(call => call.id!) : []));
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
    const hostContexts = run?.observed_host_contexts?.filter(host => host.invocation === invocation &&
      host.tool_call_id === toolCallId && text.includes(host.value)) ?? [];
    if (hostContexts.length) {
      count('child-result', 'observed-host-result', hostContexts.length);
      for (const host of hostContexts) observedHostMetadata.push({ schema: 'natlang.observed-host-result-context/1',
        origin: 'completed-child-invocation-output', invocation_id: host.invocation, tool_call_id: host.tool_call_id,
        producer_record_id: host.producer_id, result_type: host.result_type, body_sha256: host.body_sha256,
        capture_sha256: host.capture_sha256, visible_as_crisp_context: true, model_writer_target: false, recurrence_edge: false });
      return parts;
    }
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
    if (message.natlang_external_context_input === true) {
      const { natlang_external_context_input: _receiptMarker, ...contextMessage } = message;
      return contextMessage;
    }
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
    if (message.role === 'user' && Array.isArray(message.content) && index <= 1 &&
        record.messages.find(item => item.role === 'user') === message && inlineRead && plainInline &&
        inlineRead.body_block_id && inlineRead.body_source !== undefined) {
      // Authored soft bodies render as multipart openings: prompt text and the
      // Neuralese body marker are separate parts. The inline index has already
      // attested the writer/action/captures and that this body is in the child's
      // Instructions section. Replace only the exact indexed body part.
      const parts = message.content;
      const bodyParts = parts.flatMap((part, partIndex) => {
        const item = part as Record<string, unknown>;
        return item.type === 'neuralese' && item.id === inlineRead.body_block_id ? [partIndex] : [];
      });
      const flattened = parts.map(part => {
        const item = part as Record<string, unknown>;
        if (item.type === 'text' && typeof item.text === 'string') return item.text;
        if (item.type === 'neuralese' && typeof item.id === 'string') return `${item.id}`;
        return '';
      }).join('');
      const instructionStart = flattened.indexOf('Instructions:\n');
      const tail = instructionStart < 0 ? '' : flattened.slice(instructionStart + 'Instructions:\n'.length);
      const boundary = /\n\n(?:In eval\b|Eval also\b)/.exec(tail);
      const section = boundary ? tail.slice(0, boundary.index + 1) : '';
      const marker = `${inlineRead.body_block_id}`;
      const markerPartsMatch = bodyParts.length === 1 && section.split(marker).length === 2;
      const bodyPayloadMatch = bodyParts.length === 1 && ['source', 'text'].every(key => {
        const value = (parts[bodyParts[0]!] as Record<string, unknown>)[key];
        return typeof value !== 'string' || value === inlineRead.body_source;
      });
      if (markerPartsMatch && bodyPayloadMatch) {
        const bodyIndex = bodyParts[0]!;
        count('inline-instruction-read');
        const converted = parts.map((part, partIndex) => partIndex === bodyIndex ?
          { type: 'read', name: inlineRead.writer_id, source: inlineRead.body_source! } : part);
        return { ...message, content: converted };
      }
      count('inline-instruction', 'soft-body-opening-mismatch');
      return message;
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
          if (inlineRead.body_block_id && inlineRead.body_source !== undefined) {
            // A string cannot contain the attested multipart body part.
            count('inline-instruction', 'soft-body-opening-mismatch');
            return message;
          }
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
    if (message.role === 'tool' && message.tool_call_id === 'scope_0' && Array.isArray(message.content) && readerEdges.length) {
      const edge = readerEdges[0]!;
      const openingText = record.messages.find(item => item.role === 'user')?.content;
      if (typeof openingText !== 'string') throw new Error(`soft-state reader ${rowId} has no string opening`);
      const opening = /^You are inside this call: ([^\n]+)/.exec(openingText)?.[1] ?? '';
      const signatureMatch = /\(([^()]*)\)(?::|\s*=>)/.exec(opening);
      const signature = signatureMatch ? `(${signatureMatch[1]})` : '';
      if (!signatureHasExactArgumentPath(signature, edge.consumer_argument, edge.expected_type) ||
          !signatureHasExactArgumentPath(edge.consumer_signature, edge.consumer_argument, edge.expected_type))
        throw new Error(`soft-state reader ${rowId} does not expose exact ${edge.consumer_argument}: ${edge.expected_type}`);
      const matching = message.content.flatMap((part, index) => {
        const item = part as Record<string, unknown>;
        return item.type === 'neuralese' && item.id === edge.block_id ? [index] : [];
      });
      const parts = message.content as Record<string, unknown>[];
      const blockIndex = parts.findIndex(part => part.type === 'neuralese' && part.id === edge.block_id);
      const beforeBlock = parts.slice(0, blockIndex).map(part => part.text)
        .filter((part): part is string => typeof part === 'string').join('');
      const path = edge.consumer_argument.split('.');
      const typedBinding = path.length === 1
        ? beforeBlock.endsWith(`${path[0]}: ${edge.expected_type} = `)
        : path.length === 2 && new RegExp(`${path[0]}: \\{\\s*${path[1]}: ${edge.expected_type}[^}]*\\}\\s*=\\s*\\{\\s*${path[1]}: $`).test(beforeBlock);
      if (matching.length !== 1 || !typedBinding)
        throw new Error(`soft-state reader ${rowId} lacks one exact typed ${edge.consumer_argument} block`);
      // The exact body is pinned in the edge receipt and is independently checked against the writer's raw marker.
      const body = edge.body_source;
      if (createHash('sha256').update(body).digest('hex') !== edge.body_sha256)
        throw new Error(`soft-state reader ${rowId} has an invalid proof body`);
      const content = message.content.map((part, partIndex) => partIndex === matching[0] ?
        { type: 'read', name: blockName(edge), source: body } : part);
      count('soft-state-read');
      return { ...message, content };
    }
    if (message.role === 'tool' && typeof message.content === 'string') {
      if (DYNAMIC_NOTICE.test(message.content)) count('notice', 'dynamic-text');
      let parts = promptParts(message.content, 'text');
      if (message.tool_call_id === 'scope_0') parts = childResultParts(parts, 'argument-read', String(message.tool_call_id))
        .flatMap(part => part.type === 'text' ? listingParts(part.text) : [part]);
      else if (childCalls.has(String(message.tool_call_id)) || evalCalls.has(String(message.tool_call_id)))
        parts = childResultParts(parts, 'child-result', String(message.tool_call_id));
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
          const eligible = linked.filter(writer => writer.body_source !== undefined ?
            writer.body_code_source === writer.template_source.slice(1,-1) && !!writer.plan.capture_binding_plan :
            writer.template_source.slice(1,-1) === writer.template_segments[0]);
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
                parts.push({$write:{name:writer.writer_id,type:'Neuralese<string>',source:writer.body_source ?? args.code.slice(start,end),
                  ...(writer.body_source !== undefined ? { code_source: writer.body_code_source } : {})}});
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
        if (call.function.name === 'eval' && args?.finish === true && typeof args.code === 'string') {
          const edge = writerEdges.find(candidate => candidate.writer_record_id === rowId &&
            candidate.writer_call_id === invocationOf(record as Record<string, unknown>));
          if (edge) {
            const parsed = markerBody(edge);
            if (!parsed || parsed.call.id !== call.id || parsed.markerStart === undefined || parsed.markerEnd === undefined)
              throw new Error(`soft-state eval writer action mismatch for ${rowId}`);
            const code = args.code as string;
            const marker = parsed.rawMarker;
            if (code.slice(parsed.markerStart, parsed.markerEnd) !== marker)
              throw new Error(`soft-state eval marker source mismatch for ${rowId}`);
            const name = blockName(edge);
            changed = true;
            count('soft-state-write');
            return { ...call, neuralese_code: { schema: 'natlang.neuralese-code/1', mode: 'marker-output',
              code_sha256: createHash('sha256').update(code).digest('hex'),
              parts: [
                { type: 'text', text: code.slice(0, parsed.markerStart) },
                { $write: { name, type: edge.expected_type, source: parsed.body, code_source: marker } },
                { type: 'text', text: code.slice(parsed.markerEnd) },
              ],
              sites: [{ name, purpose: 'validated-runtime-soft-state-writer', block_id: edge.block_id,
                body_sha256: edge.body_sha256, writer_call_id: edge.writer_call_id,
                writer_node: edge.writer_node, writer_action: 'eval-finish-true' }] } };
          }
        }
        if (call.function.name === 'return_result' && args?.status === 'success' && 'value' in args) {
          const edge = writerEdges.find(candidate => candidate.writer_record_id === rowId &&
            candidate.writer_call_id === invocationOf(record as Record<string, unknown>));
          const sourceSha = sourceRef?.source_row_sha256;
          const typedReceipts = typedResultReceipts.filter(({ action, outcome: actionOutcome, receipt }) => {
            if (receipt.schema !== 'natlang.typed-result-write/1' || receipt.invocation_id !== invocation ||
                receipt.writer_call_id !== invocation || receipt.source_row_sha256 !== sourceSha ||
                !['typed-text-result', 'typed-text-result-field', 'typed-json-result'].includes(String(receipt.source_kind)) ||
                typeof receipt.block_id !== 'string' || !/^nz1_[a-z2-7]{20,}$/.test(receipt.block_id) ||
                typeof receipt.writer_node !== 'string' || typeof receipt.result_type !== 'string' ||
                !receipt.result_type.startsWith('Neuralese<') || typeof receipt.body_sha256 !== 'string' ||
                !/^[0-9a-f]{64}$/.test(receipt.body_sha256) || typeof receipt.body_source !== 'string' ||
                createHash('sha256').update(receipt.body_source).digest('hex') !== receipt.body_sha256 ||
                !Array.isArray(receipt.result_path) || receipt.result_path[0] !== 'return' ||
                !receipt.result_path.slice(1).every(part => typeof part === 'string' || Number.isSafeInteger(part))) return false;
            if (action.source_tool !== 'return_result' || actionOutcome.name !== 'return_result' ||
                stableJson(actionOutcome.arguments) !== stableJson(args)) return false;
            const tail = receipt.result_path.slice(1) as (string | number)[];
            const rawValue = exactPathGet(args.value, tail);
            if (receipt.source_kind === 'typed-text-result' || receipt.source_kind === 'typed-text-result-field')
              return receipt.result_type === 'Neuralese<string>' && typeof rawValue === 'string' && rawValue === receipt.body_source;
            if (receipt.body_source_basis === 'parsed-json-string-for-concrete-neuralese-result-type') {
              if (typeof rawValue !== 'string' || typeof receipt.raw_model_value_sha256 !== 'string' ||
                  createHash('sha256').update(stableJson(rawValue) ?? '').digest('hex') !== receipt.raw_model_value_sha256) return false;
              try { return stableJson(JSON.parse(rawValue)) === receipt.body_source; } catch { return false; }
            }
            return receipt.body_source_basis === 'canonical-json-of-exact-raw-model-result' &&
              stableJson(rawValue) === receipt.body_source;
          });
          if (typedReceipts.length) {
            const uniqueReceipts = typedReceipts.map(item => item.receipt).filter((receipt, at, all) => all.findIndex(other =>
              stableJson(other.result_path) === stableJson(receipt.result_path)) === at);
            const changedArgs = uniqueReceipts.reduce((current, receipt) => {
              const resultPath = receipt.result_path as (string | number)[];
              const tail = resultPath.slice(1);
              const selectedEdge = edge && edge.writer_node === receipt.writer_node && edge.block_id === receipt.block_id;
              const name = selectedEdge ? blockName(edge!) :
                `typed-result:${receipt.block_id}:${sha12(JSON.stringify([receipt.writer_node, resultPath]))}`;
              const write = { $write: { name, block_id: receipt.block_id, type: receipt.result_type,
                source: receipt.body_source, ...(receipt.source_kind === 'typed-json-result' ? { source_encoding: 'json' } : {}) } };
              return { ...current, value: exactPathSet(current.value, tail, write) };
            }, args);
            if (changedArgs !== args) {
              changed = true;
              count('typed-result-write', undefined, uniqueReceipts.length);
              return { ...call, function: { ...call.function, arguments: JSON.stringify(changedArgs) } };
            }
          }
          if (edge) {
            const parsed = markerBody(edge);
            if (!parsed) throw new Error(`soft-state writer ${rowId} has no validated marker body`);
            if (parsed.call.id !== call.id) throw new Error(`soft-state writer action mismatch for ${rowId}`);
            changed = true;
            count('soft-state-write');
            return { ...call, function: { ...call.function, arguments: JSON.stringify({ ...parsed.args,
              value: { $write: { name: blockName(edge), type: edge.expected_type, source: parsed.body } } }) } };
          }
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

  const hydratedMessages = record.messages.map(message => {
    if (!externalBodies.size || !Array.isArray(message.content)) return message;
    const parts = message.content as Record<string, unknown>[];
    return { ...message, content: parts.map(part => {
      if (part.type !== 'neuralese' || typeof part.id !== 'string' || !externalBodies.has(part.id)) return part;
      const external = externalBodies.get(part.id)!;
      // A producer that this actual invocation reads is a causal recurrence edge,
      // even though the stand-in backend supplied its body as crisp text. Preserve
      // that edge under the same name used by the earlier $write. Configured
      // function definitions remain ordinary external context: they have no
      // same-run writer and must not become synthetic reads or targets.
      return external.receipt.origin === 'same-run-producer' ?
        { type: 'read', name: `soft-state:${part.id}`, source: external.body } :
        { type: 'text', text: external.body };
    }) };
  });
  for (const [id, external] of externalBodies) {
    let occurrences = 0;
    for (const message of record.messages) if (Array.isArray(message.content))
      occurrences += (message.content as Record<string, unknown>[]).filter(part => part.type === 'neuralese' && part.id === id).length;
    if (external.receipt.origin === 'same-run-producer' && occurrences !== 1)
      throw new Error(`provider-expanded producer block lacks one exact context occurrence: ${id}`);
    if (external.receipt.origin === 'configured-function-definition' && occurrences > 1)
      throw new Error(`configured function block has ambiguous message occurrences: ${id}`);
    if (Number(external.receipt.context_occurrences) !== occurrences)
      throw new Error(`provider-expanded context occurrence count mismatch: ${id}`);
  }
  const configuredBodies = [...externalBodies].filter(([, value]) => value.receipt.origin === 'configured-function-definition')
    .filter(([, value]) => Number(value.receipt.context_occurrences) === 0)
    .map(([, value]) => value.body);
  if (configuredBodies.length) {
    if (!hydratedMessages.length || hydratedMessages[0]?.role !== 'system')
      throw new Error(`configured function body has no system context slot for ${rowId}`);
    hydratedMessages.splice(1, 0, ...configuredBodies.map(body => ({ role: 'system', content: body,
      natlang_external_context_input: true })));
  }
  const messages = hydratedMessages.map(convertMessage);
  const target = record.target ? convertMessage(record.target, record.messages.length) : undefined;
  const softStateMetadata = softEdgeForRecord ? [{
    schema: 'natlang.text-neuralese-standin-feature-conversion/1',
    role: writerEdges.length ? 'writer' : 'reader',
    block_id: softEdgeForRecord.block_id,
    writer_node: softEdgeForRecord.writer_node,
    reader_node: softEdgeForRecord.reader_node,
    argument: softEdgeForRecord.consumer_argument,
    expected_type: softEdgeForRecord.expected_type,
    body_sha256: softEdgeForRecord.body_sha256,
    ...(writerEdges.length ? { raw_marker_value: markerBody(softEdgeForRecord)?.rawMarker } : {}),
    transport_mode: options.softStateEdges?.source.transport_mode,
    learned_vectors: false,
    qualification_certificate: false,
    training_admission: false,
    validation_review_sha256: options.softStateEdges?.validation.review_sha256,
    validation_result_sha256: options.softStateEdges?.validation.result_sha256,
  }] : undefined;
  const externalContextMetadata = [...externalBodies.values()].map(value => ({
    schema: 'natlang.external-context-input/1', origin: value.receipt.origin,
    block_id: (value.receipt.block as Record<string, unknown>).id,
    type: value.type, body_sha256: createHash('sha256').update(value.body).digest('hex'),
    invocation_id: value.receipt.invocation_id,
    parent_invocation_id: value.receipt.parent_invocation_id ?? null,
    transport_provenance_sha256: value.receipt.transport_provenance_sha256,
    raw_request_sha256: value.receipt.raw_request_sha256,
    rendered_request_sha256: value.receipt.rendered_request_sha256,
    source_row_sha256: value.receipt.source_row_sha256,
    trace_sha256: (record as Record<string, unknown>).provenance &&
      ((record as Record<string, unknown>).provenance as Record<string, unknown>).trace_sha256,
      read_node: (value.receipt.block_read as Record<string, unknown>).node,
    model_turn_node: (value.receipt.model_turn as Record<string, unknown>).node,
    ...(Array.isArray(value.receipt.additional_read_turn_pairs) ? { additional_read_nodes:
      (value.receipt.additional_read_turn_pairs as Record<string, unknown>[]).map(pair =>
        ((pair.block_read as Record<string, unknown>).node)), additional_model_turn_nodes:
      (value.receipt.additional_read_turn_pairs as Record<string, unknown>[]).map(pair =>
        ((pair.model_turn as Record<string, unknown>).node)) } : {}),
    ...(value.receipt.origin === 'same-run-producer' ? { producer_write_node:
      (value.receipt.producer_write as Record<string, unknown>).node,
      producer_call_id: (value.receipt.producer_write as Record<string, unknown>).call_id,
      writer_target_selected: value.receipt.writer_target_selected === false ? false : null,
      learner_representation: value.receipt.writer_target_selected === false ?
        'typed-read-from-authenticated-runtime-writer-event-context-only' : 'typed-read-linked-to-existing-writer' } :
      { learner_representation: 'crisp-external-function-context' }),
    learned_vectors: false, qualification_certificate: false, training_admission: false,
  }));
  return { record: { ...record, messages, ...(target ? { target } : {}),
    neuralese_conversion: { version: NEURALESE_CONVERSION_VERSION, sites,
      ...(observedHostMetadata.length ? { observed_host_result_contexts: observedHostMetadata } : {}),
      ...(externalContextMetadata.length ? { external_context_inputs: externalContextMetadata } : {}),
      ...(softStateMetadata ? { soft_state_edges: softStateMetadata } : {}) } }, pieces: [...pieces.values()] };
}

function signatureHasExactParameter(signature: string, argument: string, type: string): boolean {
  const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  return new RegExp(`(?:^|\\(|, )${escapeRegex(argument)}: ${escapeRegex(type)}(?=, |\\))`).test(signature);
}

function signatureHasExactArgumentPath(signature: string, argumentPath: string, type: string): boolean {
  const path = argumentPath.split('.');
  if (path.length === 1) return signatureHasExactParameter(signature, path[0]!, type);
  if (path.length !== 2 || path.some(part => !/^[A-Za-z_$][\w$]*$/.test(part))) return false;
  const escapeRegex = (value: string) => value.replace(/[.*+?^${}()|[\\]\\]/g, '\\$&');
  return new RegExp(`(?:^|\\(|, )${escapeRegex(path[0]!)}: \\{\\s*[^{}]*\\b${escapeRegex(path[1]!)}: ${escapeRegex(type)}(?=\\s*[,}])`).test(signature);
}

function parseArguments(text: string): Record<string, unknown> | undefined {
  try { const value = JSON.parse(text); return value && typeof value === 'object' ? value : undefined; } catch { return undefined; }
}
