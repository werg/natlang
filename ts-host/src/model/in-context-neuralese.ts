/**
 * Neuralese states as the serving engines read them, without an engine. A value's text written after its producing call
 * (a typed `Neuralese<T>` result or argument: `producer_kind` 'span') is read in context, as its exact tokens where the
 * state stands (serve/engine.py `reads_in_context`). A reply captured in its own context (a learned operator's summary,
 * a `compact_history` note: 'capture') is spliced by every engine; the learner computes it in-graph from its producer
 * document (the operator call's prompt, its reply up to the content, produce_open, the content: lower_rollout_turns
 * `write_sources`). This transport keeps each state's text and gives every request a span as its text in place and a
 * capture as a `{type: "native_state", id}` content part, so a scripted replay (authored trajectories:
 * curriculum.replayReference, scripts/rl_tasks/authored_replay.mjs and reference_episodes.mjs; the replay of authored
 * teacher episodes, collector `inContextNeuralese`) runs the runtime's own Neuralese code paths (learned compaction's
 * operator calls and captures, typed `Neuralese<T>` writes and reads) and records what collection records. It holds no
 * vectors and claims no token ids; the learner computes captured states from their documents.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { digest } from '../native/hash.js';
import { MemoryNeuraleseStore, type NativeStateArtifact, type NeuraleseBlockMeta } from '../native/neuralese-store.js';
import { isNativeStateId, nativeStateSentinel, runtimeTextToParts, type NeuraleseRuntimeOptions } from '../native/neuralese.js';
import type { CapturedCall } from '../native/operator-frames.js';

export const IN_CONTEXT_DIALECT = 'native-state/in-context-text';
export const IN_CONTEXT_RECEIPT = 'in-context-text';

export type InContextNeuralese = {
  runtime: NeuraleseRuntimeOptions;
  /** The text a state is read as (its reply or written text), undefined for a state this transport did not write. */
  text(id: string): string | undefined;
  /** `driver` with every request's states replaced by their text (`neuralese: true`, its `decide` likewise). */
  wrap<T extends (request: ModelTurnRequest, ...rest: never[]) => Promise<ModelTurn>>(driver: T): T;
  /** `value` (messages, parts, strings) as `wrap` gives it to the driver: spans as their text, spliced states as parts. */
  inline<V>(value: V): V;
  /** The spliced states written so far: each with the request that produced it (its invocation and that invocation's
   * turn, counted from 0), its text, and whether it is a text inside the reply (`select`) or the reply's content. */
  captures(): { id: string; invocation_id: string; turn: number; text: string; select: boolean }[];
};

const base32 = (bytes: Uint8Array): string => {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz234567'; let bits = 0, value = 0, out = '';
  for (const byte of bytes) { value = (value << 8) | byte; bits += 8;
    while (bits >= 5) { out += alphabet[(value >>> (bits - 5)) & 31]; bits -= 5; } }
  if (bits) out += alphabet[(value << (5 - bits)) & 31];
  return out;
};
const ZERO = '0'.repeat(64);

/** A run-local in-context Neuralese transport (module docstring). `compaction` sets the runtime's compaction setting;
 * left out, the agent's own setting applies (NATLANG_NEURALESE_COMPACTION, the model-facing `learned-compaction`). */
export function createInContextNeuralese(options: { compaction?: NeuraleseRuntimeOptions['compaction'] } = {}): InContextNeuralese {
  const store = new MemoryNeuraleseStore();
  const states = new Map<string, NativeStateArtifact>();
  // A spliced state's producing request (`captures`): the index of the wrapped driver's request that produced it.
  const produced = new Map<string, { invocation: string; turn: number }>();
  const turns = new Map<string, number>();
  // A capture names its producing request by that request's messages (operator-frames.ts callContext keeps them).
  const requestOf = new WeakMap<object, { invocation: string; turn: number }>();
  let ordinal = 0;
  const original = store.getNativeState.bind(store);
  store.getNativeState = async (id: string) => states.get(id) ?? original(id);
  const record = (text: string, type: string | undefined, producer: Record<string, unknown>, kind: 'span' | 'capture',
      from?: unknown): NeuraleseBlockMeta => {
    const id = 'nzs1_' + base32(digest(JSON.stringify([++ordinal, type ?? null, text]))).slice(0, 52);
    states.set(id, { id, tensors: {}, metadata: { schema: 'neuralese.native-state/1', text, architecture: 'in-context-text',
      tokenizer_fingerprint: ZERO, checkpoint_sha256: ZERO, embedding_weight_sha256: ZERO, layer_kinds: [], producer_kind: kind } });
    const at = from && typeof from === 'object' ? requestOf.get(from) : undefined;
    if (kind === 'capture' && at) produced.set(id, at);
    const meta: NeuraleseBlockMeta = { id, dialect: IN_CONTEXT_DIALECT, ...(type ? { type } : {}), length: text.length, width: 0,
      producer: { kind: 'in-context-text', producer_kind: kind, ...producer } };
    store.note(meta);
    return meta;
  };
  // A reply captured in its own context (an operator's summary, a compact_history note) is a 'capture': the engines
  // splice it (DECISIONS 2026-11-25, capture-kind strand). A value's text written after its producing call is a 'span'.
  const capture = async (call: CapturedCall, captureOptions?: { type?: string }) =>
    call.write !== undefined ? record(call.write, captureOptions?.type, { source_kind: 'write' }, 'span') :
      record(call.select ?? call.text ?? '', captureOptions?.type, { source_kind: call.select !== undefined ? 'select' : 'operator-capture',
        ...(call.select !== undefined ? { select: true } : {}) }, 'capture', call.messages);
  const port = { dialect: IN_CONTEXT_DIALECT,
    async write(text: string, writeOptions: { type?: string; producer?: Record<string, unknown>; truncated?: boolean } = {}) {
      if (writeOptions.truncated) throw new Error('in-context Neuralese refuses an unfinished body');
      return record(text, writeOptions.type, writeOptions.producer ?? {}, 'span');
    } };
  const text = (id: string) => states.get(id)?.metadata.text;
  const spliced = (id: string) => states.get(id)?.metadata.producer_kind === 'capture';
  const known = (id: string): string => {
    const found = text(id);
    if (found === undefined) throw new Error(`in-context Neuralese: ${id} was not written in this run, so it has no text to read`);
    return found;
  };
  type Part = { type: 'text'; text: string } | { type: 'native_state'; id: string };
  /** The parts of a text: a span read as its text, a spliced state as a `native_state` part (`escape`: inside JSON
   * string literals, where no state can stand, every state is its escaped text). */
  const partsOf = (value: string, escape: boolean, read: Set<string>): Part[] => {
    const out: Part[] = [];
    const push = (piece: string) => { const last = out.at(-1); if (last?.type === 'text') last.text += piece; else out.push({ type: 'text', text: piece }); };
    for (const part of runtimeTextToParts(value)) {
      if (part.type === 'text') { push(part.text); continue; }
      if (!isNativeStateId(part.id)) throw new Error(`in-context Neuralese reads native states only, not ${part.id}`);
      if (!escape && spliced(part.id)) { known(part.id); out.push({ type: 'native_state', id: part.id }); continue; }
      read.add(part.id);
      push(escape ? JSON.stringify(known(part.id)).slice(1, -1) : known(part.id));
    }
    return out;
  };
  const content = (parts: Part[]): string | Part[] =>
    parts.every(part => part.type === 'text') ? parts.map(part => (part as { text: string }).text).join('') : parts;
  const inlineMessages = (value: unknown, read: Set<string>): unknown => {
    if (typeof value === 'string') return content(partsOf(value, false, read));
    if (Array.isArray(value)) {
      // content parts: a span part reads as its text, a spliced state stays a part
      if (value.length && value.every(part => part && typeof part === 'object' && ['text', 'native_state'].includes(String((part as { type?: unknown }).type))))
        return content(value.flatMap(part => (part as { type: string }).type === 'text' ? partsOf(String((part as { text?: unknown }).text ?? ''), false, read) :
          partsOf(nativeStateSentinel(String((part as { id?: unknown }).id)), false, read)).reduce<Part[]>((out, part) => {
          const last = out.at(-1);
          if (part.type === 'text' && last?.type === 'text') last.text += part.text; else out.push({ ...part });
          return out;
        }, []));
      return value.map(item => inlineMessages(item, read));
    }
    if (!value || typeof value !== 'object') return value;
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (key === 'arguments' && typeof item === 'string') out[key] = (content(partsOf(item, true, read)) as string);
      else out[key] = inlineMessages(item, read);
    }
    return out;
  };
  const inline = <V>(value: V): V => inlineMessages(value, new Set()) as V;
  const wrap = <T extends (request: ModelTurnRequest, ...rest: never[]) => Promise<ModelTurn>>(driver: T): T => {
    const wrapped = (async (request: ModelTurnRequest, ...rest: never[]) => {
      const invocation = request.invocation_id ?? '';
      const turn = turns.get(invocation) ?? 0;
      turns.set(invocation, turn + 1);
      if (request.messages && typeof request.messages === 'object') requestOf.set(request.messages, { invocation, turn });
      const read = new Set<string>();
      const messages = inlineMessages(request.messages, read) as ModelTurnRequest['messages'];
      const response = await driver({ ...request, messages }, ...rest);
      if (!read.size) return response;
      // The engines' receipt for each state read as its tokens (agent.ts records the read as `in-context-text`).
      const raw = response.raw_response ?? {};
      const native = (raw.native_state ?? {}) as { imports?: unknown[] };
      return { ...response, raw_response: { ...raw, native_state: { ...native, imports: [...(native.imports ?? []),
        ...[...read].map(id => ({ id, mode: IN_CONTEXT_RECEIPT, observed_native_state_id: id }))] } } };
    }) as unknown as T;
    for (const key of Object.keys(driver)) (wrapped as unknown as Record<string, unknown>)[key] = (driver as unknown as Record<string, unknown>)[key];
    const decide = (driver as unknown as { decide?: (request: { messages: unknown[] }, ...rest: unknown[]) => Promise<unknown> }).decide;
    Object.assign(wrapped, { neuralese: true,
      ...(decide ? { decide: (request: { messages: unknown[] }, ...rest: unknown[]) => decide({ ...request, messages: inline(request.messages) }, ...rest) } : {}) });
    return wrapped;
  };
  const captures = () => [...produced].map(([id, at]) => ({ id, invocation_id: at.invocation, turn: at.turn, text: known(id),
    select: (store.peek?.(id)?.producer as { select?: boolean } | undefined)?.select === true }));
  return { runtime: { store, port, dialect: IN_CONTEXT_DIALECT, capture, ...(options.compaction ? { compaction: options.compaction } : {}) },
    text, wrap, inline, captures };
}
