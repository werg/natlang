/**
 * Neuralese states as the serving engines read them by default, without an engine: `in-context-text` (serve/engine.py
 * `reads_in_context`, the fork's `native_state_registry::resolve`). A state written in its producing context (an operator
 * reply captured in place, a `Neuralese<T>` value written as the span after its producing call) is read by a consumer as
 * its exact tokens where the state stands, with no consume markers. This transport keeps each state's text and gives
 * every request the consumer's messages with that text in place of the state, so a scripted replay (authored
 * trajectories: curriculum.replayReference, scripts/rl_tasks/authored_replay.mjs and reference_episodes.mjs; the
 * replay of authored teacher episodes, collector `inContextNeuralese`) runs the runtime's own Neuralese code paths
 * (learned compaction's operator calls and captures, typed `Neuralese<T>` writes and reads) and records what the model
 * reads in collection. It holds no vectors and claims no token ids; it is never a training-time state source.
 */
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { digest } from '../native/hash.js';
import { MemoryNeuraleseStore, type NativeStateArtifact, type NeuraleseBlockMeta } from '../native/neuralese-store.js';
import { isNativeStateId, runtimeTextToParts, type NeuraleseRuntimeOptions } from '../native/neuralese.js';
import type { CapturedCall } from '../native/operator-frames.js';

export const IN_CONTEXT_DIALECT = 'native-state/in-context-text';
export const IN_CONTEXT_RECEIPT = 'in-context-text';

export type InContextNeuralese = {
  runtime: NeuraleseRuntimeOptions;
  /** The text a state is read as (its reply or written text), undefined for a state this transport did not write. */
  text(id: string): string | undefined;
  /** `driver` with every request's states replaced by their text (`neuralese: true`, its `decide` likewise). */
  wrap<T extends (request: ModelTurnRequest, ...rest: never[]) => Promise<ModelTurn>>(driver: T): T;
  /** `value` (messages, parts, strings) with every state replaced by its text. */
  inline<V>(value: V): V;
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
  let ordinal = 0;
  const original = store.getNativeState.bind(store);
  store.getNativeState = async (id: string) => states.get(id) ?? original(id);
  const record = (text: string, type: string | undefined, producer: Record<string, unknown>): NeuraleseBlockMeta => {
    const id = 'nzs1_' + base32(digest(JSON.stringify([++ordinal, type ?? null, text]))).slice(0, 52);
    states.set(id, { id, tensors: {}, metadata: { schema: 'neuralese.native-state/1', text, architecture: 'in-context-text',
      tokenizer_fingerprint: ZERO, checkpoint_sha256: ZERO, embedding_weight_sha256: ZERO, layer_kinds: [], producer_kind: 'span' } });
    const meta: NeuraleseBlockMeta = { id, dialect: IN_CONTEXT_DIALECT, ...(type ? { type } : {}), length: text.length, width: 0,
      producer: { kind: 'in-context-text', ...producer } };
    store.note(meta);
    return meta;
  };
  const capture = async (call: CapturedCall, captureOptions?: { type?: string }) =>
    record(call.write ?? call.select ?? call.text ?? '', captureOptions?.type, { source_kind: call.write !== undefined ? 'write' : 'operator-capture' });
  const port = { dialect: IN_CONTEXT_DIALECT,
    async write(text: string, writeOptions: { type?: string; producer?: Record<string, unknown>; truncated?: boolean } = {}) {
      if (writeOptions.truncated) throw new Error('in-context Neuralese refuses an unfinished body');
      return record(text, writeOptions.type, writeOptions.producer ?? {});
    } };
  const text = (id: string) => states.get(id)?.metadata.text;
  const known = (id: string): string => {
    const found = text(id);
    if (found === undefined) throw new Error(`in-context Neuralese: ${id} was not written in this run, so it has no text to read`);
    return found;
  };
  /** Text with each state sentinel replaced by its text (`escape`: inside JSON string literals). */
  const inlineText = (value: string, escape: boolean, read: Set<string>): string => {
    const parts = runtimeTextToParts(value);
    if (parts.every(part => part.type === 'text')) return value;
    return parts.map(part => {
      if (part.type === 'text') return part.text;
      if (!isNativeStateId(part.id)) throw new Error(`in-context Neuralese reads native states only, not ${part.id}`);
      read.add(part.id);
      return escape ? JSON.stringify(known(part.id)).slice(1, -1) : known(part.id);
    }).join('');
  };
  const inlineMessages = (value: unknown, read: Set<string>): unknown => {
    if (typeof value === 'string') return inlineText(value, false, read);
    if (Array.isArray(value)) {
      // content parts: a native_state part reads as its text, so the message is plain text again
      if (value.length && value.every(part => part && typeof part === 'object' && ['text', 'native_state'].includes(String((part as { type?: unknown }).type))))
        return value.map(part => (part as { type: string }).type === 'text' ? inlineText(String((part as { text?: unknown }).text ?? ''), false, read) :
          (read.add(String((part as { id?: unknown }).id)), known(String((part as { id?: unknown }).id)))).join('');
      return value.map(item => inlineMessages(item, read));
    }
    if (!value || typeof value !== 'object') return value;
    const out: Record<string, unknown> = {};
    for (const [key, item] of Object.entries(value as Record<string, unknown>)) {
      if (key === 'arguments' && typeof item === 'string') out[key] = inlineText(item, true, read);
      else out[key] = inlineMessages(item, read);
    }
    return out;
  };
  const inline = <V>(value: V): V => inlineMessages(value, new Set()) as V;
  const wrap = <T extends (request: ModelTurnRequest, ...rest: never[]) => Promise<ModelTurn>>(driver: T): T => {
    const wrapped = (async (request: ModelTurnRequest, ...rest: never[]) => {
      const read = new Set<string>();
      const messages = inlineMessages(request.messages, read) as ModelTurnRequest['messages'];
      const response = await driver(read.size ? { ...request, messages } : request, ...rest);
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
  return { runtime: { store, port, dialect: IN_CONTEXT_DIALECT, capture, ...(options.compaction ? { compaction: options.compaction } : {}) },
    text, wrap, inline };
}
