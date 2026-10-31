/** Explicit provider transport for teacher-only Neuralese values represented by exact text, never vectors. */
import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { MemoryNeuraleseStore, type NeuraleseBlockMeta } from '../native/neuralese-store.js';
import type { NeuraleseRuntimeOptions } from '../native/neuralese.js';

export const TEACHER_TEXT_NEURALESE_VERSION = 'teacher-text/1';
export const TEACHER_TEXT_NEURALESE_DIALECT = 'nd:teacher-text/1';
export const TEACHER_TEXT_NEURALESE_PROMPT = `\n\nExplicit teacher text-Neuralese transport for this collection only:\n` +
  `For a declared Neuralese<T> result, return the exact body inside <|neuralese|>BODY<|/neuralese|>, or return a plain ` +
  `string only when the declared result is Neuralese<string>. A provided teacher-text Neuralese value is displayed as a ` +
  `labeled block containing its exact JSON-escaped body. Use that body as the value; never copy its ID as content. ` +
  `Ordinary string results remain ordinary strings. This transport stores literal text only and does not provide native ` +
  `state, model vectors, token IDs, learned representations, or a student writer.`;

export type TeacherTextBlock = { id: string; type: string; body: string; body_sha256: string;
  producer_call_id: string; writer_node?: string; format?: 'teacher-text'; teacher_text?: true;
  learned_vectors?: false; tokenizer_ids?: false };
export type TeacherTextNeuralese = { runtime: NeuraleseRuntimeOptions;
  wrap<T extends (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>>(send: T): T;
  provenance: { schema: 'natlang.teacher-text-neuralese-transport/1'; mode: typeof TEACHER_TEXT_NEURALESE_VERSION;
    learned_vectors: false; qualification_certificate: false; training_admission: false;
    tokenizer: { kind: 'provider-native-unspecified'; ids_claimed: false } } };

const hex = (text: string): string => createHash('sha256').update(text, 'utf8').digest('hex');
const base32 = (bytes: Uint8Array): string => {
  const alphabet = 'abcdefghijklmnopqrstuvwxyz234567'; let bits = 0, value = 0, out = '';
  for (const byte of bytes) { value = (value << 8) | byte; bits += 8;
    while (bits >= 5) { out += alphabet[(value >>> (bits - 5)) & 31]; bits -= 5; } }
  if (bits) out += alphabet[(value << (5 - bits)) & 31];
  return out;
};
const canonical = (value: unknown): string => Array.isArray(value) ? '[' + value.map(canonical).join(',') + ']' :
  value && typeof value === 'object' ? '{' + Object.entries(value as Record<string, unknown>).sort(([a],[b])=>a.localeCompare(b))
    .map(([key,item])=>JSON.stringify(key)+':'+canonical(item)).join(',') + '}' : JSON.stringify(value);

/** A run-local store whose IDs, payload, type and writer identity are all derived from actual provider text. */
export function createTeacherTextNeuralese(): TeacherTextNeuralese {
  const store = new MemoryNeuraleseStore();
  const blocks = new Map<string, TeacherTextBlock>();
  let writeOrdinal = 0;
  const get = (id: string) => blocks.get(id);
  const recordWriter = (id: string, node: string) => {
    const found = blocks.get(id);
    if (!found || !node) throw new Error('teacher-text writer occurrence has no matching case-local block');
    if (found.writer_node && found.writer_node !== node) throw new Error('teacher-text block was assigned two writer occurrences');
    found.writer_node = node;
  };
  const port = { dialect: TEACHER_TEXT_NEURALESE_DIALECT,
    async write(text: string, options: { type?: string; producer?: Record<string, unknown>; truncated?: boolean } = {}): Promise<NeuraleseBlockMeta> {
      if (options.truncated) throw new Error('teacher-text transport refuses an unfinished Neuralese body');
      const type = options.type ?? (typeof options.producer?.result_type === 'string' ? options.producer.result_type : undefined);
      if (!type?.startsWith('Neuralese<')) throw new Error('teacher-text writer requires an exact declared Neuralese result type');
      const producer = options.producer ?? {};
      const producerCall = producer.call_id;
      if (typeof producerCall !== 'string' || !producerCall) throw new Error('teacher-text writer requires its actual producing invocation');
      const bodySha = hex(text), ordinal = ++writeOrdinal;
      const id = 'nzt1_' + base32(createHash('sha256').update(canonical({ type, body_sha256: bodySha,
        producer_call_id: producerCall, ordinal }), 'utf8').digest());
      const block: TeacherTextBlock = { id, type, body: text, body_sha256: bodySha, producer_call_id: producerCall };
      blocks.set(id, block);
      const meta: NeuraleseBlockMeta = { id, dialect: TEACHER_TEXT_NEURALESE_DIALECT, type, length: text.length, width: 0,
        producer: { kind: 'teacher-text-emulation', source_kind: producer.source_kind ?? 'typed-text-result',
          source: producer.source ?? 'return_result', marker_context: producer.marker_context ?? 'return-result',
          result_type: type, call_id: producerCall,
          text_body_sha256: bodySha, transport_version: TEACHER_TEXT_NEURALESE_VERSION } };
      store.note(meta);
      return meta;
    } };
  const runtime: NeuraleseRuntimeOptions = { store, port, teacherText: { get, recordWriter } };
  const provenance: TeacherTextNeuralese['provenance'] = { schema: 'natlang.teacher-text-neuralese-transport/1',
    mode: TEACHER_TEXT_NEURALESE_VERSION, learned_vectors: false, qualification_certificate: false,
    training_admission: false, tokenizer: { kind: 'provider-native-unspecified', ids_claimed: false } };
  const wrap = <T extends (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>>(send: T): T => {
    const renderBlock = (part: Record<string, unknown>, expanded: Map<string, TeacherTextBlock>): string => {
      if (part.type !== 'teacher_text' || typeof part.id !== 'string') throw new Error('invalid teacher-text provider part');
      const block = get(part.id);
      if (!block || typeof part.value_type === 'string' && part.value_type !== block.type)
        throw new Error('teacher-text provider part lacks its exact case-local typed body');
      if (!block.writer_node) throw new Error('teacher-text provider cannot read a block before its writer occurrence is recorded');
      expanded.set(block.id, { ...structuredClone(block), format: 'teacher-text', teacher_text: true,
        learned_vectors: false, tokenizer_ids: false });
      return `[[teacher-text Neuralese id=${block.id} type=${block.type} body=${JSON.stringify(block.body)}]]`;
    };
    const isPartList = (value: unknown): value is Record<string, unknown>[] => Array.isArray(value) && value.length > 0 &&
      value.every(part => !!part && typeof part === 'object' && ['text','teacher_text'].includes(String((part as Record<string,unknown>).type)));
    const render = async (value: unknown, expanded: Map<string, TeacherTextBlock>): Promise<unknown> => {
      if (Array.isArray(value)) {
        if (isPartList(value)) {
          const pieces = value.map(part => part.type === 'text' ? String(part.text ?? '') : renderBlock(part, expanded));
          const combined = pieces.join('');
          if (!value.some(part => part.type === 'teacher_text')) return combined;
          let parsed: unknown;
          try { parsed = JSON.parse(combined); } catch { return combined; }
          return JSON.stringify(parsed);
        }
        return Promise.all(value.map(item => render(item, expanded)));
      }
      if (!value || typeof value !== 'object') return value;
      const record = value as Record<string, unknown>;
      if (record.type === 'teacher_text') return { type: 'text', text: renderBlock(record, expanded) };
      const out: Record<string, unknown> = {};
      for (const [key, item] of Object.entries(record)) {
        if (key === 'tool_calls' && Array.isArray(item)) {
          out[key] = await Promise.all(item.map(async call => {
            if (!call || typeof call !== 'object') return call;
            const c = call as Record<string,unknown>, fn = c.function;
            if (!fn || typeof fn !== 'object') return render(c, expanded);
            const f = fn as Record<string,unknown>;
            return { ...c, function: { ...f, ...(Object.hasOwn(f,'arguments') ? { arguments: await render(f.arguments,expanded) } : {}) } };
          }));
        } else out[key] = await render(item, expanded);
      }
      return out;
    };
    const adapted = (async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
      if (request.adapters?.length || request.guidance || request.template)
        throw new Error('teacher-text transport cannot apply native adapters, native guidance, or template readout');
      const expanded = new Map<string, TeacherTextBlock>();
      const renderedMessages = await render(request.messages, expanded);
      if (!Array.isArray(renderedMessages)) throw new TypeError('teacher-text provider messages must remain an array');
      const renderedRequest = { ...request, messages: renderedMessages };
      const response = await send(renderedRequest, signal);
      const markerOutputs = [response.text, response.calls].flatMap(value => {
        const found: string[] = [];
        const visit = (item: unknown) => { if (typeof item === 'string') {
          for (const match of item.matchAll(/<\|neuralese\|>([\s\S]*?)(?:<\|\/neuralese\|>|$)/g)) found.push(match[1] ?? '');
        } else if (Array.isArray(item)) item.forEach(visit); else if (item && typeof item === 'object') Object.values(item).forEach(visit); };
        visit(value); return found;
      }).map(body => ({ body, body_sha256: hex(body), body_chars: body.length }));
      return { ...response, transport_provenance: { ...provenance, raw_request_sha256: hex(canonical(request)),
        rendered_request_sha256: hex(canonical(renderedRequest)), rendered_messages: renderedMessages,
        raw_response_sha256: response.raw_response ? hex(canonical(response.raw_response)) : null,
        expanded_input_blocks: [...expanded.values()], marker_outputs: markerOutputs } };
    }) as T;
    Object.defineProperty(adapted, 'neuralese', { value: true, enumerable: true });
    return adapted;
  };
  return { runtime, wrap, provenance };
}
