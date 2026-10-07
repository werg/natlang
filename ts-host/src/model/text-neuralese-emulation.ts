/**
 * Explicit text-channel transport for teacher runs whose provider cannot carry Neuralese content parts.
 * This is protocol emulation only: blocks contain deterministic hash stand-in vectors, never learned vectors.
 */
import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../native/neuralese-store.js';
import type { NeuraleseBlockMeta, NeuralesePort, NeuraleseStore } from '../native/neuralese-store.js';
import type { NeuraleseRuntimeOptions } from '../native/neuralese.js';

export const TEXT_NEURALESE_EMULATION_VERSION = 'text-marker-standin/3';
export const TEXT_NEURALESE_DIALECT = 'nd:text-teacher-emulation/1';
export const TEXT_NEURALESE_WIDTH = 32;

export const TEXT_NEURALESE_EMULATION_PROMPT = `\n\nDeclared Neuralese text-channel emulation for this run:\n` +
  `Use <|neuralese|>BODY<|/neuralese|> in return_result only when the current call declares ` +
  `a Neuralese<T> result. This marker is transport syntax, not a JavaScript string. In eval code, use it only as ` +
  `an unquoted value in an explicitly typed Neuralese position, for example ` +
  `const note: Neuralese<string> = <|neuralese|>the note text<|/neuralese|>; never put it inside a quoted string. ` +
  `For ordinary string results, return ordinary strings. A prior typed reference is presented as a labeled text block ` +
  `with its exact body; it is already the typed input, so pass the reference itself to a child argument declared ` +
  `Neuralese<string>. Do not try readText, read_code, or another helper to unwrap it. ` +
  `When a call's declared result is Neuralese<string>, invoke the return_result tool directly with ` +
  `{"status":"success","value":"<|neuralese|>your actual prose answer<|/neuralese|>"}; the host creates ` +
  `the typed block from that tool argument. Do not call return_result from inside eval. ` +
  `eval({code, finish:true}) completes its fresh typed expression; returning the quoted string ` +
  `"<|neuralese|>note<|/neuralese|>" is wrong because the marker is transport syntax, not a string value. ` +
  `Neuralese is a built-in type, not a callable function: do not redefine it or use read_code("Neuralese") to construct the answer.\n`;

export type TextNeuraleseEmulation = {
  store: NeuraleseStore;
  port: NeuralesePort;
  runtime: NeuraleseRuntimeOptions;
  wrap<T extends (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>>(send: T): T;
};

const sha256 = (value: string): string => createHash('sha256').update(value).digest('hex');

function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value as Record<string, unknown>)
    .sort(([a], [b]) => a.localeCompare(b)).map(([key, item]) => `${JSON.stringify(key)}:${canonical(item)}`).join(',')}}`;
  return JSON.stringify(value);
}

function markerBodies(value: unknown, out: string[] = []): string[] {
  if (typeof value === 'string') {
    const marker = /<\|neuralese\|>([\s\S]*?)(?:<\|\/neuralese\|>|$)/g;
    for (const match of value.matchAll(marker)) out.push(match[1] ?? '');
  } else if (Array.isArray(value)) for (const item of value) markerBodies(item, out);
  else if (value && typeof value === 'object') for (const item of Object.values(value)) markerBodies(item, out);
  return out;
}

function markerRenderer(store: NeuraleseStore) {
  // Neuralese tensor stores intentionally do not expose readable text. Keep an explicit case-local literal sidecar
  // keyed by the real block ID, and verify each expansion against the stored block's producer digest.
  const literalBodies = new Map<string, { type?: string; text: string; text_sha256: string }>();
  type ExpandedInputBlock = { id: string; type: string | null; body: string; body_sha256: string; learned_vectors: false };
  type ExpandedInputs = Map<string, ExpandedInputBlock>;
  const renderPart = async (part: Record<string, unknown>, expanded: ExpandedInputs) => {
    if (part.type !== 'neuralese') return part;
    if (typeof part.id !== 'string') throw new Error('text Neuralese emulation received a part without a block ID');
    const block = await store.get(part.id);
    const literal = literalBodies.get(part.id);
    if (!block) throw new Error(`text Neuralese emulation cannot render unavailable block ${part.id}`);
    if (!literal) throw new Error(`text Neuralese emulation has no stored literal body for block ${part.id}`);
    if (block.meta.producer?.text_body_sha256 !== literal.text_sha256)
      throw new Error(`text Neuralese literal digest does not match stored block ${part.id}`);
    expanded.set(part.id, { id: part.id, type: literal.type ?? null, body: literal.text,
      body_sha256: literal.text_sha256, learned_vectors: false });
    const label = `[[Neuralese text block id=${part.id}${literal.type ? ` type=${literal.type}` : ''}; exact JSON string body=${JSON.stringify(literal.text)}]]`;
    return { type: 'text', text: label };
  };
  const isParts = (value: unknown): value is Record<string, unknown>[] => Array.isArray(value) && value.length > 0 &&
    value.every(part => part && typeof part === 'object' &&
      ((part as Record<string, unknown>).type === 'text' || (part as Record<string, unknown>).type === 'neuralese'));
  const renderArgument = async (value: unknown, expanded: ExpandedInputs): Promise<unknown> => {
    if (!isParts(value)) return visit(value, expanded);
    const replacements = new Map<string, string>();
    const fragments = await Promise.all(value.map(async part => {
      if (part.type !== 'neuralese') return String(part.text ?? '');
      const block = await renderPart(part, expanded);
      const token = `__natlang_neuralese_text_block_${String(part.id)}__`;
      replacements.set(token, String(block.text ?? ''));
      return token;
    }));
    const source = fragments.join('');
    if (!replacements.size) return source;
    let parsed: unknown;
    try { parsed = JSON.parse(source); }
    catch { throw new Error('text Neuralese emulation could not preserve the JSON shape of a tool-call argument'); }
    const replace = (item: unknown): unknown => {
      if (typeof item === 'string') {
        let result = item;
        for (const [token, block] of replacements) result = result.split(token).join(block);
        return result;
      }
      if (Array.isArray(item)) return item.map(replace);
      if (item && typeof item === 'object') return Object.fromEntries(Object.entries(item).map(([key, child]) => [key, replace(child)]));
      return item;
    };
    return JSON.stringify(replace(parsed));
  };
  const visit = async (value: unknown, expanded: ExpandedInputs): Promise<unknown> => {
    if (Array.isArray(value)) {
      if (isParts(value))
        return (await Promise.all(value.map(part => renderPart(part as Record<string, unknown>, expanded))))
          .map(part => String(part.text ?? '')).join('');
      return Promise.all(value.map(item => visit(item, expanded)));
    }
    if (!value || typeof value !== 'object') return value;
    const record = value as Record<string, unknown>;
    if (record.type === 'neuralese') return renderPart(record, expanded);
    return Object.fromEntries(await Promise.all(Object.entries(record).map(async ([key, item]) => {
      if (key !== 'tool_calls' || !Array.isArray(item)) return [key, await visit(item, expanded)];
      const calls = await Promise.all(item.map(async call => {
        if (!call || typeof call !== 'object') return call;
        const callRecord = call as Record<string, unknown>, fn = callRecord.function;
        if (!fn || typeof fn !== 'object') return visit(callRecord, expanded);
        const fnRecord = fn as Record<string, unknown>;
        return { ...callRecord, function: { ...fnRecord,
          ...(Object.hasOwn(fnRecord, 'arguments') ? { arguments: await renderArgument(fnRecord.arguments, expanded) } : {}) } };
      }));
      return [key, calls];
    })));
  };
  return { literalBodies, visit };
}

/** Creates a case-local, explicitly non-learned text transport. Literal marker bodies come only from provider output. */
export function createTextNeuraleseEmulation(options: { store?: NeuraleseStore; width?: number } = {}): TextNeuraleseEmulation {
  const store = options.store ?? new MemoryNeuraleseStore();
  const width = options.width ?? TEXT_NEURALESE_WIDTH;
  if (!Number.isSafeInteger(width) || width < 1) throw new RangeError('text Neuralese stand-in width must be positive');
  const underlying = new StandInNeuralesePort(store, hashingEmbedder(width), width, TEXT_NEURALESE_DIALECT);
  const { literalBodies, visit } = markerRenderer(store);
  const port: NeuralesePort = {
    dialect: TEXT_NEURALESE_DIALECT,
    async write(text, options = {}): Promise<NeuraleseBlockMeta> {
      const resultType = options.type ?? (typeof options.producer?.result_type === 'string' ? options.producer.result_type : undefined);
      const markerContext = options.producer?.marker_context;
      const softResult = !!resultType?.startsWith('Neuralese<') && markerContext === 'return-result';
      const typedEvalSource = markerContext === 'eval-code';
      if (!softResult && !typedEvalSource)
        throw new Error(`text Neuralese marker is allowed only in a declared Neuralese<T> result or typed eval source ` +
          `(context=${String(markerContext)}, result_type=${String(resultType)})`);
      const textSha = sha256(text);
      const block = await underlying.write(text, { ...options, ...(softResult ? { type: resultType } : {}),
        producer: { kind: 'text-marker-emulation', emulation_version: TEXT_NEURALESE_EMULATION_VERSION,
          learned_vectors: false, text_body_sha256: textSha, ...(options.producer ?? {}) } });
      literalBodies.set(block.id, { ...(softResult ? { type: resultType } : {}), text, text_sha256: textSha });
      return block;
    }
  };
  const runtime: NeuraleseRuntimeOptions = { store, port };
  const wrap = <T extends (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn>>(send: T): T => {
    const adapted = (async (request: ModelTurnRequest, signal?: AbortSignal): Promise<ModelTurn> => {
      if (request.adapters?.length || request.guidance || request.template)
        throw new Error('text Neuralese emulation supports literal text blocks only, not adapters, guidance, or template readout');
      // The collector belongs to this invocation. Concurrent provider requests must not share read provenance.
      const expanded = new Map<string, { id: string; type: string | null; body: string; body_sha256: string; learned_vectors: false }>();
      const rendered = await visit(request.messages, expanded);
      if (!Array.isArray(rendered)) throw new TypeError('rendered Neuralese provider messages must remain an array');
      const renderedMessages = rendered;
      const renderedRequest = { ...request, messages: renderedMessages };
      const response = await send(renderedRequest, signal);
      const blocksRead = [...expanded.values()];
      const outputs = markerBodies({ text: response.text, calls: response.calls }).map(text => ({
        body_sha256: sha256(text), body_chars: text.length, learned_vectors: false }));
      return { ...response, transport_provenance: {
        version: TEXT_NEURALESE_EMULATION_VERSION,
        vector_semantics: 'deterministic hashingEmbedder stand-in; non-learned; no qualification/admission certificate',
        learned_vectors: false, qualification_certificate: false, training_admission: false,
        raw_request_sha256: sha256(canonical(request)),
        rendered_request_sha256: sha256(canonical(renderedRequest)),
        rendered_messages: renderedMessages,
        expanded_input_blocks: blocksRead,
        marker_outputs: outputs,
      } };
    }) as T;
    Object.defineProperty(adapted, 'neuralese', { value: true, enumerable: true });
    return adapted;
  };
  return { store, port, runtime, wrap };
}
