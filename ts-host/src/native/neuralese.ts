/**
 * Neuralese literals and references in the runtime (S0 §3, S4 §2.3, §3).
 *
 * A soft value has exactly two renderings:
 *
 * - **Reference form**, everywhere outside the model: `{ $neuralese: { type, id } }`, an ordinary data value naming a
 *   stored block.
 * - **Model form**, at the token level: `<|neuralese|>` vectors `<|/neuralese|>`. Transports carry it as content parts
 *   `{ type: "neuralese", id }` in message content and in tool-call arguments.
 *
 * Between the two, the runtime's own conversation text marks a block with a private-use sentinel around its ID. The
 * agent's text handling (cut-offs, compaction, transcript) needs no special case for it; transports turn sentinels into
 * parts, and eval compilation turns them into `__neuralese("id")`, which the checker types from context.
 */
import { constantBlock, type NeuraleseBlockMeta, type NeuralesePort, type NeuraleseStore } from './neuralese-store.js';
import type { DialectBinding, Type, TypeEnv } from './types.js';

export const NEURALESE_OPEN = '<|neuralese|>';
export const NEURALESE_CLOSE = '<|/neuralese|>';

/** Reference form of a soft value. */
export type NeuraleseRef = { readonly $neuralese: { readonly type: string; readonly id: string } };
export type NeuraleseTextPart = { type: 'text'; text: string };
export type NeuraleseContentPart = { type: 'neuralese'; id: string };
export type NativeStateContentPart = { type: 'native_state'; id: string };
export type ContentPart = NeuraleseTextPart | NeuraleseContentPart | NativeStateContentPart;
export type RuntimeContentPart = ContentPart;

const ID = /^nz1_[a-z2-7]{20,}$/;
const SENTINEL_OPEN = '', SENTINEL_CLOSE = '';
const SENTINEL = /(nz1_[a-z2-7]+)/g;
const NATIVE_SENTINEL_OPEN = '', NATIVE_SENTINEL_CLOSE = '';
const NATIVE_STATE_ID = /^nzs1_[a-z2-7]{52}$/;
const NATIVE_TENSOR_ID = /^nnt1_[a-z2-7]{52}$/;
export function isNativeTensorArtifactId(value: unknown): value is string {
  return typeof value === 'string' && NATIVE_TENSOR_ID.test(value);
}
const NATIVE_SENTINEL = /(nzs1_[a-z2-7]{52})/g;

export function isNativeStateId(value: unknown): value is string {
  return typeof value === 'string' && NATIVE_STATE_ID.test(value);
}

export function nativeStateSentinel(id: string): string {
  if (!isNativeStateId(id)) throw new TypeError(`not a native-state block ID: ${JSON.stringify(id)}`);
  return `${NATIVE_SENTINEL_OPEN}${id}${NATIVE_SENTINEL_CLOSE}`;
}

export function nativeStateSentinelIds(text: string): string[] {
  return [...text.matchAll(NATIVE_SENTINEL)].map(match => match[1]!);
}

export function runtimePartsToText(parts: readonly RuntimeContentPart[]): string {
  return parts.map(part => part.type === 'text' ? part.text :
    part.type === 'neuralese' ? neuraleseSentinel(part.id) : nativeStateSentinel(part.id)).join('');
}

export function runtimeTextToParts(text: string): RuntimeContentPart[] {
  const parts: RuntimeContentPart[] = [];
  const markers = [...text.matchAll(/(nz1_[a-z2-7]+)|(nzs1_[a-z2-7]{52})/g)];
  let last = 0;
  for (const match of markers) {
    if (match.index! > last) parts.push({ type: 'text', text: text.slice(last, match.index) });
    parts.push(match[1] ? { type: 'neuralese', id: match[1] } : { type: 'native_state', id: match[2]! });
    last = match.index! + match[0].length;
  }
  if (last < text.length || !parts.length) parts.push({ type: 'text', text: text.slice(last) });
  return parts;
}

export function nativeStateTextToParts(text: string): RuntimeContentPart[] {
  const parts: RuntimeContentPart[] = [];
  let last = 0;
  for (const match of text.matchAll(NATIVE_SENTINEL)) {
    if (match.index! > last) parts.push({ type: 'text', text: text.slice(last, match.index) });
    parts.push({ type: 'native_state', id: match[1]! });
    last = match.index! + match[0].length;
  }
  if (last < text.length || !parts.length) parts.push({ type: 'text', text: text.slice(last) });
  return parts;
}

export function isNeuraleseId(value: unknown): value is string { return typeof value === 'string' && (ID.test(value) || NATIVE_STATE_ID.test(value) || NATIVE_TENSOR_ID.test(value)); }

export function neuraleseRef(type: string, id: string): NeuraleseRef {
  if (!isNeuraleseId(id)) throw new TypeError(`not a Neuralese block ID: ${JSON.stringify(id)}`);
  return Object.freeze({ $neuralese: Object.freeze({ type, id }) });
}

export function isNeuraleseRef(value: unknown): value is NeuraleseRef {
  if (value === null || typeof value !== 'object' || Array.isArray(value)) return false;
  const keys = Object.keys(value);
  if (keys.length !== 1 || keys[0] !== '$neuralese') return false;
  const body = (value as { $neuralese?: unknown }).$neuralese;
  return !!body && typeof body === 'object' && typeof (body as { type?: unknown }).type === 'string' &&
    isNeuraleseId((body as { id?: unknown }).id) && Object.keys(body).every(key => key === 'type' || key === 'id');
}

/** The conversation-text marker for a block. */
export function neuraleseSentinel(id: string): string {
  if (isNativeTensorArtifactId(id)) throw new Error('neuralese-native-tensor-as-state: bind native adapters with withAdapters; named tensor artifacts cannot be consumed as communication state');
  if (!isNeuraleseId(id)) throw new TypeError(`not a Neuralese block ID: ${JSON.stringify(id)}`);
  return isNativeStateId(id) ? nativeStateSentinel(id) : `${SENTINEL_OPEN}${id}${SENTINEL_CLOSE}`;
}

/** The block a text consists of exactly, if it is one sentinel. */
export function soleSentinel(text: string): string | undefined {
  const match = /^(?:(nz1_[a-z2-7]+)|(nzs1_[a-z2-7]{52}))$/.exec(text);
  return match?.[1] ?? match?.[2];
}

export const hasNeuraleseSentinel = (text: string): boolean => text.includes(SENTINEL_OPEN) || text.includes(NATIVE_SENTINEL_OPEN);

/** Block IDs a text refers to. */
export function sentinelIds(text: string): string[] { return [...text.matchAll(/(nz1_[a-z2-7]+)|(nzs1_[a-z2-7]{52})/g)].map(match => match[1] ?? match[2]!); }

/** Conversation text to content parts: text runs and Neuralese blocks. */
export function textToParts(text: string): ContentPart[] { return runtimeTextToParts(text); }

export const isContentParts = (value: unknown): value is ContentPart[] => Array.isArray(value) && value.length > 0 &&
  value.every(part => part && typeof part === 'object' && ((part.type === 'text' && typeof part.text === 'string') ||
    (part.type === 'neuralese' && isNeuraleseId(part.id)) ||
    (part.type === 'native_state' && isNativeStateId(part.id))));

/** Content parts to conversation text. */
export function partsToText(parts: readonly ContentPart[]): string {
  return runtimePartsToText(parts);
}

/**
 * Eval source with its literals as calls the checker types from their context (see compiler/neuralese.ts). A template
 * that is exactly one block is a soft function body, ``nl.with({ … })`${__neuralese.body("nz1_…")}` ``.
 */
export function sourceWithLiteralCalls(code: string): string {
  const pattern = /(nz1_[a-z2-7]+)|(nzs1_[a-z2-7]{52})/g;
  const body = new RegExp('`(?:' + pattern.source + ')`', 'g');
  return code.replace(body, (_, legacy: string, native: string) =>
    `\`\${__neuralese.body(${JSON.stringify(legacy ?? native)})}\``)
    .replace(pattern, (_, legacy: string, native: string) => `__neuralese(${JSON.stringify(legacy ?? native)})`);
}

/**
 * Store each block a model wrote as marker text (`<|neuralese|>…<|/neuralese|>`) through `port`, and return the text
 * with sentinels in their place. Backends that write real blocks return parts instead; this is the stand-in path, where
 * the text inside the markers is what the stand-in port embeds. An unclosed literal is written as truncated.
 */
export async function writeLiterals(text: string, port: NeuralesePort | undefined,
  producer: Record<string, unknown> = {}): Promise<{ text: string; parts: ContentPart[]; blocks: NeuraleseBlockMeta[] }> {
  if (!text.includes(NEURALESE_OPEN)) return { text, parts: [{ type: 'text', text }], blocks: [] };
  if (!port) throw new NeuraleseUnsupportedError('the model wrote a Neuralese literal, but this runtime has no Neuralese port');
  let out = '', rest = text;
  const blocks: NeuraleseBlockMeta[] = [];
  const parts: ContentPart[] = [];
  while (true) {
    const open = rest.indexOf(NEURALESE_OPEN);
    if (open < 0) { out += rest; if (rest) parts.push({ type: 'text', text: rest }); break; }
    const prefix = rest.slice(0, open);
    out += prefix; if (prefix) parts.push({ type: 'text', text: prefix });
    const body = rest.slice(open + NEURALESE_OPEN.length);
    const close = body.indexOf(NEURALESE_CLOSE);
    const inner = close < 0 ? body : body.slice(0, close);
    const block = await port.write(inner, { producer, truncated: close < 0 });
    blocks.push(block);
    out += neuraleseSentinel(block.id);
    parts.push({ type: isNativeStateId(block.id) ? 'native_state' : 'neuralese', id: block.id });
    if (close < 0) break;
    rest = body.slice(close + NEURALESE_CLOSE.length);
  }
  return { text: out, parts, blocks };
}

/** A backend or runtime without Neuralese support was asked to carry a soft value. No text fallback exists. */
export class NeuraleseUnsupportedError extends Error {
  readonly code = 'neuralese-unsupported-backend';
  constructor(detail: string) { super(`neuralese-unsupported-backend: ${detail}`); this.name = 'NeuraleseUnsupportedError'; }
}

/** A soft value met a reader or slot of another dialect. Values are never reused across dialects (NEURALESE_DIALECTS.md). */
export class NeuraleseDialectError extends Error {
  readonly code = 'neuralese-dialect-mismatch';
  constructor(detail: string) { super(`neuralese-dialect-mismatch: ${detail}`); this.name = 'NeuraleseDialectError'; }
}

/** A native reader's explicit, checkpoint-bound adaptation capabilities from /info. */
export type NativeStateConsumer = { architecture: string; checkpoint_sha256: string;
  tokenizer_fingerprint: string; embedding_weight_sha256: string; consumption_modes: string[]; semantic_adapter_sha256?: string };

export function nativeStateConsumerFromInfo(info: Record<string, unknown>): NativeStateConsumer | undefined {
  const native = info.native_state as Record<string, unknown> | undefined;
  if (native?.enabled !== true) return undefined;
  if (typeof native.architecture !== 'string' || !native.architecture ||
      ['checkpoint_sha256', 'tokenizer_fingerprint', 'embedding_weight_sha256'].some(key =>
        typeof native[key] !== 'string' || !/^[0-9a-f]{64}$/.test(native[key] as string)) ||
      native.consumption_modes !== undefined && (!Array.isArray(native.consumption_modes) || native.consumption_modes.some(mode => typeof mode !== 'string')))
    throw new NeuraleseUnsupportedError('native consumer must declare its model identity and actual state consumption modes');
  if (native.semantic_adapter_sha256 !== undefined && (typeof native.semantic_adapter_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(native.semantic_adapter_sha256)))
    throw new NeuraleseUnsupportedError('semantic consumer must bind its exact trained adapter SHA-256');
  if (!Array.isArray(native.consumption_modes)) return undefined;
  return { architecture: native.architecture, checkpoint_sha256: native.checkpoint_sha256 as string,
    tokenizer_fingerprint: native.tokenizer_fingerprint as string,
    embedding_weight_sha256: native.embedding_weight_sha256 as string,
    consumption_modes: [...native.consumption_modes] as string[],
    ...(native.semantic_adapter_sha256 ? { semantic_adapter_sha256: native.semantic_adapter_sha256 as string } : {}) };
}

/** Runtime configuration for soft values. */
export type NeuraleseRuntimeOptions = {
  store: NeuraleseStore;
  nativeStateConsumer?: NativeStateConsumer;
  /** Writes blocks for literals a model returns as marker text (the stand-in port, or a writer in-process). */
  port?: NeuralesePort;
  /**
   * The dialect the runtime's model reads and writes, which `DefaultDialect` stands for. Defaults to the port's; set it
   * when a server writes the blocks (no port in-process). Hosts check it at startup against the server's
   * `/v1/neuralese/info` (`NatlangRuntime.readerDialect()`). It must equal the port's dialect when both are set.
   */
  dialect?: string;
  /**
   * Soft forms of the runtime's prompt pieces (system-prompts.ts), used in place of their text under a Neuralese
   * driver. Their blocks must be in `store` or on the server.
   */
  systemPrompts?: ReadonlyMap<string, { readonly text: string; readonly value: NeuraleseRef }>;
  /**
   * The Neuralese instance of the builtin `view(value, instructions?)` (neuralese/view.ts `serverViewer`): the opening
   * listing shows a view of each argument it would cut off, written for the receiving call's instructions
   * (prompt.ts `listingViewInstructions`), while the variable keeps the whole value.
   */
  view?: (site: { value: string; instructions?: string }) => Promise<NeuraleseRef | undefined>;
};

/**
 * The dialect a runtime with these options reads (DECISIONS.md 2026-10-09: dialects are static facts): the declared
 * dialect, else the port's; null for a text-only runtime.
 */
export function readerDialect(options: NeuraleseRuntimeOptions | undefined): string | null {
  const declared = options?.dialect, port = options?.port?.dialect;
  if (declared !== undefined && port !== undefined && declared !== port)
    throw new NeuraleseDialectError(`the runtime declares dialect ${JSON.stringify(declared)}, but its write port writes ` +
      `${JSON.stringify(port)}; configure a port of the declared dialect, or declare ${JSON.stringify(port)}`);
  return declared ?? port ?? null;
}

/** The dialect binding of type checks under these options (TypeEnv.dialects); undefined without Neuralese. */
export function dialectBinding(options: NeuraleseRuntimeOptions | undefined): DialectBinding | undefined {
  if (!options) return undefined;
  const dialect = readerDialect(options), store = options.store;
  return { dialect, blockDialect: id => store.peek?.(id)?.dialect ?? constantBlock(id)?.meta.dialect,
    nativeDefaultCompatible: (id, source, target) => {
      const consumer = options.nativeStateConsumer;
      if (!isNativeStateId(id) || !consumer || !/^(native|semantic)-state\//.test(source)) return false;
      const semantic = consumer.semantic_adapter_sha256 ? `semantic-state/${consumer.semantic_adapter_sha256}` : undefined;
      if (target !== `native-state/${consumer.architecture}` && target !== semantic) return false;
      if (consumer.consumption_modes.includes('best-effort-text-retokenization')) return true;
      return !!semantic && consumer.consumption_modes.includes('learned-semantic-state-deltas') &&
        (!source.startsWith('semantic-state/') || source === semantic);
    } };
}

/** A model driver that carries content parts. Transports mark themselves by setting `neuralese: true`. */
export const supportsNeuralese = (driver: unknown): boolean =>
  typeof driver === 'function' && (driver as { neuralese?: unknown }).neuralese === true;

/**
 * Request messages with sentinels turned into content parts: message `content`, and each tool call's `arguments`,
 * become part arrays where they hold blocks. Messages without blocks are unchanged.
 */
export function encodeMessages(messages: readonly unknown[]): { messages: unknown[]; blocks: number } {
  let blocks = 0;
  const encode = (text: string): string | ContentPart[] => {
    if (!hasNeuraleseSentinel(text)) return text;
    const parts = textToParts(text);
    blocks += parts.filter(part => part.type !== 'text').length;
    return parts;
  };
  const out = messages.map(message => {
    if (!message || typeof message !== 'object') return message;
    const record = message as Record<string, unknown>;
    let changed: Record<string, unknown> | undefined;
    if (typeof record.content === 'string' && hasNeuraleseSentinel(record.content))
      changed = { ...record, content: encode(record.content) };
    if (Array.isArray(record.tool_calls) && record.tool_calls.some(call =>
        typeof (call as { function?: { arguments?: unknown } })?.function?.arguments === 'string' &&
        hasNeuraleseSentinel((call as { function: { arguments: string } }).function.arguments)))
      changed = { ...(changed ?? record), tool_calls: record.tool_calls.map(call => {
        const fn = (call as { function?: { arguments?: unknown } }).function;
        return typeof fn?.arguments === 'string' && hasNeuraleseSentinel(fn.arguments) ?
          { ...(call as object), function: { ...fn, arguments: encode(fn.arguments) } } : call;
      }) };
    return changed ?? message;
  });
  return { messages: out, blocks };
}

/**
 * A model turn with its Neuralese content in conversation-text form: part arrays (from a Neuralese backend) and
 * marker text (from the stand-in path) in the reply text and in every string of the call arguments become sentinels.
 */
type TurnValueExpectation = { type: Type; env: TypeEnv };

function expectedValueTypes(expected: TurnValueExpectation | undefined): Type[] {
  if (!expected) return [];
  const type = expected.env.resolve(expected.type);
  if (type.kind === 'refined' || type.kind === 'untrusted') return expectedValueTypes({ ...expected, type: type.base });
  if (type.kind === 'union') return type.members.flatMap(member => expectedValueTypes({ ...expected, type: member }));
  return [type];
}

function expectedChild(expected: TurnValueExpectation | undefined, key: string, container: unknown): TurnValueExpectation | undefined {
  const candidates = expectedValueTypes(expected).filter(type => {
    if (type.kind !== 'record' || !container || typeof container !== 'object') return true;
    return type.fields.every(field => {
      if (!Object.hasOwn(container, field.name)) return true;
      const discriminants = expectedValueTypes({ env: expected!.env, type: field.type });
      return !discriminants.length || !discriminants.every(member => member.kind === 'lit') ||
        discriminants.some(member => member.kind === 'lit' &&
          member.value === (container as Record<string, unknown>)[field.name]);
    });
  });
  const members = candidates.flatMap(type => type.kind === 'record' ?
    type.fields.filter(field => field.name === key).map(field => field.type) :
    type.kind === 'list' || type.kind === 'dict' ? [type.element] : []);
  return members.length && expected ? { env: expected.env,
    type: members.length === 1 ? members[0]! : { kind: 'union', members } } : undefined;
}

export async function decodeTurnValue(value: unknown, port: NeuralesePort | undefined,
  producer: Record<string, unknown>, written?: NeuraleseBlockMeta[], expected?: TurnValueExpectation): Promise<unknown> {
  const valueBoundary = producer.marker_context === 'return-result' || producer.marker_context === 'other-tool-argument';
  if (typeof value === 'string') {
    const types = expectedValueTypes(expected);
    const explicitlyNeuralese = types.some(type => type.kind === 'neuralese') &&
      !types.some(type => type.kind === 'prim' && ['string', 'Blob', 'unknown'].includes(type.name) ||
        type.kind === 'lit' && typeof type.value === 'string');
    if (!value.includes(NEURALESE_OPEN) || valueBoundary && !explicitlyNeuralese) return value;
    const { text, parts, blocks } = await writeLiterals(value, port, producer);
    written?.push(...blocks);
    return valueBoundary ? parts : text;
  }
  if (isContentParts(value)) {
    if (!valueBoundary) return decodeTurnValue(partsToText(value), port, producer, written);
    // Text runs remain literal; only actual block parts confer transport identity.
    return value.some(part => part.type !== 'text') ? value.map(part => ({ ...part })) :
      value.map(part => (part as NeuraleseTextPart).text).join('');
  }
  if (Array.isArray(value)) return Promise.all(value.map((item, index) =>
    decodeTurnValue(item, port, producer, written, expectedChild(expected, String(index), value))));
  if (value && typeof value === 'object') {
    const entries = await Promise.all(Object.entries(value).map(async ([key, item]) =>
      [key, await decodeTurnValue(item, port, producer, written, expectedChild(expected, key, value))]));
    return Object.fromEntries(entries);
  }
  return value;
}
