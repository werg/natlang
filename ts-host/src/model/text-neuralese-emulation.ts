/**
 * Explicit text-channel transport for teacher runs whose provider cannot carry Neuralese content parts.
 * This is protocol emulation only: blocks contain deterministic hash stand-in vectors, never learned vectors.
 */
import { createHash } from 'node:crypto';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { MemoryNeuraleseStore, StandInNeuralesePort, hashingEmbedder } from '../native/neuralese-store.js';
import type { NeuraleseBlockMeta, NeuralesePort, NeuraleseStore } from '../native/neuralese-store.js';
import type { NeuraleseRuntimeOptions } from '../native/neuralese.js';
import { COMBINATORS, type StandardLibrary } from '../neuralese/combinators.js';
import { hexDigest } from '../native/hash.js';

export const TEXT_NEURALESE_EMULATION_VERSION = 'text-marker-standin/2';
export const TEXT_NEURALESE_PROMPT_REVISION = 'text-marker-guidance/7';
export const TEXT_NEURALESE_DIALECT = 'nd:text-teacher-emulation/1';
export const TEXT_NEURALESE_WIDTH = 32;

export const TEXT_NEURALESE_EMULATION_PROMPT = `\n\nDeclared Neuralese text-channel emulation for this run:\n` +
  `Use <|neuralese|>BODY<|/neuralese|> in return_result only when the current call declares ` +
  `a Neuralese<T> result. This marker is transport syntax, not a JavaScript string. In eval code, use it only as ` +
  `an unquoted value in an explicitly typed Neuralese position, for example ` +
  `const note: Neuralese<string> = <|neuralese|>the note text<|/neuralese|>. A quoted occurrence is ordinary string content, ` +
  `not marker transport syntax; at a Neuralese<string> result boundary, the configured writer stores it as literal body text. ` +
  `Everything between the markers is literal body text: \${name} is stored exactly as written, never evaluated or interpolated. ` +
  `To pass a computed soft value, return the variable itself from eval when it has the declared result type ` +
  `(for example, return notes;), or pass it directly as a typed child argument. Never put a variable name between marker delimiters. ` +
  `For ordinary string results, return ordinary strings. When a prior Neuralese value is bound as a variable in eval, ` +
  `use String(notes), \`\u0024{notes}\`, string concatenation, or JSON.stringify(notes) when ordinary text is needed; ` +
  `the declared text read conversion handles these string positions automatically in this call. If a child argument expects ` +
  `Neuralese<T>, pass the original typed variable unchanged to preserve its soft type. Return that same typed variable from eval ` +
  `when it matches the declared result type. A displayed [[Neuralese text block ...]] label is a human-readable preview. ` +
  `A complete unchanged label may be resolved as a Neuralese result only when its ID, type, and exact body digest match ` +
  `a typed value visible in this call; partial, altered, or nonvisible labels remain strings and fail type checking. Prefer ` +
  `the bound variable in eval; readText and read_code do not unpack a displayed label. ` +
  `When the declared final result is exactly Neuralese<string>, return the computed plain text as the result; the configured ` +
  `Neuralese writer materializes it as a typed block, whether you return it from eval, use eval({code, finish:true}), ` +
  `or stage it with return_result(text) inside eval. A direct return_result tool call may also carry a plain string value. ` +
  `A literal <|neuralese|>BODY<|/neuralese|> remains an option when the body itself is literal text; markers are not needed ` +
  `to wrap a computed variable. Do not put a variable name between marker delimiters ` +
  `or expect \${...} inside a marker body to interpolate. No general text conversion applies to other Neuralese<T> types: ` +
  `pass their existing typed value or use an exact block marker in an explicitly typed position. ` +
  `For an ordinary string result, any characters that look like Neuralese marker delimiters are literal string content; ` +
  `return them exactly without wrapping or interpreting them as a block. ` +
  `eval({code, finish:true}) completes its fresh typed expression. In JavaScript, a quoted marker is an ordinary string, ` +
  `not a soft block; when the declared result is Neuralese<string>, that string is written as literal body text. ` +
  `Neuralese is a built-in type, not a callable function: do not redefine it or use read_code("Neuralese") to construct the answer.\n`;

export type TextNeuraleseEmulation = {
  store: NeuraleseStore;
  port: NeuralesePort;
  runtime: NeuraleseRuntimeOptions;
  /** Explicit StandardLibrary whose declared read export has an authenticated text implementation. */
  standardLibrary?: StandardLibrary;
  /** Resolves to the explicit non-learned library used by the text backend. */
  standardLibraryReady: Promise<StandardLibrary>;
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

function markerRenderer(store: NeuraleseStore, getStandardLibrary: () => StandardLibrary | undefined) {
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
    const standardLibrary = getStandardLibrary();
    const readSource = standardLibrary?.textReadSource?.bodyId === part.id ? standardLibrary.textReadSource : undefined;
    if (!literal && !readSource) throw new Error(`text Neuralese emulation has no authenticated literal/read source for block ${part.id}`);
    if (literal && block.meta.producer?.text_body_sha256 !== literal.text_sha256)
      throw new Error(`text Neuralese literal digest does not match stored block ${part.id}`);
    if (readSource && (readSource.sourceSha256 !== hexDigest(readSource.source) || readSource.learnedVectors !== false ||
        readSource.type !== 'Neuralese<(v: Neuralese<unknown>) => unknown>' ||
        readSource.bodyId !== standardLibrary?.bodies.read))
      throw new Error(`text read source does not match the configured library's declared read body ${part.id}`);
    const type = readSource?.type ?? literal?.type;
    const bodyText = readSource?.source ?? literal?.text;
    const bodySha = readSource?.sourceSha256 ?? literal!.text_sha256;
    expanded.set(part.id, { id: part.id, type: type ?? null, body: bodyText!,
      body_sha256: bodySha!, learned_vectors: false });
    const label = `[[Neuralese text block id=${part.id}${type ? ` type=${type}` : ''}; exact JSON string body=${JSON.stringify(bodyText)}]]`;
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
export function createTextNeuraleseEmulation(options: { store?: NeuraleseStore; width?: number;
  standardLibrary?: StandardLibrary } = {}): TextNeuraleseEmulation {
  const store = options.store ?? new MemoryNeuraleseStore();
  const width = options.width ?? TEXT_NEURALESE_WIDTH;
  if (!Number.isSafeInteger(width) || width < 1) throw new RangeError('text Neuralese stand-in width must be positive');
  const underlying = new StandInNeuralesePort(store, hashingEmbedder(width), width, TEXT_NEURALESE_DIALECT);
  let standardLibrary = options.standardLibrary;
  const standardLibraryReady = options.standardLibrary ? Promise.resolve(options.standardLibrary) : (async () => {
    const bodies = {} as Record<keyof typeof COMBINATORS, string>;
    for (const [name, entry] of Object.entries(COMBINATORS) as [keyof typeof COMBINATORS, (typeof COMBINATORS)[keyof typeof COMBINATORS]][]) {
      const sourceSha256 = hexDigest(entry.text);
      const meta = await underlying.write(entry.text, { type: `Neuralese<${entry.type}>`, producer: {
        kind: 'text-provider-standard-library', source_sha256: sourceSha256, learned_vectors: false,
      } });
      bodies[name] = meta.id;
    }
    return {
      dialect: underlying.dialect, width, bodies,
      textReadSource: { schema: 'natlang.text-read-source/1', export: 'read', bodyId: bodies.read,
        type: `Neuralese<${COMBINATORS.read.type}>`, source: COMBINATORS.read.text,
        sourceSha256: hexDigest(COMBINATORS.read.text), learnedVectors: false },
    } satisfies StandardLibrary;
  })().then(library => (standardLibrary = library));
  const { literalBodies, visit } = markerRenderer(store, () => standardLibrary);
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
      if (request.adapters?.length || request.guidance)
        throw new Error('text Neuralese emulation supports literal text blocks only, not adapters or guidance');
      const template = request.template;
      const textReadout = !!template;
      const configuredRead = standardLibrary?.textReadSource;
      if (template && (template.call !== 'return_result' || template.value !== 'decode' || template.length !== undefined ||
          template.passes !== undefined))
        throw new Error('text Neuralese emulation only supports the declared read(decode) template contract');
      if (template) {
        if (!configuredRead)
          throw new Error('text Neuralese template readout needs an explicitly declared text read source; vector-only libraries are unsupported');
        if (configuredRead.bodyId !== standardLibrary?.bodies.read)
          throw new Error('text Neuralese read source does not match the configured library read body');
      }
      // The collector belongs to this invocation. Concurrent provider requests must not share read provenance.
      const expanded = new Map<string, { id: string; type: string | null; body: string; body_sha256: string; learned_vectors: false }>();
      const rendered = await visit(request.messages, expanded);
      if (!Array.isArray(rendered)) throw new TypeError('rendered Neuralese provider messages must remain an array');
      const returnTool = request.tools.find(tool => (tool as { function?: { name?: unknown } }).function?.name === 'return_result');
      if (template && !returnTool) throw new Error('text Neuralese readout request has no return_result tool contract');
      const readInstruction = template ? `This is a typed Neuralese read call. Interpret the exact visible input block using the displayed declared result type and read instruction. ` +
        `Return one call to return_result with arguments {"status":"success","value":<the decoded value>}. ` +
        `The value must match the declared result type exactly. Do not use any other tool and do not return an explanation.` : undefined;
      const renderedMessages = textReadout ? [{ role: 'system', content: readInstruction! }, ...rendered] : rendered;
      const { template: _template, ...withoutTemplate } = request;
      const renderedRequest = { ...(template ? withoutTemplate : request),
        ...(template ? { tools: [returnTool!], tool_choice: 'required' as const } : {}), messages: renderedMessages };
      const response = await send(renderedRequest, signal);
      if (template && (!Array.isArray(response.calls) || response.calls.length !== 1 || response.calls[0]?.[0] !== 'return_result' ||
          response.calls[0]?.[1]?.status !== 'success' || !Object.hasOwn(response.calls[0][1], 'value')))
        throw new Error('text Neuralese readout provider did not return exactly one successful typed return_result value');
      if (template?.value_type === 'string' && typeof response.calls?.[0]?.[1]?.value !== 'string')
        throw new Error('text Neuralese readout provider returned a value that does not match the declared string result type');
      const blocksRead = [...expanded.values()];
      const outputs = markerBodies({ text: response.text, calls: response.calls }).map(text => ({
        body_sha256: sha256(text), body_chars: text.length, learned_vectors: false }));
      return { ...response, transport_provenance: {
        version: TEXT_NEURALESE_EMULATION_VERSION,
        prompt_revision: TEXT_NEURALESE_PROMPT_REVISION,
        vector_semantics: 'deterministic hashingEmbedder stand-in; non-learned; no qualification/admission certificate',
        learned_vectors: false, qualification_certificate: false, training_admission: false,
        raw_request_sha256: sha256(canonical(request)),
        rendered_request_sha256: sha256(canonical(renderedRequest)),
        rendered_messages: renderedMessages,
        expanded_input_blocks: blocksRead,
        marker_outputs: outputs,
        ...(template ? { text_template_readout: { schema: 'natlang.text-template-readout/1', call: template.call,
          value: template.value, value_type: template.value_type ?? null, read_body_id: configuredRead!.bodyId,
          read_source_sha256: configuredRead!.sourceSha256, learned_vectors: false,
          qualification_certificate: false, training_admission: false } } : {}),
      } };
    }) as T;
    Object.defineProperty(adapted, 'neuralese', { value: true, enumerable: true });
    return adapted;
  };
  return { store, port, runtime, get standardLibrary() { return standardLibrary; }, standardLibraryReady, wrap };
}
