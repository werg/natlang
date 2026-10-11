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
import { constantBlock } from './neuralese-store.js';
export const NEURALESE_OPEN = '<|neuralese|>';
export const NEURALESE_CLOSE = '<|/neuralese|>';
const ID = /^nz1_[a-z2-7]{20,}$/;
const SENTINEL_OPEN = '', SENTINEL_CLOSE = '';
const SENTINEL = /(nz1_[a-z2-7]+)/g;
export function isNeuraleseId(value) { return typeof value === 'string' && ID.test(value); }
export function neuraleseRef(type, id) {
    if (!isNeuraleseId(id))
        throw new TypeError(`not a Neuralese block ID: ${JSON.stringify(id)}`);
    return Object.freeze({ $neuralese: Object.freeze({ type, id }) });
}
export function isNeuraleseRef(value) {
    if (value === null || typeof value !== 'object' || Array.isArray(value))
        return false;
    const keys = Object.keys(value);
    if (keys.length !== 1 || keys[0] !== '$neuralese')
        return false;
    const body = value.$neuralese;
    return !!body && typeof body === 'object' && typeof body.type === 'string' &&
        isNeuraleseId(body.id) && Object.keys(body).every(key => key === 'type' || key === 'id');
}
/** The conversation-text marker for a block. */
export function neuraleseSentinel(id) {
    if (!isNeuraleseId(id))
        throw new TypeError(`not a Neuralese block ID: ${JSON.stringify(id)}`);
    return `${SENTINEL_OPEN}${id}${SENTINEL_CLOSE}`;
}
/** The block a text consists of exactly, if it is one sentinel. */
export function soleSentinel(text) {
    const match = /^(nz1_[a-z2-7]+)$/.exec(text);
    return match?.[1];
}
export const hasNeuraleseSentinel = (text) => text.includes(SENTINEL_OPEN);
/** Block IDs a text refers to. */
export function sentinelIds(text) { return [...text.matchAll(SENTINEL)].map(match => match[1]); }
/** Conversation text to content parts: text runs and Neuralese blocks. */
export function textToParts(text) {
    const parts = [];
    let last = 0;
    for (const match of text.matchAll(SENTINEL)) {
        if (match.index > last)
            parts.push({ type: 'text', text: text.slice(last, match.index) });
        parts.push({ type: 'neuralese', id: match[1] });
        last = match.index + match[0].length;
    }
    if (last < text.length || !parts.length)
        parts.push({ type: 'text', text: text.slice(last) });
    return parts;
}
export const isContentParts = (value) => Array.isArray(value) && value.length > 0 &&
    value.every(part => part && typeof part === 'object' && ((part.type === 'text' && typeof part.text === 'string') ||
        (part.type === 'neuralese' && isNeuraleseId(part.id))));
/** Content parts to conversation text. */
export function partsToText(parts) {
    return parts.map(part => part.type === 'text' ? part.text : neuraleseSentinel(part.id)).join('');
}
/**
 * Eval source with its literals as calls the checker types from their context (see compiler/neuralese.ts). A template
 * that is exactly one block is a soft function body, ``nl.with({ … })`${__neuralese.body("nz1_…")}` ``.
 */
export function sourceWithLiteralCalls(code) {
    return code.replace(new RegExp('`' + SENTINEL.source + '`', 'g'), (_, id) => `\`\${__neuralese.body(${JSON.stringify(id)})}\``)
        .replace(SENTINEL, (_, id) => `__neuralese(${JSON.stringify(id)})`);
}
/**
 * Store each block a model wrote as marker text (`<|neuralese|>…<|/neuralese|>`) through `port`, and return the text
 * with sentinels in their place. Backends that write real blocks return parts instead; this is the stand-in path, where
 * the text inside the markers is what the stand-in port embeds. An unclosed literal is written as truncated.
 */
export async function writeLiterals(text, port, producer = {}) {
    if (!text.includes(NEURALESE_OPEN))
        return { text, blocks: [] };
    if (!port)
        throw new NeuraleseUnsupportedError('the model wrote a Neuralese literal, but this runtime has no Neuralese port');
    let out = '', rest = text;
    const blocks = [];
    while (true) {
        const open = rest.indexOf(NEURALESE_OPEN);
        if (open < 0) {
            out += rest;
            break;
        }
        out += rest.slice(0, open);
        const body = rest.slice(open + NEURALESE_OPEN.length);
        const close = body.indexOf(NEURALESE_CLOSE);
        const inner = close < 0 ? body : body.slice(0, close);
        const block = await port.write(inner, { producer, truncated: close < 0 });
        blocks.push(block);
        out += neuraleseSentinel(block.id);
        if (close < 0)
            break;
        rest = body.slice(close + NEURALESE_CLOSE.length);
    }
    return { text: out, blocks };
}
/** A backend or runtime without Neuralese support was asked to carry a soft value. No text fallback exists. */
export class NeuraleseUnsupportedError extends Error {
    code = 'neuralese-unsupported-backend';
    constructor(detail) { super(`neuralese-unsupported-backend: ${detail}`); this.name = 'NeuraleseUnsupportedError'; }
}
/** A soft value met a reader or slot of another dialect. Values are never reused across dialects (NEURALESE_DIALECTS.md). */
export class NeuraleseDialectError extends Error {
    code = 'neuralese-dialect-mismatch';
    constructor(detail) { super(`neuralese-dialect-mismatch: ${detail}`); this.name = 'NeuraleseDialectError'; }
}
/**
 * The dialect a runtime with these options reads (DECISIONS.md 2026-10-09: dialects are static facts): the declared
 * dialect, else the port's; null for a text-only runtime.
 */
export function readerDialect(options) {
    const declared = options?.dialect, port = options?.port?.dialect;
    if (declared !== undefined && port !== undefined && declared !== port)
        throw new NeuraleseDialectError(`the runtime declares dialect ${JSON.stringify(declared)}, but its write port writes ` +
            `${JSON.stringify(port)}; configure a port of the declared dialect, or declare ${JSON.stringify(port)}`);
    return declared ?? port ?? null;
}
/** The dialect binding of type checks under these options (TypeEnv.dialects); undefined without Neuralese. */
export function dialectBinding(options) {
    if (!options)
        return undefined;
    const dialect = readerDialect(options), store = options.store;
    return { dialect, blockDialect: id => store.peek?.(id)?.dialect ?? constantBlock(id)?.meta.dialect };
}
/** A model driver that carries content parts. Transports mark themselves by setting `neuralese: true`. */
export const supportsNeuralese = (driver) => typeof driver === 'function' && driver.neuralese === true;
/**
 * Request messages with sentinels turned into content parts: message `content`, and each tool call's `arguments`,
 * become part arrays where they hold blocks. Messages without blocks are unchanged.
 */
export function encodeMessages(messages) {
    let blocks = 0;
    const encode = (text) => {
        if (!hasNeuraleseSentinel(text))
            return text;
        const parts = textToParts(text);
        blocks += parts.filter(part => part.type === 'neuralese').length;
        return parts;
    };
    const out = messages.map(message => {
        if (!message || typeof message !== 'object')
            return message;
        const record = message;
        let changed;
        if (typeof record.content === 'string' && hasNeuraleseSentinel(record.content))
            changed = { ...record, content: encode(record.content) };
        if (Array.isArray(record.tool_calls) && record.tool_calls.some(call => typeof call?.function?.arguments === 'string' &&
            hasNeuraleseSentinel(call.function.arguments)))
            changed = { ...(changed ?? record), tool_calls: record.tool_calls.map(call => {
                    const fn = call.function;
                    return typeof fn?.arguments === 'string' && hasNeuraleseSentinel(fn.arguments) ?
                        { ...call, function: { ...fn, arguments: encode(fn.arguments) } } : call;
                }) };
        return changed ?? message;
    });
    return { messages: out, blocks };
}
/**
 * A model turn with its Neuralese content in conversation-text form: part arrays (from a Neuralese backend) and
 * marker text (from the stand-in path) in the reply text and in every string of the call arguments become sentinels.
 */
export async function decodeTurnValue(value, port, producer, written) {
    if (typeof value === 'string') {
        if (!value.includes(NEURALESE_OPEN))
            return value;
        const { text, blocks } = await writeLiterals(value, port, producer);
        written?.push(...blocks);
        return text;
    }
    if (isContentParts(value))
        return decodeTurnValue(partsToText(value), port, producer, written);
    if (Array.isArray(value))
        return Promise.all(value.map(item => decodeTurnValue(item, port, producer, written)));
    if (value && typeof value === 'object') {
        const entries = await Promise.all(Object.entries(value).map(async ([key, item]) => [key, await decodeTurnValue(item, port, producer, written)]));
        return Object.fromEntries(entries);
    }
    return value;
}
