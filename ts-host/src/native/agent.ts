import { nativeProtocolFailure, nativeProtocolFeedback } from './native-protocol.js';
import { renderAuthoredTextGuidance } from '../adaptation/prompts.js';
import { fingerprint } from '../adaptation/identity.js';
import type { AuthoredTextGuidance, NativeProgramGuidance } from '../adaptation/prompts.js';
import { formatType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, Reject, coerce, isLive, liveId, liveLabel, problems } from './values.js';
import type { Value } from './values.js';
import { COMPACTION_NOTE_CHARS, runSeedKey, type NativeResult, type NativeSession } from './runtime.js';
import { undeclaredServiceType } from './introspection.js';
import type { ModelTurn, ModelTurnDelta, ModelTurnOptions, ModelTurnRequest } from '../contracts.js';
import { deriveSeed } from './trace.js';
import { AUTOMATIC_NOTE, COMPACTION_NOTICE, directoryReducerPrompt, scopedFileToolNames, FUNCTION_TOOLS_PROMPT, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN,
  LAST_TURN_NOTICE, TOOLS_PROMPT, promptAtNlDepthLimit, type FileToolSurface } from './prompt.js';
import { canGenerateNl, runInFrame } from '../runtime/context.js';
import { adoptImportedNativeArtifacts } from './nz-file.js';
import { FileHandle, FolderHandle, fileListingText, type Folder } from './scoped-fs.js';
import { SHOWN_CHARS, SCOPE_VALUE_CHARS, shownCharsFor, note as cutNote } from './cutoff.js';

/** Scope and input values show at least SCOPE_VALUE_CHARS, and never less than a tool output of the call may. */
const scopeValueChars = (session: NativeSession) => Math.max(SCOPE_VALUE_CHARS, session.pages.budget);
const nativeProgramGuidanceWrites = new WeakMap<NeuraleseStore, Map<string, Promise<string>>>();
const nativeProgramGuidancePins = new WeakMap<NeuraleseStore, Map<string, Promise<void>>>();

async function pinNativeProgramGuidance(store: NeuraleseStore, id: string): Promise<void> {
  let pins = nativeProgramGuidancePins.get(store);
  if (!pins) nativeProgramGuidancePins.set(store, pins = new Map());
  let pending = pins.get(id);
  if (!pending) {
    pending = store.pin(id);
    pins.set(id, pending);
    void pending.catch(() => { if (pins!.get(id) === pending) pins!.delete(id); });
  }
  await pending;
}
import { listingViewInstructions, viewNote } from './prompt.js';
import { untrustedBlock, type UntrustedRegistry } from './untrusted.js';
import { writtenReturnResult } from './pseudo-call.js';
import { decodeTurnValue, encodeMessages, isContentParts, isNativeAdapterRef, isNeuraleseRef, neuraleseSentinel, neuraleseRefSentinel, nativeStateSentinel, NeuraleseUnsupportedError, supportsNeuralese,
  nativeStateSentinelIds, runtimeTextToParts, type NeuraleseRuntimeOptions } from './neuralese.js';
import type { NativeStateArtifact, NeuraleseStore } from './neuralese-store.js';
import { blockInput, graphNode, hasProducerOccurrence, invocationNodeId, producerOccurrences, producerScopeForTrace } from './graph.js';
import { failureFeedback, type RefinementChecker, type RefinementFailure, type RefinementJudge } from './refinement.js';
import { containsRefinement } from './types.js';
import { DECISION_SYSTEM_PROMPT, decisionPrompt, decisionScorer, finiteValues, softmax } from './decision.js';
import type { NeuraleseBlockMeta } from './neuralese-store.js';
import {activeRecorder} from './model-recording.js';
import { hexDigest } from './hash.js';
import { activeSystemPrompts, softenMessages, renderPromptMessages, type PromptRendering } from './system-prompts.js';

/** The code tools, as offered. Kept here so data collected under earlier wording can be migrated to it exactly. */
export const READ_CODE_DESCRIPTION = 'Inspect a named item this call can use but does not show: source of a function in the program\'s codebase; ' +
  'documentation for eval built-ins (nl, iterateOn, transcript, decide); a declaration for an external service or importable package ' +
  '("pkg" lists its exports, "pkg.name" shows one); the type-only member shape of a host service object visible to this call; ' +
  'or the exact schema and instructions of a native tool currently offered to this call. Tool schemas and service shapes are ' +
  'metadata, not program function source or service implementation. Standard JavaScript built-ins and methods are used directly ' +
  'through their TypeScript library types; read_code does not show their library implementation.';
export const EDIT_CODE_DESCRIPTION = 'Edit a function in the program\'s codebase: replace one exact or uniquely fuzzy span of its source. ' +
  'The function is validated before the edit becomes live.';
/**
 * Each assistant turn goes back into the conversation with its reasoning. The template decides whether to show it:
 * some drop earlier reasoning, others keep it (Ling's "preserved thinking"), and there a turn sent without it reads
 * as an empty <think></think> that teaches the model to skip thinking. The runtime's own opening turns carry a short
 * thought for the same reason.
 */
const thought = (reasoning: string | undefined) => reasoning ? { reasoning_content: reasoning } : {};
/** Exact provider proof of an immediate empty EOS; an empty decoded string alone is not sufficient. */
function immediateEmptyEos(response: ModelTurn): boolean {
  if (response.completion_tokens !== 0 || response.truncated || response.text !== '' ||
      (response.calls?.length ?? 0) !== 0) return false;
  const choices = response.raw_response?.choices;
  if (!Array.isArray(choices) || !choices.length) return false;
  const choice = choices[0] as Record<string, unknown> | undefined;
  const message = choice?.message as Record<string, unknown> | undefined;
  const content = message?.content;
  const toolCalls = message?.tool_calls;
  return choice?.finish_reason === 'stop' && (content === null || content === '') &&
    (toolCalls === undefined || (Array.isArray(toolCalls) && toolCalls.length === 0));
}
/**
 * The finishing tool. The last sentence is for judgments: asked "does review recommend the product?", a model answered
 * "it does not" with status blocked, and the caller got an error for a perfectly good answer.
 */
export const RETURN_RESULT_DESCRIPTION = 'Finish the call. With status "success", value is the result and must have the declared ' +
  'return type; if a complete typed result is already staged, you may omit value to return that exact stored result. With status ' +
  '"blocked" (required information is missing; do not guess) or "failed" (the instructions require an ' +
  'invalid or contradictory operation), give the reason instead of a value. A negative answer (false, no, none, zero, an empty ' +
  'list) is a result like any other: return it with status "success".';
/** The finishing tool's description before its sentence on negative answers; kept for migrating collected data. */
export const RETURN_RESULT_DESCRIPTION_BEFORE = 'Finish the call. With status "success", value is the result and must have the ' +
  'declared return type. With status "blocked" (required information is missing; do not guess) or "failed" (the instructions ' +
  'require an invalid or contradictory operation), give the reason instead of a value.';

/** Named in every opening, so a model that looks for them knows they exist and where their documentation is. */
export const BUILT_INS_LINE = 'Eval also has the built-ins nl, iterateOn, transcript and decide; read_code shows how to use each.';
/** Inline children have the same delegation tools as named calls. */
export const INLINE_BUILT_INS_LINE = BUILT_INS_LINE;
export const NL_DEPTH_LIMIT_BUILT_INS_LINE = 'Eval has iterateOn, transcript and decide; read_code shows how to use each. ' +
  'This is the fifth ad hoc layer: make further judgments here or call an existing named function from a file.';
export const OPENING_THOUGHT = "I'll start by reading this call's arguments into the eval scope.";
export const FOLDER_THOUGHT = "Next I'll list the files in this call's folder.";
export const DIFF_CODE_DESCRIPTION = 'Show the changes made to functions of the program\'s codebase in this call.';

/** Assistant turns the model has taken, not counting the runtime's pre-filled scope calls. */
export function modelTurnsSoFar(messages: readonly Record<string, unknown>[]): number {
  return messages.filter(message => message.role === 'assistant' &&
    !((message.tool_calls as { id?: string }[] | undefined) ?? []).some(call => String(call.id).startsWith('scope_'))).length;
}

export type NativeModelDriver = ((request: ModelTurnRequest, signal?: AbortSignal, options?: ModelTurnOptions) => Promise<ModelTurn> | ModelTurn) & {
  /** Scripted host drivers may ask to route fixed answers by the actual native return contract. */
  readonly requiresResultContract?: boolean };

export type NativeReviewOptions = { driver?: NativeModelDriver; threshold?: number;
  scope?: 'values' | 'actions'; withdrawalPolicy?: 'caller' | 'retry';
  prompt?: 'baseline' | 'repeat_instructions' | 'checklist';
  order?: 'reason_first' | 'decision_first' };

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  type: 'function', function: { name, description, parameters: { type: 'object', properties,
    required, additionalProperties: false } },
});

/**
 * The call's arguments as its caller gave them, as the opening eval's result shows them (read_inputs() returns
 * the same values in eval). A long value is cut off, and its name holds all of it; an argument the caller left out is undefined.
 */
export function inputsListing(session: NativeSession, views: Readonly<Record<string, string>> = {}): string {
  const lam = session.lam;
  if (lam.type.kind !== 'lambda') return '{}';
  session.markUntrusted();
  const root = lam.projectTransaction?.folder;
  return lam.type.params.fields.map(field => {
    const value = Object.hasOwn(lam.args, field.name) ? lam.args[field.name]! : undefined;
    // A large value written as a view (the builtin view's Neuralese instance) shows the view; the variable holds the value itself.
    const shown = views[field.name] ? neuraleseSentinel(views[field.name]!) + viewNote(field.name) :
      renderValue(value, { root, holder: field.name, budget: scopeValueChars(session),
        liveIdentity: session.runtime.displayLiveId, untrusted: session.runtime.untrusted });
    const opening = value instanceof FileHandle && value.folder === root ? (() => {
      const stat = root!.listFiles().find(entry => entry.path === value.path);
      if (!stat || stat.bytes > 4000) return '  // Read this file with read_file or file.readText() before answering.';
      try { return `\n  // File contents:\n${JSON.stringify(new TextDecoder('utf-8', { fatal: true }).decode(root!.readBytesSync(value.path)))}`; }
      catch { return '  // Binary file: use file.readBytes() before answering.'; }
    })() : '';
    return `${field.name}: ${formatType(field.type)}${field.optional ? ' | undefined' : ''} = ${shown}${opening}`;
  }).join('\n');
}

/** The folder handle API a directory reducer's eval sees, as TypeScript declarations. */
const FOLDER_DECLARATIONS = [
  'interface Entry { readonly name: string; readonly path: string; readonly relativePath: string; readonly parent: Folder | null; exists(): Promise<boolean>;',
  '  stat(): Promise<{ path: string, kind: "file" | "folder", bytes: number }>; remove(): Promise<void>;',
  '  /** Like mv: moveTo("done/") or moveTo(folder.dir("done")) moves into that folder; moveTo("done/a.md") renames. */',
  '  moveTo(destination: Folder | FileHandle | string): Promise<void>; }',
  'interface FileHandle extends Entry { readText(startLine?: number, endLine?: number): Promise<string>; readJson(): Promise<unknown>;',
  '  readBytes(): Promise<Uint8Array>; writeText(content: string): Promise<void>; writeJson(value: unknown): Promise<void>;',
  '  writeBytes(content: Uint8Array): Promise<void>; editText(find: string, replaceWith: string, fuzzy?: boolean): Promise<unknown>; }',
  'interface Folder extends Entry { file(path: string): FileHandle; dir(path: string): Folder; entries(pattern?: string): Promise<Entry[]>;',
  '  files(pattern?: string): Promise<FileHandle[]>; folders(pattern?: string): Promise<Folder[]>; diff(): Promise<unknown>;',
  '  /** Run a directory reducer on this folder and keep the file changes it commits. */',
  '  apply(reducer: Function, ...args: unknown[]): Promise<unknown>;',
  '  snapshot(): FolderSnapshot; at(sourceId:string):FolderSnapshot; propose<R>(reducer: Function, ...args: unknown[]): Promise<FolderProposal<R>>;',
  '  accept<R>(proposal: FolderProposal<R>): Promise<FolderSnapshot>; select(snapshot: FolderSnapshot): Promise<void>; }',
  'interface FolderSnapshot extends Folder {readonly digest:string;branch():Folder;}',
  'interface FolderProposal<R> {readonly folder:FolderSnapshot;readonly value:R;readonly baseRevision:number;readonly diff:unknown;}',
];

const DEFAULT_CONTEXT_TOKENS = 16384;

/**
 * The context budget when none is configured: the model's context window as its server reports it (a driver's
 * `contextWindow()`), less an eighth (at least 2048 tokens) for the reply; DEFAULT_CONTEXT_TOKENS when unknown.
 */
/** Tokens kept free between the estimated prompt and the reply limit once the window is known. */
const REPLY_MARGIN = 64;

/**
 * The prompt size and context window a server states when it refuses a request as longer than its context, for the
 * servers we run: llama.cpp (`exceed_context_size_error` with `n_prompt_tokens`, `n_ctx`) and vLLM ("maximum context
 * length is W tokens ... your prompt contains at least P input tokens", or "(P in the messages, C in the completion)").
 */
export function contextOverflow(message: string): { promptTokens: number; window: number | null } | undefined {
  const llama = /exceed_context_size_error[\s\S]*?"n_prompt_tokens":(\d+)/.exec(message);
  if (llama) {
    const window = /"n_ctx":(\d+)/.exec(message);
    return { promptTokens: Number(llama[1]), window: window ? Number(window[1]) : null };
  }
  const window = /maximum context length is (\d+) tokens/.exec(message);
  if (!window) return undefined;
  const prompt = /prompt contains at least (\d+) input tokens/.exec(message) ?? /\((\d+) in the messages/.exec(message) ??
    /request has (\d+) input tokens/.exec(message);
  return prompt ? { promptTokens: Number(prompt[1]), window: Number(window[1]) } : undefined;
}

export async function defaultContextBudget(driver: unknown): Promise<number> {
  const ask = (driver as { contextWindow?: unknown } | null)?.contextWindow;
  let window: unknown;
  try { window = typeof ask === 'function' ? await (ask as () => Promise<unknown>).call(driver) : undefined; } catch { window = undefined; }
  if (typeof window !== 'number' || !Number.isInteger(window) || window < 4096) return DEFAULT_CONTEXT_TOKENS;
  return window - Math.max(2048, Math.floor(window / 8));
}

/**
 * Compaction: everything between the kept start (the opening, and the pinned note once there is one) and the latest
 * exchange (the last assistant turn and what followed it) leaves the conversation. It is all in transcript, and every
 * value the model stored is still in eval's scope. Returns how many messages were removed.
 */
export function collapseHistory(messages: Record<string, unknown>[], keptStart: number): number {
  let latest = messages.length - 1;
  while (latest >= keptStart && messages[latest]!.role !== 'assistant') latest--;
  if (latest <= keptStart) return 0;
  messages.splice(keptStart, latest - keptStart);
  return latest - keptStart;
}


function schemaOf(type: Type, env: TypeEnv, depth = 0): Record<string, unknown> {
  if (depth > 5) return {};
  const resolved = env.resolve(type);
  if (resolved.kind === 'refined' || resolved.kind === 'untrusted') return schemaOf(resolved.base, env, depth);
  if (resolved.kind === 'prim' && resolved.name === 'unknown') return {};
  if (resolved.kind === 'prim') return { type: { string: 'string', Blob: 'string', number: 'number',
    boolean: 'boolean', null: 'null', Folder: 'object', FileHandle: 'object' }[resolved.name as Exclude<typeof resolved.name, 'unknown'>],
    ...(['Folder', 'FileHandle'].includes(resolved.name) ? { 'x-natlang': `${resolved.name.toLowerCase()}-handle` } : {}) };
  if (resolved.kind === 'lit') return { const: resolved.value };
  if (resolved.kind === 'union') return resolved.members.every(part => env.resolve(part).kind === 'lit') ?
    { enum: resolved.members.map(part => (env.resolve(part) as Extract<Type, { kind: 'lit' }>).value) } :
    { anyOf: resolved.members.map(part => schemaOf(part, env, depth + 1)) };
  if (resolved.kind === 'list') return { type: 'array', items: schemaOf(resolved.element, env, depth + 1) };
  if (resolved.kind === 'dict') return { type: 'object', additionalProperties: schemaOf(resolved.element, env, depth + 1) };
  if (resolved.kind === 'record') return { type: 'object',
    properties: Object.fromEntries(resolved.fields.map(field => [field.name, schemaOf(field.type, env, depth + 1)])),
    required: resolved.fields.filter(field => !field.optional).map(field => field.name), additionalProperties: false };
  return {};
}

function pythonJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value)
    .map(([key, item]) => `${JSON.stringify(key)}: ${pythonJson(item)}`).join(', ')}}`;
  return JSON.stringify(value);
}

function referencedTypeAliases(signatures: string[], definitions: Record<string, string>,
  declarations: Record<string, string> = {}): string[] {
  const found = new Set<string>();
  const visit = (text: string): void => {
    for (const name of Object.keys(definitions)) {
      if (found.has(name) || !new RegExp(`\\b${name}\\b`).test(text)) continue;
      found.add(name);
      visit(declarations[name] ?? definitions[name]!);
    }
  };
  for (const signature of signatures) visit(signature);
  // A class or method-bearing interface is shown as its TypeScript declaration, not its live-value alias.
  return [...found].map(name => declarations[name] ?? `type ${name} = ${definitions[name]};`);
}

/**
 * The one way a value is shown to the model: a TypeScript literal cut at item and field boundaries when it is long
 * (scopeExpression), or a preview when it has no literal form (live values). `holder` is where all of it is.
 */
export function renderValue(value: Value | unknown, options: { holder?: string; budget?: number; root?: Folder; liveIdentity?: (value: object) => number;
  /** Where untrusted text came from; a string it knows is shown as a quoted data block. */ untrusted?: UntrustedRegistry } = {}): string {
  if (value === undefined) return 'undefined';
  const requested=options.budget??SHOWN_CHARS;
  const budget=Number.isFinite(requested)?Math.max(0,requested):Infinity;
  // Render once at the requested budget. Building the complete literal first made even a
  // 2,000-character scope preview allocate the entire value (including combinatorial arrays).
  return scopeExpression(value, options.root, options.holder, budget, { nodes: 0, depth: 0, untrusted: options.untrusted }) ?? previewValue(value as Value, options.holder, options.liveIdentity);
}

/** Native coefficient artifacts act on model weights and stay out of the text-space scope. */
const isAdapterValue = isNativeAdapterRef;
function adapterCells(lam: { captures?: Record<string, { get(): unknown }> }): { id: string; scale: number }[] {
  const out: { id: string; scale: number }[] = [];
  for (const cell of Object.values(lam.captures ?? {})) {
    let value: unknown;
    try { value = cell.get(); } catch { continue; }
    if (isAdapterValue(value)) out.push({ id: value.$nativeAdapter.id, scale: 1 });
  }
  return out;
}
const capturedValue = (cell: { get(): unknown }) => { try { return cell.get(); } catch { return undefined; } };

/** A short preview of a value that has no literal form; `holder` names where all of it is (see cutoff.ts). */
function previewValue(value: Value, holder?: string, identity?: (value: object) => number): string {
  if (isNeuraleseRef(value)) return neuraleseRefSentinel(value);
  if (isLive(value)) return livePreview(value as object, identity);
  if (typeof value === 'string') {
    const text = value.trimEnd(), lines = text.split('\n');
    if (text.length > 400 || lines.length > 8)
      return `${JSON.stringify(lines[0]!.slice(0, 80))} ${cutNote(`cut off: ${text.length - Math.min(80, lines[0]!.length)} of ${text.length} characters not shown`, { holder })}`;
    return lines.length > 1 ? '\n' + lines.map(line => `      | ${line}`).join('\n') : JSON.stringify(text);
  }
  if (Array.isArray(value)) {
    const head = value.slice(0, 3).map(item => previewValue(item, undefined, identity)).join(', ');
    return `[${head}${value.length > 3 ? `, ${cutNote(`cut off: ${value.length - 3} of ${value.length} items not shown`, { holder })}` : ''}]`;
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    const head = entries.slice(0, 6).map(([key, item]) => `${key}: ${previewValue(item, undefined, identity)}`).join(', ');
    return `{ ${head}${entries.length > 6 ? `, ${cutNote(`cut off: ${entries.length - 6} of ${entries.length} fields not shown`, { holder })}` : ''} }`;
  }
  if (value === null) return 'null';
  return String(value);
}

/** Bounded preview of a live host value: type, stable identity, and a short observation. */
export function livePreview(value: object, identity: (value: object) => number = liveId): string {
  const label = liveLabel(value), id = identity(value);
  let detail = '';
  try {
    const tag = Object.prototype.toString.call(value);
    if (tag === '[object Date]') detail = ` ${(value as Date).toISOString()}`;
    else if (tag === '[object Map]' || tag === '[object Set]') detail = ` size ${(value as Map<unknown, unknown>).size}`;
    else if (typeof value === 'function') detail = (value as Function).length ? ` (${(value as Function).length} parameters)` : '';
    // A class that says what its instances are is shown by that instead of its fields.
    else if (typeof (value as { toString?: unknown }).toString === 'function' &&
        (value as { toString: unknown }).toString !== Object.prototype.toString) detail = `: ${String(value)}`;
    else {
      const keys = Object.keys(value).slice(0, 6);
      if (keys.length) detail = ` { ${keys.join(', ')}${Object.keys(value).length > 6 ? ', …' : ''} }`;
    }
  } catch { /* an exotic object may refuse inspection */ }
  return `[${label} #${id}${detail}; live value, use it in eval]`;
}

const isPlainRecord = (value: object) => Object.prototype.toString.call(value) === '[object Object]' &&
  (Object.getPrototypeOf(value) === null || Object.getPrototypeOf(Object.getPrototypeOf(value)) === null);

/**
 * TypeScript that evaluates to a scope value: a literal for data, `folder.file(...)` for a handle into the
 * reducer's folder, `new Date(...)`/`new Map(...)`/`new Set(...)`/`new Uint8Array(...)` for those built-ins.
 * A value beyond the budget is cut off with a comment saying `holder` (the variable) holds all of it.
 * Undefined when no expression produces the value (an opaque host object).
 */
function scopeExpression(value: unknown, root: Folder | undefined, holder?: string, budget = SHOWN_CHARS,
  state: { nodes: number; depth: number; untrusted?: UntrustedRegistry } = { nodes: 0, depth: 0 }): string | undefined {
  if (++state.nodes > 8192 || state.depth >= 64)
    return cutNote('inspection limit reached', { holder });
  state.depth++;
  try {
  const sequence = <T,>(items: Iterable<T>, count: number, open: string, close: string, noun: string,
    render: (item: T, left: number) => string | undefined) => {
    const shown: string[] = [];
    let used = 0;
    for (const item of items) {
      if (used >= budget || state.nodes >= 8192) break;
      const text = render(item, budget - used);
      if (text === undefined) return undefined;
      shown.push(text); used += text.length + 2;
    }
    if (shown.length === count) return `${open}${shown.join(', ')}${close}`;
    return `${open}${shown.join(', ')}${shown.length ? ', ' : ''}${cutNote(`cut off: ${count - shown.length} of ${count} ${noun} not shown`, { holder })}${close}`;
  };
  if (value === null || typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : String(value);
  if (typeof value === 'string') {
    // Text from outside the program is data: a labelled block, never a bare literal (native/untrusted.ts).
    const source = state.untrusted?.sourceOf(value);
    if (source !== undefined) {
      if (value.length <= Math.max(16, budget)) return untrustedBlock(value, source);
      return `${untrustedBlock(value.slice(0, Math.max(0, budget)), source)}${cutNote(`cut off: ${value.length - Math.max(0, budget)} of ${value.length} characters not shown`, { holder })}`;
    }
    // Long record keys can exhaust the preview budget before their values.
    // Preserve tiny labels such as "0"/"1" rather than showing an empty string
    // with a truncation note; the literal is shorter than that note anyway.
    if (value.length <= Math.max(16, budget)) return JSON.stringify(value);
    return `${JSON.stringify(value.slice(0, Math.max(0, budget)))} ${cutNote(`cut off: ${value.length - Math.max(0, budget)} of ${value.length} characters not shown`, { holder })}`;
  }
  if (typeof value !== 'object' || value === undefined) return undefined;
  if (value instanceof FileHandle || value instanceof FolderHandle) {
    if (!root || value.folder !== root) return undefined;
    return value instanceof FileHandle ? `folder.file(${JSON.stringify(value.path)})` : value.path ? `folder.dir(${JSON.stringify(value.path)})` : 'folder';
  }
  const tag = Object.prototype.toString.call(value);
  if (tag === '[object Date]') return `new Date(${JSON.stringify((value as Date).toISOString())})`;
  if (tag === '[object Uint8Array]') return sequence(value as Uint8Array, (value as Uint8Array).length,
    'new Uint8Array([', '])', 'bytes', item => String(item));
  if (tag === '[object Set]') return sequence(value as Set<unknown>, (value as Set<unknown>).size, 'new Set([', '])', 'members',
    (item, left) => scopeExpression(item, root, holder, left, state));
  if (tag === '[object Map]') return sequence(value as Map<unknown, unknown>, (value as Map<unknown, unknown>).size,
    'new Map([', '])', 'entries', ([key, item], left) => {
    const keyText = scopeExpression(key, root, holder, left, state);
    const itemText = keyText === undefined ? undefined : scopeExpression(item, root, holder, left - keyText.length, state);
    return itemText === undefined ? undefined : `[${keyText}, ${itemText}]`;
  });
  if (Array.isArray(value)) {
    // An item's own cut-offs name the item (`rows[2].text`), which is where all of that part is.
    let index = 0;
    return sequence(value, value.length, '[', ']', 'items', (item, left) =>
      scopeExpression(item, root, holder ? `${holder}[${index++}]` : undefined, left, state));
  }
  // A soft value is shown as its literal: the model reads the block itself (the transport sends it as a part).
  if (isNeuraleseRef(value)) return neuraleseRefSentinel(value);
  if (!isPlainRecord(value)) return undefined;
  const entries: [string, unknown][] = [];
  let fieldCount = 0;
  for (const key in value) if (Object.hasOwn(value, key)) {
    if (entries.length < (Number.isFinite(budget) ? 24 : 8192)) entries.push([key, (value as Record<string, unknown>)[key]]);
    fieldCount++;
  }
  // Give ordinary argument records an overview, rather than allowing a large first field
  // to hide every later field. Tiny nested records show their keys and a precise holder.
  if (Number.isFinite(budget) && budget < 128 && entries.length > 2)
    return `{ ${cutNote('fields: '+entries.map(([key])=>key).join(', ')+(fieldCount>entries.length?', …':'')+'; values not shown', {holder})} }`;
  const visible = entries;
  const overhead = visible.reduce((sum,[key])=>sum+key.length+4,0);
  const available = budget - overhead;
  const fields = visible.map(([key,item])=>{
    const identifier=/^[A-Za-z_$][\w$]*$/.test(key);
    return { label: identifier?key:JSON.stringify(key), item,
      holder: holder?(identifier?holder+'.'+key:holder+'['+JSON.stringify(key)+']'):undefined };
  });
  // Field budgets by water-filling: a field whose whole literal fits in an even share keeps it, and what it leaves
  // goes to the longer fields, so a record that fits is shown whole and a long field cannot hide the short ones.
  const budgets = fields.map(()=>budget);
  if (Number.isFinite(budget)) {
    const probe = { nodes: state.nodes, depth: state.depth, untrusted: state.untrusted };
    const needs = fields.map(field=>{
      const text=scopeExpression(field.item,root,field.holder,Math.max(0,available),probe);
      return text===undefined?0:text.length;
    });
    let left = Math.max(0, available), open = fields.map((_,i)=>i);
    for (let settled = true; open.length; ) {
      settled = true;
      const share = Math.floor(left / open.length);
      for (const i of [...open]) if (needs[i]! <= share) { budgets[i] = needs[i]!; left -= needs[i]!; open = open.filter(j=>j!==i); settled = false; }
      if (settled) { for (const i of open) budgets[i] = Math.max(0, Math.floor(left / open.length)); break; }
    }
  }
  const shown:string[]=[];
  for(const [index,field] of fields.entries()){
    const text=scopeExpression(field.item,root,field.holder,budgets[index]!,state);
    if(text===undefined)return undefined;
    shown.push(field.label+': '+text);
  }
  if(visible.length<fieldCount)shown.push(cutNote('cut off: '+(fieldCount-visible.length)+' of '+fieldCount+' fields not shown',{holder}));
  return '{ '+shown.join(', ')+' }';
  } finally { state.depth--; }
}

/** The native model loop. Program state stays in NativeSession, never in the model history. */
export class NativeToolAgent {
  readonly proposals: Record<string, unknown>[] = [];
  readonly reviews: Record<string, unknown>[] = [];
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; maxSeconds?: number; promptRendering?: PromptRendering; systemPrompt?: string | (() => string);
      programGuidance?: NativeProgramGuidance; authoredTextGuidance?: AuthoredTextGuidance;
      review?: NativeReviewOptions;
      maxFailureRepairs?: number;
      /** The readout's system prompt (default DECISION_SYSTEM_PROMPT; the kernel adds iteration addenda). */
      decisionSystemPrompt?: () => string;
      /** Which finite-typed calls answer by decision readout (native/decision.ts); default `declared`. */
      decisionReadout?: 'declared' | 'finite-returns';
      /**
       * Guided generation on natlang's own servers (reference and llama.cpp fork; others never receive it): the reply
       * opens a tool call when one is required, call names are checked, and eval code is checked line by line for
       * repetition and TypeScript syntax, with rejected lines rolled back and resampled. `true` or settings
       * (`{ repeat?, syntax?, retries?, tools? }`); off by default.
       */
      guidance?: boolean | { repeat?: number; syntax?: boolean; retries?: number; tools?: string[] };
      /** The file tools a directory reducer offers (prompt.ts FileToolSurface; default all). */
      fileTools?: FileToolSurface;
      /**
       * Checks `Is<T, P>` slots of the call's result before it is accepted (native/refinement.ts). A failure goes back
       * to the model as a tool error within the repair budget; past it the call fails with the refinement's code.
       */
      refinement?: { checker: RefinementChecker; judge?: RefinementJudge; escalation?: RefinementJudge;
        /** Optional host-owned return obligation, checked with the same repair feedback as refined slots. */
        returnGuard?: (runId: string) => Promise<RefinementFailure[]> };
      /** Tensor store and write port for soft values (S0 §3). */
      neuralese?: NeuraleseRuntimeOptions;
      /**
       * Receives the deltas of the call's model turns as they stream (plans/STREAMING.md §1.6), with the turn's number
       * (as its `model_request` trace events give it). A turn taken again after a failed request first gets a `reset`.
       * Without it the driver is called with no turn options.
       */
      onDelta?: (turn: number, delta: ModelTurnDelta) => void;
      /**
       * Context budget in prompt tokens (default 16384; null never compacts). Past three quarters of it the oldest
       * tool outputs are elided until the prompt is back under half.
       */
      contextTokens?: number | null } = {}) {
    if (options.maxFailureRepairs !== undefined &&
        (!Number.isInteger(options.maxFailureRepairs) || options.maxFailureRepairs < 0))
      throw new RangeError('maxFailureRepairs must be a non-negative integer');
    if (options.contextTokens !== undefined && options.contextTokens !== null &&
        (!Number.isInteger(options.contextTokens) || options.contextTokens < 1024))
      throw new RangeError('contextTokens must be an integer of at least 1024, or null');
  }

  /**
   * Views of the arguments whose listing would be cut off, when the runtime has view's Neuralese instance and the driver
   * carries Neuralese: block IDs by argument name. Each is a call of `view(value, instructions)` at its Neuralese
   * representation, its instructions the receiving call's (prompt.ts `listingViewInstructions`). Only plain data is
   * viewed; a failed view keeps the preview.
   */
  private async views(session: NativeSession): Promise<Record<string, string>> {
    const view = this.options.neuralese?.view;
    const lam = session.lam;
    if (!view || !supportsNeuralese(this.driver) || lam.type.kind !== 'lambda') return {};
    const root = lam.projectTransaction?.folder;
    const out: Record<string, string> = {};
    for (const field of lam.type.params.fields) {
      if (!Object.hasOwn(lam.args, field.name)) continue;
      const value = lam.args[field.name]!;
      if (!renderValue(value, { root, holder: field.name, budget: scopeValueChars(session),
        liveIdentity: session.runtime.displayLiveId, untrusted: session.runtime.untrusted }).includes('<<cut off:')) continue;
      let text: string | undefined;
      try { text = JSON.stringify(value); } catch { text = undefined; }
      if (text === undefined || value instanceof FileHandle) continue;
      const instructions = listingViewInstructions(field.name, formatType(field.type), typeof lam.body === 'string' ? lam.body.replace(/\n+$/, '') : '');
      try {
        const written = await view({ value: text, instructions });
        if (written) {
          out[field.name] = written.$neuralese.id;
          session.runtime.trace.emit('view', { call_id: session.runtime.currentCallId ?? null, argument: field.name,
            representation: 'neuralese', block: written.$neuralese.id, chars: text.length });
        }
      } catch (error) {
        session.runtime.trace.emit('view', { call_id: session.runtime.currentCallId ?? null, argument: field.name,
          representation: 'neuralese', error: String((error as Error)?.message ?? error).slice(0, 300) });
      }
    }
    return out;
  }

  /** Messages with the runtime's prompt pieces in their soft forms, when a bank is configured and the driver is Neuralese. */
  private softened<M extends Record<string, unknown>>(messages: readonly M[]): M[] {
    const bank = activeSystemPrompts(this.options.neuralese?.systemPrompts);
    return bank?.size && supportsNeuralese(this.driver) ? softenMessages(messages, bank) : [...messages];
  }

  /** Record only actually completed native writes, even when the containing turn failed. */
  private async nativeWriteOrigins(response: ModelTurn, session: NativeSession, turn: number, turnNode?: string): Promise<Map<string,string[]>> {
    const nativeBlocks = (response.raw_response?.native_state as { blocks?: Record<string, unknown>[] } | undefined)?.blocks ?? [];
    const producer = {call_id:session.runtime.currentCallId??null,turn};
    const nativeOrigins=new Map<string,string[]>();
    for (const block of nativeBlocks.filter(block => typeof block.id === 'string')) {
      const id = block.id as string;
      const artifact = await this.options.neuralese?.store.getNativeState?.(id);
      if (!artifact) throw new Error(`native-state producer artifact ${id} was not archived before the agent turn`);
      const tokenIds = artifact?.tensors.token_ids;
      const embeddings = artifact?.tensors.embeddings;
      if (!tokenIds || !embeddings) throw new Error(`native-state producer artifact ${id} lacks token/embedding tensors`);
      const node=graphNode(session.runtime.trace, 'block_write', { call_id: producer.call_id,
        turn: turnNode ?? '', block: id, format: 'native-state', producer: 'native-state-producer',
        source_kind: 'native-state-producer', learned_vectors: false,
        ...(typeof block.type === 'string' ? { result_type: block.type } : {}),
        ...(typeof artifact.metadata.text === 'string' ? { text_body_sha256: hexDigest(artifact.metadata.text) } : {}),
        length: tokenIds.shape[0], width: embeddings.shape[1], truncated: false, stops: [],
        native_identity: { architecture: artifact.metadata.architecture,
          checkpoint_sha256: artifact.metadata.checkpoint_sha256,
          tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
          embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 },
        tensor_inventory: Object.fromEntries(Object.entries(artifact.tensors)
          .map(([name, tensor]) => [name, { dtype: tensor.dtype, shape: tensor.shape }])),
        metadata: block },
        turnNode ? [{ node: turnNode, port: 'turn' }] : []);
      if(node)nativeOrigins.set(id,[...(nativeOrigins.get(id)??[]),node]);
    }
    return nativeOrigins;
  }

  /**
   * Turn the Neuralese content of a reply into conversation text: blocks a backend wrote (content parts) and literals
   * written as marker text (stored through the write port) become block markers in the reply and call arguments.
   */
  private async decodeNeuralese(response: ModelTurn, session: NativeSession, turn: number, turnNode?: string,
      allowAssistantResult = true): Promise<ModelTurn> {
    const carries = (value: unknown): boolean => typeof value === 'string' ?
      value.includes('<|neuralese|>') || nativeStateSentinelIds(value).length > 0 :
      Array.isArray(value) ? value.some(carries) : !!value && typeof value === 'object' &&
        (String((value as { type?: unknown }).type) === 'native_state' ||
          Object.values(value).some(carries));
    const nativeBlocks = (response.raw_response?.native_state as { blocks?: Record<string, unknown>[] } | undefined)?.blocks ?? [];
    const rawChoices = response.raw_response?.choices as Array<{ message?: { content?: unknown } }> | undefined;
    const content = rawChoices?.[0]?.message?.content;
    if (!carries(response.text) && !carries(response.calls) && !carries(content) && !nativeBlocks.length) return response;
    const returnType = session.lam.type.kind === 'lambda' ? formatType(session.lam.type.returns) : undefined;
    const producer = { call_id: session.runtime.currentCallId ?? null, turn,
      ...(returnType ? { result_type: returnType } : {}) };
    const port = this.options.neuralese?.port;
    const expectedReturn = session.lam.type.kind === 'lambda' ? session.env.resolve(session.lam.type.returns) : undefined;
    const nativeStateText = async (id: string): Promise<string | undefined> => {
      const artifact = await this.options.neuralese?.store.getNativeState?.(id);
      return artifact?.metadata.text;
    };
    const nativeOrigins=await this.nativeWriteOrigins(response,session,turn,turnNode);
    const originUses=new Map<string,number>();
    const withNativeOrigins=(value:unknown):unknown=>{
      if(Array.isArray(value))return value.map(withNativeOrigins);
      if(value&&typeof value==='object'){
        const part=value as {type?:unknown;id?:unknown};
        if(part.type==='native_state'&&typeof part.id==='string'){
          const nodes=nativeOrigins.get(part.id),index=originUses.get(part.id)??0;
          if(nodes?.length){
            originUses.set(part.id,index+1);
            // A repeated reference to one produced state retains that origin.
            // Distinct equal writes are matched in actual output occurrence order.
            const node=nodes.length===1?nodes[0]:nodes[index];
            if(!node)throw new Error('neuralese-native-grad-invalid: native output has no unique producer occurrence');
            return {...value,source_occurrence:node};
          }
          return value;
        }
        return Object.fromEntries(Object.entries(value).map(([key,item])=>[key,withNativeOrigins(item)]));
      }
      return value;
    };
    const written: NeuraleseBlockMeta[] = [];
    const decodeCall = async ([name, args]: [string, Record<string, unknown>], assistantCompletion = false): Promise<[string, Record<string, unknown>]> => {
      const decodedArgs: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(args)) {
        const context = name === 'eval' && key === 'code' ? 'eval-code' :
          name === 'return_result' && key === 'value' ? 'return-result' : 'other-tool-argument';
        const resultValue = name === 'return_result' && key === 'value' &&
          (args.status === undefined || args.status === 'success');
        decodedArgs[key] = await decodeTurnValue(withNativeOrigins(value), port, { ...producer, marker_context: context }, written,
          resultValue && expectedReturn ? { type: expectedReturn, env: session.env } : undefined,
          assistantCompletion ? nativeStateText : undefined);
      }
      return [name, decodedArgs];
    };
    const structuredResult = allowAssistantResult && !response.calls?.length && !response.truncated && isContentParts(content) &&
      content.some(part => part.type !== 'text');
    const decoded = { ...response,
      ...(structuredResult ? { text: '' } : response.text === undefined ? {} : { text: await decodeTurnValue(response.text, port,
        { ...producer, marker_context: 'assistant-text' }, written, undefined, nativeStateText) as string }),
      ...(response.calls === undefined ? {} : { calls: await Promise.all(response.calls.map(call => decodeCall(call))) as ModelTurn['calls'] }) };
    if (structuredResult) {
      // Actual producer parts are a typed return, not an opaque-ID string.
      // Literal strings never enter this branch, even if they spell a marker.
      decoded.text = '';
      decoded.calls = [await decodeCall(['return_result', { status: 'success', value: content }], true)];
    }
    // Each native state the turn wrote is a node. A producer span ends at the model's end marker, so it records no stops.
    const reportedArtifacts = new Set<string>();
    for (const block of written) {
      const reported = (block.producer ?? {}) as Record<string, unknown>;
      if (reportedArtifacts.has(block.id)) continue;
      reportedArtifacts.add(block.id);
      graphNode(session.runtime.trace, 'block_write', { call_id: producer.call_id, turn: turnNode ?? '', block: block.id,
        length: block.length, truncated: !!block.truncated, stops: Array.isArray(reported.stops) ? reported.stops : [],
        ...Object.fromEntries(['position', 'temperature', 'seed']
          .filter(key => reported[key] !== undefined)
          .map(key => [key, reported[key]])) }, turnNode ? [{ node: turnNode, port: 'turn' }] : []);
    }
    return decoded;
  }

  /**
   * The graph nodes of one model turn: a `block_read` for each block the request carried (positions are
   * [message, part] in the request; a server that reports payload positions gives token positions), and the
   * `model_turn` with its sampling settings and the tool calls it chose.
   */
  private async modelTurnNode(session: NativeSession, messages: readonly unknown[], response: ModelTurn, turn: number, readout?: 'decision'): Promise<string | undefined> {
    const trace = session.runtime.trace;
    const callId = session.runtime.options.runId;
    const turnId = `${callId}#turn${turn}`;
    const resolveReadSource = (block: string, declared?: string): string => {
      // A producer may live in a child or parent invocation's local trace. The
      // active episode recorder has already authenticated those actual graph
      // nodes; include them alongside this request's trace without consulting
      // the global recent-producer cache (which can refer to another episode).
      const recorded=activeRecorder()?.nativeStateWriters?.(block)??[];
      const writers=[...new Set([...trace.events.filter(event=>event.kind==='block_write'&&event.block===block&&
        typeof event.node==='string')
        .map(event=>String(event.node)),...producerOccurrences(block,producerScopeForTrace(trace)),...recorded])];
      if (declared === 'external') {
        if (writers.length) throw new Error(`neuralese-native-grad-invalid: native read ${block} declares external despite an in-episode writer`);
        return 'external';
      }
      if (declared !== undefined) {
        // Keep the actual episode recorder's cross-invocation writers and the
        // task-scoped inference index. Neither can borrow another task's state.
        if (!writers.includes(declared) &&
            !hasProducerOccurrence(block, declared, session.runtime.episodeBudget))
          throw new Error(`neuralese-native-grad-invalid: native read ${block} names no matching in-episode writer ${declared}`);
        return declared;
      }
      if (writers.length > 1)
        throw new Error(`neuralese-native-grad-invalid: native read ${block} has multiple in-episode writers and no occurrence binding`);
      return writers[0] ?? 'external';
    };
    const scopedReads = response.raw_response?.__natlang_native_read_occurrences;
    const preparedMessageOffset = Array.isArray(scopedReads) ? 1 : 0;
    const reads: { node: string; port: string; block: string; source: string; stop: boolean }[] = [];
    type Use={block:string;stop?:true;source?:string};
    const findRefs = (value: unknown): Use[] => {
      if (typeof value === 'string') {
        return runtimeTextToParts(value).flatMap<Use>(part=>part.type==='text'?[]:
          [{block:part.id,...(part.stop_gradient?{stop:true as const}:{}),...(part.source_occurrence?{source:part.source_occurrence}:{})}]);
      } else if (Array.isArray(value)) return value.flatMap(findRefs);
      else if (value && typeof value === 'object') {
        const record = value as { type?: unknown; id?: unknown;stop_gradient?:unknown;source_occurrence?:unknown };
        if (record.type === 'native_state' && typeof record.id === 'string') {
          return [{block:record.id,...(record.stop_gradient===true?{stop:true}:{}),...(typeof record.source_occurrence==='string'?{source:record.source_occurrence}:{})}];
        }
        return Object.values(record).flatMap(findRefs);
      }
      return [];
    };
    for (const [index, message] of messages.entries()) {
      const record = message as { content?: unknown; tool_calls?: { function?: { arguments?: unknown } }[] };
      // Match serving's recursive argument renderer: only recognized part LISTS
      // import state. A singleton lookalike object is ordinary JSON data.
      const argumentRefs = (value: unknown): Use[] => {
        if (Array.isArray(value)) {
          const parts = value.length > 0 && value.every(part => part && typeof part === 'object' &&
            ['text','native_state'].includes((part as {type:string}).type));
          return parts ? findRefs(value) : value.flatMap(argumentRefs);
        }
        if (value && typeof value === 'object') return Object.values(value).flatMap(argumentRefs);
        return [];
      };
      const contentParts = typeof record.content === 'string' ? runtimeTextToParts(record.content) :
        Array.isArray(record.content) ? record.content : record.content === undefined ? [] : [record.content];
      const contentUses = contentParts.map(findRefs);
      const argumentUses = (record.tool_calls ?? []).map(call => {
        const value = call.function?.arguments;
        return argumentRefs(typeof value === 'string' ? value.trim() ? JSON.parse(value) : {} : value);
      });
      for (const [part, uses] of [...contentUses, ...argumentUses].entries()) {
        for (const use of uses) {
          const block=use.block;
          const artifact = await this.options.neuralese?.store.getNativeState?.(block);
          if (!artifact) throw new NeuraleseUnsupportedError(`native-state consumer artifact ${block} is unavailable for provenance`);
          const source = resolveReadSource(block, use.source);
          const node = graphNode(trace, 'block_read', { call_id: callId, turn: turnId, block,
            ...(use.stop?{stop_gradient:true}:{}),
            format: 'native-state', native_identity: {
              architecture: artifact.metadata.architecture, checkpoint_sha256: artifact.metadata.checkpoint_sha256,
              tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
              embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 },
            positions: [index + preparedMessageOffset, part] }, [blockInput(block, 'block', source, producerScopeForTrace(trace))]);
          if (node) reads.push({ node, port: 'read', block, source, stop: !!use.stop });
        }
      }
    }
    // `withNativeStates` adds exact archived-state parts in the transport
    // after NativeToolAgent builds graphMessages. The server response carries
    // their prepared request positions and resolved writer occurrence so the
    // execution graph includes the very import the model consumed.
    if (Array.isArray(scopedReads)) for (const raw of scopedReads) {
      if (!raw || typeof raw !== 'object') throw new Error('neuralese-native-grad-invalid: scoped native read lost its request occurrence');
      const item = raw as { id?: unknown; positions?: unknown; source_occurrence?: unknown; stop_gradient?: unknown };
      if (typeof item.id !== 'string' || !Array.isArray(item.positions) || item.positions.length !== 1 ||
          !Array.isArray(item.positions[0]) || item.positions[0].length !== 2 ||
          !item.positions[0].every(Number.isSafeInteger) || item.positions[0].some((value: number) => value < 0) ||
          item.source_occurrence !== undefined && typeof item.source_occurrence !== 'string' ||
          item.stop_gradient !== undefined && item.stop_gradient !== true)
        throw new Error('neuralese-native-grad-invalid: scoped native read has invalid prepared prompt positions');
      const [messageIndex, partIndex] = item.positions[0] as [number, number];
      const artifact = await this.options.neuralese?.store.getNativeState?.(item.id);
      if (!artifact) throw new NeuraleseUnsupportedError(`native-state consumer artifact ${item.id} is unavailable for provenance`);
      const source = resolveReadSource(item.id, typeof item.source_occurrence === 'string' ? item.source_occurrence : undefined);
      const node = graphNode(trace, 'block_read', { call_id: callId, turn: turnId, block: item.id,
        ...(item.stop_gradient === true ? { stop_gradient: true } : {}), format: 'native-state', native_identity: {
          architecture: artifact.metadata.architecture, checkpoint_sha256: artifact.metadata.checkpoint_sha256,
          tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
          embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 }, positions: [messageIndex, partIndex] },
        [blockInput(item.id, 'block', source, producerScopeForTrace(trace))]);
      if (node) reads.push({ node, port: 'read', block: item.id, source, stop: item.stop_gradient === true });
    }
    // The model server's captured prompt is the authoritative record of what
    // entered the model. A caller can supply context outside `messages`, so
    // reconcile that inventory here instead of silently omitting a consumed
    // state from the episode graph.
    const capturedPrompt = response.raw_response?.x_natlang_rl_capture as { prompt?: unknown } | undefined;
    if (Array.isArray(capturedPrompt?.prompt)) {
      const remaining = [...reads];
      for (const [promptIndex, raw] of capturedPrompt.prompt.entries()) {
        if (!raw || typeof raw !== 'object' || (raw as { kind?: unknown }).kind !== 'native_state') continue;
        const segment = raw as { id?: unknown; source_state_id?: unknown; source_occurrence?: unknown;
          stop_gradient?: unknown; semantic_adaptation?: unknown };
        if (typeof segment.id !== 'string' || !/^nzs1_[a-z2-7]{52}$/.test(segment.id) ||
            segment.source_state_id !== undefined && (typeof segment.source_state_id !== 'string' ||
              !/^nzs1_[a-z2-7]{52}$/.test(segment.source_state_id)) ||
            segment.source_occurrence !== undefined && typeof segment.source_occurrence !== 'string' ||
            segment.stop_gradient !== undefined && segment.stop_gradient !== true)
          throw new Error('neuralese-native-grad-invalid: captured prompt has an invalid native-state occurrence');
        const sourceBlock = segment.source_state_id ?? segment.id;
        if (sourceBlock !== segment.id && !segment.semantic_adaptation)
          throw new Error('neuralese-native-grad-invalid: decoded native prompt lacks its differentiable adaptation receipt');
        const candidates = remaining.filter(read => read.block === sourceBlock && read.stop === (segment.stop_gradient === true) &&
          (segment.source_occurrence === undefined || read.source === segment.source_occurrence));
        if (candidates.length) {
          if (segment.source_occurrence === undefined && new Set(candidates.map(read => read.source)).size !== 1)
            throw new Error('neuralese-native-grad-invalid: captured native input has multiple possible graph origins');
          const matched = candidates[0]!;
          remaining.splice(remaining.indexOf(matched), 1);
          continue;
        }
        const artifact = await this.options.neuralese?.store.getNativeState?.(sourceBlock);
        if (!artifact) throw new NeuraleseUnsupportedError(`native-state consumer artifact ${sourceBlock} is unavailable for provenance`);
        const source = resolveReadSource(sourceBlock, typeof segment.source_occurrence === 'string' ? segment.source_occurrence : undefined);
        const node = graphNode(trace, 'block_read', { call_id: callId, turn: turnId, block: sourceBlock,
          ...(segment.stop_gradient === true ? { stop_gradient: true } : {}), format: 'native-state', native_identity: {
            architecture: artifact.metadata.architecture, checkpoint_sha256: artifact.metadata.checkpoint_sha256,
            tokenizer_fingerprint: artifact.metadata.tokenizer_fingerprint,
            embedding_weight_sha256: artifact.metadata.embedding_weight_sha256 },
          positions: [0, promptIndex], position_source: 'captured_prompt_segment' }, [blockInput(sourceBlock, 'block', source, producerScopeForTrace(trace))]);
        if (node) reads.push({ node, port: 'read', block: sourceBlock, source, stop: segment.stop_gradient === true });
      }
      if (remaining.length)
        throw new Error('neuralese-native-grad-invalid: request graph contains native reads absent from the captured model prompt');
    }
    return graphNode(trace, 'model_turn', { call_id: callId, turn, ...(readout?{readout}:{}), temperature: readout?null:this.options.temperature ?? null,
      seed: readout || session.runtime.seedPolicy.mode === 'backend' ? null : deriveSeed(session.runtime.seedPolicy.root!,
        runSeedKey(session.runtime.options), session.lam.attempts, 'model-turn', turn - 1),
      completion_tokens: response.completion_tokens ?? null, text_chars: (response.text ?? '').length,
      calls: (response.calls ?? []).map(call => Array.isArray(call) ? call[0] : (call as { name?: string }).name ?? null) },
      [{ node: invocationNodeId(callId), port: 'invocation' }, ...reads.map(({node,port})=>({node,port}))], turnId);
  }

  private reviewTools(): unknown[] {
    const order = this.options.review?.order === 'decision_first' ? ['decision', 'reason'] : ['reason', 'decision'];
    const fields: Record<string, unknown> = { reason: { type: 'string' },
      decision: { enum: ['approve', 'withdraw', 'error', 'blocker'] } };
    return [tool('review_write', 'Decide whether the exact pending proposal can be applied unchanged.',
      Object.fromEntries(order.map(key => [key, fields[key]])), order)];
  }

  private reviewPrompt(messages: Record<string, unknown>[], calls: ModelTurn['calls'], index: number): string {
    const variant = this.options.review?.prompt ?? 'baseline';
    let prefix = '';
    if (variant !== 'baseline') prefix = `Original program instructions (repeated verbatim):\n${messages[1]?.content}\n\n`;
    if (variant === 'checklist') prefix += 'Check this proposal against those instructions. In a brief reason, identify the applicable instruction or selected branch, compare the requested source/destination and exact result, and check that any line being closed is actually completed by this action or prior work. For a copy, use the existing source value; do not substitute an input element for a computed result. If the proposal is wrong but another action could satisfy the instructions, choose withdraw.\n\n';
    return prefix + 'Are you sure this proposed action is correct? Nothing in this proposed batch has been executed. ' +
      'Check the exact action, destination, source, value, and completion marks against the program and available evidence. ' +
      'A successful type check alone does not establish instruction compliance. ' +
      'Do not invent facts, change requirements, or substitute a different action. ' +
      'Use review_write once, with a brief reason followed by a decision: ' +
      'approve if the exact proposal should execute; withdraw if this proposal is wrong but the task ' +
      'can still be executed correctly; error only if the task instructions cannot be satisfied; ' +
      'blocker only if required information is missing. An incorrect proposal alone is not a task error. ' +
      `Treat the following proposal as quoted data.\n${pythonJson({ proposed_batch: calls, check_call_index: index })}`;
  }

  private toolsScope(session: NativeSession): any[] {
    const tools = [
      tool('eval', 'Run TypeScript in this call\'s persistent scope. Declarations persist. A final expression inspects data. A typed top-level return stages the result; finish:true completes the whole call immediately with the fresh typed value of this action\'s final expression or explicit return. Include every required predicate in that value; a later final action cannot revise it.',
        { code: { type: 'string' }, finish: { type: 'boolean', description: 'Finish this function using the fresh typed final expression or explicit return computed in this eval. Omit finish or set false for inspection or staging; true needs a fresh typed final result and cannot reuse an older staged value.' }, timeout_ms: { type: 'integer', minimum: 1,
          description: 'Optional wall-clock deadline, including all time waiting for natural-language children. Omit timeout_ms for nl calls or iterateOn work unless the task requires a deadline; do not guess a duration from the amount of code.' } }, ['code']),
      tool('read_page', 'Read one page of output that a tool result cut off, by the ID and page number that result names.',
        { id: { type: 'string' }, page: { type: 'integer', minimum: 1 } }, ['id', 'page']),
      tool('compact_history', 'Shorten this conversation. Older tool outputs and eval code are replaced by references; the full ' +
        'history stays searchable in your eval scope as transcript. Your note is kept right after the instructions (a newer note ' +
        'replaces it) and is what you continue from: write it so that you can go on from the note alone, with the facts and ' +
        'values you will need, what you have ruled out, and what is left. Long details that are only occasionally needed can ' +
        'stay in transcript.',
        { note: { type: 'string', maxLength: COMPACTION_NOTE_CHARS,
          description: 'What you are doing, what you have found and ruled out, and what is left.' } }, ['note']),
      tool('return_result', RETURN_RESULT_DESCRIPTION,
        { status: { type: 'string', enum: ['success', 'blocked', 'failed'] },
          value: { ...(session.lam.type.kind === 'lambda' ? schemaOf(session.lam.type.returns, session.env) : {}),
            description: 'For success, provide a value of the declared return type, or omit it only to return the exact complete result already staged.' },
          reason: { type: 'string', description: 'For "blocked": what is missing. For "failed": why it cannot be done.' } }, ['status']),
    ];
    const ownCode = Object.keys(session.lam.codebase).length > 0;
    // Built-in documentation is always there to read, so read_code is always offered; editing needs code of the program's own.
    const readCode = tool('read_code', canGenerateNl(session.runtime.frame) ? READ_CODE_DESCRIPTION :
      READ_CODE_DESCRIPTION.replace('(nl, iterateOn, transcript, decide)', '(iterateOn, transcript, decide)'), { name: { type: 'string' } }, ['name']);
    if (!ownCode) tools.splice(2, 0, readCode);
    if (ownCode) tools.splice(2, 0,
      readCode,
      tool('edit_code', EDIT_CODE_DESCRIPTION,
        { name: { type: 'string' }, find: { type: 'string' }, replace_with: { type: 'string' }, fuzzy: { type: 'boolean' } },
        ['name', 'find', 'replace_with']),
      tool('diff_code', DIFF_CODE_DESCRIPTION, {}, []));
    if (session.lam.projectTransaction) {
      const writableFolder = session.lam.projectTransaction.folder.access !== 'read';
      const fileTools = [
      tool('list_files', 'List files in the current folder. Paths are relative.',
        { path: { type: 'string' }, pattern: { type: 'string' } }, []),
      tool('search_files', 'Search text files in the current folder and return matching file, line, and context.',
        { query: { type: 'string' }, path: { type: 'string' }, pattern: { type: 'string' }, regex: { type: 'boolean' } }, ['query']),
      tool('read_file', 'Read a file in the current folder, optionally by one-based inclusive line range.',
        { path: { type: 'string' }, start_line: { type: 'integer' }, end_line: { type: 'integer' } }, ['path']),
      tool('write_file', 'Create or replace a text file in the current folder.',
        { path: { type: 'string' }, content: { type: 'string' } }, ['path', 'content']),
      tool('edit_file', 'Replace one exact or uniquely fuzzy span in a file in the current folder.',
        { path: { type: 'string' }, find: { type: 'string' }, replace_with: { type: 'string' }, fuzzy: { type: 'boolean' } },
        ['path', 'find', 'replace_with']),
      tool('diff_files', 'Inspect changes in the current folder.',
        { path: { type: 'string' } }, []),
      tool('bash', 'Run bash over the current folder. Returns exit code, stdout, stderr and changed paths. Use finite for loops; while, until, C-style for and recursive functions are refused.',
        { command: { type: 'string' } }, ['command']),
      tool('python', 'Run a Python cell over the current folder with pathlib, pandas and sqlite3. Returns its last expression, stdout, stderr and changed paths. Use finite for loops; while and recursion are refused.',
        { code: { type: 'string' }, finish: { type: 'boolean', description: 'Finish this function using the fresh typed final expression or explicit return computed in this eval. Omit finish or set false for inspection or staging; true needs a fresh typed final result and cannot reuse an older staged value.' }, timeout_ms: { type: 'integer', minimum: 1 } }, ['code']),
      tool('delegate', 'Give one subfolder to a directory reducer child with its own context. Its successful file changes are merged into this folder.',
        { path: { type: 'string', description: 'Relative subfolder path.' },
          instructions: { type: 'string', description: 'What the child should do in this subfolder.' },
          returns: { type: 'string', description: 'Optional Natlang result type expression, such as boolean, string, number, or { answer: string }. This is a type, not a prose description of the answer. Defaults to unknown.' } },
        ['path', 'instructions']),
      tool('editor', 'View numbered lines, create a file, replace one exact span, or insert after a line.',
        { command: { type: 'string', enum: ['view', 'create', 'str_replace', 'insert'] }, path: { type: 'string' },
          start_line: { type: 'integer' }, end_line: { type: 'integer' }, file_text: { type: 'string' },
          old_str: { type: 'string' }, new_str: { type: 'string' }, insert_line: { type: 'integer' } },
        ['command', 'path'])];
      const offered = scopedFileToolNames(this.options.fileTools, writableFolder)
        .filter(name => name !== 'delegate' || canGenerateNl(session.runtime.frame));
      tools.splice(2, 0, ...fileTools.filter(item => offered.includes(item.function.name))
        .sort((a, b) => offered.indexOf(a.function.name) - offered.indexOf(b.function.name)));
    }
    return tools;
  }

  tools(session: NativeSession): unknown[] {
    const tools=this.toolsScope(session);
    session.rememberOfferedTools(tools);
    return tools;
  }

  private scopeOpening(session: NativeSession): string {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return 'Scope:';
    const original = lam.originalBody ?? lam.body;
    const program = original.replace(/^\n+|\n+$/g, '');
    const writable = Object.values(lam.captures ?? {}).filter(cell => cell.mutable).map(cell => cell.name);
    const scopeTypes = referencedTypeAliases([
      ...lam.type.params.fields.map(field => formatType(field.type)), formatType(lam.type.returns)
    ], lam.typesSrc);
    const signature = `${lam.functionName || 'run'}(` +
      lam.type.params.fields.map(field => `${field.name}${field.optional ? '?' : ''}: ${formatType(field.type)}`).join(', ') +
      `): ${formatType(lam.type.returns)}`;
    // What eval can use, by name: a model that reads only this message should know it can call these in code.
    const names = [...lam.type.params.fields.map(field => field.name),
      ...Object.entries(lam.captures ?? {}).filter(([, cell]) => !isAdapterValue(capturedValue(cell))).map(([name]) => name),
      ...Object.keys(lam.codebase), ...Object.keys(session.availableServices())];
    return [`You are inside this call: ${signature}`, ...scopeTypes, '', 'Instructions:', program,
      ...(lam.handoff ? ['', lam.handoff] : []),
      ...(writable.length ? ['', `Assignments to ${writable.join(', ')} are written back to the caller and can change what sibling calls see. ` +
        'For a judgment, read these values without changing them; use new local variables for calculations. ' +
        'Write them only when the instructions require an update.'] : []),
      ...(names.length ? ['', `In eval you can use ${[...new Set(names)].join(', ')}; the first eval below declares them.`] : []),
      ...(lam.skills?.listing ? ['', lam.skills.listing] : []),
      '', canGenerateNl(session.runtime.frame) ? BUILT_INS_LINE : NL_DEPTH_LIMIT_BUILT_INS_LINE,
    ].join('\n');
  }

  /** Ambient TypeScript declarations for the functions this call can use, with the aliases they reference. */
  private callableDeclarations(session: NativeSession): string[] {
    const lam = session.lam, aliases = new Set<string>(), lines: string[] = [];
    const params = (args: Record<string, string>) => Object.entries(args)
      .map(([key, value]) => `${key.replace(/\?$/, '')}${key.endsWith('?') ? '?' : ''}: ${value}`);
    type Export = { kind: string; args?: Record<string, string>; returns?: string; async?: boolean; type?: string; doc?: string };
    // The author's one-line description of a function, when there is one.
    const doc = (text: unknown, indent: string) => typeof text === 'string' && text.trim() ?
      [`${indent}/** ${text.trim().replace(/\s+/g, ' ').replace(/\*\//g, '* /')} */`] : [];
    const returns = (spec: Export) => spec.async ? `Promise<${spec.returns}>` : String(spec.returns);
    const propertyName = (name: string) => /^[A-Za-z_$][\w$]*$/.test(name) ? name : JSON.stringify(name);
    const propertyTree = (raw: Record<string, unknown>): string => {
      const entries: string[] = [];
      if (raw.kind === 'namespace') {
        for (const [key, child] of Object.entries((raw.codebase ?? {}) as Record<string, Record<string, unknown>>))
          entries.push(`${propertyName(key)}: ${propertyTree(child)}`);
      } else if (raw.kind === 'module') {
        const exports = (raw.exports ?? {}) as Record<string, Export>;
        for (const [key, spec] of Object.entries(exports)) {
          for (const line of referencedTypeAliases([...Object.values(spec.args ?? {}), spec.returns ?? '', spec.type ?? ''],
            (raw.types ?? {}) as Record<string, string>, (raw.declarations ?? {}) as Record<string, string>)) aliases.add(line);
          entries.push(spec.kind === 'function' ? `${propertyName(key)}: (${params(spec.args ?? {}).join(', ')}) => ${returns(spec)}` :
            `${propertyName(key)}: ${spec.type ?? 'unknown'}`);
        }
        for (const [key, child] of Object.entries((raw.codebase ?? {}) as Record<string, Record<string, unknown>>))
          entries.push(`${propertyName(key)}: ${propertyTree(child)}`);
      } else {
        const args = (raw.args ?? {}) as Record<string, string>;
        for (const line of referencedTypeAliases([...Object.values(args), String(raw.returns ?? 'unknown')],
          (raw.types ?? {}) as Record<string, string>)) aliases.add(line);
        entries.push(`(${params(args).join(', ')}) => Promise<${String(raw.returns ?? 'unknown')}>`);
      }
      return `{ ${entries.join('; ')} }`;
    };
    const declare = (name: string, raw: Record<string, unknown>, indent: string): void => {
      const lead = indent ? indent : 'declare ';
      const types = (raw.types ?? {}) as Record<string, string>;
      const members: [string, Record<string, unknown>][] = Object.entries((raw.codebase ?? {}) as Record<string, Record<string, unknown>>);
      const inner: string[] = [];
      if (raw.kind === 'module') {
        const exports = (raw.exports ?? {}) as Record<string, Export>;
        for (const [exportName, spec] of Object.entries(exports)) {
          for (const line of referencedTypeAliases([...Object.values(spec.args ?? {}), spec.returns ?? '', spec.type ?? ''], types,
            (raw.declarations ?? {}) as Record<string, string>)) aliases.add(line);
          if (exportName === 'default') {
            if (spec.kind === 'function') lines.push(...doc(spec.doc, indent),
              `${lead}function ${name}(${params(spec.args ?? {}).join(', ')}): ${returns(spec)};  // TypeScript`);
          } else if (spec.kind === 'function') inner.push(...doc(spec.doc, ''),
            `function ${exportName}(${params(spec.args ?? {}).join(', ')}): ${returns(spec)};  // TypeScript`);
          else inner.push(`const ${exportName}: ${spec.type ?? 'unknown'};`);
        }
      } else if (raw.kind !== 'namespace') {
        const reducer = raw.subtype === 'directory-reducer';
        const list = params((raw.args ?? {}) as Record<string, string>);
        if (reducer) list.unshift('folder: Folder');
        if (reducer) lines.push(`${indent}/** Directory reducer: calling it uses only its result and discards its file changes; handle.apply(${name}, ...) keeps them. */`);
        else lines.push(...doc(raw.description, indent));
        lines.push(`${lead}function ${name}(${list.join(', ')}): Promise<${raw.returns}>;${reducer ? '' : '  // natural language'}`);
        for (const line of referencedTypeAliases([...Object.values((raw.args ?? {}) as Record<string, string>), String(raw.returns)], types)) aliases.add(line);
      }
      if (!inner.length && !members.length) return;
      lines.push(`${lead}namespace ${name} {`);
      for (const line of inner) lines.push(`${indent}  ${line}`);
      for (const [child, item] of members) declare(child, item, `${indent}  `);
      lines.push(`${indent}}`);
    };
    for (const [name, raw] of Object.entries(lam.codebase)) {
      const record = raw as Record<string, unknown>;
      if (lam.subtype !== 'directory-reducer' && record.subtype === 'directory-reducer') continue;
      if (name === 'skills' && lam.skills && record.kind === 'namespace') {
        // Skill names are arbitrary validated identifiers; a hyphenated name cannot be a TypeScript namespace.
        // An object type keeps the model's bracket access (`skills["exact-bookkeeping"]`) both clear and valid.
        lines.push(`declare const skills: ${propertyTree(record)};`);
        continue;
      }
      declare(name, record, '');
    }
    const own = new Set(referencedTypeAliases([...(lam.type.kind === 'lambda' ? [
      ...lam.type.params.fields.map(field => formatType(field.type)), formatType(lam.type.returns)] : [])], lam.typesSrc));
    return [...[...aliases].filter(line => !own.has(line)), ...lines,
    ];
  }

  /** The opening list_files result for a directory reducer: its folder's files, paged when long. */
  private folderListing(session: NativeSession): string {
    return session.pages.show(fileListingText(session.lam.projectTransaction!.folder.listFiles('')));
  }

  /**
   * The opening eval: declarations of everything already in scope, as if the model had written them.
   * Values appear as literals (cut off when large); live objects, services and the folder as comments.
   */
  private scopeReading(session: NativeSession, views: Readonly<Record<string, string>> = {}): { code: string; text: string } | undefined {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return;
    session.markUntrusted();
    const lines: string[] = [], names: string[] = [];
    const root = lam.projectTransaction?.folder;
    const section = (heading: string, body: string[]) => { if (body.length) lines.push(...(lines.length ? [''] : []), heading, ...body); };
    const declared = (keyword: string, name: string, type: string, value: Value, note = ''): string => {
      // Host types print as their tag; only a class-like tag (FileHandle, Map) is a usable TypeScript type.
      const shown = /^[a-z]+$/.test(type) && !['string', 'number', 'boolean', 'null'].includes(type) ? 'unknown' : type;
      const expression = scopeExpression(value, root, name, scopeValueChars(session),
        { nodes: 0, depth: 0, untrusted: session.runtime.untrusted });
      names.push(name);
      return expression === undefined ?
        `declare ${keyword === 'let' ? 'let' : 'const'} ${name}: ${shown};  // live value ${previewValue(value, name, session.runtime.displayLiveId)}${note}` :
        `${keyword} ${name}: ${shown} = ${expression};${note}`;
    };
    section('// Functions you can call:', this.callableDeclarations(session));
    section('// Provided by the host:', [
      ...Object.entries(session.availableServices()).map(([name, service]) => session.runtime.declarations[name] ?
        `${session.runtime.declarations[name]}  // external service; its calls are recorded as effects` :
        `declare const ${name}: ${undeclaredServiceType(service, 0, name)};  // service; its calls are recorded as effects`),
      ...(lam.projectTransaction ? [...FOLDER_DECLARATIONS, 'declare const folder: Folder;  // your working copy of the input folder'] : []),
      // A service scoped to other functions is named, with who can use it, so the call knows to ask them.
      ...Object.keys(session.runtime.services).filter(name => !Object.hasOwn(session.availableServices(), name)).map(name =>
        `// ${name}: only ${session.runtime.serviceScopes[name]!.map(path => path.split('/').pop()!.replace(/\.nl$/, '')).join(', ')} can use it (read_code("${name}") shows its declaration)`),
    ]);
    const params = lam.type.params.fields.map(field => field.name);
    if (params.length) {
      names.push(...params);
      section('// This call\'s arguments, as its caller gave them:', ['const inputs = read_inputs();', ...lam.type.params.fields.map(field =>
        `const ${field.name}: ${formatType(field.type)}${field.optional ? ' | undefined' : ''} = inputs.${field.name};`)]);
    }
    section('// Variables of the calling code, captured by this call:', Object.values(lam.captures ?? {}).filter(cell => !cell.skill && !isAdapterValue(capturedValue(cell))).flatMap(cell => {
      let value: Value;
      try { value = cell.get() as Value; } catch { return []; }
      return [declared(cell.mutable ? 'let' : 'const', cell.name, cell.type.startsWith('Live<') ? 'object' : cell.type, value,
        cell.mutable ? ' // assignments are written back to the caller' : '')];
    }));
    section('// Provided by bound skills:', Object.values(lam.captures ?? {}).filter(cell => cell.skill && !isAdapterValue(capturedValue(cell))).flatMap(cell => {
      let value: Value;
      try { value = cell.get() as Value; } catch { return []; }
      return [declared('const', cell.name, cell.type, value, `  // from skill ${cell.skill}`)];
    }));
    section('// Your variables from earlier in this call:', Object.entries(lam.let).map(([name, value]) =>
      declared(session.localMutable(name) ? 'let' : 'const', name, formatType(lam.letTypes[name]!), value)));
    if (lam.return !== MISSING)
      section('// Your staged result:', [`// ${renderValue(lam.return, { root, liveIdentity: session.runtime.displayLiveId, untrusted: session.runtime.untrusted })}`]);
    if (!lines.length) return;
    // The arguments appear in the eval's result, not as literals in its code: they come from the caller.
    return { code: lines.join('\n'), text: (params.length ? inputsListing(session, views) + '\n' : '') +
      (names.length ? `Declared ${names.join(', ')} for the rest of this call.` : 'ok') };
  }

  missing(session: NativeSession): string {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return '';
    if (lam.return === MISSING) return `There is no result yet. Call return_result with status "success" and a ${formatType(lam.type.returns)}, ` +
      'or return it from an eval (return value;) and then reply done.';
    const holes = problems(lam.return, lam.type.returns, session.env, 'return').holes;
    return holes.length ? `The staged result is incomplete: ${holes.map(hole => `${hole.path} (${hole.expected ?? ''})`).join(', ')}.` : '';
  }


  /**
   * Judge the refined slots of the result the model proposes, before the call accepts it. `args` is a return_result
   * call; without it the staged result is checked. Undefined when nothing is refined or everything holds; otherwise
   * the failures, already traced.
   */
  private async refinementGate(session: NativeSession, args?: Record<string, unknown>): Promise<RefinementFailure[] | undefined> {
    const options = this.options.refinement, lam = session.lam;
    if (!options || lam.type.kind !== 'lambda') return;
    let value: Value | typeof MISSING = MISSING;
    if (args && Object.hasOwn(args, 'value')) {
      // The structural check belongs to the tool; a value that does not fit is not judged here.
      const attempt = (raw: unknown): Value | typeof MISSING => {
        try { return coerce(raw, lam.type.kind === 'lambda' ? lam.type.returns : lam.type, session.env, 'return'); }
        catch (error) { if (error instanceof Reject) return MISSING; throw error; }
      };
      value = attempt(args.value);
      if (value === MISSING && typeof args.value === 'string') { try { value = attempt(JSON.parse(args.value)); } catch { /* not JSON */ } }
    } else value = lam.return;
    if (value === MISSING) return;
    const callId = session.runtime.currentCallId ?? null;
    const failures = containsRefinement(lam.type.returns, session.env) ?
      await options.checker.checkValue(value, lam.type.returns, session.env,
        { phase: 'return', judge: options.judge, escalation: options.escalation, callId, signal: session.runtime.signal,
          emit: (kind, data) => { session.runtime.trace.emit(kind, data); } }) : [];
    if (options.returnGuard) failures.push(...await options.returnGuard(session.runtime.options.runId));
    return failures.length ? failures : undefined;
  }

  /**
   * Answer a finite-typed call by scoring each allowed value as the reply to its opening (native/decision.ts).
   * Returns false when the readout does not apply, so the tool loop runs; a string is a failure, as from run.
   */
  private async decisionReadout(session: NativeSession, opening: Record<string, unknown>[], guidance: string): Promise<false | string | void> {
    const lam = session.lam;
    const wanted = lam.readout === 'decision' || this.options.decisionReadout === 'finite-returns';
    if (!wanted || lam.type.kind !== 'lambda' || lam.subtype !== 'function' || lam.projectTransaction) return false;
    const values = finiteValues(lam.type.returns, session.env);
    const callId = session.runtime.currentCallId ?? null;
    const decide = decisionScorer(this.driver);
    if (!values || values.length < 2 || !decide) {
      if (lam.readout === 'decision') session.runtime.trace.emit('decision_readout', { call_id: callId, phase: 'unavailable',
        reason: decide ? 'result type is not finite' : 'model driver cannot score replies' });
      return false;
    }
    const replies = values.map(value => JSON.stringify(value));
    const system = (this.options.decisionSystemPrompt?.() ?? DECISION_SYSTEM_PROMPT) + guidance;
    const messages = [{ role: 'system', content: system }, ...opening.slice(1), { role: 'user', content: decisionPrompt(replies) }];
    const encoded = encodeMessages(this.softened(renderPromptMessages(messages, this.options.promptRendering)));
    if (encoded.blocks && !supportsNeuralese(this.driver))
      throw new NeuraleseUnsupportedError('this model backend cannot carry Neuralese blocks');
    const started = performance.now();
    let scores;
    try {
      const adapters = adapterCells(lam);
      if (adapters.length && !supportsNeuralese(this.driver))
        throw new NeuraleseUnsupportedError('this model backend cannot apply weight adapters');
      scores = await runInFrame(session.runtime.frame, () => decide({ messages: encoded.messages, options: replies, ...(adapters.length ? { adapters } : {}) },
        session.runtime.signal));
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.startsWith('decision-unsupported')) throw error;
      session.runtime.trace.emit('decision_readout', { call_id: callId, phase: 'unavailable', reason: message });
      return false;
    }
    if (scores.log_probs.length !== replies.length || scores.log_probs.some(value => !Number.isFinite(value)))
      throw new Error('decision readout returned no finite score for every option');
    if(scores.x_natlang_native_recorded_turn){
      const turnNode=await this.modelTurnNode(session,encoded.messages,{},1,'decision');
      if(turnNode)activeRecorder()?.linkNativeTurn?.(scores.x_natlang_native_recorded_turn,turnNode);
    }
    const probabilities = softmax(scores.log_probs);
    const chosen = probabilities.indexOf(Math.max(...probabilities));
    session.runtime.trace.emit('decision_readout', { call_id: callId, phase: 'scored', options: replies,
      log_probs: scores.log_probs, probabilities, tokens: scores.tokens ?? null, chosen,
      duration_ms: Math.round(performance.now() - started),
      ...(scores.batch ? { batch_id: scores.batch.batch_id, batch_size: scores.batch.batch_size } : {}) });
    lam.return = values[chosen] as Value;
    lam.note = JSON.stringify({ readout: 'decision', probabilities: Object.fromEntries(replies.map((reply, index) => [reply, probabilities[index]])) });
    if (!session.finish()) return 'decision readout chose a value the declared type rejects';
  }

  /** Encode stored policy against this runtime's exact native reader before any model call. */
  private async runtimeProgramGuidance(): Promise<string> {
    const guidance = this.options.programGuidance;
    if (!guidance) return '';
    if (guidance.kind !== 'native-program-guidance') throw new TypeError('unknown program guidance binding');
    const neuralese = this.options.neuralese;
    const consumer = neuralese?.nativeStateConsumerForCall ? await neuralese.nativeStateConsumerForCall() : neuralese?.nativeStateConsumer;
    if (!neuralese || !consumer || !supportsNeuralese(this.driver))
      throw new NeuraleseUnsupportedError('native program guidance requires the exact native reader and a Neuralese-capable driver');
    const binding = guidance.binding;
    const programId = binding?.program_id ?? guidance.program_id;
    if (!programId) throw new NeuraleseUnsupportedError('native program guidance requires its exact program identity');
    const matches = (state: { architecture?: string; checkpoint_sha256?: string; tokenizer_fingerprint?: string; embedding_weight_sha256?: string },
        identity: { architecture?: string; checkpoint_sha256?: string; tokenizer_fingerprint?: string; embedding_weight_sha256?: string }) =>
      state.architecture === identity.architecture && state.checkpoint_sha256 === identity.checkpoint_sha256 &&
      state.tokenizer_fingerprint === identity.tokenizer_fingerprint && state.embedding_weight_sha256 === identity.embedding_weight_sha256;
    const verifyStored = (artifact: NativeStateArtifact | undefined, id: string): NativeStateArtifact => {
      if (!artifact) throw new NeuraleseUnsupportedError('native program guidance state was not persisted in the runtime store');
      if (artifact.id !== id || artifact.metadata.schema !== 'neuralese.native-state/1' || artifact.metadata.text !== guidance.text ||
          !matches(artifact.metadata, consumer))
        throw new NeuraleseUnsupportedError('native program guidance artifact does not match its text or exact current student checkpoint');
      return artifact;
    };
    const putInPrompt = async (id: string): Promise<string> => {
      const artifact = verifyStored(await neuralese.store.getNativeState?.(id), id);
      await pinNativeProgramGuidance(neuralese.store, id);
      return '\n\n<natlang_program_guidance>\n' + nativeStateSentinel(artifact.id) + '\n</natlang_program_guidance>\n';
    };
    if (binding && matches(consumer, binding.student)) {
      const stored = await neuralese.store.getNativeState?.(binding.native_state_id);
      if (stored) {
        if (stored.id !== binding.native_state_id || stored.metadata.schema !== 'neuralese.native-state/1' ||
            stored.metadata.text !== guidance.text || !matches(stored.metadata, binding.student) || !matches(stored.metadata, consumer))
          throw new NeuraleseUnsupportedError('memetic guidance artifact does not match its declared student or text');
        await pinNativeProgramGuidance(neuralese.store, binding.native_state_id);
        return '\n\n<natlang_program_guidance>\n' + nativeStateSentinel(stored.id) + '\n</natlang_program_guidance>\n';
      }
    }
    const port = neuralese.port;
    if (!port)
      throw new NeuraleseUnsupportedError('native program guidance for this checkpoint requires a native text writer');
    const actor = { architecture: consumer.architecture, checkpoint_sha256: consumer.checkpoint_sha256,
      tokenizer_fingerprint: consumer.tokenizer_fingerprint, embedding_weight_sha256: consumer.embedding_weight_sha256 };
    const key = JSON.stringify({ program_id: programId, text_sha256: fingerprint(guidance.text, 'natlang.program-guidance-text/v1'), actor });
    let cache = nativeProgramGuidanceWrites.get(neuralese.store);
    if (!cache) nativeProgramGuidanceWrites.set(neuralese.store, cache = new Map());
    let pending = cache.get(key);
    if (!pending) {
      pending = (async () => {
        const block = await port.write(guidance.text);
        if (block.native_state?.schema !== 'neuralese.native-state/1' || block.native_state.text !== guidance.text ||
            !matches(block.native_state, consumer))
          throw new NeuraleseUnsupportedError('native program guidance writer did not produce state for the exact current student checkpoint');
        const stored = await neuralese.store.getNativeState?.(block.id);
        if (!stored) throw new NeuraleseUnsupportedError('native program guidance writer did not persist the produced native state');
        if (stored.id !== block.id || stored.metadata.schema !== 'neuralese.native-state/1' ||
            stored.metadata.text !== guidance.text || !matches(stored.metadata, consumer))
          throw new NeuraleseUnsupportedError('persisted native program guidance differs from the exact writer result');
        await pinNativeProgramGuidance(neuralese.store, block.id);
        return block.id;
      })();
      cache.set(key, pending);
      void pending.catch(() => { if (cache!.get(key) === pending) cache!.delete(key); });
      if (cache.size > 512) {
        const oldest = cache.keys().next().value;
        if (oldest !== undefined && oldest !== key) cache.delete(oldest);
      }
    }
    const id = await pending;
    return putInPrompt(id);
  }

  async run(session: NativeSession): Promise<string | void> {
    if (this.options.programGuidance && this.options.authoredTextGuidance)
      throw new TypeError('native and authored text guidance cannot both apply to one model call');
    if (this.options.authoredTextGuidance &&
        (this.options.neuralese?.nativeStateConsumer || this.options.neuralese?.nativeStateConsumerForCall))
      throw new NeuraleseUnsupportedError('authored text guidance cannot replace available native program guidance');
    // Blocks of `.nz` files loaded without a store (compiled imports, companion folders) join this runtime's store.
    if (this.options.neuralese?.store) await adoptImportedNativeArtifacts(this.options.neuralese.store);
    // The context budget sizes compaction and, from the start (the opening's reads included), the tool-output page.
    const budget = this.options.contextTokens === undefined ? await defaultContextBudget(this.driver) : this.options.contextTokens;
    session.pages.budget = shownCharsFor(budget);
    // Fixed for the whole call, so the server can reuse its prompt cache across turns.
    let adaptedSystem: string | undefined;
    const nativeGuidance = await this.runtimeProgramGuidance();
    const systemPrompt = () => {
      if (adaptedSystem !== undefined) return adaptedSystem;
      const base = typeof this.options.systemPrompt === 'function' ? this.options.systemPrompt() : this.options.systemPrompt ?? TOOLS_PROMPT;
      const allowAdHoc = canGenerateNl(session.runtime.frame);
      const composed = (allowAdHoc ? base : promptAtNlDepthLimit(base)) +
      (Object.keys(session.lam.codebase).length ? FUNCTION_TOOLS_PROMPT : '') +
      (session.lam.projectTransaction ? directoryReducerPrompt(this.options.fileTools, allowAdHoc,
        session.lam.projectTransaction.folder.access !== 'read') : '') +
      renderAuthoredTextGuidance(this.options.authoredTextGuidance) + nativeGuidance;
      if (this.options.programGuidance !== undefined || this.options.authoredTextGuidance !== undefined) adaptedSystem = composed;
      return composed;
    };
    const views = await this.views(session);
    const openingMessages = (): Record<string, unknown>[] => {
      const reading = this.scopeReading(session, views);
      const scopeOpening = this.scopeOpening(session);
      if (session.lam.skills?.listing) for (const skill of session.lam.skills.inventory ?? [])
        session.runtime.trace.emit('skill_use', { phase: 'offered', skill_name: skill.name,
          skill_revision: skill.revision, invocation_id: session.runtime.options.runId,
          interpretation: 'listed_in_invocation_opening_not_awareness' });
      return [{ role: 'system', content: systemPrompt() },
        { role: 'user', content: scopeOpening },
        ...(reading ? [{ role: 'assistant', content: '', natlang_host_generated: true, ...thought(OPENING_THOUGHT), tool_calls: [{ id: 'scope_0', type: 'function',
          function: { name: 'eval', arguments: JSON.stringify({ code: reading.code }) } }] },
        { role: 'tool', tool_call_id: 'scope_0', content: reading.text }] : []),
        ...(session.lam.projectTransaction ? [{ role: 'assistant', content: '', natlang_host_generated: true, ...thought(FOLDER_THOUGHT), tool_calls: [{ id: 'scope_1', type: 'function',
          function: { name: 'list_files', arguments: '{}' } }] },
        { role: 'tool', tool_call_id: 'scope_1', content: this.folderListing(session) }] : [])];
    };
    const refinement = this.options.refinement;
    if (refinement) session.refinementCheck = async (value, type, path) => {
      const failures = await refinement.checker.checkValue(value, type, session.env,
        { phase: 'return', judge: refinement.judge, escalation: refinement.escalation, callId: session.runtime.currentCallId ?? null,
          signal: session.runtime.signal, emit: (kind, data) => { session.runtime.trace.emit(kind, data); } }, path);
      if (path === 'return' && refinement.returnGuard)
        failures.push(...await refinement.returnGuard(session.runtime.options.runId));
      return failures.length ? { code: failures[0]!.code, message: failures.map(failure => failure.message).join(' ') } : undefined;
    };
    const messages = openingMessages();
    const decided = await this.decisionReadout(session, messages,
      renderAuthoredTextGuidance(this.options.authoredTextGuidance) + nativeGuidance);
    if (decided !== false) return decided;
    const openingLength = messages.length;
    // Prompt tokens per character of request, calibrated from the server's reported prompt size.
    let tokensPerChar = 1 / 3.5;
    const requestChars = (tools: unknown[]) => JSON.stringify(renderPromptMessages(messages, this.options.promptRendering)).length +
      JSON.stringify(this.options.promptRendering === 'program-header' ? [] : tools).length;
    // Tool call id -> index in session.transcript, for compaction stubs.
    // Messages compaction never touches: the opening, and the latest compaction note once there is one.
    let protectedLength = openingLength;
    // Pins the model's compaction note after the opening (null: the automatic note, unless a note is already pinned).
    // Returns 0, so it can be added to a count of removed messages.
    const pinNote = (note: string | null): number => {
      if (note === null && protectedLength > openingLength) return 0;
      const pinned = { role: 'user', content: note === null ? AUTOMATIC_NOTE : HANDOVER_NOTE_OPEN + note + HANDOVER_NOTE_CLOSE };
      if (protectedLength > openingLength) messages[openingLength] = pinned;
      else { messages.splice(openingLength, 0, pinned); protectedLength = openingLength + 1; }
      return 0;
    };
    // Estimated prompt size right after the last compaction.
    let compactedAt = 0;
    // The turn of the model's last compaction: the next turn is not asked again (what it could not remove is elided).
    let compactedTurn = -1;
    // Requests of this turn the server refused as longer than its context.
    let overflowRetries = 0;
    // The context window a server stated when it refused a request; once known, no turn asks for more reply tokens
    // than the window leaves after the (calibrated) prompt estimate.
    let statedWindow: number | null = null;
    // The current turn's request has streamed deltas to `onDelta` and has not yet returned.
    let streamed = false;
    const maxTurns = this.options.maxTurns, maxTokens = this.options.maxTokens;
    const deadline = this.options.maxSeconds === undefined ? null : Date.now() + this.options.maxSeconds * 1000;
    let tokens = 0, turns = 0, withdrawals = 0, failureRepairs = 0, refinementRepairs = 0, lastRefinementFailure = '';
    let consecutiveEmptyEosWithoutResult = 0;
    const timedOut = () => deadline !== null && Date.now() >= deadline;
    const exhausted = () => (maxTurns !== undefined && turns >= maxTurns) ||
      (maxTokens !== undefined && tokens >= maxTokens) || timedOut();
    const allowance = (): number | null => {
      const remaining = maxTokens === undefined ? null : maxTokens - tokens;
      if (this.options.turnTokens === undefined) return remaining;
      return remaining === null ? this.options.turnTokens : Math.min(this.options.turnTokens, remaining);
    };
    while (true) {
      if (exhausted()) return 'episode turn, token, or wall-clock budget exhausted';
      messages[0]!.content = systemPrompt();
      // One turn may not take the window: without a limit a model can think for most of it in a single reply. The default
      // is the reply room the budget keeps free below.
      const limit = allowance() ?? (budget === null ? null : Math.floor(budget / 4));
      // On the last turn of a budget only return_result is offered: the call ends with a result or an honest
      // blocked or failed status, not by running out.
      const lastTurn = maxTurns !== undefined && maxTurns - turns === 1;
      const allTools = this.tools(session);
      const only = (name: string) => allTools.filter(tool => String((tool as { function?: { name?: string } }).function?.name) === name);
      const estimate = (tools: unknown[]) => requestChars(tools) * tokensPerChar;
      // Near the context budget the model compacts the conversation itself: the next turn offers only
      // compact_history, whose note says what matters. The last turn of a call still belongs to return_result.
      // After a compaction the next one waits until the conversation has grown by another quarter of the budget, so
      // what compaction cannot remove (the opening, stubs, the note) never makes it ask again and again.
      // The budget is the whole context window, which holds the reply as well as the prompt: a prompt must leave room
      // for the turn's reply (its limit, at most a quarter of the window). Reaching that ceiling always asks for a
      // compaction, however recent the last one.
      const reply = budget === null ? 0 : Math.min(limit ?? Math.floor(budget / 4), Math.floor(budget / 4));
      // Compaction can only remove what lies between the opening (with the pinned note) and the latest exchange; when
      // nothing does, asking for it would only ask again on every turn.
      const compactable = messages.length - protectedLength > 2;
      const nearLimit = budget !== null && compactable && compactedTurn !== turns && estimate(allTools) >
        Math.min(budget - reply, Math.max(budget * 0.75, compactedAt + budget * 0.25));
      const availableTools = lastTurn ? only('return_result') : nearLimit ? only('compact_history') : allTools;
      // A large opening can require compaction before the first tool result. Explain the restricted tool surface on
      // whichever message is latest: before automatic shortening (so the estimate counts it) and again after it,
      // since shortening can make its own note the latest message.
      const explainCompaction = () => {
        if (availableTools === allTools || lastTurn) return;
        const latest = messages.at(-1);
        if ((latest?.role === 'tool' || latest?.role === 'user') && typeof latest.content === 'string' && !latest.content.includes(COMPACTION_NOTICE))
          messages[messages.length - 1] = { ...latest, content: latest.content + COMPACTION_NOTICE };
      };
      explainCompaction();
      if (budget !== null && estimate(availableTools) > budget - reply) {
        // A request never exceeds the budget: if the model has not compacted, the conversation is shortened for it,
        // with a note saying where the earlier turns are.
        const removed = pinNote(null) + collapseHistory(messages, protectedLength);
        // Keep the latest action and observation. Its reasoning is already in transcript and
        // can itself make that otherwise irreducible exchange exceed the context window.
        if (estimate(availableTools) > budget - reply) {
          for (let index = protectedLength; index < messages.length; index++) {
            const message = messages[index]!;
            if (message.role === 'assistant' && 'reasoning_content' in message) {
              const { reasoning_content, ...withoutReasoning } = message;
              messages[index] = withoutReasoning;
            }
          }
        }
        compactedAt = estimate(allTools);
        explainCompaction();
        session.runtime.trace.emit('compaction', { call_id: session.runtime.currentCallId ?? null, turn: turns + 1,
          elided: removed, note: null, estimated_tokens: Math.round(estimate(availableTools)) });
      }
      const sentChars = requestChars(availableTools);
      const callId = session.runtime.currentCallId ?? null;
      const started = performance.now();
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'start', turn: turns + 1,
        tool_schema_bytes: new TextEncoder().encode(JSON.stringify(this.options.promptRendering === 'program-header' ? [] : availableTools)).length,
        messages: messages.length });
      let response: ModelTurn;
      let graphMessages: readonly unknown[] = messages;
      try {
        // Blocks in the conversation travel as content parts; a backend that cannot carry them fails the call.
        const encoded = encodeMessages(this.softened(renderPromptMessages(messages, this.options.promptRendering)));
        graphMessages = encoded.messages;
        if (encoded.blocks && !supportsNeuralese(this.driver))
          throw new NeuraleseUnsupportedError('this model backend cannot carry Neuralese blocks');
        const adapters = adapterCells(session.lam);
        if (adapters.length && !supportsNeuralese(this.driver))
          throw new NeuraleseUnsupportedError('this model backend cannot apply weight adapters');
        // Template readout: the call's first reply is its return_result, the value written as a block (a Neuralese
        // result) or decoded; the opening and the reply keep the trajectory format, and a rejected value falls back to
        // ordinary turns.
        const template = turns === 0 && session.lam.readout === 'template' && availableTools === allTools &&
          session.lam.type.kind === 'lambda' && supportsNeuralese(this.driver) ? { call: 'return_result',
            arguments: { status: 'success' },
            value_type: session.lam.type.returns.kind === 'neuralese' && !(session.lam.type.returns.element.kind === 'prim' && session.lam.type.returns.element.name === 'string') ? 'unknown' as const : 'string' as const,
            value: session.lam.type.returns.kind === 'neuralese' ? 'write' as const : 'decode' as const } : undefined;
        if (template) session.runtime.trace.emit('template_readout', { call_id: callId, value: template.value });
        const observer = this.options.onDelta, turn = turns + 1;
        const turnOptions: ModelTurnOptions | undefined = observer || this.driver.requiresResultContract ? {
          ...(observer ? { onDelta: (delta: ModelTurnDelta) => { streamed = true; observer(turn, delta); } } : {}),
          ...(this.driver.requiresResultContract ? { acceptsResult: (value: unknown): boolean => {
            if (session.lam.type.kind !== 'lambda') return false;
            try {
              const type = session.lam.type.returns;
              return !problems(coerce(value, type, session.env, 'return'), type, session.env, 'return').holes.length;
            } catch (error) {
              if (error instanceof Reject) return false;
              throw error;
            }
          } } : {}) } : undefined;
        const turnRequest: ModelTurnRequest = { ...(callId ? { invocation_id: callId } : {}), ...(adapters.length ? { adapters } : {}),
          ...(session.executionOccurrence() ? { native_execution_occurrence: session.executionOccurrence() } : {}),
          ...(template ? { template } : {}),
          ...(this.options.guidance && supportsNeuralese(this.driver) ? { guidance: this.options.guidance } : {}),
          messages: encoded.messages, tools: this.options.promptRendering === 'program-header' ? [] : availableTools,
          // A turn that offers one tool it must use (the compaction turn, the last turn) requires a tool call.
          ...(availableTools !== allTools && this.options.promptRendering !== 'program-header' ? { tool_choice: 'required' as const } : {}),
          ...(this.options.temperature === undefined ? {} : { temperature: this.options.temperature }),
          seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            deriveSeed(session.runtime.seedPolicy.root!, runSeedKey(session.runtime.options), session.lam.attempts, 'model-turn', turns),
          max_tokens: statedWindow === null ? limit :
            Math.max(1, Math.min(limit ?? statedWindow, statedWindow - Math.ceil(estimate(availableTools)) - REPLY_MARGIN)) };
        // Drivers read the invocation's tuned stack from the current frame when called; browsers keep no frame
        // across the awaits above, so the call starts inside its own frame on every platform.
        response = await runInFrame(session.runtime.frame, () => turnOptions ? this.driver(turnRequest, session.runtime.signal, turnOptions) :
          this.driver(turnRequest, session.runtime.signal));
        streamed = false;
      } catch (error) {
        // What the failed request streamed belongs to an abandoned attempt, whether the turn is taken again or not.
        if (streamed) { streamed = false; this.options.onDelta!(turns + 1, { type: 'reset' }); }
        session.runtime.trace.emit('model_request', { call_id: callId, phase: 'error', turn: turns + 1,
          duration_ms: Math.round(performance.now() - started),
          error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` });
        // A server that refuses the request as too long says how long it was: learn the true size and take the turn
        // again, which compacts it. Twice at most per turn; beyond that the estimate is not the problem.
        const measured = contextOverflow(error instanceof Error ? error.message : '');
        if (measured && overflowRetries < 2 && sentChars > 0) {
          overflowRetries++;
          tokensPerChar = Math.max(tokensPerChar, measured.promptTokens / sentChars);
          if (measured.window !== null) statedWindow = measured.window;
          continue;
        }
        throw error;
      }
      overflowRetries = 0;
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'end', turn: turns + 1,
        duration_ms: Math.round(performance.now() - started),
        prompt_tokens: response.prompt_tokens ?? null, completion_tokens: response.completion_tokens ?? null,
        ...(response.scheduling ? { batch_id: response.scheduling.batch_id, batch_size: response.scheduling.batch_size,
          in_flight: response.scheduling.in_flight, queue_wait_ms: response.scheduling.queue_wait_ms,
          schedule_priority: response.scheduling.priority } : {}) });
      const turnNode = await this.modelTurnNode(session, graphMessages, response, turns + 1);
      if (response.prompt_tokens !== undefined && sentChars > 0) tokensPerChar = response.prompt_tokens / sentChars;
      turns++;
      session.runtime.checkInterruption();
      if(response.protocol_failure){
        const failure=nativeProtocolFailure(response.raw_response);
        if(!failure)throw new Error('neuralese-native-protocol-integrity: failed turn lost its server receipt');
        await this.nativeWriteOrigins(response,session,turns,turnNode);
        const recorded=response.raw_response?.x_natlang_native_recorded_turn;
        if(turnNode&&typeof recorded==='string')activeRecorder()?.linkNativeTurn?.(recorded,turnNode);
        tokens+=response.completion_tokens===undefined?limit??0:Math.max(1,response.completion_tokens);
        session.runtime.trace.emit('model_protocol_failure',{call_id:callId,turn:turns,
          code:failure.code,failed_turn:turnNode,failure});
        if(timedOut()||(maxTokens!==undefined&&tokens>maxTokens))return 'episode token or wall-clock budget exhausted';
        // Record the actual non-executable reply before the same call repairs it.
        // Neither a parsed prefix call nor an unfinished block becomes a result.
        messages.push({role:'assistant',content:response.text??'',...thought(response.reasoning)},
          {role:'user',content:nativeProtocolFeedback(failure)});
        continue;
      }
      response = await this.decodeNeuralese(response, session, turns, turnNode, availableTools === allTools || lastTurn);
      const recordedNativeTurn=response.raw_response?.x_natlang_native_recorded_turn;
      if(turnNode&&typeof recordedNativeTurn==='string')activeRecorder()?.linkNativeTurn?.(recordedNativeTurn,turnNode);
      // A reply that only writes out return_result({...}) is the tool's request; it is carried out as one.
      const written = !response.calls?.length && !response.truncated ? writtenReturnResult(response.text ?? '') : undefined;
      if (written) response = { ...response, text: '', calls: [['return_result', written]] };
      const calls = response.calls ?? [];
      const exactEmptyEos = immediateEmptyEos(response);
      if (!exactEmptyEos || calls.length || response.truncated) consecutiveEmptyEosWithoutResult = 0;
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'generated', turn: turns, calls, text: response.text ?? '' });
      tokens += response.completion_tokens === undefined ? limit ?? 0 : Math.max(1, response.completion_tokens);
      if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
        return 'episode token or wall-clock budget exhausted';
      if (!response.calls?.length && availableTools !== allTools && !lastTurn && !response.truncated) {
        // The compaction turn was answered without the tool: its text is not a result. Compact without a note.
        consecutiveEmptyEosWithoutResult = 0;
        const elided = pinNote(null) + collapseHistory(messages, protectedLength);
        compactedAt = estimate(allTools);
        session.runtime.trace.emit('compaction', { call_id: session.runtime.currentCallId ?? null, turn: turns,
          elided, note: null, estimated_tokens: Math.round(estimate(allTools)) });
        continue;
      }
      if (!response.calls?.length) {
        // A reply without a tool call ends the turn: it returns the staged result, or, for a string-typed
        // call with nothing staged, the reply's text is the result.
        if (!response.truncated) await session.acceptTextResult(response.text ?? '');
        const missing = this.missing(session);
        if (!response.truncated && exactEmptyEos && missing) {
          consecutiveEmptyEosWithoutResult++;
          if (consecutiveEmptyEosWithoutResult >= 2) {
            const detail = 'model_incomplete: terminated_without_result (repeated immediate EOS after one recovery turn)';
            session.runtime.trace.emit('model_incomplete', { call_id: session.runtime.currentCallId ?? null,
              reason: 'terminated_without_result', evidence: 'two_consecutive_zero_token_stop_replies',
              recovery_turns: 1 });
            return detail;
          }
        } else if (!missing) consecutiveEmptyEosWithoutResult = 0;
        // A reply without a tool call that leaves a refinement-rejected result unrepaired is a failed repair too: without
        // this count a model that only says "done" is nudged for ever (a scripted or stuck model never runs out of turns).
        if (!response.truncated && missing && lastRefinementFailure && this.options.refinement &&
            ++refinementRepairs > this.options.refinement.checker.repairBudget(this.options.maxFailureRepairs)) return lastRefinementFailure;
        if (!response.truncated && !missing && this.options.refinement) {
          const failures = await this.refinementGate(session);
          if (failures) {
            // The staged result does not satisfy its refinement: unstage it and tell the model, within the repair budget.
            session.lam.return = MISSING;
            lastRefinementFailure = `${failures[0]!.code}: ${failures[0]!.message}`;
            if (++refinementRepairs > this.options.refinement.checker.repairBudget(this.options.maxFailureRepairs))
              return `${failures[0]!.code}: ${failures[0]!.message}`;
            messages.push({ role: 'assistant', content: response.text ?? '', ...thought(response.reasoning) },
              { role: 'user', content: failureFeedback(failures).replace(/^rejected\n/, '') });
            continue;
          }
        }
        if (!response.truncated && !missing && session.finish()) { session.lam.note = response.text ?? ''; return; }
        // Code written into a reply has not run; saying so is what a model that wrote its eval out as text needs.
        const unrun = /```(?:ts|typescript|js|javascript)?\s*\n/.test(response.text ?? '') ?
          'The code in your reply was not run: code runs only when you call eval with it. ' : '';
        const feedback = response.truncated ? `Your reply was cut off ${limit === null ? 'at the length limit' : `at the ${limit}-token limit`} before any tool call. ` +
          (session.lam.return !== MISSING && !missing ? 'A complete typed result is already staged. Call return_result with {status: "success"} and omit value to return that exact result, or reply "done" to return it without a tool call.' :
            'Take the next step with one tool call.') :
          unrun + (missing || 'The staged result is incomplete.');
        if (response.truncated && ++failureRepairs > (this.options.maxFailureRepairs ?? Infinity)) {
          session.runtime.trace.emit('model_incomplete', { call_id: callId, turn: turns,
            reason: 'truncated_without_action', evidence: 'completed_truncated_reply_without_executable_action',
            failed_continuations: failureRepairs, repair_limit: this.options.maxFailureRepairs });
          return 'model_incomplete: truncated_without_action (configured repair limit reached)';
        }
        messages.push({ role: 'assistant', content: response.text ?? '', ...thought(response.reasoning) }, { role: 'user', content: feedback });
        continue;
      }
      const proposal: Record<string, unknown> = { calls, value_confidence: response.value_confidence ?? [],
        released: false, messages: [...messages] };
      this.proposals.push(proposal);
      let withdrawn = false;
      const review = this.options.review;
      if (review) for (const [index, [name, args]] of calls.entries()) {
        const rawConfidence = response.value_confidence?.[index] as number | { geometric_mean?: number } | null | undefined;
        const confidence = typeof rawConfidence === 'number' ? rawConfidence : rawConfidence?.geometric_mean;
        const lowValue = review.threshold !== undefined && confidence !== undefined && confidence < review.threshold;
        const structural = review.scope === 'actions' &&
          ['eval', 'edit_code', 'edit_file', 'write_file'].includes(name);
        if (!lowValue && !structural) continue;
        if (exhausted())
          return 'careful review budget exhausted before applying proposal';
        const budget = allowance();
        const fork = [...messages, { role: 'user', content: this.reviewPrompt(messages, calls, index) }];
        const answer = await (review.driver ?? this.driver)({ messages: review.driver && review.driver !== this.driver ? fork : renderPromptMessages(fork, this.options.promptRendering),
          tools: this.options.promptRendering === 'program-header' && (!review.driver || review.driver === this.driver) ? [] : this.reviewTools(),
          temperature: 0, seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            deriveSeed(session.runtime.seedPolicy.root!, runSeedKey(session.runtime.options), session.lam.attempts, 'review', turns),
          max_tokens: budget }, session.runtime.signal);
        turns++; tokens += answer.completion_tokens === undefined ? budget ?? 0 : Math.max(1, answer.completion_tokens);
        const audit: Record<string, unknown> = { proposal: this.proposals.length - 1, call_index: index,
          confidence: rawConfidence ?? null, messages: fork, calls: answer.calls ?? [], text: answer.text ?? '',
          order: review.order ?? 'reason_first', trigger: structural ? 'structural' : 'confidence',
          prompt_variant: review.prompt ?? 'baseline' };
        this.reviews.push(audit);
        session.runtime.checkInterruption();
        if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
          return 'careful review budget exhausted before applying proposal';
        if (answer.calls?.length !== 1 || answer.calls[0]![0] !== 'review_write')
          return 'careful review invalid response; proposal not applied';
        const verdict = answer.calls[0]![1], decision = verdict.decision, reason = verdict.reason;
        if (!['approve', 'withdraw', 'error', 'blocker'].includes(String(decision)) || typeof reason !== 'string')
          return 'careful review invalid verdict; proposal not applied';
        audit.decision = decision;
        if (decision === 'withdraw' && review.withdrawalPolicy === 'retry' && withdrawals < 1) {
          withdrawals++; withdrawn = true;
          proposal.withdrawn = true;
          session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
            phase: 'withdrawn', turn: turns, calls });
          messages.push({ role: 'user', content: 'The pending batch was withdrawn before execution. No action in it happened. Reconsider the original instructions from the unchanged workspace. Do not change requirements to obtain a result. This is the only reconsideration.' });
          break;
        }
        if (decision !== 'approve') return `careful review ${decision}: ${reason}`;
      }
      if (withdrawn) continue;
      proposal.released = true;
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'released', turn: turns, calls });
      const raw = calls.map(([name, args], i) => {
        const original = response.raw_calls?.[i] as Record<string, unknown> | undefined;
        const fn = original?.function as Record<string, unknown> | undefined;
        return original && fn && typeof fn.name === 'string' && typeof fn.arguments === 'string' ?
          { ...original, id: typeof original.id === 'string' && original.id ? original.id : `call_${turns}_${i}` } :
          { id: `call_${turns}_${i}`, type: 'function', function: { name, arguments: JSON.stringify(args) } };
      });
      const results: NativeResult[] = [];
      let refinementFailures: RefinementFailure[] | undefined;
      session.turn = turns;
      session.turnReasoning = response.reasoning;
      const previousFailureSerial = session.failureSerial;
      for (const [index, [name, args]] of calls.entries()) {
        if (timedOut()) return 'episode wall-clock budget exhausted';
        let appliedArgs = args;
        if (name === 'return_result' && this.options.refinement && (appliedArgs.status ?? 'success') === 'success') {
          const failures = await this.refinementGate(session, appliedArgs);
          if (failures) {
            refinementFailures = failures;
            results.push({ kind: 'rejected', text: failureFeedback(failures), codes: failures.map(failure => failure.code) });
            break;
          }
        }
        const result: NativeResult = await session.applyAsync(name, appliedArgs,
          typeof raw[index]!.id === 'string' ? raw[index]!.id as string : undefined);
        results.push(result);
        if (result.kind === 'blocked') return result.text;
        if (['blocked', 'budget', 'completed'].includes(result.kind)) break;
        // A later action may depend on this one's result. In particular, never
        // finish an older staged answer after the computation in this batch failed.
        if (['error', 'rejected', 'refused'].includes(result.kind) || session.failureSerial > previousFailureSerial) break;
      }
      messages.push({ role: 'assistant', content: '', tool_calls: raw.slice(0, results.length), ...thought(response.reasoning) });
      // Near the end of the turn budget the model is told how many turns are left, so a task that cannot be finished
      // ends with an honest blocked or failed rather than by running out.
      const left = maxTurns === undefined ? Infinity : maxTurns - turns;
      const notice = left === 1 ? LAST_TURN_NOTICE :
        left <= 4 && left > 0 ? `\n\n[${left} turns left in this call. If the task cannot be finished, call return_result with status "blocked" and what is missing, or status "failed" and why.]` : '';
      let note: string | undefined;
      for (const [index, result] of results.entries()) {
        const [name, args] = calls[index]!, id = String(raw[index]!.id);
        messages.push({ role: 'tool', tool_call_id: id, content: result.text + (index === results.length - 1 ? notice : '') });
        if (name === 'compact_history' && result.kind === 'ok') note = String(args.note).trim();
      }
      if (note !== undefined) {
        // The note speaks for everything before it: the conversation becomes the opening, the note, and the compaction
        // call with its result.
        const elided = pinNote(note) + collapseHistory(messages, protectedLength);
        compactedAt = requestChars(this.tools(session)) * tokensPerChar;
        compactedTurn = turns;
        session.runtime.trace.emit('compaction', { call_id: session.runtime.currentCallId ?? null, turn: turns,
          elided, note, estimated_tokens: Math.round(compactedAt) });
      }
      if (results.at(-1)?.kind === 'budget') return 'action or tool-call budget exhausted';
      const repairLimit = this.options.maxFailureRepairs;
      // A refinement rejected a proposed result: from the gate above, or from an eval that returned or finished one.
      const refinedResult = results.find(result => result.codes?.some(code => code === 'refinement-unsatisfied' || code === 'refinement-undecided'));
      if (refinementFailures || refinedResult) {
        lastRefinementFailure = refinementFailures ? `${refinementFailures[0]!.code}: ${refinementFailures[0]!.message}` :
          /refinement-(?:unsatisfied|undecided): [^\n]*/.exec(refinedResult!.text)?.[0] ?? 'refinement-unsatisfied: the result does not satisfy its refined type';
        if (++refinementRepairs > this.options.refinement!.checker.repairBudget(repairLimit)) return lastRefinementFailure;
        continue;
      }
      if (session.failureSerial > previousFailureSerial) {
        if (++failureRepairs > (repairLimit ?? Infinity))
          return `eval repair limit reached: ${session.failureDebug?.message ?? 'failure'}`;
        continue;
      }
      if (!session.failureDebug && results.some((result, index) => calls[index]?.[0] === 'eval' && result.kind === 'ok'))
        failureRepairs = 0;
      // A rejected tool call is reported back to the model like a failed eval, within the same repair budget.
      const failed = results.find(result => ['rejected', 'refused'].includes(result.kind));
      if (failed) {
        if (++failureRepairs > (repairLimit ?? Infinity)) return `repair limit reached: ${failed.text}`;
        continue;
      }
      if (results.at(-1)?.kind === 'completed') return;
    }
  }
}
