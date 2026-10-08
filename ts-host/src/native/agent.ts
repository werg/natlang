import { programGuidance } from '../adaptation/prompts.js';
import { formatType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, isLive, liveId, liveLabel, problems } from './values.js';
import type { Value } from './values.js';
import { COMPACTION_NOTE_CHARS, type NativeResult, type NativeSession } from './runtime.js';
import { undeclaredServiceType } from './introspection.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { deriveSeed } from './trace.js';
import { AUTOMATIC_NOTE, COMPACTION_NOTICE, directoryReducerPrompt, fileToolNames, FUNCTION_TOOLS_PROMPT, HANDOVER_NOTE_CLOSE, HANDOVER_NOTE_OPEN,
  LAST_TURN_NOTICE, TOOLS_PROMPT, promptAtNlDepthLimit, type FileToolSurface } from './prompt.js';
import { canGenerateNl } from '../runtime/context.js';
import { adoptImportedBlocks } from './nz-file.js';
import { FileHandle, FolderHandle, fileListingText, type Folder } from './scoped-fs.js';
import { SHOWN_CHARS, note as cutNote } from './cutoff.js';
import { digestNote } from './prompt.js';
import { decodeTurnValue, encodeMessages, isNeuraleseRef, neuraleseSentinel, NeuraleseUnsupportedError, supportsNeuralese,
  sentinelIds, type NeuraleseRuntimeOptions } from './neuralese.js';
import { resolveNeuralesePreviews } from './neuralese-preview.js';
import { blockInput, graphNode, invocationNodeId } from './graph.js';
import { DECISION_SYSTEM_PROMPT, decisionPrompt, decisionScorer, finiteValues, softmax } from './decision.js';
import type { NeuraleseBlockMeta } from './neuralese-store.js';
import { activeSystemPrompts, softenMessages } from './system-prompts.js';

/** The code tools, as offered. Kept here so data collected under earlier wording can be migrated to it exactly. */
export const READ_CODE_DESCRIPTION = 'Inspect a named item this call can use but does not show: source of a function in the program\'s codebase; ' +
  'documentation for eval built-ins (nl, iterateOn, transcript); a declaration for an external service or importable package ' +
  '("pkg" lists its exports, "pkg.name" shows one); the type-only member shape of a host service object visible to this call; ' +
  'or the exact schema and instructions of a native tool currently offered to this call. Tool schemas and service shapes are ' +
  'metadata, not program function source or service implementation.';
export const EDIT_CODE_DESCRIPTION = 'Edit a function in the program\'s codebase: replace one exact or uniquely fuzzy span of its source. ' +
  'The function is validated before the edit becomes live.';
/**
 * Each assistant turn goes back into the conversation with its reasoning. The template decides whether to show it:
 * some drop earlier reasoning, others keep it (Ling's "preserved thinking"), and there a turn sent without it reads
 * as an empty <think></think> that teaches the model to skip thinking. The runtime's own opening turns carry a short
 * thought for the same reason.
 */
const thought = (reasoning: string | undefined) => reasoning ? { reasoning_content: reasoning } : {};
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
export const BUILT_INS_LINE = 'Eval also has the built-ins nl, iterateOn and transcript; read_code shows how to use each.';
/** Inline children have the same delegation tools as named calls. */
export const INLINE_BUILT_INS_LINE = BUILT_INS_LINE;
export const NL_DEPTH_LIMIT_BUILT_INS_LINE = 'Eval has iterateOn and transcript; read_code shows how to use each. ' +
  'This is the fifth ad hoc layer: make further judgments here or call an existing named function from a file.';
export const OPENING_THOUGHT = "I'll start by reading this call's arguments into the eval scope.";
export const FOLDER_THOUGHT = "Next I'll list the files in this call's folder.";
export const DIFF_CODE_DESCRIPTION = 'Show the changes made to functions of the program\'s codebase in this call.';

/** Assistant turns the model has taken, not counting the runtime's pre-filled scope calls. */
export function modelTurnsSoFar(messages: readonly Record<string, unknown>[]): number {
  return messages.filter(message => message.role === 'assistant' &&
    !((message.tool_calls as { id?: string }[] | undefined) ?? []).some(call => String(call.id).startsWith('scope_'))).length;
}

export type NativeModelDriver = (request: ModelTurnRequest, signal?: AbortSignal) => Promise<ModelTurn> | ModelTurn;
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
export function inputsListing(session: NativeSession, digests: Readonly<Record<string, string>> = {}): string {
  const lam = session.lam;
  if (lam.type.kind !== 'lambda') return '{}';
  const root = lam.projectTransaction?.folder;
  return lam.type.params.fields.map(field => {
    const value = Object.hasOwn(lam.args, field.name) ? lam.args[field.name]! : undefined;
    // A large value written as a digest (DECISIONS.md 43) shows the digest; the variable holds the value itself.
    const shown = digests[field.name] ? neuraleseSentinel(digests[field.name]!) + digestNote(field.name) :
      renderValue(value, { root, holder: field.name, liveIdentity: session.runtime.displayLiveId });
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
export function renderValue(value: Value | unknown, options: { holder?: string; budget?: number; root?: Folder; liveIdentity?: (value: object) => number } = {}): string {
  if (value === undefined) return 'undefined';
  const requested=options.budget??SHOWN_CHARS;
  const budget=Number.isFinite(requested)?Math.max(0,requested):Infinity;
  // Render once at the requested budget. Building the complete literal first made even a
  // 2,000-character scope preview allocate the entire value (including combinatorial arrays).
  return scopeExpression(value, options.root, options.holder, budget) ?? previewValue(value as Value, options.holder, options.liveIdentity);
}

/** Whether a captured value is a weight adapter (`Adapter`): it acts on the model's weights for this call's turns and
 * is not shown in the scope (a block sentinel there would be read as text-space Neuralese). */
const isAdapterValue = (value: unknown) => isNeuraleseRef(value) && value.$neuralese.type === 'Adapter';
function adapterCells(lam: { captures?: Record<string, { get(): unknown }> }): { id: string; scale: number }[] {
  const out: { id: string; scale: number }[] = [];
  for (const cell of Object.values(lam.captures ?? {})) {
    let value: unknown;
    try { value = cell.get(); } catch { continue; }
    if (isAdapterValue(value)) out.push({ id: (value as { $neuralese: { id: string } }).$neuralese.id, scale: 1 });
  }
  return out;
}
const capturedValue = (cell: { get(): unknown }) => { try { return cell.get(); } catch { return undefined; } };

/** A short preview of a value that has no literal form; `holder` names where all of it is (see cutoff.ts). */
function previewValue(value: Value, holder?: string, identity?: (value: object) => number): string {
  if (isNeuraleseRef(value)) return neuraleseSentinel(value.$neuralese.id);
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
  state = { nodes: 0, depth: 0 }): string | undefined {
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
  if (isNeuraleseRef(value)) return neuraleseSentinel(value.$neuralese.id);
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
    const probe = { nodes: state.nodes, depth: state.depth };
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
      temperature?: number; maxSeconds?: number; systemPrompt?: string | (() => string); programGuidance?: string;
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
      /** Tensor store and write port for soft values (S0 §3). */
      neuralese?: NeuraleseRuntimeOptions;
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
   * Digests of the arguments whose listing would be cut off (DECISIONS.md 43), when the runtime has a digester and the
   * driver carries Neuralese: block IDs by argument name. Only plain data is digested; a failed digest keeps the preview.
   */
  private async digests(session: NativeSession): Promise<Record<string, string>> {
    const digest = this.options.neuralese?.digest;
    const lam = session.lam;
    if (!digest || !supportsNeuralese(this.driver) || lam.type.kind !== 'lambda') return {};
    const root = lam.projectTransaction?.folder;
    const out: Record<string, string> = {};
    for (const field of lam.type.params.fields) {
      if (!Object.hasOwn(lam.args, field.name)) continue;
      const value = lam.args[field.name]!;
      if (!renderValue(value, { root, holder: field.name, liveIdentity: session.runtime.displayLiveId }).includes('<<cut off:')) continue;
      let text: string | undefined;
      try { text = JSON.stringify(value); } catch { text = undefined; }
      if (text === undefined || value instanceof FileHandle) continue;
      const instructions = typeof lam.body === 'string' ? lam.body : '';
      try {
        const written = await digest({ name: field.name, type: formatType(field.type), value: text, instructions });
        if (written) {
          out[field.name] = written.$neuralese.id;
          session.runtime.trace.emit('digest', { call_id: session.runtime.currentCallId ?? null, argument: field.name,
            block: written.$neuralese.id, chars: text.length });
        }
      } catch (error) {
        session.runtime.trace.emit('digest', { call_id: session.runtime.currentCallId ?? null, argument: field.name,
          error: String((error as Error)?.message ?? error).slice(0, 300) });
      }
    }
    return out;
  }

  /** Messages with the runtime's prompt pieces in their soft forms, when a bank is configured and the driver is Neuralese. */
  private softened<M extends Record<string, unknown>>(messages: readonly M[]): M[] {
    const bank = activeSystemPrompts(this.options.neuralese?.systemPrompts);
    return bank?.size && supportsNeuralese(this.driver) ? softenMessages(messages, bank) : [...messages];
  }

  /**
   * Turn the Neuralese content of a reply into conversation text: blocks a backend wrote (content parts) and literals
   * written as marker text (stored through the write port) become block markers in the reply and call arguments.
   */
  private async decodeNeuralese(response: ModelTurn, session: NativeSession, turn: number, turnNode?: string): Promise<ModelTurn> {
    const carries = (value: unknown): boolean => typeof value === 'string' ? value.includes('<|neuralese|>') :
      Array.isArray(value) ? value.some(carries) : !!value && typeof value === 'object' &&
        ((value as { type?: unknown }).type === 'neuralese' || Object.values(value).some(carries));
    if (!carries(response.text) && !carries(response.calls)) return response;
    const returnType = session.lam.type.kind === 'lambda' ? formatType(session.lam.type.returns) : undefined;
    const producer = { call_id: session.runtime.currentCallId ?? null, turn,
      ...(returnType ? { result_type: returnType } : {}) };
    const port = this.options.neuralese?.port;
    const expectedReturn = session.lam.type.kind === 'lambda' ? session.env.resolve(session.lam.type.returns) : undefined;
    const returnsString = expectedReturn?.kind === 'prim' && expectedReturn.name === 'string';
    const written: NeuraleseBlockMeta[] = [];
    const decodeCall = async ([name, args]: [string, Record<string, unknown>]): Promise<[string, Record<string, unknown>]> => {
      const decodedArgs: Record<string, unknown> = {};
      for (const [key, value] of Object.entries(args)) {
        const context = name === 'eval' && key === 'code' ? 'eval-code' :
          name === 'return_result' && key === 'value' ? 'return-result' : 'other-tool-argument';
        // At a declared string result boundary, marker-looking characters are string content. Preserve them byte for
        // byte instead of treating them as a transport block; marker decoding remains type-directed everywhere else.
        const literalStringResult = name === 'return_result' && key === 'value' && returnsString &&
          (args.status === undefined || args.status === 'success') && typeof value === 'string';
        decodedArgs[key] = literalStringResult ? value :
          await decodeTurnValue(value, port, { ...producer, marker_context: context }, written);
      }
      return [name, decodedArgs];
    };
    const decoded = { ...response,
      ...(response.text === undefined ? {} : { text: await decodeTurnValue(response.text, port,
        { ...producer, marker_context: 'assistant-text' }, written) as string }),
      ...(response.calls === undefined ? {} : { calls: await Promise.all(response.calls.map(decodeCall)) as ModelTurn['calls'] }) };
    // Each block the turn wrote is a node. Stop decisions and distribution parameters come from a writer that reports
    // them (a Neuralese server); the stand-in writer has none.
    for (const block of written) {
      const reported = (block.producer ?? {}) as Record<string, unknown>;
      graphNode(session.runtime.trace, 'block_write', { call_id: producer.call_id, turn: turnNode ?? '', block: block.id,
        length: block.length, truncated: !!block.truncated, stops: Array.isArray(reported.stops) ? reported.stops : [],
        ...Object.fromEntries(['position', 'temperature', 'seed', 'mean', 'scale', 'distribution',
          'emulation_version', 'learned_vectors', 'text_body_sha256', 'result_type', 'marker_context']
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
  private modelTurnNode(session: NativeSession, messages: readonly unknown[], response: ModelTurn, turn: number): string | undefined {
    const trace = session.runtime.trace;
    const callId = session.runtime.options.runId;
    const turnId = `${callId}#turn${turn}`;
    const reads: { node: string; port: string; block: string }[] = [];
    const seen = new Set<string>();
    messages.forEach((message, index) => {
      const record = message as { content?: unknown; tool_calls?: { function?: { arguments?: unknown } }[] };
      const texts = [record.content, ...(record.tool_calls ?? []).map(call => call.function?.arguments)];
      texts.forEach((text, part) => {
        if (typeof text !== 'string') return;
        for (const block of sentinelIds(text)) {
          if (seen.has(block)) continue;
          seen.add(block);
          const node = graphNode(trace, 'block_read', { call_id: callId, turn: turnId, block, positions: [index, part] }, [blockInput(block, 'block')]);
          if (node) reads.push({ node, port: 'read', block });
        }
      });
    });
    return graphNode(trace, 'model_turn', { call_id: callId, turn, temperature: this.options.temperature ?? null,
      seed: session.runtime.seedPolicy.mode === 'backend' ? null : deriveSeed(session.runtime.seedPolicy.root!,
        session.runtime.options.seedId ?? session.runtime.options.runId, session.lam.attempts, 'model-turn', turn - 1),
      completion_tokens: response.completion_tokens ?? null, text_chars: (response.text ?? '').length,
      calls: (response.calls ?? []).map(call => Array.isArray(call) ? call[0] : (call as { name?: string }).name ?? null) },
      [{ node: invocationNodeId(callId), port: 'invocation' }, ...reads], turnId);
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
        { code: { type: 'string' }, finish: { type: 'boolean', description: 'Finish this function using the fresh typed final expression or explicit return computed in this eval. Use false or omit for inspection or staging; never finishes an older staged value.' }, timeout_ms: { type: 'integer', minimum: 1,
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
      READ_CODE_DESCRIPTION.replace('(nl, iterateOn, transcript)', '(iterateOn, transcript)'), { name: { type: 'string' } }, ['name']);
    if (!ownCode) tools.splice(2, 0, readCode);
    if (ownCode) tools.splice(2, 0,
      readCode,
      tool('edit_code', EDIT_CODE_DESCRIPTION,
        { name: { type: 'string' }, find: { type: 'string' }, replace_with: { type: 'string' }, fuzzy: { type: 'boolean' } },
        ['name', 'find', 'replace_with']),
      tool('diff_code', DIFF_CODE_DESCRIPTION, {}, []));
    if (session.lam.projectTransaction) {
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
        { code: { type: 'string' }, finish: { type: 'boolean', description: 'Finish this function using the fresh typed final expression or explicit return computed in this eval. Use false or omit for inspection or staging; never finishes an older staged value.' }, timeout_ms: { type: 'integer', minimum: 1 } }, ['code']),
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
      const offered = fileToolNames(this.options.fileTools).filter(name => name !== 'delegate' || canGenerateNl(session.runtime.frame));
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
  private scopeReading(session: NativeSession, digests: Readonly<Record<string, string>> = {}): { code: string; text: string } | undefined {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return;
    const lines: string[] = [], names: string[] = [];
    const root = lam.projectTransaction?.folder;
    const section = (heading: string, body: string[]) => { if (body.length) lines.push(...(lines.length ? [''] : []), heading, ...body); };
    const declared = (keyword: string, name: string, type: string, value: Value, note = ''): string => {
      // Host types print as their tag; only a class-like tag (FileHandle, Map) is a usable TypeScript type.
      const shown = /^[a-z]+$/.test(type) && !['string', 'number', 'boolean', 'null'].includes(type) ? 'unknown' : type;
      const expression = scopeExpression(value, root, name);
      names.push(name);
      return expression === undefined ?
        `declare ${keyword === 'let' ? 'let' : 'const'} ${name}: ${shown};  // live value ${previewValue(value, name, session.runtime.displayLiveId)}${note}` :
        `${keyword} ${name}: ${shown} = ${expression};${note}`;
    };
    section('// Functions you can call:', this.callableDeclarations(session));
    section('// Provided by the host:', [
      ...Object.entries(session.availableServices()).map(([name, service]) => session.runtime.declarations[name] ?
        `${session.runtime.declarations[name]}  // external service; its calls are recorded as effects` :
        `declare const ${name}: ${undeclaredServiceType(service)};  // service; its calls are recorded as effects`),
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
      section('// Your staged result:', [`// ${renderValue(lam.return, { root, liveIdentity: session.runtime.displayLiveId })}`]);
    if (!lines.length) return;
    // The arguments appear in the eval's result, not as literals in its code: they come from the caller.
    return { code: lines.join('\n'), text: (params.length ? inputsListing(session, digests) + '\n' : '') +
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
   * Answer a finite-typed call by scoring each allowed value as the reply to its opening (native/decision.ts).
   * Returns false when the readout does not apply, so the tool loop runs; a string is a failure, as from run.
   */
  private async decisionReadout(session: NativeSession, opening: Record<string, unknown>[]): Promise<false | string | void> {
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
    const system = (this.options.decisionSystemPrompt?.() ?? DECISION_SYSTEM_PROMPT) + programGuidance(this.options.programGuidance ?? '');
    const messages = [{ role: 'system', content: system }, ...opening.slice(1), { role: 'user', content: decisionPrompt(replies) }];
    const encoded = encodeMessages(this.softened(messages));
    if (encoded.blocks && !supportsNeuralese(this.driver))
      throw new NeuraleseUnsupportedError('this model backend cannot carry Neuralese blocks');
    const started = performance.now();
    let scores;
    try {
      const adapters = adapterCells(lam);
      if (adapters.length && !supportsNeuralese(this.driver))
        throw new NeuraleseUnsupportedError('this model backend cannot apply weight adapters');
      scores = await decide({ messages: encoded.blocks ? encoded.messages : messages, options: replies, ...(adapters.length ? { adapters } : {}) }, session.runtime.signal);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      if (!message.startsWith('decision-unsupported')) throw error;
      session.runtime.trace.emit('decision_readout', { call_id: callId, phase: 'unavailable', reason: message });
      return false;
    }
    if (scores.log_probs.length !== replies.length || scores.log_probs.some(value => !Number.isFinite(value)))
      throw new Error('decision readout returned no finite score for every option');
    const probabilities = softmax(scores.log_probs);
    const chosen = probabilities.indexOf(Math.max(...probabilities));
    session.runtime.trace.emit('decision_readout', { call_id: callId, phase: 'scored', options: replies,
      log_probs: scores.log_probs, probabilities, tokens: scores.tokens ?? null, chosen,
      duration_ms: Math.round(performance.now() - started) });
    lam.return = values[chosen] as Value;
    lam.note = JSON.stringify({ readout: 'decision', probabilities: Object.fromEntries(replies.map((reply, index) => [reply, probabilities[index]])) });
    if (!session.finish()) return 'decision readout chose a value the declared type rejects';
  }

  async run(session: NativeSession): Promise<string | void> {
    // Blocks of `.nz` files loaded without a store (compiled imports, companion folders) join this runtime's store.
    if (this.options.neuralese?.store) await adoptImportedBlocks(this.options.neuralese.store);
    // Fixed for the whole call, so the server can reuse its prompt cache across turns.
    let adaptedSystem: string | undefined;
    const systemPrompt = () => {
      if (adaptedSystem !== undefined) return adaptedSystem;
      const base = typeof this.options.systemPrompt === 'function' ? this.options.systemPrompt() : this.options.systemPrompt ?? TOOLS_PROMPT;
      const allowAdHoc = canGenerateNl(session.runtime.frame);
      const composed = (allowAdHoc ? base : promptAtNlDepthLimit(base)) +
      (Object.keys(session.lam.codebase).length ? FUNCTION_TOOLS_PROMPT : '') +
      (session.lam.projectTransaction ? directoryReducerPrompt(this.options.fileTools, allowAdHoc) : '') +
      programGuidance(this.options.programGuidance ?? '');
      if (this.options.programGuidance !== undefined) adaptedSystem = composed;
      return composed;
    };
    const digests = await this.digests(session);
    const openingMessages = (): Record<string, unknown>[] => {
      const reading = this.scopeReading(session, digests);
      const scopeOpening = this.scopeOpening(session);
      if (session.lam.skills?.listing) for (const skill of session.lam.skills.inventory ?? [])
        session.runtime.trace.emit('skill_use', { phase: 'offered', skill_name: skill.name,
          skill_revision: skill.revision, invocation_id: session.runtime.options.runId,
          interpretation: 'listed_in_invocation_opening_not_awareness' });
      return [{ role: 'system', content: systemPrompt() },
        { role: 'user', content: scopeOpening },
        ...(reading ? [{ role: 'assistant', content: '', ...thought(OPENING_THOUGHT), tool_calls: [{ id: 'scope_0', type: 'function',
          function: { name: 'eval', arguments: JSON.stringify({ code: reading.code }) } }] },
        { role: 'tool', tool_call_id: 'scope_0', content: reading.text }] : []),
        ...(session.lam.projectTransaction ? [{ role: 'assistant', content: '', ...thought(FOLDER_THOUGHT), tool_calls: [{ id: 'scope_1', type: 'function',
          function: { name: 'list_files', arguments: '{}' } }] },
        { role: 'tool', tool_call_id: 'scope_1', content: this.folderListing(session) }] : [])];
    };
    const messages = openingMessages();
    const decided = await this.decisionReadout(session, messages);
    if (decided !== false) return decided;
    const openingLength = messages.length;
    const budget = this.options.contextTokens === undefined ? DEFAULT_CONTEXT_TOKENS : this.options.contextTokens;
    // Prompt tokens per character of request, calibrated from the server's reported prompt size.
    let tokensPerChar = 1 / 3.5;
    const requestChars = (tools: unknown[]) => JSON.stringify(messages).length + JSON.stringify(tools).length;
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
    const maxTurns = this.options.maxTurns, maxTokens = this.options.maxTokens;
    const deadline = this.options.maxSeconds === undefined ? null : Date.now() + this.options.maxSeconds * 1000;
    let tokens = 0, turns = 0, withdrawals = 0, failureRepairs = 0;
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
      const nearLimit = budget !== null && compactedTurn !== turns && estimate(allTools) >
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
        tool_schema_bytes: new TextEncoder().encode(JSON.stringify(availableTools)).length,
        messages: messages.length });
      let response: ModelTurn;
      try {
        // Blocks in the conversation travel as content parts; a backend that cannot carry them fails the call.
        const encoded = encodeMessages(this.softened(messages));
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
        response = await this.driver({ ...(callId ? { invocation_id: callId } : {}), ...(adapters.length ? { adapters } : {}),
          ...(template ? { template } : {}),
          ...(this.options.guidance && supportsNeuralese(this.driver) ? { guidance: this.options.guidance } : {}),
          messages: encoded.blocks ? encoded.messages : messages, tools: availableTools,
          // A turn that offers one tool it must use (the compaction turn, the last turn) requires a tool call.
          ...(availableTools !== allTools ? { tool_choice: 'required' as const } : {}),
          ...(this.options.temperature === undefined ? {} : { temperature: this.options.temperature }),
          seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            deriveSeed(session.runtime.seedPolicy.root!, session.runtime.options.seedId ?? session.runtime.options.runId, session.lam.attempts, 'model-turn', turns),
          max_tokens: limit }, session.runtime.signal);
      } catch (error) {
        session.runtime.trace.emit('model_request', { call_id: callId, phase: 'error', turn: turns + 1,
          duration_ms: Math.round(performance.now() - started),
          error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` });
        // A server that refuses the request as too long says how long it was: learn the true size and take the turn
        // again, which compacts it. Twice at most per turn; beyond that the estimate is not the problem.
        const measured = /exceed_context_size_error[\s\S]*?"n_prompt_tokens":(\d+)/.exec(error instanceof Error ? error.message : '');
        if (measured && overflowRetries < 2 && sentChars > 0) {
          overflowRetries++;
          tokensPerChar = Math.max(tokensPerChar, Number(measured[1]) / sentChars);
          continue;
        }
        throw error;
      }
      overflowRetries = 0;
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'end', turn: turns + 1,
        duration_ms: Math.round(performance.now() - started),
        prompt_tokens: response.prompt_tokens ?? null, completion_tokens: response.completion_tokens ?? null });
      const turnNode = this.modelTurnNode(session, messages, response, turns + 1);
      if (response.prompt_tokens !== undefined && sentChars > 0) tokensPerChar = response.prompt_tokens / sentChars;
      turns++;
      session.runtime.checkInterruption();
      response = await this.decodeNeuralese(response, session, turns, turnNode);
      const calls = response.calls ?? [];
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'generated', turn: turns, calls, text: response.text ?? '' });
      tokens += response.completion_tokens === undefined ? limit ?? 0 : Math.max(1, response.completion_tokens);
      if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
        return 'episode token or wall-clock budget exhausted';
      if (!response.calls?.length && availableTools !== allTools && !lastTurn && !response.truncated) {
        // The compaction turn was answered without the tool: its text is not a result. Compact without a note.
        const elided = pinNote(null) + collapseHistory(messages, protectedLength);
        compactedAt = estimate(allTools);
        session.runtime.trace.emit('compaction', { call_id: session.runtime.currentCallId ?? null, turn: turns,
          elided, note: null, estimated_tokens: Math.round(estimate(allTools)) });
        continue;
      }
      if (!response.calls?.length) {
        // A reply without a tool call ends the turn: it returns the staged result, or, for a string-typed
        // call with nothing staged, the reply's text is the result.
        if (!response.truncated) session.acceptTextResult(response.text ?? '');
        const missing = this.missing(session);
        if (!response.truncated && !missing && session.finish()) { session.lam.note = response.text ?? ''; return; }
        // Code written into a reply has not run; saying so is what a model that wrote its eval out as text needs.
        const unrun = /```(?:ts|typescript|js|javascript)?\s*\n/.test(response.text ?? '') ?
          'The code in your reply was not run: code runs only when you call eval with it. ' : '';
        const feedback = response.truncated ? `Your reply was cut off ${limit === null ? 'at the length limit' : `at the ${limit}-token limit`} before any tool call. ` +
          (session.lam.return !== MISSING && !missing ? 'A complete typed result is already staged. Call return_result with {status: "success"} and omit value to return that exact result, or reply "done" to return it without a tool call.' :
            'Take the next step with one tool call.') :
          unrun + (missing || 'The staged result is incomplete.');
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
        const answer = await (review.driver ?? this.driver)({ messages: fork, tools: this.reviewTools(),
          temperature: 0, seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            deriveSeed(session.runtime.seedPolicy.root!, session.runtime.options.seedId ?? session.runtime.options.runId, session.lam.attempts, 'review', turns),
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
      session.turn = turns;
      session.turnReasoning = response.reasoning;
      const previousFailureSerial = session.failureSerial;
      for (const [index, [name, args]] of calls.entries()) {
        if (timedOut()) return 'episode wall-clock budget exhausted';
        let appliedArgs = args;
        const expected = session.lam.type.kind === 'lambda' ? session.lam.type.returns : undefined;
        if (name === 'return_result' && args.status !== 'blocked' && args.status !== 'failed' &&
            Object.hasOwn(args, 'value') && expected && this.options.neuralese?.store) {
          const captures: unknown[] = [];
          for (const cell of Object.values(session.lam.captures ?? {})) {
            try { captures.push(cell.get()); } catch { /* inaccessible live captures are not authorization */ }
          }
          const visible = [...Object.values(session.lam.args), ...Object.values(session.lam.let), ...captures];
          const normalized = await resolveNeuralesePreviews(args.value, expected, session.env, visible,
            this.options.neuralese.store);
          if (normalized.resolutions.length) {
            appliedArgs = { ...args, value: normalized.value };
            session.runtime.trace.emit('neuralese_preview_resolution', { schema: 'natlang.neuralese-preview-resolution/1',
              call_id: session.runtime.options.runId, resolutions: normalized.resolutions });
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
