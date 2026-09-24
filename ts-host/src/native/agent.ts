import { formatType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, isLive, liveId, liveLabel, problems } from './values.js';
import type { Value } from './values.js';
import { COMPACTION_NOTE_CHARS, type NativeResult, type NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { deriveSeed } from './trace.js';
import { DIRECTORY_REDUCER_PROMPT, FUNCTION_TOOLS_PROMPT, TOOLS_PROMPT } from './prompt.js';
import { FileHandle, FolderHandle, fileListingText, type Folder } from './scoped-fs.js';
import { PAGE_CHARS } from './evaluator.js';
import type { PageStore } from './pages.js';

/** Assistant turns the model has taken, not counting the runtime's pre-filled scope calls. */
export function modelTurnsSoFar(messages: readonly Record<string, unknown>[]): number {
  return messages.filter(message => message.role === 'assistant' &&
    !((message.tool_calls as { id?: string }[] | undefined) ?? []).some(call => String(call.id).startsWith('scope_'))).length;
}

export type NativeModelDriver = (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
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
 * the same values in eval). Long values are paged; an argument the caller left out is undefined.
 */
export function inputsListing(session: NativeSession): string {
  const lam = session.lam;
  if (lam.type.kind !== 'lambda') return '{}';
  const root = lam.projectTransaction?.folder;
  return lam.type.params.fields.map(field => {
    const value = Object.hasOwn(lam.args, field.name) ? lam.args[field.name]! : undefined;
    const shown = value === undefined ? 'undefined' : scopeExpression(value, root, session.pages) ?? previewValue(value);
    return `${field.name}: ${formatType(field.type)}${field.optional ? ' | undefined' : ''} = ${shown}`;
  }).join('\n');
}

/** The folder handle API a directory reducer's eval sees, as TypeScript declarations. */
const FOLDER_DECLARATIONS = [
  'interface Entry { readonly name: string; readonly relativePath: string; readonly parent: Folder | null; exists(): Promise<boolean>;',
  '  stat(): Promise<{ path: string, kind: "file" | "folder", bytes: number }>; remove(): Promise<void>;',
  '  /** Like mv: moveTo("done/") or moveTo(folder.dir("done")) moves into that folder; moveTo("done/a.md") renames. */',
  '  moveTo(destination: Folder | FileHandle | string): Promise<void>; }',
  'interface FileHandle extends Entry { readText(startLine?: number, endLine?: number): Promise<string>; readJson(): Promise<unknown>;',
  '  readBytes(): Promise<Uint8Array>; writeText(content: string): Promise<void>; writeJson(value: unknown): Promise<void>;',
  '  writeBytes(content: Uint8Array): Promise<void>; editText(find: string, replaceWith: string, fuzzy?: boolean): Promise<unknown>; }',
  'interface Folder extends Entry { file(path: string): FileHandle; dir(path: string): Folder; entries(pattern?: string): Promise<Entry[]>;',
  '  files(pattern?: string): Promise<FileHandle[]>; folders(pattern?: string): Promise<Folder[]>; diff(): Promise<unknown>;',
  '  /** Run a directory reducer on this folder and keep the file changes it commits. */',
  '  apply(reducer: Function, ...args: unknown[]): Promise<unknown>; }',
];

const DEFAULT_CONTEXT_TOKENS = 16384;
/** Appended to the latest tool result when the next turn must compact. */
const COMPACTION_NOTICE = '\n\n[This conversation is near its context limit. Call compact_history with a short note on what you are ' +
  'doing and what is left; the full history stays available in transcript.]';
/** Messages at the end of the conversation that budget compaction keeps whole if it can: the latest exchanges. */
const RECENT_MESSAGES = 6;
/** What replaces an old tool output when the conversation is compacted; `entry` is its transcript index. */
export const elidedOutput = (entry?: number) => '[Output elided to keep this conversation within its context budget' +
  (entry === undefined ? '. Values it stored are still in scope.]' : `; transcript[${entry}].output holds it.]`);
/** What replaces the code of an old eval call when outputs alone do not bring the conversation under budget. */
export const elidedCode = (entry?: number) => '// Code elided to keep this conversation within its context budget' +
  (entry === undefined ? '. Its declarations are still in scope.' : `; transcript[${entry}].code holds it.`);
const ELIDED = /^\[Output elided |^\/\/ Code elided /;

/**
 * Deterministic compaction: replace the oldest tool outputs after the opening with a stub, oldest first, until
 * `over()` is false; if that is not enough, replace the code of the oldest eval calls the same way. The opening and
 * the last `recent` messages stay whole; the messages keep their order and number. `entry` maps a tool message, or
 * a tool call's id, to its transcript index, which the stub names so the model can read the original with code.
 * Returns how many outputs and calls were elided.
 */
export function compactMessages(messages: Record<string, unknown>[], openingLength: number, recent: number,
  over: () => boolean, entry: (callId: string) => number | undefined = () => undefined): number {
  let elided = 0;
  for (let index = openingLength; index < messages.length - recent && over(); index++) {
    const message = messages[index]!;
    if (message.role !== 'tool' || typeof message.content !== 'string' || ELIDED.test(message.content)) continue;
    const stub = elidedOutput(entry(String(message.tool_call_id)));
    if (message.content.length <= stub.length) continue;
    messages[index] = { ...message, content: stub };
    elided++;
  }
  for (let index = openingLength; index < messages.length - recent && over(); index++) {
    const message = messages[index]!;
    if (message.role !== 'assistant' || !Array.isArray(message.tool_calls)) continue;
    let changed = false;
    const calls = (message.tool_calls as Array<Record<string, unknown>>).map(call => {
      const fn = call.function as { name?: unknown; arguments?: unknown } | undefined;
      if (fn?.name !== 'eval' || typeof fn.arguments !== 'string') return call;
      let args: Record<string, unknown>;
      try { args = JSON.parse(fn.arguments) as Record<string, unknown>; } catch { return call; }
      const stub = elidedCode(entry(String(call.id)));
      if (typeof args.code !== 'string' || ELIDED.test(args.code) || args.code.length <= stub.length) return call;
      changed = true;
      return { ...call, function: { ...fn, arguments: JSON.stringify({ ...args, code: stub }) } };
    });
    if (changed) { messages[index] = { ...message, tool_calls: calls }; elided++; }
  }
  return elided;
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

function previewValue(value: Value): string {
  if (isLive(value)) return livePreview(value as object);
  if (typeof value === 'string') {
    const text = value.trimEnd(), lines = text.split('\n');
    if (text.length > 400 || lines.length > 8)
      return `${JSON.stringify(lines[0]!.slice(0, 80))} … CUT OFF: only the beginning of ${lines.length} lines, ${text.length} characters. Read it before using it.`;
    return lines.length > 1 ? '\n' + lines.map(line => `      | ${line}`).join('\n') : JSON.stringify(text);
  }
  if (Array.isArray(value)) {
    const head = value.slice(0, 3).map(previewValue).join(', ');
    const more = value.length > 3 ? `, … ${value.length - 3} more (read to see)` : '';
    return `${value.length} items: [${head}${more}]`;
  }
  if (value && typeof value === 'object') {
    const entries = Object.entries(value);
    const head = entries.slice(0, 6).map(([key, item]) => `${key}: ${previewValue(item)}`).join(', ');
    const more = entries.length > 6 ? `, … ${entries.length - 6} more fields (read to see)` : '';
    return `{ ${head}${more} }`;
  }
  if (value === null) return 'null';
  return String(value);
}

/** Bounded preview of a live host value: type, stable identity, and a short observation. */
export function livePreview(value: object): string {
  const label = liveLabel(value), id = liveId(value);
  let detail = '';
  try {
    const tag = Object.prototype.toString.call(value);
    if (tag === '[object Date]') detail = ` ${(value as Date).toISOString()}`;
    else if (tag === '[object Map]' || tag === '[object Set]') detail = ` size ${(value as Map<unknown, unknown>).size}`;
    else if (typeof value === 'function') detail = (value as Function).length ? ` (${(value as Function).length} parameters)` : '';
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
 * A value beyond the page budget is cut off with a comment naming the read_page ID that holds all of it.
 * Undefined when no expression produces the value (an opaque host object).
 */
function scopeExpression(value: unknown, root: Folder | undefined, pages: PageStore, budget = PAGE_CHARS): string | undefined {
  const whole = () => {
    let text: string;
    try { text = typeof value === 'string' ? value : JSON.stringify(value) ?? String(value); } catch { text = String(value); }
    const { id, count } = pages.add(text);
    return `the whole value is ${count === 1 ? 'one page' : `${count} pages`}: read_page("${id}", 1)`;
  };
  const sequence = <T,>(items: T[], open: string, close: string, noun: string, render: (item: T, left: number) => string | undefined) => {
    const shown: string[] = [];
    let used = 0;
    for (const item of items) {
      if (used >= budget) break;
      const text = render(item, budget - used);
      if (text === undefined) return undefined;
      shown.push(text); used += text.length + 2;
    }
    if (shown.length === items.length) return `${open}${shown.join(', ')}${close}`;
    return `${open}${shown.join(', ')}${shown.length ? ', ' : ''}/* cut off: ${shown.length} of ${items.length} ${noun} shown; ${whole()} */${close}`;
  };
  if (value === null || typeof value === 'boolean') return String(value);
  if (typeof value === 'number') return Number.isFinite(value) ? JSON.stringify(value) : String(value);
  if (typeof value === 'string') {
    if (value.length <= budget) return JSON.stringify(value);
    return `${JSON.stringify(value.slice(0, Math.max(0, budget)))} /* cut off: ${Math.max(0, budget)} of ${value.length} characters shown; ${whole()} */`;
  }
  if (typeof value !== 'object' || value === undefined) return undefined;
  if (value instanceof FileHandle || value instanceof FolderHandle) {
    if (!root || value.folder !== root) return undefined;
    return value instanceof FileHandle ? `folder.file(${JSON.stringify(value.path)})` : value.path ? `folder.dir(${JSON.stringify(value.path)})` : 'folder';
  }
  const tag = Object.prototype.toString.call(value);
  if (tag === '[object Date]') return `new Date(${JSON.stringify((value as Date).toISOString())})`;
  if (tag === '[object Uint8Array]') return sequence(Array.from(value as Uint8Array), 'new Uint8Array([', '])', 'bytes', item => String(item));
  if (tag === '[object Set]') return sequence([...(value as Set<unknown>)], 'new Set([', '])', 'members',
    (item, left) => scopeExpression(item, root, pages, left));
  if (tag === '[object Map]') return sequence([...(value as Map<unknown, unknown>)], 'new Map([', '])', 'entries', ([key, item], left) => {
    const keyText = scopeExpression(key, root, pages, left);
    const itemText = keyText === undefined ? undefined : scopeExpression(item, root, pages, left - keyText.length);
    return itemText === undefined ? undefined : `[${keyText}, ${itemText}]`;
  });
  if (Array.isArray(value)) return sequence(value, '[', ']', 'items', (item, left) => scopeExpression(item, root, pages, left));
  if (!isPlainRecord(value)) return undefined;
  return sequence(Object.entries(value), '{ ', ' }', 'fields', ([key, item], left) => {
    const text = scopeExpression(item, root, pages, left);
    return text === undefined ? undefined : `${/^[A-Za-z_$][\w$]*$/.test(key) ? key : JSON.stringify(key)}: ${text}`;
  });
}

/** The native model loop. Program state stays in NativeSession, never in the model history. */
export class NativeToolAgent {
  readonly proposals: Record<string, unknown>[] = [];
  readonly reviews: Record<string, unknown>[] = [];
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; maxSeconds?: number; systemPrompt?: string | (() => string);
      review?: NativeReviewOptions;
      maxFailureRepairs?: number;
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
      tool('eval', 'Run TypeScript in this call\'s persistent scope. Declarations persist. A top-level return value of the declared type is staged as the call\'s result; the final expression is only shown.',
        { code: { type: 'string' }, timeout_ms: { type: 'integer', minimum: 1,
          description: 'Optional: fail this eval if it has not finished after this many milliseconds.' } }, ['code']),
      tool('read_page', 'Read one page of output that a tool result cut off, by the ID and page number that result names.',
        { id: { type: 'string' }, page: { type: 'integer', minimum: 1 } }, ['id', 'page']),
      tool('compact_history', 'Shorten this conversation. Older tool outputs and eval code are replaced by references, and the ' +
        'full history of this call stays in your eval scope as transcript (every call with its code and complete output), where ' +
        'you can inspect it with code. Your note is kept right after the instructions (a newer note replaces it) and is what you ' +
        'continue from, so it need not repeat details: say what you are doing and what is left, and refer to transcript for the rest.',
        { note: { type: 'string', maxLength: COMPACTION_NOTE_CHARS,
          description: 'What you are doing, what you have found, and what is left; details can stay in transcript.' } }, ['note']),
      tool('return_result', 'Finish the call. With status "success", value is the result and must have the declared return type. ' +
        'With status "blocked" (required information is missing; do not guess) or "failed" (the instructions require an invalid ' +
        'or contradictory operation), give the reason instead of a value.',
        { status: { type: 'string', enum: ['success', 'blocked', 'failed'] },
          value: session.lam.type.kind === 'lambda' ? schemaOf(session.lam.type.returns, session.env) : {},
          reason: { type: 'string', description: 'For "blocked": what is missing. For "failed": why it cannot be done.' } }, ['status']),
    ];
    if (Object.keys(session.lam.codebase).length) tools.splice(2, 0,
      tool('read_function', 'Read the source of an imported function by its listed name.',
        { name: { type: 'string' } }, ['name']),
      tool('edit_function', 'Replace one exact or uniquely fuzzy span in an imported function source. The function is validated before the edit becomes live.',
        { name: { type: 'string' }, find: { type: 'string' }, replace_with: { type: 'string' }, fuzzy: { type: 'boolean' } },
        ['name', 'find', 'replace_with']),
      tool('diff_functions', 'Inspect source changes made to imported functions in this call.', {}, []));
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
        { path: { type: 'string' } }, [])];
      tools.splice(2, 0, ...fileTools);
    }
    return tools;
  }

  tools(session: NativeSession): unknown[] {
    return this.toolsScope(session);
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
    return [`You are inside this call: ${signature}`, ...scopeTypes, '', 'Instructions:', program,
      ...(writable.length ? ['', `Assignments to ${writable.join(', ')} are written back to the caller.`] : []),
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
  private scopeReading(session: NativeSession): { code: string; text: string } | undefined {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return;
    const lines: string[] = [], names: string[] = [];
    const root = lam.projectTransaction?.folder;
    const section = (heading: string, body: string[]) => { if (body.length) lines.push(...(lines.length ? [''] : []), heading, ...body); };
    const declared = (keyword: string, name: string, type: string, value: Value, note = ''): string => {
      // Host types print as their tag; only a class-like tag (FileHandle, Map) is a usable TypeScript type.
      const shown = /^[a-z]+$/.test(type) && !['string', 'number', 'boolean', 'null'].includes(type) ? 'unknown' : type;
      const expression = scopeExpression(value, root, session.pages);
      names.push(name);
      return expression === undefined ?
        `declare ${keyword === 'let' ? 'let' : 'const'} ${name}: ${shown};  // live value ${previewValue(value)}${note}` :
        `${keyword} ${name}: ${shown} = ${expression};${note}`;
    };
    section('// Functions you can call:', this.callableDeclarations(session));
    section('// Provided by the host:', [
      ...Object.entries(session.runtime.services).map(([name, service]) =>
        `declare const ${name}: { ${Object.keys(service as object).map(key => `${key}: Function`).join('; ')} };  // service; its calls are recorded as effects`),
      ...(lam.projectTransaction ? [...FOLDER_DECLARATIONS, 'declare const folder: Folder;  // your working copy of the input folder'] : []),
    ]);
    const params = lam.type.params.fields.map(field => field.name);
    if (params.length) {
      names.push(...params);
      section('// This call\'s arguments, as its caller gave them:', ['const inputs = read_inputs();', ...lam.type.params.fields.map(field =>
        `const ${field.name}: ${formatType(field.type)}${field.optional ? ' | undefined' : ''} = inputs.${field.name};`)]);
    }
    section('// Variables of the calling code, captured by this call:', Object.values(lam.captures ?? {}).flatMap(cell => {
      let value: Value;
      try { value = cell.get() as Value; } catch { return []; }
      return [declared(cell.mutable ? 'let' : 'const', cell.name, cell.type.startsWith('Live<') ? 'object' : cell.type, value,
        cell.mutable ? ' // assignments are written back to the caller' : '')];
    }));
    section('// Your variables from earlier in this call:', Object.entries(lam.let).map(([name, value]) =>
      declared(session.localMutable(name) ? 'let' : 'const', name, formatType(lam.letTypes[name]!), value)));
    if (lam.return !== MISSING)
      section('// Your staged result:', [`// ${scopeExpression(lam.return, root, session.pages) ?? previewValue(lam.return)}`]);
    if (!lines.length) return;
    // The arguments appear in the eval's result, not as literals in its code: they come from the caller.
    return { code: lines.join('\n'), text: (params.length ? inputsListing(session) + '\n' : '') +
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


  async run(session: NativeSession): Promise<string | void> {
    // Fixed for the whole call, so the server can reuse its prompt cache across turns.
    const systemPrompt = () => (typeof this.options.systemPrompt === 'function'
      ? this.options.systemPrompt() : this.options.systemPrompt ?? TOOLS_PROMPT) +
      (Object.keys(session.lam.codebase).length ? FUNCTION_TOOLS_PROMPT : '') +
      (session.lam.subtype === 'directory-reducer' ? DIRECTORY_REDUCER_PROMPT : '');
    const openingMessages = (): Record<string, unknown>[] => {
      const reading = this.scopeReading(session);
      return [{ role: 'system', content: systemPrompt() },
        { role: 'user', content: this.scopeOpening(session) },
        ...(reading ? [{ role: 'assistant', content: '', tool_calls: [{ id: 'scope_0', type: 'function',
          function: { name: 'eval', arguments: JSON.stringify({ code: reading.code }) } }] },
        { role: 'tool', tool_call_id: 'scope_0', content: reading.text }] : []),
        ...(session.lam.projectTransaction ? [{ role: 'assistant', content: '', tool_calls: [{ id: 'scope_1', type: 'function',
          function: { name: 'list_files', arguments: '{}' } }] },
        { role: 'tool', tool_call_id: 'scope_1', content: this.folderListing(session) }] : [])];
    };
    const messages = openingMessages();
    const openingLength = messages.length;
    const budget = this.options.contextTokens === undefined ? DEFAULT_CONTEXT_TOKENS : this.options.contextTokens;
    // Prompt tokens per character of request, calibrated from the server's reported prompt size.
    let tokensPerChar = 1 / 3.5;
    const requestChars = (tools: unknown[]) => JSON.stringify(messages).length + JSON.stringify(tools).length;
    // Tool call id -> index in session.transcript, for compaction stubs.
    const transcriptEntries = new Map<string, number>();
    // Messages compaction never touches: the opening, and the latest compaction note once there is one.
    let protectedLength = openingLength;
    // Estimated prompt size right after the last compaction.
    let compactedAt = 0;
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
      const limit = allowance();
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
      const nearLimit = budget !== null && estimate(allTools) > Math.max(budget * 0.75, compactedAt + budget * 0.25);
      const availableTools = lastTurn ? only('return_result') : nearLimit ? only('compact_history') : allTools;
      if (availableTools !== allTools && !lastTurn) {
        // Say why only compact_history is offered, on the latest tool result, as the turns-left notice does.
        const latest = messages.at(-1);
        if (latest?.role === 'tool' && typeof latest.content === 'string' && !latest.content.includes(COMPACTION_NOTICE))
          messages[messages.length - 1] = { ...latest, content: latest.content + COMPACTION_NOTICE };
      }
      if (budget !== null && estimate(availableTools) > budget) {
        // A request never exceeds the budget: if the model has not compacted, the oldest outputs are elided without
        // a note. Program state lives in the eval scope and every output in transcript, so no values are lost.
        // Keep the latest exchanges if that is enough; otherwise keep only the last call and its result.
        let elided = 0;
        for (const recent of [RECENT_MESSAGES, 2])
          if (estimate(availableTools) > budget * 0.5)
            elided += compactMessages(messages, protectedLength, recent, () => estimate(availableTools) > budget * 0.5,
              callId => transcriptEntries.get(callId));
        compactedAt = estimate(allTools);
        session.runtime.trace.emit('compaction', { call_id: session.runtime.currentCallId ?? null, turn: turns + 1,
          elided, note: null, estimated_tokens: Math.round(estimate(availableTools)) });
      }
      const sentChars = requestChars(availableTools);
      const callId = session.runtime.currentCallId ?? null;
      const started = performance.now();
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'start', turn: turns + 1,
        tool_schema_bytes: new TextEncoder().encode(JSON.stringify(availableTools)).length,
        messages: messages.length });
      let response: ModelTurn;
      try {
        response = await this.driver({ messages, tools: availableTools,
          // A turn that offers one tool it must use (the compaction turn, the last turn) requires a tool call.
          ...(availableTools !== allTools ? { tool_choice: 'required' as const } : {}),
          ...(this.options.temperature === undefined ? {} : { temperature: this.options.temperature }),
          seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.runtime.options.runId, session.lam.attempts, 'model-turn', turns),
          max_tokens: limit });
      } catch (error) {
        session.runtime.trace.emit('model_request', { call_id: callId, phase: 'error', turn: turns + 1,
          duration_ms: Math.round(performance.now() - started),
          error: `${error instanceof Error ? error.name : 'Error'}: ${error instanceof Error ? error.message : String(error)}` });
        throw error;
      }
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'end', turn: turns + 1,
        duration_ms: Math.round(performance.now() - started),
        prompt_tokens: response.prompt_tokens ?? null, completion_tokens: response.completion_tokens ?? null });
      if (response.prompt_tokens !== undefined && sentChars > 0) tokensPerChar = response.prompt_tokens / sentChars;
      turns++;
      session.runtime.checkInterruption();
      const calls = response.calls ?? [];
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'generated', turn: turns, calls, text: response.text ?? '' });
      tokens += response.completion_tokens === undefined ? limit ?? 0 : Math.max(1, response.completion_tokens);
      if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
        return 'episode token or wall-clock budget exhausted';
      if (!response.calls?.length && availableTools !== allTools && !lastTurn) {
        // The compaction turn was answered without the tool: its text is not a result. Compact without a note.
        const elided = compactMessages(messages, protectedLength, RECENT_MESSAGES, () => estimate(allTools) > budget! * 0.5,
          callId => transcriptEntries.get(callId));
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
        const feedback = response.truncated ? `Your reply was cut off at the ${limit}-token limit before any tool call. Take the next step with one tool call.` :
          missing || 'The staged result is incomplete.';
        messages.push({ role: 'assistant', content: response.text ?? '' }, { role: 'user', content: feedback });
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
          ['eval', 'edit_function', 'edit_file', 'write_file'].includes(name);
        if (!lowValue && !structural) continue;
        if (exhausted())
          return 'careful review budget exhausted before applying proposal';
        const budget = allowance();
        const fork = [...messages, { role: 'user', content: this.reviewPrompt(messages, calls, index) }];
        const answer = await (review.driver ?? this.driver)({ messages: fork, tools: this.reviewTools(),
          temperature: 0, seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.runtime.options.runId, session.lam.attempts, 'review', turns),
          max_tokens: budget });
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
      const previousFailureSerial = session.failureSerial;
      for (const [index, [name, args]] of calls.entries()) {
        if (timedOut()) return 'episode wall-clock budget exhausted';
        const result: NativeResult = await session.applyAsync(name, args);
        results.push(result);
        if (result.kind === 'blocked') return result.text;
        if (['blocked', 'budget', 'completed'].includes(result.kind)) break;
      }
      messages.push({ role: 'assistant', content: '', tool_calls: raw.slice(0, results.length) });
      // Near the end of the turn budget the model is told how many turns are left, so a task that cannot be finished
      // ends with an honest blocked or failed rather than by running out.
      const left = maxTurns === undefined ? Infinity : maxTurns - turns;
      const notice = left === 1 ? '\n\n[This is your last turn in this call: call return_result with status "success" and the result, or status "blocked" with what is missing, or status "failed" with why.]' :
        left <= 4 && left > 0 ? `\n\n[${left} turns left in this call. If the task cannot be finished, call return_result with status "blocked" and what is missing, or status "failed" and why.]` : '';
      let note: string | undefined;
      for (const [index, result] of results.entries()) {
        const [name, args] = calls[index]!, id = String(raw[index]!.id);
        messages.push({ role: 'tool', tool_call_id: id, content: result.text + (index === results.length - 1 ? notice : '') });
        transcriptEntries.set(id, session.transcript.length);
        session.transcript.push({ turn: turns, tool: name, ...(name === 'eval' && typeof args.code === 'string' ? { code: args.code } : {}),
          arguments: structuredClone(args), output: session.pages.expand(result.text) });
        if (name === 'compact_history' && result.kind === 'ok') note = String(args.note).trim();
      }
      if (note !== undefined) {
        // Everything older than the latest exchange moves to transcript; the note is kept after the opening.
        const pinned = { role: 'user', content: `Your note from compacting this conversation: ${note}\n\n` +
          'Continue from where this note leaves off. The full history of this call is in transcript; look something up there ' +
          'only when you have a specific question about it.' };
        if (protectedLength > openingLength) messages[openingLength] = pinned;
        else { messages.splice(openingLength, 0, pinned); protectedLength = openingLength + 1; }
        // The note speaks for everything before it: only the compaction call and its result stay whole.
        const elided = compactMessages(messages, protectedLength, 2, () => true, callId => transcriptEntries.get(callId));
        compactedAt = requestChars(this.tools(session)) * tokensPerChar;
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
