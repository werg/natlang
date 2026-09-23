import { fitsType, formatType, parseType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, dump, isPending, problems } from './values.js';
import type { Value } from './values.js';
import type { NativeResult, NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { deriveSeed } from './trace.js';
import { DIRECTORY_REDUCER_PROMPT, EXPLICIT_TOOLS_PROMPT } from './prompt.js';
import { isLazyDict } from './host-tree.js';

export type NativeModelDriver = (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
export type NativeReviewOptions = { driver?: NativeModelDriver; threshold?: number;
  scope?: 'values' | 'actions'; withdrawalPolicy?: 'caller' | 'retry';
  prompt?: 'baseline' | 'repeat_instructions' | 'checklist';
  order?: 'reason_first' | 'decision_first' };

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  type: 'function', function: { name, description, parameters: { type: 'object', properties,
    required, additionalProperties: false } },
});

const CHECKPOINT_REQUEST = 'Before continuing this same task in a fresh conversation, leave yourself a concise working note. ' +
  'State only unresolved decisions or facts that are not obvious from the program and workspace. ' +
  'For an unfinished loop, name its current accumulator path and rounds completed; never restart from its initial value. ' +
  'The workspace, line marks, and effects will be shown again; do not restate them. ' +
  'Do not execute a tool or claim the task is finished. Reply with the note only, at most 800 characters.';


function schemaOf(type: Type, env: TypeEnv, depth = 0): Record<string, unknown> {
  if (depth > 5) return {};
  const resolved = env.resolve(type);
  if (resolved.kind === 'prim') return { type: { string: 'string', Blob: 'string', number: 'number',
    boolean: 'boolean', null: 'null', Folder: 'object', FileHandle: 'object' }[resolved.name],
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

function textSpans(text: string, maximum = 96): string[] {
  const lines = text.match(/.*(?:\r?\n|$)/g)?.filter(Boolean) ?? [], out: string[] = [];
  for (let width = 1; width <= 3; width++) for (let start = 0; start + width <= lines.length; start++) {
    const span = lines.slice(start, start + width).join('');
    if (span.trim() && span.length <= 600 && text.split(span).length === 2) {
      out.push(span);
      const trimmed = span.replace(/[\r\n]+$/, '');
      if (trimmed && trimmed !== span && text.split(trimmed).length === 2) out.push(trimmed);
    }
  }
  return [...new Set(out)].sort((a, b) => a.length - b.length || text.indexOf(a) - text.indexOf(b)).slice(0, maximum);
}

function mergedSchemas(values: unknown[], fallback: Record<string, unknown> = {}): Record<string, unknown> {
  const unique = [...new Map(values.filter(value => value && typeof value === 'object')
    .map(value => [JSON.stringify(value), value as Record<string, unknown>])).values()];
  return unique.length === 1 ? structuredClone(unique[0]!) : unique.length ? { anyOf: structuredClone(unique) } : fallback;
}

function pythonJson(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(pythonJson).join(', ')}]`;
  if (value && typeof value === 'object') return `{${Object.entries(value)
    .map(([key, item]) => `${JSON.stringify(key)}: ${pythonJson(item)}`).join(', ')}}`;
  return JSON.stringify(value);
}

function pendingLine(value: Extract<Value, { nodeKind: string }>): string {
  let line = `${formatType(value.type)}  ${value.status}`;
  if (value.nodeKind === 'map' && value.slots) line += `  ${value.slots.filter(item => !isPending(item)).length} of ${value.slots.length} reduced`;
  if (value.nodeKind === 'fold' && value.acc !== MISSING) line += `  at ${value.at} of ${Array.isArray(value.over) ? value.over.length : '?'}`;
  if (value.nodeKind === 'iterate' && value.state !== MISSING) line += `  iteration ${value.iteration} of max ${String(value.max)}`;
  if (value.status === 'quiesced' && value.note) line += `  "${value.note.slice(0, 60)}"`;
  return line;
}

function previewValue(value: Value): string {
  if (isLazyDict(value)) return `[${value.label}; lazy read-only record]`;
  if (isPending(value)) return `[${pendingLine(value)}]`;
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
  if (value && typeof value === 'object')
    return `{ ${Object.entries(value).slice(0, 6).map(([key, item]) => `${key}: ${previewValue(item)}`).join(', ')} }`;
  if (value === null) return 'null';
  return String(value);
}

function scopePreviewValue(value: Value): string {
  if (isLazyDict(value)) return `[${value.label}; inspect with read_value or use it in eval]`;
  if (!isPending(value)) {
    try {
      const encoded = JSON.stringify(value);
      if (encoded !== undefined && encoded.length <= 800) return encoded;
    } catch { /* fall back to the bounded structural preview */ }
  }
  return previewValue(value);
}

function scopeProgramListing(body: string, marks: Record<number, string>): string {
  const lines = body.replace(/^\n+|\n+$/g, '').split('\n'), width = String(lines.length).length;
  return lines.map((line, index) => {
    const number = index + 1, text = line.trim();
    const markable = !!text && !text.startsWith('#') && !text.startsWith('function ');
    const box = !markable ? '   ' : marks[number] === 'done' ? '[x]' : marks[number] === 'skipped' ? '[-]' : '[ ]';
    return `${String(number).padStart(width)} ${box} ${line}`.trimEnd();
  }).join('\n');
}

function stateParts(value: Value, type: Type, env: TypeEnv, path: string,
  filled: string[], todo: string[], subs: string[], depth = 0): void {
  if (value === MISSING) { todo.push(`${path} (${formatType(type)})`); return; }
  if (isPending(value)) { subs.push(`${path} [${pendingLine(value)}]`); return; }
  const resolved = env.resolve(type);
  if (resolved.kind === 'record' && value && typeof value === 'object' && !Array.isArray(value) && depth < 3) {
    for (const field of resolved.fields) {
      if (Object.hasOwn(value, field.name)) stateParts((value as Record<string, Value>)[field.name]!,
        field.type, env, `${path}/${field.name}`, filled, todo, subs, depth + 1);
      else if (!field.optional) todo.push(`${path}/${field.name} (${formatType(field.type)})`);
    }
    return;
  }
  if (Array.isArray(value) && value.some(isPending)) {
    subs.push(`${path} [${value.filter(item => !isPending(item)).length} of ${value.length} items done]`);
    return;
  }
  filled.push(`${path} = ${previewValue(value)}`);
}

/** The native model loop. Program state stays in NativeSession, never in the model history. */
export class NativeToolAgent {
  readonly proposals: Record<string, unknown>[] = [];
  readonly reviews: Record<string, unknown>[] = [];
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; maxSeconds?: number; systemPrompt?: string;
      validationFeedback?: 'caller' | 'local'; review?: NativeReviewOptions;
      segmentTurns?: number | null; segmentMessages?: number | null } = {}) {
    if (options.segmentTurns !== undefined && options.segmentTurns !== null &&
        (!Number.isInteger(options.segmentTurns) || options.segmentTurns < 1))
      throw new RangeError('segmentTurns must be positive or null');
    if (options.segmentMessages !== undefined && options.segmentMessages !== null &&
        (!Number.isInteger(options.segmentMessages) || options.segmentMessages < 5))
      throw new RangeError('segmentMessages must be at least 5 or null');
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

  private pendingMarks(session: NativeSession): number[] {
    return this.unmarkedLines(session);
  }

  private unmarkedLines(session: NativeSession): number[] {
    return (session.lam.originalBody ?? session.lam.body).replace(/^\n+|\n+$/g, '').split('\n')
      .flatMap((line, index) => {
        const text = line.trim(), number = index + 1;
        return text && !text.startsWith('#') && !text.startsWith('function ') &&
          !Object.hasOwn(session.lam.marks, number) ? [number] : [];
      });
  }

  private toolsScope(session: NativeSession): any[] {
    const tools = [
      tool('eval', 'Execute TypeScript in the persistent scope. Parameters and declarations persist. A compatible final expression becomes the function result.',
        { code: { type: 'string' } }, ['code']),
      tool('read_value', 'Inspect a variable or field/index selection without executing code.',
        { expression: { type: 'string' },
          start: { type: 'integer', minimum: 0, description: 'Zero-based character offset for string or item index for a list.' },
          end: { type: 'integer', minimum: 0, description: 'Exclusive character or item offset, as in JavaScript slice().' } }, ['expression']),
      tool('mark_lines', 'Close one instruction line or inclusive contiguous range after its work succeeded. Use skipped only for an untaken branch.',
        { start: { type: 'integer' }, end: { type: 'integer' }, skipped: { type: 'boolean' } }, ['start']),
      tool('report_blocker', 'End without a result because required information is missing. Do not guess.',
        { missing: { type: 'string' } }, ['missing']),
      tool('report_error', 'End without a result because the instructions require an invalid or contradictory operation.',
        { message: { type: 'string' } }, ['message']),
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
      tool('commit', 'Set the typed result and retain selected folder changes. Include and exclude entries are relative glob patterns.',
        { value: schemaOf(session.lam.type.kind === 'lambda' ? session.lam.type.returns : parseType('null'), session.env),
          include: { type: 'array', items: { type: 'string' } },
          exclude: { type: 'array', items: { type: 'string' } } }, ['value']),
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
    session.surfaceName = 'scope-eval-v1';
    if (!this.missing(session) && !this.unmarkedLines(session).length) return [];
    return this.toolsScope(session);
  }

  opening(session: NativeSession): string {
    const lam = session.lam, type = lam.type;
    if (type.kind !== 'lambda') return 'Workspace:';
    const lines = ['Workspace:'];
    if (lam.continuationNote)
      lines.push('  Earlier working note (check against the workspace): ' + lam.continuationNote);
    if (lam.journal.length) {
      const omitted = Math.max(0, lam.journal.length - 4);
      lines.push(`  Effects already attempted: ${lam.journal.length}` +
        (omitted ? ` (last 4 shown; read args@effects for all ${lam.journal.length})` : ''));
      for (const entry of lam.journal.slice(-4)) lines.push('    ' + pythonJson(entry));
    }
    for (const field of type.params.fields) lines.push(`  args/${field.name} (${formatType(field.type)}, read-only): ` +
      (Object.hasOwn(lam.args, field.name) ? previewValue(lam.args[field.name]!) : 'not supplied'));
    for (const [name, localType] of Object.entries(lam.letTypes)) {
      const value = lam.let[name];
      if (value !== undefined && value !== MISSING)
        lines.push(`  let/${name} (${name in lam.fnCopies ? `a copy of ${name}, editable` : formatType(localType)}): ` +
          (name in lam.fnCopies ? '' : previewValue(value)));
    }
    const filled: string[] = [], todo: string[] = [], subs: string[] = [];
    stateParts(lam.return, type.returns, session.env, 'return', filled, todo, subs);
    lines.push(`  return (${formatType(type.returns)}): ` +
      (lam.return === MISSING ? 'not written yet' : todo.length || subs.length ? 'partly written' : 'written'));
    if (filled.length && (todo.length || subs.length)) lines.push(`    written so far: ${filled.join('; ')}`);
    if (todo.length && lam.return !== MISSING) lines.push(`    still missing: ${todo.join(', ')}`);
    if (subs.length) lines.push(`    sub-tasks: ${subs.join('; ')}`);
    return lines.join('\n');
  }

  private scopeOpening(session: NativeSession): string {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return 'Scope:';
    const original = lam.originalBody ?? lam.body;
    const program = scopeProgramListing(original, lam.marks);
    const inputs = lam.type.params.fields.map(field => `  ${field.name}: ${formatType(field.type)} = ` +
      (Object.hasOwn(lam.args, field.name) ? scopePreviewValue(lam.args[field.name]!) : 'missing'));
    const imports = Object.entries(lam.codebase).filter(([, raw]) =>
      lam.subtype === 'directory-reducer' || (raw as Record<string, unknown>).subtype !== 'directory-reducer').map(([name, raw]) => {
      const fn = raw as Record<string, unknown>;
      const kind = fn.subtype === 'directory-reducer' ? 'directory reducer' :
        Object.hasOwn(fn, 'code') ? 'TypeScript' : 'natural language';
      const parameters = Object.entries(fn.args as Record<string, string> ?? {})
        .map(([key, value]) => `${key.replace(/\?$/, '')}: ${value}`);
      if (fn.subtype === 'directory-reducer') parameters.unshift('folder: Folder');
      return `  ${name} [${kind}] (${parameters.join(', ')}): Promise<${fn.returns}>`;
    });
    const locals = Object.entries(lam.let).map(([name, value]) =>
      `  ${name}: ${formatType(lam.letTypes[name]!)} = ${scopePreviewValue(value)}`);
    return ['Execute the natural-language function line by line.', '', 'Program:', program, '',
      'Scope:',
      ' parameters',
      ...(inputs.length ? inputs : ['  (none)']),
      ' imports (immutable live bindings)', ...(imports.length ? imports : ['  (none)']),
      ' locals', ...(locals.length ? locals : ['  (none)']),
      ` result: ${formatType(lam.type.returns)} — ${lam.return === MISSING ? 'unset' : 'set'}`].join('\n');
  }

  missing(session: NativeSession): string {
    const lam = session.lam;
    if (lam.type.kind !== 'lambda') return '';
    if (lam.return === MISSING) return `\`return\` has not been written yet. Write a ${formatType(lam.type.returns)} to \`return\`.`;
    const filled: string[] = [], todo: string[] = [], subs: string[] = [];
    stateParts(lam.return, lam.type.returns, session.env, 'return', filled, todo, subs);
    if (subs.length) return `A sub-task has not been run yet: ${subs.map(item => item.split(' [')[0]).join(', ')}. Run it.`;
    return todo.length ? `\`return\` is missing: ${todo.join(', ')}.` : '';
  }

  async run(session: NativeSession): Promise<string | void> {
    const openingMessages = (): Record<string, unknown>[] => [
      { role: 'system', content: (this.options.systemPrompt ?? EXPLICIT_TOOLS_PROMPT) +
        (session.lam.subtype === 'directory-reducer' ? DIRECTORY_REDUCER_PROMPT : '') },
      { role: 'user', content: this.scopeOpening(session) },
    ];
    const messages = openingMessages();
    const maxTurns = this.options.maxTurns, maxTokens = this.options.maxTokens;
    const deadline = this.options.maxSeconds === undefined ? null : Date.now() + this.options.maxSeconds * 1000;
    let tokens = 0, nudges = 0, turns = 0, withdrawals = 0, segmentTurns = 0;
    let checkpointReady = true;
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
      const rollover = this.options.segmentTurns === undefined ? 6 : this.options.segmentTurns;
      const itemLimit = this.options.segmentMessages === undefined ? 12 : this.options.segmentMessages;
      if (((rollover !== null && segmentTurns >= rollover) ||
           (itemLimit !== null && messages.length >= itemLimit)) && checkpointReady &&
          (this.missing(session) || this.pendingMarks(session).length)) {
        const budget = allowance();
        const checkpointLimit = budget === null ? 512 : Math.min(512, budget);
        const checkpointMessages = [...messages, { role: 'user', content: CHECKPOINT_REQUEST }];
        const callId = session.runtime.currentCallId ?? null;
        const started = performance.now();
        session.runtime.trace.emit('model_request', { call_id: callId, phase: 'start',
          purpose: 'checkpoint', turn: turns + 1, messages: checkpointMessages.length });
        const response = await this.driver({ messages: checkpointMessages, tools: [],
          temperature: this.options.temperature ?? 0.2,
          seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'checkpoint', turns),
          max_tokens: checkpointLimit });
        session.runtime.trace.emit('model_request', { call_id: callId, phase: 'end',
          purpose: 'checkpoint', turn: turns + 1, duration_ms: Math.round(performance.now() - started),
          prompt_tokens: response.prompt_tokens ?? null, completion_tokens: response.completion_tokens ?? null });
        turns++;
        tokens += response.completion_tokens === undefined ? checkpointLimit :
          Math.max(1, response.completion_tokens);
        session.runtime.checkInterruption();
        if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
          return 'episode token or wall-clock budget exhausted';
        session.lam.continuationNote = (response.text ?? '').trim().slice(0, 800);
        session.runtime.trace.emit('checkpoint', { call_id: callId, note: session.lam.continuationNote, turn: turns });
        session.runtime.observeState('after-checkpoint');
        messages.splice(0, messages.length, ...openingMessages());
        segmentTurns = 0;
        checkpointReady = true;
        continue;
      }
      const limit = allowance();
      const availableTools = this.tools(session);
      const callId = session.runtime.currentCallId ?? null;
      const started = performance.now();
      session.runtime.trace.emit('model_request', { call_id: callId, phase: 'start', turn: turns + 1,
        tool_schema_bytes: new TextEncoder().encode(JSON.stringify(availableTools)).length,
        messages: messages.length });
      let response: ModelTurn;
      try {
        response = await this.driver({ messages, tools: availableTools,
          temperature: this.options.temperature ?? 0.2,
          seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'model-turn', turns),
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
      turns++;
      segmentTurns++;
      session.runtime.checkInterruption();
      const calls = response.calls ?? [];
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'generated', turn: turns, calls, text: response.text ?? '' });
      tokens += response.completion_tokens === undefined ? limit ?? 0 : Math.max(1, response.completion_tokens);
      if (timedOut() || (maxTokens !== undefined && tokens > maxTokens))
        return 'episode token or wall-clock budget exhausted';
      if (!response.calls?.length) {
        if (this.options.validationFeedback !== 'local' && this.missing(session))
          return `validation failed: ${this.missing(session)}`;
        const marks = this.pendingMarks(session);
        if (marks.length) {
          if (++nudges > 2) return `validation failed: unfinished lines: ${marks.join(', ')}`;
          messages.push({ role: 'assistant', content: response.text ?? '' },
            { role: 'user', content: `Lines still marked [ ]: ${marks.join(', ')}. Mark completed work done and untaken work skipped (skipped=true).` });
          checkpointReady = false;
          continue;
        }
        if (session.finish()) { session.lam.note = response.text ?? ''; return; }
        if (this.options.validationFeedback !== 'local') return `validation failed: ${this.missing(session)}`;
        if (++nudges > 2) return `replied without writing \`return\`: ${(response.text ?? '').slice(0, 280)}`;
        const missing = this.missing(session);
        messages.push({ role: 'assistant', content: response.text ?? '' },
          { role: 'user', content: missing || 'return is incomplete' });
        checkpointReady = false;
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
          ['eval', 'edit_function', 'edit_file', 'write_file', 'commit', 'mark_lines'].includes(name);
        if (!lowValue && !structural) continue;
        if (exhausted())
          return 'careful review budget exhausted before applying proposal';
        const budget = allowance();
        const fork = [...messages, { role: 'user', content: this.reviewPrompt(messages, calls, index) }];
        const answer = await (review.driver ?? this.driver)({ messages: fork, tools: this.reviewTools(),
          temperature: 0, seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'review', turns),
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
          checkpointReady = false;
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
      for (const [index, [name, args]] of calls.entries()) {
        if (timedOut()) return 'episode wall-clock budget exhausted';
        const result: NativeResult = await session.applyAsync(name, args);
        results.push(result);
        if (result.kind === 'blocked') return result.text;
        if (['blocked', 'budget', 'completed'].includes(result.kind)) break;
      }
      messages.push({ role: 'assistant', content: '', tool_calls: raw.slice(0, results.length) });
      for (const [index, result] of results.entries())
        messages.push({ role: 'tool', tool_call_id: raw[index]!.id, content: result.text });
      if (!this.missing(session) && !this.pendingMarks(session).length && session.finish()) return;
      checkpointReady = !results.some(result => ['rejected', 'refused', 'error'].includes(result.kind)) &&
        ['eval', 'edit_file', 'mark_lines', 'commit'].includes(calls[results.length - 1]?.[0] ?? '');
      if (results.at(-1)?.kind === 'budget') return 'action or tool-call budget exhausted';
      const failed = results.find(result => ['rejected', 'refused'].includes(result.kind));
      if (this.options.validationFeedback !== 'local' && failed)
        return `validation failed: ${failed.text}`;
      if (results.at(-1)?.kind === 'completed') return;
    }
  }
}
