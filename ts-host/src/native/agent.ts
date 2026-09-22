import { fitsType, formatType, parseType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, dump, isPending, problems } from './values.js';
import type { Value } from './values.js';
import type { NativeResult, NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../contracts.js';
import { deriveSeed } from './trace.js';
import { EXPLICIT_TOOLS_PROMPT, TOOLS_PROMPT } from './prompt.js';
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

const newLocal = { type: 'string', 'x-natlang': 'new-local',
  pattern: '^let/[a-z_][a-z0-9_]*$',
  description: 'let/<name>: a new local, created by this call' };
const explicitNewLocal = { type: 'string', 'x-natlang': 'new-local',
  pattern: '^let/[a-z_][a-z0-9_]*$', description: 'a new local destination' };
const futureLocal = { type: 'string', pattern: '^let/[a-z_][a-z0-9_]*$',
  description: 'a local produced by another call in this batch' };
const CHECKPOINT_REQUEST = 'Before continuing this same task in a fresh conversation, leave yourself a concise working note. ' +
  'State only unresolved decisions or facts that are not obvious from the program and workspace. ' +
  'For an unfinished loop, name its current accumulator path and rounds completed; never restart from its initial value. ' +
  'The workspace, line marks, and effects will be shown again; do not restate them. ' +
  'Do not execute a tool or claim the task is finished. Reply with the note only, at most 800 characters.';


function schemaOf(type: Type, env: TypeEnv, depth = 0): Record<string, unknown> {
  if (depth > 5) return {};
  const resolved = env.resolve(type);
  if (resolved.kind === 'prim') return { type: { Text: 'string', Blob: 'string', Num: 'number',
    Bool: 'boolean', Null: 'null', Folder: 'object', FileHandle: 'object' }[resolved.name],
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

type Slot = { path: string; type: Type; value: Value; writable: boolean };
function slots(path: string, type: Type, value: Value, env: TypeEnv, writable: boolean, depth = 0): Slot[] {
  const out: Slot[] = [{ path, type, value, writable }];
  if (isLazyDict(value)) return out;
  if (depth >= 3) return out;
  if (isPending(value)) {
    if (value.nodeKind === 'lambda' && value.type.kind === 'lambda') {
      out.push({ path: `${path}/args`, type: value.type.params, value: value.args as Value,
        writable: false });
      for (const field of value.type.params.fields)
        if (!Object.hasOwn(value.args, field.name)) out.push({ path: `${path}/args/${field.name}`,
          type: field.type, value: MISSING, writable: writable && value.status !== 'running' });
    }
    return out;
  }
  const resolved = env.resolve(type);
  if (resolved.kind === 'record') for (const field of resolved.fields) {
    const child = value && value !== MISSING && typeof value === 'object' && !Array.isArray(value) &&
      Object.hasOwn(value, field.name) ? (value as Record<string, Value>)[field.name]! : MISSING;
    out.push(...slots(`${path}/${field.name}`, field.type, child, env, writable, depth + 1));
  } else if (resolved.kind === 'list' && Array.isArray(value)) {
    value.forEach((child, index) => out.push(...slots(`${path}/${index}`, resolved.element, child, env, writable, depth + 1)));
    if (writable) out.push({ path: `${path}/+`, type: resolved.element, value: MISSING, writable });
  } else if (resolved.kind === 'dict' && value && value !== MISSING && typeof value === 'object' && !Array.isArray(value))
    for (const [key, child] of Object.entries(value))
      out.push(...slots(`${path}/${key}`, resolved.element, child, env, writable, depth + 1));
  return out;
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
      segmentTurns?: number | null; segmentMessages?: number | null;
      toolSchema?: 'tools-v3' | 'tools-v4' | 'scope-eval-v1' } = {}) {
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

  private openMarks(session: NativeSession): number[] {
    if (!Object.keys(session.lam.marks).length) return [];
    return this.unmarkedLines(session);
  }

  private pendingMarks(session: NativeSession): number[] {
    return this.options.toolSchema === 'scope-eval-v1' ? this.unmarkedLines(session) : this.openMarks(session);
  }

  private unmarkedLines(session: NativeSession): number[] {
    return (session.lam.originalBody ?? session.lam.body).replace(/^\n+|\n+$/g, '').split('\n')
      .flatMap((line, index) => {
        const text = line.trim(), number = index + 1;
        return text && !text.startsWith('#') && !text.startsWith('function ') &&
          !Object.hasOwn(session.lam.marks, number) ? [number] : [];
      });
  }

  private toolsV3(session: NativeSession): any[] {
    const names = [...Object.keys(session.lam.codebase), ...Object.keys(session.lam.fnCopies).map(name => `let/${name}`)];
    const lam = session.lam;
    const all: Slot[] = [];
    if (lam.type.kind === 'lambda') {
      all.push(...slots('args', lam.type.params, lam.args as Value, session.env, false));
      all.push(...slots('return', lam.type.returns, lam.return, session.env, true));
    }
    for (const [name, type] of Object.entries(lam.letTypes))
      all.push(...slots(`let/${name}`, type, lam.let[name] ?? MISSING, session.env, true));
    const zone = (path: string) => path.startsWith('let/') ? 0 : path.startsWith('return') ? 1 : 2;
    const rank = (slot: Slot) => [slot.path.split('/').some(part => /^\d+$/.test(part)) ? 1 : 0,
      slot.path.split('/').length > 3 ? 1 : 0, zone(slot.path), slot.path.split('/').length];
    const present = all.filter(slot => slot.value !== MISSING).sort((a, b) => {
      const left = rank(a), right = rank(b);
      for (let index = 0; index < left.length; index++) if (left[index] !== right[index]) return left[index]! - right[index]!;
      return 0;
    });
    const readable = present.map(slot => slot.path);
    const writable = all.filter(slot => slot.writable &&
      !['lambda', 'map', 'fold', 'iterate'].includes(session.env.resolve(slot.type).kind)).slice(0, 48);
    const definable = all.filter(slot => slot.writable).slice(0, 48);
    const textSlots = all.filter(slot => slot.writable && typeof slot.value === 'string').map(slot => slot.path);
    for (const slot of all) if (slot.writable && isPending(slot.value) &&
      slot.value.nodeKind === 'lambda' && slot.value.kind === 'instructions')
      textSlots.push(`${slot.path}/instructions`);
    if (!Object.keys(lam.codebase).length && lam.kind === 'instructions') textSlots.unshift('instructions');
    const path = { type: 'string' };
    const slotPaths = writable.map(slot => slot.path);
    const definitionPaths = definable.map(slot => slot.path);
    const valueSchemas = [...new Map(writable.map(slot => {
      const schema = schemaOf(slot.type, session.env); return [JSON.stringify(schema), schema] as const;
    })).values()];
    const scalarShapes = ['string', 'number', 'boolean', 'null', 'object', 'array'].map(type => ({ type }));
    const writeShapes = [...new Map([...valueSchemas.flatMap(schema => Object.keys(schema).length ? [schema] : scalarShapes),
      ...scalarShapes].map(schema => [JSON.stringify(schema), schema])).values()];
    const writeValue = { description: 'The value itself, complete (not wrapped in an object).', anyOf: writeShapes };
    const fitting = (target: Type): string[] => present.filter(slot =>
      !isPending(slot.value) && fitsType(slot.type, target, session.env)).map(slot => slot.path).slice(0, 48);
    const valueAlternatives = writable.map(slot => ({ path: { const: slot.path },
      type: { const: formatType(slot.type) }, value: schemaOf(slot.type, session.env) }));
    const sourceAlternatives = writable.flatMap(slot => {
      const sources = fitting(slot.type).filter(path => path !== slot.path && !path.startsWith(`${slot.path}/`));
      return sources.length ? [{ path: { const: slot.path }, type: { const: formatType(slot.type) },
        source: { enum: sources } }] : [];
    });
    const writeAlternatives: Record<string, unknown>[] = [...valueAlternatives,
      { path: newLocal, type: { type: 'string' }, value: {} }, ...sourceAlternatives,
      ...Object.keys(lam.codebase).map(name => ({ path: newLocal, type: { const: `Function<${name}>` } }))];
    const readAlternatives: Record<string, unknown>[] = lam.type.kind === 'lambda' && lam.type.params.fields.length ?
      [{ path: { const: 'args' } }] : [];
    if (lam.journal.length) readAlternatives.push({ path: { const: 'args@effects' } });
    for (const slot of present.slice(0, 48)) {
      readAlternatives.push({ path: { const: slot.path } });
      const length = typeof slot.value === 'string' ? slot.value.split('\n').length :
        Array.isArray(slot.value) ? slot.value.length : 0;
      if (length > 1 && length <= 60) {
        const positions = Array.from({ length }, (_, i) => i + (typeof slot.value === 'string' ? 1 : 0));
        readAlternatives.push({ path: { const: slot.path }, start: { enum: positions }, end: { enum: positions } });
      }
    }
    for (const name of Object.keys(lam.codebase)) readAlternatives.push({ path: { const: `codebase/${name}` } });
    const lazyPaths = all.filter(slot => isLazyDict(slot.value)).map(slot => slot.path);
    for (const lazyPath of lazyPaths) readAlternatives.push({
      path: { type: 'string', pattern: `^${lazyPath.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:/.+)?$` },
      start: { type: 'integer' }, end: { type: 'integer' }, 'x-optional': ['start', 'end'] });
    const inputNames = [...new Set(names.flatMap(name => {
      const definition = (name.startsWith('let/') ? lam.fnCopies[name.slice(4)] : lam.codebase[name]) as
        Record<string, unknown> | undefined;
      return Object.keys(definition?.args as Record<string, unknown> ?? {});
    }).map(name => name.replace(/\?$/, '')))];
    const unmarked = names.length ? this.unmarkedLines(session) : [];
    const doneLine = { anyOf: [{ enum: unmarked },
      { type: 'array', items: { enum: unmarked }, minItems: 1, maxItems: 2 }] };
    const done = { ...doneLine,
      description: 'line or inclusive [first, last] range that this action finishes; EVERY line in the range is marked done on success. Never include an untaken branch.' };
    const writeProperties: Record<string, unknown> = {
      path: { type: 'string', description: '`return`, a part of it, or let/<name>' },
      type: { type: 'string', description: 'the type of what is written, e.g. Bool[], Text[], Num, Text, { name: Text, count: Num }, or a type name of this task' },
      value: writeValue,
      source: { type: 'string', description: 'instead of `value`: the path of an existing value to copy' },
    };
    if (unmarked.length) writeProperties.done = done;
    const tools = [
      tool('read', 'Inspect a value only when you need its contents to make a decision. Workspace paths can be passed directly to `call` without reading them first; do not walk through collection items merely to pass the collection to a function. Optional line or item range for long values. `codebase/<function>` shows the text of a function; host-backed Dict inputs list and read their entries through their normal `args/...` paths; `args@effects` shows the full effect journal.', {
        path: lazyPaths.length ? { anyOf: [{ type: 'string', description: 'what to read', enum: [...new Set([
          ...(lam.journal.length ? ['args@effects'] : []), ...readable,
          ...Object.keys(lam.codebase).map(name => `codebase/${name}`)])] },
          ...lazyPaths.map(path => ({ type: 'string', pattern: `^${path.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}(?:/.+)?$` }))] } : { type: 'string', description: 'what to read', enum: [...new Set([
          ...(lam.journal.length ? ['args@effects'] : []), ...readable,
          ...Object.keys(lam.codebase).map(name => `codebase/${name}`)])] },
        start: { type: 'integer' }, end: { type: 'integer' } }, ['path']),
      tool('write', 'Write a value into the workspace: into `return`, or into a local `let/<name>` (a new name creates the local; `type` says what it holds). Supply `value` or `source`; a type alone is not a value. The value must be complete; to reuse a value that already exists, give `source` (its path) instead of `value`. Source copying preserves the value and type; it does not wrap or convert. Use the destination requested by the program; do not append a field name to make incompatible types fit. To change how a function works, copy it first: type `Function<name>` with path `let/<copy>`, then `edit` `let/<copy>/instructions`, then `call` it as `let/<copy>`.', writeProperties, ['path', 'type']),
      tool('edit', 'Replace text: `old` must occur exactly once in the text at `path`. Use it to delete finished steps from `instructions` (new = ""), to substitute a result into them, or to adapt a copied function.', {
        path: { type: 'string', description: 'a text', ...(textSlots.length ? { enum: textSlots } : {}) },
        old: { type: 'string' }, new: { type: 'string' } }, ['path', 'old', 'new']),
      tool('run_code', 'Run code for exact work (counting, arithmetic, sorting, string operations). Your inputs are in `args`, your locals in `locals`. The value of the last expression comes back to you. Select an available engine.', {
        code: { type: 'string' }, engine: { enum: ['typescript-host'] } }, ['code', 'engine']),
    ];
    (tools[0]!.function.parameters as Record<string, unknown>)['x-natlang-alternatives'] = readAlternatives;
    (tools[1]!.function.parameters as Record<string, unknown>)['x-natlang-alternatives'] = writeAlternatives;
    (tools[1]!.function.parameters as Record<string, unknown>).anyOf = [{ required: ['value'] }, { required: ['source'] },
      ...(Object.keys(lam.codebase).length ? [{ properties: { type: {
        enum: Object.keys(lam.codebase).map(name => `Function<${name}>`) } }, required: ['type'] }] : [])];
    const literalShapes = new Map<string, Record<string, unknown>[]>();
    for (const name of names) {
      const definition = (name.startsWith('let/') ? lam.fnCopies[name.slice(4)] : lam.codebase[name]) as
        Record<string, unknown> | undefined;
      for (const [rawName, type] of Object.entries(definition?.args as Record<string, string> ?? {})) {
        const parameter = rawName.replace(/\?$/, ''), shape = schemaOf(parseType(type), session.env);
        const shapes = literalShapes.get(parameter) ?? [];
        if (!shapes.some(existing => JSON.stringify(existing) === JSON.stringify(shape))) shapes.push(shape);
        literalShapes.set(parameter, shapes);
      }
    }
    const literalProperties = Object.fromEntries([...literalShapes].map(([name, shapes]) =>
      [name, shapes.length === 1 ? shapes[0] : { anyOf: shapes }]));
    const callProperties: Record<string, unknown> = {
      function: { enum: names }, to: { type: 'string' },
      inputs: { type: 'object', properties: Object.fromEntries(inputNames.map(name => [name, { type: 'string' }])), additionalProperties: false },
      values: { type: 'object', properties: literalProperties, additionalProperties: false },
      over: { type: 'string' },
      init: { anyOf: ['string', 'number', 'boolean', 'null', 'object', 'array'].map(type => ({ type })) },
      until: { type: 'string' }, max: { type: 'integer' },
    };
    if (unmarked.length) callProperties.done = done;
    if (names.length) {
      const destinations = { anyOf: [{ enum: definitionPaths }, newLocal] };
      const callAlternatives: Record<string, unknown>[] = [];
      const checkNames = Object.entries(lam.codebase).flatMap(([name, raw]) => {
        const definition = raw as Record<string, unknown>;
        const params = Object.keys(definition.args as Record<string, string> ?? {}).filter(param => !param.endsWith('?'));
        return definition.returns === 'Bool' && params.length === 1 ? [name] : [];
      });
      for (const name of names) {
        const definition = (name.startsWith('let/') ? lam.fnCopies[name.slice(4)] : lam.codebase[name]) as
          Record<string, unknown> | undefined;
        const params = definition?.args as Record<string, string> ?? {};
        const namesAndTypes = Object.entries(params).map(([raw, type]) => [raw.replace(/\?$/, ''), type, raw.endsWith('?')] as const);
        const properties = Object.fromEntries(namesAndTypes.map(([param, type]) =>
          [param, { type: 'string', description: `workspace path to ${type}` }]));
        const valueProperties = Object.fromEntries(namesAndTypes.map(([param, type]) =>
          [param, schemaOf(parseType(type), session.env)]));
        const required = namesAndTypes.filter(([param, , optional]) => !optional && param in properties).map(([param]) => param);
        const inputs = { type: 'object', properties, required, additionalProperties: false };
        const base = { function: { const: name }, to: destinations };
        callAlternatives.push(namesAndTypes.length ? { ...base, inputs } : base);
        if (namesAndTypes.length) callAlternatives.push({ ...base, inputs: { ...inputs, required: [] },
          values: { type: 'object', properties: valueProperties, required: [], additionalProperties: false },
          'x-optional': ['inputs', 'values'] });
        const lists = present.filter(slot => Array.isArray(slot.value)).map(slot => slot.path);
        if (lists.length && namesAndTypes.length) {
          callAlternatives.push({ ...base, over: { enum: lists }, inputs: { ...inputs, required: [] },
            'x-optional': ['inputs'] });
          if ('acc' in properties && 'item' in properties) {
            const rest = Object.fromEntries(Object.entries(properties).filter(([param]) => param !== 'acc' && param !== 'item'));
            callAlternatives.push({ ...base, over: { enum: lists }, init: {},
              ...(Object.keys(rest).length ? { inputs: { ...inputs, properties: rest, required: [] },
                'x-optional': ['inputs'] } : {}) });
          }
        }
        if (checkNames.length && namesAndTypes.length) {
          const starts = [...new Set(namesAndTypes.flatMap(([, type]) => {
            try { return fitting(parseType(type)); } catch { return []; }
          }))];
          callAlternatives.push({ ...base, init: { enum: starts.length ? starts : present.map(slot => slot.path).slice(0, 48) },
            until: { enum: checkNames }, max: { type: 'integer' }, inputs: { ...inputs, required: [] },
            'x-optional': ['inputs'] });
        }
      }
      for (const name of names) for (const slot of present.filter(slot => isPending(slot.value) &&
        ['unreduced', 'quiesced'].includes(slot.value.status)).slice(0, 4))
        callAlternatives.push({ function: { const: name }, to: { enum: [slot.path] } });
      const call = tool('call', 'Call one of your functions and put its result at `to` (`return`, a part of it, or a local `let/<name>`). `inputs` maps parameters to workspace paths; `values` supplies typed literal parameters. Do not bind one parameter both ways. With `over`: call it once for every item of that list (the item goes to the one parameter you left out). Do not put the mapped item in inputs or values. The result is the list of results. With `over` and `init`, omit both item and acc from inputs and values: carry `acc` through the list. With `init`, `until`, `max`: repeat from the value at `init` until the function `until` says true, at most `max` times. Calling again with only `function` and `to` retries what did not finish.', callProperties, ['function', 'to']);
      (call.function.parameters as Record<string, unknown>)['x-natlang-alternatives'] = callAlternatives;
      tools.push(call);
    }
    if (unmarked.length) for (const name of ['write', 'call']) {
      const entry = tools.find(item => item.function.name === name);
      const alternatives = (entry?.function.parameters as Record<string, unknown> | undefined)?.['x-natlang-alternatives'];
      if (!Array.isArray(alternatives)) continue;
      for (const alternative of alternatives) {
        alternative.done = doneLine;
        alternative['x-optional'] = [...(alternative['x-optional'] ?? []), 'done'];
      }
    }
    if (unmarked.length) {
      const markable = [...new Set([...unmarked, ...(lam.originalBody ?? lam.body).replace(/^\n+|\n+$/g, '').split('\n')
        .flatMap((line, index) => line.trim().startsWith('function ') ? [index + 1] : [])])].sort((a, b) => a - b);
      const mark = tool('mark_done', 'Mark lines of your program as finished. `start` alone for one line, `start` and `end` for an inclusive range (EVERY line between the endpoints). Add skipped=true when the lines did not apply, such as the branch of an `if` that was not taken. Mark a line only after everything it asks for is finished.', {
        start: { type: 'integer' }, end: { type: 'integer' },
        skipped: { type: 'boolean' } }, ['start']);
      (mark.function.parameters as Record<string, unknown>)['x-natlang-alternatives'] = [
        { start: { enum: markable }, skipped: { const: true }, 'x-optional': ['skipped'] },
        { start: { enum: markable }, end: { enum: markable }, skipped: { const: true }, 'x-optional': ['skipped'] },
      ];
      tools.push(mark);
    }
    tools.push(tool('report_blocker', 'The task cannot be done as asked: the inputs do not determine the result, or a rule does not cover the case. Say exactly what is missing. This ends the task without a result; do not guess instead.', { missing: { type: 'string' } }, ['missing']));
    tools.push(tool('report_error', 'The executed instructions cannot be satisfied: a contradiction, invalid operation, or incompatible required result prevents correct completion. Explain the error. This ends the task without a result. Do not change the requirements to succeed. Use report_blocker for missing information instead.', { message: { type: 'string' } }, ['message']));
    return tools;
  }

  /** Model-facing tools-v4: one operation per tool and positional, path-only calls. */
  private toolsV4(session: NativeSession): any[] {
    const legacy = this.toolsV3(session);
    const byName = Object.fromEntries(legacy.map(entry => [entry.function.name, entry]));
    const lam = session.lam, all: Slot[] = [];
    if (lam.type.kind === 'lambda') {
      all.push(...slots('args', lam.type.params, lam.args as Value, session.env, false));
      all.push(...slots('return', lam.type.returns, lam.return, session.env, true));
    }
    for (const [name, type] of Object.entries(lam.letTypes))
      all.push(...slots(`let/${name}`, type, lam.let[name] ?? MISSING, session.env, true));
    const definitions: Record<string, Record<string, unknown>> = {};
    for (const [name, definition] of Object.entries(lam.codebase)) definitions[name] = definition as Record<string, unknown>;
    for (const [name, definition] of Object.entries(lam.fnCopies)) definitions[`let/${name}`] = definition as Record<string, unknown>;
    const envFor = (definition: Record<string, unknown>) => session.env.child(Object.fromEntries(
      Object.entries(definition.types as Record<string, string> ?? {}).filter(([name]) => !session.env.lookup(name))
        .map(([name, value]) => [name, parseType(value)])));
    const references = (typeText: string, definition: Record<string, unknown>) => {
      const target = parseType(typeText), env = envFor(definition), paths: string[] = [];
      for (const slot of all) if (slot.value !== MISSING && !isPending(slot.value)) {
        try { if (fitsType(slot.type, target, env)) paths.push(slot.path); } catch { /* incompatible alias */ }
      }
      const existing = [...new Set(paths)].slice(0, 48);
      return existing.length ? { anyOf: [{ enum: existing }, futureLocal] } : structuredClone(futureLocal);
    };
    const destinations = () => {
      const existing = [...new Set(all.filter(slot => slot.writable && !slot.path.startsWith('args')).map(slot => slot.path))].slice(0, 48);
      return { anyOf: [...(existing.length ? [{ enum: existing }] : []), structuredClone(explicitNewLocal)] };
    };
    const positionalInputs = (definition: Record<string, unknown>, skip: number) => {
      const ordered = Object.entries(definition.args as Record<string, string> ?? {}).slice(skip);
      let optional = false;
      for (const [raw] of ordered) { if (raw.endsWith('?')) optional = true; else if (optional) return null; }
      return { type: 'array', prefixItems: ordered.map(([, type]) => references(type, definition)), items: {},
        minItems: ordered.filter(([raw]) => !raw.endsWith('?')).length, maxItems: ordered.length };
    };
    const result = [structuredClone(byName.read)];
    const code = byName.run_code.function;
    result.push(tool('run_code', code.description, { engine: structuredClone(code.parameters.properties.engine),
      code: structuredClone(code.parameters.properties.code) }, ['engine', 'code']));

    const valueAlternatives: Record<string, unknown>[] = [], copyAlternatives: Record<string, unknown>[] = [];
    const functionAlternatives: Record<string, unknown>[] = [];
    for (const alternative of byName.write.function.parameters['x-natlang-alternatives'] ?? []) {
      if (alternative.value !== undefined && alternative.path?.['x-natlang'] !== 'new-local')
        valueAlternatives.push({ destination: alternative.path, type: alternative.type, value: alternative.value });
      else if (alternative.source !== undefined)
        copyAlternatives.push({ source: alternative.source, destination: alternative.path });
      else if (String(alternative.type?.const ?? '').startsWith('Function<'))
        functionAlternatives.push({ function: { const: String(alternative.type.const).slice(9, -1) }, save_as: alternative.path });
    }
    const localTypes: [string, Record<string, unknown>][] = ['Num', 'Text', 'Bool', 'Null', 'Num[]', 'Text[]',
      'Bool[]', 'Dict<Num>', 'Dict<Text>', 'Dict<Bool>'].map(name => [name, schemaOf(parseType(name), session.env)]);
    for (const slot of all) { const shape = schemaOf(slot.type, session.env); if (Object.keys(shape).length) localTypes.push([formatType(slot.type), shape]); }
    for (const definition of Object.values(definitions)) {
      const env = envFor(definition);
      for (const typeText of [...Object.values(definition.args as Record<string, string> ?? {}), String(definition.returns)]) {
        const shape = schemaOf(parseType(typeText), env); if (Object.keys(shape).length) localTypes.push([typeText, shape]);
      }
    }
    const seen = new Set<string>();
    for (const [typeText, shape] of localTypes) {
      const key = `${typeText}\0${JSON.stringify(shape)}`; if (seen.has(key)) continue; seen.add(key);
      valueAlternatives.push({ destination: structuredClone(explicitNewLocal), type: { const: typeText }, value: shape });
      if (seen.size >= 32) break;
    }
    for (const slot of all) if (slot.value !== MISSING && !isPending(slot.value))
      copyAlternatives.push({ source: { const: slot.path }, destination: structuredClone(explicitNewLocal) });
    const explicitTool = (name: string, description: string, properties: Record<string, unknown>, required: string[], alternatives: unknown[]) => {
      const entry = tool(name, description, properties, required);
      if (alternatives.length) (entry.function.parameters as Record<string, unknown>)['x-natlang-alternatives'] = alternatives;
      return entry;
    };
    result.push(explicitTool('write_value', 'Write one literal value. Choose its destination and type before generating the value.',
      { destination: { type: 'string' }, type: { type: 'string' }, value: {} },
      ['destination', 'type', 'value'], valueAlternatives));
    if (copyAlternatives.length) result.push(explicitTool('copy_value', 'Copy a value between workspace paths.',
      { source: { type: 'string' }, destination: { type: 'string' } }, ['source', 'destination'], copyAlternatives));
    if (functionAlternatives.length) result.push(explicitTool('copy_function', 'Make an editable local copy of a named function.',
      { function: { enum: Object.keys(lam.codebase) }, save_as: structuredClone(explicitNewLocal) }, ['function', 'save_as'], functionAlternatives));

    const offered = byName.edit?.function.parameters.properties.path.enum ?? [];
    if (byName.edit) {
      const alternatives: Record<string, unknown>[] = [];
      for (const slot of all) if (offered.includes(slot.path) && typeof slot.value === 'string') {
        const spans = textSpans(slot.value);
        if (spans.length) alternatives.push({ path: { const: slot.path }, find: { enum: spans }, replace_with: { type: 'string' } });
        alternatives.push({ path: { const: slot.path }, find: { type: 'string' }, fuzzy: { const: true }, replace_with: { type: 'string' } });
      }
      result.push(explicitTool('edit_text', 'Replace existing text. Use fuzzy only for one unambiguous inexact selection.',
        { path: { enum: offered }, find: { type: 'string' }, fuzzy: { type: 'boolean' }, replace_with: { type: 'string' } },
        ['path', 'find', 'replace_with'], alternatives));
    }

    const calls: Record<string, unknown>[] = [], maps: Record<string, unknown>[] = [];
    const folds: Record<string, unknown>[] = [], repeats: Record<string, unknown>[] = [];
    const checks = Object.entries(definitions).filter(([, definition]) => definition.returns === 'Bool' &&
      Object.keys(definition.args as object ?? {}).length > 0);
    for (const [name, definition] of Object.entries(definitions)) {
      const args = Object.entries(definition.args as Record<string, string> ?? {}), metadata = {
        function: { const: name }, 'x-natlang-parameters': args.map(([raw]) => raw.replace(/\?$/, '')),
        'x-natlang-types': args.map(([, type]) => type) };
      const direct = positionalInputs(definition, 0);
      if (direct) calls.push({ ...metadata, ...(direct.maxItems ? { inputs: direct } : {}),
        ...(direct.maxItems && direct.minItems === 0 ? { 'x-optional': ['inputs'] } : {}), save_as: destinations() });
      if (args.length) {
        const extra = positionalInputs(definition, 1);
        if (extra) maps.push({ ...metadata, items: references(`(${args[0]![1]})[]`, definition),
          ...(extra.maxItems ? { inputs: extra } : {}),
          ...(extra.maxItems && extra.minItems === 0 ? { 'x-optional': ['inputs'] } : {}), save_as: destinations() });
      }
      if (args.length >= 2) {
        const env = envFor(definition), extra = positionalInputs(definition, 2);
        let valid = false; try { valid = fitsType(parseType(String(definition.returns)), parseType(args[0]![1]), env); } catch { /* no fold */ }
        if (valid && extra) folds.push({ ...metadata, items: references(`(${args[1]![1]})[]`, definition),
          initial: references(args[0]![1], definition), ...(extra.maxItems ? { inputs: extra } : {}),
          ...(extra.maxItems && extra.minItems === 0 ? { 'x-optional': ['inputs'] } : {}), save_as: destinations() });
      }
      if (args.length) {
        const env = envFor(definition), extra = positionalInputs(definition, 1);
        let state = false; try { state = fitsType(parseType(String(definition.returns)), parseType(args[0]![1]), env); } catch { /* no repeat */ }
        const until = checks.flatMap(([check, checkDefinition]) => {
          const first = Object.values(checkDefinition.args as Record<string, string> ?? {})[0];
          try { return first && Object.keys(checkDefinition.args as object).filter(raw => !raw.endsWith('?')).length === 1 &&
            fitsType(parseType(args[0]![1]), parseType(first), envFor(checkDefinition)) ? [check] : []; } catch { return []; }
        });
        if (state && extra && until.length) repeats.push({ ...metadata, initial: references(args[0]![1], definition),
          ...(extra.maxItems ? { inputs: extra } : {}), ...(extra.maxItems && extra.minItems === 0 ? { 'x-optional': ['inputs'] } : {}),
          until: { enum: until }, at_most: { type: 'integer' }, save_as: destinations() });
      }
    }
    const addMode = (name: string, description: string, alternatives: Record<string, unknown>[], fields: string[]) => {
      if (!alternatives.length) return;
      const properties = Object.fromEntries(fields.map(field => [field, mergedSchemas(alternatives.map(alt => alt[field]),
        field === 'inputs' ? { type: 'array', items: { type: 'string' } } : {})]));
      result.push(explicitTool(name, description, properties, fields.filter(field => field !== 'inputs'), alternatives));
    };
    addMode('run_function', 'Run a function once. `inputs` lists workspace paths in parameter order.', calls,
      ['function', 'inputs', 'save_as']);
    addMode('for_each', 'Run a function for every item in `items`. The item fills parameter 1; `inputs` fills the rest.', maps,
      ['function', 'items', 'inputs', 'save_as']);
    addMode('fold', 'Fold `items`. Accumulator fills parameter 1, item parameter 2, and `inputs` fills the rest.', folds,
      ['function', 'items', 'initial', 'inputs', 'save_as']);
    addMode('repeat', 'Repeat a state transition. State fills parameter 1 and `inputs` fills the rest.', repeats,
      ['function', 'initial', 'inputs', 'until', 'at_most', 'save_as']);
    const pending = all.filter(slot => isPending(slot.value) && ['unreduced', 'quiesced'].includes(slot.value.status)).map(slot => slot.path);
    if (pending.length) result.push(explicitTool('resume', 'Continue a pending computation with its retained inputs and progress.',
      { computation: { enum: pending } }, ['computation'], pending.map(path => ({ computation: { const: path } }))));
    if (byName.mark_done) { const mark = structuredClone(byName.mark_done); mark.function.name = 'mark_lines'; result.push(mark); }
    result.push(structuredClone(byName.report_blocker), structuredClone(byName.report_error));
    return result;
  }

  private toolsScope(session: NativeSession): any[] {
    const tools = [
      tool('eval', 'Execute TypeScript in the persistent typed scope and the same host as crisp code. Declarations persist; imported functions are called with await and positional values.',
        { code: { type: 'string' } }, ['code']),
      tool('read_value', 'Inspect a variable or field/index selection without executing code.',
        { expression: { type: 'string' },
          start: { type: 'integer', minimum: 0, description: 'Zero-based character offset for Text or item index for a list.' },
          end: { type: 'integer', minimum: 0, description: 'Exclusive character or item offset, as in JavaScript slice().' } }, ['expression']),
      tool('write_value', 'Transport an already supplied literal into a top-level scope variable. For normal program work, including literal decisions, prefer eval declarations. as_type is needed only when inference is ambiguous.',
        { name: { type: 'string', pattern: '^[A-Za-z_$][A-Za-z0-9_$]*$' }, value: {}, as_type: { type: 'string' } },
        ['name', 'value']),
      tool('return_value', 'Stage one existing variable as this function\'s typed result. End the turn naturally after all instruction lines are closed.',
        { variable: { type: 'string', pattern: '^[A-Za-z_$][A-Za-z0-9_$]*$' } }, ['variable']),
      tool('mark_lines', 'Close one instruction line or inclusive contiguous range after its work succeeded. Use skipped only for an untaken branch.',
        { start: { type: 'integer' }, end: { type: 'integer' }, skipped: { type: 'boolean' } }, ['start']),
      tool('report_blocker', 'End without a result because required information is missing. Do not guess.',
        { missing: { type: 'string' } }, ['missing']),
      tool('report_error', 'End without a result because the instructions require an invalid or contradictory operation.',
        { message: { type: 'string' } }, ['message']),
    ];
    if (Object.keys(session.lam.codebase).length || session.lam.projectTransaction) {
      const roots = `codebase/${session.lam.projectTransaction ? ' or project/' : ''}`;
      const fileTools = [
      tool('list_files', `List files beneath ${roots}. Codebase file identity is fixed during a run.`,
        { path: { type: 'string' }, pattern: { type: 'string' } }, []),
      tool('search_files', `Search text files beneath ${roots} and return matching file, line, and context.`,
        { query: { type: 'string' }, path: { type: 'string' }, pattern: { type: 'string' }, regex: { type: 'boolean' } }, ['query']),
      tool('read_file', `Read a file beneath ${roots}, optionally by one-based inclusive line range.`,
        { path: { type: 'string' }, start_line: { type: 'integer' }, end_line: { type: 'integer' } }, ['path']),
      tool('write_file', 'Replace a text file. project/ may create files; codebase/ is limited to existing files.',
        { path: { type: 'string' }, content: { type: 'string' } }, ['path', 'content']),
      tool('edit_file', `Replace one exact or uniquely fuzzy span in a file beneath ${roots}.`,
        { path: { type: 'string' }, find: { type: 'string' }, replace_with: { type: 'string' }, fuzzy: { type: 'boolean' } },
        ['path', 'find', 'replace_with']),
      tool('diff_files', `Inspect the current overlay delta beneath ${roots}.`,
        { path: { type: 'string' } }, [])];
      if (session.lam.projectTransaction) fileTools.unshift(
        tool('commit', 'Stage an existing typed variable and select project changes. Patterns are relative to project/.',
        { value: { type: 'string', pattern: '^[A-Za-z_$][A-Za-z0-9_$]*$' },
          include: { type: 'array', items: { type: 'string' } },
          exclude: { type: 'array', items: { type: 'string' } } }, ['value']));
      tools.splice(4, 0, ...fileTools);
    }
    return tools;
  }

  tools(session: NativeSession): unknown[] {
    session.surfaceName = this.options.toolSchema ?? 'tools-v4';
    if (this.options.toolSchema === 'scope-eval-v1' &&
        !this.missing(session) && !this.unmarkedLines(session).length) return [];
    return this.options.toolSchema === 'tools-v3' ? this.toolsV3(session) :
      this.options.toolSchema === 'scope-eval-v1' ? this.toolsScope(session) : this.toolsV4(session);
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
      (Object.hasOwn(lam.args, field.name) ? previewValue(lam.args[field.name]!) : 'missing'));
    const imports = Object.entries(lam.codebase).map(([name, raw]) => {
      const fn = raw as Record<string, unknown>;
      return `  ${name}(${Object.entries(fn.args as Record<string, string> ?? {})
        .map(([key, value]) => `${key.replace(/\?$/, '')}: ${value}`).join(', ')}): Promise<${fn.returns}>`;
    });
    const locals = Object.entries(lam.let).map(([name, value]) =>
      `  ${name}: ${formatType(lam.letTypes[name]!)} = ${previewValue(value)}`);
    return ['Execute the natural-language function line by line.', '', 'Program:', program, '', 'Scope:',
      ' parameters (immutable lexical bindings; there is no inputs or args object)',
      ...(inputs.length ? inputs : ['  (none)']),
      ' imports (immutable live bindings)', ...(imports.length ? imports : ['  (none)']),
      ' locals', ...(locals.length ? locals : ['  (none)']),
      ` result: ${formatType(lam.type.returns)} — ${lam.return === MISSING ? 'not staged' : 'staged'}`].join('\n');
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
    const openingMessages = (): Record<string, unknown>[] => {
    const lam = session.lam;
    const output = lam.type.kind === 'lambda' ? formatType(lam.type.returns) : 'unknown';
    const signatures = Object.entries(lam.codebase).map(([name, raw]) => {
      const fn = raw as Record<string, unknown>;
      const args = Object.entries(fn.args as Record<string, string> ?? {}).map(([key, type]) => `${key}: ${type}`).join(', ');
      return { signature: `${name}(${args}) -> ${fn.returns}`, description: String(fn.description ?? '') };
    });
    const width = Math.max(0, ...signatures.map(item => item.signature.length));
    const functions = signatures.map(item => `  ${item.signature.padEnd(width)}   ${item.description}`.trimEnd());
    const original = lam.originalBody ?? lam.body;
    const numbered = original.replace(/^\n+|\n+$/g, '').split('\n');
    const program = functions.length ? numbered.map((line, index) => {
      const text = line.trim(), markable = text && !text.startsWith('#') && !text.startsWith('function ');
      return `${String(index + 1).padStart(String(numbered.length).length)} ${markable ?
        lam.marks[index + 1] === 'done' ? '[x]' : lam.marks[index + 1] === 'skipped' ? '[-]' : '[ ]' : '   '} ${line}`.trimEnd();
    }).join('\n') + '\n\nThe lines are numbered. [ ] is still to do, [x] is done, [-] did not apply. Mark lines done as you finish them.' : lam.body.trim();
    if (this.options.toolSchema === 'scope-eval-v1') return [
      { role: 'system', content: this.options.systemPrompt ?? EXPLICIT_TOOLS_PROMPT },
      { role: 'user', content: this.scopeOpening(session) },
    ];
    const messages: Record<string, unknown>[] = [
      { role: 'system', content: (this.options.systemPrompt ??
        (this.options.toolSchema === 'tools-v3' ? TOOLS_PROMPT : EXPLICIT_TOOLS_PROMPT)) +
        '\nFor run_code, always name an engine offered in its current tool schema.' },
      { role: 'user', content: `${program}\n\nWrite the result to \`return\` (${output}).` +
        (functions.length ? `\n\nFunctions you can call:\n${functions.join('\n')}` : '') },
    ];
    const opening = this.opening(session);
    if (lam.type.kind === 'lambda' && (lam.type.params.fields.length || lam.continuationNote ||
        lam.journal.length || Object.keys(lam.let).length || lam.return !== MISSING))
      messages.push({ role: 'assistant', content: '', tool_calls: [{ id: 'call_0', type: 'function',
      function: { name: 'read', arguments: '{"path":"args"}' } }] },
      { role: 'tool', tool_call_id: 'call_0', content: opening });
    return messages;
    };
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
          (['call', 'mark_done', 'edit'].includes(name) || 'done' in args || 'source' in args);
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
      checkpointReady = !results.some(result => ['rejected', 'refused', 'error'].includes(result.kind)) &&
        ['write', 'call', 'edit', 'mark_done'].includes(calls[results.length - 1]?.[0] ?? '');
      if (results.at(-1)?.kind === 'budget') return 'action or tool-call budget exhausted';
      const failed = results.find(result => ['rejected', 'refused'].includes(result.kind));
      if (this.options.validationFeedback !== 'local' && failed)
        return `validation failed: ${failed.text}`;
      if (results.at(-1)?.kind === 'completed') return;
    }
  }
}
