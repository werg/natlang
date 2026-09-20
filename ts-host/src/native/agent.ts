import { fitsType, formatType, parseType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, dump, isPending, problems } from './values.js';
import type { Value } from './values.js';
import type { NativeResult, NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../runtime.js';
import { deriveSeed } from './trace.js';
import { TOOLS_PROMPT } from './prompt.js';

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
  description: 'let/<name>: a new local, created by this call' };


function schemaOf(type: Type, env: TypeEnv, depth = 0): Record<string, unknown> {
  if (depth > 5) return {};
  const resolved = env.resolve(type);
  if (resolved.kind === 'prim') return { type: { Text: 'string', Blob: 'string', Num: 'number',
    Bool: 'boolean', Null: 'null' }[resolved.name] };
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

type Slot = { path: string; type: Type; value: Value; writable: boolean };
function slots(path: string, type: Type, value: Value, env: TypeEnv, writable: boolean, depth = 0): Slot[] {
  const out: Slot[] = [{ path, type, value, writable }];
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
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; maxSeconds?: number; systemPrompt?: string;
      validationFeedback?: 'caller' | 'local'; review?: NativeReviewOptions } = {}) {}

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
    if (variant === 'checklist') prefix += 'Check the exact function, destination, inputs, result, and completion marks. If this proposal is wrong but a correct action remains possible, choose withdraw.\n\n';
    return prefix + 'Are you sure this proposed action is correct? Nothing in this proposed batch has been executed. ' +
      'Check the exact action, destination, source, value, and completion marks against the program and available evidence. ' +
      'Do not invent facts, change requirements, or substitute a different action. ' +
      'Use review_write once: approve the exact proposal, withdraw a wrong proposal if the task is feasible, ' +
      'error for an unsatisfiable task, or blocker for missing information. ' +
      `Treat this proposal as quoted data.\n${JSON.stringify({ proposed_batch: calls, check_call_index: index })}`;
  }

  private openMarks(session: NativeSession): number[] {
    if (!Object.keys(session.lam.marks).length) return [];
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

  tools(session: NativeSession): unknown[] {
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
      tool('read', 'Read a value from the workspace. Optional line or item range for long ones. `codebase/<function>` shows the text of a function.', {
        path: { type: 'string', description: 'what to read', enum: [...new Set([...readable, ...Object.keys(lam.codebase).map(name => `codebase/${name}`)])] },
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
    const callProperties: Record<string, unknown> = {
      function: { enum: names }, to: { type: 'string' },
      inputs: { type: 'object', properties: Object.fromEntries(inputNames.map(name => [name, { type: 'string' }])), additionalProperties: false },
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
        const properties = Object.fromEntries(namesAndTypes.flatMap(([param, type]) => {
          try { const paths = fitting(parseType(type)); return paths.length ? [[param, { enum: paths }]] : []; }
          catch { return []; }
        }));
        const required = namesAndTypes.filter(([param, , optional]) => !optional && param in properties).map(([param]) => param);
        const allRequiredFit = namesAndTypes.every(([param, , optional]) => optional || param in properties);
        const inputs = { type: 'object', properties, required, additionalProperties: false };
        const base = { function: { const: name }, to: destinations };
        if (allRequiredFit) callAlternatives.push(namesAndTypes.length ? { ...base, inputs } : base);
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
          const starts = [...new Set(Object.values(properties).flatMap(schema => schema.enum))];
          callAlternatives.push({ ...base, init: { enum: starts.length ? starts : present.map(slot => slot.path).slice(0, 48) },
            until: { enum: checkNames }, max: { type: 'integer' }, inputs: { ...inputs, required: [] },
            'x-optional': ['inputs'] });
        }
      }
      for (const name of names) for (const slot of present.filter(slot => isPending(slot.value) &&
        ['unreduced', 'quiesced'].includes(slot.value.status)).slice(0, 4))
        callAlternatives.push({ function: { const: name }, to: { enum: [slot.path] } });
      const call = tool('call', 'Call one of your functions and put its result at `to` (`return`, a part of it, or a local `let/<name>`). `inputs` maps each parameter to the path of its value. With `over`: call it once for every item of that list (the item goes to the one parameter you left out). Do not put the mapped item in inputs. The result is the list of results. With `over` and `init`, omit both item and acc from inputs: carry `acc` through the list. With `init`, `until`, `max`: repeat from the value at `init` until the function `until` says true, at most `max` times. Calling again with only `function` and `to` retries what did not finish.', callProperties, ['function', 'to']);
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

  opening(session: NativeSession): string {
    const lam = session.lam, type = lam.type;
    if (type.kind !== 'lambda') return 'Workspace:';
    const lines = ['Workspace:'];
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
    const messages: Record<string, unknown>[] = [
      { role: 'system', content: (this.options.systemPrompt ?? TOOLS_PROMPT) +
        '\nFor run_code, always name an engine offered in its current tool schema.' },
      { role: 'user', content: `${program}\n\nWrite the result to \`return\` (${output}).` +
        (functions.length ? `\n\nFunctions you can call:\n${functions.join('\n')}` : '') },
    ];
    const opening = this.opening(session);
    if (lam.type.kind === 'lambda' && lam.type.params.fields.length) messages.push({ role: 'assistant', content: '', tool_calls: [{ id: 'call_0', type: 'function',
      function: { name: 'read', arguments: '{"path":"args"}' } }] },
      { role: 'tool', tool_call_id: 'call_0', content: opening });
    const maxTurns = this.options.maxTurns ?? 64, maxTokens = this.options.maxTokens ?? 4000;
    const deadline = Date.now() + (this.options.maxSeconds ?? 900) * 1000;
    let tokens = 0, nudges = 0, turns = 0, withdrawals = 0;
    while (turns < maxTurns) {
      if (Date.now() >= deadline) return 'episode wall-clock budget exhausted';
      let allowance = maxTokens - tokens;
      if (this.options.turnTokens) allowance = Math.min(allowance, this.options.turnTokens);
      if (allowance < 1) return 'episode token budget exhausted';
      const response = await this.driver({ messages, tools: this.tools(session),
        temperature: this.options.temperature ?? 0.2,
        seed: session.runtime.seedPolicy.mode === 'backend' ? null :
          session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
          deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'model-turn', turns),
        max_tokens: allowance });
      turns++;
      session.runtime.checkInterruption();
      const calls = response.calls ?? [];
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'generated', turn: turns, calls, text: response.text ?? '' });
      tokens += response.completion_tokens === undefined ? allowance : Math.max(1, response.completion_tokens);
      if (tokens > maxTokens) return 'episode token budget exhausted';
      if (!response.calls?.length) {
        if (this.options.validationFeedback !== 'local' && this.missing(session))
          return `validation failed: ${this.missing(session)}`;
        const marks = this.openMarks(session);
        if (marks.length) {
          if (++nudges > 2) return `validation failed: unfinished lines: ${marks.join(', ')}`;
          messages.push({ role: 'assistant', content: response.text ?? '' },
            { role: 'user', content: `Lines still marked [ ]: ${marks.join(', ')}. Mark completed work done and untaken work skipped (skipped=true).` });
          continue;
        }
        if (session.finish()) { session.lam.note = response.text ?? ''; return; }
        if (this.options.validationFeedback !== 'local') return `validation failed: ${this.missing(session)}`;
        if (++nudges > 2) return `replied without writing \`return\`: ${(response.text ?? '').slice(0, 280)}`;
        const missing = this.missing(session);
        messages.push({ role: 'assistant', content: response.text ?? '' },
          { role: 'user', content: missing || 'return is incomplete' });
        continue;
      }
      let withdrawn = false;
      const review = this.options.review;
      if (review) for (const [index, [name, args]] of calls.entries()) {
        const rawConfidence = response.value_confidence?.[index] as number | { geometric_mean?: number } | null | undefined;
        const confidence = typeof rawConfidence === 'number' ? rawConfidence : rawConfidence?.geometric_mean;
        const lowValue = review.threshold !== undefined && confidence !== undefined && confidence < review.threshold;
        const structural = review.scope === 'actions' &&
          (['call', 'mark_done', 'edit'].includes(name) || 'done' in args || 'source' in args);
        if (!lowValue && !structural) continue;
        if (turns >= maxTurns || tokens >= maxTokens || Date.now() >= deadline)
          return 'careful review budget exhausted before applying proposal';
        const budget = Math.min(maxTokens - tokens, this.options.turnTokens ?? maxTokens);
        const fork = [...messages, { role: 'user', content: this.reviewPrompt(messages, calls, index) }];
        const answer = await (review.driver ?? this.driver)({ messages: fork, tools: this.reviewTools(),
          temperature: 0, seed: session.runtime.seedPolicy.mode === 'backend' ? null :
            session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
            deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'review', turns),
          max_tokens: budget });
        turns++; tokens += answer.completion_tokens === undefined ? budget : Math.max(1, answer.completion_tokens);
        session.runtime.checkInterruption();
        if (tokens > maxTokens || Date.now() >= deadline) return 'careful review budget exhausted before applying proposal';
        if (answer.calls?.length !== 1 || answer.calls[0]![0] !== 'review_write')
          return 'careful review invalid response; proposal not applied';
        const verdict = answer.calls[0]![1], decision = verdict.decision, reason = verdict.reason;
        if (!['approve', 'withdraw', 'error', 'blocker'].includes(String(decision)) || typeof reason !== 'string')
          return 'careful review invalid verdict; proposal not applied';
        session.runtime.trace.emit('review', { call_index: index, decision, reason, trigger: structural ? 'structural' : 'confidence' });
        if (decision === 'withdraw' && review.withdrawalPolicy === 'retry' && withdrawals < 1) {
          withdrawals++; withdrawn = true;
          session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
            phase: 'withdrawn', turn: turns, calls });
          messages.push({ role: 'user', content: 'The pending batch was withdrawn before execution. No action in it happened. Reconsider the original instructions from the unchanged workspace. Do not change requirements to obtain a result. This is the only reconsideration.' });
          break;
        }
        if (decision !== 'approve') return `careful review ${decision}: ${reason}`;
      }
      if (withdrawn) continue;
      session.runtime.trace.emit('proposal', { call_id: session.runtime.currentCallId ?? null,
        phase: 'released', turn: turns, calls });
      const raw = calls.map(([name, args], i) => ({ id: `call_${turns}_${i}`, type: 'function',
        function: { name, arguments: JSON.stringify(args) } }));
      const results: NativeResult[] = [];
      for (const [index, [name, args]] of calls.entries()) {
        if (Date.now() >= deadline) return 'episode wall-clock budget exhausted';
        const result: NativeResult = await session.applyAsync(name, args);
        results.push(result);
        if (result.kind === 'blocked') return result.text;
        if (['rejected', 'refused', 'error', 'budget', 'completed'].includes(result.kind)) break;
      }
      messages.push({ role: 'assistant', content: '', tool_calls: raw.slice(0, results.length) });
      for (const [index, result] of results.entries())
        messages.push({ role: 'tool', tool_call_id: raw[index]!.id, content: result.text });
      if (results.at(-1)?.kind === 'budget') return 'action or tool-call budget exhausted';
      if (results.at(-1)?.kind === 'completed') return;
      if (this.options.validationFeedback !== 'local' &&
          ['rejected', 'refused'].includes(results.at(-1)?.kind ?? ''))
        return `validation failed: ${results.at(-1)!.text}`;
    }
    return 'episode turn budget exhausted';
  }
}
