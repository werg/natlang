import { formatType } from './types.js';
import type { Type, TypeEnv } from './types.js';
import { MISSING, dump, isPending, problems } from './values.js';
import type { Value } from './values.js';
import type { NativeResult, NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../runtime.js';
import { deriveSeed } from './trace.js';

export type NativeModelDriver = (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  type: 'function', function: { name, description, parameters: { type: 'object', properties,
    required, additionalProperties: false } },
});

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
  if (depth >= 3 || isPending(value)) return out;
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

/** The native model loop. Program state stays in NativeSession, never in the model history. */
export class NativeToolAgent {
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; maxSeconds?: number; systemPrompt?: string } = {}) {}

  private openMarks(session: NativeSession): number[] {
    if (!Object.keys(session.lam.marks).length) return [];
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
    const readable = all.filter(slot => slot.value !== MISSING).map(slot => slot.path);
    const writable = all.filter(slot => slot.writable && !isPending(slot.value)).slice(0, 48);
    const textSlots = all.filter(slot => slot.writable && typeof slot.value === 'string').map(slot => slot.path);
    const path = { type: 'string' };
    const slotPaths = writable.map(slot => slot.path);
    const valueSchemas = [...new Map(writable.map(slot => {
      const schema = schemaOf(slot.type, session.env); return [JSON.stringify(schema), schema] as const;
    })).values()];
    const writeValue = valueSchemas.length === 1 ? valueSchemas[0] : { anyOf: valueSchemas };
    const inputNames = [...new Set(names.flatMap(name => {
      const definition = (name.startsWith('let/') ? lam.fnCopies[name.slice(4)] : lam.codebase[name]) as
        Record<string, unknown> | undefined;
      return Object.keys(definition?.args as Record<string, unknown> ?? {});
    }).map(name => name.replace(/\?$/, '')))];
    const tools = [
      tool('read', 'Read a value from the workspace. Use a range for long text or lists.', {
        path: { type: 'string', enum: [...new Set([...readable, ...Object.keys(lam.codebase).map(name => `codebase/${name}`)])] },
        start: { type: 'integer' }, end: { type: 'integer' } }, ['path']),
      tool('write', 'Write a complete typed value to return or a local.', {
        path: { anyOf: [{ enum: slotPaths }, { type: 'string', pattern: '^let/[A-Za-z_][A-Za-z0-9_]*$' }] },
        type: { type: 'string' }, value: writeValue, source: { type: 'string', enum: readable } }, ['path', 'type']),
      tool('edit', 'Replace one exact occurrence in a text value.', {
        path: { type: 'string', enum: textSlots }, old: { type: 'string' }, new: { type: 'string' } }, ['path', 'old', 'new']),
      tool('run_code', 'Run exact TypeScript work in the selected engine.', {
        code: { type: 'string' }, engine: { enum: ['typescript-host'] } }, ['code', 'engine']),
    ];
    if (names.length) tools.push(tool('call', 'Call a checked function and place its result at to.', {
      function: { enum: names }, to: { anyOf: [{ enum: slotPaths }, { type: 'string', pattern: '^let/[A-Za-z_][A-Za-z0-9_]*$' }] },
      inputs: { type: 'object', properties: Object.fromEntries(inputNames.map(name => [name, { type: 'string', enum: readable }])), additionalProperties: false }, over: { type: 'string', enum: readable },
      init: {}, until: { type: 'string' }, max: { type: 'integer' } }, ['function', 'to']));
    if (names.length) tools.push(tool('mark_done', 'Mark completed or untaken lines of the program.', {
      start: { type: 'integer' }, end: { type: 'integer' }, skipped: { type: 'boolean' } }, ['start']));
    tools.push(tool('report_blocker', 'Explain information missing from the task.', { missing: { type: 'string' } }, ['missing']));
    tools.push(tool('report_error', 'Explain an unsatisfiable or invalid instruction.', { message: { type: 'string' } }, ['message']));
    return tools;
  }

  opening(session: NativeSession): string {
    const args = session.lam.args;
    const fields = session.lam.type.kind === 'lambda' ? session.lam.type.params.fields : [];
    return ['Workspace:', ...fields.map(field => `  args/${field.name} (${formatType(field.type)}, read-only): ` +
      (Object.hasOwn(args, field.name) ? JSON.stringify(dump(args[field.name]!)) : 'not supplied')),
    `  return (${session.lam.type.kind === 'lambda' ? formatType(session.lam.type.returns) : 'unknown'}): ` +
      (session.lam.return === MISSING ? 'not written yet' : 'written')].join('\n');
  }

  async run(session: NativeSession): Promise<string | void> {
    const lam = session.lam;
    const output = lam.type.kind === 'lambda' ? formatType(lam.type.returns) : 'unknown';
    const functions = Object.entries(lam.codebase).map(([name, raw]) => {
      const fn = raw as Record<string, unknown>;
      const args = Object.entries(fn.args as Record<string, string> ?? {}).map(([key, type]) => `${key}: ${type}`).join(', ');
      return `  ${name}(${args}) -> ${fn.returns}: ${fn.description ?? ''}`;
    });
    const original = lam.originalBody ?? lam.body;
    const program = functions.length ? original.replace(/^\n+|\n+$/g, '').split('\n').map((line, index) => {
      const text = line.trim(), markable = text && !text.startsWith('#') && !text.startsWith('function ');
      return `${index + 1} ${markable ? lam.marks[index + 1] === 'done' ? '[x]' : lam.marks[index + 1] === 'skipped' ? '[-]' : '[ ]' : '   '} ${line}`;
    }).join('\n') : lam.body.trim();
    const messages: Record<string, unknown>[] = [
      { role: 'system', content: this.options.systemPrompt ?? 'Interpret the program. Use tools to complete return; do not invent missing facts.' },
      { role: 'user', content: `${program}\n\nWrite the result to \`return\` (${output}).` +
        (functions.length ? `\n\nFunctions you can call:\n${functions.join('\n')}` : '') },
    ];
    const opening = this.opening(session);
    if (lam.type.kind === 'lambda' && lam.type.params.fields.length) messages.push({ role: 'assistant', content: '', tool_calls: [{ id: 'call_0', type: 'function',
      function: { name: 'read', arguments: '{"path":"args"}' } }] },
      { role: 'tool', tool_call_id: 'call_0', content: opening });
    const maxTurns = this.options.maxTurns ?? 64, maxTokens = this.options.maxTokens ?? 4000;
    const deadline = Date.now() + (this.options.maxSeconds ?? 900) * 1000;
    let tokens = 0, nudges = 0;
    for (let turn = 0; turn < maxTurns; turn++) {
      if (Date.now() >= deadline) return 'episode wall-clock budget exhausted';
      let allowance = maxTokens - tokens;
      if (this.options.turnTokens) allowance = Math.min(allowance, this.options.turnTokens);
      if (allowance < 1) return 'episode token budget exhausted';
      const response = await this.driver({ messages, tools: this.tools(session),
        temperature: this.options.temperature ?? 0.2,
        seed: session.runtime.seedPolicy.mode === 'backend' ? null :
          session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
          deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'model-turn', turn),
        max_tokens: allowance });
      session.runtime.checkInterruption();
      tokens += response.completion_tokens ?? allowance;
      if (tokens > maxTokens) return 'episode token budget exhausted';
      if (!response.calls?.length) {
        const marks = this.openMarks(session);
        if (marks.length) {
          if (++nudges > 2) return `validation failed: unfinished lines: ${marks.join(', ')}`;
          messages.push({ role: 'assistant', content: response.text ?? '' },
            { role: 'user', content: `Lines still marked [ ]: ${marks.join(', ')}. Mark completed work done and untaken work skipped (skipped=true).` });
          continue;
        }
        if (session.finish()) return;
        if (++nudges > 2) return `replied without writing \`return\`: ${(response.text ?? '').slice(0, 280)}`;
        const missing = lam.return === MISSING ? `return has not been written (${output})` :
          lam.type.kind === 'lambda' ? problems(lam.return, lam.type.returns, session.env, 'return').holes.map(d => d.path).join(', ') : '';
        messages.push({ role: 'assistant', content: response.text ?? '' },
          { role: 'user', content: missing || 'return is incomplete' });
        continue;
      }
      const calls = response.calls;
      const raw = calls.map(([name, args], i) => ({ id: `call_${turn}_${i}`, type: 'function',
        function: { name, arguments: JSON.stringify(args) } }));
      const results: NativeResult[] = [];
      for (const [index, [name, args]] of calls.entries()) {
        if (Date.now() >= deadline) return 'episode wall-clock budget exhausted';
        const result: NativeResult = await session.applyAsync(name, args);
        results.push(result);
        if (result.kind === 'blocked') return result.text;
        if (['rejected', 'refused', 'error', 'budget', 'completed'].includes(result.kind)) break;
      }
      messages.push({ role: 'assistant', content: response.text ?? '', tool_calls: raw.slice(0, results.length) });
      for (const [index, result] of results.entries())
        messages.push({ role: 'tool', tool_call_id: raw[index]!.id, content: result.text });
      if (results.at(-1)?.kind === 'budget') return 'action or tool-call budget exhausted';
    }
    return 'episode turn budget exhausted';
  }
}
