import { formatType } from './types.js';
import { MISSING, dump, problems } from './values.js';
import type { NativeResult, NativeSession } from './runtime.js';
import type { ModelTurn, ModelTurnRequest } from '../runtime.js';
import { deriveSeed } from './trace.js';

export type NativeModelDriver = (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;

const tool = (name: string, description: string, properties: Record<string, unknown>, required: string[]) => ({
  type: 'function', function: { name, description, parameters: { type: 'object', properties,
    required, additionalProperties: false } },
});

/** The native model loop. Program state stays in NativeSession, never in the model history. */
export class NativeToolAgent {
  constructor(readonly driver: NativeModelDriver,
    readonly options: { maxTurns?: number; maxTokens?: number; turnTokens?: number;
      temperature?: number; systemPrompt?: string } = {}) {}

  tools(session: NativeSession): unknown[] {
    const names = Object.keys(session.lam.codebase);
    const path = { type: 'string' };
    const tools = [
      tool('read', 'Read a value from the workspace.', { path, start: { type: 'integer' }, end: { type: 'integer' } }, ['path']),
      tool('write', 'Write a complete typed value to return or a local.', {
        path, type: { type: 'string' }, value: {}, source: path }, ['path', 'type']),
      tool('edit', 'Replace one exact occurrence in a text value.', { path, old: { type: 'string' }, new: { type: 'string' } }, ['path', 'old', 'new']),
      tool('run_code', 'Run exact TypeScript work in the selected engine.', {
        code: { type: 'string' }, engine: { enum: ['typescript-host'] } }, ['code', 'engine']),
    ];
    if (names.length) tools.push(tool('call', 'Call a checked function and place its result at to.', {
      function: { enum: names }, to: path, inputs: { type: 'object' }, over: path,
      init: {}, until: { type: 'string' }, max: { type: 'integer' } }, ['function', 'to']));
    tools.push(tool('report_blocker', 'Explain information missing from the task.', { missing: { type: 'string' } }, ['missing']));
    tools.push(tool('report_error', 'Explain an unsatisfiable or invalid instruction.', { message: { type: 'string' } }, ['message']));
    return tools;
  }

  opening(session: NativeSession): string {
    const args = session.lam.args;
    const fields = session.lam.type.kind === 'lambda' ? session.lam.type.params.fields : [];
    return fields.map(field => `${field.name} (${formatType(field.type)}): ` +
      (Object.hasOwn(args, field.name) ? JSON.stringify(dump(args[field.name]!)) : 'not supplied')).join('\n');
  }

  async run(session: NativeSession): Promise<string | void> {
    const lam = session.lam;
    const output = lam.type.kind === 'lambda' ? formatType(lam.type.returns) : 'unknown';
    const messages: Record<string, unknown>[] = [
      { role: 'system', content: this.options.systemPrompt ?? 'Interpret the program. Use tools to complete return; do not invent missing facts.' },
      { role: 'user', content: `${lam.body.trim()}\n\nWrite the result to \`return\` (${output}).` },
    ];
    const opening = this.opening(session);
    if (opening) messages.push({ role: 'assistant', content: '', tool_calls: [{ id: 'call_0', type: 'function',
      function: { name: 'read', arguments: '{"path":"args"}' } }] },
      { role: 'tool', tool_call_id: 'call_0', content: opening });
    const maxTurns = this.options.maxTurns ?? 64, maxTokens = this.options.maxTokens ?? 4000;
    let tokens = 0;
    for (let turn = 0; turn < maxTurns; turn++) {
      let allowance = maxTokens - tokens;
      if (this.options.turnTokens) allowance = Math.min(allowance, this.options.turnTokens);
      if (allowance < 1) return 'episode token budget exhausted';
      const response = await this.driver({ messages, tools: this.tools(session),
        temperature: this.options.temperature ?? 0.2,
        seed: session.runtime.seedPolicy.mode === 'backend' ? null :
          session.runtime.seedPolicy.mode === 'compatibility' ? 0 :
          deriveSeed(session.runtime.seedPolicy.root!, session.path, session.lam.attempts, 'model-turn', turn),
        max_tokens: allowance });
      tokens += response.completion_tokens ?? allowance;
      if (tokens > maxTokens) return 'episode token budget exhausted';
      if (!response.calls?.length) {
        if (session.finish()) return;
        const missing = lam.return === MISSING ? `return has not been written (${output})` :
          lam.type.kind === 'lambda' ? problems(lam.return, lam.type.returns, session.env, 'return').holes.map(d => d.path).join(', ') : '';
        messages.push({ role: 'assistant', content: response.text ?? '' },
          { role: 'user', content: missing || 'return is incomplete' });
        continue;
      }
      const calls = response.calls;
      const raw = calls.map(([name, args], i) => ({ id: `call_${turn}_${i}`, type: 'function',
        function: { name, arguments: JSON.stringify(args) } }));
      messages.push({ role: 'assistant', content: response.text ?? '', tool_calls: raw });
      for (const [index, [name, args]] of calls.entries()) {
        const result: NativeResult = await session.applyAsync(name, args);
        messages.push({ role: 'tool', tool_call_id: raw[index]!.id, content: result.text });
        if (result.kind === 'blocked') return result.text;
      }
    }
    return 'episode turn budget exhausted';
  }
}
