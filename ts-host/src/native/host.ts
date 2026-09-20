import { writeFileSync } from 'node:fs';
import { TypeScriptEnvironment } from '../environment.js';
import type { RunRequest, RunResult } from '../runtime.js';
import { NativeToolAgent } from './agent.js';
import { checkedDefinitions } from './codebase.js';
import { NativeRuntime, type NativeStream } from './runtime.js';
import { loadFunctionFile } from './source.js';
import { TypeEnv } from './types.js';
import { buildPending, coerce, dump, isPending, type Pending } from './values.js';

/** Python-free host. Its interpreter remains opt-in while differential parity is expanded. */
export class NativeNatlangHost {
  readonly environment: TypeScriptEnvironment;
  private readonly ownsEnvironment: boolean;
  private closed = false;
  private running = false;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; mode?: 'fresh' | 'retained' } = {}) {
    this.ownsEnvironment = !options.environment;
    this.environment = options.environment ?? new TypeScriptEnvironment({ host: options.host, mode: options.mode });
  }

  async run(request: RunRequest): Promise<RunResult> {
    if (this.closed) throw new Error('natlang host is disposed');
    if (this.running) throw new Error('concurrent runs cannot share a native eval environment');
    if (request.signal?.aborted) throw new Error('natlang run aborted');
    if (request.mapWorkers !== undefined && request.mapWorkers !== 1)
      throw new RangeError('native Map workers must be 1 until concurrent Map parity is implemented');
    if (request.typescript === false) throw new Error('native host requires the TypeScript eval engine');
    this.running = true;
    let runtime: NativeRuntime | undefined;
    try {
      let root: Pending;
      if (request.source.kind === 'program') root = buildPending(request.source.program);
      else if (request.source.kind === 'definitions') root = checkedDefinitions(
        request.source.entries as Parameters<typeof checkedDefinitions>[0], request.source.root).instantiate(request.inputs);
      else root = loadFunctionFile(request.source.path);
      if (request.source.kind !== 'definitions' && request.inputs) {
        if (!isPending(root) || root.nodeKind !== 'lambda' || root.type.kind !== 'lambda')
          throw new TypeError('inputs require a Lambda program');
        const env = new TypeEnv().child(root.types);
        for (const [name, value] of Object.entries(request.inputs)) {
          const field = root.type.params.fields.find(item => item.name === name);
          if (!field) throw new TypeError(`${name} is not a program parameter`);
          root.args[name] = coerce(value, field.type, env, `args/${name}`);
        }
      }
      const streams = Object.entries(request.streams ?? {});
      if (streams.some(([name]) => name !== 'over') || (streams.length && root.nodeKind !== 'fold'))
        throw new TypeError('only a root Fold over stream is supported');
      const iterator = streams[0]?.[1][Symbol.asyncIterator]();
      const stream: NativeStream | undefined = iterator && { async poll() {
        try { const step = await iterator.next(); return step.done ? { kind: 'closed' } : { kind: 'item', value: step.value }; }
        catch (error) { return { kind: 'failed', detail: error instanceof Error ? error.message : String(error) }; }
      } };
      const agent = request.modelTurn ? new NativeToolAgent(request.modelTurn, {
        maxTurns: request.options?.model?.max_turns, maxTokens: request.options?.model?.max_tokens,
        turnTokens: request.options?.model?.turn_tokens, temperature: request.options?.model?.temperature,
        maxSeconds: request.options?.model?.max_seconds }) : undefined;
      runtime = new NativeRuntime({ environment: this.environment, stream,
        agent: agent ? session => agent.run(session) : undefined,
        capabilities: request.capabilities as Record<string, (args: unknown[]) => unknown>,
        maxEpisodes: request.options?.max_episodes, maxDepth: request.options?.max_depth,
        runId: request.options?.run_id, seedPolicy: request.options?.seed?.mode ? {
          mode: request.options.seed.mode, root: request.options.seed.root } : undefined });
      const outcome = await runtime.runRoot(root);
      if (request.tracePath) writeFileSync(request.tracePath,
        runtime.trace.events.map(event => JSON.stringify(event)).join('\n') + '\n');
      return { outcome: { kind: outcome.outcome.kind, path: outcome.outcome.path, detail: outcome.outcome.detail },
        value: dump(outcome.value), emitted: outcome.emitted,
        trace: request.tracePath ? runtime.trace.events as Record<string, unknown>[] : null,
        run_id: runtime.options.runId };
    } finally {
      runtime?.close();
      this.running = false;
    }
  }

  close(): void { this.closed = true; if (this.ownsEnvironment) this.environment.close(); }
}
