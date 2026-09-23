import { existsSync, readFileSync, readdirSync, statSync, writeFileSync } from 'node:fs';
import { extname, join } from 'node:path';
import { randomUUID } from 'node:crypto';
import YAML from 'yaml';
import { TypeScriptEnvironment } from '../environment.js';
import { applicationCapabilityPrompt } from '../application-packages.js';
import type { RunRequest, RunResult } from '../contracts.js';
import { NativeToolAgent, type NativeReviewOptions } from './agent.js';
import { TOOLS_PROMPT } from './prompt.js';
import { checkedDefinitions } from './codebase.js';
import { NativeRuntime, type NativeStream } from './runtime.js';
import { loadFunctionFile } from './source.js';
import { NativeSourceWorkspace } from './workspace.js';
import { TypeEnv, type Type } from './types.js';
import { buildPending, coerce, dump, isPending, type Pending } from './values.js';

export type NativeRunRequest = RunRequest & { review?: NativeReviewOptions; parallelMapSafe?: boolean;
  validationFeedback?: 'caller' | 'local' };

/** Default Python-free natlang host. */
export class NativeNatlangHost {
  readonly environment: TypeScriptEnvironment;
  private readonly ownsEnvironment: boolean;
  private closed = false;
  private running = false;

  constructor(options: { environment?: TypeScriptEnvironment; host?: object; mode?: 'fresh' | 'retained';
    workspace?: string; network?: boolean } = {}) {
    this.ownsEnvironment = !options.environment;
    this.environment = options.environment ?? new TypeScriptEnvironment({ host: options.host, mode: options.mode,
      workspace: options.workspace, network: options.network });
  }

  async run(request: NativeRunRequest): Promise<RunResult> {
    if (this.closed) throw new Error('natlang host is disposed');
    if (this.running) throw new Error('concurrent runs cannot share a native eval environment');
    if (request.signal?.aborted) throw new Error('natlang run aborted');
    if (request.mapWorkers !== undefined && (!Number.isInteger(request.mapWorkers) || request.mapWorkers < 1))
      throw new RangeError('mapWorkers must be positive');
    this.running = true;
    let runtime: NativeRuntime | undefined;
    let runFailed = false;
    const unbind: (() => void)[] = [];
    try {
      let root: Pending;
      let fileStreams: Record<string, unknown> = {};
      if (request.source.kind === 'program') root = buildPending(request.source.program);
      else if (request.source.kind === 'definitions') root = checkedDefinitions(
        request.source.entries as Parameters<typeof checkedDefinitions>[0], request.source.root).instantiate(request.inputs);
      else if (['.nl', '.ts'].includes(extname(request.source.path))) root = loadFunctionFile(request.source.path,
        { packageImports: this.environment.scopeCapabilities.allowModules });
      else {
        const doc = YAML.parse(readFileSync(request.source.path, 'utf8')) as Record<string, unknown>;
        root = buildPending(doc.program ?? doc);
        fileStreams = doc.streams as Record<string, unknown> ?? {};
      }
      if (request.source.kind !== 'definitions' && request.inputs) {
        if (!isPending(root) || root.nodeKind !== 'lambda' || root.type.kind !== 'lambda')
          throw new TypeError('inputs require a Lambda program');
        const env = new TypeEnv().child(root.types);
        for (const [name, value] of Object.entries(request.inputs)) {
          const field = root.type.params.fields.find(item => item.name === name);
          if (!field) throw new TypeError(`${name} is not a program parameter`);
          const imported = typeof value === 'string' && value.length < 256 && !value.includes('\n') && existsSync(value) ?
            this.importPath(value, field.type, env) : value;
          root.args[name] = coerce(imported, field.type, env, `args/${name}`);
        }
      }
      const streams = Object.entries({ ...fileStreams, ...request.streams });
      if (streams.some(([name]) => name !== 'over') || (streams.length && root.nodeKind !== 'fold'))
        throw new TypeError('only a root Fold over stream is supported');
      const source = streams[0]?.[1] as AsyncIterable<unknown> | Iterable<unknown> | undefined;
      const iterator = source && (Symbol.asyncIterator in source ? source[Symbol.asyncIterator]() :
        (async function* () { yield* source as Iterable<unknown>; })()[Symbol.asyncIterator]());
      const stream: NativeStream | undefined = iterator && { async poll() {
        try { const step = await iterator.next(); return step.done || step.value === '$close' ? { kind: 'closed' } : { kind: 'item', value: step.value }; }
        catch (error) { return { kind: 'failed', detail: error instanceof Error ? error.message : String(error) }; }
      } };
      const agent = request.modelTurn ? new NativeToolAgent(request.modelTurn, {
        systemPrompt: TOOLS_PROMPT + applicationCapabilityPrompt(this.environment.scopeCapabilities),
        maxTurns: request.options?.model?.max_turns, maxTokens: request.options?.model?.max_tokens,
        turnTokens: request.options?.model?.turn_tokens, temperature: request.options?.model?.temperature,
        segmentTurns: request.options?.model?.segment_turns,
        segmentMessages: request.options?.model?.segment_messages,
        maxSeconds: request.options?.model?.max_seconds, validationFeedback: request.validationFeedback,
        review: request.review }) : undefined;
      runtime = new NativeRuntime({ environment: this.environment, stream,
        agent: agent ? session => agent.run(session) : undefined,
        capabilities: request.capabilities as Record<string, (args: unknown[]) => unknown>,
        maxEpisodes: request.options?.max_episodes, maxDepth: request.options?.max_depth,
        maxActions: request.options?.max_actions, maxToolCalls: request.options?.max_tool_calls,
        mapWorkers: request.mapWorkers, parallelModelSafe: request.parallelMapSafe,
        runId: request.options?.run_id ?? randomUUID(), signal: request.signal, timeoutMs: request.timeoutMs,
        seedPolicy: request.options?.seed?.mode ? {
        mode: request.options.seed.mode, root: request.options.seed.root } : undefined });
      for (const value of Object.values(this.environment.host))
        if (value instanceof NativeSourceWorkspace) unbind.push(value.bindParent(runtime));
      let timer: ReturnType<typeof setTimeout> | undefined;
      let abortListener: (() => void) | undefined;
      const interruption = new Promise<never>((_, reject) => {
        if (request.timeoutMs !== undefined)
          timer = setTimeout(() => reject(new Error('natlang run timed out; external effects may have occurred')),
            request.timeoutMs);
        if (request.signal) {
          abortListener = () => reject(new Error('natlang run aborted; external effects may have occurred'));
          request.signal.addEventListener('abort', abortListener, { once: true });
          if (request.signal.aborted) abortListener();
        }
      });
      let outcome;
      try { outcome = await Promise.race([runtime.runRoot(root), interruption]); }
      finally {
        if (timer) clearTimeout(timer);
        if (abortListener) request.signal?.removeEventListener('abort', abortListener);
      }
      return { outcome: { kind: outcome.outcome.kind, path: outcome.outcome.path, detail: outcome.outcome.detail },
        value: dump(outcome.value), emitted: outcome.emitted,
        trace: request.tracePath ? runtime.trace.events as Record<string, unknown>[] : null,
        run_id: runtime.options.runId };
    } catch (error) {
      runFailed = true;
      throw error;
    } finally {
      let traceFailure: unknown;
      if (request.tracePath && runtime) {
        try { writeFileSync(request.tracePath,
          runtime.trace.events.map(event => JSON.stringify(event)).join('\n') + '\n'); }
        catch (error) { if (!runFailed) traceFailure = error; }
      }
      for (const release of unbind.reverse()) release();
      runtime?.close();
      this.running = false;
      if (traceFailure) throw traceFailure;
    }
  }

  private importPath(path: string, type: Type, env: TypeEnv): unknown {
    const resolved = env.resolve(type);
    if (statSync(path).isDirectory()) {
      const files = readdirSync(path).filter(name => !name.startsWith('.')).sort();
      if (resolved.kind === 'list') return files.map(name => this.importPath(join(path, name), resolved.element, env));
      if (resolved.kind === 'dict') return Object.fromEntries(files.map(name =>
        [name.replace(/\.[^.]*$/, ''), this.importPath(join(path, name), resolved.element, env)]));
      throw new TypeError(`${path} is a directory but the parameter type is not a list or Dict`);
    }
    const text = readFileSync(path, 'utf8');
    return ['.json', '.yaml', '.yml'].includes(extname(path)) || !(resolved.kind === 'prim' && resolved.name === 'string') ?
      YAML.parse(text) : text;
  }

  close(): void { this.closed = true; if (this.ownsEnvironment) this.environment.close(); }
}
