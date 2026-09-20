import { spawn, type ChildProcessWithoutNullStreams } from 'node:child_process';
import { existsSync } from 'node:fs';
import { createInterface } from 'node:readline';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { EvalFailure, TypeScriptEnvironment, portable, type EvalRequest } from './environment.js';

const PROTOCOL = 'natlang-ts-host/1';
const SOURCE_ROOT = resolve(fileURLToPath(new URL('../..', import.meta.url)));
const DEFAULT_ROOT = existsSync(join(SOURCE_ROOT, 'natlang', 'ts_host_bridge.py')) ? SOURCE_ROOT : process.cwd();

export type Source =
  | { kind: 'program'; program: Record<string, unknown> }
  | { kind: 'definitions'; entries: Record<string, Record<string, unknown>>; root: string }
  | { kind: 'file'; path: string };
export type RunOptions = { seed?: { mode?: 'compatibility' | 'derived' | 'backend'; root?: number;
  version?: string }; model?: { temperature?: number; max_turns?: number; max_tokens?: number;
  max_seconds?: number; turn_tokens?: number }; world_seed?: number;
  max_episodes?: number; max_depth?: number; run_id?: string };
export type ModelTurnRequest = { messages: unknown[]; tools: unknown[]; temperature: number;
  seed: number | null; max_tokens: number };
export type ModelTurn = { calls?: [string, Record<string, unknown>][]; text?: string;
  raw_calls?: unknown[]; completion_tokens?: number;
  value_confidence?: (number | { geometric_mean?: number } | null)[];
  raw_response?: Record<string, unknown> };
export type RunRequest = { source: Source; inputs?: Record<string, unknown>;
  options?: RunOptions; mapWorkers?: number; tracePath?: string;
  typescript?: boolean; modelTurn?: (request: ModelTurnRequest) => Promise<ModelTurn> | ModelTurn;
  streams?: Record<string, AsyncIterable<unknown>>;
  capabilities?: Record<string, (args: unknown[]) => Promise<unknown> | unknown>;
  signal?: AbortSignal; timeoutMs?: number };
export type RunResult = { outcome: { kind: string; path: string; detail: string };
  value: unknown; emitted: unknown[]; trace: Record<string, unknown>[] | null; run_id: string };

type WireMessage = { protocol: string; kind: string; id?: number; [key: string]: unknown };

/** One run per Python process, preserving the complete natlang interpreter semantics. */
export class NatlangHost {
  readonly environment: TypeScriptEnvironment;
  readonly python: string;
  readonly cwd: string;
  private closed = false;
  private active = new Set<ChildProcessWithoutNullStreams>();

  constructor(options: { environment?: TypeScriptEnvironment; python?: string; cwd?: string } = {}) {
    this.environment = options.environment ?? new TypeScriptEnvironment();
    this.python = options.python ?? process.env.NATLANG_PYTHON ?? 'python3';
    this.cwd = options.cwd ?? DEFAULT_ROOT;
  }

  run(request: RunRequest): Promise<RunResult> {
    if (this.closed) return Promise.reject(new Error('natlang host is disposed'));
    const child = spawn(this.python, ['-u', '-m', 'natlang.ts_host_bridge'], {
      cwd: this.cwd, stdio: ['pipe', 'pipe', 'pipe'],
      env: { ...process.env, PYTHONPATH: [this.cwd, process.env.PYTHONPATH].filter(Boolean).join(':') },
    });
    this.active.add(child);
    const lines = createInterface({ input: child.stdout, crlfDelay: Infinity });
    const streams = Object.fromEntries(Object.entries(request.streams ?? {}).map(([part, source]) =>
      [part, source[Symbol.asyncIterator]()])) as Record<string, AsyncIterator<unknown>>;
    const start = { protocol: PROTOCOL, kind: 'start', source: request.source,
      inputs: request.inputs ?? {}, options: request.options ?? {},
      map_workers: request.mapWorkers ?? 1, trace_path: request.tracePath,
      stream_parts: Object.keys(streams),
      capabilities: Object.keys(request.capabilities ?? {}),
      environment_mode: this.environment.mode,
      typescript: request.typescript ?? true };
    return new Promise<RunResult>((resolveRun, rejectRun) => {
      let settled = false;
      let stderr = '';
      let timer: NodeJS.Timeout | undefined;
      const fail = (error: Error) => {
        if (settled) return;
        settled = true;
        child.kill();
        cleanup();
        rejectRun(error);
      };
      const finish = (value: RunResult) => {
        if (settled) return;
        settled = true;
        cleanup();
        child.stdin.end();
        resolveRun(value);
      };
      const abort = () => fail(new Error('natlang run aborted; external effects may have occurred'));
      const cleanup = () => {
        if (timer) clearTimeout(timer);
        request.signal?.removeEventListener('abort', abort);
        this.active.delete(child);
      };
      if (request.signal?.aborted) { abort(); return; }
      request.signal?.addEventListener('abort', abort, { once: true });
      if (request.timeoutMs !== undefined) {
        if (!Number.isFinite(request.timeoutMs) || request.timeoutMs < 1) {
          fail(new RangeError('timeoutMs must be positive')); return;
        }
        timer = setTimeout(() => fail(new Error('natlang run timed out; external effects may have occurred')), request.timeoutMs);
      }
      child.stderr.on('data', chunk => { stderr = (stderr + String(chunk)).slice(-4096); });
      child.on('error', error => fail(error));
      child.on('close', code => {
        this.active.delete(child);
        if (!settled) fail(new Error(`natlang bridge exited ${code}: ${stderr}`));
      });
      lines.on('line', line => {
        if (settled) return;
        void (async () => {
          let message: WireMessage;
          try { message = JSON.parse(line) as WireMessage; }
          catch { throw new Error('invalid JSON from natlang bridge'); }
          if (message.protocol !== PROTOCOL) throw new Error('natlang bridge protocol mismatch');
          if (message.kind === 'complete') {
            finish({ outcome: message.outcome as RunResult['outcome'], value: message.value,
              emitted: message.emitted as unknown[], trace: message.trace as RunResult['trace'],
              run_id: message.run_id as string });
            return;
          }
          if (message.kind === 'fatal') {
            fail(new Error(`${message.type}: ${message.error}`)); return;
          }
          if (typeof message.id !== 'number') throw new Error('bridge request missing id');
          try {
            let value: unknown;
            if (message.kind === 'model_turn') {
              if (!request.modelTurn) throw new Error('this run has no model-turn callback');
              value = await request.modelTurn(message as unknown as ModelTurnRequest);
            } else if (message.kind === 'eval') {
              if (request.typescript === false) throw new Error('TypeScript engine is disabled');
              value = this.environment.execute(message as unknown as EvalRequest);
            } else if (message.kind === 'stream_poll') {
              const part = message.part as string;
              const iterator = streams[part];
              if (!iterator) throw new Error(`unknown stream part ${part}`);
              const step = await iterator.next();
              value = step.done ? { kind: 'closed' } : { kind: 'item', value: step.value };
            } else if (message.kind === 'capability') {
              const name = message.name as string;
              const capability = request.capabilities?.[name];
              if (!capability) throw new Error(`unknown capability ${name}`);
              if (!Array.isArray(message.args)) throw new TypeError('capability arguments must be a list');
              value = await capability(message.args);
            } else throw new Error(`unknown bridge request ${message.kind}`);
            child.stdin.write(JSON.stringify({ protocol: PROTOCOL, kind: 'reply', id: message.id,
              value: portable(value) }) + '\n');
          } catch (error) {
            child.stdin.write(JSON.stringify({ protocol: PROTOCOL, kind: 'reply', id: message.id,
              error: error instanceof Error ? error.message : String(error),
              events: error instanceof EvalFailure ? error.events : [] }) + '\n');
          }
        })().catch(error => fail(error instanceof Error ? error : new Error(String(error))));
      });
      try { child.stdin.write(JSON.stringify(start) + '\n'); }
      catch (error) { fail(error instanceof Error ? error : new Error(String(error))); }
    });
  }

  close(): void {
    if (this.closed) return;
    this.closed = true;
    for (const child of this.active) child.kill();
    this.active.clear();
    this.environment.close();
  }
}
