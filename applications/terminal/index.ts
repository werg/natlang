/**
 * Semantic terminal: a trusted recipe terminal. Natlang chooses one exact recipe for a request and
 * explains actual completions; `RecipeTerminal` owns the job promises, correlates completion IDs,
 * and rejects forged completions. Recipes run argv without a shell. A request that arrives while a
 * job is active is declined for resubmission, and a cancellation never claims a rollback.
 */
import { spawn } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { isAbsolute, relative, resolve } from 'node:path';
import type { FolderHandle } from '@natlang/node';
import interpret from './interpret.nl';
import explain from './explain.nl';
import type { Recipe, TerminalEvent } from './types.js';

export type * from './types.js';
export type RecipeResult = { status: 'ok' | 'failed' | 'unknown', detail: string };
export type RunnableRecipe = Recipe & { run(context: { signal: AbortSignal, requestId: string, jobId: string }): Promise<RecipeResult> };
export type Outcome = { request_id: string, job_id: string, status: string, detail: string };
export type Session = { revision: number, active_request: string, active_job: string, status: string, messages: string[], history: Outcome[] };
type Job = { id: string, requestId: string, recipeId: string, cancelRequested: boolean, result: RecipeResult | null,
  promise: Promise<unknown> | null, controller: AbortController, completion?: TerminalEvent };

export const emptyTerminalSession = (): Session => ({ revision: 0, active_request: '', active_job: '', status: 'idle', messages: [], history: [] });

export class RecipeTerminal {
  private readonly recipes: Map<string, RunnableRecipe>;
  private readonly jobs = new Map<string, Job>();
  private readonly requests = new Set<string>();
  private nextJob = 1;
  private readonly events: Record<string, unknown>[] = [];
  private readonly listeners = new Set<(event: TerminalEvent) => void>();

  constructor(recipes: RunnableRecipe[]) {
    this.recipes = new Map(recipes.map(recipe => [recipe.id, recipe]));
    if (this.recipes.size !== recipes.length) throw new Error('duplicate recipe IDs');
  }

  catalog(): Recipe[] { return [...this.recipes.values()].map(({ id, description }) => ({ id, description })); }

  start(requestId: string, recipeId: string): { id: string, request_id: string, status: string, detail: string } {
    if (!requestId || this.requests.has(requestId)) throw new Error(`duplicate request: ${requestId}`);
    const recipe = this.recipes.get(recipeId);
    if (!recipe) throw new Error(`unknown recipe: ${recipeId}`);
    this.requests.add(requestId);
    const id = `job-${this.nextJob++}`;
    const job: Job = { id, requestId, recipeId, cancelRequested: false, result: null, promise: null, controller: new AbortController() };
    this.jobs.set(id, job);
    this.events.push({ operation: 'terminal.start', request_id: requestId, job_id: id, recipe_id: recipeId });
    job.promise = Promise.resolve().then(() => recipe.run({ signal: job.controller.signal, requestId, jobId: id })).then(result => {
      job.result = { status: ['ok', 'failed', 'unknown'].includes(result?.status) ? result.status : 'unknown', detail: String(result?.detail ?? '') };
      return this.complete(job);
    }, error => {
      job.result = { status: 'unknown', detail: error instanceof Error ? error.message : String(error) };
      return this.complete(job);
    });
    return { id, request_id: requestId, status: 'running', detail: '' };
  }

  cancel(jobId: string): { id: string, request_id: string, status: string, detail: string } {
    const job = this.jobs.get(jobId);
    if (!job) throw new Error(`unknown job: ${jobId}`);
    job.cancelRequested = true;
    job.controller.abort();
    this.events.push({ operation: 'terminal.cancel-requested', job_id: jobId });
    return { id: jobId, request_id: job.requestId, status: 'cancel-requested', detail: '' };
  }

  async wait(requestId: string): Promise<TerminalEvent> {
    const job = [...this.jobs.values()].find(row => row.requestId === requestId);
    if (!job) throw new Error(`no job for request: ${requestId}`);
    await job.promise;
    return { ...job.completion! };
  }

  private complete(job: Job): RecipeResult {
    const result = job.result!;
    const detail = job.cancelRequested ? `Cancellation was requested. Actual result: ${result.detail}` : result.detail;
    this.events.push({ operation: 'terminal.complete', request_id: job.requestId, job_id: job.id, status: result.status,
      cancel_requested: job.cancelRequested });
    job.completion = { kind: 'complete', id: `completion-${job.id}`, request_id: job.requestId, job_id: job.id, text: '',
      status: result.status, detail };
    for (const listener of this.listeners) {
      try { listener({ ...job.completion }); }
      catch (error) { this.events.push({ operation: 'terminal.listener-failed', job_id: job.id,
        detail: error instanceof Error ? error.message : String(error) }); }
    }
    return result;
  }

  subscribe(listener: (event: TerminalEvent) => void): () => void {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  }

  close(): void {
    for (const job of this.jobs.values()) if (!job.result) job.controller.abort();
    this.listeners.clear();
  }

  /** True when `event` is exactly the recorded completion of its job. */
  confirm(event: TerminalEvent): boolean {
    const job = this.jobs.get(event.job_id);
    return !!job?.completion && job.completion.request_id === event.request_id &&
      job.completion.status === event.status && job.completion.detail === event.detail;
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/** Reduce one user or job event into the session. */
export async function step(terminal: RecipeTerminal, session: Session, item: TerminalEvent, files?: FolderHandle): Promise<Session> {
  const say = (message: string, changes: Partial<Session> = {}): Session => ({ ...session, ...changes, messages: [...session.messages, message] });
  switch (item.kind) {
    case 'request': {
      if (!item.id || item.id === session.active_request || session.history.some(row => row.request_id === item.id))
        return say('Duplicate or empty request ID.');
      if (session.status === 'running' || session.status === 'cancel-requested')
        return say(`Request ${item.id} was not accepted while ${session.active_request} is active; resubmit it after completion.`);
      const catalog = terminal.catalog();
      const recipe = await interpret(item.text, catalog, files);
      if (!catalog.some(row => row.id === recipe))
        return say(`No supported recipe for: ${item.text}`, { revision: session.revision + 1, status: 'unsupported' });
      const job = terminal.start(item.id, recipe);
      return say(`Started ${recipe} for ${item.id}.`, { revision: session.revision + 1, active_request: item.id, active_job: job.id, status: job.status });
    }
    case 'complete': {
      if (!terminal.confirm(item)) return say(`Unverified completion ignored: ${item.job_id}`);
      const message = await explain(item);
      const history = [...session.history, { request_id: item.request_id, job_id: item.job_id, status: item.status, detail: item.detail }];
      if (item.request_id !== session.active_request || item.job_id !== session.active_job)
        return say(`Stale result ${item.job_id}: ${message}`, { history });
      return say(message, { history, revision: session.revision + 1, active_request: '', active_job: '', status: item.status });
    }
    case 'cancel':
      if (!session.active_job || item.request_id !== session.active_request) return say(`No active job for ${item.request_id}.`);
      terminal.cancel(session.active_job);
      return say(`Cancellation requested for ${session.active_job}; awaiting actual outcome.`,
        { revision: session.revision + 1, status: 'cancel-requested' });
    case 'recover':
      if (session.status !== 'running' && session.status !== 'cancel-requested') return session;
      return say(`The prior host stopped while ${session.active_job} was active. Its outcome is unknown; inspect external effects before retrying.`,
        { revision: session.revision + 1, active_request: '', active_job: '', status: 'unknown' });
    default:
      return say(`Ignored unknown event kind: ${(item as TerminalEvent).kind}`);
  }
}

function inside(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

export type CommandRecipe = { id: string, description: string, argv: string[], cwd?: string, env?: Record<string, string>, timeoutMs?: number };

/** Exact argv recipes for a trusted local workspace. No shell text is evaluated. */
export class CommandRecipeLibrary {
  private readonly root: string;
  private readonly outputBytes: number;
  private readonly timeoutMs: number;
  private readonly definitions: (CommandRecipe & { cwd: string })[];

  constructor(root: string, definitions: CommandRecipe[], { outputBytes = 64 * 1024, timeoutMs = 10 * 60_000 } = {}) {
    this.root = realpathSync(root);
    this.outputBytes = outputBytes;
    this.timeoutMs = timeoutMs;
    this.definitions = definitions.map(definition => {
      if (!definition.id || !definition.description || !Array.isArray(definition.argv) || !definition.argv.length ||
          definition.argv.some(value => typeof value !== 'string'))
        throw new Error('command recipe needs id, description, and string argv');
      const cwd = resolve(this.root, definition.cwd ?? '.');
      if (!inside(this.root, cwd)) throw new Error(`recipe cwd escapes workspace: ${definition.id}`);
      return { ...definition, cwd };
    });
    if (new Set(this.definitions.map(row => row.id)).size !== this.definitions.length) throw new Error('duplicate command recipe IDs');
  }

  recipes(): RunnableRecipe[] {
    return this.definitions.map(definition => ({ id: definition.id, description: definition.description,
      run: context => this.run(definition, context) }));
  }

  run(definition: CommandRecipe & { cwd: string }, { signal }: { signal?: AbortSignal } = {}): Promise<RecipeResult> {
    return new Promise(resolveResult => {
      const child = spawn(definition.argv[0]!, definition.argv.slice(1), { cwd: definition.cwd, shell: false,
        stdio: ['ignore', 'pipe', 'pipe'], env: definition.env ? { ...process.env, ...definition.env } : process.env });
      let stdout = Buffer.alloc(0), stderr = Buffer.alloc(0), settled = false, timedOut = false, cancelled = false;
      const capture = (field: 'stdout' | 'stderr', chunk: Buffer) => {
        const value = Buffer.concat([field === 'stdout' ? stdout : stderr, chunk]);
        const bounded = value.subarray(Math.max(0, value.length - this.outputBytes));
        if (field === 'stdout') stdout = bounded; else stderr = bounded;
      };
      child.stdout.on('data', chunk => capture('stdout', chunk));
      child.stderr.on('data', chunk => capture('stderr', chunk));
      const abort = () => {
        cancelled = true; child.kill('SIGTERM');
        setTimeout(() => { if (!settled) child.kill('SIGKILL'); }, 5000).unref();
      };
      const duration = Math.max(1, Math.min(definition.timeoutMs ?? this.timeoutMs, 24 * 60 * 60_000));
      const timer = setTimeout(() => { timedOut = true; child.kill('SIGTERM'); }, duration);
      timer.unref();
      const finish = (status: RecipeResult['status'], detail: string) => {
        if (settled) return;
        settled = true; clearTimeout(timer);
        signal?.removeEventListener('abort', abort);
        resolveResult({ status, detail });
      };
      child.on('error', error => finish('unknown', error.message));
      child.on('exit', (code, processSignal) => {
        const output = [stdout.toString('utf8').trim(), stderr.toString('utf8').trim()].filter(Boolean).join('\n');
        if (cancelled || timedOut) return finish('unknown',
          `${cancelled ? 'Cancellation' : 'Timeout'} requested; exit=${code}; signal=${processSignal}; ${output}`.trim());
        finish(code === 0 ? 'ok' : 'failed', `exit=${code}; ${output || '(no output)'}`);
      });
      signal?.addEventListener('abort', abort, { once: true });
      if (signal?.aborted) abort();
    });
  }
}

export function natlangWorkspaceRecipes(root: string): CommandRecipeLibrary {
  return new CommandRecipeLibrary(root, [
    { id: 'repository-status', description: 'inspect concise Git working tree status', argv: ['git', 'status', '--short'] },
    { id: 'repository-diff', description: 'summarize current uncommitted Git changes', argv: ['git', 'diff', '--stat'] },
    { id: 'list-files', description: 'list workspace files tracked or visible to ripgrep', argv: ['rg', '--files'] },
    { id: 'test-typescript-host', description: 'build and test the natlang TypeScript host',
      argv: ['npm', '--prefix', 'ts-host', 'test'], timeoutMs: 60 * 60_000 },
    { id: 'build-typescript-host', description: 'build the natlang TypeScript host and browser bundle',
      argv: ['npm', '--prefix', 'ts-host', 'run', 'build'], timeoutMs: 60 * 60_000 },
  ]);
}
