/**
 * Semantic terminal: a trusted recipe terminal. Natlang chooses one exact recipe for a request and
 * explains actual completions; `RecipeTerminal` owns the job promises, correlates completion IDs,
 * and rejects forged completions. Recipes run argv without a shell. A request that arrives while a
 * job is active is declined for resubmission, and a completion carries the fact that cancellation was requested
 * (`cancel_requested`) beside the actual result. The recipe catalog, its limits and the wording of the session's
 * messages are data files (recipes.json, messages.json).
 */
import { spawn } from 'node:child_process';
import { realpathSync } from 'node:fs';
import { isAbsolute, relative, resolve } from 'node:path';
import { builtin, untrusted, type FolderHandle, type Untrusted } from '@natlang/node';
import chooseRecipe from './interpret/chooseRecipe.nl';
import explain from './explain/explain.nl';
import { loadMessages, loadRecipeData, renderMessage, type CommandRecipe, type MessageTable, type RecipeData, type RecipeLimits } from './data.js';
import type { Recipe, ResultStatus, SessionStatus, TerminalEvent } from './types.js';

export type * from './types.js';
export { loadMessages, loadRecipeData, parseRecipeData, renderMessage, type CommandRecipe, type MessageTable, type RecipeData, type RecipeLimits } from './data.js';
export type RecipeResult = { status: ResultStatus, detail: string };
export type RunnableRecipe = Recipe & { run(context: { signal: AbortSignal, requestId: string, jobId: string }): Promise<RecipeResult> };
export type Outcome = { request_id: string, job_id: string, status: string, detail: string, cancel_requested?: boolean };
/** active_job is empty exactly when status is not running or cancel-requested. */
export type Session = { revision: number, active_request: string, active_job: string, status: SessionStatus, messages: string[], history: Outcome[] };
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

  start(requestId: string, recipeId: string): { id: string, request_id: string, status: 'running', detail: string } {
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

  cancel(jobId: string): { id: string, request_id: string, status: 'cancel-requested', detail: string } {
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
    this.events.push({ operation: 'terminal.complete', request_id: job.requestId, job_id: job.id, status: result.status,
      cancel_requested: job.cancelRequested });
    job.completion = { kind: 'complete', id: `completion-${job.id}`, request_id: job.requestId, job_id: job.id, text: '',
      status: result.status, detail: result.detail, cancel_requested: job.cancelRequested };
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
      job.completion.status === event.status && job.completion.detail === event.detail &&
      job.completion.cancel_requested === event.cancel_requested;
  }

  drainEvents(): Record<string, unknown>[] { return this.events.splice(0); }
}

/**
 * Reduce one user or job event into the session. The wording of the session's own messages comes from `messages`
 * (messages.json); the explanation of a completion is the only text a model writes for the user.
 */
export async function step(terminal: RecipeTerminal, session: Session, item: TerminalEvent, files?: FolderHandle,
    messages: MessageTable = loadMessages()): Promise<Session> {
  const say = (name: string, values: Record<string, string | number> = {}, changes: Partial<Session> = {}): Session =>
    ({ ...session, ...changes, messages: [...session.messages, renderMessage(messages, name, values)] });
  switch (item.kind) {
    case 'request': {
      if (!item.id || item.id === session.active_request || session.history.some(row => row.request_id === item.id))
        return say('duplicate-request');
      if (session.status === 'running' || session.status === 'cancel-requested')
        return say('request-busy', { request: item.id, active: session.active_request });
      const catalog = terminal.catalog();
      const note = files ? await builtin('readNote')(item.text, files) as Untrusted<string> : untrusted('', 'workspace file');
      const recipe = await chooseRecipe(item.text, catalog, note);
      if (!catalog.some(row => row.id === recipe))
        return say('no-recipe', { text: item.text }, { revision: session.revision + 1, status: 'unsupported' });
      const job = terminal.start(item.id, recipe);
      return say('started', { recipe, request: item.id }, { revision: session.revision + 1, active_request: item.id, active_job: job.id, status: job.status });
    }
    case 'complete': {
      if (!terminal.confirm(item) || item.status === '') return say('unverified-completion', { job: item.job_id });
      const explanation = await explain({ request_id: item.request_id, job_id: item.job_id, status: item.status,
        detail: untrusted(item.detail, 'command output'), cancel_requested: item.cancel_requested });
      const history = [...session.history, { request_id: item.request_id, job_id: item.job_id, status: item.status, detail: item.detail,
        cancel_requested: item.cancel_requested }];
      if (item.request_id !== session.active_request || item.job_id !== session.active_job)
        return say('stale-result', { job: item.job_id, explanation }, { history });
      return { ...session, messages: [...session.messages, explanation], history, revision: session.revision + 1,
        active_request: '', active_job: '', status: item.status };
    }
    case 'cancel':
      if (!session.active_job || item.request_id !== session.active_request) return say('no-active-job', { request: item.request_id });
      terminal.cancel(session.active_job);
      return say('cancel-requested', { job: session.active_job }, { revision: session.revision + 1, status: 'cancel-requested' });
    case 'recover':
      if (session.status !== 'running' && session.status !== 'cancel-requested') return session;
      return say('recovered', { job: session.active_job },
        { revision: session.revision + 1, active_request: '', active_job: '', status: 'unknown' });
    default:
      return say('unknown-event', { kind: String((item as TerminalEvent).kind) });
  }
}

function inside(root: string, path: string): boolean {
  const rel = relative(root, path);
  return rel === '' || (!rel.startsWith('..') && !isAbsolute(rel));
}

/** Exact argv recipes for a trusted local workspace. No shell text is evaluated. */
export class CommandRecipeLibrary {
  private readonly root: string;
  private readonly limits: RecipeLimits;
  private readonly definitions: (CommandRecipe & { cwd: string })[];

  constructor(root: string, definitions: CommandRecipe[], limits: Partial<RecipeLimits> = {}) {
    this.root = realpathSync(root);
    this.limits = { ...loadRecipeData().limits, ...limits };
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
        const bounded = value.subarray(Math.max(0, value.length - this.limits.outputBytes));
        if (field === 'stdout') stdout = bounded; else stderr = bounded;
      };
      child.stdout.on('data', chunk => capture('stdout', chunk));
      child.stderr.on('data', chunk => capture('stderr', chunk));
      const abort = () => {
        cancelled = true; child.kill('SIGTERM');
        setTimeout(() => { if (!settled) child.kill('SIGKILL'); }, this.limits.killDelayMs).unref();
      };
      const duration = Math.max(1, Math.min(definition.timeoutMs ?? this.limits.timeoutMs, this.limits.maxTimeoutMs));
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

/** The workspace's recipes: the catalog and limits of recipes.json (or the data given). */
export function natlangWorkspaceRecipes(root: string, data: RecipeData = loadRecipeData()): CommandRecipeLibrary {
  return new CommandRecipeLibrary(root, data.recipes, data.limits);
}
