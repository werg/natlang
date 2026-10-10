/**
 * The companion on the agent's live stream (plans/STREAMING.md §2). While the agent's reply streams, pi-durable hands
 * each provider event to the generation's `onStream` observers (vendor/durable generation.ts streamResponse); the
 * companion reads them as hints:
 *
 * - reasoning and text: what they name (workspace files, code symbols) is looked up early, by the hint policy
 *   (pluggable: `crisp` reads paths and quoted names, `nl` is `hints.nl`, `shadow` runs both). A named file is read and,
 *   when the companion does not know its version, summarized (`summarize.nl`, as the background task does);
 *   a named symbol's definitions are found with grep;
 * - a tool call, as soon as its arguments are complete (they parse as a whole JSON object, or the call ended): its view
 *   intent is captured (host/views.ts `viewIntent`), the file a `read` names is learned as above, and the file an `edit`
 *   or `write` names is read once (a warm, read-only read).
 *
 * Everything is speculative until the turn's terminal message (`afterResponse`, `settle`): work counts only when that
 * message still holds what started it (the text the hint came from, or the call with the same ID, name and arguments).
 * Confirmed results are committed (file knowledge to `pi.companion.files`, research notes to the conversation's
 * `pi.companion` document, which the next request's companion section shows); unconfirmed work is aborted and dropped.
 * A re-sent request (its parts start again at a content index already seen), a failed or aborted attempt, and a new
 * attempt discard the turn's speculation at once.
 *
 * Cost: the observer is synchronous and returns at once; the crisp policy is a regular expression over new text; at
 * most `MAX_RUNNING` jobs run at a time for the whole companion and at most `MAX_JOBS` per turn, so nothing named means
 * nothing done, and the agent never waits for any of it.
 */
import type { Context } from '@earendil-works/chord';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { AssistantMessage, AssistantMessageEvent, ToolCall } from '@earendil-works/pi-ai';
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableMode, type PluggableSetting } from 'natlang:runtime';
import type { Harness } from '../../vendor/durable/src/harness/harness.ts';
import type { StreamHookApi } from '../../vendor/durable/src/harness/types.ts';
import type { ConversationId } from '../../vendor/durable/src/types.ts';
import type { ExecutionEnv } from '../../vendor/durable/src/env/index.ts';
import { resolvePath } from '../../host/paths.ts';
import type { FileKnowledge, FileSummary, StreamHints } from '../../types.ts';
import { viewIntent } from '../../host/views.ts';
import { CompanionDoc, CompanionFiles, companionService, definitionPattern, valueOf, workspacePath } from './workspace.ts';
import hints from './hints.nl';
import summarize from './summarize.nl';

/** Jobs running at once, for the whole companion. */
const MAX_RUNNING = 2;
/** Jobs one turn may start. */
const MAX_JOBS = 8;
/** Files one turn may summarize (a model call each). */
const MAX_SUMMARIES = 3;
/** Names a scan of new text may return, per kind. */
const MAX_NAMES = 3;
/** Characters of new text before a scan: the crisp policy reads small pieces, the natural-language one larger ones. */
const CHUNK = { crisp: 80, nl: 1_500 } as const;
/** Lines of grep output a symbol lookup reads. */
const SYMBOL_LINES = 3;

/** What the companion's stream helpers did, for logs and tests. */
export type SpeculationEvent =
  | { type: 'job'; conversationId: ConversationId; job: string; source: 'text' | 'call' }
  | { type: 'start'; conversationId: ConversationId; job: string }
  | { type: 'prepare'; conversationId: ConversationId; callId: string; name: string; intent: string }
  | { type: 'discard'; conversationId: ConversationId; reason: 'reset' | 'failed' | 'aborted' | 'superseded' | 'unconfirmed'; jobs: string[] }
  | { type: 'commit'; conversationId: ConversationId; job: string; notes: string[]; files: string[] };

/** What started a job: a piece of streamed reasoning or text, or a complete tool call (arguments as canonical JSON). */
type Source = { kind: 'text'; text: string } | { kind: 'call'; id: string; name: string; arguments: string };
/** What a finished job found: notes for the next request, and file knowledge to keep. */
type Outcome = { notes: string[]; knowledge: FileKnowledge[] };
type Job = {
  key: string;
  sources: Source[];
  run(signal: AbortSignal): Promise<Outcome>;
  state: 'queued' | 'running' | 'done' | 'dropped';
  outcome?: Outcome;
  controller: AbortController;
};

/** Canonical JSON: object keys sorted, so equal arguments compare equal whatever their order. */
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(',')}]`;
  if (value && typeof value === 'object') return `{${Object.keys(value).sort()
    .map(key => `${JSON.stringify(key)}:${canonical((value as Record<string, unknown>)[key])}`).join(',')}}`;
  return JSON.stringify(value) ?? 'null';
}

const PATH = /(?<![\w./:@-])((?:\.{1,2}\/)?(?:[\w@-][\w.@-]*\/)*[\w@-][\w.@-]*\.[A-Za-z][A-Za-z0-9]{0,7})(?![\w/-])/g;
const QUOTED = /`([A-Za-z_$][\w$]*(?:(?:\.|::|#)[A-Za-z_$][\w$]*)*)(?:\(\))?`/g;
const CALLED = /(?<![\w$.`])([A-Za-z_$][\w$]{2,})\(\)/g;

/**
 * The crisp hint policy: the paths `text` writes (a token with a directory or a file extension of letters, not part of
 * a URL) and the symbols it quotes (`name`, `a.name`, `name()`) or writes as calls (name()), at most MAX_NAMES each.
 */
export function hintsOf(text: string): StreamHints {
  const files: string[] = [], symbols: string[] = [];
  for (const [, path] of text.matchAll(PATH)) {
    // "e.g", "i.e" and the like: a bare name needs two characters before or after its last dot.
    const name = path!.split('/').pop()!, dot = name.lastIndexOf('.');
    if (!path!.includes('/') && dot < 2 && name.length - dot - 1 < 2) continue;
    if (!files.includes(path!) && files.length < MAX_NAMES) files.push(path!);
  }
  for (const pattern of [QUOTED, CALLED]) for (const [, quoted] of text.matchAll(pattern)) {
    const name = quoted!.split(/\.|::|#/).pop()!;
    if (files.some(file => file.endsWith(quoted!)) || symbols.includes(name) || symbols.length >= MAX_NAMES) continue;
    symbols.push(name);
  }
  return { files, symbols };
}

/** The streamed parts of one turn's attempt and the work they started. */
class Turn {
  readonly controller = new AbortController();
  /** Content index → the part's kind and text so far. */
  readonly parts = new Map<number, { kind: string; text: string; scanned: number }>();
  /** The highest content index started: a start at or below it means the request was sent again. */
  started = -1;
  readonly jobs = new Map<string, Job>();
  /** Call ID → the call as it completed and its intent. */
  readonly intents = new Map<string, { name: string; arguments: string; intent: string }>();
  /** Content index → a tool call's argument text, while it streams. */
  readonly argumentText = new Map<number, string>();
  readonly prepared = new Set<number>();
  summaries = 0;
  /** The conversation's environment and the agent's working directory, resolved at the turn's first job. */
  workspace?: Promise<{ env: ExecutionEnv | undefined; cwd: string }>;
  /** Set by `settle`: the confirmed jobs, committed as they finish. */
  settled = false;
  readonly conversationId: ConversationId;
  readonly taskId: number;
  readonly attempt: number;
  readonly api: StreamHookApi;
  readonly context: Context;
  constructor(conversationId: ConversationId, taskId: number, attempt: number, api: StreamHookApi, context: Context) {
    this.conversationId = conversationId;
    this.taskId = taskId;
    this.attempt = attempt;
    this.api = api;
    this.context = context;
  }
}

export type CompanionStreamOptions = {
  natlang: NatlangRuntime;
  harness(): Harness;
  onReport?(error: unknown): void;
  onSpeculation?(event: SpeculationEvent): void;
  /** The hint policy: `crisp` (default), `nl` (`hints.nl`) or `shadow` (both, the natural-language hints used). */
  hints?: PluggableSetting;
};

/** The companion's stream helpers: one speculative turn per conversation, bounded work, committed on confirmation. */
export class CompanionStream {
  readonly #turns = new Map<ConversationId, Turn>();
  /** Per conversation, the generation task of the latest settled reply: whose research the next request shows. */
  readonly #latest = new Map<ConversationId, number>();
  readonly #queue: { turn: Turn; job: Job }[] = [];
  #running = 0;
  readonly #mode: PluggableMode;
  readonly #options: CompanionStreamOptions;

  constructor(options: CompanionStreamOptions) {
    this.#options = options;
    this.#mode = pluggableMode(options.hints, 'crisp');
  }

  #emit(event: SpeculationEvent): void {
    try { this.#options.onSpeculation?.(event); } catch (error) { this.#options.onReport?.(error); }
  }

  /** The generation's `onStream` hook: read one provider event. Synchronous; work it starts runs in the background. */
  observe(event: AssistantMessageEvent | { type: 'neuralese'; contentIndex: number }, attempt: number, api: StreamHookApi, context: Context): void {
    let turn = this.#turns.get(api.conversationId);
    if (event.type === 'start' || !turn || turn.taskId !== api.taskId || turn.attempt !== attempt) {
      if (turn && !turn.settled) this.#discard(turn, 'superseded');
      turn = new Turn(api.conversationId, api.taskId, attempt, api, context);
      this.#turns.set(api.conversationId, turn);
      const current = turn;
      if (api.signal.aborted) { this.#discard(current, 'aborted'); return; }
      api.signal.addEventListener('abort', () => { if (!current.settled) this.#discard(current, 'aborted'); }, { once: true });
    }
    if (turn.settled || turn.controller.signal.aborted) return;
    switch (event.type) {
      case 'error': this.#discard(turn, 'failed'); return;
      case 'done': if (event.message.stopReason === 'deferred') this.#discard(turn, 'aborted'); return;
      case 'text_start': case 'thinking_start': case 'toolcall_start': case 'neuralese': {
        if (event.contentIndex <= turn.started) {
          // The request was sent again (a reset) or the final turn replaced the streamed parts: start over.
          this.#discard(turn, 'reset');
          turn = new Turn(api.conversationId, api.taskId, attempt, api, context);
          this.#turns.set(api.conversationId, turn);
        }
        turn.started = event.contentIndex;
        if (event.type === 'text_start' || event.type === 'thinking_start')
          turn.parts.set(event.contentIndex, { kind: event.type.slice(0, -6), text: '', scanned: 0 });
        if (event.type === 'toolcall_start') turn.argumentText.set(event.contentIndex, '');
        return;
      }
      case 'text_delta': case 'thinking_delta': {
        const part = turn.parts.get(event.contentIndex);
        if (!part) return;
        part.text += event.delta;
        const fresh = part.text.slice(part.scanned);
        if (fresh.length < CHUNK[this.#mode === 'crisp' ? 'crisp' : 'nl']) return;
        // Up to the last whitespace: a name still streaming stays for the next scan.
        const cut = Math.max(fresh.lastIndexOf(' '), fresh.lastIndexOf('\n'));
        if (cut <= 0) return;
        this.#scan(turn, fresh.slice(0, cut));
        part.scanned += cut;
        return;
      }
      case 'text_end': case 'thinking_end': {
        const part = turn.parts.get(event.contentIndex);
        if (!part) return;
        const fresh = part.text.slice(part.scanned);
        part.scanned = part.text.length;
        if (fresh.trim()) this.#scan(turn, fresh);
        return;
      }
      case 'toolcall_delta': {
        const text = (turn.argumentText.get(event.contentIndex) ?? '') + event.delta;
        turn.argumentText.set(event.contentIndex, text);
        if (turn.prepared.has(event.contentIndex)) return;
        // Complete arguments are a whole JSON object: nothing can follow its closing brace.
        let args: unknown;
        try { args = JSON.parse(text); } catch { return; }
        const call = event.partial.content[event.contentIndex];
        if (call?.type !== 'toolCall' || !args || typeof args !== 'object' || Array.isArray(args)) return;
        this.#prepare(turn, event.contentIndex, { ...call, arguments: args as ToolCall['arguments'] }, event.partial);
        return;
      }
      case 'toolcall_end':
        if (!turn.prepared.has(event.contentIndex)) this.#prepare(turn, event.contentIndex, event.toolCall, event.partial);
        return;
      default: return;
    }
  }

  /**
   * The generation's terminal message (`afterResponse`): commit what it confirms, drop the rest. Returns the intents
   * captured for the message's calls (call ID → intent), each equal to what the message itself gives.
   */
  async settle(message: AssistantMessage, conversationId: ConversationId, taskId: number, context: Context): Promise<Map<string, string>> {
    const turn = this.#turns.get(conversationId);
    const intents = new Map<string, string>();
    this.#latest.set(conversationId, taskId);
    if (!turn || turn.taskId !== taskId || turn.settled || turn.controller.signal.aborted) {
      // Nothing streamed for this reply (or its speculation was discarded): the last reply's research is no longer news.
      await this.#commit({ conversationId, taskId }, [], true, context);
      return intents;
    }
    turn.settled = true;
    const written = message.content.flatMap(part => part.type === 'text' ? [part.text] : part.type === 'thinking' ? [part.thinking] : []).join('\n');
    const calls = new Map(message.content.filter((part): part is ToolCall => part.type === 'toolCall')
      .map(call => [call.id, { name: call.name, arguments: canonical(call.arguments) }]));
    const confirmed = (source: Source) => source.kind === 'text' ? written.includes(source.text) :
      calls.get(source.id)?.name === source.name && calls.get(source.id)?.arguments === source.arguments;
    for (const [id, captured] of turn.intents) {
      const call = calls.get(id);
      if (call?.name === captured.name && call.arguments === captured.arguments &&
          captured.intent === viewIntent(message, { name: call.name, arguments: JSON.parse(call.arguments) })) intents.set(id, captured.intent);
    }
    const dropped: string[] = [];
    const kept: Job[] = [];
    for (const job of turn.jobs.values()) {
      job.sources = job.sources.filter(confirmed);
      if (job.sources.length) { kept.push(job); continue; }
      if (job.state !== 'done') job.controller.abort(new Error('the reply did not keep what started this work'));
      job.state = 'dropped';
      dropped.push(job.key);
    }
    if (dropped.length) this.#emit({ type: 'discard', conversationId, reason: 'unconfirmed', jobs: dropped });
    // This turn's research replaces the last turn's; jobs still running add theirs when they finish.
    const done = kept.filter(job => job.state === 'done');
    await this.#commit(turn, done, true, context);
    if (kept.every(job => job.state === 'done') && this.#turns.get(conversationId) === turn) this.#turns.delete(conversationId);
    return intents;
  }

  /** Commit finished, confirmed jobs: their knowledge, and their notes as this turn's research. */
  async #commit(turn: { conversationId: ConversationId; taskId: number }, jobs: readonly Job[], replace: boolean, context: Context): Promise<void> {
    // Notes of a reply that is no longer the conversation's latest are not shown with a later one's.
    const latest = this.#latest.get(turn.conversationId) === turn.taskId;
    const notes = latest ? jobs.flatMap(job => job.outcome?.notes ?? []) : [];
    const knowledge = jobs.flatMap(job => job.outcome?.knowledge ?? []);
    if (!notes.length && !knowledge.length && !replace) return;
    try {
      // Nothing to replace the last research with: clear it, without a commit when there is none.
      if (!notes.length && !knowledge.length &&
          !(await this.#options.harness().snapshot(CompanionDoc, turn.conversationId, context))?.research) return;
      const conversation = await this.#options.harness().conversation(turn.conversationId, context);
      await conversation?.commit(async tx => {
        if (knowledge.length) {
          const files = await tx.doc(CompanionFiles);
          for (const known of knowledge) files.files[known.path] = JSON.parse(JSON.stringify(known)) as never;
        }
        const doc = await tx.doc(CompanionDoc, turn.conversationId);
        if (replace) {
          if (notes.length) doc.research = { turn: turn.taskId, notes } as never; else delete doc.research;
        } else if (notes.length) {
          if (doc.research?.turn === turn.taskId) doc.research.notes.push(...notes);
          else doc.research = { turn: turn.taskId, notes } as never;
        }
      }, context);
    } catch (error) { this.#options.onReport?.(error); return; }
    for (const job of jobs) this.#emit({ type: 'commit', conversationId: turn.conversationId, job: job.key,
      notes: job.outcome?.notes ?? [], files: (job.outcome?.knowledge ?? []).map(known => known.path) });
  }

  #discard(turn: Turn, reason: Extract<SpeculationEvent, { type: 'discard' }>['reason']): void {
    if (turn.controller.signal.aborted) return;
    turn.controller.abort(new Error(`speculation discarded: ${reason}`));
    const jobs = [...turn.jobs.values()];
    for (const job of jobs) { if (job.state !== 'done') job.controller.abort(turn.controller.signal.reason); job.state = 'dropped'; }
    if (this.#turns.get(turn.conversationId) === turn) this.#turns.delete(turn.conversationId);
    this.#emit({ type: 'discard', conversationId: turn.conversationId, reason, jobs: jobs.map(job => job.key) });
  }

  /** Run the hint policy over new streamed text and queue a lookup for each name it returns. */
  #scan(turn: Turn, text: string): void {
    const source: Source = { kind: 'text', text };
    const queue = (found: StreamHints) => {
      if (turn.controller.signal.aborted) return;
      for (const path of (found.files ?? []).slice(0, MAX_NAMES)) this.#fileJob(turn, path, source, true);
      for (const symbol of (found.symbols ?? []).slice(0, MAX_NAMES)) this.#symbolJob(turn, symbol, source);
    };
    if (this.#mode === 'crisp') { queue(hintsOf(text)); return; }
    // The natural-language policy is itself a job: it waits for a slot like the lookups it starts.
    this.#enqueue(turn, `hints:${turn.jobs.size}`, source, async signal => {
      const known = Object.keys((await this.#options.harness().snapshot(CompanionFiles, BACKGROUND_CONTEXT))?.files ?? {}).slice(0, 200);
      const choose = pluggable({ crisp: (written: string) => hintsOf(written),
        nl: async (written: string) => await hints(written, known) as StreamHints }, this.#mode, { name: 'companion.hints' });
      queue(await this.#options.natlang.run(() => choose(text), { signal, name: `companion.hints#${turn.taskId}` }) as StreamHints);
      return { notes: [], knowledge: [] };
    });
  }

  /** Capture a complete call's intent and prepare what it will need. */
  #prepare(turn: Turn, index: number, call: ToolCall, partial: AssistantMessage): void {
    turn.prepared.add(index);
    const intent = viewIntent(partial, call);
    turn.intents.set(call.id, { name: call.name, arguments: canonical(call.arguments), intent });
    this.#emit({ type: 'prepare', conversationId: turn.conversationId, callId: call.id, name: call.name, intent });
    const path = (call.arguments as { path?: unknown }).path;
    if (typeof path !== 'string' || !path) return;
    const source: Source = { kind: 'call', id: call.id, name: call.name, arguments: canonical(call.arguments) };
    if (call.name === 'read') this.#fileJob(turn, path, source, false);
    else if (call.name === 'edit' || call.name === 'write') this.#warmJob(turn, path, source);
  }

  /**
   * The workspace as a job sees it: the conversation's environment, at the agent's working directory. Resolved once per
   * turn while the generation runs (its runtime and context end with its phase; confirmed jobs may finish later), then
   * read through the harness in the background context: a job's own signal is what cancels it.
   */
  async #workspace(turn: Turn) {
    const { env, cwd } = await turn.workspace!;
    return { env, cwd, service: companionService(this.#options.harness(), BACKGROUND_CONTEXT, env, cwd) };
  }

  /**
   * Learn a file: what the companion knows about its version, else its summary (at most MAX_SUMMARIES per turn). A
   * file the agent's text names gives a note; one a read call names only knowledge (the agent reads it whole), and the
   * note that a named file does not exist is a warning worth having before the agent relies on it.
   */
  #fileJob(turn: Turn, path: string, source: Source, noted: boolean): void {
    const run = async (signal: AbortSignal): Promise<Outcome> => {
      const learned: FileKnowledge[] = [];
      const { cwd, service } = await this.#workspace(turn);
      const local = workspacePath(cwd, path);
      if (local === undefined) return { notes: [], knowledge: [] };
      const file = await service.file(local);
      if (!file) return { notes: noted ? [`${local} does not exist in the workspace.`] : [], knowledge: [] };
      let purpose = file.known?.purpose;
      if (!file.known && turn.summaries < MAX_SUMMARIES) {
        turn.summaries++;
        const summary = await this.#options.natlang.run(() => summarize(local, file.text), { signal,
          name: `companion.summarize#${local}` }) as FileSummary;
        signal.throwIfAborted();
        // Held, not committed: the reply's terminal message decides (`settle`).
        learned.push(JSON.parse(JSON.stringify({ path: local, hash: file.hash, purpose: summary.purpose, symbols: summary.symbols,
          notes: summary.notes })) as FileKnowledge);
        purpose = summary.purpose;
      }
      return { notes: noted && purpose ? [`${local}: ${purpose}`] : [], knowledge: learned };
    };
    this.#enqueue(turn, `file:${path}`, source, run);
  }

  /** Read a file an edit or write will change, once, so the environment has it at hand. Read-only; keeps nothing. */
  #warmJob(turn: Turn, path: string, source: Source): void {
    this.#enqueue(turn, `warm:${path}`, source, async () => {
      const { env, cwd } = await this.#workspace(turn);
      const local = workspacePath(cwd, path);
      if (!env || local === undefined) return { notes: [], knowledge: [] };
      const absolute = resolvePath(cwd, local);
      const info = await env.fileInfo(absolute, BACKGROUND_CONTEXT);
      if (info.ok && info.value.kind === 'file') valueOf(await env.readTextFile(absolute, BACKGROUND_CONTEXT));
      return { notes: [], knowledge: [] };
    });
  }

  /** Find where a symbol is defined: grep for a definition keyword followed by the name. */
  #symbolJob(turn: Turn, symbol: string, source: Source): void {
    if (!/^[A-Za-z_$][\w$]*$/.test(symbol)) return;
    this.#enqueue(turn, `symbol:${symbol}`, source, async () => {
      const { service } = await this.#workspace(turn);
      const lines = await service.search(definitionPattern(symbol));
      return { notes: lines.slice(0, SYMBOL_LINES).map(line => `\`${symbol}\` is defined at ${line}`), knowledge: [] };
    });
  }

  /** Queue a job of `turn` (one per key; a later source of the same work is added to it), within the turn's cap. */
  #enqueue(turn: Turn, key: string, source: Source, run: (signal: AbortSignal) => Promise<Outcome>): void {
    if (turn.controller.signal.aborted) return;
    const existing = turn.jobs.get(key);
    if (existing) { if (!existing.sources.some(known => canonical(known) === canonical(source))) existing.sources.push(source); return; }
    if (turn.jobs.size >= MAX_JOBS) return;
    if (!turn.workspace) {
      const api = turn.api;
      turn.workspace = (async () => {
        const env = await api.env(turn.context);
        return { env, cwd: (await api.agent(turn.context)).cwd ?? env?.cwd ?? '/' };
      })();
      turn.workspace.catch(() => {});
    }
    const job: Job = { key, sources: [source], run, state: 'queued', controller: new AbortController() };
    turn.jobs.set(key, job);
    this.#emit({ type: 'job', conversationId: turn.conversationId, job: key, source: source.kind });
    this.#queue.push({ turn, job });
    this.#pump();
  }

  /** Start queued jobs while slots are free; a dropped job is skipped. */
  #pump(): void {
    while (this.#running < MAX_RUNNING && this.#queue.length) {
      const { turn, job } = this.#queue.shift()!;
      if (job.state !== 'queued') continue;
      job.state = 'running';
      this.#running++;
      this.#emit({ type: 'start', conversationId: turn.conversationId, job: job.key });
      void (async () => {
        try {
          const outcome = await job.run(job.controller.signal);
          job.controller.signal.throwIfAborted();
          job.outcome = outcome;
          job.state = 'done';
          // Confirmed already: commit now. Before `settle`, the outcome waits for it.
          if (turn.settled) {
            await this.#commit(turn, [job], false, BACKGROUND_CONTEXT);
            if ([...turn.jobs.values()].every(other => other.state === 'done' || other.state === 'dropped') &&
                this.#turns.get(turn.conversationId) === turn) this.#turns.delete(turn.conversationId);
          }
        } catch (error) {
          if (!job.controller.signal.aborted) this.#options.onReport?.(error);
          if (job.state !== 'done') job.state = 'dropped';
        } finally {
          this.#running--;
          this.#pump();
        }
      })();
    }
  }
}
