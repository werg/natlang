/**
 * The `durable` service: one task invocation's reads of committed state, its commit of write operations, admission,
 * hooks, section rendering, memos and the durable clock. Bound to pi-durable's `TaskRuntime` for the invocation, so
 * every operation rejects once the invocation has ended.
 */
import type { Context } from '@earendil-works/chord';
import type { Message } from '@earendil-works/pi-ai';
import { ONCE_EFFECTS } from '@natlang/node';
import { getCurrentTools } from '@earendil-works/pi-ai/utils/transcript';
import type { ConversationId, EntryId, TaskId, TaskRuntime } from '../vendor/durable/src/types.ts';
import type { Agent as PiAgent, ContextView as PiContextView, SubmissionDraft as PiSubmissionDraft } from '../vendor/durable/src/harness/types.ts';
import { InboxDoc } from '../vendor/durable/src/harness/inbox.ts';
import { LiveDoc } from '../vendor/durable/src/harness/live.ts';
import { planSystemEntries, replaySections } from '../vendor/durable/src/harness/prompt.ts';
import { estimateContext } from '../vendor/durable/src/harness/compaction.ts';
import { applyOps, type ApplyScope } from '../ops.ts';
import type { CommitResult, ContextView, EntryRecord, Expect, InboxItem, LiveState, Op, SubmissionDraft, SubmissionRecord, TaskOutcome, TaskRecord } from '../types.ts';

/** What a host hands the durable service besides the runtime: admission, and which implementations are selected. */
export type DurableHost = {
  /** Admit a submission to `conversationId` (the selected admission: crisp or the natural-language `admit`). */
  submit(conversationId: ConversationId, draft: SubmissionDraft, context: Context): Promise<number>;
  submission(id: number, context: Context): Promise<SubmissionRecord | undefined>;
  /** The raw active range through `at` (default: the newest entry): the newest head marker and every entry from its head on. */
  scan(conversationId: ConversationId, at: number | undefined, context: Context): Promise<{ head: EntryRecord | null; entries: EntryRecord[] }>;
  /** "crisp" or "natural-language", per pluggable point. */
  implementation(point: 'context' | 'scheduler' | 'admission' | 'planning'): 'crisp' | 'natural-language';
};

type Runtime = TaskRuntime<unknown, unknown, unknown, Record<string, unknown>>;

/** Strict JSON copy: what a natural-language call receives, never a live draft. */
export const plain = <T>(value: T): T => value === undefined ? value : JSON.parse(JSON.stringify(value)) as T;

/** pi-durable's ContextView plus the prompt state it has shown: the sections and the offered tools, replayed. */
export function contextView(view: PiContextView): ContextView {
  const messages = view.messages as Message[];
  return plain({
    head: view.head ?? null,
    entries: view.entries,
    contributions: view.contributions,
    messages,
    sections: [...replaySections(messages)].map(([key, text]) => ({ key, text })),
    tools: getCurrentTools(messages),
  }) as unknown as ContextView;
}

/** The port's view as pi-durable reads it: no head is `undefined` there, not `null`. */
const piView = (view: ContextView): never => ({ ...view, head: view.head ?? undefined }) as never;

/** What one phase invocation shares between its services: a failure after which it may commit nothing more. */
export type PhaseState = { failed?: string };

export function durableService(runtime: Runtime, context: Context, host: DurableHost, agent: PiAgent, phase: PhaseState = {}) {
  const conversationId = runtime.conversationId;
  const scope: ApplyScope = { taskId: runtime.taskId, conversationId, now: () => runtime.now() };
  return {
    // A hook handler runs once per phase, as in pi: an executor that calls it again (a re-run eval) gets the earlier
    // outcome instead of running the extension's code twice.
    [ONCE_EFFECTS]: ['hook'],
    /** The model context of this conversation through entry `at` (default: the newest entry): pi-durable's derivation. */
    async view(at?: number): Promise<ContextView> {
      return contextView(await runtime.context(conversationId, context, at === undefined ? {} : { at: at as EntryId }));
    },
    async scan(at?: number) { return plain(await host.scan(conversationId, at, context)); },
    /** pi-durable's planSystemEntries, as the port's plan ({ message, edits? }); the crisp side of harness/planSystem. */
    planSystem(view: ContextView, desired: { key: string; text: string }[], tools: unknown[], now: number) {
      const planned = planSystemEntries(piView(view), new Map(desired.map(section => [section.key, section.text])), tools as never, now);
      return plain(planned.map(draft => ({ message: draft.model![0], ...(draft.edits?.length ? { edits: draft.edits } : {}) })));
    },
    /** pi-durable's estimateContext; the crisp side of harness/estimate. */
    estimate(view: ContextView, extra: Message[]): number { return estimateContext(piView(view), extra as never); },
    implementation(point: 'context' | 'scheduler' | 'admission' | 'planning') { return host.implementation(point); },
    async entry(id: number): Promise<EntryRecord | null> { return plain(await runtime.entry(id as EntryId, context)) as EntryRecord ?? null; },
    async task(id: number): Promise<TaskRecord | null> { return plain(await runtime.getTask(id as TaskId, context)) as unknown as TaskRecord ?? null; },
    async outcomes(ids: number[]): Promise<TaskOutcome[]> {
      return plain((await runtime.outcomes(ids as TaskId[], context))) as unknown as TaskOutcome[];
    },
    async live(): Promise<LiveState> { return plain(await runtime.snapshot(LiveDoc, conversationId, context) ?? {}) as unknown as LiveState; },
    async inbox(): Promise<InboxItem[]> { return plain((await runtime.snapshot(InboxDoc, conversationId, context))?.items ?? []) as unknown as InboxItem[]; },
    async submission(id: number): Promise<SubmissionRecord | null> { return plain(await host.submission(id, context)) ?? null; },
    async commit(ops: Op[], expect?: Expect): Promise<CommitResult> {
      if (phase.failed) throw new Error(`This phase already failed (${phase.failed}); it commits nothing more. ` +
        'End this call with return_result status "failed" and that reason.');
      let result: CommitResult | undefined;
      await runtime.commit(async (tx, current) => {
        const applied = await applyOps(tx, scope, current as never, ops, expect);
        result = applied.result;
        return applied.next;
      }, context);
      return result!;
    },
    async submit(draft: SubmissionDraft): Promise<number> { return host.submit(conversationId, draft, context); },
    /** The extensions that handle hook `name` for this task kind, in order. */
    async hooks(name: string): Promise<string[]> {
      const names: string[] = [];
      await runtime.hooks.each(name, () => { names.push(name); });
      return names.map((_, index) => `${name}#${index}`);
    },
    /** Call handler `index` of hook `name` with `args`; a throw comes back as `error`. */
    async hook(name: string, index: number, args: unknown[]): Promise<{ value?: unknown, error?: string }> {
      let position = 0, outcome: { value?: unknown, error?: string } = { error: `hook ${name} has no handler ${index}` };
      await runtime.hooks.each(name, async handler => {
        if (position++ !== index) return;
        try { outcome = { value: plain(await (handler as (...a: unknown[]) => unknown)(...args, runtime, context)) }; }
        catch (error) {
          if (runtime.signal.aborted) throw error;
          outcome = { error: error instanceof Error ? error.message : String(error) };
        }
      });
      return outcome;
    },
    /**
     * Render section `key` of the agent with the shown sections: its text, `omit` when the section has nothing to show,
     * or `error` when its renderer threw (reported already).
     */
    async renderSection(key: string, shown: Record<string, string>): Promise<{ text?: string, omit?: true, error?: string }> {
      const section = agent.sections.find(item => item.key === key);
      if (!section) return { error: `the agent has no section ${key}` };
      try {
        const env = await runtime.env(context).catch(error => { if (runtime.signal.aborted) throw error; runtime.report(error); return undefined; });
        const text = await section.render({ conversationId, agent, env, shown, read: runtime }, context);
        return text === undefined ? { omit: true } : { text };
      } catch (error) {
        if (runtime.signal.aborted) throw error;
        runtime.report(error);
        return { error: error instanceof Error ? error.message : String(error) };
      }
    },
    report(error: unknown): void {
      // A caught error is reported as it is; text becomes an Error with that message.
      runtime.report(error instanceof Error ? error : new Error(String(error).replace(/^Error: /, '')));
    },
    async sleep(until: number): Promise<void> { await runtime.sleep(until, context); },
    now(): number { return runtime.now(); },
    async memo(name: string, candidate?: unknown): Promise<unknown> {
      return candidate === undefined ? plain(await runtime.memo(name, context)) ?? null : plain(await runtime.memo(name, candidate as never, context));
    },
  };
}

export type DurableService = ReturnType<typeof durableService>;

/** What the model reads of the durable service. */
export const DURABLE_DECLARATION = `/**
 * The task's durable state, bound to this task invocation. Types are those of types.ts. Every read is committed state.
 * Writes go through commit(), all or nothing.
 */
/** The model context of this conversation through entry at (default: the newest entry), as pi-durable derives it. */
export function view(at?: number): Promise<ContextView>;
/** pi-durable's system-entry plan (what harness/planSystem computes when planning is crisp). */
export function planSystem(view: ContextView, desired: { key: string, text: string }[], tools: AgentTool[], now: number): { message: SystemMessage, edits?: ContextEdit[] }[];
/** pi-durable's context estimate (what harness/estimate computes when planning is crisp). */
export function estimate(view: ContextView, extra: Message[]): number;
/** The raw active range through at: the newest head marker (null when none) and every entry from its head through at. */
export function scan(at?: number): Promise<{ head: EntryRecord | null; entries: EntryRecord[] }>;
/** Which implementation the host selected for a pluggable point. */
export function implementation(point: "context" | "scheduler" | "admission" | "planning"): "crisp" | "natural-language";
export function entry(id: number): Promise<EntryRecord | null>;
export function task(id: number): Promise<TaskRecord | null>;
/** Outcomes of terminal tasks, in the order given. */
export function outcomes(ids: number[]): Promise<TaskOutcome[]>;
/** This conversation's pi.live document. */
export function live(): Promise<LiveState>;
/** This conversation's queued submissions, in ID order. */
export function inbox(): Promise<InboxItem[]>;
export function submission(id: number): Promise<SubmissionRecord | null>;
/**
 * Apply ops atomically, in order (see Op), after checking expect (see Expect). Rejects with "state changed: …" when a
 * guard fails (nothing is written: read again and decide again), and when this task was aborted or the invocation
 * ended (stop: someone else owns the state now).
 */
export function commit(ops: Op[], expect?: Expect): Promise<CommitResult>;
/** Admit a submission to this conversation and return its ID. A requestId seen before returns the earlier ID. */
export function submit(draft: SubmissionDraft): Promise<number>;
/** The handlers of hook name for this task kind, in extension order; empty when there are none. */
export function hooks(name: string): Promise<string[]>;
/** Call handler index of hook name with args. A throw is returned as error, not reported: report it yourself where the rule says so. A handler runs once per phase: the same call again returns the same outcome. */
export function hook(name: string, index: number, args: unknown[]): Promise<{ value?: unknown; error?: string }>;
/** Render the agent's section key given the shown sections (key to text): text, omit, or error (already reported). */
export function renderSection(key: string, shown: Record<string, string>): Promise<{ text?: string; omit?: true; error?: string }>;
/** Report a failure the harness keeps going after: pass a caught error itself (catch (e) { durable.report(e) }), or a text. */
export function report(error: unknown): void;
/** Wait until the durable clock reaches until (milliseconds). */
export function sleep(until: number): Promise<void>;
/** The durable clock, in milliseconds. */
export function now(): number;
/** Read a memo of this task (null when absent), or store candidate unless one exists and return the winner. */
export function memo(name: string, candidate?: unknown): Promise<unknown>;`;
