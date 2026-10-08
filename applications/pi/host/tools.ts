/**
 * The `tools` service, bound to one tool task: pi's tool execution machinery. It resolves the tool in the phase's
 * agent, applies its argument repair, and runs it with pi's execution api (identity, environment, output buffer,
 * throttled progress into the call's `pi.live.tools` slot, details, durable operations). The policy around it — the
 * order of checks and hooks, intent, the error-to-result rules, assembly, bounding and settlement — is `tool/call` and
 * `tool/run`.
 */
import { ONCE_EFFECTS } from '@natlang/node';
import { type Context, copyJson } from '@earendil-works/chord';
import { awaitWithContext } from '@earendil-works/chord/context';
import type { ToolCall } from '@earendil-works/pi-ai';
import type { Task, TaskId, TaskOptions, TaskRuntime } from '../vendor/durable/src/types.ts';
import type { Agent, ToolDiagnostic, ToolExecutionApi, ToolExecutionResult, ToolRegistration } from '../vendor/durable/src/harness/types.ts';
import { OutputBuffer, PROGRESS_BYTES_PER_SECOND } from '../vendor/durable/src/harness/output.ts';
import { publishProgress, type Reported } from '../vendor/durable/src/harness/tool.ts';
import type { OutputLimits } from '../types.ts';
import { plain } from './durable.ts';

type Runtime = TaskRuntime<unknown, unknown, unknown, Record<string, unknown>>;

/** What one execution produced, for `tool/run`. */
export type Execution = {
  /** What the tool returned; absent when it threw. */
  result?: ToolExecutionResult;
  /** The thrown error's text. */
  error?: string;
  /** The output the call kept, within its limits: text, and what was dropped. */
  retained: { text: string; droppedBytes: number; droppedLines: number };
  /** Diagnostics the tool reported while running, in order. */
  diagnostics: ToolDiagnostic[];
  /** The last details the tool reported while running. */
  details?: unknown;
  /** Rounded execution time; absent when building the environment threw. */
  durationMs?: number;
};

export function toolsService(runtime: Runtime, context: Context, agent: Agent, callId: string) {
  const find = (name: string): ToolRegistration => {
    const tool = agent.tools.find(item => item.name === name);
    if (!tool) throw new Error(`the agent has no tool ${name}`);
    return tool;
  };
  return {
    // Running the tool is the call's external effect: an executor that runs it again within the phase (a retried eval)
    // gets the earlier execution, not a second run (pi runs a tool once per execute phase).
    [ONCE_EFFECTS]: ['execute'],
    /** The tool's argument repair, if it has one: the repaired arguments, or the error its repair threw. */
    prepare(name: string, args: Record<string, unknown>): { args: Record<string, unknown> } | { error: string } {
      const tool = find(name);
      if (!tool.prepareArguments) return { args };
      try { return { args: plain(tool.prepareArguments(args)) as Record<string, unknown> }; }
      catch (error) { return { error: error instanceof Error ? error.message : String(error) }; }
    },

    async execute(name: string, args: Record<string, unknown>, limits: OutputLimits): Promise<Execution> {
      const tool = find(name);
      const call = { type: 'toolCall', id: callId, name, arguments: args } as ToolCall;
      const reported: Reported = { output: new OutputBuffer(limits), limits, diagnostics: [], details: undefined };
      const progress = publishProgress(runtime as never, reported, context);
      let ended = false;
      const assertLive = (): void => { if (ended) throw new Error(`Tool call ${call.id} has settled`); };
      const api: Omit<ToolExecutionApi, 'env'> = {
        taskId: runtime.taskId,
        conversationId: runtime.conversationId,
        callId,
        registry: runtime.registry,
        agent: runtime.agent,
        models: runtime.models,
        output: (chunk, skipped) => { assertLive(); if (reported.output.push(chunk, skipped)) progress.mark(); },
        outputWindow: limits.retain === 'tail' ? { maxBytes: limits.maxBytes, maxLines: limits.maxLines,
          minIntervalMs: runtime.settings.progress.outputIntervalMs, bytesPerSecond: PROGRESS_BYTES_PER_SECOND } : undefined,
        diagnostic: diagnostic => {
          assertLive();
          reported.diagnostics.push(copyJson(diagnostic, { omitUndefinedProperties: true }) as ToolDiagnostic);
          progress.mark();
        },
        details: async (value, detailsContext) => {
          assertLive();
          detailsContext.abortSignal?.throwIfAborted();
          reported.details = copyJson(value, { omitUndefinedProperties: true });
          const committed = progress.markAndWait();
          committed.catch(() => {});
          return awaitWithContext(committed, detailsContext);
        },
        commit: async (change, commitContext) => {
          let result: Awaited<ReturnType<typeof change>> | undefined;
          await runtime.commit(async tx => { result = await change(tx); return undefined; }, commitContext);
          return result as Awaited<ReturnType<typeof change>>;
        },
        memo: runtime.memo,
        createTask: async <I, S extends { phase: string }, R, H extends object>(task: Task<I, S, R, H>, input: I,
          options: Omit<TaskOptions, 'conversationId'>, taskContext: Context): Promise<TaskId<R>> => {
          let id: TaskId<R> | undefined;
          await runtime.commit(async tx => { id = await tx.createTask(task, input, options); return undefined; }, taskContext);
          return id!;
        },
        getTask: runtime.getTask,
        waitForTask: runtime.waitForTask,
        conversation: runtime.conversation,
        snapshot: runtime.snapshot,
        snapshotAsOf: runtime.snapshotAsOf,
        watchDoc: runtime.watchDoc,
      };
      let result: ToolExecutionResult | undefined, error: string | undefined, durationMs: number | undefined;
      try {
        const env = await runtime.env(context);
        const startedAt = performance.now();
        try { result = await tool.execute(args as never, { ...api, env }, context); }
        finally { durationMs = Math.round(performance.now() - startedAt); }
      } catch (thrown) {
        if (runtime.signal.aborted) {
          ended = true;
          for (const waiter of await progress.stop()) waiter.reject(thrown);
          throw thrown;
        }
        error = thrown instanceof Error ? thrown.message : String(thrown);
      }
      ended = true;
      reported.output.end();
      // Details waiting for a progress commit resolve now; the settling commit follows from tool/run.
      for (const waiter of await progress.stop()) waiter.resolve();
      const snapshot = reported.output.snapshot();
      return plain({ ...(result ? { result } : {}), ...(error !== undefined ? { error } : {}),
        retained: { text: snapshot.text, droppedBytes: snapshot.droppedBytes, droppedLines: snapshot.droppedLines },
        diagnostics: reported.diagnostics, ...(reported.details !== undefined ? { details: reported.details } : {}),
        ...(durationMs !== undefined ? { durationMs } : {}) });
    },
  };
}

export type ToolsService = ReturnType<typeof toolsService>;

export const TOOLS_DECLARATION = `/** The tool machinery of this tool call: the agent's tool implementations, run with pi's execution api. */
/** The tool's own argument repair, if it has one: the repaired arguments, or the error its repair threw (unchanged args when it has none). */
export function prepare(name: string, args: Record<string, unknown>): { args: Record<string, unknown> } | { error: string };
/**
 * Run tool name with args, keeping its running output within limits. Returns what it produced: result (absent when it
 * threw), error (the thrown error's text), retained (the kept output text and the bytes and lines dropped), the
 * diagnostics and last details it reported while running, and durationMs (absent when its environment failed).
 * Rejects only when the call was aborted.
 */
export function execute(name: string, args: Record<string, unknown>, limits: OutputLimits): Promise<{
  result?: ToolExecutionResult; error?: string; retained: { text: string; droppedBytes: number; droppedLines: number };
  diagnostics: ToolDiagnostic[]; details?: unknown; durationMs?: number }>;`;
