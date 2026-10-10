/**
 * pi's foreground subagent tool (coding-agent experimental/durable/subagent.ts): each call runs a self-contained task
 * in a child conversation owned by the call's task and returns the child's answer. Aborting the call aborts the
 * child; the child outlives the call. The procedure is `subagent.nl`; the `delegation` service holds what runs on the
 * commit line (finding or creating the child in one commit) and the Harness handles it waits on.
 */
import type { Context } from '@earendil-works/chord';
import type { AssistantMessage } from '@earendil-works/pi-ai';
import { Type } from 'typebox';
import type { NatlangRuntime } from 'natlang:runtime';
import { AssistantEntry } from '../../vendor/durable/src/entries.ts';
import { configure } from '../../vendor/durable/src/harness/agent.ts';
import type { Extension, ToolExecutionApi, ToolExecutionResult } from '../../vendor/durable/src/harness/types.ts';
import type { ConversationId, EntryId } from '../../vendor/durable/src/types.ts';
import subagentTask from './subagent.nl';
import { toolError } from '../coding-tools/index.ts';

export function delegationService(api: ToolExecutionApi, context: Context, self: () => Extension) {
  return {
    /** This tool call's task ID. */
    taskId: (): number => api.taskId,
    /**
     * The child conversation of this call: the one this call's task already owns (a rerun after a crash finds it), or
     * a new one owned by the task, created in the same commit as a copy of this conversation's agent without the
     * subagent extension (so it cannot delegate further).
     */
    child: (): Promise<number> => api.commit(async tx => {
      const existing = (await tx.scanConversations({ ownerTaskId: api.taskId }, 1)).items[0];
      if (existing !== undefined) return existing.id as number;
      const created = await tx.createConversation({ ownership: { kind: 'task', taskId: api.taskId } });
      await configure(tx, created.id, { extensions: { remove: [self()] } });
      return created.id as number;
    }, context),
    /** Publish this call's running details (for UIs: which conversation runs the task). */
    details: (value: unknown): Promise<void> => api.details(JSON.parse(JSON.stringify(value)), context),
    /** Submit task as input to conversation child with requestId (a resubmission returns the same submission), wait until it settles, and return the settled record: status "done" with answer, or "unanswered" with reason. */
    ask: async (child: number, task: string, requestId: string) => {
      const handle = await api.conversation(child as ConversationId, context);
      if (!handle) throw new Error(`Conversation ${child} does not exist`);
      const settled = await (await handle.submit({ type: 'input', content: task, requestId }, context)).wait(context);
      return JSON.parse(JSON.stringify(settled)) as { type: string; status: string; answer?: number; reason?: string };
    },
    /** The text of answer entry `answer`: its assistant message's text items joined with no separator ("" when there is none). */
    answerText: async (answer: number): Promise<string> => {
      const entry = await api.commit(tx => tx.entry(AssistantEntry, answer as EntryId), context);
      const message = entry?.model?.[0] as AssistantMessage | undefined;
      return message?.content.flatMap(content => content.type === 'text' ? [content.text] : []).join('') ?? '';
    },
  };
}

export const DELEGATION_DECLARATION = `/** Child conversations of this subagent call. */
/** This tool call's task ID. */
export function taskId(): number;
/** The child conversation of this call: the one it already owns, or a new one (a copy of this agent that cannot delegate further). */
export function child(): Promise<number>;
/** Publish this call's running details for UIs. */
export function details(value: unknown): Promise<void>;
/** Submit task as input to conversation child with requestId (resubmitting returns the same submission), wait until it settles, and return it: status "done" with answer (an entry ID), or "unanswered" with reason. */
export function ask(child: number, task: string, requestId: string): Promise<{ type: string; status: string; answer?: number; reason?: string }>;
/** The text of answer entry answer. */
export function answerText(answer: number): Promise<string>;`;

/** The subagent extension; its function runs on `natlang`. */
export function subagent(natlang: NatlangRuntime): Extension {
  const extension: Extension = {
    name: 'subagent',
    tools: [{
      name: 'subagent',
      description: 'Delegate a self-contained task to a subagent with the same tools and get its answer back. Give it everything it needs to know; it does not see this conversation.',
      parameters: Type.Object({ task: Type.String({ description: 'What the subagent should do' }) }),
      // A rerun after a crash finds the child it created and the submission it made.
      replay: 'safe',
      execute: async (args, api, context): Promise<ToolExecutionResult> => {
        const signal = context.abortSignal;
        try {
          return await natlang.run(() => subagentTask(args as never), {
            services: { delegation: delegationService(api, context, () => extension) },
            serviceDeclarations: { delegation: DELEGATION_DECLARATION }, ...(signal ? { signal } : {}), name: `tool:subagent#${api.callId}` }) as ToolExecutionResult;
        } catch (error) { throw toolError(error, signal); }
      },
    }],
  } as Extension;
  return extension;
}
