/**
 * pi on natlang: pi-durable's harness with its built-in task kinds (`pi.generation`, `pi.tool`, `pi.compaction`)
 * carried out by natural-language functions, and pi's coding agent as extensions.
 *
 * The host is pi-durable itself, vendored at f10993b (`vendor/durable`): the Session line and SQLite storage, the
 * scheduler's mechanics, the registry and the Harness API. The registry the Harness reads substitutes the
 * natural-language task kinds for pi's built-ins by name, so every task pi-durable creates (a run's generation, a
 * round's tool tasks, compactions) runs here. See PORT.md.
 */
import type { Context } from '@earendil-works/chord';
import { BACKGROUND_CONTEXT } from '@earendil-works/chord/context';
import type { Models } from '@earendil-works/pi-ai';
import type { NatlangRuntime } from '@natlang/node';
import { Harness, type HarnessOptions, type HarnessSettings } from './vendor/durable/src/index.ts';
import type { Extension, Registry } from './vendor/durable/src/harness/types.ts';
import type { ConversationId, EntryId, EntryRecord as PiEntryRecord, Storage, SubmissionId } from './vendor/durable/src/types.ts';
import { naturalLanguageTask, type Entry, type TaskHost } from './host/tasks.ts';
import { substituteTasks } from './host/registry.ts';
import { admissionPolicy, schedulerPolicy } from './host/policies.ts';
import { LiveDoc } from './vendor/durable/src/harness/live.ts';
import type { EntryRecord, SubmissionDraft } from './types.ts';
import generation from './generation.nl';
import tool from './tool.nl';
import compaction from './compaction.nl';

export type Implementation = 'crisp' | 'natural-language';

export type PiOptions = {
  storage: Storage;
  /** Runs the natural-language functions (the executor). */
  natlang: NatlangRuntime;
  /** pi-ai model access: the agent's provider models. */
  models: Models;
  registry: Registry;
  settings?: HarnessSettings;
  env?: HarnessOptions['env'];
  now?: () => number;
  onReport?: (error: unknown) => void;
  onPhase?: TaskHost['onPhase'];
  /** Pluggable hot paths; crisp by default. */
  implementations?: { context?: Implementation; scheduler?: Implementation; admission?: Implementation };
  /** Runs per phase before the task faults (default 2). */
  attempts?: number;
};

/** Open pi on `options.storage`: a Harness whose built-in task kinds are natural-language functions. */
export async function openPi(options: PiOptions, context: Context = BACKGROUND_CONTEXT): Promise<Harness> {
  let harness: Harness | undefined;
  const implementations = { context: 'crisp', scheduler: 'crisp', admission: 'crisp', ...options.implementations } as const;
  const host: TaskHost = {
    natlang: options.natlang,
    attempts: options.attempts,
    onPhase: options.onPhase,
    implementation: point => implementations[point],
    async submit(conversationId: ConversationId, draft: SubmissionDraft, submitContext: Context): Promise<number> {
      const conversation = await harness!.conversation(conversationId, submitContext);
      if (!conversation) throw new Error(`Conversation ${conversationId} does not exist`);
      return (await conversation.submit(draft as never, submitContext)).id;
    },
    async submission(id: number, submissionContext: Context) {
      const handle = await harness!.submission(id as SubmissionId, submissionContext);
      return handle ? JSON.parse(JSON.stringify(await handle.status(submissionContext))) : undefined;
    },
    async scan(conversationId: ConversationId, at: number | undefined, scanContext: Context) {
      const conversation = await harness!.conversation(conversationId, scanContext);
      if (!conversation) throw new Error(`Conversation ${conversationId} does not exist`);
      const view = await conversation.context(scanContext, at === undefined ? {} : { at: at as EntryId });
      const tail = at ?? view.entries.at(-1)?.id;
      const entries: PiEntryRecord[] = [];
      if (tail !== undefined) {
        let cursor: Parameters<typeof conversation.entries>[2];
        for (let page = 0; page < 1_000_000; page++) {
          const result = await conversation.entries({ ...(view.head?.head !== undefined ? { minEntryId: view.head.head } : {}),
            maxEntryId: tail as EntryId, order: 'ascending' }, 256, cursor, scanContext);
          entries.push(...result.items);
          if (!result.next) break;
          cursor = result.next;
        }
      }
      return JSON.parse(JSON.stringify({ head: view.head ?? null, entries })) as { head: EntryRecord | null; entries: EntryRecord[] };
    },
  };
  const tasks = [
    naturalLanguageTask(host, 'pi.generation', 1, () => ({ phase: 'prepare', attempt: 1 }), ['prepare', 'request', 'retry', 'poll', 'tools'], generation as Entry),
    naturalLanguageTask(host, 'pi.tool', 1, () => ({ phase: 'call' }), ['call', 'execute'], tool as Entry),
    naturalLanguageTask(host, 'pi.compaction', 1, () => ({ phase: 'select' }), ['select', 'summarize', 'retry'], compaction as Entry),
  ];
  const policyHost = { natlang: options.natlang, now: options.now ?? Date.now,
    live: async (id: number, liveContext: Context) => (await harness!.snapshot(LiveDoc, id as ConversationId, liveContext)) ?? {} };
  const scheduler = schedulerPolicy(policyHost as never, implementations.scheduler);
  const admission = admissionPolicy(policyHost as never, implementations.admission);
  harness = await Harness.open(options.storage, {
    models: options.models,
    registry: substituteTasks(options.registry, tasks),
    ...(options.settings ? { settings: options.settings } : {}),
    ...(options.env ? { env: options.env } : {}),
    ...(options.now ? { now: options.now } : {}),
    ...(options.onReport ? { onReport: options.onReport } : {}),
    ...(scheduler ? { schedulerPolicy: scheduler } : {}),
    ...(admission ? { admission } : {}),
  }, context);
  return harness;
}

export type { Extension, Harness };
export { substituteTasks };
