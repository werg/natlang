/**
 * The Harness options that make pi-durable run the port: the natural-language task kinds substituted by name in the
 * registry, the services' host (admission, submission reads, the raw range scan), and the selected scheduler and
 * admission policies. Shared by `openPi` and the conformance shim, so both open the same port.
 */
import type { Context } from '@earendil-works/chord';
import type { NatlangRuntime } from '@natlang/node';
import type { Harness, HarnessOptions } from '../vendor/durable/src/index.ts';
import type { ConversationId, EntryId, EntryRecord as PiEntryRecord, SubmissionId } from '../vendor/durable/src/types.ts';
import { LiveDoc } from '../vendor/durable/src/harness/live.ts';
import type { EntryRecord, SubmissionDraft } from '../types.ts';
import { admissionPolicy, schedulerPolicy } from './policies.ts';
import { substituteTasks } from './registry.ts';
import { naturalLanguageTask, type Entry, type TaskHost } from './tasks.ts';

export type Implementation = 'crisp' | 'natural-language';
export type Implementations = { context: Implementation; scheduler: Implementation; admission: Implementation; planning: Implementation };

/** The natural-language entries of the three task kinds. */
export type Entries = { generation: Entry; tool: Entry; compaction: Entry };

export type PortOptions = {
  natlang: NatlangRuntime;
  entries: Entries;
  implementations?: Partial<Implementations>;
  attempts?: number;
  onPhase?: TaskHost['onPhase'];
  now?: () => number;
};

/**
 * `options` with the port's registry and policies, and a `bind` to call with the opened Harness (the services read
 * and admit through it).
 */
export function portOptions<T extends HarnessOptions>(options: T, port: PortOptions): { options: T; bind(harness: Harness): void } {
  let harness: Harness | undefined;
  const opened = () => { if (!harness) throw new Error('the Harness is not open yet'); return harness; };
  const implementations: Implementations = { context: 'crisp', scheduler: 'crisp', admission: 'crisp', planning: 'crisp', ...port.implementations };
  const conversation = async (id: ConversationId, context: Context) => {
    const found = await opened().conversation(id, context);
    if (!found) throw new Error(`Conversation ${id} does not exist`);
    return found;
  };
  const host: TaskHost = {
    natlang: port.natlang,
    attempts: port.attempts,
    onPhase: port.onPhase,
    implementation: point => implementations[point],
    async submit(conversationId: ConversationId, draft: SubmissionDraft, context: Context): Promise<number> {
      return (await (await conversation(conversationId, context)).submit(draft as never, context)).id;
    },
    async submission(id: number, context: Context) {
      const handle = await opened().submission(id as SubmissionId, context);
      return handle ? JSON.parse(JSON.stringify(await handle.status(context))) : undefined;
    },
    async scan(conversationId: ConversationId, at: number | undefined, context: Context) {
      const found = await conversation(conversationId, context);
      // The bounds as pi-durable captures them: the newest head marker at or before the tail.
      const view = await found.context(context, at === undefined ? {} : { at: at as EntryId });
      const tail = at ?? (view.entries.length ? Math.max(...view.entries.map(entry => entry.id)) : undefined);
      const entries: PiEntryRecord[] = [];
      if (tail !== undefined) {
        let cursor: Parameters<typeof found.entries>[2];
        for (let page = 0; page < 1_000_000; page++) {
          const result = await found.entries({ ...(view.head?.head !== undefined ? { minEntryId: view.head.head } : {}),
            maxEntryId: tail as EntryId, order: 'ascending' }, 256, cursor, context);
          entries.push(...result.items);
          if (!result.next) break;
          cursor = result.next;
        }
      }
      return JSON.parse(JSON.stringify({ head: view.head ?? null, entries })) as { head: EntryRecord | null; entries: EntryRecord[] };
    },
  };
  const tasks = [
    naturalLanguageTask(host, 'pi.generation', 1, () => ({ phase: 'prepare', attempt: 1 }), ['prepare', 'request', 'retry', 'poll', 'tools'], port.entries.generation),
    naturalLanguageTask(host, 'pi.tool', 1, () => ({ phase: 'call' }), ['call', 'execute'], port.entries.tool),
    naturalLanguageTask(host, 'pi.compaction', 1, () => ({ phase: 'select' }), ['select', 'summarize', 'retry'], port.entries.compaction),
  ];
  const policyHost = { natlang: port.natlang, now: port.now ?? options.now ?? Date.now,
    live: async (id: number, context: Context) => (await opened().snapshot(LiveDoc, id as ConversationId, context)) ?? {} };
  const scheduler = schedulerPolicy(policyHost as never, implementations.scheduler);
  const admission = admissionPolicy(policyHost as never, implementations.admission);
  return {
    options: { ...options, registry: substituteTasks(options.registry, tasks),
      ...(scheduler ? { schedulerPolicy: scheduler } : {}), ...(admission ? { admission } : {}) },
    bind(opened: Harness) { harness = opened; },
  };
}
