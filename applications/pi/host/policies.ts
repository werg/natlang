/**
 * The natural-language implementations of the pluggable scheduler policy and of admission (PORT.md "The scheduler",
 * owner decision 4), selected by a pluggable mode (`crisp`, `nl`, `shadow`). pi-durable's crisp code stays the default:
 * `schedulerPolicy` and `admissionPolicy` return undefined for `crisp`, so the vendored Harness keeps its own inline
 * policy and `admitSubmission`. In `shadow` each scheduler policy point runs `pluggable()` over pi's rules restated
 * over facts (`crispSchedulerPolicy`) and the natural-language call, serves the natural-language decision and records
 * whether the two agree. Admission has no shadow mode: each side commits its own decision.
 *
 * The vendored scheduler reads each policy point's facts on the Session line, calls the policy outside any commit, and
 * commits the decision guarded on the records it read (vendor/PATCHES.md). Here each policy point is one
 * natural-language call (`scheduler/*.nl`) with a narrow `scheduler` service; its cleanup operations are applied with
 * the port's write operations (ops.ts) in the scheduler's commit. Admission (`admit.nl`) gets an `admission` service
 * bound to the conversation: committed reads, and one guarded commit of admission operations.
 */
import type { Context } from '@earendil-works/chord';
import { pluggable, pluggableMode, type NatlangRuntime, type PluggableSetting } from 'natlang:runtime';
import { copyJson } from '@earendil-works/chord';
import { UserEntry } from '../vendor/durable/src/entries.ts';
import { ConversationBusy } from '../vendor/durable/src/errors.ts';
import { startRun } from '../vendor/durable/src/harness/generation.ts';
import { InboxDoc } from '../vendor/durable/src/harness/inbox.ts';
import { LiveDoc } from '../vendor/durable/src/harness/live.ts';
import { crispSchedulerPolicy } from '../vendor/durable/src/harness/policy.ts';
import type { AdmissionPolicy, AdmissionRequest, SchedulerPolicy } from '../vendor/durable/src/harness/types.ts';
import type { ConversationId, EntryDraft, SubmissionId, TaskId, Tx } from '../vendor/durable/src/types.ts';
import { applyOps } from '../ops.ts';
import type { AdmissionExpect, AdmissionOp, AdmissionState, AdmitResult, InboxItem, LiveState, Op, SubmissionDraft, SubmissionRecord } from '../types.ts';
import pass from '../scheduler/pass.nl';
import step from '../scheduler/step.nl';
import reconcile from '../scheduler/reconcile.nl';
import abortTask from '../scheduler/abortTask.nl';
import abortConversation from '../scheduler/abortConversation.nl';
import admit from '../admit.nl';
import { StateChanged } from '../ops.ts';

export type PolicyHost = {
  natlang: NatlangRuntime;
  /** A conversation's committed `pi.live` (the Harness's `snapshot(LiveDoc, id)`), for cleanup decisions. */
  live(conversationId: number, context: Context): Promise<LiveState>;
  now(): number;
  /** Calls per policy point before its failure propagates (default 3). */
  attempts?: number;
  onCall?: (event: { point: string; attempt: number; error?: string }) => void;
  /** Further services every policy call gets (task services replace the runtime's). */
  services?: Record<string, object>;
};

const plain = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

export const SCHEDULER_DECLARATION = `/** Committed reads for the scheduler's decisions. */
/** The conversation's pi.live document (empty when it has none). */
export function live(conversationId: number): Promise<LiveState>;`;

export const ADMISSION_DECLARATION = `/** Admission to one conversation: committed reads, and one guarded commit. Types are those of types.ts. */
/** The conversation's admission facts at one committed point. */
export function state(): Promise<AdmissionState>;
/** The submission of this conversation that carries requestId, or null. */
export function byRequest(requestId: string): Promise<SubmissionRecord | null>;
/**
 * Apply ops in one commit, in order (see AdmissionOp), after checking expect (see AdmissionExpect). Returns the ID of the
 * submission the commit created and the user IDs a boundary placed. Rejects with "state changed: …" when a guard fails;
 * nothing is written then: read again and decide again.
 */
export function commit(ops: AdmissionOp[], expect: AdmissionExpect): Promise<{ id: number; placed: number[] }>;`;

/**
 * A scheduler decision as shadow mode compares it: without its cleanup (pi-durable's crisp settlement has none) and
 * with the wording of its messages reduced to whether there is one.
 */
const decisionShape = (decision: unknown): string => JSON.stringify(decision, (key, value) =>
  key === 'cleanup' ? undefined : (key === 'message' || key === 'report') && typeof value === 'string' ? true : value);

/** The selected scheduler policy: undefined for `crisp` (pi-durable's inline policy), else the natural-language one, compared with pi's rules in `shadow`. */
export function schedulerPolicy(host: PolicyHost, setting: PluggableSetting): SchedulerPolicy | undefined {
  const mode = pluggableMode(setting, 'crisp');
  if (mode === 'crisp') return undefined;
  type Point = Exclude<keyof SchedulerPolicy, 'applyCleanup'>;
  const call = async <T>(point: Point, fn: (facts: never) => Promise<unknown>, facts: unknown, context: Context): Promise<T> => {
    const services = { ...host.services, scheduler: { live: (conversationId: number) => host.live(conversationId, context).then(plain) } };
    const decide = pluggable({ crisp: () => crispSchedulerPolicy[point](plain(facts) as never, context) as unknown,
      nl: () => fn(plain(facts) as never) }, mode, { name: `pi.scheduler.${point}`, same: (exact, judged) => decisionShape(exact) === decisionShape(judged) });
    const attempts = Math.max(1, host.attempts ?? 3);
    for (let attempt = 1; ; attempt++) {
      try {
        const decision = await host.natlang.run(decide, { services, serviceDeclarations: { scheduler: SCHEDULER_DECLARATION }, name: `scheduler/${point}` });
        host.onCall?.({ point, attempt });
        return decision as T;
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        host.onCall?.({ point, attempt, error: message });
        if (attempt >= attempts) throw error;
      }
    }
  };
  return {
    pass: (facts, context) => call('pass', pass as never, facts, context),
    step: (facts, context) => call('step', step as never, facts, context),
    reconcile: (facts, context) => call('reconcile', reconcile as never, facts, context),
    abortTask: (facts, context) => call('abortTask', abortTask as never, facts, context),
    abortConversation: (facts, context) => call('abortConversation', abortConversation as never, facts, context),
    async applyCleanup(tx, record, cleanup) {
      const ops = cleanup as Op[];
      if (!Array.isArray(ops) || ops.length === 0) return;
      await applyOps(tx as unknown as Tx, { taskId: record.id as TaskId, conversationId: record.conversationId, now: host.now },
        undefined as never, ops, undefined);
    },
  };
}

/** The admission service of one conversation (see ADMISSION_DECLARATION). */
export function admissionService(request: AdmissionRequest, context: Context) {
  const { conversationId, session } = request;
  return {
    async state(): Promise<AdmissionState> {
      const live = (await session.snapshot(LiveDoc, conversationId, context) ?? {}) as LiveState;
      const inbox = ((await session.snapshot(InboxDoc, conversationId, context)) as { items?: InboxItem[] } | undefined)?.items ?? [];
      // A table read in a commit that writes nothing: nothing is published.
      const head = await session.commit(async tx => (await tx.latestHeadMarker(conversationId))?.head, context);
      return plain({ run: live.run?.taskId ?? null, inbox, activeStart: head ?? null,
        steeringMode: request.queueModes.steeringMode, followUpMode: request.queueModes.followUpMode });
    },
    async byRequest(requestId: string): Promise<SubmissionRecord | null> {
      const found = await session.commit(tx => tx.submissionByRequest(conversationId, requestId), context);
      return found === undefined ? null : plain(found) as unknown as SubmissionRecord;
    },
    async commit(ops: AdmissionOp[], expect: AdmissionExpect): Promise<{ id: number; placed: number[] }> {
      if (!Array.isArray(ops) || ops.length === 0) throw new Error('commit takes a non-empty list of admission operations');
      return session.commit(async tx => {
        // Table reads first.
        if (expect.requestAbsent !== undefined && (await tx.submissionByRequest(conversationId, expect.requestAbsent)) !== undefined) {
          throw new StateChanged(`request ${expect.requestAbsent} was admitted meanwhile`);
        }
        const live = await tx.doc(LiveDoc, conversationId);
        const owner = (live.run?.taskId ?? null) as number | null;
        if (owner !== expect.run) throw new StateChanged(`the run belongs to ${owner ?? 'no task'}, not ${expect.run ?? 'no task'}`);
        const inbox = await tx.doc(InboxDoc, conversationId);
        const ids = inbox.items.map(item => item.id as number);
        if (JSON.stringify(ids) !== JSON.stringify(expect.inbox)) throw new StateChanged(`the inbox holds [${ids.join(', ')}], not [${expect.inbox.join(', ')}]`);
        let created: number | undefined;
        const placed: number[] = [];
        const requestId = (draft: SubmissionDraft) => draft.requestId === undefined ? {} : { requestId: draft.requestId };
        for (const [index, op] of ops.entries()) {
          const where = `admission operation ${index} (${op?.op})`;
          switch (op?.op) {
            case 'queue': {
              const { draft } = op;
              const { id } = await tx.createSubmission({ conversationId, ...requestId(draft), type: draft.type, status: 'queued' } as never);
              const value = copyJson((draft.type === 'write' ? draft.entry : draft.content) as never, { omitUndefinedProperties: true });
              if (draft.type === 'write') inbox.items.push({ id, mode: 'write', entry: value as never });
              else inbox.items.push({ id, mode: draft.whenBusy === 'steer' ? 'steer' : 'followUp', content: value as never });
              created = id;
              break;
            }
            case 'boundary': {
              const { selection } = op;
              const resolve = (id: number) => id === 0 ? created : id;
              const items = inbox.items;
              const indexOf = (id: number | undefined) => {
                const at = items.findIndex(item => item.id === id);
                if (at < 0) throw new StateChanged(`submission ${id} is not in the inbox`);
                return at;
              };
              const removed: number[] = [];
              for (const write of selection.writes ?? []) {
                const at = indexOf(resolve(write.id)), item = items[at]!;
                if (item.mode !== 'write') throw new Error(`${where}: inbox item ${item.id} is ${item.mode}, not a write`);
                removed.push(at);
                if (write.stale) { tx.settleSubmission(item.id, { status: 'unanswered', reason: 'stale' }); continue; }
                const entry = await tx.appendEntry(conversationId, item.entry as unknown as EntryDraft);
                tx.placeSubmission(item.id, entry.id);
              }
              for (const user of selection.users ?? []) {
                const at = indexOf(resolve(user)), item = items[at]!;
                if (item.mode === 'write') throw new Error(`${where}: inbox item ${item.id} is a write; list it under writes`);
                removed.push(at);
                const entry = await tx.appendEntry(UserEntry, conversationId, { model: [{ role: 'user', content: item.content as never, timestamp: request.now }] });
                tx.placeSubmission(item.id, entry.id);
                placed.push(item.id);
              }
              for (const at of [...new Set(removed)].sort((a, b) => b - a)) items.splice(at, 1);
              if (placed.length > 0) await startRun(tx, conversationId, live, placed as SubmissionId[]);
              break;
            }
            case 'write': {
              const entry = await tx.appendEntry(conversationId, op.draft.entry as unknown as EntryDraft);
              created = (await tx.createSubmission({ conversationId, ...requestId(op.draft), type: 'write', status: 'done', entry: entry.id } as never)).id;
              break;
            }
            case 'stale':
              created = (await tx.createSubmission({ conversationId, ...requestId(op.draft), type: 'write', status: 'unanswered', reason: 'stale' } as never)).id;
              break;
            case 'input': {
              const entry = await tx.appendEntry(UserEntry, conversationId, { model: [{ role: 'user', content: op.draft.content as never, timestamp: request.now }] });
              created = (await tx.createSubmission({ conversationId, ...requestId(op.draft), type: 'input', status: 'placed', entry: entry.id } as never)).id;
              await startRun(tx, conversationId, live, [created as SubmissionId]);
              break;
            }
            default:
              throw new Error(`${where}: unknown admission operation; they are queue, boundary, write, stale and input`);
          }
        }
        if (created === undefined) throw new Error('an admission commit creates one submission: queue, write, stale or input');
        return { id: created, placed };
      }, context);
    },
  };
}

/**
 * The selected admission: undefined for `crisp` (pi-durable's `admitSubmission`), else `admit.nl`. There is no `shadow`:
 * both sides admit by committing, so running both would admit the submission twice.
 */
export function admissionPolicy(host: Pick<PolicyHost, 'natlang' | 'services'>, setting: PluggableSetting): AdmissionPolicy | undefined {
  const mode = pluggableMode(setting, 'crisp');
  if (mode === 'shadow') throw new TypeError('admission is "crisp" or "nl", not "shadow": each side commits its own admission, so both cannot run.');
  if (mode === 'crisp') return undefined;
  return {
    async admit(request, context) {
      const result = await host.natlang.run(() => (admit as unknown as (draft: SubmissionDraft) => Promise<AdmitResult>)(plain(request.draft) as SubmissionDraft), {
        services: { ...host.services, admission: admissionService(request, context) }, serviceDeclarations: { admission: ADMISSION_DECLARATION },
        name: `admit#${request.conversationId}` });
      if (result.busy) throw new ConversationBusy(request.conversationId as ConversationId);
      if (result.conflict !== undefined) throw new Error(result.conflict);
      if (typeof result.id !== 'number') throw new Error('admission returned no submission ID');
      return result.id as SubmissionId;
    },
  };
}
