/**
 * The write operations of `durable.commit` (types.ts `Op`): a natural-language function decides outside the commit
 * line, then hands the host one list of operations with guards on what it read. The host applies them here, inside one
 * pi-durable commit, in order, with pi-durable's own record helpers so the invariants they carry (usage counted with
 * every assistant entry, the stored tool result equal to what the model sees, run inputs settled with the run) hold.
 *
 * pi-durable's transaction requires table reads before the first table write, so guards and boundary facts are read
 * first.
 */
import type { Draft } from '@earendil-works/chord';
import type { AssistantMessage, ToolCall } from '@earendil-works/pi-ai';
import { UserEntry, SystemEntry } from './vendor/durable/src/entries.ts';
import type { ConversationId, EntryDraft, EntryId, NextTaskState, RunningTask, SubmissionId, TaskId, Tx } from './vendor/durable/src/types.ts';
import { addTools } from './vendor/durable/src/harness/agent.ts';
import { createCompaction } from './vendor/durable/src/harness/compaction.ts';
import { appendAssistant, convertPartial, GenerationTask, handOver, startRun } from './vendor/durable/src/harness/generation.ts';
import { InboxDoc } from './vendor/durable/src/harness/inbox.ts';
import { clearProgress, compactionStatus, endRun, LiveDoc, removeCompactionStatus, type LiveState } from './vendor/durable/src/harness/live.ts';
import { appendToolResult, harnessError, ToolTask } from './vendor/durable/src/harness/tool.ts';
import type { ToolExecutionResult } from './vendor/durable/src/harness/types.ts';
import { recordUsage } from './vendor/durable/src/harness/usage.ts';
import type { CommitResult, Expect, Op } from './types.ts';

/** A guard failed: the state changed after the function read it. The function reads again and decides again. */
export class StateChanged extends Error {
  constructor(what: string) { super(`state changed: ${what}; read the state again and decide again`); this.name = 'StateChanged'; }
}

/** An operation list the host cannot apply as written. */
export class InvalidOperation extends Error {
  constructor(message: string) { super(message); this.name = 'InvalidOperation'; }
}

export type ApplyScope = {
  readonly taskId: TaskId;
  readonly conversationId: ConversationId;
  readonly now: () => number;
};

type Ids = Record<string, number>;

/** Resolve "$name" to an ID created earlier in the commit; numbers pass through. */
function id(ids: Ids, value: number | string | undefined, field: string): number | undefined {
  if (value === undefined) return undefined;
  if (typeof value === 'number') return value;
  if (typeof value === 'string' && value.startsWith('$') && Object.hasOwn(ids, value.slice(1))) return ids[value.slice(1)];
  if (typeof value === 'string' && /^\d+$/.test(value)) return Number(value);
  throw new InvalidOperation(`${field}: ${JSON.stringify(value)} names nothing created earlier in this commit; give an ID or "$name" of an earlier operation's as`);
}

/** Replace every string that is exactly "$name" of an earlier operation's `as` by that ID, throughout `value`. */
function resolveRefs(ids: Ids, value: unknown): unknown {
  if (typeof value === 'string') return value.startsWith('$') && Object.hasOwn(ids, value.slice(1)) ? ids[value.slice(1)] : value;
  if (Array.isArray(value)) return value.map(item => resolveRefs(ids, item));
  if (value && typeof value === 'object') return Object.fromEntries(Object.entries(value).map(([key, item]) => [key, resolveRefs(ids, item)]));
  return value;
}

function name(ids: Ids, as: string | undefined, value: number): void {
  if (as === undefined) return;
  if (Object.hasOwn(ids, as)) throw new InvalidOperation(`two operations are named ${JSON.stringify(as)}`);
  ids[as] = value;
}

/** Strict JSON without undefined properties, as pi-durable stores it. */
const json = <T>(value: T): T => JSON.parse(JSON.stringify(value)) as T;

/**
 * Apply `ops` in one transaction. Returns the next state for pi-durable's runtime commit (or undefined) and the commit
 * result for the function.
 */
export async function applyOps(tx: Tx, scope: ApplyScope, current: RunningTask<unknown, unknown, unknown>, ops: readonly Op[],
  expect: Expect | undefined): Promise<{ next: NextTaskState<unknown, unknown> | undefined; result: CommitResult }> {
  if (!Array.isArray(ops)) throw new InvalidOperation('commit takes a list of operations');
  const { conversationId } = scope;
  // Table reads first.
  if (expect?.tail !== undefined) {
    const newest = (await tx.scanEntries({ conversationId }, 1)).items[0]?.id;
    if (newest !== expect.tail) throw new StateChanged(`the newest entry is ${newest ?? 'none'}, not ${expect.tail}`);
  }
  const placing = ops.some(op => (op as Op | undefined)?.op === 'place');
  let activeStart = placing ? (await tx.latestHeadMarker(conversationId))?.head : undefined;
  const live = await tx.doc(LiveDoc, conversationId) as Draft<LiveState>;
  if (expect && 'run' in expect && expect.run !== undefined) {
    const owner = live.run?.taskId ?? null;
    if (owner !== expect.run) throw new StateChanged(`the run belongs to ${owner === null ? 'no task' : `task ${owner}`}, not ${expect.run === null ? 'no task' : `task ${expect.run}`}`);
  }
  const inbox = await tx.doc(InboxDoc, conversationId);
  if (expect?.inbox !== undefined) {
    const ids = inbox.items.map(item => item.id as number);
    if (JSON.stringify(ids) !== JSON.stringify(expect.inbox)) throw new StateChanged(`the inbox holds [${ids.join(', ')}], not [${expect.inbox.join(', ')}]`);
  }

  const ids: Ids = {};
  const placed: number[] = [];
  let next: NextTaskState<unknown, unknown> | undefined;
  const self = scope.taskId;
  const slotOf = (callId: string) => {
    const slot = live.tools?.find(item => item.callId === callId);
    if (!slot) throw new InvalidOperation(`pi.live.tools has no slot for call ${JSON.stringify(callId)}`);
    return slot;
  };
  const myStatus = () => compactionStatus(live, self);

  for (const [index, op] of ops.entries()) {
    if (!op || typeof op !== 'object') throw new InvalidOperation(`operation ${index} is not an operation object`);
    const where = `operation ${index} (${op.op})`;
    if (next !== undefined) throw new InvalidOperation(`${where}: next must be the last operation`);
    switch (op.op) {
      case 'appendAssistant': {
        const entry = await appendAssistant(tx, conversationId, json(op.message) as unknown as AssistantMessage);
        name(ids, op.as, entry.id);
        break;
      }
      case 'appendToolResult': {
        if (!op.call?.id || !op.call?.name) throw new InvalidOperation(`${where}: call needs id and name`);
        const result: ToolExecutionResult = op.error ? harnessError(op.error.code, op.error.message) :
          json(op.result ?? {}) as unknown as ToolExecutionResult;
        const call = { type: 'toolCall', id: op.call.id, name: op.call.name, arguments: {} } as ToolCall;
        const entry = await appendToolResult(tx, conversationId, call, result, scope.now(), op.durationMs);
        name(ids, op.as, entry.id);
        break;
      }
      case 'appendUser': {
        const entry = await tx.appendEntry(UserEntry, conversationId, { model: [{ role: 'user', content: json(op.content) as never, timestamp: scope.now() }] });
        name(ids, op.as, entry.id);
        break;
      }
      case 'appendSystem': {
        if (op.message?.role !== 'system') throw new InvalidOperation(`${where}: message must be a system message`);
        const entry = await tx.appendEntry(SystemEntry, conversationId, { model: [json(op.message) as never],
          ...(op.edits?.length ? { edits: json(op.edits) as never } : {}) });
        name(ids, op.as, entry.id);
        break;
      }
      case 'appendEntry': {
        if (typeof op.entry?.kind !== 'string' || !op.entry.kind) throw new InvalidOperation(`${where}: entry needs a kind`);
        const draft = json(op.entry) as unknown as EntryDraft;
        const entry = await tx.appendEntry(conversationId, draft);
        if (draft.head !== undefined && activeStart !== undefined) activeStart = draft.head === 'self' ? entry.id : draft.head;
        name(ids, op.as, entry.id);
        break;
      }
      case 'convertPartial':
        await convertPartial(tx, live, conversationId);
        break;
      case 'createTask': {
        const ownership = op.owner === 'self' ? { kind: 'task' as const, taskId: self } : { kind: 'conversation' as const };
        let task: TaskId;
        if (op.kind === 'pi.tool') {
          const input = op.input as { assistant?: unknown, callId?: unknown };
          if (typeof input?.callId !== 'string') throw new InvalidOperation(`${where}: a pi.tool task's input is {assistant, callId}`);
          task = await tx.createTask(ToolTask, { assistant: id(ids, input.assistant as number | string, `${where} input.assistant`) as EntryId, callId: input.callId },
            { ownership, conversationId });
        } else if (op.kind === 'pi.generation') task = await tx.createTask(GenerationTask, {}, { ownership, conversationId });
        else throw new InvalidOperation(`${where}: kind must be pi.tool or pi.generation; use createCompaction for compactions`);
        name(ids, op.as, task);
        break;
      }
      case 'createCompaction': {
        if (op.ifNone && (live.compactions?.length ?? 0) > 0) break;
        const task = await createCompaction(tx, conversationId, { reason: op.reason, ...(op.instructions ? { instructions: op.instructions } : {}) },
          op.owner === 'self' ? self : undefined);
        name(ids, op.as, task);
        break;
      }
      case 'liveGeneration':
        if (op.value === null) delete live.generation;
        else live.generation = json(op.value) as never;
        break;
      case 'clearTools':
        delete live.tools;
        break;
      case 'setTools':
        live.tools = op.slots.map((slot: Extract<Op, { op: 'setTools' }>['slots'][number]) => json({ callId: slot.callId, name: slot.name, status: slot.status,
          ...(slot.taskId !== undefined ? { taskId: id(ids, slot.taskId, `${where} taskId`) } : {}),
          ...(slot.entry !== undefined ? { entry: id(ids, slot.entry, `${where} entry`) } : {}) })) as never;
        break;
      case 'slot': {
        const slot = slotOf(op.callId);
        if (op.taskId !== undefined) slot.taskId = id(ids, op.taskId, `${where} taskId`) as TaskId;
        if (op.status !== undefined) slot.status = op.status;
        if (op.entry !== undefined) slot.entry = id(ids, op.entry, `${where} entry`) as EntryId;
        if (op.clearProgress || op.status === 'done') clearProgress(slot);
        break;
      }
      case 'compactionStatus': {
        if (op.remove) { removeCompactionStatus(live, self); break; }
        const status = myStatus();
        if (!status) break;
        if (op.attempt !== undefined) status.attempt = op.attempt;
        if (op.retry === null) delete status.retry;
        else if (op.retry !== undefined) status.retry = json(op.retry);
        break;
      }
      case 'startRun':
        await startRun(tx, conversationId, live, op.inputs.map((input: number | string) => id(ids, input, `${where} inputs`) as SubmissionId));
        break;
      case 'endRun':
        endRun(tx, live, self, resolveRefs(ids, json(op.settlement)) as never);
        break;
      case 'handOver':
        handOver(live, self, id(ids, op.to, `${where} to`) as TaskId);
        break;
      case 'runInputs':
        if (live.run?.taskId === self) live.run.inputs.push(...op.append.map((input: number | string) => id(ids, input, `${where} append`) as SubmissionId));
        break;
      case 'place': {
        const { selection } = op;
        const items = inbox.items;
        const indexOf = (item: number) => {
          const at = items.findIndex(entry => entry.id === item);
          if (at < 0) throw new StateChanged(`submission ${item} is not in the inbox`);
          return at;
        };
        const removed: number[] = [];
        for (const write of selection.writes ?? []) {
          const at = indexOf(write.id), item = items[at]!;
          if (item.mode !== 'write') throw new InvalidOperation(`${where}: inbox item ${write.id} is ${item.mode}, not a write`);
          removed.push(at);
          if (write.stale) { tx.settleSubmission(item.id, { status: 'unanswered', reason: 'stale' }); continue; }
          const draft = item.entry as unknown as EntryDraft;
          const entry = await tx.appendEntry(conversationId, draft);
          if (draft.head !== undefined) activeStart = draft.head === 'self' ? entry.id : draft.head;
          tx.placeSubmission(item.id, entry.id);
        }
        for (const user of selection.users ?? []) {
          const at = indexOf(user), item = items[at]!;
          if (item.mode === 'write') throw new InvalidOperation(`${where}: inbox item ${user} is a write; list it under writes`);
          removed.push(at);
          const entry = await tx.appendEntry(UserEntry, conversationId, { model: [{ role: 'user', content: item.content as never, timestamp: scope.now() }] });
          tx.placeSubmission(item.id, entry.id);
          placed.push(item.id);
        }
        for (const at of [...new Set(removed)].sort((a, b) => b - a)) items.splice(at, 1);
        break;
      }
      case 'usage':
        if (op.bucket !== 'models' && op.bucket !== 'tools') throw new InvalidOperation(`${where}: bucket is models or tools`);
        await recordUsage(tx, conversationId, op.bucket, op.key, json(op.usage) as never);
        break;
      case 'addTools':
        await addTools(tx, conversationId, op.names);
        break;
      case 'next': {
        const state = resolveRefs(ids, json(op.state)) as typeof op.state;
        if (state?.status === 'running') next = { status: 'running', checkpoint: json(state.checkpoint) };
        else if (state?.status === 'waiting') next = { status: 'waiting', checkpoint: json(state.checkpoint), policy: state.policy ?? 'allSettled',
          on: state.on.map((member: number | string) => id(ids, member, `${where} on`) as TaskId) };
        else if (state?.status === 'terminal') next = { status: 'terminal', outcome: json(state.outcome) as never };
        else throw new InvalidOperation(`${where}: state.status is running, waiting or terminal`);
        break;
      }
      default:
        throw new InvalidOperation(`${where}: unknown operation; the operations are listed in the Op type`);
    }
  }
  void current; void activeStart;
  return { next, result: { ids, placed } };
}
