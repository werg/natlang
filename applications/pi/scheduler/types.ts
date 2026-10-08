import type { JsonValue, Op } from '../types.ts';

/**
 * Facts and decisions of the scheduler's policy (spec §5, PORT.md "The scheduler"). The host reads the facts on the
 * Session line; a function decides and returns a decision; the host commits it, guarded on the records read, and asks
 * again when a guard fails. No function here writes anything.
 *
 * THE OWNERSHIP TREE (spec §5.4, §5.5). These rules are stated once, here, and every scheduler function applies them.
 * - A task's parent is its owner task when it has `owner`, otherwise its conversation. A conversation's parent is the
 *   task that owns it, or none for an ownerless root conversation. Edges never change; terminal tasks stay on the walk.
 * - `above` is the walk up from a task's parent (or from a queued conversation itself): owner tasks and conversations
 *   in order, ending at an ownerless conversation, or at a step of kind "unknown" (an edge not loaded yet).
 * - LIVE: a task among the facts' `tasks` (pending, running, waiting or completing). Terminal tasks are not live.
 * - OWNED LIVE WORK of task X: every live task T with `background` false such that X appears in T.above as a "task"
 *   step before the walk passes a background owner. Precisely: walk T.above in order; at each "task" step, T counts as
 *   owned live work of that step's task; stop after a "task" step whose `background` is true, and stop at "unknown".
 * - IN SCOPE of conversation C (optionally crossing background owners): walk `above` in order; a "conversation" step
 *   with id C means in scope; a "task" step with `background` true means not in scope, unless crossing; "unknown"
 *   means not known (treat as not in scope wherever a rule asks for "in scope"); reaching the end means not in scope.
 *   For the scope of every ownerless conversation (`null`), reaching the end means in scope instead.
 * - CANCELLATION INTENT of a task: it is not terminal, and it is abort-marked or it is completing with an outcome other
 *   than "completed".
 * - BELOW CANCELLED (a walk `above`): walk in order; "unknown" gives false; skip "conversation" steps; at a "task" step,
 *   if that task has cancellation intent the answer is true; otherwise, if it is background, the answer is false; a
 *   terminal task never counts but the walk passes it. Reaching the end gives false.
 * - FAILED: a task completing or terminal with an outcome other than "completed".
 */

/** One step of a walk up the ownership tree; see THE OWNERSHIP TREE. */
export type OwnerStep = {
  kind: "task" | "conversation" | "unknown";
  id?: number;
  background?: boolean;
  /** For a task step: its status, "terminal" once it ended. */
  status?: "pending" | "running" | "waiting" | "completing" | "terminal";
  abortRequested?: boolean;
  /** A completing task's held outcome status. */
  outcome?: string;
};

/**
 * A live task. `version`: the definition version it was stored with. `on` and `policy`: what a waiting task waits for.
 * `outcome`, `message`, `reason`: a completing task's held outcome (its status, error message, orphan reason).
 * `invocation`: the in-memory invocation running it now ("run" or "abort"), or null.
 */
export type TaskFact = {
  id: number;
  conversationId: number;
  kind: string;
  version: number;
  status: "pending" | "running" | "waiting" | "completing";
  abortRequested: boolean;
  background: boolean;
  owner?: number;
  on?: number[];
  policy?: "allSettled" | "failFast";
  outcome?: string;
  message?: string;
  reason?: string;
  invocation: "run" | "abort" | null;
  above: OwnerStep[];
};

/** The registered definition of a kind: its version, and whether it can migrate older records. */
export type DefinitionFact = { kind: string; version: number; migrate: boolean };

export type BlockedReason = "missing_task" | "task_too_old" | "migration_failed";

/**
 * Facts of a scheduling pass. `tasks`: every live task, in order. `definitions`: the registered definitions (a kind
 * absent here has none). `migrationFailed`: tasks whose migration already failed under the definition registered now.
 * `idleScopes`: scopes someone waits to become idle: a conversation ID, or null for every ownerless conversation.
 */
export type PassFacts = { tasks: TaskFact[]; definitions: DefinitionFact[]; migrationFailed: number[]; idleScopes: (number | null)[] };

/** Task faulted or orphaned by the scheduler, for its cleanup. */
export type CleanupTask = { id: number; kind: string; conversationId: number };
/** The scheduler's outcome: faulted with its message, or orphaned with its reason. */
export type SchedulerOutcome = { status: "faulted" | "orphaned"; message?: string; reason?: string };

/**
 * A pass's decision: the tasks to reserve now (mode "abort" for an abort-marked task, `migrate` when its definition is
 * newer), the abort-marked tasks to settle "orphaned" with their reason and cleanup (operations from `cleanup`), and
 * for each idle scope whether it is idle.
 */
export type PassDecision = {
  reserve: { id: number; mode: "run" | "abort"; migrate: boolean }[];
  orphan: { id: number; reason: BlockedReason; cleanup: Op[] }[];
  idle: { scope: number | null; idle: boolean }[];
};

/**
 * Facts of a step, decided before each phase of a run invocation and once after an abort handler.
 * - `mode`: "run", or "abort" after the abort handler returned or threw;
 * - `task`: the task as committed now, null when it no longer exists or is terminal;
 * - `closing`: the Harness is closing;
 * - `previous`: null before the first phase; otherwise the phase that returned: its `phase` name, `error` when it threw
 *   (or, in abort mode, when the abort handler threw), and `unchanged` when the checkpoint after it equals the one it
 *   started with (compared by the host as JSON with key order ignored);
 * - `definition`: the registry's current definition for the task's kind: `same` as the one this invocation runs,
 *   `exists`, and `canReserve` (same version, or newer with a migration);
 * - `reported`: that definition was already reported as unable to take over.
 */
export type StepFacts = {
  mode: "run" | "abort";
  task: { id: number; conversationId: number; kind: string; status: "pending" | "running" | "waiting" | "completing"; abortRequested: boolean; checkpoint?: JsonValue } | null;
  closing: boolean;
  previous: { phase?: string; error?: string; unchanged?: boolean } | null;
  definition: { same: boolean; exists: boolean; canReserve: boolean };
  reported: boolean;
};

/**
 * A step's decision: "continue" (run the next phase), "stop" (end the invocation, write nothing), "fault" (end it and
 * settle the task "faulted" with `message`; `cleanup` from `cleanup`), or "handover" (back to pending under the new
 * definition). `report`: a message to report once.
 */
export type StepDecision = { kind: "continue" | "stop" | "fault" | "handover"; message?: string; report?: string; cleanup?: Op[] };

/** A member of a failFast waiter's `on`: whether it is live, its status ("terminal" or "missing" when not live), its outcome. */
export type MemberFact = { id: number; live: boolean; status: "pending" | "running" | "waiting" | "completing" | "terminal" | "missing"; abortRequested: boolean; outcome?: string };

/**
 * Facts of a reconcile. `tasks`: every live task. `failFast`: waiting tasks with policy failFast flagged for a check,
 * with each member of their `on`. `queued`: conversations with queued submissions, each with the walk `above` starting
 * at the conversation itself (empty unless a cascade is pending).
 */
export type ReconcileFacts = { tasks: TaskFact[]; failFast: { id: number; members: MemberFact[] }[]; queued: { conversation: number; above: OwnerStep[] }[] };

/**
 * A reconcile's decision: tasks to abort-mark, conversations whose queued inputs to withdraw, and completing tasks to
 * finalize, in order, each with its cleanup when its held outcome is faulted or orphaned.
 */
export type ReconcileDecision = { mark: number[]; withdraw: number[]; finalize: { id: number; cleanup?: Op[] }[] };

/**
 * Facts of `Harness.abortTask(id)`. `task`: the task as committed (any status, "terminal" included), or null when no
 * task has that ID. `invocation`: its active invocation. `tasks`: every live task. `blocked`: why the registered
 * definition cannot take it (a migration was attempted), or null when it can; computed only when it has no invocation
 * and is not completing.
 */
export type AbortTaskFacts = {
  id: number;
  task: { id: number; conversationId: number; kind: string; version: number; status: "pending" | "running" | "waiting" | "completing" | "terminal"; abortRequested: boolean; background: boolean; owner?: number; on?: number[]; policy?: "allSettled" | "failFast"; outcome?: string; message?: string; reason?: string; above: OwnerStep[] } | null;
  invocation: "run" | "abort" | null;
  tasks: TaskFact[];
  blocked: BlockedReason | null;
};

/**
 * abortTask's decision: "reject" (no such task; `message`), "terminal" (already ended; nothing written), "orphan"
 * (settle "orphaned" with `reason` and `cleanup` now), or "mark" (set the abort mark). `join`: wait until the active
 * run invocation ends.
 */
export type AbortTaskDecision = { kind: "reject" | "terminal" | "orphan" | "mark"; message?: string; reason?: BlockedReason; join: boolean; cleanup?: Op[] };

/**
 * Facts of `Conversation.abort()`: the conversation, whether to cross background owners, every live task, and the
 * conversations with queued submissions with the walk `above` from each conversation itself.
 */
export type AbortConversationFacts = { conversation: number; background: boolean; tasks: TaskFact[]; queued: { conversation: number; above: OwnerStep[] }[] };

/** Tasks to abort-mark, conversations whose queued inputs to withdraw, and tasks to wait for until terminal. */
export type AbortConversationDecision = { mark: number[]; withdraw: number[]; waitFor: number[] };
