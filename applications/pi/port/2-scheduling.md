# Group 2: scheduling and the public Harness surface

Source: `/home/werg/src/pi/packages/durable` at f10993b. Files: `src/harness/{scheduler,harness,submissions,inbox,task-graph,events,output,view,util,json,types}.ts`.
Spec: `docs/spec.md` §2.2 (public surface), §5.1–5.5 (tasks, scheduler, abort, structured concurrency), §6 (submissions and
inbox), §9.3–9.5 (conversation view, agent events, task graph).

## How this group splits

**Policy as natural language (fn).** The decisions that make up "what happens next" are eight functions. Each gets its
facts as data and returns a decision. The host applies the writes.

| natlang unit | contract (inputs → output) | rules | callers | callees |
|---|---|---|---|---|
| `admitSubmission` | (tx, conversationId, draft, now, queue modes) → SubmissionId, or throws `ConversationBusy` / request-type conflict | R1 | host `submit` (Conversation, bound handle), `reset`, compaction summary placement (compaction group) | `applyBoundary`, `startRun` (generation group), tx service |
| `applyBoundary` | (tx, boundary {conversationId, inbox items, steeringMode, followUpMode, activeStart}, at = postTools or final, now) → {users: placed input IDs, reset} | R2 | `admitSubmission`; generation `answer` and `finishToolRound` (generation group) | tx service (append entry, place/settle submission, inbox document) |
| `classifyTask` | ({status, abortRequested, kind, storedVersion, hasInvocation, ownedLive ids, onLive ids, fit}, purpose = reserve or inspect) → running / completing / waiting{on} / blocked{reason, error?} / ready{mode run or abort, migrates} | R4 | scheduler host reservation pass; `inspect` | none (facts supplied by crisp helpers) |
| `decideStep` | ({mode, task {id, kind, status, abortRequested, checkpoint} or missing, closing, previous phase {checkpoint, error?} or none, registry change facts}) → continue / stop / fault{message} / handover, plus an optional report | R5 | scheduler host invocation driver (before each phase, and once after an abort handler) | none |
| `reconcile` | (live tasks with {id, background, abortRequested, belowCancelled}; failFast waiters with members {id, live, failed}; queued conversations {id, belowCancelled}; finalizable completing tasks {id, outcome}) → {mark ids, withdraw conversation ids, finalize [{id, outcome, cleanup}]} | R8 | scheduler host (after trigger commits, and at open) | none (the host writes; crisp helpers make the facts) |
| `abortTask` | ({exists, status, abortRequested, invocation mode or none, hasOwnedLive, fit}) → reject / terminal / orphan{reason} / mark, plus joinRun | R7 | `Harness.abortTask` | none |
| `abortConversation` | ({background flag, live tasks {id, background, abortRequested, inScope}, queued conversations {id, inScope}}) → {mark ids, withdraw conversation ids, waitFor ids} | R7 | `Conversation.abort`, bound handle `abort` | none |
| `abortSubmission` | (record or none, conversationId filter or none) → not_found / aborted (settle and remove from inbox) / already_placed / settled | R7 | `Harness.abortSubmission`, `Submission.abort` | `removeInboxItem` (crisp), tx service |

**Policy kept as crisp helpers.** These carry exact policy rules from spec §5.4/§5.5, but each is a walk over the
ownership tree or a fixed check with a fixed message. A small model would do them less reliably than code, so the
scheduler host calls them and passes their results to the fns as data: ownership walks (`walkUp`, `chainKnown`,
`ownedLive`, `inScope`, `belowCancelled`, `idle`), `cancellationIntent`, `failedOutcome`, `fitDefinition`,
`canReserve`, `migrateRecord`, `finalizable`, `holdOrTerminate`, `commitTaskState`, `validateWait`, `withState`,
`jsonEqual`, `withdrawQueuedInputs`, `removeInboxItem`, and output bounding.

**Mechanics as host services.** These are the Session commit line, the live-record mirror and commit listener, task
invocations (abort controllers, done promises, joins), timers and sleeps, waiters, recovery at open, close,
observation mounts (conversation view, task graph), and progress pacing. pi-durable has **no leases**. One Harness owns
one Storage. The in-memory invocation map plus the durable `running` status are the only execution ownership. A crash
leaves `running` records, and open turns them back into `pending`.

**Out.** The experimental agent-event adapter (`events.ts`). The coding agent's durable frontend renders the
conversation view and task graph and never uses it.

No decision here is scored. `decide()` is not needed: every rule below is exact.

## Policy rules (exact)

### R1 Admission: `admitSubmission` (spec §6)

This all happens in one commit. Table reads must come before the first table write.

1. **Dedupe.** If `draft.requestId` is set and a submission of this conversation already carries it:
   - If its type differs from `draft.type`, throw `Request ${requestId} already identifies a submission of type ${existing.type}`.
   - Otherwise return its ID and write nothing.
2. **Busy check.** The conversation is busy when the conversation's `pi.live.run` is present. Creating the idle
   `pi.live` document does not make it busy.
3. **Reject.** If busy, and the draft is an input with `whenBusy: "reject"`, throw `ConversationBusy`
   (`Conversation ${id} is busy`). Write nothing.
4. **Read boundary facts.** If idle, read the boundary facts now: the active-range start (the newest head marker's
   `head`), the inbox items, and the current `steeringMode` and `followUpMode`.
5. **Queue path** (busy, or idle with a non-empty inbox):
   - Create the submission `{conversationId, requestId?, type, status: "queued"}`.
   - Append an inbox item at the end. Its payload is strict JSON with undefined properties omitted:
     - a write becomes `{id, mode: "write", entry}`;
     - an input becomes `{id, mode: whenBusy === "steer" ? "steer" : "followUp", content}`.
     So `followUp` is the default, and `reject` while idle queues as a follow-up.
   - If busy, return the ID.
   - If idle, apply a `final` boundary (R2) with `now`. If it placed user items, `startRun` with those IDs: create a
     conversation-owned `pi.generation` task and set `pi.live.run = {taskId, inputs}`. Return the ID.
   - An older queued item is selected before the new one. With `one-at-a-time` modes the new input usually stays
     queued.
6. **Direct path** (idle, empty inbox):
   - **Write.** It is stale if its `entry.head` is a number, the active-range start is known, and `entry.head` is less
     than the start:
     - stale: create `{type: "write", status: "unanswered", reason: "stale"}` and append no entry;
     - otherwise: append the entry as drafted and create `{type: "write", status: "done", entry: entryId}`.

     A write never starts a run.
   - **Input.** Append a `pi.user` entry with `model: [{role: "user", content, timestamp: now}]`. Create
     `{type: "input", status: "placed", entry}`. `startRun` with `[id]`.

### R2 Boundary selection: `applyBoundary` (spec §6)

Steering versus follow-up:

1. `reset` := the inbox holds a write whose `entry.head === "self"`.
2. `final` := `at === "final"` or `reset`. A queued reset turns a `postTools` boundary into `final`.
3. **Writes.** Every write item is selected at every boundary.
4. **Steers.** Items with mode `steer`: all when `steeringMode === "all"`, else only the first in inbox order (the
   lowest ID).
5. **Follow-ups.** Only when `final`: all when `followUpMode === "all"`, else only the first.
6. **Place writes first**, in inbox order:
   - A write is stale when its `head` is a number, the active-range start is known, and `head` is less than the start.
     A stale write settles `unanswered` with reason `stale`.
   - Otherwise append the entry draft unchanged. If it has a `head`, move the active-range start to it (`"self"` means
     the new entry's own ID), so later writes in the same boundary are judged against it. Then place the submission
     (a write becomes `done` with its entry).
7. **Then place the selected user items**, in inbox order. For each: append a `pi.user` entry
   `{role: "user", content, timestamp: now}`, place the submission (an input becomes `placed` with its entry), and
   collect its ID.
8. **Remove** every write and every selected user item from the inbox by position. The remaining items keep their
   order.

Consequences:

- User items queued before a reset or summary run after it, in the new context.
- Queued user items never become stale.
- With the default modes (`one-at-a-time` for both), a boundary places at most one steer, and at most one follow-up at
  a final boundary.

### R3 Turn boundaries as the callers apply them (generation group; summary for context)

- **Idle submission with a non-empty inbox:** a `final` boundary (R1.5).
- **Final answer** (`answer`):
  - Run the `final` boundary.
  - If an `onYield` continuation exists and the boundary selected no user item and no reset: append the continuation as
    a user message and hand `pi.live.run` to a new generation. The inputs stay open.
  - Otherwise: settle the run's inputs `done` with the answer entry, remove `run`, and start a new run with any placed
    users.
- **Tool round done** (`finishToolRound`), terminate or handoff case. Terminate means every call of the round asked to
  terminate. With several handoffs, the last in call order wins. A handoff is appended first as a `pi.reset` entry
  with `head: "self"` and a user message, and the active-range start moves to it. Then:
  - run the `final` boundary;
  - settle the inputs `done` with the tool-calling answer;
  - start a new run with the placed users.
- **Tool round done, other cases** (`postTools` boundary):
  - If the boundary selected a reset: the run's inputs settle `unanswered` with reason `reset`, and the placed users
    start a new run.
  - Otherwise: the placed steer IDs are appended to `run.inputs`, and a new generation takes the run. Generation
    continues even with nothing queued.
- **Failure, abort handler, fault, orphan:** the run's inputs settle `unanswered`. The inbox is left alone.

### R4 What runs next: scheduling pass and `classifyTask` (spec §5.1, §5.4)

**When a pass runs.** A pass is triggered by `resume()`, by any commit that changes a task, by a registry change, and
by the end of any invocation.

**What a pass does.** It is one commit that reserves *every* eligible live task at once. There is no priority, no
concurrency limit and no fairness order. Everything eligible starts concurrently, in mirror order.

**`classifyTask`, per live task (first match wins):**

1. It has an active invocation → `running`. Skip.
2. Its status is `completing` → `completing`. Skip. A held outcome never runs code again, needs no definition, and is
   never migrated.
3. Its waits-on set is non-empty → `waiting{on}`. Skip. The waits-on set is:
   - if `abortRequested`: its live ordinary owned work (from `ownedLive`). Abort runs bottom-up. A marked waiting task
     leaves its wait early, and does not wait for members of `on` it does not own.
   - else, if status is `waiting`: the members of `on` that are still live.
   - else: empty.
4. The definition fit (R4a) is blocked → `blocked{reason}`.
   - In a pass, an abort-marked blocked task settles `orphaned` with that reason, through `holdOrTerminate`. Its owned
     work is already gone (rule 3), so it becomes terminal at once and an orphaned outcome never holds.
   - An unmarked blocked task stays `pending` or `waiting`. It still counts for idle and is reconsidered on every
     registry change.
5. Inspection only: the task needs migration but the definition has no `migrate` → `blocked{migration_failed}`, with
   error `Task ${kind} version ${defVersion} has no migration from ${storedVersion}`.
6. Otherwise → `ready{mode: abortRequested ? "abort" : "run", migrates: versions differ}`.

**Reserving a `ready` task** (inside the pass commit):

- If it migrates, run `migrate(input, checkpoint, storedVersion)` (`migrateRecord`):
  - If `migrate` is missing or throws: remember (definition object, error), report once, and treat the task as
    `blocked{migration_failed}`.
  - Otherwise: the record gets the definition's `version` and the migrated `input` and checkpoint.
- Write status `running` with the (migrated) checkpoint. Skip this write when the record is already `running` and was
  not migrated: a step write that storage rejected leaves the task `running`, and the task reruns without a rewrite.
  The `waiting` fields `on` and `policy` are dropped.
- Register the in-memory invocation inside the commit. Start it after the commit.

**R4a `fitDefinition(record, definition)`:**

- no definition → `missing_task`
- equal version → fits, no migration
- definition version lower than the stored version → `task_too_old`
- this same definition object already failed to migrate this task → `migration_failed` (with the error)
- otherwise → fits, with migration

`canReserve(definition, record)` is true when the versions are equal, or the definition is newer and has `migrate`.

### R5 Step precedence: `decideStep` (spec §5.1, rules 1–6)

This runs on the commit line before every phase (including the first), and once after an abort handler returns. The
first match wins. The numbering here is this document's own. The spec's rules map as follows: spec 1 = R5.1,
spec 2 = R5.2, spec 3 = R5.4, spec 4 = R5.6, spec 5 = R5.8, spec 6 = R5.7. The code checks "unchanged" before
"progress", which is equivalent.

1. The task is missing, or not `running` (it is terminal, `completing`, `waiting`, or `pending`) → stop, write
   nothing. A `waiting` state ends the invocation even when its checkpoint is unchanged.
2. The Harness is closing → stop, write nothing. The checkpoint and abort mark are kept for reopen.
3. **Abort mode only** (after the abort handler) → fault. The message is the handler's thrown error message, or else
   `Abort handler of task ${id} returned without a terminal outcome`.
4. Run mode, and `abortRequested` → stop, write nothing. The abort invocation starts when R4 finds no owned work live.
   This rule comes before rule 6, so a handler that threw because it was signalled does not fault.
5. No phase has run yet → continue.
6. The previous phase threw → fault with its message (`error.message`, else `String(error)`).
7. The checkpoint after the phase is structurally equal to the one it started with (`jsonEqual`: object key order
   ignored, arrays ordered) → fault `Task ${kind} phase ${phase} returned without durable progress`.
8. Progress was made: take a fresh registry snapshot. If the task's kind now resolves to a *different* definition
   object than this invocation's:
   - if that definition exists and `canReserve` → **handover**: write status `pending` with the same checkpoint (memos
     and abort mark kept) and stop. The next pass reserves it under the new definition.
   - otherwise → report once per resolved definition `Task ${id} keeps running under its old ${kind} definition`
     (cause `missing_task` when absent, else `incompatible_task`), and continue under the old definition.
9. Continue: the host runs `phases[checkpoint.phase]`.

A fault is written with `holdOrTerminate(record, {status: "faulted", error: {message}})`. The invocation ends inside
the step commit. If storage rejects the step's write, the invocation ends anyway, the task stays `running`, and the
next pass reruns it.

### R6 Ownership tree (crisp helpers; spec §5.4, §5.5)

- **Parents.**
  - A task's parent is its `owner` task if it has one (a child task, always in the same conversation), else its
    conversation.
  - A conversation's parent is its owner task, or none (an ownerless root).
  - Edges are immutable. Terminal tasks remain nodes that walks pass through.
- **`walkUp(start)`** yields each owner task (with node `{conversationId, owner?, background}`) and each conversation,
  from `start` up to an ownerless conversation. It yields `unknown` and stops when an edge or node is not loaded.
  - Node lookup order: this commit's staged record, then the live mirror, then cached terminal nodes.
  - Edge lookup order: this commit's staged conversation, then the cache.
- **`ownedLive`** maps each task to its live ordinary owned work. For each live, non-background candidate task X: walk
  up from X's parent and add X to every owner task met, up to *and including* the first background owner, then stop.
  An unknown edge also stops the walk.
- **`inScope(start, scope, crossBackground)`**:
  - reaching the scope's conversation → `true`;
  - meeting a background owner task first, when `crossBackground` is false → `false`;
  - reaching the top → `true` for the `roots` scope, `false` for a conversation scope;
  - an unloaded edge → `unknown`.
- **`cancellationIntent(task)`** is true when the task is not terminal and either has `abortRequested`, or is
  `completing` with an outcome other than `completed`.
- **`failedOutcome(task)`** is true when the task is `completing` or terminal with an outcome other than `completed`.
- **`belowCancelled(start)`** walks up over owner tasks:
  - unknown edge → false;
  - a *live* owner with cancellation intent → true;
  - an owner that is background → false (a background owner with its own intent already returned true);
  - terminal owners never count, but the walk passes through them;
  - reaching the top → false.
- **`holdOrTerminate(record, outcome)`** writes the outcomes the scheduler decides, `faulted` and `orphaned`:
  - if the task has live ordinary owned work in this commit's candidates → write `completing{outcome}`;
  - else → write terminal `{outcome}` and, in the same commit, run the Harness cleanup `settleSchedulerOutcome`. That
    cleanup lives in `live.ts` (live group): it ends the run with inputs `unanswered` (reason `faulted` with the
    message as detail, or the orphan reason), marks a tool slot done, or removes a compaction status.
- **`commitTaskState`** applies a state that a runtime commit returns:
  - `waiting` → run `validateWait` first.
  - `terminal` → load the owner chains of every task staged in this commit, so that work the same commit creates
    counts. If owned work is live, write `completing{outcome}` instead. The task-written terminal does its own
    settlement, so there is no cleanup hook.
  - Memos are dropped whenever a record becomes `completing` or terminal (`withState`).

### R7 Abort (spec §2.2, §5.4)

**`abortTask(id)`.** One commit, then an optional join.

1. Unknown ID → reject `Task ${id} does not exist`.
2. Terminal → return `terminal`, write nothing.
3. No active invocation, not `completing`, no live ordinary owned work, and the current registry cannot take it →
   settle `orphaned{reason}` now and return `marked`. "Cannot take it" means R4a is blocked, or the migration attempt
   fails. pi does run `migrate` here to find out, and records a failure as with reservation.
4. Otherwise set `abortRequested` (if not already set) and return `marked`. A `completing` task is only marked: its
   final record keeps the held outcome and the mark, and the mark cascades.
5. After the commit, if a *run* invocation was active, wait until it ends. The new mark signals it, and its next step
   stops it (R5 rule 4).

It does not wait for the task to become terminal. It applies to background tasks too. It does not enable scheduling.

What follows from other rules:

- the cascade (R8) marks the task's ordinary owned work;
- R4 starts the abort invocation once that work is gone;
- the abort handler must commit a terminal outcome, or R5 rule 3 faults the task;
- a definition that cannot take the task means `orphaned`.

**`abortConversation(conversationId, background)`.**

1. One commit:
   - A task is reached when (`background` is set, or the task is not background) and
     `inScope(task's parent, {conversation}, crossBackground: background)` is true.
   - Mark each reached task (if not already marked).
   - Withdraw the queued inputs (R7w) of each conversation that has queued submissions and is `inScope` under the same
     crossing rule. The addressed conversation is always in scope.
2. If `background` is set: wait for each reached task to become terminal. Background work created after the commit is
   neither marked nor awaited.
3. Wait until the conversation is idle (R9).

Marks signal run invocations through the commit listener. Cancelling the caller's context only cancels the wait.

**R7w `withdrawQueuedInputs(conversation)`.** Every inbox item with mode `steer` or `followUp` settles `unanswered`
with reason `aborted` and is removed. Write items stay for later placement.

**`abortSubmission(id, conversationId?)`.**

- Unknown, or `conversationId` given and different → `not_found`.
- `queued` → settle `unanswered{reason: "aborted"}`, remove its inbox item, return `aborted`.
- `placed` → `already_placed` (no write).
- `done` or `unanswered` → `settled`.

The handle form `Submission.abort()` turns `not_found` into the rejection `Submission ${id} does not exist`.

### R8 Cascades, joins, finalization: `reconcile` (spec §5.4, §5.5)

**`reconcile`.** One commit. It runs after trigger commits and at open. If its commit fails, its flags are restored and
it retries with the next commit of any kind.

1. Load the owner chains of every live task. When a cascade is pending, also load those of every conversation with
   queued submissions.
2. **Cascade.** Mark every live, non-background, unmarked task that is `belowCancelled(its parent)`. Marks are derived
   again on every pass, because loading edges can reveal a cancelled owner. This also catches work created below a
   cancelled owner after its cascade.
3. **failFast.** For each waiting task W with policy `failFast` that is flagged for a check:
   - W is flagged at open, when it starts waiting, and when a member of its `on` newly fails.
   - If any member of `W.on` holds or ended with an outcome other than `completed` (live record, else stored record),
     mark every live member of `W.on` that has not itself failed.
   - W itself is not marked. It resumes once all of `on` is terminal and reads the outcomes with `runtime.outcomes()`.
4. **Withdraw.** For each conversation with queued submissions that is `belowCancelled`, apply R7w. The cancelled
   task's own conversation is not below it, so it keeps its queue.
5. **Finalize** (`finalizable`, a fixpoint over this commit's candidates). Each `completing` task with no live
   ordinary owned work gets its terminal record, with its held outcome and abort mark kept. For `faulted` and
   `orphaned`, also run `settleSchedulerOutcome` in this commit. Finalizing one task can free its owner, so repeat
   until none are left. The loop is bounded by the number of completing tasks.

**Triggers.** The commit listener queues a reconcile when:

- a task becomes terminal (its owner may finalize);
- a task is newly abort-marked;
- a task newly becomes `completing` (this is a cascade when its outcome is not `completed`);
- a task newly waits `failFast`;
- a failFast member newly fails;
- a queued input arrives below a cancelled owner, or where the chain is unknown;
- a live task appears below a cancelled owner, or where the chain is unknown;
- the Harness opens.

**Wait validation** (`validateWait`, when a runtime commit returns `waiting`):

- the invocation is in abort mode → `Abort handler of task ${id} cannot wait`;
- a member is the task itself or any owner task above it (across conversations) →
  `Task ${id} cannot wait on itself or its owner ${member}`;
- a member is missing (not staged, not live, not stored) → `Task ${member} does not exist`;
- the policy is `failFast` and a member's `owner` is not this task →
  `Task ${id} can wait failFast only on tasks it owns; ${member} is not one`.

Members that are already terminal are allowed. An empty `on` resumes at the next pass. Tasks that the waiting task does
not own are only allowed with `allSettled`. Creating owned work under an owner that is `completing`, terminal, or
abort-marked is rejected by the transaction group (§3.3).

### R9 Idle and enabling (spec §2.2, §5.4)

**Idle.** `idle(scope)`: no live, non-background task whose walk up from its parent is in scope.

- Conversation scope: the walk reaches that conversation without crossing a background owner.
- Harness scope: the walk reaches an ownerless root without crossing a background owner.
- An unknown edge counts as *inside*.
- Waiting, blocked, deadline-sleeping and `completing` tasks are live.

Idle waiters are rechecked after every task commit and every reconcile. A new wait first queues one reconcile.

**Enabling.** Scheduling starts disabled at open. `resume()` enables it, and throws `Harness is closed` after close.

- Calls that enable scheduling: `Conversation.submit`, `compact`, `abort` and `waitForIdle`; `Submission.wait`;
  `Harness.waitForTask` and `waitForIdle`; and `submit` through a bound handle.
- Calls that never enable it: `inspect`, `getTask`, `submission`, `Submission.status`, `usage`, conversation reads,
  views, the task graph, `abortTask`, `abortSubmission`.

## Durability mechanics: the service interface the policy uses

- **M1 Commit line** (`session` service).
  - `commitWith(callback)` runs serially. It is atomic across records and documents. Table reads are only allowed
    before the first table write (otherwise `ReadAfterWrite`). Publication happens only after storage settles.
  - Other operations: `readOnLine`, `subscribeCommits` (synchronous, on the line), `subscribeClose`.
  - Every task transition is decided in a callback on the line: reservation, marks, runtime commits, reconcile, and
    steps.
  - A runtime commit is *gated*: it rereads the task, then rejects if:
    - the invocation ended → `Task ${id} invocation has ended`;
    - the Harness is closing → `Harness is closed`;
    - the task is gone → `Task ${id} is terminal`;
    - the task is not running → `Task ${id} is ${status}`;
    - a run invocation's task carries a mark → `Task ${id} has a durable abort mark`.
  - An invocation ends *inside* the step commit that decides its end. So a runtime commit it queued earlier either
    lands first and counts, or arrives later and rejects.
- **M2 Candidates.** Hold and finalize decisions inside a commit judge the commit's staged records
  (`tx.stagedTasks()`, `tx.stagedConversations()`) over the committed ones. This is the "overlay". Work that a commit
  creates holds that commit's own terminal.
- **M3 Live mirror.** The commit listener keeps a map of every non-terminal task record (pending, running, waiting,
  completing), updated synchronously on the line. Its effects:
  - When a task becomes terminal, it resolves the task waiters with the terminal record.
  - When a new abort mark appears, it signals the task's run invocation.
  - It queues reconciles (R8 triggers) and scheduling passes.
- **M4 Invocations.** An invocation is the in-memory record `{taskId, conversationId, mode, abort controller, context,
  watches, ended, done}`, at most one per task. Ending is idempotent: it removes the entry, stops the watches, and
  aborts the controller with `Task ${id} invocation has ended`. After that, every runtime operation rejects with the
  same message.
- **M5 Recovery at open.**
  - One commit scans pending, running, waiting and completing records (256 per page). Every `running` record becomes
    `pending`, keeping its checkpoint, mark and memos. Every waiting `failFast` record is flagged for a check.
  - Then one reconcile runs: it re-derives cascades a crash left unapplied and finalizes held outcomes.
  - Nothing dispatches until `resume()`. There are no leases or heartbeats.
- **M6 Failure handling.**
  - A failed reservation commit drops its invocations, is reported, and the pass retries at the next kick.
  - A failed reconcile is retried at the next commit.
  - A failed step write: see R5.
  - A non-fatal extension failure goes to `onReport`.
- **M7 Close.**
  - The seal marks the Harness closing, unsubscribes from the registry, rejects task and idle waiters with
    `Harness is closed`, drops the context cache, stops the expiry timer, and signals every invocation.
  - `beforeClose` joins all invocations before storage closes.
  - Close writes no outcome: running work resumes after reopen.
- **M8 Context cache** (optimization only). Per conversation, the scheduler keeps the last context range a task read.
  A busy conversation keeps it. An idle one keeps it for `contextRetentionMs` (default 600000). Expiry is checked on
  task changes and by one unreferenced timer.
- **M9 Mapping to natlang.**
  - The commit line plus the listener plus the kick/drain and reconcile queues correspond to an `EventLoop`, one per
    Harness:
    - events: API calls, commit publications, registry changes, invocation ends;
    - `reduce`: applies one decision (fn results) and the writes;
    - `onCommit`: the storage commit, awaited before publish, which matches "publication after storage";
    - `context.after(...)`: dispatches the follow-up passes;
    - `wakeAt(state)`: carries sleeps, retry backoffs and context expiry.
  - Event-ID duplicate suppression can carry `requestId` dedupe only partly. Admission must still return the existing
    ID, and must throw on a type conflict.
  - `KeyedEventLoop` cannot key by conversation, because ownership walks and the shared Session line cross
    conversations.
  - A directory reducer's copy-and-`folder.apply` gives the same atomicity as a commit, if the store is a folder.
  - The phase driver is open-ended: tasks can have unbounded phases, such as deferred polling. So it is a host loop,
    not an `iterateOn` with a fixed limit.

---

## durable/src/harness/scheduler.ts — the durable task scheduler of one Harness: mirrors live tasks, reserves and drives invocations, applies abort cascades, joins and holds, and serves the task runtime API

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| internal type records: `AnyTaskRecord`, `RunnableTaskRecord`, `Checkpoint`, `Erased*`, `Resolution`, `Fit`, `Phase`, `Reservation`, `ReportedTask`, `PhaseResult`, `Decision`, `Up`, `Step`, `Scope` | 40–130 | Type views. `Resolution` = ready{task, record (maybe migrated)} or blocked{reason}. `Fit` = {task, migrates} or {reason, error?}. `Phase` = {snapshot(), task(), agent promise}. `Reservation` = {invocation, task, snapshot}. `ReportedTask` = the replacement definition already reported. `PhaseResult` = {checkpoint, failure?}. `Decision` = continue / stop / {fault}. `Up` = {task} or {conversation}. `Step` adds {unknown}. `Scope` = {conversation} or {roots}. | crisp | state records of the scheduler host | Types only. |
| constants | 51–54 | `SCAN_PAGE_SIZE` = 256. `MAX_TIMER_DELAY` = 2147483647 ms (a longer sleep waits in several steps). `LIVE_STATUSES` = pending, running, waiting, completing. | crisp | scheduler host | Fixed values. |
| `BlockedReason` | 57 | `missing_task` / `task_too_old` / `migration_failed`. Derived, never persisted. | crisp | state record | Enum. |
| `Invocation` | 69–81 | In-memory execution: taskId, conversationId, mode (run or abort), AbortController, a context cancelled by it, the set of watches it acquired, ended flag, done promise and finish. | service | scheduler host state (M4) | Cancellation contexts are host infrastructure. |
| `SchedulerOutcome`, `InvocationBinding` | 106–109 | `SchedulerOutcome`: `faulted{error{message}}` or `orphaned{reason}`, the only outcomes the scheduler writes itself. `InvocationBinding`: {signal, check()} for bound conversation handles. | crisp (outcome) / service (binding) | state records | Data, and a signal binding. |
| `TaskNode`, `Overlay` | 112, 124–127 | `TaskNode` = {conversationId, owner?, background}, the immutable ownership fields. `Overlay` = the current commit's staged task records and staged conversation owner edges (taskId or null). | crisp | ownership helpers (R6, M2) | Plain data for walks. |
| `TaskSchedulerOptions` | 132–157 | Injected collaborators: session, storage, registry reader, models; `agent(conversationId, snapshot, ctx)`, resolved at most once per phase; `settings()`, read at each access; `env(conversationId, ctx)`; clock; `report`; `settleOutcome(tx, record, outcome)` (the Harness cleanup hook); `withdrawInputs(tx, conversationId)`; `conversation(id, binding, ctx)` (bound-handle factory); a base context with no caller cancellation. | service | composition (harness.ts constructor) | Wiring of host services. |
| `TaskScheduler` state + constructor | 174–236 | State record: live mirror (taskId→record); context cache (conversation→{range, idleSince}); expiry timer {at, timer}; invocations (taskId→Invocation); task waiters (taskId); idle waiters (conversation, or undefined for the whole Harness); failed migrations (taskId→{definition, error}); edges (conversation→owner taskId or null); conversationOwners set; settled nodes of terminal owners; failFast check set; flags reconcileScheduled, cascadePending, enabled, closing, dirty, draining; registry unsubscribe. The constructor stores the options. | service | "scheduler host" service | Mirror and in-memory execution state. |
| `open` | 239–264 | M5. Subscribes to commits (`#observe`), close (`#seal`) and registry changes (`#kick`). One commit: read every table first (scan the four live statuses, 256 per page); load records into the mirror; rewrite `running` as `pending` with the same checkpoint (the spread keeps the mark and memos); flag waiting `failFast` records. Then set cascadePending and queue a reconcile. Dispatches nothing. | service | scheduler host; called by `Harness.open` | Crash recovery is durability mechanics. |
| `resume` | 267–270 | Sets enabled, then kicks a pass. Idempotent. The kick does nothing once closing. | service | scheduler host; R9 | State toggle. |
| `join` | 273–275 | Waits until every current invocation's `done` has settled. Writes nothing. | service | `beforeClose` | Join of host executions. |
| `abort` (abortTask) | 282–304 | R7 abortTask. One commit: read the task (unknown → reject); terminal → `terminal`; if no invocation and not completing, load chains, and if no owned work is live and resolution under the current snapshot is blocked → `#terminate` orphaned{reason} → `marked`; else set `abortRequested` if unset → `marked`. After the commit, await the run invocation's `done` (bounded by the caller's context). | fn (decision) + service (commit, join) | fn `abortTask`; called by `Harness.abortTask` | Abort semantics are policy. The commit and join are mechanics. |
| `waitForTask` | 306–316 | On the line, so that no terminal publication falls between check and registration: closing → reject `Harness is closed`; live → register a waiter, resolved with the terminal record by `#observe`; else read storage: missing → reject `Task ${id} does not exist`; else resolve with the stored (terminal) record. Cancelling the context cancels only this wait. | service | scheduler host; `Harness.waitForTask`, runtime | Waiter registry. |
| `waitForIdle` | 322–327 | Closing → reject. `idle(scope)` (R9) → resolve. Else queue a reconcile and register an idle waiter for the conversation, or for undefined (the Harness). | service | scheduler host; Conversation, Harness, bound handle | Waiting is mechanics. The predicate is crisp `idle`. |
| `abortConversation` | 334–352 | R7 abortConversation. One commit: load chains, including those of conversations with queued submissions; mark the reached tasks; withdraw queued inputs of in-scope conversations. Then (with `background`) `waitForTask` each reached task, then `waitForIdle(conversation)`. | fn (selection) + service (commit, waits) | fn `abortConversation`; called by `Conversation.abort`, bound handle | Which tasks and queues an abort reaches is policy. |
| `#observe` | 356–427 | Commit listener (M3) for each publication. Per task change: (a) a newly failed outcome is collected; (b) terminal → remove from the mirror, forget its failed migration and failFast flag, keep its node if it owns a loaded conversation, resolve its waiters, queue a reconcile; (c) a new abort mark → cascadePending, and abort the controller of its *run* invocation; (d) newly completing → queue a reconcile, and set cascadePending if it has cancellation intent; (e) newly waiting failFast → flag and queue; (f) update the mirror. For each newly failed task, flag every live failFast waiter whose `on` contains it. Record new conversation owner edges. A queued *input* submission whose chain is unknown, or below a cancelled owner → cascadePending. Each updated live record with an unknown chain → queue a reconcile; non-background, unmarked and below a cancelled owner → cascadePending. If cascadePending → queue a reconcile (this also retries a failed one). If any task changed → `#settleIdle` and kick a pass. | service | scheduler host (natlang: EventLoop after-commit dispatch, M9) | Synchronous listener on the line. The triggers only queue the idempotent `reconcile`. |
| `#contextRetentionMs` | 433–440 | `settings().contextRetentionMs`. If the settings getter throws: report, return 0 (the cache is only a cache). | crisp | scheduler host (M8) | Settings read with fallback. |
| `#settleIdle` | 443–461 | Resolves idle waiters whose scope is idle. For each kept context: idle for at least the retention → drop it; busy → clear idleSince; idle and retention > 0 → set idleSince if unset; retention 0 → drop. Then reschedule the expiry timer. | service | scheduler host (M8, R9) | Waiters and the cache. |
| `#scheduleExpiry` | 468–494 | One timer at the earliest `idleSince + retention`, with delay `min(max(0, at − now), MAX_TIMER_DELAY)`, then `#settleIdle`. The timer is unreferenced so it never keeps the process alive. Where timers cannot be unreferenced (Cloudflare Workers) there is no timer, and task changes alone check expiry. No timer once closing. | service | scheduler host (M8) | Clock and timer. |
| `#scheduleReconcile` | 498–502 | Coalesces: at most one queued reconcile (a microtask). None once closing. | service | scheduler host | Event scheduling. |
| `#reconcile` | 510–549 | R8. Takes and clears cascadePending and the failFast flags; one commit (skipped if closing): load scopes; mark the cascade; failFast marks; withdraw below cancelled owners; finalize. On error: restore cascadePending and the flags, report (unless closing). Always `#settleIdle` afterwards. | fn (decisions) + service (commit, retry) | fn `reconcile`; called by the scheduler host | Cascades, joins and finalization are structured-concurrency policy. |
| `#anyFailed` | 552–558 | Whether any ID's record (mirror, else storage) has `failedOutcome`. | inline | in fn `reconcile` (as the member fact `failed`) | One predicate per member. |
| `#finalize` | 565–582 | R8.5. Loop: candidates = mirror with staged records; owned = `ownedLive(candidates)`; done = completing candidates not in owned; none → return; write each terminal with its held outcome; faulted or orphaned → `settleOutcome`. Repeat. | crisp (`finalizable` fixpoint) + inline write rule | fixpoint in the scheduler host, write rule in fn `reconcile` | An exact fixpoint over the ownership tree. |
| `#loadScopes`, `#loadChain`, `#setEdge` | 588–629 | Load the owner chain from storage (task records and conversation records) into the edge and node caches, from a start up to its ownerless root. Terminal nodes go to `settled`. Setting a non-null edge records the owner in `conversationOwners`. `loadScopes(queued)` loads the chain of every live task. With `queued`, it also scans queued submissions (256 per page) and loads their conversations' chains, returning those conversation IDs. | service | scheduler host | Storage reads into a cache. |
| `#edge`, `#node`, `#above`, `#chainKnown`, `#liveRecords` | 632–680 | R6 walk primitives with candidate precedence. `#liveRecords` yields mirror records replaced by staged candidates (terminal candidates dropped), plus staged non-terminal records not in the mirror. | crisp | ownership helpers (`walkUp`, `chainKnown`, `candidates`) | Exact graph plumbing. |
| `#ownedLive` | 686–700 | R6 `ownedLive`. | crisp | ownership helpers | Exact walk. Encodes the "ordinary owned work" rule. |
| `#inScope` | 707–717 | R6 `inScope`. | crisp | ownership helpers | Exact walk. |
| `#belowCancelled` | 723–732 | R6 `belowCancelled`. | crisp | ownership helpers | Exact walk. |
| `#seal` | 735–745 | M7 close listener. | service | scheduler host | Close mechanics. |
| `#kick`, `#drain` | 747–768 | Pass loop. `kick`: set dirty; return if draining, disabled or closing; else set draining and queue `drain` as a microtask (never commit synchronously inside a listener). `drain`: while dirty and enabled and not closing: clear dirty, `#reserve`, `#start` each reservation. Report errors unless closing. Finally clear draining, and kick again if dirty (a wakeup during a failed pass still needs its pass). | service | scheduler host (natlang: EventLoop event plus `after`) | Loop and wakeup mechanics. |
| `#reserve` | 771–814 | R4 pass. One commit (nothing if disabled or closing): load chains; owned = `ownedLive`; registry snapshot taken lazily, once per pass; for each mirror record, `classifyTask` → skip, orphan, or reserve (migrate, write `running` if needed, create the invocation). If the commit fails, the created invocations are deleted and finished, and the error is rethrown. | service (commit, registration) + fn per task | calls fn `classifyTask` per live task; crisp `migrateRecord` | "What runs next" is policy. Reservation is mechanics. |
| `#waitingOn` | 820–824 | R4.3 waits-on set. | inline | in fn `classifyTask` | One rule with three cases. |
| `#resolve` | 827–843 | Reservation-time resolution: `fitDefinition`; when it fits and migrates, run `migrate` (R4 reservation). Missing `migrate` → error `Task ${kind} version ${v} has no migration from ${stored}`. Results are copied as strict JSON. | crisp (`migrateRecord`) + service (runs registered task code) | scheduler host | Version plumbing around task code. |
| `#fit` | 845–853 | R4a. | crisp | `fitDefinition` | Exact version comparison. |
| `inspect` | 860–871 | Read-only, on the line: load chains, `ownedLive`, then `classifyTask` (purpose inspect, no migration run) for each mirror record. `scheduling` = `closing` if closing, else `running` if enabled, else `paused`. | service (read) + fn per task | calls fn `classifyTask`; called by `Harness.inspect` | Reports exactly what the pass would do. |
| `#inspectTask` | 873–888 | R4 with inspection rule 5: invocation → running; completing → completing; waits-on non-empty → waiting{on}; blocked fit → blocked{reason, error?}; migrates without `migrate` → blocked migration_failed; else ready{migrates}. | fn | `classifyTask` (shared with `#reserve`) | One classifier keeps inspection and scheduling consistent. |
| `#createInvocation` | 890–906 | Creates the Invocation (M4) and registers it in the map, on the line. | service | scheduler host | In-memory execution. |
| `#start` | 908–922 | Runs `#run` or `#runAbort` detached. On throw → report. Finally: end the invocation, resolve `done`, kick a pass. | service | scheduler host | Execution mechanics. |
| `#run` | 925–945 | Phase driver: loop { `#step` with `decideStep`; stop or closing → return; reset the phase's agent resolution; call `phases[checkpoint.phase](current, runtime, invocationContext)`; record {checkpoint} or {checkpoint, failure} }. | service | scheduler host; phase handlers are task fns (generation/tool/compaction groups) | Open-ended host loop (M9). |
| `#decide` | 951–984 | R5, run mode. | fn | `decideStep` | Step precedence is scheduling policy. |
| `#runAbort` | 987–1001 | Abort driver: current = mirror record; missing or closing → return. Call `definition.abort(current, runtime, ctx)` once and catch its error. Then `#step` with the abort-mode decision (R5.3). | service + rule in fn | driver in host; rule in `decideStep` | The handler call is mechanics. Its follow-up is policy. |
| `#step` | 1009–1031 | One commit: current = the mirror record if `running`; decision = `decideStep` when running and not closing, else stop. Continue → return current. Otherwise end the invocation inside the commit; on fault → `holdOrTerminate` faulted{message}. If the commit throws, end the invocation and report unless closing. | service | scheduler host (M1) | Commit-ordering mechanics. Rules 1–2 are in `decideStep`. |
| `#terminate` | 1037–1045 | `holdOrTerminate` (R6). | crisp (+ cleanup service call) | `holdOrTerminate` | Exact hold rule. |
| `#commitState` | 1052–1069 | `commitTaskState` (R6). | crisp | `commitTaskState`, used by runtime `commit` | Exact hold rule on candidates. |
| `#validateWait` | 1075–1097 | R8 wait validation, with its four messages. | crisp | `validateWait` | Fixed checks and messages. |
| `#end` | 1100–1107 | M4 ending. | service | scheduler host | Cancellation context. |
| `#idle` | 1110–1116 | R9 `idle`. | crisp | ownership helpers | Exact walk. |
| runtime `agent` | 1122–1130, 1151–1152 | Rejects after end. Resolves the conversation's agent at most once per phase handler, at first use, with the phase's registry snapshot and the invocation context. The shared promise's failure is observed. Reset before each phase. | service | runtime API (agent group resolves) | Memoized host call. |
| runtime `hooks.each` | 1131–1144 | For each handler map from `agentHooks(agent, taskName)` (agent group), in order: if `map[name]` is a function, invoke it (bound). A throw is reported and the next handler runs. If the invocation is signalled, the throw propagates. | service | runtime API; hook semantics in the hooks group | Iteration over registered extension code, with an error-isolation rule. |
| runtime plain members | 1147–1160, 1270–1277 | taskId, conversationId, signal, models; `settings` resolved at each access; `registry` = the phase snapshot; `env(ctx)` → `HarnessOptions.env` through `buildEnv`; `now()` and `report()` throw `Task ${id} invocation has ended` after the end. | service | runtime API | Host accessors. |
| runtime `commit` | 1161–1169 | Gated commit (M1). The callback `(tx, current)` may return the next state, which `commitTaskState` applies in the same commit. | service | runtime API | Commit mechanics. |
| runtime `memo` | 1170–1185 | `memo(name)`: read the committed own-property memo of the mirror record. `memo(name, candidate)`: gated commit; an existing winner is returned; else write `memos[name] = candidate` and return it (first writer wins; memos vanish at completing/terminal). | service | runtime API | Durable first-writer-wins primitive. |
| runtime `sleep`, `#sleep`, `delay` | 1186, 1313–1324, 1410–1422 | Loop: throw if the invocation is signalled or the context cancelled; remaining = until − now; ≤ 0 → return; wait `min(remaining, MAX_TIMER_DELAY)`. The clock is rechecked after every timer. | service | runtime API (clock) | Timers. |
| runtime `watchDoc`, `#watchDoc` | 1187, 1326–1337 | Session `watchDoc`. If the invocation ended meanwhile: stop the watch and throw. Otherwise track it in `invocation.watches`, untracked once closed. | service | runtime API | Watchers are host infrastructure. |
| runtime reads: `snapshot`, `snapshotAsOf`, `getTask`, `entry`, `outcomes` | 1188–1241 | Committed reads; all reject after the end. `entry(token?, id)`: the entry visible from the task's conversation, returned with a token only when `kind` matches, else undefined. `outcomes(ids)`: on the line, each must be terminal (else `Task ${id} is not terminal`); returns their outcomes in order. | service | runtime API | Storage reads. |
| runtime `waitForTask` | 1200–1203 | Scheduler `waitForTask` with the invocation signal added, so it rejects when the invocation ends. | service | runtime API | Waiter. |
| runtime `conversation` | 1216–1228 | Harness factory: a bound handle with binding {invocation signal, check that throws after the end}, or undefined if the conversation is missing. | service | runtime API | Bound handle. |
| runtime `context` | 1242–1269 | `readContextFrom` (context group), starting from the conversation's cached range. Keeps the new range unless closing, ended, or a newer range is already kept: busy conversation → keep with idleSince undefined; idle and retention > 0 → keep, with idleSince carried over or now, and reschedule expiry. | service | runtime API (M8) | Cache around a context read. |
| `#read`, `#gated` | 1282–1310 | `#read`: rejects after the end. `#gated`: the gated commit rules and messages of M1; the commit carries {conversationId, taskId} scope (entries get `byTaskId`). | service | runtime API | Commit gating. |
| `cancellationIntent`, `failedOutcome`, `parentOf`, `nodeOf`, `overlayOf` | 1341–1367 | R6 definitions. `overlayOf` reads `tx.stagedTasks()` and `tx.stagedConversations()`. | crisp | ownership helpers | Exact predicates. |
| `memoOf` | 1370–1373 | Own-property memo only, so names like `toString` never resolve to inherited properties. | crisp | runtime `memo` | Exact lookup. |
| `withState` | 1396–1400 | Replaces the record's state. Removes `memos` when the new state is `completing` or `terminal`. | crisp | `commitTaskState`, `holdOrTerminate`, reservation | Exact record rule. |
| `canReserve` | 1403–1408 | Same version, or a newer definition with `migrate`. | crisp | fact for `decideStep` | Exact comparison. |
| `jsonEqual` | 1425–1435 | Structural JSON equality: object key order ignored, arrays ordered, primitives by `===`. | crisp | fact for `decideStep` (rule 7) | Exact comparison. |
| `missingMigration`, `endedError`, `sessionMethod`, `erased` | 1376–1393 | Error texts `Task ${kind} version ${v} has no migration from ${stored}` and `Task ${id} invocation has ended`. Overload forwarding and type erasure. | crisp | scheduler host | Messages and typing plumbing. |

**Notes.**

- **Callees in other groups:** `agentHooks` and the agent resolver (agent group); `readContextFrom` (context);
  `settleSchedulerOutcome` (live); the phase handlers of `pi.generation`, `pi.tool` and `pi.compaction` (generation,
  tool and compaction groups); session commit, transaction staging and storage (session/storage group).
- **Callers:** `harness.ts` only.
- **State:** the scheduler-host state record above, entirely in memory, rebuilt at open. Durable truth is the task
  records.
- **Facts passed in, not walked.** The fns `classifyTask`, `decideStep`, `reconcile`, `abortTask` and
  `abortConversation` cannot reach the ownership helpers from their own companion folders. So the host computes the
  facts with the crisp walks and passes them in as data. This keeps the walks in one place.
- **Owner decision: hot path.** `decideStep` runs before *every* phase of every task. `classifyTask` runs per live task
  on every pass. `reconcile` runs after most task commits. All three run *on the commit line*. As natural-language
  calls, model latency would serialize every commit of the Harness. They reduce to small decision tables once the facts
  are supplied. Choose between:
  - fn as CRITERIA says, deciding off the line on the mirror and committing with a recheck of the record;
  - crisp, with these rule texts as the spec. I would keep the admission, boundary and abort fns natural language
    either way.
- **Owner decision: migration.** Task-definition versioning and `migrate` assume task definitions are code objects
  with identity, because handover compares definition *objects*. natlang needs an equivalent definition identity, for
  example a content hash of the `.nl` source, for handover and for "retry a failed migration only under a different
  definition".

## durable/src/harness/harness.ts — the public Harness and Conversation surface over one Session: opening, conversation creation, and wiring of scheduler, submissions, views and task graph

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `SCAN_PAGE_SIZE`, `CreateTarget`, `CreateOptions`, `ConversationHost` | 57–81 | Page size 256. CreateTarget = root, independent{ownership}, or fork{parentId, at, ownership}. CreateOptions = {agent?, init?}. ConversationHost = {harness, storage, tasks, submissions, views, now, create()}. | crisp | state records | Data. |
| `ConversationImpl` state | 83–90 | Stateless handle {id, host}; compare by `id`. | crisp | state record | Handle. |
| `ConversationImpl.agent` | 92–94 | Resolve this conversation's `pi.agent` with the current registry snapshot and settings. Reads `{}` when absent; never writes. | service | host API (agent group does the resolution) | Delegation. |
| `ConversationImpl.configure` | 96–98 | `configure(tx, id, change)` in its own commit. | service | host API (agent group) | Delegation. |
| `ConversationImpl.submit` | 100–102 | `Submissions.submit(id, draft)`. | service | host API → fn `admitSubmission` | Wrapper. |
| `ConversationImpl.compact` | 104–108 | `resume()`, then one commit `createCompaction(tx, id, {reason: "manual", instructions?})`. Returns the task ID, not the summary. The task is conversation-owned and not background, so abort cancels it and idle includes it; it does not make the conversation busy. | crisp | host API → compaction group | Fixed data built around another group's creator. |
| `ConversationImpl.reset` | 110–117 | Submits a write draft `{kind: "pi.reset", head: "self", model?: [{role: "user", content: handoff, timestamp: now}]}`; `model` only when a handoff is given. Resolves after admission. While busy, it is placed at the next boundary, where it acts as R2 reset. | crisp | host API → fn `admitSubmission` | Builds a fixed draft. |
| `ConversationImpl.commit` | 119–121 | Session commit bound to this conversation; `tx.createTask` defaults to it. | service | host API | Commit line. |
| `ConversationImpl.context` | 123–125 | `readContext` for this conversation, optionally `at` an entry. | service | host API (context group) | Delegation. |
| `ConversationImpl.entries` | 127–140 | Scan entries on the line with `conversationId` forced to this conversation. Only `minEntryId`, `maxEntryId` and `order` are copied from the query. Paged. | service | host API | A storage scan with a binding rule. |
| `ConversationImpl.fork` | 142–144 | `#create({kind: "fork", parentId: id, at, ownership})` with `agent`/`init`. | service | host API | Creation. |
| `ConversationImpl.abort` | 146–149 | `resume()`, then `abortConversation(id, options.background === true)`. | service | host API → fn `abortConversation` | Wrapper. |
| `ConversationImpl.waitForIdle` | 151–154 | `resume()`, then `waitForIdle(id)`. | service | host API (R9) | Wrapper. |
| `ConversationImpl.viewState`, `watch` | 156–162 | Conversation view mount as a read-only state, or as an exact-frame watch. | service | host API (view.ts) | Observation. |
| `HarnessImpl` state + constructor | 166–212 | Composition root. It wires: the settings resolver; the scheduler (agent resolver per snapshot, env builder, clock default `Date.now`, `settleSchedulerOutcome`, `withdrawQueuedInputs`, a bound-handle factory that returns undefined for a missing conversation, a scheduler context without caller cancellation); `Submissions` (queue modes = resolved settings, `resume` = scheduler resume); `TaskGraphView`; `ConversationViews`. | service | host | Wiring. |
| `resolveAgent` | 215–223 | Registry snapshot (the given one or the current one) plus the committed `pi.agent` plus resolved settings → `resolveAgent(...)` (agent group); reports extension failures. | service | host; scheduler per phase | Delegation. |
| `buildEnv` | 226–231 | No `env` option → undefined. Else read `cwd` from `pi.agent` and call `env({conversationId, cwd?, read: harness}, ctx)` off the line, at each use. | service | host; runtime `env` | Environment factory. |
| `openTasks`, `close`, `beforeClose` | 234–236, 323–331 | `openTasks` → scheduler `open` (M5). `close`: set the closed flag, then Session close. `beforeClose` (after the seal, before storage closes) joins all invocations; it writes no outcome. | service | host | Lifecycle. |
| `resume` | 238–241 | Throws `Harness is closed` after close; else enables scheduling (R9). | service | host API | State toggle. |
| `getTask`, `submission` | 243–261 | Committed task record on the line (does not enable scheduling). `submission(id)` → handle or undefined (reacquire after reopen). | service | host API | Reads. |
| `abortSubmission`, `abortTask` | 263–273 | Delegate to `Submissions.abort` / scheduler `abort`. | service | host API → fns `abortSubmission`, `abortTask` | Wrappers. |
| `waitForTask`, `waitForIdle` | 275–283 | `resume()`, then wait. Harness idle = the roots scope (R9). | service | host API | Wrappers. |
| `inspect` | 249–257 | On the line: scheduler `inspect(current snapshot)`, plus queued and placed submissions (two scans, 256 per page), merged and sorted by ID → `{scheduling, tasks, submissions}`. Writes nothing; does not enable scheduling. | service | host API (+ fn `classifyTask`) | Read-only projection. |
| `usage` | 286–296 | Scan every conversation (256 per page) on the line, then read each `pi.usage` snapshot (each at its own point) and sum them into the UsageDoc initial value with `addUsageState` (usage group). Totals only grow. | crisp (sum) over service reads | host API | Arithmetic over reads. |
| `taskGraph`, `watchTaskGraph` | 298–304 | Task graph mount as a read-only state, or as an exact-frame watch. | service | host API (task-graph.ts) | Observation. |
| `root`, `conversation`, `createConversation` | 306–321 | `root(ctx, {agent, init})` = `#create(root)`. `conversation(id)`: throws if closed; reads the record → handle or undefined. `createConversation({ownership, agent, init})` = `#create(independent)`. Ownership is always explicit; it is never inferred. | service | host API | Creation and lookup. |
| `#create` | 333–350 | Throws if closed. One commit: if the target is root and conversation 1 (`ROOT_CONVERSATION_ID`) exists → return 1, ignoring `agent`/`init`, with no write. Else create the root, fork `(parentId, at, {ownership})`, or create `({ownership})`; the creation hook runs inside that call. Then `configure(agent)` if given, then `init(tx, id)` if given. Returns a handle. | service | host | Atomic creation (the root-idempotence rule is part of it). |
| `conversationCreated` | 357–364 | Creation hook in every commit that creates or forks a conversation (raw `tx.createConversation` too). It creates `pi.live` `{}`, `pi.inbox` `{items: []}`, `pi.usage`, and `pi.provider` (fresh UUIDv7). Then `createAgent(tx, record)`: a fork keeps the asOf copy of its parent's agent; a task-owned conversation copies its owner conversation's stored agent; an ownerless one gets `{}` (agent group). Then `HarnessOptions.conversationCreated`. | crisp | host (document creation via services) | Fixed initialization order. |
| `#assertOpen` | 366–368 | Throws `Harness is closed`. | crisp | host | Guard. |
| `boundConversation` | 375–406 | Invocation-bound `ConversationHandle`. Every operation first checks the binding (throws `Task ${id} invocation has ended`) and runs under the invocation's signal. `submit` (input drafts only) returns a submission whose `status`/`wait`/`abort` are bound too. `abort` → `abortConversation`. `waitForIdle`. Admitted work stays durable after the invocation ends. | service | runtime `conversation`, tool API | Cancellation contexts are host infrastructure. |
| `Harness.open` | 411–437 | Throws if the context is already aborted. Every built-in task must be in the registry snapshot, else `Registry lacks built-in tasks ${names}; create it with createRegistry()`. Construct, then `openTasks`. On failure: close without the caller's signal (a close error goes to `onReport`) and rethrow the open error. | service | host entry point | Lifecycle. |

**Notes.**

- **This file is almost entirely host service.** The public surface is the natlang host API that applications call.
  Its behaviour comes from the fns it delegates to (`admitSubmission`, `abortTask`, `abortConversation`,
  `abortSubmission`, `classifyTask`) and from other groups: agent (`resolveAgent`, `configure`, `createAgent`),
  compaction (`createCompaction`), context (`readContext`), usage (`addUsageState`), live (`settleSchedulerOutcome`).
- **The only harness-level policies here** are: the progress-enabling rule (R9); root idempotence; explicit ownership;
  and the creation-hook order (built-in documents, then the agent copy, then the host hook, then the convenience
  `agent`, then `init`).
- **State:** a closed flag plus the composed services.

## durable/src/harness/submissions.ts — admission, waits, and withdrawal of durable submissions; decides what a busy or idle conversation does with new input

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `AbortResult`, `Submissions` state + constructor | 20–51 | State: {session, storage, clock, queueModes getter (read at each admission, on the line), resume, waiters by submission ID, closed}. Subscribes to commits (resolves waiters of submissions that became `done`/`unanswered`) and to close (closed, reject all `Harness is closed`). | service | submissions host | Waiters and wiring. |
| `submit` | 54–61 | `resume()`; one commit `admitSubmission(tx, conversationId, draft, now(), queueModes())` → handle. Returns after durable admission, not settlement. | service | host API → fn `admitSubmission` | Commit wrapper. |
| `get`, `status` | 64–73 | On the line: `get` → handle or undefined. `status` → record, or reject `Submission ${id} does not exist`. | service | host API | Reads. |
| `wait` | 75–87 | `resume()`; on the line: missing → reject `Submission ${id} does not exist`; settled → resolve at once (even while closing); closed → reject `Harness is closed`; else register a waiter. Cancelling the context cancels only the wait; it never withdraws. An input settles at its answer or terminal failure; a write settles when placed or when it becomes unplaceable. | service | host API | Waiter. |
| `abort` | 90–103 | R7 `abortSubmission`, in one commit. | fn | `abortSubmission`; called by `Harness.abortSubmission`, `SubmissionHandle.abort` | Withdrawal is submission policy with its own commit and contract. |
| `#observe` | 105–110 | Resolve waiters for every submission change whose status is `done` or `unanswered`. | service | submissions host | Listener. |
| `SubmissionHandle` | 113–135 | {id}; `status` and `wait` delegate. `abort` maps `not_found` to the rejection `Submission ${id} does not exist`. | service | host API | Handle. |
| `isSettled` | 137–139 | Status `done` or `unanswered`. | inline | `wait`, `#observe` | One comparison. |
| `admitSubmission` | 148–207 | R1. | fn | `admitSubmission`; called by `submit`, compaction summary placement (compaction group), `reset` | The central "busy conversation" policy. Its own commit, retried exactly once per `requestId`. |

**Notes.**

- **Layout (owner decision).** `admitSubmission` has callers in two modules (host API, compaction). It calls
  `applyBoundary` (inbox) and `startRun` (generation group). `applyBoundary` itself is also called by generation's
  `answer` and `finishToolRound`. A natlang fn can only call its companion folder, so these need one of:
  - (a) a shared "conversation control" service that the host exposes by invoking the fns (one source; my
    recommendation);
  - (b) copies in each companion folder (drift risk);
  - (c) placing generation's boundary commits in the same folder.
- **`startRun`, `endRun` and `handOver`** (generation and live groups) are one-line data writes. They belong with
  whichever option is chosen.
- **`requestId` dedupe** could ride on EventLoop event IDs only partly (M9).

## durable/src/harness/inbox.ts — the per-conversation queue of submissions (`pi.inbox`) and the turn-boundary selection that places queued items

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `InboxItem`, `InboxState` | 8–14 | Item = {id, mode: steer or followUp, content (UserInput as JSON)} or {id, mode: write, entry (EntryDraft as plain JSON)}. State = {items}, in ID order. | crisp | state record | Data. |
| `InboxDoc` | 16–24 | Document definition: kind `pi.inbox`, version 1, conversation scope, history `latest`, fork `initial` (a fork starts with an empty inbox), `initial()` = `{items: []}`, a complete base whenever `items` is empty. | service | document definition (document service) | Chord storage policy. |
| `QueueModes`, `Boundary`, `BoundaryResult` | 27–40 | QueueModes = {steeringMode, followUpMode}, each `all` or `one-at-a-time`. Boundary = {conversationId, inbox draft, steeringMode, followUpMode, activeStart (mutable)}. BoundaryResult = {users: placed input IDs, reset}. | crisp | records passed to `applyBoundary` | Data. |
| `prepareBoundary` | 46–56 | Reads before the commit's first table write: the newest head marker's `head` (activeStart, may be absent) and the inbox; takes the queue modes as they are now, on the line. | crisp | caller-side helper for `admitSubmission` and generation | Read ordering is commit mechanics. Its output is data. |
| `applyBoundary` | 65–106 | R2. | fn | `applyBoundary`; called by `admitSubmission`, generation `answer`, `finishToolRound` | Steering/follow-up selection and placement order are core turn policy. |
| `isStale` | 109–111 | `typeof entry.head === "number"` and activeStart known and `entry.head < activeStart`. | inline | in `applyBoundary` and `admitSubmission` (one sentence each) | One comparison. |
| `removeInboxItem` | 114–118 | Remove the item with this ID if present; the caller settles the submission. | crisp | `abortSubmission` | Exact list edit. |
| `withdrawQueuedInputs` | 124–132 | R7w: from the end backwards, every non-write item settles `unanswered{reason: "aborted"}` and is removed; writes stay. | crisp | `reconcile` and `abortConversation` writes (scheduler host), passed to the scheduler as `withdrawInputs` | An exact write used by two fns. |

**Notes.**

- **Positional removal.** Spec §6 requires it: selected and stale items are removed positionally, and the remaining
  order is preserved. Chord must express scattered removals without carrying the retained values. A natlang document
  service needs the same diff behaviour.
- **Applying a boundary is an action.** `applyBoundary` writes entries and placements through the tx service. The fn
  performs the boundary; it does not only plan it.
- **Callers that decide what happens after a boundary.** These are in generation (R3): continuation, steer joining,
  reset-ends-run, and starting a successor run. The generation group must use R2's result exactly as R3 describes.

## durable/src/harness/task-graph.ts — the structural view of every live task (status, phase, wait, owned conversations) for UIs and debugging (spec §9.5)

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `TaskGraphState`, `TaskGraphNode`, `TaskGraph`, `TaskGraphWatch` | 19–49 | State = {status: pending or running, phase}, or {waiting, phase, on, policy}, or {completing, outcome status}. Node = {id, kind, conversationId, owner?, background, abortRequested, state, conversations (owned, ascending)}. Graph = {tasks keyed by decimal ID}. Watch = an exact-frame watch handle. | crisp | state records (public protocol) | Data. |
| `TaskGraphView` state + constructor | 62–79 | At most one mount {value, observers}. Advances the mount from each commit publication. On close: closed, end every observer, drop the mount. | service | observation service | Chord observation is host infrastructure. |
| `state`, `watch`, `#attach` | 82–129 | On the line: build the mount if absent, then create the observer from the current value. If closed → `Harness is closed`; if the context is aborted → throw its reason. Register. Detach removes the observer, and drops the mount with the last one. `state` → a read-only replicated state. `watch` → an exact-frame watch that stops when its context is cancelled. Neither enables scheduling. | service | host API `taskGraph` / `watchTaskGraph` | Observation. |
| `#build` | 131–148 | Scan the four live statuses (256 per page), sort by ID; for each task, scan the conversations with `ownerTaskId` = its ID, sort their IDs → node. Holds the line while it runs. | service | observation service | Storage scans. |
| `advance` | 152–190 | Per publication. Task changes: terminal → delete `tasks[id]` if present; else build a node with the previous `conversations`, and set it if its JSON differs. Then conversation changes that have an owner whose node exists: append, sort, set `tasks[id].conversations`. Apply the ops and send each observer one batch; no ops → no revision. | crisp | inside the observation service | Exact projection. |
| `nodeOf`, `stateOf`, `phaseOf` | 192–222 | Record → node without input, checkpoint payload, outcome payload or memos. `phase` = `checkpoint.phase`. Completing (and terminal, never reached) → {completing, outcome: outcome.status}. | crisp | inside the observation service | Exact projection. |

**Notes.**

- **Read-only.** The task graph is a read-only consumer. The coding agent's durable runtime uses it (`runtime.ts`).
- **Ports with the document/observation service**, whatever represents Chord states and watches in natlang. It has no
  natural-language part.
- **Caveats.** `running` in the graph is the durable status, not "has an invocation". Blocked state and live members
  of `on` come only from `inspect()`.

## durable/src/harness/events.ts — experimental adapter translating one conversation's committed publications into coding-agent-style events (spec §9.4)

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `MessageChange`, `SnapshotEvent`, `AgentEvent`, `AgentEventStream`, `Block`, `QueuedItem` | 24–98 | Event protocol: snapshot {entries, run?, generation?, tools, compactions, inbox ids and modes, agent, usage}; run_start/run_end{inputs}; turn_start/turn_end; message_start/update/end; tool_execution_start/update/end; inbox_update; submission; auto_retry_start/end; deferred_poll; entry_appended; agent_changed; usage_changed; task_failed; compaction_start/end. A stream has a snapshot, `start(listener)`, `stop()`, and `closed`. | out | — | Experimental, and its protocol may change. The coding agent's durable frontend renders `ConversationView` and never uses it. |
| `parts`, `snapshotOf`, `queued` | 101–134 | Read `pi.live`, `pi.inbox`, `pi.agent` and `pi.usage` from a view. Snapshot: absent run/generation omitted; tools and compactions default to `[]`; agent default `{}`; usage default initial; inbox → [{id, mode}]. | out | — | As above. |
| `watchEvents` | 140–182 | Attaches to the conversation's view mount on the line. Scans this conversation's `pi.generation` tasks that are `completing` (page 100) into a `held` set. Builds the snapshot. Creates a watch whose overflow (more than 100 undelivered batches) replaces everything with one snapshot of the newest view. Each publication → `translate` → one batch. The acquisition context governs the lifetime. | out | — | As above. |
| `translate` | 195–347 | One batch per publication touching the conversation, in spec §9.4 order: (1) `tool_execution_start` for slots newly running (args from the tool task's checkpoint `arguments` in this commit, else `{}`); (2) `message_start` of the first partial, or `message_update` {usage, changes}; (3) `tool_execution_update` for still-running slots; (4) `auto_retry_start`/`auto_retry_end`/`deferred_poll`; (5) entries in append order, each preceded by its tool ends (a slot became done; created done; or vanished, with the result entry of this commit), then `entry_appended` (no messages) or `message_start` (skipped for the first assistant entry after a streamed partial) + `message_end`; (6) tool ends without an entry; (7) `compaction_end`, `task_failed` (faulted message or orphaned reason), one `turn_end` (a generation newly completing, or terminal without an earlier hold), `run_end` (run removed, or its first input changed); (8) `submission` records in ID order; (9) `inbox_update`, `agent_changed`, `usage_changed` when the document object changed; (10) `compaction_start`, `run_start`, and `turn_start` (the run's task changed to a `pi.generation` changed in this commit). | out | — | As above. If ever needed (a JSON print mode), port it as crisp, never natural language. |
| `resultOf`, `messageChanges`, `toolUpdate`, `startsWith`, `PARTIAL_PATH` | 187–425 | `messageChanges`: ops on `docs.pi.live.generation.message` → splice into `content` = *_start per block; append to a block's text/thinking = a delta; append inside `arguments` = `toolcall_delta` with the relative path; other change in a block = `block` (once per batch); other change, or replacement of the whole message or generation = `message`; usage-only change = no changes. `toolUpdate`: trims and appends of `tools[i].output` (a set op, or a changed value with neither → `{set}`), plus details (null when removed) and diagnostics (`[]` when removed). | out | — | As above. |

**Notes.**

- **Why out.** natlang applications follow the Elm-style model: they render from state (`ConversationView` through an
  EventLoop view), not from event streams. Print mode awaits its own `Submission`.
- **If the owner wants the adapter after all,** for example to feed the existing coding-agent JSON mode, it is a pure
  deterministic projection of view diffs. It would be crisp in the observation service.

## durable/src/harness/output.ts — bounded retention of running tool output and adaptive pacing of progress commits

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `OutputLimits`, `BoundedOutput`, `OutputSlice` | 5–16 | Limits = {maxBytes, maxLines, retain: head or tail}. Bounded = {text, droppedBytes, droppedLines}. Slice adds `bytes`. | crisp | state records | Data. |
| `sanitizeOutput` | 19, 25–27 | Removes `/[\x00-\x08\x0b-\x1f￹-￻]/g`: control characters including CR and VT/FF; tab and LF stay; U+FFF9–FFFB are removed. | crisp | tool group (`OutputBuffer.snapshot`) | Exact filter. |
| `boundOutput` | 34–44 | Text → UTF-8 bytes. Head or tail range, decoded with the BOM kept as text. Returns the same string when nothing was dropped. Also returns `bytes`, `droppedBytes`, and `droppedLines` = lines(all) − lines(kept). | crisp | tool group (results, snapshot) | Byte/line truncation (CRITERIA example of crisp). |
| `headRange` | 46–61 | maxLines or maxBytes 0 → empty. Else end after the maxLines-th newline (or everything). If that is over maxBytes: cut after the last newline before maxBytes; with no newline, cut at maxBytes on a character boundary. | crisp | `boundOutput` | Exact. |
| `tailRange` | 63–84 | Either limit 0 → empty at the end. A trailing newline ends the last line. Walk back to the start of the maxLines-th last line. If more than maxBytes remain: the first line starting inside the last maxBytes bytes, or, when the last line alone is too long, a cut on a character boundary at len − maxBytes. | crisp | `boundOutput` | Exact. |
| `characterEnd`, `characterStart`, `lineCount`, `lines`, `countNewlines` | 87–105, 250–258 | Step back or forward over UTF-8 continuation bytes (`0b10xxxxxx`). Line count = newlines, plus 1 when the text is non-empty and does not end in a newline. | crisp | output helpers; `characterEnd` is also used by the `read` tool | Exact. |
| `OutputBuffer` | 112–227 | State record: limits, a streaming UTF-8 decoder, started flag, stored chunks {text, bytes, newlines}, stored totals, a `full` flag (head), stream totals, endsWithNewline. **`push(chunk, skipped?)`**: a string chunk or a skip first flushes an incomplete earlier character as U+FFFD; only a BOM at the very start of byte output is dropped; a skip requires tail retention (else throws `Skipped output requires tail retention`) and clears the stored text but counts its bytes and newlines. **`#accept`**: head stops storing once stored bytes exceed maxBytes or newlines reach maxLines; tail drops leading chunks while the rest still exceeds maxBytes+1 bytes or maxLines+1 newlines. **`end()`** flushes the decoder. **`snapshot()`** → {sanitized bounded text, droppedBytes = total − kept, droppedLines = total lines − kept lines}, and compacts tail storage to `tailMargin`. | crisp | tool group (tool output API) | Exact streaming truncation. |
| `tailMargin` | 233–247 | The shortest suffix with more than maxBytes bytes or more than maxLines newlines (or all of the text), so a later tail window can still find its first line. | crisp | `OutputBuffer` | Exact. |
| `PROGRESS_BYTES_PER_SECOND`, `Progress` | 261–341 | 100 KiB/s. Adaptive commit pacing: the first change after an idle period commits at once; each commit delays the next by `max(minIntervalMs, bytes·1000/102400)` ms from its start. One commit in flight; changes meanwhile coalesce. `markAndWait` settles with the commit that includes the change. On a write error: next delay = minIntervalMs, waiters reject, `onError`. `stop()`: no more commits, await the one in flight, return the pending waiters for the caller's final commit. Default intervals 100 ms (`ProgressPolicy`). | service | tool group (running output / details commits) | Timer-driven commit throttling is durability mechanics. |

**Notes.**

- **Caller.** This file belongs to the tool-execution group's caller (`tool.ts`). It is listed here only because it is
  in this group's file list.
- **Port as is.** Everything except `Progress` is crisp and can be ported verbatim. `Progress` becomes part of the
  host's progress-commit service.
- **What an nl tool function sees.** It only calls `api.output(chunk)`. Retention is the host's job.

## durable/src/harness/view.ts — the per-conversation structural view mount (active entries plus built-in documents) as a Chord state and watch (spec §9.3)

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `ConversationView`, `ViewObserver`, `MOUNTED`, `Mount`, `VIEWS` | 26–65 | View = {conversation record, entries (head marker, then non-head entries from its head), docs keyed by kind: `pi.agent`, `pi.live`, `pi.inbox`, `pi.provider`, `pi.usage`; absent documents are absent}. Observer = {advance?, publication?, closeSession}. Mount = {value, document incarnation and version per kind, observers}. | crisp (view record) / service (mount) | public protocol; observation service | Data and host state. |
| `conversationViews` | 68–72 | The Harness's mounts via a WeakMap; otherwise `Not a Harness`. | service | `events.ts` | Lookup. |
| `ConversationViews` constructor | 84–97 | Registers; each publication advances every mount; close ends observers and clears mounts. | service | observation service | Listener. |
| `state`, `watch`, `attach` | 100–154 | `attach`, on the line: the mount (built if absent) → observer from the current value (the creator may read storage, still on the line). Closed → `Harness is closed`; context aborted → throw. Register; detach drops the mount with its last observer. `state` → read-only replicated state; `watch` → exact-frame watch (100 pending frames, overflow replaced by the full value) bound to its context. | service | host API `viewState` / `watch` | Observation. |
| `#build` | 156–170 | Conversation record (missing → `Conversation ${id} does not exist`); capture context bounds and active entries (context group); load each mounted document on the line, recording incarnation ID and version. | service | observation service | Reads. |
| `advance` | 174–223 | Per publication, for this conversation. **Entries** (ID order): no head → splice at the end; head marker H → entries become `[H, ...entries from the first non-head entry with id ≥ H.head]` (one splice at the front). **Documents** of the mounted kinds without a key: retirement of the mounted incarnation → delete `docs[kind]`; same incarnation and version → its ops prefixed with `["docs", kind]`; otherwise → set the whole value. Ops = document ops then entry ops. If any: apply, then `observer.advance`. Every observer always gets `publication(before, after, ops)`. | crisp | inside the observation service | Exact projection. |
| `prefixed` | 226–244 | Moves a Chord op under a prefix; a root replace (`r`) becomes a set (`s`) of the prefix. The op kinds are p, m, s, d, a, t. | crisp | `advance` | Exact. |

**Notes.**

- **The view is the primary UI surface.** It is what the coding agent's durable TUI and runtime render, and it is what
  an Elm-style natlang app would hold as state.
- **It is pure host observation.** That means Chord-equivalent replicated state, exact frames, and overflow
  replacement.
- **Dependency.** `captureContextBounds` and `activeEntries` (context group).
- **Caveat.** A raw head write that targets behind the mounted range shows only the mounted entries (spec §12).

## durable/src/harness/util.ts — keyed waiters, paginated scans, and the closed error shared by the Harness modules

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `Waiters` | 5–42 | Keyed pending waits. `add(key, ctx)`: an already-aborted context → reject with its reason; otherwise register; on context abort, remove and reject with the reason. `resolve(key, value)` settles and removes every waiter of the key. `rejectAll(error)`. `keys()`. Each waiter settles once. | service | scheduler, submissions | Promise/cancellation plumbing. |
| `scanAll` | 45–54 | Concatenates pages until `next` is undefined. | crisp | scheduler, harness, task graph, events | Pagination. |
| `closedError` | 56–58 | `Error("Harness is closed")`. | crisp | everywhere | Fixed message. |

**Notes.**

- **Waiters map to host awaitables.** In natlang a wait is a service call that resolves with the record. Cancelling the
  caller's call cancels only that wait, never the work.

## durable/src/harness/json.ts — leaf-wise assignment into Chord drafts so streaming writes publish appends, not whole values

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `assignJson`, `isRecord` | 10–31 | Assign `value` at `target[key]` leaf by leaf. Record onto record: delete keys absent from value, recurse per key. Array onto an array no longer than value: recurse per index, push extras. Else set only if different (`!==`). This lets Chord record a longer string as an append instead of a full set. | crisp | generation group (partial message), tool group (slot details) | Draft write-amplification plumbing; belongs with the document service. |

**Notes.**

- **Only needed if the natlang document service diffs drafts like Chord.** A service that diffs whole values itself
  makes it unnecessary. No natural-language part.

## durable/src/harness/types.ts — the public types of the Harness: submissions, tasks, tools, extensions, registry, agent, settings, inspection, conversation and Harness interfaces

| unit | lines | what it holds | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `ModelRef` | 47–50 | {provider, modelId}, resolved through pi-ai `Models`. | crisp | state record | Data. |
| `UserInput` | 52 | pi-ai `UserMessage["content"]` (string or content blocks). | crisp | state record | Data. |
| `SubmissionDraft`, `InputSubmissionDraft` | 55–72 | {requestId?} plus either input {content, whenBusy?: steer / followUp / reject} or write {entry: EntryDraft}. Input-only variant for bound handles. | crisp | `admitSubmission` input | Data. |
| `SettledSubmissionRecord` | 74–76 | A SubmissionRecord with status `done` or `unanswered`. | crisp | state record | Data. |
| `Submission` | 79–84 | Handle interface: id, `status()`, `wait()`, `abort()` → aborted / already_placed / settled. | service | host API | Interface. |
| `SettledTask` | 86–88 | A TaskRecord whose state is terminal (the durable result receipt). | crisp | state record | Data. |
| `AnyTask` | 91–101 | Erased task definition {name, version, initial, phases (by phase name), abort, migrate?, hooks?}. | crisp | registry record (definitions are task fns from other groups) | Data shape. |
| `ConversationAbortOptions` | 104–111 | {background?}: cross background boundaries (R7). | crisp | `abortConversation` input | Data. |
| `HooksOf` | 114 | Type-level extraction of a task's hook map. | out | — | TypeScript typing only. |
| `ConversationHandle` | 120–127 | Invocation-bound interface: id, `submit(input draft)`, `abort(options?)`, `waitForIdle()`. | service | runtime/tool API | Interface. |
| `ToolControl` | 130–134 | {addTools?: names, terminate?: true, handoff?: text}, the post-tools controls of a tool result (R3). | crisp | state record (tool/generation groups) | Data. |
| `ToolDiagnostic` | 137–141 | {severity info / warn / error, message, code?}. | crisp | state record | Data. |
| `ToolExecutionResult` | 143–154 | {content?, isError?, details?, diagnostics?, usage?, control?}. Omitted content means the retained output; omitted details means the last `details()`. | crisp | tool group | Data. |
| `ToolExecutionMode`, `QueueMode` | 157–160 | `parallel` or `sequential`; `all` or `one-at-a-time`. | crisp | settings | Enums. |
| `ToolExecutionApi` | 166–206 | Per-call tool API: ids, registry snapshot, `agent()`, models, env, `output(chunk, skipped?)`, `outputWindow`, `diagnostic()`, `details()`, `commit()`, `memo()`, `createTask()`, `getTask()`, `waitForTask()`, `conversation()`, plus document reads and watches. Every operation rejects after the invocation ends. | service | tool group | Interface. |
| `ToolRegistration` | 212–236 | pi-ai Tool plus replay safe/unsafe (default unsafe), executionMode, `prepareArguments` (pure), outputLimits {maxBytes, maxLines, retain}, `execute(args, api, ctx)`. | crisp (record) | tool/extension groups | Data plus a fn reference. |
| `PromptInput`, `PromptSection` | 239–257 | Section render input {conversationId, agent, env, shown sections, read}. Section {key, render(), tag? (default true: wrap the text in a tag named after `key`)}. | crisp | prompt group | Data plus a fn reference. |
| `HookRegistration`, `Wrap`, `Extension` | 260–277 | Hook {task name, handlers}. Wrap {tool name or section key, pure `wrap`}. Extension {name, tools?, sections?, hooks?, wraps?, tasks?}. | crisp | registry/extension group | Data. |
| `RegistrySnapshot`, `RegistryReader`, `Registry` | 280–305 | Immutable snapshot: installed(), extension(name), tools(), sections(), tasks(), task(name). Reader: snapshot() and subscribe() (wakes the scheduler). Registry: install (replace in place by name) and uninstall. | service | registry service | Process-local code registry. |
| `AgentState`, `AgentChange`, `Agent` | 308–345 | Stored choices {model?, thinkingLevel?, extensions? (array or {add, remove}), tools? (array or {remove}), instructions?, cwd?}. A change: a given field replaces, `null` clears, `undefined` leaves. Resolved agent {model?, thinkingLevel, extensions, tools, sections, instructions?, cwd?}. | crisp | agent group | Data. |
| `ConversationInit`, `ConversationCreateOptions` | 351–358 | `init(tx, id)`, run in the creating commit after the hook and agent change (table reads throw). Options {ownership, agent?, init?}. | crisp | host API | Data plus a callback. |
| `ConversationStreamOptions`, `ConversationRetryPolicy`, `CompactionPolicy`, `ProgressPolicy` | 361–402 | Stream {transport?, timeoutMs?, maxRetries?, maxRetryDelayMs?, headers?, metadata?, cacheRetention?, deferred?}. Retry defaults {enabled true, maxRetries 3, baseDelayMs 2000, maxAgentDelayMs 60000}. Compaction defaults {enabled true, reserveTokens 16384, keepRecentTokens 20000, backgroundTokens 32768}. Progress defaults {partialIntervalMs 100, outputIntervalMs 100}. | crisp | settings records (generation/compaction/tool groups) | Data. |
| `CompactionReason`, `CompactionResult` | 405–411 | manual / threshold / overflow. Result {entryId?, submissionId?}. | crisp | compaction group | Data. |
| `HarnessSettings`, `Settings` | 414–443 | Host settings (all optional, synchronous getters allowed, never copied). Resolved settings: defaults as above, plus `toolExecution` parallel, `steeringMode` one-at-a-time, `followUpMode` one-at-a-time, `contextRetentionMs` 600000; object fields merged key by key; read anew for each decision (queue modes on the line). | crisp | settings record | Data. |
| `EnvTarget`, `HarnessOptions` | 446–469 | Env target {conversationId, cwd?, read}. Options {models, registry, settings?, env?, conversationCreated?, now?, onReport?}. | crisp | host composition | Data plus callbacks. |
| `TaskInspection`, `HarnessInspection` | 472–499 | Per live task {record, state: running / ready{migrates} / waiting{on} / completing / blocked{reason, error?}}. Inspection {scheduling: paused / running / closing, tasks, submissions (queued and placed, in ID order)}. | crisp | output of `classifyTask` / `inspect` | Data. |
| `ContextView` | 502–511 | {head marker?, raw active entries, per-entry contributions, model messages}. | crisp | context group | Data. |
| `Conversation`, `ConversationWatch` | 514–569 | Handle interface (§2.2): agent, configure, submit, reset, compact, commit, context, entries, fork, abort, waitForIdle, viewState, watch. | service | host API | Interface. |
| `Harness` | 572–614 | Session plus resume, root, conversation, createConversation, getTask, inspect, submission, abortSubmission, abortTask, waitForTask, waitForIdle, usage, taskGraph, watchTaskGraph. | service | host API | Interface. |
| `HookApi`, `HookResult` | 617–626 | What a hook can use: ids, models, committed reads, the asking task's memos. Result: a value, undefined, or a promise. | service (api) / crisp (result) | hooks group | Interface. |
| `GenerationHooks`, `ToolHooks`, `CompactionHooks` | 629–679 | Hook contracts. `beforeRequest(messages)` → replacement messages for that request only. `afterResponse(message)`. `onYield(answer)` → {continue: UserInput}; the first one wins. `afterTools(assistant, results)`. `beforeTool(call)` → {arguments?, block?}; the first block wins, and a throw blocks. `afterTool(call, result)` → a replacement result. `beforeCompact({reason, entries, messages, firstKept, instructions?})` → decline, or {summary}; the first decision wins. | crisp (contracts) | hooks group; called through runtime `hooks.each` | Signatures of extension hook fns. Their semantics belong to the hooks group. |

**Notes.**

- **What this file is.** Only state records and service interfaces. The public protocol records (`SubmissionDraft`,
  `TaskInspection`, `HarnessInspection`, `Settings`, `AgentState`) are the typed contracts of the natlang fns and the
  host API.
- **Settings are read live**, so policy fns must receive the resolved value for each decision. Queue modes are read at
  the moment the boundary is decided.

## Owner decisions

1. **Hot-path scheduler decisions** (`decideStep`, `classifyTask`, `reconcile`) run on the commit line, before every
   phase and in every pass. Natural language (as CRITERIA says, deciding off the line and committing with a recheck)
   or crisp with these rule texts?
2. **Shared conversation-control fns.** `admitSubmission` and `applyBoundary`, together with generation's
   `startRun`/`endRun`/`handOver`, are called from several modules. Expose them as one host-provided
   "conversation control" service, copy them per folder, or co-locate generation's boundary commits?
3. **Ownership walks** (`ownedLive`, `inScope`, `belowCancelled`, `idle`, `finalizable`) are classified crisp,
   although they encode spec §5.4/§5.5 policy. Confirm.
4. **The experimental agent-event adapter (`events.ts`) is classified out.** Confirm, or keep it as a crisp projection
   for a JSON/print mode.
5. **Task-definition identity and migration.** Handover and failed-migration retry compare definition *objects*.
   natlang needs a definition identity (for example a hash of the `.nl` source and version) and a `migrate` form.
6. **The context cache (M8)** is a performance cache. Keep it as a host service, or drop it?
