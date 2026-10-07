# Group 3: session, records and storage (pi-durable)

Source: `/home/werg/src/pi/packages/durable/src` at `f10993b`. Spec: `packages/durable/docs/spec.md` §1–4, §9.1–9.2,
§10, §11 (plus §2.2 and §8.1 where they define what a record means). Paths below are relative to `packages/`.

**Overall split.** Nearly all of this group is durability machinery. It becomes **one host service, `durable`**
(the Session kernel plus one Storage backend), described in "The `durable` service interface" below. The record
shapes become a crisp `records` type module that natlang signatures use. The parts that are *harness semantics*
(what entry kinds mean, how a model's transcript is derived from entries, what a fork inherits, document
policies, submission and task record lifecycles) are spelled out as exact rules in "Harness semantics carried by
these records" (H1–H8). The natural-language functions of the other groups (generation, tools, compaction,
submissions, scheduler) must carry these rules in their instructions or call crisp helpers that implement them.

---

## durable/src/types.ts — record shapes, the transaction surface, and the Session and Storage interfaces everything in pi-durable is written against

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `JsonObject`, ID brands (`ConversationId`, `EntryId`, `TaskId<R>`, `SubmissionId`, `DocumentId`), `Seq` | 14–36 | All IDs are plain safe integers from **one Session-global namespace** shared by every record kind (a number belongs to at most one table). Brands are compile-time only. `Seq` numbers one atomic commit; it strictly increases, gaps allowed, and is a separate type so an entity ID is never used as a document point in time. | crisp | `records` type module; used in every natlang signature | type declarations only |
| `ROOT_CONVERSATION_ID` | 38–39 | The root conversation always has ID `1`. Allocators therefore start minting at `2` (memory `nextId = 2`, SQLite `next_id = '2'`). | crisp | `records` constant; read by the harness `root()` (other group) and the commit service | fixed protocol rule |
| Document semantics types (`LatestConversationSemantics`, `RewindableConversationSemantics`, `DocumentSemantics`) | 41–60 | `scope` is `session`, `conversation` or `task`. Only conversation documents declare `history` (`latest` keeps current state only; `rewindable` keeps history for as-of reads) and `fork` (`current` or `initial` for latest; `asOf`, `current` or `initial` for rewindable — `asOf` requires `rewindable`). | crisp | `records` | declarative policy data; meaning in H4/H5 |
| Document definition types (`CheckpointInfo`, `CommonDocDefinition`, `DocDefinition`, `DocFamilyDefinition`, `DocToken`, `DocFamilyToken`, the eight scoped token aliases) | 62–134 | A definition is `{ kind, version, initial(), migrate?(value, fromVersion), checkpointWhen?(value, ops, { deltasSinceBase }) }` plus its semantics; a family has `family: true` and `initial(seed)`. `kind` is persisted protocol; `version` a positive integer; `checkpointWhen` returning true stores the change as a complete base. A token is `{ definition }`; the aliases only narrow which owner arguments an access takes. | crisp | `records`; each definition lives with its owning module (live, inbox, usage, provider, agent; H5) | data plus small pure functions |
| Task-execution types (`RunningTask`, `NextTaskState`, `PhaseHandler`, `HookRunner`, `TaskRuntime`, `TaskDefinition`, `Task`) | 136–261 | Contract of task code: phase map, `abort`, `migrate`, hooks; runtime `commit`, `memo`, `getTask`, `waitForTask`, `outcomes`, `conversation`, `entry`, `context`, `now`, `report`, `sleep`, `watchDoc`, `snapshot`, `snapshotAsOf`. A phase must commit a changed checkpoint or terminal outcome or the task faults. | crisp | type declarations only; behaviour classified by the task/scheduler group | lives in this file but is §5 semantics |
| `TaskOwnership`, `JoinPolicy`, `TaskOptions` | 263–280 | Ownership is required: `{ kind: "conversation" }` (top-level task) or `{ kind: "task", taskId }` (child task, same conversation). `conversationId` defaults to the owner task's conversation, else the commit's bound conversation; it is required otherwise. `background: true` only for conversation-owned tasks (excluded from ordinary idle waits, conversation aborts and cascades). Join policy `failFast` or `allSettled`. | crisp | `records`; rules enforced by `durable.commit` (createTask row) | record shape |
| `ConversationOwnership`, `ConversationRecord` | 282–298 | Record `{ id, parent?: { conversationId, at: EntryId }, owner?: { conversationId, taskId } }`, immutable once written. `parent` = history inheritance (fork, inclusive through `at`); `owner` = creator task, for attribution, subtree abort and subtree idle waits, not access control, and it stays after the task ends. Callers pass only `{ kind: "ownerless" }` or `{ kind: "task", taskId }`; the store derives `owner.conversationId` from the task record. | crisp | `records` | record; rules H3/H4 |
| `ContextEdit`, `EntryRecord`, `EntryDraft`, `TypedEntry`, `TypedEntryDraft`, `Entry` | 300–352 | Entry `{ id, conversationId, kind, model?: Message[], data?: JSON, head?: EntryId, edits?: ContextEdit[], byTaskId? }`. `model` is what the model sees (absent for display or bookkeeping entries); `data` is for views, extensions and bookkeeping; `head` starts a new active context at that entry; an edit `{ target, action: "omit" }` or `{ target, action: "replace", messages }` overrides an earlier visible entry's model contribution. A draft has no `id`/`conversationId`/`byTaskId` and may set `head: "self"`. `Entry<D>` token = `{ kind, is(entry) }`. | crisp | `records` | record; meaning H1/H2 |
| `SubmissionRecord`, `SubmissionSettlement`, `SubmissionCreate` | 354–439 | `{ id, conversationId, requestId?, type: "input" or "write", status, entry?, answer?, reason?, detail? }`. Input: `queued` → `placed` (entry) → `done` (entry, answer) or `unanswered` (reason, detail?, keeps entry if placed). Write: `queued` → `done` (entry) or `unanswered` (no entry). Settlement is `{ status: "done", answer }` or `{ status: "unanswered", reason, detail? }`. `requestId` is a host dedup key scoped to the conversation. | crisp | `records` | state record; transitions H6 |
| `TaskOutcomeError`, `TaskOutcome`, `TaskState`, `TaskRecord` | 441–560 | Record `{ id, conversationId, kind, version, input, owner?, background, abortRequested, startedAt?, endedAt?, state, memos? }`. State: `pending`/`running` `{ checkpoint }`, `waiting` `{ checkpoint, on: TaskId[], policy }`, `completing` `{ outcome }`, `terminal` `{ outcome }`. Outcome: `completed { result }`, `failed { error: { message, detail? }, result? }`, `aborted { reason?, result? }`, `orphaned { reason }`, `faulted { error }`. `memos` only while live. | crisp | `records` | state record; stamps H7 |
| `DocumentRecord`, `DocumentCreate` | 562–606 | Persisted incarnation `{ id, kind, key?, createdAt: Seq, retiredAt?: Seq, scope, history?, fork? }`; `id` never reused; create record = same without the two Seq stamps. | service | `durable` internal | storage bookkeeping; natlang code sees document values only |
| Scan types (`Page`, `Cursor`, `ScanOrder`, `ConversationQuery`, `EntryQuery`, `TaskQuery`, `SubmissionQuery`) | 608–659 | Page `{ items, next? }`; cursor is opaque JSON that carries the scan order; conversation/task/submission scans default `ascending`, entry scans default `descending`. Filters: conversations by `ownerConversationId`, `ownerTaskId`; entries by inclusive `minEntryId`/`maxEntryId`; tasks by `conversationId`, `kind`, `status`, `abortRequested`, `background`; submissions by `conversationId`, `status`. | service | parameters of `durable` reads | storage interface |
| Document storage types (`DocumentPoint`, `DocumentAddress`, `DocumentQuery`, `DocumentContent`, `DocumentCopySource`, `StoredDocument`) | 661–705 | Point = a `Seq` or `"current"`. Address = `{ kind, scope, key? }` (no key = singleton). Content = `{ version, kind: "base", value }` or `{ version, kind: "delta", ops }`. Stored document = `{ record, version, value, deltasSinceBase }`. | service | `durable` internal | storage machinery |
| `StorageWrite` | 707–724 | One atomic batch item: `conversation`, `entry`, `task`, `submission` (complete records), `document.create` (record + base), `document.copy` (record + source id/point), `document.change` (id + content), `document.retire` (id). | service | `durable` internal | commit batch format |
| Commit publication types (`DocumentCommitChange`, `TableCommitChange`, `CommitChange`, `CommitPublication`) | 726–760 | After a successful commit: `{ seq, changes }`, order unspecified. Table changes are the records as written. Document change = `{ type: "document", record, conversationId (owner's; task docs derive it from the task), version, value (null when retired), ops (empty for create/retire) }` or `{ type: "document.copy", record, conversationId, source }` for an unaccessed fork copy. | service | payload of `durable.onCommit` events | observation machinery |
| `Tx` | 762–861 | Transaction surface of one commit: table reads, conversation create/fork, entry append, task create, raw submission create/settle/place, typed document get-or-create (`doc`) and `retireDoc`. Full rules per method in session/transaction.ts below. | service | `durable.commit` | atomic commit line |
| Observation types (`DocumentState`, `WatchEnd`, `WatchHandle`, `DocumentWatch`, `DocumentReader`, `DocumentObserver`) | 863–918 | Watch handle `{ value, start(listener), stop() → WatchEnd, closed }`; end reasons `stopped`, `cancelled`, `session_closed`, `retired`, `listener_error`. `DocumentReader` = `snapshot` + `snapshotAsOf`. | service | `durable.watchDoc`, `durable.snapshot*` | watchers are host infrastructure |
| `Session` | 920–1005 | `commit`, `close`, `subscribeCommits`, `subscribeClose`, `snapshot`, `documentState`, `snapshotAsOf`, `watchDoc`. | service | `durable` | the kernel |
| `Storage` | 1007–1107 | Backend contract: `commit(writes) → Seq`, `mintId`, keyed lookups and ordered scans of every table, `findDocument`, `document(id, at)`, `scanDocuments`, `close`. Storage trusts the Session for semantic validity; it enforces atomicity, global ID ownership, immutable conversation/entry creation, document record consistency and detached values. | service | `durable` backend | §10 storage contract |

Notes. This file is the shared vocabulary. Only the record types (conversation, entry, submission, task, document
definition) appear in natlang signatures; every Storage/Tx/observation type stays inside the host service. The
task-execution types are listed for completeness; the scheduler group owns their behaviour.

## durable/src/entries.ts — the built-in transcript entry kinds

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `defineEntry(kind)` | 6–12 | If `kind` is not a non-empty string throw `TypeError("Entry kind must be a non-empty string")`; return `{ kind, is(entry) = entry !== undefined && entry.kind === kind }`. | crisp | `records.defineEntry`; called by extension definitions | pure plumbing |
| Built-in entry tokens `UserEntry` (`pi.user`), `AssistantEntry` (`pi.assistant`), `SystemEntry` (`pi.system`), `ToolResultEntry` (`pi.tool-result`, data `{ diagnostics }`), `ResetEntry` (`pi.reset`), `CompactionEntry` (`pi.compaction`, data `{ reason }`) | 14–34 | The six kinds the harness writes and reads; exact meaning, payload, head use and writers in H1. | crisp | `records` constants; meaning goes into the instructions of the writers (submission placement, generation, tool, compaction, reset) and of `deriveContext` | the token is data; the meaning is harness semantics (H1) |

Notes. Kind names starting `pi.` are reserved by convention only (spec §12); an extension that reuses `pi.system`
would have its entries replayed as system messages.

## durable/src/tasks.ts — task definition helper

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `defineTask(definition)` | 4–8 | Returns `{ definition }` unchanged; a task only becomes runnable when installed in the registry. | crisp | trivial wrapper; task group | identity |

## durable/src/documents.ts — document definition helpers and typed-access checks

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `defineDoc`, `defineDocFamily`, `validateDefinition` | 38–99 | Validate then return `{ definition }`. Validation: `version` must be a safe integer ≥ 1, else `TypeError("Document <kind> version must be a positive integer")`. (`fork: "asOf"` with `history: "latest"` is prevented by the TypeScript types only.) | crisp | `records.defineDoc`; called where each document is declared | pure check |
| `resolveAddress`, `ownerId` (+ `AnyDocDefinition`, `AnyDocToken`, `ResolvedAddress`) | 82–134 | Parse the overloaded argument list by the definition's scope: session → no owner; conversation → next arg is the conversation ID; task → next arg is the task ID (must be a safe integer, else `TypeError("Document <kind> requires a <scope> ID")`); a family then takes the key; returns the address, its string identity and the index of the next argument (seed or context). | service | `durable` internal | in natlang the caller passes an explicit address record `{ kind, scope, key? }`, so no overload parsing is needed |
| `addressId(address)` | 137–145 | Stable identity string `JSON.stringify([kind, scopeKind, ownerId or null, key or null])`. | service | `durable` internal (tracker cache key, fork duplicate check) | cache plumbing |
| `documentCreate(definition, address, id)` | 148–165 | Build the create record: `{ id, kind, key?, scope }`, plus `history` and `fork` from the definition for conversation scope. | service | `durable` internal | storage record assembly |
| `checkRecordScope(definition, record)` | 168–176 | Reject access whose definition disagrees with the persisted scope kind, or for conversation documents with the persisted `history`/`fork`: `TypeError("Document <id> (<kind>) does not match the supplied definition semantics")`. A migration cannot change lifetime semantics. | service | `durable` internal, on every typed access | integrity check |
| `checkRecordVersion(definition, record, version)` | 179–190 | Stored version > definition version → `Error("Document <id> (<kind>) has newer version <v> than <d>")`; stored < definition and no `migrate` → `Error("Document <id> (<kind>) requires migration from version <v>")`. | service | `durable` internal | version gate (rule in H5) |
| `materializeDocument`, `materializeDocumentValue` | 193–208 | Check scope and version; same version → value as stored; older → `copyJson(definition.migrate(value, storedVersion))` (copy rejects non-strict JSON). | service | `durable` internal; `migrate` itself is a crisp function of the definition | migration mechanics; the migration function is the definition owner's crisp code |

Notes. None of the built-in documents has a `migrate` or is a family (all five are version-1 singletons; H5), so
families, seeds and migration are extension-facing mechanisms. They still need to exist in the service because
extension code and future versions use them.

## durable/src/ids.ts — brand casts

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `idFromNumber`, `seqFromNumber` | 4–11 | Identity casts applying a compile-time brand. | out | — | no runtime effect; natlang IDs are plain numbers |

## durable/src/errors.ts — error classes callers distinguish

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `ReadAfterWrite` | 3–9 | Thrown when a transaction reads a table after its first table write: message `Tx.<method>() cannot read tables after the first table write`. | service | error code of `durable.commit` | transaction discipline (H8) |
| `StorageRejected` | 11–17 | Storage refused a batch before any durable effect; the Session rolls back and stays usable. Any other storage error after admission poisons the Session. | service | `durable` internal error class | durability machinery |
| `ConversationBusy` | 19–28 | `Conversation <id> is busy`; carries `conversationId`. Raised when an input with `whenBusy: "reject"` reaches a busy conversation. | inline | in the submission-admission function (submissions group) as its "busy" result | it is a harness admission outcome, not storage |

## durable/src/index.ts — package exports

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| export list | 1–185 | Re-exports definitions, entry tokens, errors, harness pieces, `createSession`, `MemoryStorage`, `defineTask`, all record types, `ROOT_CONVERSATION_ID`. | out | — | module surface; the natlang layout decides its own (records module, `durable` service, harness folders) |

## durable/src/session/session.ts — the Session kernel: one mutation line, the loaded-document cache, committed publication, reads and watches

State record `SessionImpl`: `storage`; `documents` (map address identity → `LoadedDocument`, the tracker cache);
`commitListeners`; `closeListeners`; `host` (callbacks handed to transactions); `tail` (promise chain = the
mutation line); `closing` (close promise once started); `poison` (`{ error }` after an uncertain storage failure).

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `createSession(storage, { now })` | 48–51 | Open a kernel over one backend; `now` (default `Date.now`) stamps task `startedAt`/`endedAt`. | service | `durable.open` (host start-up) | infrastructure |
| constructor / transaction host | 69–84 | Builds the host the transactions use: `storage`, `now`, `cached(address)`, `load(...)`, `install(doc)`, `evict(address, recordId)` (only if that incarnation is still cached), `conversationCreated(tx, record)`. | service | `durable` internal | wiring |
| `commit(change, context)` / `commitWith(change, context, scope)` | 86–102 | Reject if closed (`Session is closed`) or poisoned; queue one transaction on the line. `scope = { conversationId?, taskId? }`: `conversationId` is the default conversation for `tx.createTask()`, `taskId` is stamped as `byTaskId` on every appended entry (task runtime commits). | service | `durable.commit(..., scope)`; called by every harness module that writes | atomic commit line |
| `readOnLine(job)` | 105–115 | Run a read-only job on the line so several reads see one committed state (used for context bounds, views, scheduler passes, task graph). | service | `durable.readTogether(reads)` | consistency mechanism |
| `conversationDocumentOnLine(token, conversationId)` | 121–133 | For a job already on the line: load (or reuse) the current incarnation, check scope/version, return `{ record, version, value }` or `undefined`. | service | `durable` internal (conversation view mount) | read mechanics |
| `snapshot(token, owner…, [key], context)` | 135–180 | Current value of a document or `undefined` when absent. Never creates or persists. Serves the cached tracker value directly if its value version equals the token's version; otherwise loads on the line (migrating in memory only). Value is a shared immutable revision (callers must not mutate). | service | `durable.snapshot(definition, address)`; called by NL functions that read documents (agent resolution, inbox, live, usage) | committed read |
| `documentState(...)` | 182–238 | Chord `AttachedReplicatedState` over one incarnation: hydrated with the current revision, then every later committed frame; `null` after retirement; disposal detaches. | out | — | Chord replicated-state bridge for UI clients; no harness logic uses it (only `types.ts` and `session.ts` mention it); a natlang frontend uses snapshots plus `onCommit` events |
| `watchDoc(token, owner…, [key], context)` | 240–302 | On the line: load the current incarnation (`undefined` if absent; never creates), check definition, attach a `CommittedWatch` that receives every later committed change of exactly this incarnation. If the context is cancelled before or during acquisition, clean up and reject with the abort reason (or `AbortError` "The operation was aborted"); later cancellation ends the watch as `cancelled`. | service | `durable.watchDoc`; exposed to task code and tools (scheduler `runtime.watchDoc`, tool `api.watchDoc`); no built-in harness logic calls it | watcher infrastructure; mapping in "Observation" below |
| `snapshotAsOf(token, conversationId, [key], at, context)` | 304–349 | Only rewindable conversation documents (`TypeError("Session.snapshotAsOf() requires a conversation document")`; `at` must be an integer, `TypeError("Session.snapshotAsOf() requires an entry ID")`). On the line: entry `at` must be visible from the conversation (`Entry <at> is not visible from conversation <c>`); use the conversation that owns that entry (may be an ancestor); find the incarnation alive at that entry's commit; `undefined` if none; materialize at that seq (failure: `Historical document <id> (<kind>) cannot be read`); migrate in memory. | service | `durable.snapshotAsOf`; callers: task runtime and tools (passthrough) | historical read |
| `close(context)` | 351–369 | First call: synchronously notify and clear close listeners (they seal admission and stop watches), then run `beforeClose()` (the Harness joins invocations), then on the line clear commit listeners and the cache and close storage (without the caller's abort signal). Every call awaits the same promise; cancelling a call only cancels that wait. | service | `durable.close` (host shutdown) | lifecycle |
| `conversationCreated(tx, record)` hook | 375–377 | Runs inside every transaction that creates or forks a conversation, after the conversation record is staged. Plain Session: nothing. The Harness overrides it (harness.ts 357–364): create `pi.live`, `pi.inbox`, `pi.usage`, `pi.provider`, then `createAgent` (H4), then `HarnessOptions.conversationCreated`. | service | hook point of `durable.commit`; the Harness rule is a crisp creation procedure owned by the harness group | mechanism here, rule in H4 |
| `beforeClose()` hook | 380–382 | Runs after admission is sealed, before storage closes; must not reject. | service | hook point | lifecycle |
| `subscribeCommits(listener)` | 385–389 | Register a synchronous post-adoption listener `(publication, context)`; returns an idempotent disposer. Listener must not throw, block, or call Session APIs. | service | `durable.onCommit` → host dispatches events (views, task graph, scheduler, submission waits, frontend) | publication |
| `subscribeClose(listener)` | 392–396 | Register a synchronous close listener; disposer returned. | service | `durable.onClose` | lifecycle |
| `unloadDocuments()` | 399–403 | On the line, drop every cached tracker; later access reloads from storage. | out | — | cache maintenance used by no module outside session.ts (tests) |
| `#runCommit(change, context, scope)` | 405–445 | The commit algorithm: (1) assert healthy; throw if context already aborted. (2) New `Transaction`; run the callback. (3) Callback throws → `settleFailure()` (abort every draft, drain pending ops) and rethrow. (4) `settleSuccess()` → write batch. (5) Empty batch → discard, return result (no storage call, no publication). (6) `storage.commit(writes)` without the caller's abort signal (admitted commits are not interrupted). (7) Storage error: discard; if not `StorageRejected`, poison the Session; rethrow. (8) `tx.adopt(seq)`; if adoption throws, poison (memory is behind durable state). (9) Publish. | service | `durable.commit` core | the atomic commit line |
| `#publish(seq, writes, documents)` | 447–467 | If any listener: changes = every table write (`conversation`, `entry`, `task`, `submission`) as written plus every document change from adoption; call each listener with `{ seq, changes }` synchronously. | service | `durable.onCommit` | publication |
| `#attachDocument(definition, loaded, create)` | 473–501 | Check scope/version; create the observer with the current tracker value; subscribe to commits and forward only changes whose `record.id` equals this incarnation, with operations from `observedOperations`, skipping empty operation lists (a migration-only base changes nothing for an observer already at that version); document states get frames without the caller's cancellation; subscribe to close → `observer.closeSession()`. Returns `{ observer, detach }`. | service | `durable` internal (watchDoc) | watcher machinery |
| `#loadDocument(definition, addressId, address)` | 503–528 | Cache hit only if the cached value version equals the token's version; otherwise evict and reload: `findDocument(address, "current")` → `undefined` if absent; `document(id, "current")` (missing → `Current document <id> (<kind>) cannot be read`); materialize (migrate in memory); cache `{ addressId, record, storedVersion, valueVersion = token version, deltasSinceBase, tracker }`. | service | `durable` internal | cache mechanics |
| `#enqueue(job)` | 530–537 | Append the job to the promise chain; the chain continues whether the job succeeds or fails. | service | `durable` internal: the mutation line | serialization mechanism (an EventLoop could carry it; see natlang mechanisms) |
| `#assertUsable`, `#assertHealthy` | 539–550 | Closed → `Session is closed`; poisoned → `Session is poisoned by a failed commit after storage admission; reopen it` (cause = original error). | service | `durable` internal | lifecycle guards |
| `observedOperations(observed, change)` | 557–565 | Retired (`value === null`) → `[["r", null]]`; same version as the observer → the change's ops; different version → set the observer's version and return a root replacement `[["r", value]]`. | service | `durable` internal | observation rule |
| `cancellationError(signal)` | 567–569 | The signal's reason if it is an `Error`, else `DOMException("The operation was aborted", "AbortError")`. | service | `durable` internal | abort mapping |

Notes. Callers outside this file: `commitWith` (harness, scheduler, submissions), `readOnLine` (context, scheduler,
view, submissions, harness, task graph), `subscribeCommits`/`subscribeClose` (view, scheduler, task graph,
submissions, coding-agent frontend), `watchDoc`/`snapshotAsOf` (passthrough to task and tool code). Everything here
is the host service; nothing in it is a harness decision. The one harness rule attached here is the creation hook
(H4), which is implemented by the Harness subclass in another group.

## durable/src/session/transaction.ts — one commit callback's transaction: table reads, staged writes, document drafts, batch assembly and adoption

State record `Transaction`: `host`, `context`, `scope { conversationId?, taskId? }`, `pendingOperations` (every
in-flight async Tx call), `sealed`, `hasTableWrite`, `writes` (conversation and entry writes staged eagerly),
`createdConversationIds`, `forkSourceConversationIds`, `forkSourceDocumentIds`, `tasksById` (per task: memoized
committed read, candidate write `create` or `replace`, publication conversation), `submissions` (created this
commit), `submissionChanges` (settlements/placements in staging order), `plans`, `documents` (every acquisition or
retirement in order), `latestDocumentByAddress`.

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `applySubmissionChange(current, change)` | 61–79 | The submission state machine. A record already `done` or `unanswered` is returned unchanged (settling twice is a no-op). Placement: current must be `queued` (else `Submission <id> is not queued`); an input becomes `placed` with `entry`, a write becomes `done` with `entry`. Settlement `done`: current must be `placed` (else `Submission <id> is not a placed input`). Otherwise merge the settlement fields into the record (an `unanswered` input keeps its `entry`). | service | `durable.commit` assembly; the rule is H6 and must also be stated to the submissions group | pure rule executed inside the commit |
| `LoadedDocument`, `TransactionHost`, `TransactionTask`, `TransactionScope`, `DocumentTarget`, `DocumentEntry`, `DocumentPlan` | 85–186 | Internal records: cached incarnation `{ addressId, record, storedVersion, valueVersion, deltasSinceBase, tracker }`; host callbacks; per-task staging; scope; a staged document's provenance (`loaded`, `created`, `fork-copy`, `retire-only`); its write/publication plan. | service | `durable` internal | machinery |
| Table reads: `conversation`, `entry(id)` / `entry(token, id)`, `task`, `scanConversations`, `scanEntries`, `latestHeadMarker`, `scanTasks`, `submission` (internal), `submissionByRequest` | 229–278 | Read **committed** storage state (not this commit's staged writes). Each read fails with `ReadAfterWrite` once any table write was staged, and with `Transaction has settled` after the callback settled. `entry(id)` is a global lookup without visibility check; the typed form returns `undefined` for another kind. `latestHeadMarker(c)` = newest visible entry with a `head`, no cutoff. | service | `durable.commit` Tx reads | transaction mechanics; discipline in H8 |
| `createConversation`, `createRootConversation`, `forkConversation`, `#stageConversation` | 282–343 | Mint an ID (root: reserved `1`). Task ownership: the owner task (latest candidate in this commit, else committed) must exist (`Conversation owner task <id> does not exist`); `owner = { conversationId: task.conversationId, taskId }`. Record `{ id, parent?, owner? }`. For a fork, compute the document copies (forks.ts), stage each as a `fork-copy`, remember source docs and the parent. Stage the conversation write, then run the creation hook in this same commit. Returns the inert record. | service | `durable.commit` Tx `createConversation`/`forkConversation`; callers: harness root/create/fork, subagent tools | machinery; rules H3/H4 |
| `appendEntry` / `#appendEntry` | 345–381 | Conversation must exist or be created in this commit (`Conversation <id> does not exist`). Mint an ID. Record = draft fields + `id`, `conversationId`, `head` (`"self"` → the new ID), `byTaskId` = the commit scope's task; undefined fields omitted; the copy rejects non-strict JSON. Staged immediately; returns the record. | service | `durable.commit` Tx `appendEntry`; callers: submission placement, generation, tool, compaction, reset | machinery; meaning H1/H2 |
| `createTask(task, input, options)` | 383–426 | For `ownership.kind === "task"`: owner (candidate or committed) must exist (`Task owner <id> does not exist`); `background: true` → `TypeError("A child task cannot be background")`; an explicit different `conversationId` → `A child task lives in its owner's conversation <c>`. Conversation = owner's, else `options.conversationId`, else scope's; none → `TypeError("Tx.createTask() requires options.conversationId")`; it must exist. Checkpoint = `definition.initial(input)`. Record `{ id, conversationId, kind: definition.name, version, input, owner?, background: false by default, abortRequested: false, state: { status: "pending", checkpoint } }`. Owner liveness is validated again at assembly. | service | `durable.commit` Tx `createTask`; callers: submissions (run task), generation (tool tasks), compaction, tools | machinery; rules H7 |
| `createSubmission(create)` | 428–439 | Conversation must exist; mint ID; copy record. No admission rules (no busy check, no inbox, no placement). | service | `durable.commit` Tx; caller: submissions group's admission function | raw record write |
| `settleSubmission(id, settlement)`, `placeSubmission(id, entry)` | 441–456 | Count as table writes; stage the change; resolved at assembly against this commit's latest record (created here or committed), so they work after other writes. | service | `durable.commit` Tx; callers: submissions, inbox placement, live/run finish | machinery; state machine H6 |
| `setTask(record)` (internal) | 458–474 | Replace a task's whole record. If this commit already holds a terminal candidate: `Task <id> already has a terminal candidate`; changing conversation: `Task <id> cannot change conversations`. A task created in this commit stays a `create`. Stamps times (`#stampTimes`). | service | `durable.commit` internal op used by the scheduler and task runtime | machinery; stamps H7 |
| `stagedTasks()`, `stagedConversations()` | 476–488 | Candidate task records and conversation records staged so far. | service | `durable.commit` internal (scheduler inspects what a commit created) | getters |
| `doc(token, owner…, [key, seed])` | 491–540 | Typed get-or-create of a draft. Task-scoped: rejected if this commit holds a terminal candidate for the task (`Task <id> is terminal`). Memoized per address: a second call returns the same draft; a pending `fork-copy` at the address is materialized on first access. After `retireDoc` at the same address, a new incarnation is created (no load). Family seed is copied; the first call's seed wins. | service | `durable.commit` Tx `doc`; callers: almost every harness commit (44 uses) | Chord drafts; natlang shape in "Commit" below |
| `retireDoc(token, owner…, [key])` | 542–585 | Resolve without creating. Already retiring → no-op. Pending fork copy → mark it retired (copied and retired in one batch). Acquired draft → retire after persisting its final content. Otherwise look up the current incarnation (cache, else storage) and stage a retirement (absent → nothing). | service | `durable.commit` Tx `retireDoc` | machinery |
| `#acquire(entry, seed, skipLoad)` | 587–622 | Load the current incarnation (unless replacing one retired this commit); if found, check scope/version and begin a tracked change. If absent: conversation must exist; task must exist (`Task <id> does not exist`) and not be terminal (`Task <id> is terminal`); value = `initial(seed)` or `initial()` (copied, strict JSON); mint an ID; begin a change on a new tracker. | service | `durable` internal | machinery |
| `#acquireForkCopy` | 624–649 | Read the stored source at its point (`Fork source document <id> cannot be read`); kind, key, history and fork must match the copy record (`… does not match the copied record`); migrate to the token's version; the copy becomes an ordinary create with that value. | service | `durable` internal | machinery |
| `#findRetirement` | 651–659 | Find the current record (cache first, then storage); check scope; stage `retire-only`. | service | `durable` internal | machinery |
| `settleFailure()`, `settleSuccess()`, `discard()` | 663–696 | Failure: seal, abort every change, wait for pending ops. Success: seal; if any Tx operation is still pending, abort and throw `Session commit callback settled before its pending Tx operations`; prepare every open change (revokes every draft); assemble; any error aborts every change. Discard aborts every prepared change. | service | `durable` internal | commit protocol |
| `adopt(seq)` | 699–753 | After storage success, per plan: stamp `createdAt = seq` on new records and `retiredAt = seq` on retirements; adopt the prepared value into its tracker by pointer swap (a new incarnation unless it also retires; a loaded one only if it had operations); advance `storedVersion`, `deltasSinceBase` (reset on base, +1 on delta); install new incarnations in the cache; evict retired ones. Produces publications: retirement → `{ value: null, version: undefined, ops: [] }`; fork copy → `document.copy`; otherwise `{ version, value, ops }` (empty ops for creations). | service | `durable` internal | commit protocol |
| `#assemble()` | 755–847 | Build the batch in this order: (1) a plan per staged document; (2) reject writes to fork sources; (3) validate owners; (4) every replaced task must exist (`Task <id> does not exist`), not be terminal (`Task <id> is already terminal`), keep its conversation; (5) for every task whose candidate is `terminal`, retire all its task documents (staged ones and, for existing tasks, every current one found by scanning its task scope, page size 256); (6) resolve each published document's owning conversation (task docs via their task record) before storage so adoption needs no reads; (7) apply submission changes in staging order; (8) writes = staged conversations and entries, then submissions, then tasks, then document content (evaluating `checkpointWhen` last, only for loaded deltas; true → replace the delta by a base), then retirements. | service | `durable` internal | commit protocol; the terminal-retires-task-docs rule is H5 |
| `#validateOwners()` | 855–873 | For every new task-owned conversation and every new child task, the owner's final candidate (or committed record) must exist (`<Conversation owner task or Task owner> <id> does not exist`), must not be `terminal` or `completing` (`… is <status>`), and must not be abort-marked (`… is abort-marked`). So a task cannot create owned work in the commit that finishes it, nor while abort-marked. | service | `durable` internal | structured-concurrency invariant (spec §5.5) enforced at commit (H7) |
| `#rejectForkSourceWrites(plans)` | 875–894 | In a fork commit: changing or retiring a selected source document → `Cannot change fork source document <id> in the fork transaction`; changing any `fork: "current"` document of the forked parent → `Cannot fork conversation <c> while changing its current-policy documents`. | service | `durable` internal | fork consistency (H4) |
| Guards: `#abortChanges`, `#assertOpen`, `#assertTaskDocumentsOpen`, `#track`, `#read`, `#write`, `#requireConversation` | 896–946 | Abort all drafts; `Transaction has settled` after sealing; terminal-task doc guard; register pending ops; reads fail after first table write (`ReadAfterWrite`); writes set `hasTableWrite`; conversation existence (created here or in storage; `Conversation <id> does not exist`). | service | `durable` internal | machinery |
| `#stampTimes(value, candidate)` | 952–963 | `startedAt` = earlier candidate's, else the record's, else `now()` if status is `running`; `endedAt` likewise with status `terminal`. Once set they carry over. | service | `durable` internal | lifecycle stamp (H7) |
| `#taskEntry`, `#currentTask`, `#committedTask` | 965–983 | Per-task staging slot; latest candidate else committed record; committed read memoized once per commit. These internal reads do not trigger `ReadAfterWrite`. | service | `durable` internal | machinery |
| `planDocument(entry)` | 987–1034 | `created` → `document.create` with a base of the prepared value. `fork-copy` → `document.copy`. `retire-only` → no content. `loaded` → if stored version < definition version: `document.change` base (even with no ops); else if ops non-empty: `document.change` delta; else no content. | service | `durable` internal | base/delta selection (§3.5) |
| `publishes(plan)` | 1040–1042 | Publish every creation, copy and retirement, and a loaded document that writes content (including a migration-only base). | service | `durable` internal | publication rule |

Notes. The transaction is entirely mechanism. The harness depends on its rules, not its code: reads before writes,
document read-your-writes, atomicity, owner liveness, fork-source restrictions, task documents retiring with their
task. These are collected in H4–H8 so that callers (and any crisp commit procedures) state them.

## durable/src/session/observation.ts — committed-state sources and watches

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| Constants `MAX_PENDING_WATCH_FRAMES = 100`, `RETIREMENT_OPERATIONS = [["r", null]]` | 12–18 | Watch buffer bound and the canonical retirement frame. | service | `durable` internal | watch parameters |
| `CommittedStateSource` + `SessionSourceAttachment` | 24–143 | Chord `ReplicatedStateSource`: `attach()` captures `{ value, cursor }` atomically and buffers later frames; `activate(listener)` drains synchronously; `advance(value, ops)` increments the cursor, a `null` value retires it, frames are delivered in a microtask; `closeSession()` disposes all attachments; last disposal releases the commit subscription. A listener throw disposes only that attachment. | service | host-only bridge used by `documentState` (out), the conversation view and the task graph (other groups) | Chord replication for UI mounts; keep only if the natlang frontend binds Chord states |
| `CommittedWatch`: `start`, `stop`, `observeCancellation`, `cancel`, `closeSession`, `advance`, `#schedule`, `#drain`, `#terminate`, `#finishIfReady` | 155–297 | `value` = acquisition revision until delivery, then the last delivered value. `start(listener)` once (`Watch is already started`; after end `Watch is stopped`); never calls inline (microtask). `advance`: ignored after end or retirement; when 100 frames are pending, drop them and queue one root replacement `[["r", newest]]` (with the newest commit's context); else queue the exact frame. `#drain` delivers one frame at a time, awaiting the listener, with the frame's context minus cancellation; a listener throw ends the watch `listener_error`; a delivered `null` ends it `retired`. `stop()` (idempotent) → `stopped`; cancellation → `cancelled`; Session close → `session_closed`. First termination reason wins; end detaches immediately and drops pending frames; `closed` resolves when no callback is running. | service | `durable.watchDoc` handle; mapping in "Observation" | watcher infrastructure (§9.2) |
| `toError` | 299–301 | Non-Error throw → `new Error(String(error))`. | service | `durable` internal | trivial |

## durable/src/session/forks.ts — which documents a fork copies

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `prepareForkDocumentCopies(storage, parent, at, child)` | 15–59 | Entry `at` must be visible from `parent` (`Entry <at> is not visible from conversation <parent>`). Collect (1) the `fork: "asOf"` documents of the conversation **that owns entry `at`** (may be an ancestor of `parent`), alive at that entry's commit seq, then (2) the `fork: "current"` documents of `parent` alive now. `initial` documents are not copied. Page size 256. | service | `durable` internal (forkConversation) | storage-level copy selection; the policy itself is H4 |
| `collectCopies(...)` | 61–97 | For each alive conversation document whose `fork` equals the policy: mint a new ID; child create record keeps `kind`, `key`, `history`, `fork`, scope = child; source = `{ id, at }`. If two sources map to the same child address: `Fork selects multiple source documents for <kind>[/<key>]`. | service | `durable` internal | machinery |

## durable/src/storage/scan.ts — cursor helpers shared by the built-in backends

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `scanStart(requested, cursor, fallback)` | 14–33 | Invalid `order` → `TypeError("Invalid scan order: <x>")`. No cursor → `{ order: requested ?? fallback, after: undefined }`. Cursor `{ after: safe integer, order? }` else `TypeError("Invalid storage cursor")`; order = cursor's (or fallback for old cursors); a different requested order → `TypeError("The cursor continues a <order> scan; the query asks for <requested>")`. | service | backend internal | storage plumbing |
| `nextCursor(id, order)` | 36–38 | `{ after: id, order }`. | service | backend internal | storage plumbing |

## durable/src/storage/memory.ts — the reference Storage backend (in memory, detached copies)

State record `MemoryStorage.state`: `recordTypes` (ID → table), conversations (+ sorted IDs, indexes by owner
conversation and owner task), entries (+ sorted IDs per conversation, head-carrying IDs per conversation, commit seq
per entry), tasks (+ sorted IDs, IDs per status), submissions (+ IDs, IDs per status, requestId index per
conversation), documents (record + revisions `[{ base or delta, seq }]`), address index (`ids`, `currentId`), IDs per
scope. Counters `nextId = 2`, `nextSeq = 1`; `closed`.

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| Value helpers `clone`, `freeze`, `cursorId`, `lowerBound`, `upperBound`, `insertSorted`, `removeSorted`, `insertMapId`, `scopeKey`, `addressKey`, `recordAddressKey`, `tableContaining`, `page`, `scanIndexes`, `documentDeltaBatches` | 52–220 | Deep copy (keeps null-prototype objects, defines `__proto__` safely); deep freeze; sorted-array binary search and insert/remove; address keys `JSON.stringify([kind, scopeKey, ["singleton"] or ["family", key]])`; `page` collects up to `limit + 1` and sets `next` from the last returned ID only when more exist; `scanIndexes` walks ascending after the cursor or descending before it; delta replay throws `Document <id> crosses a stored version boundary without a base` on a base or version change inside the tail. | service | backend internal | storage plumbing |
| `isAliveAt(record, at)`, `isCurrentOnly(record)` | 191–197 | Alive at `"current"` iff not retired; at seq `s` iff `createdAt <= s < retiredAt` (no upper bound when unretired). Current-only = not a conversation document, or `history: "latest"`. | service | backend internal | lifetime rule (§3.2) |
| `commit(writes)` / `prepareCommit(writes, seq)` | 262–288 | Seq must be ≥ next (`Commit sequence <s> does not strictly increase`). Deep-copy and freeze the writes, resolve fork copies, check global IDs, prepare and check document actions; nothing changes until `apply()`, which applies once. | service | backend | atomicity |
| `resolveDocumentCopies(writes)` | 290–328 | Turn each `document.copy` into a `document.create` with the source's **stored** version and value at its point. Source changed in the same batch, unreadable, or mismatched (scope, kind, key, history, fork) → `StorageRejected("Document copy <id> was rejected")` with the cause. | service | backend | fork copy (§10) |
| `applyPreparedCommit` | 330–425 | Insert/replace table records and their indexes (status indexes move on status change; requestId index follows the record); entries record their commit seq and head index; then apply document actions; `nextId = max(nextId, id + 1)`; `nextSeq = seq + 1`. | service | backend | storage |
| `mintId()` | 427–431 | Return `nextId++` (`ID space is exhausted` past safe integers). | service | backend | ID allocation |
| `conversation`, `task`, `submission`, `submissionByRequest` | 433–604 | Keyed lookups returning detached copies; `submissionByRequest` by `(conversationId, requestId)`. | service | backend | keyed reads |
| `scanConversations`, `scanTasks`, `scanSubmissions` | 439–593 | Ordered by ID (default ascending); use the owner-task index, else owner-conversation index, else all (filters conjunctive); tasks/submissions use the status index when `status` is given, then filter the remaining fields. | service | backend | scans |
| `entry(id)` / `entry(conversationId, id)` | 465–489 | Global form: entry + its commit seq. Conversation form: only if visible through the conversation's fork ancestry (see `visibleEntries`). | service | backend | visibility rule (H3) |
| `findLatestHeadMarker(c, atOrBefore)` | 491–514 | Unknown conversation → `Unknown conversation: <c>`. In `c`, the newest head-carrying entry with ID ≤ cap (cap = cutoff or ∞); if none, move to the parent with cap = min(cap, parent.at); none at the root → `undefined`. | service | backend | context bound (H2) |
| `scanEntries(query, limit, cursor)` | 516–539 | Default descending. A cursor narrows the bound the scan moves away from (descending: max = min(max, after − 1); ascending: min = max(min, after + 1)). Descending uses `visibleEntries`, ascending `visibleEntriesAscending`; page of `limit`. | service | backend | fork-aware history scan |
| `findDocument(address, at)` | 606–621 | Current: the address's `currentId` record. Historical: the first incarnation at the address alive at `at`. | service | backend | document lookup |
| `document(id, at)` / `materializeDocument` | 623–676 | Unknown → `undefined`. Numeric point on a current-only document → `Document <id> does not retain historical content`. Not alive at point → `undefined`. Take revisions with seq ≤ point, the newest base (missing → `Document <id> is missing a required base`), replay the following deltas (all same version) → `{ record, version: base version, value, deltasSinceBase }`. | service | backend | replay |
| `scanDocuments(query, limit, cursor)` | 636–653 | Incarnations in one exact scope alive at the point, optional kind, ascending by ID only. | service | backend | document scan |
| `visibleEntries`, `visibleEntriesAscending` | 678–731 | Descending: entries of `c` with ID in [min, cap] newest first, then the parent's with cap = min(cap, parent.at), up the chain; stop early when cap < min. Ascending: the same segments, root segment first. | service | backend | **visible history rule (H3)** — the one piece of semantics in the backend |
| `checkGlobalIds(writes)` | 733–753 | Conversation, entry and document IDs must be new (`ID <id> already belongs to <table>`) and written once (`ID <id> is written more than once`); task and submission IDs may be rewritten in their own table but never claimed by another (`ID <id> is written as two record types`). | service | backend | global ID ownership |
| `prepareDocumentActions`, `checkDocumentActions`, `applyDocumentActions` | 755–859 | At most one content command per incarnation (`Document <id> has more than one content command`), retire at most once. Checks: unknown document, create of an existing one, change of a retired one, delta without base, delta across a version (`version transition requires a base`), more than one current incarnation per address. Apply: create → record with `createdAt` (and `retiredAt` if also retired), first revision; content on current-only + base → replace all revisions (reclaim); else append; retire → `retiredAt`, current-only drops revisions; maintain `currentId`. | service | backend | document storage |
| `close`, `assertOpen` | 655–657, 861–863 | Mark closed; later calls throw `MemoryStorage is closed`. | service | backend | lifecycle |

## durable/src/storage/jsonl/* and durable/src/storage/sqlite/* — the persistent backends (skimmed: how they differ from memory)

| unit | lines | what it does (concrete: inputs, outputs, rules) | decision | natlang unit / caller | why |
|---|---|---|---|---|---|
| `JsonlStorage` (jsonl/storage.ts) + `openNodeJsonlStorage` (jsonl/node.ts) | storage.ts 1–845; node.ts 1–14 | Wraps a `MemoryStorage` as its index and validator. Commit: `memory.prepareCommit` → encode → append each affected sidecar (`doc-<id>.jsonl`, `task-<id>.jsonl`), flush them when `fsync: true`, append one marker line to `main.jsonl` (format 1, seq, operations referencing sidecar ordinals) → apply in memory → reclaim sidecars of current-only documents/finished tasks (rewrite via `.reclaim` temp + rename). Open replays confirmed records, removes torn lines and unconfirmed tails; missing confirmed data → `JsonlCorruptionError`; any failed append poisons the backend (`JsonlStoragePoisonedError`). Uses the portable `FileSystem` capability. | service | `durable` backend option | persistence (§11.3) |
| `SqliteStorage` (sqlite/storage.ts), schema (sqlite/migrations.ts), adapters (sqlite/node.ts, sqlite/cloudflare.ts, sqlite/database.ts) | storage.ts 1–934; migrations.ts 1–125 | One SQL transaction per commit; tables `durable_metadata` (`next_id`, `next_seq`), `record_ids` (global ID ownership), `conversations` (owner indexes), `entries` (`commit_seq`, partial index on `head`), `tasks` (status/kind/abort/background indexes), `submissions` (request index), `documents` (address and scope indexes), `document_revisions` (base/delta per seq; reclaimed by `DELETE` for current-only documents). `mintId` counts in memory from `next_id`, persisted as the max at commit. Schema migrations must be contiguous from 1; a newer schema is refused. Adapters: `node:sqlite` (used by the coding-agent frontend, `openNodeSqliteStorage`) and Cloudflare Durable Object SQL. | service | `durable` backend option (the frontend's default) | persistence (§11.2) |

Notes. All three backends pass the same conformance suite; memory is the reference semantics. For the port, the
backend is a host choice with no effect on harness logic.

---

## The `durable` service interface (what the harness logic needs)

One host service, implemented in TypeScript, owns the Session line and the backend. Natural-language functions and
crisp helpers reach it only through these operations. Values returned are detached JSON; document values are shared
immutable revisions (never mutate; copy first).

**Reads** (committed state only; each call sees one committed point; `readTogether` makes several reads at one
point, replacing `readOnLine`):

| operation | returns | rules |
|---|---|---|
| `conversation(id)` | `ConversationRecord` or undefined | keyed |
| `scanConversations({ ownerConversationId?, ownerTaskId?, order? }, limit, cursor?)` | page | default ascending by ID; filters conjunctive and indexed (subtree traversal, "find the child I created" as in the subagent tool) |
| `entry(id)` | entry or undefined | global lookup, no visibility check |
| `visibleEntry(conversationId, id)` | `{ entry, commitSeq }` or undefined | only if visible through the fork ancestry (H3) |
| `latestHeadMarker(conversationId, atOrBefore?)` | entry with `head` or undefined | newest visible head-carrying entry at or before the cutoff |
| `scanEntries({ conversationId, minEntryId?, maxEntryId?, order? }, limit, cursor?)` | page | default **descending**; inclusive bounds; fork-aware |
| `task(id)`, `scanTasks({ conversationId?, kind?, status?, abortRequested?, background?, order? }, …)` | record / page | default ascending |
| `submission(id)`, `scanSubmissions({ conversationId?, status?, order? }, …)`, `submissionByRequest(conversationId, requestId)` | record / page | default ascending |
| `snapshot(definition, address)` | value or undefined | never creates; migrates in memory only |
| `snapshotAsOf(definition, conversationId, [key], atEntry)` | value or undefined | rewindable conversation documents only (H5) |
| `readTogether([read…])` | results | all at one committed point |

Paging: `limit` is the maximum page size; `next` is present only when more items exist; cursors are opaque, carry
their order, and a different explicit order with a cursor is an error.

**Commit** (one atomic batch; nothing is visible until it succeeds). A commit runs with an optional scope
`{ conversationId?, taskId? }` (default conversation for new tasks; `byTaskId` stamp on entries). One commit can:

- read tables first (any number of the reads above, committed state), then
- create conversations (`ownership` ownerless or task) and forks (`parent`, `at`, `ownership`); each runs the
  creation procedure (H4) in the same commit;
- append any number of entries (draft: `kind`, `model?`, `data?`, `head?` an entry ID or `"self"`, `edits?`);
- create tasks (`kind` of a registered definition, `input`, `ownership`, `conversationId?`, `background?`);
  replace task records (scheduler and task runtime only);
- create raw submissions; settle (`done` + answer, or `unanswered` + reason) or place (at an entry) submissions;
- get-or-create and edit any number of documents (each address once per commit, by draft or operations); retire
  documents.

It returns the created records/IDs. Guarantees and failure modes:

1. Atomic across every record and document write (§1 inv. 1); only committed state is observable (inv. 3).
2. IDs are global, never reused; entries and conversation records are immutable (inv. 5).
3. Publication (`onCommit`) happens only after storage success (inv. 2), as one `{ seq, changes }` per commit.
4. A failing callback, preparation, validation or `checkpointWhen` rolls back with nothing written (inv. 8).
5. `StorageRejected` rolls back; any other storage failure poisons the Session (`Session is poisoned … reopen it`);
   it must be reopened.
6. Once admitted, a commit is not interrupted by caller cancellation.
7. Error vocabulary the callers must handle: `ReadAfterWrite`, the owner-liveness errors, fork-source errors,
   `Conversation <id> does not exist`, `Task <id> is terminal`, version/semantics mismatch, `Session is closed`.

Recommended natlang shape (owner decision 2): every recurring commit of the harness becomes a **named crisp commit
procedure** registered with the service and run on the line with a Tx (for example "place queued inbox items",
"record a generation attempt", "settle a submission with its answer"), so the line is never held by a model and a
small model only calls `durable.commit("procedure", args)`. Raw `commit(tx => …)` in eval stays available for
extension tool code that needs custom writes, with H8 stated in the tool-authoring instructions.

**Observation.** `onCommit(listener)` → the host dispatches a `committed` event `{ seq, changes }` to the
interested EventLoops (views, task graph, submission waiters, frontend titles); the listener itself never calls the
service. `watchDoc(definition, address)` → a handle; for natural-language task and tool code map it to a blocking
service call `nextDocChange(handle, until?)` returning `{ value, ops }` or the `WatchEnd` reason (convergent: under
backlog >100 the next result is the whole newest value). `onClose(listener)`. `documentState` is out.

---

## Harness semantics carried by these records (exact rules)

These are not mechanics; the other groups' natural-language functions (or the crisp helpers they call) must state
them.

### H1. Entry kinds (spec §8.1, entries.ts)

| kind | `model` | `data` | `head` | written by | rule for readers |
|---|---|---|---|---|---|
| `pi.user` | `[UserMessage]`, timestamp = Harness clock at placement (= admission when placed at once) | none | none | input submission placement; `onYield` continuations | ordinary model input |
| `pi.assistant` | `[AssistantMessage]` with any stop reason: answers, failed attempts with error text and usage, converted partials with stop reason `aborted` | none | none | generation | excluded from later requests when stop reason is `aborted`, `error` or `deferred` (H2 step 6); no separate usage/notice kind exists |
| `pi.system` | `[SystemMessage]` with `content: ""`; `sections` is an ordered named patch (string adds/replaces, `null` removes); `toolsRemoved` applies before `toolsAdded` | none | none | generation preparation (baseline, later deltas) | replaying all system messages in order yields the effective prompt and tool set |
| `pi.tool-result` | `[ToolResultMessage]` whose content ends with the rendered diagnostics block | `{ diagnostics: ToolDiagnostic[] }`, always present (possibly `[]`) | none | tool tasks; generation for calls to tools its request did not offer (`tool_unavailable`) | ordered after its call (H2 step 7) |
| `pi.reset` | absent (plain reset) or `[UserMessage]` with the handoff text | none | always `"self"` | `Conversation.reset()` through a write submission; generation `tools` phase for the `handoff` control | starts a new context at itself |
| `pi.compaction` | `[UserMessage]` with the wrapped summary | `{ reason: CompactionReason }` | the first kept entry | compaction tasks, directly or through a write submission | context = this summary, then the kept entries |

### H2. Heads, edits and the model transcript (spec §2.1; implemented in harness/context.ts, another group)

Input: conversation `c`, optional cutoff `at` (must be visible, else `Entry <at> is not visible from conversation
<c>`). Output: view `{ head, entries, contributions, messages }`.

1. `tail` = `at`, or the newest visible entry; no entries → `{ head: undefined, entries: [], contributions: [],
   messages: [] }`.
2. `H` = newest visible entry with a `head`, ID ≤ `tail` (walking fork ancestry, H3).
3. Range = visible entries with ID from `H.head` (or the transcript start) through `tail`, oldest first.
4. Edits: for every entry in the range, oldest to newest, every edit sets `edit[target]` (the newest edit per
   target wins). Edits held by older head markers in the range still count; edits outside the range do not.
5. Active entries = `H`, then the range's entries that carry no `head` (older head markers and `H`'s own position
   drop out); without `H`, the whole range.
6. Contribution per active entry: `omit` → nothing; `replace` → the edit's messages; otherwise `model` (or
   nothing). Then drop assistant messages whose stop reason is `aborted`, `error` or `deferred`.
7. Flatten. Place each assistant's tool results directly after it, in tool-call order: for an assistant with
   `toolCall` parts, look from the next message up to the next assistant message and take the first tool result
   per call ID. A call without a result gets a synthesized result `{ role: "toolResult", toolCallId, toolName,
   content: [{ type: "text", text: "Tool result unavailable: history ends before this call completed." }],
   isError: true, details: { reason: "missing_result" }, timestamp: <assistant timestamp> }`. Tool results not
   claimed by a preceding call are dropped.
8. Lead with system: find the first message whose role is not `user`; if it is at index > 0 and is a `system`
   message, move it to the front.
9. `contributions` are the per-entry lists after step 6 (before 7–8); stored entries keep committed order.

Recommendation (for the context group): steps 1–9 are exact plumbing over many messages and should be a **crisp**
`deriveContext` helper (its incremental `readContextFrom` variant gives identical results), while the decisions
that create heads and edits (compaction, reset, write submissions) stay natural language.

### H3. Conversations, ownership and visible history (spec §2, §2.1, §10)

- `parent` and `owner` are independent. `parent` gives inherited history and historical documents; `owner` gives
  attribution, subtree abort and idle traversal and survives the owner task's end. Ownership is always explicit
  (ownerless or a task); the owner's conversation is derived from the task.
- Visible history of `c` with cap `m` (cutoff, or ∞): `c`'s entries with ID ≤ `m`; then the parent's entries with
  ID ≤ min(`m`, `c.parent.at`); then that conversation's parent with the cap lowered again by its own
  `parent.at`; up to a conversation without parent. Newest-first scans read child before parent; oldest-first
  scans read the root segment first, then each fork's segment.
- Entry IDs are global and increasing, so a child's own entries all sort after its fork point.

### H4. Forks and conversation creation (spec §2.2, §3.7; forks.ts; harness.ts 357–364; agent.ts `createAgent`)

- `fork(P, at E, ownership)`: `E` must be visible from `P`. The child's transcript is everything visible from `P`
  through `E`, even if `E`'s commit also appended later entries.
- Documents follow each record's persisted policy: `asOf` → value of the conversation that owns `E`, at `E`'s
  commit (document state at `E` is the final state of that commit); `current` → `P`'s committed value when the
  fork commit runs; `initial` → nothing copied, created from the definition on first access. Copies keep kind,
  key, history, fork, stored version and value (unknown definitions survive), with new IDs and a fresh base.
  Task documents and tasks are never copied; session documents are shared.
- A fork commit must not also change `P`'s `current` documents or any selected source ("commit the parent change
  first").
- Creation procedure, in the same commit, for every created or forked conversation: create `pi.live`, `pi.inbox`,
  `pi.usage`, `pi.provider` (fresh `sessionId` UUIDv7, never copied); `pi.agent`: a fork keeps its `asOf` copy; a
  new task-owned conversation gets a copy of every field of the owner conversation's stored `pi.agent` (later owner
  changes do not reach it); a new ownerless conversation gets `{}`; then the host's `conversationCreated`. A plain
  Session creates no documents.
- `snapshotAsOf` on a child at an inherited entry reads the ancestor's historical instance.

### H5. Documents (spec §3)

- Address: singleton `{ kind, scope }`, family member `{ kind, scope, key }`; scope `session`, `conversation(c)` or
  `task(t)`. Each incarnation lives `createdAt ≤ seq < retiredAt`; retiring and recreating makes a new incarnation.
- Only a commit's typed access creates (`initial()` or, for a family, `initial(seed)` with the first call's seed);
  reads, states and watches return `undefined` when absent. Conversation documents need an existing conversation;
  task documents need an existing, non-terminal task, and **all of a task's documents retire in the commit that
  makes it terminal** (including ones created in that commit). Never reference a task document from the outcome
  that retires it.
- Versions: stored = definition → use; stored < definition → `migrate(value, storedVersion)` (pure, returns a
  complete current value; no `migrate` → error); stored > definition → error. Reads migrate in memory only; the
  next commit access writes a full current-version base. Kinds are protocol: renaming needs explicit copy and
  retirement.
- Storage form: creation and version changes store a base; otherwise `checkpointWhen` decides base vs delta; an
  empty change writes and publishes nothing. `latest` keeps only current state; `rewindable` keeps history.
- Built-in definitions (all version 1, singletons, no migration):

| kind | scope / history / fork | initial | checkpoint (store a base when) | owner module |
|---|---|---|---|---|
| `pi.agent` | conversation / rewindable / asOf | `{}` | every change | harness/agent.ts |
| `pi.provider` | conversation / latest / initial | `{ sessionId: uuidv7() }` | every change | harness/provider.ts |
| `pi.usage` | conversation / latest / initial | `{ models: {}, tools: {} }` | every change | harness/usage.ts |
| `pi.inbox` | conversation / latest / initial | `{ items: [] }` | `items.length === 0` | harness/inbox.ts |
| `pi.live` | conversation / latest / initial | `{}` | `generation` absent and no tool slot with status `running` | harness/live.ts |

  Only `pi.agent` needs history (fork `asOf` and `snapshotAsOf`).

### H6. Submission lifecycle (types.ts 354–439; transaction.ts `applySubmissionChange`)

- Input: `queued` → `placed { entry }` → `done { entry, answer }`; `queued` or `placed` → `unanswered { reason,
  detail? }` (a placed input keeps its entry). Write: `queued` → `done { entry }` or `unanswered`.
- Placing requires `queued`; answering (`done`) requires a placed input; settling an already settled submission
  changes nothing. `requestId` is unique per conversation and makes a retried submit exactly-once
  (`submissionByRequest`). Raw creation applies no admission rules; busy checks, `whenBusy`, inbox queueing and
  placement belong to the submissions group.

### H7. Task records (types.ts 441–560; transaction.ts `createTask`, `setTask`, `#validateOwners`, `#stampTimes`)

- New task: `state = { status: "pending", checkpoint: definition.initial(input) }`, `abortRequested: false`,
  `background` default false, `version` = definition version, `kind` = definition name.
- A child task lives in its owner's conversation and cannot be background. New owned work (child task or
  task-owned conversation) requires an owner that exists and is not `completing`, `terminal` or abort-marked,
  judged on its final record in the same commit.
- Task records are replaced whole; a task never changes conversation; nothing follows `terminal`. `startedAt` is
  stamped at the first change to `running`, `endedAt` at the change to `terminal`; both carry over. `memos` exist
  only while live. Terminal records remain queryable; large results belong in entries or long-lived documents.

### H8. Transaction discipline for every caller (spec §4, §12)

Read every table row you need before the first table write (after it, table reads fail `ReadAfterWrite`);
document reads and edits stay available and see this commit's own edits; table reads never see this commit's
writes. Use the records and IDs that creation returns instead of reading them back. Await every Tx operation
before the commit returns. Never call a model, tool, process, network, human, a nested commit or a wait inside a
commit. Values must be strict JSON (assigning `undefined` deletes a property). Only the Harness appends to a busy
conversation; other writers use a write submission.

---

## Where natlang's own mechanisms could carry a mechanism

| pi-durable mechanism | natlang mechanism | fit and gaps |
|---|---|---|
| Session mutation line (`#enqueue`), commit-then-publish (`#runCommit`, `#publish`) | `EventLoop`: serial application, `onCommit` awaited before publish, failed commit keeps previous state | Good fit for the line and publication order. Gaps: no poisoning (an uncertain storage failure must stop the loop); a reducer may run natlang steps, which pi forbids inside a commit; `KeyedEventLoop` per conversation would **break cross-conversation atomicity** (subagent creation writes a child conversation and reads/validates the parent task in one commit), so the line must stay Session-wide. |
| Submission `requestId` exactly-once | `EventLoop` duplicate event-ID suppression (`seenEventIds`) | Same idea, but pi's key is per conversation and stored in the submissions table; keep pi's rule. |
| Transaction drafts, prepare/abort, adopt only on success | Directory reducer + `folder.apply` (isolated copy, kept on success; folder writers serialize) | Workable for **latest** documents stored as JSON files in a session folder. Gaps: no delta/base history (needed for `pi.agent` `asOf` forks and `snapshotAsOf`), no commit `Seq`, no operation batches for watches, table invariants (immutable entries, global IDs, owner liveness) still need crisp checks, and a reducer is model-run so the serialized writer would be held through a model call. |
| Watches and commit listeners | `EventLoop` events (`context.after`, `wakeAt`) | Map each watch/commit listener to events dispatched into the consumer's loop; overflow coalescing matches pi's convergent (not audit) contract. |
| JSONL backend | `TerminalSessionStore` (atomic checkpoint + journal, single writer) | Too narrow: no indexed tables, scans or document history; not a substitute. |

---

## Owner decisions

1. **Reuse or reimplement the durability layer.** pi-durable's `session/` and `storage/` are already host
   TypeScript with a conformance suite; importing them as the `durable` service (with Chord) keeps crash-recovery
   semantics exact. Reimplementing on EventLoop/`folder.apply` loses rewindable history and needs the table
   invariants rebuilt. Recommendation: reuse.
2. **How natural-language code writes.** Named crisp commit procedures run on the line (recommended), raw
   `commit(tx => …)` in eval (Chord drafts must then work inside the eval sandbox), or declarative write plans with
   optimistic guards. This decides how every other group's "commit" steps are written.
3. **Backend.** SQLite (the coding-agent frontend's choice, `openNodeSqliteStorage`), JSONL, or memory.
4. **Context derivation placement.** CRITERIA lists context assembly as harness behaviour; H2's mechanics are exact
   and long-input, so I recommend crisp `deriveContext` with natural-language policy around it.
5. **Chord UI bridges.** `documentState`/`CommittedStateSource` marked out (and needed by the conversation view and
   task graph only if the natlang frontend binds Chord states).

## Counts (rows in the tables above)

| decision | rows |
|---|---|
| fn | 0 |
| inline | 1 |
| implicit | 0 |
| crisp | 14 |
| service | 91 |
| out | 4 |
| total | 110 |

No unit of this group is an `fn`: its harness semantics (H1–H8) are rules that the natural-language functions of the
generation, tool, compaction, submission and scheduler groups must state, plus the crisp `deriveContext`
recommended in H2 (counted by the context group, not here).
