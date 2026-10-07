# pi: a port of pi-durable to natlang, unit by unit

Status: for owner review. Nothing has been translated yet.

Source: pi at `f10993b`. Two parts are ported:

- `packages/durable`: the harness. Its normative spec is `docs/spec.md`, cited here as §n.
- `packages/coding-agent/src/experimental/durable`: the coding agent's frontend.

The appendices in `port/` are the inventory. They list every unit with its source lines, its exact rules and a first
decision:

1. `port/1-agent-turn.md`: generation, tool, compaction, context, prompt, agent, registry, provider, usage, live.
2. `port/2-scheduling.md`: scheduler, Harness API, submissions, inbox, task graph, views, output. Rules R1–R9.
3. `port/3-session-state.md`: Session, transactions, documents, storage. Rules H1–H8.
4. `port/4-tools-frontend.md`: read, write, edit, bash, the execution environment, the frontend, subagent.
5. `port/5-libraries.md`: what pi-durable uses of pi-ai and Chord. The `ai` service.

This document makes the final decisions. Where it differs from an appendix, it says so; "Overrides" lists every such
case.

## What is ported

- **The harness's behaviour.** That covers:
  - the three durable task kinds that make up a run (`pi.generation`, `pi.tool`, `pi.compaction`);
  - admission of submissions, and steering and follow-up at turn boundaries;
  - context derivation, system prompt planning and compaction;
  - the tool-call policy.
- **The coding agent.** That covers:
  - the tools read, write, edit and bash, and the subagent;
  - pi's system prompt and settings;
  - a print-mode CLI.
- **Reused as host code, not ported.** pi-durable's Session, SQLite storage, task scheduler, registry and Harness API,
  and pi-ai's providers. They are durable-execution and provider infrastructure (see Policy).
- **Out.** The TUI and its view model, the experimental agent-event adapter, the Chord UI bridges, and powershell. The
  Harness API stays pi's, so pi's TUI could run on the port unchanged.

The agent that the harness hosts is a provider model, reached through pi-ai (`ai.turn`). The natural-language
functions are the harness around it. They run on natlang's executor, a small fast model. None of them is the agent.

## Decisions

- **fn**: its own natural-language function.
- **inline**: instructions inside its caller.
- **implicit**: left to the model.
- **crisp**: a TypeScript helper or value in a callable folder, or a record type.
- **op**: a write operation of the `durable` service. Crisp, applied by the host inside a commit.
- **host**: pi-durable's or pi-ai's own TypeScript, reused behind a service.
- **out**: not ported.

## Policy

- **Natural language: the harness's behaviour.** That is everything that decides:
  - what the agent model sees: context derivation, the compaction cut, the summary request, the system prompt plan;
  - what happens next in a run:
    - each task phase, and the classification of a response (deferral, retry, overflow, failure);
    - the end of a turn and the tool round's control;
    - steering and follow-up selection, and admission;
  - how a tool call is handled: lookup, repair, validation, hooks, intent, replay, result;
  - each tool's procedure, and the subagent.

  A deterministic rule is natural language when it is the harness's semantics, as dominators are the compiler's
  content. The model runs it in eval from the steps the instructions give.
- **Crisp: plumbing.**
  - record types and constants;
  - verbatim model-visible text: tool declarations, the system prompt sections, the summarization prompts;
  - byte, line and character mechanics: truncation, UTF-8 sizes, line endings, BOM, tolerant-match normalization,
    diffs, image sniffing;
  - fixed transcript formats: the serialized conversation and the `<harness>` diagnostics block;
  - the extension linker (agent resolution, settings merge). These are configuration merges that the host's per-phase
    dispatch needs.
- **Host: durable execution and the outside world.**
  - **Reused from pi-durable:**
    - the Session line and the storage;
    - the scheduler: reservation, the phase driver, recovery, abort cascades, ownership walks, idle;
    - the registry.

    These run inside the commit line, on every phase. They are host code for the same reason nldb's redo log is.
  - **The outside world:**
    - `ai` is pi-ai;
    - `env` is files and processes;
    - `resources` is context files and skills.
- **Natural language never runs inside a commit.** pi-durable's invariant 4 forbids external effects inside the commit
  line, and a model call is one. A function therefore:
  1. reads committed state;
  2. decides;
  3. gives the host one atomic list of write operations, with guards.

  The host checks the guards inside the commit. A failed guard means the state changed under the function: it reads
  again and decides again.
- **Analyses are computed by a caller and passed down where one caller can do it.** Where several functions need the
  same one, see "Shared functions".
- **Executors are small models.**
  - Each function is one task with a typed contract, and its rules are spelled out.
  - Verbatim text is a crisp value that the function copies, never retypes.
  - Byte and line arithmetic is never left to the model.

## Architecture

### The host and the task kinds

`index.ts` opens a pi-durable Harness on SQLite storage. Its registry's built-in tasks are the natural-language ones.
This needs one patch to pi-durable, because its registry hard-codes `BUILTIN_TASKS`.

Each task kind is registered with its name, version and initial checkpoint. Its phase handlers call the kind's
natural-language entry with the phase facts:

```ts
{ task: { id, kind, conversationId, input, checkpoint }, mode: "run" | "abort",
  agent, settings, sessionId, now }
```

- `agent` is the phase's resolved agent. It holds:
  - the model and thinking level;
  - the tools, as declarations with `replay` and `executionMode`;
  - the section renderers, the instructions and the cwd.
- **The entry returns once it has committed the task's next state.** A signalled invocation stops without writing.
  The scheduler's step rules then apply unchanged (appendix 2, R5), so a phase that returns without durable progress
  faults its task.
- **A task definition's identity is a hash of its `.nl` sources plus its version.** pi's handover rule compares
  definitions, and this gives it something to compare.

### Services

**`durable`, bound to the task invocation.**

- **Reads of committed state:**
  - `scanContext(conversation, at?)` returns `{head, entries}`, the active range in the raw;
  - `entry`, `task`, `outcomes(taskIds)`;
  - `live()`, `inbox()`, `submission(id)`, `submissionByRequest`.
- `commit(ops, expect?)` applies an atomic list of write operations (below) and returns the references it created.
- `submit(conversation, draft)` runs admission (`admit`) and returns the submission ID.
- **Phase utilities:**
  - `hooks(name)` returns the selected extensions' handlers for this task kind, in extension order;
  - `report(error)`;
  - `sleep(until)` and `now()` on the durable clock;
  - `memo`.

**`ai`** is appendix 5, A.11: `model`, `available`, `turn(model, messages, options, live?)`, `poll`, `cancel`,
`failure`, `estimateTokens`, `validateArguments` and `newSessionId`. With `live`, `turn` publishes the throttled
partials into `pi.live` itself.

**`tools`, bound to a tool task.** `execute(tool, args, limits)` runs the tool's implementation with its execution
api, output buffer and progress throttle. It returns:

```ts
{ result | error, retained, diagnostics, details, durationMs }
```

**`env`, bound to a tool call.** pi's ExecutionEnv (file system and shell), plus:

- `toolPath(path)` and `readPath(path)`: pi's path normalization and read-path variants;
- `lock(path, body)`: the file mutation lock;
- `runShell(command, timeout)`: execution with streaming into the call's output, and spilling.

**`resources`** gives the project context files and the skills for a cwd.

### Write operations

A commit is a list of these operations. The host applies them atomically on the Session line, in order. An operation
may name its result (`as: "t1"`), and a later operation may refer to it (`"$t1"`).

| Operation | Effect | Invariant it carries |
|---|---|---|
| `appendAssistant(message)` | Records the usage under `models["provider/model"]`, then appends a `pi.assistant` entry. | Every assistant entry's usage is counted. |
| `appendToolResult(call, result \| error {code, message}, durationMs?)` | Renders the diagnostics as a `<harness>` text item, records the tool's usage, and appends a `pi.tool-result` entry. | The stored message is exactly what the model sees. |
| `appendUser(content)`, `appendSystem(message)`, `appendEntry(draft)` | Append an entry. `appendEntry` is for resets and summaries (`head`). | Only the run's task appends to a busy conversation. |
| `convertPartial()` | Appends a committed partial, if any, as an aborted assistant message. | The scheduler's cleanup uses the same rule. |
| `createTask(kind, input, owner)` | Creates a task. `owner` is `self` or `conversation`. | — |
| `createCompaction(reason, owner?, instructions?, ifNone?)` | Creates a compaction task and its status entry. | — |
| `live(set, delete)`, `slot(callId, …)`, `compactionStatus(…)` | Edit fields of `pi.live`. | — |
| `startRun(inputs)`, `endRun(settlement)`, `handOver(to)`, `runInputs(append)` | Run control. | The run's inputs settle with the run. |
| `place(selection)` | Places a boundary's selection. Writes are appended, and stale writes settle. User entries are appended and placed. The items leave the inbox. Returns the placed input IDs. | Positional removal; the remaining items keep their order. |
| `queue(draft)`, `settle(submission, …)`, `createSubmission(…)` | Admission writes. | Dedupe by request ID. |
| `usage(bucket, key, usage)` | Adds to `pi.usage`. | Only own keys count. |
| `configure(change)`, `addTools(names)` | Edit `pi.agent`. | — |
| `next(state)` | Sets the task's next state: `running {checkpoint}`, `waiting {on, policy, checkpoint}`, or `terminal {outcome}`. | The scheduler's gates (appendix 2, M1). |

`expect` holds the guards: `run` (the run's task, or none), `inbox` (the item IDs read), and `tail` (the newest entry
read).

### A run, as the functions see it

1. The user submits. `admit` places the input, or queues it, and starts a run with a `pi.generation` task.
2. `generation` runs in phases:
   - **prepare**: derive the context, render and plan the system prompt, and maybe compact first;
   - **request**: call the agent model;
   - **classify**: decide what the response means.
3. Then one of two things happens:
   - **The response is an answer.** `answer` applies the final boundary, settles the inputs, and starts a run for any
     queued input.
   - **The response calls tools.** `start-tool-round` creates one `pi.tool` task per call. Each runs `tool`: `call`,
     then `run`, which runs the tool's own function. When the tasks have settled, `finish-tool-round` applies the
     post-tools boundary and hands the run to the next generation.

## Layout

```
applications/pi/
  PORT.md, port/              this plan and the inventory
  natlang.json, package.json
  index.ts                    host: Harness on SQLite, the task kinds, the services
  main.ts                     print-mode CLI
  services.ts                 durable, ai, tools, env, resources, with their declarations
  types.ts                    records with doc comments: messages, entries, live state, checkpoints, ops
  text/                       package pi-text: crisp text mechanics shared by the tools and the tool task
  harness/                    shared functions (see "Shared functions")
    context.nl  context/texts.ts
    cut.nl
    estimate.nl
    boundary.nl
  generation.nl
  generation/
    prepare.nl  prepare/plan-system.nl
    request.nl
    classify.nl
    answer.nl
    start-tool-round.nl
    finish-tool-round.nl
    abort.nl
  tool.nl
  tool/
    call.nl
    run.nl  run/result.ts
    from-slot.ts
  compaction.nl
  compaction/
    select.nl
    summarize.nl  summarize/transcript.ts
  admit.nl
  extensions/
    coding-tools/  index.ts (declarations), read.nl, read/{selection,image}.ts, write.nl,
                   edit.nl, edit/{apply.nl, apply/text.ts, diff.ts, prepare.ts}, bash.nl
    pi-prompt/     sections.ts
    subagent/      index.ts, subagent.nl
  tasks/                      live coding tasks (kept)
```

That is 26 natural-language functions.

## Functions and their contracts

| Function | Arguments | Result |
|---|---|---|
| `generation` | phase facts | Commits the task's next state. |
| `generation/prepare` | facts, checkpoint, view, model | Commits either a wait on a blocking compaction, or the system entries plus the `request` checkpoint. |
| `generation/prepare/plan-system` | the view's system state (head, shown sections, offered tools, system entry IDs), desired sections, desired tools, now | 0, 1 or 2 system entry drafts. |
| `generation/request` | facts, checkpoint, messages through the cutoff, model | The terminal `AssistantMessage`. It first commits the attempt's live state. |
| `generation/classify` | facts, request record, message | Commits poll, compaction, retry or failure. Otherwise returns `answer` or `tools`. |
| `generation/answer` | facts, message | Commits the end of the turn. |
| `generation/start-tool-round` | facts, message, offered tool names, the agent's tools | Commits the round. |
| `generation/finish-tool-round` | facts, the `tools` checkpoint, outcomes, slots | Commits the next generation, or the end of the run. |
| `generation/abort` | facts, checkpoint | Commits `aborted`. |
| `tool` | phase facts | Commits the task's next state. |
| `tool/call` | facts, call, resolved tool | Either commits a settled result, or commits the intent and returns `{args}`. |
| `tool/run` | facts, call, tool, args | Commits the settlement. |
| `compaction` | phase facts | Commits the task's next state. |
| `compaction/select` | facts, view, model | Commits completion with nothing to do, a summary placed, or the `summarize` checkpoint. |
| `compaction/summarize` | facts, pinned request, view through the tail | Commits a summary, a retry or a failure. |
| `admit` | conversation, draft, now | Commits the admission and returns the submission ID. |
| `context` | head marker, entries in range | `ContextView`: `{head, entries, contributions, messages, sources, sections, tools}`. `sources` gives each message's entry; `sections` and `tools` are the shown prompt state. |
| `cut` | view, keepRecentTokens | The index of the first entry kept, or `null`. |
| `estimate` | view, extra messages | Tokens. |
| `boundary` | inbox items, queue modes, `postTools` or `final`, active start | A selection: `{writes: [{id, stale}], users: [ids], reset, final}`. |
| `read`, `write`, `edit`, `bash`, `subagent` | the tool's parameters | `ToolExecutionResult`. |
| `edit/apply` | normalized content, edits, path | `{base, content}`, or the error message. |

## Unit decisions

The appendices give each unit's exact rules. The tables below give each unit's decision.

### generation.ts: the `pi.generation` task (appendix 1)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Records: input, checkpoints (prepare, request, retry, poll, tools), result, the in-memory request | crisp | `types.ts` | Data. |
| Registration: `pi.generation`, version 1, initial `prepare {attempt: 1}` | host | `index.ts` | A registration record. |
| Phase routing. The phase's model: the agent's in prepare, the pinned one in request and poll. With no model or an unknown one, the run fails with `no_model`. | fn | `generation` | The model check moves up from three phases into the entry. |
| `prepare`, in order: the compaction outcome after a blocking compaction; the context; sections; the system plan; the threshold; the request commit | fn | `generation/prepare` | A decision chain that ends in one next state. |
| Section rendering: each section in order, wrapped in `<key>` tags unless untagged. A section that throws keeps its shown text, and the error is reported. | inline | `generation/prepare` | A short loop with two rules. The renderers are extension code. |
| System plan: head rebaseline, section patches, tool changes | fn | `generation/prepare/plan-system` | Override: crisp → fn. It decides what the model is told about prompt changes. |
| Threshold: blocking or background compaction, from the estimate, the window, the reserve and background tokens, and whether a cut exists | inline | `generation/prepare` | Arithmetic, stated where it is used. |
| `request`: convert a leftover partial, set the live generation, run the `beforeRequest` chain, build the options, call `ai.turn` with live partials | fn | `generation/request` | Its own contract: committed context in, terminal message out. |
| Streaming with throttled partials (`streamResponse`) | host | `ai.turn(…, live)` | Streams, timers and throttled commits. |
| `retry`: sleep, then the next attempt goes back to prepare | inline | `generation` | Two steps. |
| `poll`: sleep until `pollAt`, call `ai.poll`, then classify | inline | `generation` | Three steps. |
| `tools`: start the next call of a sequential round, or finish the round | inline | `generation` | One commit, no decision. |
| `classify`, in order: deferred → poll; `afterResponse`; tool use → tools; stop or length → answer; overflow → blocking compaction; retryable → retry; anything else → fail | fn | `generation/classify` | The retry and error policy. Override: its commits were crisp helpers; now they are operation lists it builds itself. |
| `answer`: `onYield`; final boundary; continuation or end of the run; a new run for placed input | fn | `generation/answer` | The end-of-turn policy. |
| `startToolRound`: the offered-tools check, sequential or parallel, slots, tasks | fn | `generation/start-tool-round` | The round's setup rules. |
| `finishToolRound`: controls, `afterTools`, terminate, handoff, added tools, post-tools boundary, the next generation | fn | `generation/finish-tool-round` | The round's control policy. |
| Abort handler: cancel a deferred response, write results for calls never started, end the run `aborted` | fn | `generation/abort` | Its own trigger and contract. |
| `readCalls` | inline | `generation/abort` | One lookup. |
| `createToolTask`, `createGeneration`, `startRun`, `handOver` | op | `createTask`, `startRun`, `handOver` | One write each. |
| `failModelError`, `failNoModel` | op, texts inline | `endRun` + `next`, with the message texts in each caller | Override: helper → operations. |
| `convertPartial` | op | `convertPartial` | Shared with the scheduler's cleanup, which runs inside a commit. |
| `appendAssistant` | op | `appendAssistant` | Carries the usage invariant. |
| `DEFAULT_POLL_AFTER_MS` (5000) | inline | `generation/classify` | One constant in one rule. |

### tool.ts: the `pi.tool` task (appendix 1)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Records: input `{assistant, callId}`; checkpoints `call` and `execute {arguments, replay}`; result `{entryId, control?}` | crisp | `types.ts` | Data. |
| Registration: `pi.tool`, version 1 | host | `index.ts` | A registration record. |
| Phase routing. Read the call from the assistant entry; a missing call faults with `Entry <a> has no tool call <id>`. | fn | `tool` | — |
| `call`, in order: lookup, repair, validation, `beforeTool`, validation again, intent | fn | `tool/call` | The order of checks and hooks is the tool-call policy (§7.3, §8.4). |
| Repair (`prepareArguments`) | inline | `tool/call` | One step. Each tool's repair is its own crisp code. |
| Validation | host | `ai.validateArguments` | pi's exact coercion and message. |
| `execute`, recovery after intent: rerun only when both the stored replay and the current one are `safe`; otherwise settle `failed` with "interrupted", from the slot | inline | `tool` | One rule. |
| Abort: settle `aborted`, from the slot | inline | `tool` | One sentence. |
| `run`: limits, `tools.execute`, error to result, assembly, `afterTool`, bounding, settlement | fn | `tool/run` | Error policy and hooks. |
| Execution api, output buffer, progress throttle, waiters | host | `tools` | Buffers, timers and promises. |
| Result assembly: content from retained output, details, the order of diagnostics | crisp | `tool/run/result.ts` `assemble` | A fixed format. |
| Bounding (`boundContent`, `truncated`) | crisp | `tool/run/result.ts` `bound` | Byte and line truncation. |
| `settle` | op | `appendToolResult`, `slot`, `next` | Writes. |
| `fromSlot` | crisp | `tool/from-slot.ts` | A format. |
| `harnessError`, the error codes | op | `appendToolResult` with `error {code, message}` | A format. |
| `appendToolResult`, `renderDiagnostics` | op | `appendToolResult` | A format, plus the usage invariant. |
| `errorText` | implicit | — | Trivial. |

### compaction.ts: the `pi.compaction` task (appendix 1)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Records; `TOOL_RESULT_MAX_CHARS`; the summary prefix and suffix | crisp | `types.ts`, `compaction/summarize/transcript.ts` | Data. |
| The summarization prompts, verbatim | crisp | `compaction/summarize/transcript.ts` | Model-visible text. |
| Registration: `pi.compaction`, version 1 | host | `index.ts` | A registration record. |
| Phase routing. Model resolution. `no_model` removes the status. | fn | `compaction` | — |
| `select`: the cut, `beforeCompact`, pinning the summary request | fn | `compaction/select` | A decision chain with a hook (§8.7). |
| `summarize`: the summarized messages, the request, then a summary, a retry or a failure; usage | fn | `compaction/summarize` | Its own classification rules. |
| `retry` | inline | `compaction` | Two steps. |
| Abort | inline | `compaction` | One sentence. |
| `createCompaction` | op | `createCompaction` | A write. |
| `selectCut`, `isCandidate` | fn | `harness/cut` | Override: crisp → fn. What compaction keeps is harness content. |
| `estimateContext` | fn | `harness/estimate` | Override: crisp → fn. |
| `summarizedMessages` | inline | `compaction/select`, `compaction/summarize` | The view's ordered messages from entries before the cut. A cut never separates a call from its result, so this equals pi's reordering of that prefix. |
| `summaryText`, `summaryFailure` | inline | `compaction/summarize` | A few sentences. |
| `summaryPrompt`, `serializeConversation` | crisp | `compaction/summarize/transcript.ts` | Exact formats. |
| `place`, `placeSummary` | op + host | Blocking: `appendEntry` in the completing commit. Conversation-owned: `durable.submit` as a write with request ID `compaction:<taskId>`, then the completing commit. | Override: pi does this in one commit. Here admission is a function outside the commit line, and the request ID makes the pair idempotent. |
| `complete`, `failNoModel` | op | `compactionStatus`, `next` | Writes. |

### context.ts: the model context (appendix 1, H2)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Constants: the excluded stop reasons, the missing-result text | crisp | `harness/context/texts.ts` | Data. |
| Bounds and range reads (`captureContextBounds`, `rangeQuery`, `scanRange`, `readContext`) | host | `durable.scanContext` | Storage reads. |
| Derivation, in order: newest edit per target; active entries; contributions (omit, replace, failed messages dropped); tool results after their calls, with missing ones synthesized; the system message first. Also the shown sections and offered tools, replayed from system messages. | fn | `harness/context` | Override: crisp → fn. What the model sees is the harness's first concern. The rules are spelled out from §2.1. |
| Incremental cache (`readContextFrom`, `extendRange`, `settle`, the range record) | out in the pure variant; host in the optimized one | — | An optimization of the same rules. |
| `freezeJson` | out | — | A JavaScript aliasing guard. |
| `activeEntries` | host | views | A storage read plus a filter, for UIs. |

### prompt.ts: the system prompt and tools (appendix 1, §7.4)

| Part | Decision | Unit | Why |
|---|---|---|---|
| `replaySections` | fn | part of `harness/context` (the shown sections) | The same replay as the offered tools. |
| `renderSections` | inline | `generation/prepare` | A short loop. |
| `planSystemEntries`, `planSections`, `planTools`, `systemEntry` | fn | `generation/prepare/plan-system` | Override: crisp → fn. |
| `toToolDeclaration`, `declarationsEqual` | inline | `generation/prepare/plan-system` | Compare the JSON of name, description, parameters and constrained sampling; write only those fields. |

### agent.ts, registry.ts, provider.ts, usage.ts, define.ts, live.ts (appendix 1)

| Part | Decision | Unit | Why |
|---|---|---|---|
| Defaults: retry, compaction, progress, context retention | crisp | host settings | Data. |
| The `pi.agent` document | host | — | Storage. |
| `resolveSettings` | crisp | host | A merge over the defaults. |
| `configure`, `addTools` | op | `configure`, `addTools` | Field-wise writes. |
| `createAgent` (a subagent's conversation copies its owner's agent) | host | conversation creation | Runs inside the creating commit. |
| `agentHooks` | host | `durable.hooks` | A filter in the dispatcher. |
| `resolveAgent`, `selectExtensions`, `applyWrap` | crisp | host; the result is passed to every phase as `agent` | A linker over registry data. Borderline: see owner decision 6. |
| The registry: snapshot, install, publish, `validateExtension` | host | — | Process-local state (§7.5). |
| `pi.provider`, `ensureProviderSessionId` | host | the `sessionId` phase fact | Storage. |
| `pi.usage`, `recordUsage`, `addUsage` | host, op | inside `appendAssistant`, `appendToolResult` and `usage` | Arithmetic with an own-key rule. |
| `addUsageState` | host | `Harness.usage()` | A sum. |
| `defineExtension`, `defineTool` | out | — | Type inference only. |
| `section`, `hook`, `wrapTool`, `wrapSection` | crisp | extension declarations | Constructors. |
| Live records: run, generation, slots, compaction statuses | crisp | `types.ts` | Data. |
| The `pi.live` document | host | — | Storage. |
| `endRun`, the slot operations, the compaction status operations | op | `endRun`, `slot`, `compactionStatus` | Writes. |
| `settleSchedulerOutcome` | crisp | host | Runs inside the scheduler's own commit. |

### Scheduling and the public surface (appendix 2)

| Part | Decision | Unit | Why |
|---|---|---|---|
| scheduler.ts: reservation, the phase driver, step rules, recovery, reconcile, abort cascades, ownership walks, idle, the context cache | host | reused | Override: appendix 2 makes `classifyTask`, `decideStep`, `reconcile`, `abortTask` and `abortConversation` natural language. They run on the commit line before every phase and in every pass. A model call there breaks invariant 4 and holds the whole Session behind a model. They are the durable-execution runtime, not the harness's behaviour. Their rules are R4–R9. |
| harness.ts: the Harness and Conversation API, conversation creation, root idempotence | host | reused | `submit` and `reset` go through `admit`. |
| `admitSubmission` (R1), in order: dedupe, busy, reject, queue, direct write, direct input | fn | `admit` | The busy-conversation policy. It decides outside the commit, guarded on the run and the inbox. |
| `abortSubmission` | crisp | host | Override: fn → host. A four-way status switch. |
| Submission waits and handles | host | — | Waiters. |
| `applyBoundary` (R2): the selection | fn | `harness/boundary` | Override: split. Choosing the writes, steers and follow-ups, detecting a reset, and judging staleness are a function of the inbox. |
| `applyBoundary`: the placement | op | `place(selection)` | Placing is writing. |
| `prepareBoundary` | host | reads | Read ordering is commit mechanics. |
| `removeInboxItem`, `withdrawQueuedInputs` | crisp | host | Exact list edits used by the abort runtime. |
| task-graph.ts, view.ts, output.ts (bounding, pacing), util.ts, json.ts | host | — | Observation and machinery. |
| events.ts | out | — | Experimental, unused by the frontend. |
| types.ts | crisp | `types.ts` | Records. |

### Session and storage (appendix 3)

- **Host, reused:** the Session line, transactions, documents, forks, observation, and the SQLite backend.
- **The record rules H1–H8 are stated where they are used:**
  - entry kinds;
  - heads and edits;
  - forks;
  - the submission and task lifecycles;
  - the transaction discipline.

  The functions that write and derive these records state the rules, and the operations enforce them.

### Tools and the environment (appendix 4)

| Part | Decision | Unit | Why |
|---|---|---|---|
| The `coding-tools` extension: read, write, edit, bash, in that order | crisp | `extensions/coding-tools/index.ts` | A declaration. Its order is model-visible. |
| Tool declarations, verbatim | crisp | `extensions/coding-tools/index.ts` | Compared at every request (§7.4), so they must be byte-stable. |
| `requireEnv` | inline | each tool | One check. |
| `normalizeToolPath`, `resolveToolPath`, the read-path variants | host | `env.toolPath`, `env.readPath` | Path hygiene next to the env's own resolution rules. Three tools share it. |
| `read`: resolve the path, open, make up to two attempts while the file changes, select | fn | `coding-tools/read` | Its own contract and retry rule. |
| Read selection (`readTextSelection`, `readHead`, the slice rules, the continuation messages) | crisp | `read/selection.ts`, taking the opened reader | Exact slicing, byte counts and formatted numbers. |
| Image sniffing | crisp | `read/image.ts` | Byte rules. |
| Truncation: `truncateHeadOf`, `formatSize`, `utf8ByteLength`, limits | crisp | the `pi-text` package | Shared with the tool task. |
| `write`: resolve, lock, write, message | fn | `coding-tools/write` | Its own contract. See owner decision 7. |
| `edit`, in order: validate, resolve, lock, info, read, BOM and line endings, apply, write, diff | fn | `coding-tools/edit` | Ordered steps with exact messages. |
| Applying the edits, in order: empty, find (exact, then tolerant), unique, overlap, replacement, no change; the verbatim messages | fn | `coding-tools/edit/apply` | Override: crisp → fn. This is the edit tool's semantics. The mechanics stay crisp. |
| Line endings, BOM, tolerant normalization, find, count, replacements, the line-preserving replacement | crisp | `edit/apply/text.ts` | Index arithmetic and character classes. |
| Diff string, unified patch | crisp | `edit/diff.ts` | For renderers only. |
| `prepareEditArguments` | crisp | `edit/prepare.ts` (the tool's `prepareArguments`) | A pure repair. |
| `bash`: check the timeout, run, map the outcome to messages | fn | `coding-tools/bash` | Its own outcome rules. |
| Shell execution with streaming, spill and window | host | `env.runShell` | Processes and callbacks. |
| The file mutation lock | host | `env.lock` | Concurrency. |
| powershell | out | — | Not installed by the frontend. |
| ExecutionEnv, the Node env, decoders, the line scanner, the watcher | host | `env` | The outside world. |

### The frontend (appendix 4)

| Part | Decision | Unit | Why |
|---|---|---|---|
| main.ts, runtime.ts, tui.ts, the view model | out | — | Presentation. pi's TUI can run on the port, which keeps its Harness API. |
| Print-mode CLI: open a session, submit, wait, print the answer and tool activity | host (new) | `main.ts` | For evaluation and tests. |
| Session directories and the lock | host | `index.ts` | Storage location. |
| harness-setup: HTTP, settings, registry composition, environments, initial model | host + crisp | `index.ts` | Configuration and the catalog. |
| The `pi-prompt` extension: seven sections, verbatim, with their conditions | crisp | `extensions/pi-prompt/sections.ts` | Must be deterministic: any change appends a system delta and breaks provider caches (§7.4). |
| Loading context files and skills | host | `resources` | File reads. |
| `subagent`, in order: find or create the child (one commit), details, submit with a request ID, wait, answer text | fn | `extensions/subagent/subagent` | Delegation with its own replay rule. |
| `ensureChildConversation` | crisp | an op of the tool's api | A commit body that reads before it writes. |

## Shared functions (owner decision 3)

Four functions are needed by more than one other:

- `context`: by `generation` and `compaction`;
- `cut`: by `generation/prepare`, `generation/classify` and `compaction/select`;
- `estimate`: by `generation/prepare`;
- `boundary`: by `generation/answer`, `generation/finish-tool-round` and `admit`.

natlang lets a function call only its own companion folder. The options:

1. **A `uses:` frontmatter key (new primitive; recommended).** A function lists the package functions it may call
   besides its folder, for example `uses: [harness/cut]`. The list is explicit, per function, and there is one source.
2. **Copies in each folder.** They will drift.
3. **Compute in the common caller and pass the result down.** This works for `context` inside `generation`. It does
   not work for `cut`, which classify needs only after an overflow. Nor for `boundary`, which runs after hooks inside
   `answer`.
4. **A service whose host implementation calls the function.** This hides harness code behind a service, and services
   stand for the world outside the program.

Crisp code that several folders share (UTF-8 sizes, truncation) goes in a declared workspace package, `pi-text`. Crisp
modules may import declared packages.

## Overrides of the appendices

1. **Natural-language functions, with crisp in the optimized variant:**
   - context derivation (`deriveRange`, `selectActive`, `contribute`, `orderToolResults`, `missingResult`,
     `leadWithSystem`, `replaySections`, `getCurrentTools`): `harness/context`;
   - the cut (`selectCut`, `isCandidate`): `harness/cut`;
   - the estimate (`estimateContext`): `harness/estimate`;
   - the system plan (`planSystemEntries`, `planSections`, `planTools`): `generation/prepare/plan-system`.
2. **Commit helpers become operations.** The classify commits, `failRun`, `settle`, `placeSummary`, `complete` and the
   slot and status helpers are now write operations. The functions compose them.
3. **The scheduler's policy is host code:** `classifyTask`, `decideStep`, `reconcile`, `abortTask`,
   `abortConversation` and `abortSubmission`.
4. **`applyBoundary` splits** into the `boundary` selection function and the `place` operation.
5. **Summary placement for background compactions** goes through admission, so it takes two commits.
6. **The edit tool's matching policy** (`applyEditsToNormalizedContent`) becomes a function, `edit/apply`.
7. **The no-model check** moves from each phase into the entries.

## The optimized variant

The original request asks for a pure end-to-end version and an optimized one. Here the pure version runs every
function above.

| | Per model turn | Per tool call |
|---|---|---|
| Executor calls in the pure version | about 10 | 4 |

The optimized version cuts that down. It keeps the same contracts and swaps in crisp implementations of:

- the hot-path analyses: `context` (with pi's incremental cache), `cut`, `estimate` and `plan-system`;
- the entries' routing;
- `read`, `write` and `bash`.

How a build selects the variant follows from owner decision 3.

## Verification

- **Conformance.** pi-durable's harness suites run against the port with the natural-language task kinds substituted:
  generation, generation-recovery, compaction, context, prompt, inbox, submissions, tools, tools-recovery, structured
  and tasks. That is 322 tests.
  - The agent is pi's faux provider. The harness runs on a real executor.
  - The tests check exact entries and documents, so they test whether the functions are exact.
- **Scripted unit tests** per function, with a scripted executor, as in nldb.
- **Live runs** on the coding tasks in `tasks/`, with a provider model as the agent.

## Owner decisions

1. **Reuse pi-durable's host.** That is the Session, SQLite storage, the scheduler, the registry and the Harness API,
   vendored at `f10993b`, with one patch so the built-in tasks can be injected. *Recommended.*
2. **How functions write.** Recommended: operation lists with guards. The alternatives:
   - named crisp procedures, one per whole step, which moves the commits' content into crisp code;
   - raw `commit(tx => …)` in eval, which puts a model inside the commit line.
3. **Shared functions.** The `uses:` primitive. *Recommended.*
4. **The scheduler's policy as host code** (override 3). *Recommended.*
5. **Derivations as natural language** in the pure variant, crisp in the optimized one (override 1). This costs
   executor calls per turn (see the optimized variant). *Recommended.*
6. **Agent resolution stays crisp.** It is a linker over registry data, needed by the host's hook dispatch. The
   alternative is natural language at every phase. *Recommended crisp.*
7. **Tools.** Recommended: tools as functions in the pure variant; `read`, `write` and `bash` crisp in the optimized
   one. `write` is the thinnest: resolve, lock, write, message.
8. **Round cap.** pi has no limit on tool rounds; a run's rounds are durable tasks, and abort bounds them.
   *Recommended: no cap, as pi.*
9. **Summary writer.** The conversation's own model, through `ai.turn`, with pi's prompts. *Recommended.*
10. **Extensions in natlang.** An extension is a folder with:
    - declarations (crisp);
    - tool implementations (functions or crisp);
    - sections (crisp renderers or literal text, because renderers must be deterministic);
    - hooks (functions or crisp).

    Recommended: wrappers crisp only; reload as pi.
11. **Presentation state in `pi.live`.** Keep the partials and tool progress. The host writes them (`ai.turn`,
    `tools.execute`); functions never do. *Recommended.*
12. **Tool replay.** Keep `replay: safe/unsafe` and the intent checkpoint. Every tool declares whether it can rerun.
    *Recommended.*
13. **Storage.** SQLite, as the frontend. *Recommended.*
14. **Prompt text.** Recommended: verbatim, including the false `PI_*` bullet, so the port and pi-durable can be
    compared on identical prompts. The alternative is to drop that bullet.
15. **Per-model compaction overrides.** Ignored, as in pi's frontend. *Recommended.*
16. **Executor failures.** When the executor fails a phase (failed, blocked, or a type error), rerun the phase once
    from its checkpoint, then fault the task, as pi faults a phase that throws. *Recommended.*
17. **Models for the live runs.**
    - Agent: the teacher at :8082, as an OpenAI-compatible provider in pi-ai's `models.json`.
    - Executor: the same endpoint, or a smaller model.

## Changes from today

- **Removed.** The current `applications/pi`, which has `pi.nl` as the agent, plus `pi/`, `harness/`, `pure.ts`,
  `agent.ts`, `tools.ts` and `services.ts`. This layout replaces them.
- **Kept.** `tasks/` and the live scripts.
- **The live batch.** The batch armed for the next teacher window still runs the current app, so its pi results
  describe the design this replaces.
- **Skills.** The skills' "An agent harness as a program" pattern changes with this port.
