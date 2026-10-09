/**
 * The records the harness functions read and write. Each alias reaches the model with its doc comments, so the rules a
 * record carries are written here once and the functions refer to them.
 *
 * Shapes follow pi-durable (`vendor/durable`, spec `docs/spec.md`, cited as §n) and pi-ai. Identifiers (entries, tasks,
 * submissions, conversations) are positive integers that grow with time: a larger ID is newer.
 */

/** Any JSON value. */
export type JsonValue = unknown;
export type JsonObject = Record<string, JsonValue>;

// ---------------------------------------------------------------------------------------------------------------
// Models and messages (pi-ai)

/** A model in the pi-ai catalog. */
export type ModelRef = { provider: string; modelId: string };
/** "off" requests no reasoning. */
export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/** What the harness rules need to know about a model (`ai.model(ref)`). */
export type ModelInfo = {
  provider: string;
  modelId: string;
  name: string;
  /** Tokens one request may hold; 0 when unknown (then no compaction threshold applies). */
  contextWindow: number;
  /** Longest answer the model can produce; 0 when unknown. */
  maxTokens: number;
  /** False: only "off" exists. */
  reasoning: boolean;
};

export type TextContent = { type: "text"; text: string; textSignature?: string };
export type ThinkingContent = { type: "thinking"; thinking: string; thinkingSignature?: string; redacted?: boolean };
export type ImageContent = { type: "image"; data: string; mimeType: string };
/**
 * A Neuralese block in a message: the standard content part (ts-host contracts.ts `ModelContentPart`), not in pi-ai's
 * own content union. It reaches the model as the block itself, never as text, so only an agent model whose declared
 * reader is a Neuralese dialect can be sent it; any other model fails the request with neuralese-unsupported-backend.
 */
export type NeuraleseContent = { type: "neuralese"; id: string; value_type?: "string" | "unknown" };
/** A call the model makes. `id` is unique within its assistant message. */
export type ToolCall = { type: "toolCall"; id: string; name: string; arguments: JsonObject; thoughtSignature?: string };

/** Token counts and cost of one model response or tool execution. */
export type Usage = {
  input: number;
  output: number;
  cacheRead: number;
  cacheWrite: number;
  cacheWrite1h?: number;
  reasoning?: number;
  totalTokens: number;
  cost: { input: number; output: number; cacheRead: number; cacheWrite: number; total: number };
};

/** A response the provider finishes later; poll it with `ai.poll`. `pollAfterMs` is the provider's suggested wait. */
export type DeferredHandle = { provider: string; modelId: string; api: string; id: string; expiresAt?: number; pollAfterMs?: number; data?: JsonValue };

/** What a model is offered as a tool: only these four fields ever enter the transcript (§7.4). */
export type ToolDeclaration = { name: string; description: string; parameters: JsonObject; constrainedSampling?: JsonValue };

/**
 * A positional prompt change (`pi.system` entry). `content` is always "". `sections` sets section texts by key (a
 * string sets or re-adds, null deletes). `toolsAdded` offers tools; `toolsRemoved` withdraws them by name.
 */
export type SystemMessage = {
  role: "system";
  content: string;
  sections?: Record<string, string | null>;
  toolsAdded?: ToolDeclaration[];
  toolsRemoved?: { name: string }[];
  timestamp: number;
};
/** User content: a string, or text, image and Neuralese items. */
export type UserContent = string | (TextContent | ImageContent | NeuraleseContent)[];
export type UserMessage = { role: "user"; content: UserContent; timestamp: number };
/**
 * A provider response. Store it exactly as returned: it carries provider fields not listed here (responseId,
 * signatures) that are needed to send it back.
 * - stopReason "toolUse": `content` holds the tool calls to run; "stop" or "length": a final answer;
 * - "error": the request failed, `errorMessage` says why (see `ai.failure`); "aborted": cancelled;
 * - "deferred": the provider finishes it later; `deferred` is the handle to poll.
 */
export type AssistantMessage = {
  role: "assistant";
  content: (TextContent | ThinkingContent | ToolCall | NeuraleseContent)[];
  api: string;
  provider: string;
  model: string;
  usage: Usage;
  stopReason: "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred";
  errorMessage?: string;
  deferred?: DeferredHandle;
  timestamp: number;
};
export type ToolResultMessage = {
  role: "toolResult";
  toolCallId: string;
  toolName: string;
  content: (TextContent | ImageContent | NeuraleseContent)[];
  details?: JsonValue;
  usage?: Usage;
  isError: boolean;
  durationMs?: number;
  timestamp: number;
};
export type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

// ---------------------------------------------------------------------------------------------------------------
// Transcript entries (§2.1, §8.1)

/** Overrides one earlier entry's contribution to model context: "omit" drops it, "replace" contributes `messages`. */
export type ContextEdit = { target: number; action: "omit" | "replace"; messages?: Message[] };

/**
 * One transcript entry. Kinds:
 * - "pi.user": `model` is [UserMessage], user input;
 * - "pi.assistant": `model` is [AssistantMessage], a provider response with any stop reason;
 * - "pi.system": `model` is [SystemMessage], a prompt or tool change;
 * - "pi.tool-result": `model` is [ToolResultMessage]; `data.diagnostics` holds the structured diagnostics;
 * - "pi.reset": `head` is its own ID; `model` absent or [UserMessage] with handoff text: a new context starts here;
 * - "pi.compaction": a summary; `model` is [UserMessage] with the wrapped summary, `head` is the first entry kept.
 *
 * An entry with `head` is a head marker: model context starts at the newest head marker, which contributes its own
 * messages first, followed by the entries from ID `head` on that carry no `head` themselves.
 */
export type EntryRecord = {
  id: number;
  conversationId: number;
  kind: string;
  model?: Message[];
  data?: JsonValue;
  head?: number;
  edits?: ContextEdit[];
  /** The task that appended it. */
  byTaskId?: number;
};

/** An entry to append. `head: "self"` makes it a head marker that starts at its own new ID. */
export type EntryDraft = { kind: string; model?: Message[]; data?: JsonValue; head?: number | "self"; edits?: ContextEdit[] };

/**
 * A conversation's model context as of a tail entry (§2.1). Derived from committed entries by `context`.
 * - `head`: the newest head marker at or before the tail, if any;
 * - `entries`: the active entries: the head marker first (when there is one), then every entry from its `head` through
 *   the tail that carries no `head`; the last one is the tail;
 * - `contributions`: per entry of `entries`, its model messages after edits, without assistant messages whose stop
 *   reason is aborted, error or deferred;
 * - `messages`: what the next request sends: all contributions in order, each assistant message's tool results moved
 *   right after it in call order (a missing result synthesized as an error), and the first system message moved to the
 *   front when only user messages precede it;
 * - `sections` and `tools`: the prompt state the transcript has shown so far: section texts by key in order, and the
 *   offered tools in order, replayed from the system messages of `messages`.
 */
export type ContextView = {
  head: EntryRecord | null;
  entries: EntryRecord[];
  contributions: Message[][];
  messages: Message[];
  sections: { key: string; text: string }[];
  tools: ToolDeclaration[];
};

// ---------------------------------------------------------------------------------------------------------------
// Tools (§7.3)

/** A model-visible remark about a call (truncation, an error): never part of the tool's data. */
export type ToolDiagnostic = { severity: "info" | "warn" | "error"; message: string; code?: string };

/** What a tool result asks of the run after its round: add tools, end the run (`terminate`), or start over (`handoff`). */
export type ToolControl = { addTools?: string[]; terminate?: boolean; handoff?: string };

/**
 * A tool's result. `content` omitted: its retained output becomes the content. `details` is for UIs. `diagnostics` are
 * rendered for the model after the content. `usage` is the spend of the execution itself.
 */
export type ToolExecutionResult = {
  content?: (TextContent | ImageContent | NeuraleseContent)[];
  isError?: boolean;
  details?: JsonValue;
  diagnostics?: ToolDiagnostic[];
  usage?: Usage;
  control?: ToolControl;
};

/** How much of a tool's output the model sees: at most `maxBytes` and `maxLines`, keeping the "head" or the "tail". */
export type OutputLimits = { maxBytes: number; maxLines: number; retain: "head" | "tail" };

/**
 * A tool as the agent offers it. `replay` "safe" means an interrupted run may simply run again; "unsafe" (the default)
 * means it may have partly run. `executionMode` "sequential" makes its whole round run one call at a time.
 * `outputLimits` fields not given default to 51200 bytes, 2000 lines and "head".
 */
export type AgentTool = {
  name: string;
  description: string;
  parameters: JsonObject;
  constrainedSampling?: JsonValue;
  replay?: "safe" | "unsafe";
  executionMode?: "parallel" | "sequential";
  outputLimits?: { maxBytes?: number; maxLines?: number; retain?: "head" | "tail" };
};

/** A prompt section of the agent, rendered in order before each request. `tag` false: not wrapped in `<key>` tags. */
export type AgentSection = { key: string; tag: boolean };

/** The conversation's agent as the phase resolved it (§7.1): fixed for the phase. */
export type Agent = {
  /** Absent: no model is configured. */
  model?: ModelRef;
  thinkingLevel: ThinkingLevel;
  /** The tools a request offers, in order. */
  tools: AgentTool[];
  /** Extension sections, then "instructions" when set. */
  sections: AgentSection[];
  instructions?: string;
  cwd?: string;
};

// ---------------------------------------------------------------------------------------------------------------
// Settings (§7.1)

export type RetryPolicy = { enabled: boolean; maxRetries: number; baseDelayMs: number; maxAgentDelayMs?: number };
/**
 * Automatic compaction (§8.7). Requests block to compact above `contextWindow - reserveTokens`; a background compaction
 * starts `backgroundTokens` below that (0 disables it). A summary keeps about `keepRecentTokens` of recent context.
 */
export type CompactionPolicy = { enabled: boolean; reserveTokens: number; keepRecentTokens: number; backgroundTokens: number };
/** How many queued items of one mode a boundary places: the first ("one-at-a-time") or all. */
export type QueueMode = "all" | "one-at-a-time";
/** Request options pinned with a request. */
export type StreamOptions = {
  transport?: string;
  timeoutMs?: number;
  maxRetries?: number;
  maxRetryDelayMs?: number;
  headers?: Record<string, string>;
  metadata?: JsonObject;
  cacheRetention?: "none" | "short" | "long";
  deferred?: boolean | { window?: "15m" | "1h" | "24h" };
};
export type Settings = {
  stream: StreamOptions;
  retry: RetryPolicy;
  compaction: CompactionPolicy;
  toolExecution: "parallel" | "sequential";
  steeringMode: QueueMode;
  followUpMode: QueueMode;
};

// ---------------------------------------------------------------------------------------------------------------
// Live state (`pi.live`, §8.2) and the inbox (`pi.inbox`, §6)

/**
 * One call of the current tool round, in call order. `taskId` is set once its tool task exists; a call never started
 * has none. `status`: "pending" (not yet running), "running" (intent recorded), "done". `entry` is its result entry.
 * `output`, `droppedBytes`, `droppedLines`, `details` and `diagnostics` are the running progress the host publishes.
 */
export type ToolSlot = {
  callId: string;
  name: string;
  taskId?: number;
  status: "pending" | "running" | "done";
  entry?: number;
  output?: string;
  droppedBytes?: number;
  droppedLines?: number;
  details?: JsonValue;
  diagnostics?: ToolDiagnostic[];
};

/** A compaction in progress. `blocking`: a generation waits for it. */
export type CompactionStatus = { taskId: number; reason: CompactionReason; blocking: boolean; attempt: number; retry?: { at: number; error: string } };
export type CompactionReason = "manual" | "threshold" | "overflow";

/**
 * Run control and presentation of one conversation.
 * - `run`: present exactly while the conversation is busy. `taskId` is the generation that owns the run; `inputs` are
 *   the input submissions the run will settle.
 * - `generation`: the current request attempt; `message` is its committed partial answer, `retry` a scheduled retry,
 *   `deferred` a scheduled poll.
 * - `tools`: the current round's slots, in call order.
 * - `compactions`: compactions in progress, in task ID order.
 */
export type LiveState = {
  run?: { taskId: number; inputs: number[] };
  generation?: { attempt: number; message?: AssistantMessage; retry?: { at: number; error: string }; deferred?: { pollAt: number } };
  tools?: ToolSlot[];
  compactions?: CompactionStatus[];
};

/**
 * A queued submission, in ID order. "steer" and "followUp" items are user input waiting for a turn boundary; a "write"
 * item is an entry to append (a summary, a reset).
 */
export type InboxItem = { id: number; mode: "steer" | "followUp" | "write"; content?: UserContent; entry?: EntryDraft };

/** What a submission asks for: user input that may start a run, or an entry write. */
export type SubmissionDraft = {
  requestId?: string;
  type: "input" | "write";
  /** input: the user content. */
  content?: UserContent;
  /** input: what a busy conversation does with it: queue as a steer, as a follow-up (default), or reject. */
  whenBusy?: "steer" | "followUp" | "reject";
  /** write: the entry. */
  entry?: EntryDraft;
};

/** How a submission ended: answered ("done", with the answer entry) or "unanswered" with a reason. */
export type Settlement = { status: "done"; answer: number } | { status: "unanswered"; reason: string; detail?: JsonValue };

/**
 * A submission. "queued": in the inbox; "placed": its entry is in the transcript and a run owns it; "done" and
 * "unanswered": settled.
 */
export type SubmissionRecord = {
  id: number;
  conversationId: number;
  requestId?: string;
  type: "input" | "write";
  status: "queued" | "placed" | "done" | "unanswered";
  entry?: number;
  answer?: number;
  reason?: string;
};

/**
 * What a turn boundary places (§6), chosen by `boundary`:
 * - `writes`: every write item in inbox order; `stale` ones (a numeric `head` before the active range) settle
 *   "unanswered" instead of being appended;
 * - `users`: the steer and follow-up item IDs placed, in inbox order;
 * - `reset`: a write with `head: "self"` is placed; `final`: the boundary counts as final.
 */
export type BoundarySelection = { writes: { id: number; stale: boolean }[]; users: number[]; reset: boolean; final: boolean };

// ---------------------------------------------------------------------------------------------------------------
// Tasks (§5)

export type TaskOutcome = {
  status: "completed" | "failed" | "aborted" | "orphaned" | "faulted";
  result?: JsonValue;
  error?: { message: string; detail?: JsonValue };
  reason?: string;
};

/**
 * A task record. `owner` is the task that owns it (absent: its conversation owns it). `state.status`:
 * "pending" (ready), "running", "waiting" (on the tasks in `on`), "completing" (outcome held until owned work ends),
 * "terminal".
 */
export type TaskRecord = {
  id: number;
  conversationId: number;
  kind: string;
  version: number;
  input: JsonValue;
  owner?: number;
  background: boolean;
  abortRequested: boolean;
  state: { status: "pending" | "running" | "waiting" | "completing" | "terminal"; checkpoint?: JsonValue; on?: number[]; outcome?: TaskOutcome };
};

/**
 * A task's next state, committed with the `next` operation:
 * - running: continue at `checkpoint`;
 * - waiting: park until every task in `on` is terminal, then resume at `checkpoint`. A task created in the same commit
 *   is "$name" here, in `on` and in the checkpoint. `policy` "allSettled" waits for all; "failFast" aborts the others when one fails;
 * - terminal: done with `outcome`. While work the task owns is still live, it is held as completing.
 */
export type NextState =
  | { status: "running"; checkpoint: JsonValue }
  | { status: "waiting"; on: (number | string)[]; policy: "allSettled" | "failFast"; checkpoint: JsonValue }
  | { status: "terminal"; outcome: TaskOutcome };

/** The tool task's checkpoint: before intent ("call"), or after intent with the final arguments ("execute"). */
export type ToolCheckpoint = { phase: "call" } | { phase: "execute"; arguments: JsonObject; replay: "safe" | "unsafe" };

/**
 * The generation task's checkpoint. `compacted` (the blocking compaction this generation already ran) carries over
 * through every phase, so a generation compacts at most once.
 * - prepare: build the prompt; `overflow` is set when the last attempt overflowed and `compacted` compacted it;
 * - request: call the model with the context through entry `cutoff`, with the pinned model and options;
 * - retry: wait until `until`, then prepare attempt + 1;
 * - poll: wait until `pollAt`, then poll the deferred response `handle`;
 * - tools: wait for the round's tool tasks; `pending` lists call IDs of a sequential round not yet started.
 */
export type GenerationCheckpoint =
  | { phase: "prepare"; attempt: number; compacted?: number; overflow?: string }
  | { phase: "request"; attempt: number; compacted?: number; model: ModelRef; thinkingLevel: ThinkingLevel; streamOptions: StreamOptions; cutoff: number }
  | { phase: "retry"; attempt: number; compacted?: number; until: number }
  | { phase: "poll"; attempt: number; compacted?: number; model: ModelRef; cutoff: number; handle: DeferredHandle; pollAt: number }
  | { phase: "tools"; assistant: number; tools: number[]; pending: string[] };

/** The compaction task's input: why it runs and the user's focus for a manual one. */
export type CompactionInput = { reason: CompactionReason; instructions?: string };

/**
 * What every task-kind entry function receives: the task as reserved, whether this invocation runs a phase ("run") or
 * the abort handler ("abort"), the phase's agent and settings, the conversation's provider session ID and the clock.
 */
export type PhaseFacts = {
  task: { id: number; kind: string; conversationId: number; input: JsonValue; checkpoint: JsonValue };
  mode: "run" | "abort";
  agent: Agent;
  settings: Settings;
  sessionId: string;
  now: number;
  /** Set when this phase runs again: why the earlier attempt failed (an error, or a return without a commit). */
  previousAttempt?: string;
};

// ---------------------------------------------------------------------------------------------------------------
// Write operations (`durable.commit`)

/**
 * One write of a commit. `durable.commit(ops, expect)` applies a list of them atomically, in order. An operation with
 * `as: "name"` names what it creates (an entry or a task ID); a later operation of the same commit refers to it as
 * "$name" in an ID field and anywhere inside next's state (its `on` and its checkpoint), and `commit` returns the IDs
 * by name.
 *
 * - appendAssistant: append a "pi.assistant" entry and count its usage. Every assistant message goes through it.
 * - appendToolResult: append the "pi.tool-result" entry of `call`: `result`, or a harness error {code, message}. The
 *   host renders diagnostics into the content as the model will see them, and counts the tool's usage.
 * - appendUser: append a "pi.user" entry with `content`, stamped now.
 * - appendSystem: append a "pi.system" entry with `message` (and `edits`).
 * - appendEntry: append any entry draft (a reset, a summary).
 * - convertPartial: append the committed partial answer of `pi.live.generation`, if any, as an aborted assistant entry.
 * - createTask: create a "pi.tool" (input {assistant, callId}) or "pi.generation" (input {}) task, owned by this task
 *   ("self") or by the conversation.
 * - createCompaction: create a compaction task and list its status. `owner: "self"`: this generation waits for it
 *   (blocking); absent: conversation-owned, in the background unless the reason is "manual". `ifNone`: only when
 *   no compaction is listed.
 * - liveGeneration: set `pi.live.generation` (null deletes it); clearTools: delete `pi.live.tools`;
 *   setTools: set `pi.live.tools` to `slots` (taskId may be "$name").
 * - slot: change the slot of `callId`: `taskId`, `status`, `entry`; `clearProgress` removes its running output.
 * - compactionStatus: change this compaction task's status (`attempt`, `retry`, null deletes retry) or `remove` it.
 * - startRun: start a run for placed `inputs`: a new conversation-owned generation takes `pi.live.run`.
 * - endRun: if this task owns the run, settle each of its inputs with `settlement` and remove `run`; always remove
 *   `generation` and `tools`.
 * - handOver: if this task owns the run, give it to task `to` (inputs move with it).
 * - runInputs: append placed input IDs to the run this task owns.
 * - place: place a boundary selection (see BoundarySelection); returns the placed user IDs as `placed`.
 * - usage: add `usage` to the conversation's ledger under `models["provider/model"]` or `tools[name]`.
 * - addTools: add tool names to the conversation's stored tool filter (a tool round's addTools control).
 * - next: set this task's next state. A list without `next` leaves the state as it is.
 */
export type Op =
  | { op: "appendAssistant"; message: AssistantMessage; as?: string }
  | { op: "appendToolResult"; call: { id: string; name: string }; result?: ToolExecutionResult; error?: { code: string; message: string }; durationMs?: number; as?: string }
  | { op: "appendUser"; content: UserContent; as?: string }
  | { op: "appendSystem"; message: SystemMessage; edits?: ContextEdit[]; as?: string }
  | { op: "appendEntry"; entry: EntryDraft; as?: string }
  | { op: "convertPartial" }
  | { op: "createTask"; kind: "pi.tool" | "pi.generation"; input: JsonObject; owner: "self" | "conversation"; as?: string }
  | { op: "createCompaction"; reason: CompactionReason; owner?: "self"; instructions?: string; ifNone?: boolean; as?: string }
  | { op: "liveGeneration"; value: { attempt: number; retry?: { at: number; error: string }; deferred?: { pollAt: number } } | null }
  | { op: "clearTools" }
  | { op: "setTools"; slots: { callId: string; name: string; taskId?: number | string; status: "pending" | "running" | "done"; entry?: number | string }[] }
  | { op: "slot"; callId: string; taskId?: number | string; status?: "pending" | "running" | "done"; entry?: number | string; clearProgress?: boolean }
  | { op: "compactionStatus"; attempt?: number; retry?: { at: number; error: string } | null; remove?: boolean }
  | { op: "startRun"; inputs: number[] }
  | { op: "endRun"; settlement: Settlement }
  | { op: "handOver"; to: number | string }
  | { op: "runInputs"; append: number[] }
  | { op: "place"; selection: BoundarySelection }
  | { op: "usage"; bucket: "models" | "tools"; key: string; usage: Usage }
  | { op: "addTools"; names: string[] }
  | { op: "next"; state: NextState };

/**
 * Guards a commit checks before writing: the state must still be what the caller read. A failed guard writes nothing
 * and rejects with "state changed: …"; read again and decide again.
 * - run: the task that owns `pi.live.run`, or null when there is none;
 * - inbox: the IDs of the inbox items read, in order;
 * - tail: the newest entry ID read.
 */
export type Expect = { run?: number | null; inbox?: number[]; tail?: number };

/**
 * What a commit created, by `as` name, and the user IDs a `place` operation placed. committed is set when the list had
 * a next operation: the task's state it committed, for example "running in phase summarize" or "terminal (completed)".
 */
export type CommitResult = { ids: Record<string, number>; placed: number[]; committed?: string };

/**
 * A function's answer when it committed the task's next state itself: committed says which (a CommitResult's
 * committed). Its caller commits nothing more for this phase and returns that text.
 */
export type Committed = { committed: string };

// ---------------------------------------------------------------------------------------------------------------
// Compaction (appended by the compaction port)

/**
 * The compaction task's checkpoints as pi-durable stores them: the summary request's fields sit beside `phase`
 * ("retry" adds `until`). summarize: ask the pinned model for the summary of the entries before `firstKept`, reading
 * the context as of entry `tail`; retry: wait until `until`, then summarize again with attempt + 1.
 */
export type CompactionPhase =
  | { phase: "select" }
  | { phase: "summarize"; attempt: number; model: ModelRef; thinkingLevel: ThinkingLevel; streamOptions: StreamOptions; maxTokens: number; tail: number; firstKept: number }
  | { phase: "retry"; attempt: number; model: ModelRef; thinkingLevel: ThinkingLevel; streamOptions: StreamOptions; maxTokens: number; tail: number; firstKept: number; until: number };

/**
 * A summary for the compaction entry to place: its text, the first entry it keeps, and the usage of the summarization
 * response that wrote it under `usageKey` ("provider/model"), when a model wrote it.
 */
export type SummaryToPlace = { summary: string; firstKept: number; usageKey?: string; usage?: Usage };

// ---------------------------------------------------------------------------------------------------------------
// Admission (`admit`, spec §6, appendix 2 R1)

/**
 * A conversation's admission facts, read at one committed point: `run`, the task that owns `pi.live.run` (null: the
 * conversation is idle); `inbox`, the queued items in ID order; `activeStart`, the newest head marker's `head` (null when
 * there is none); the queue modes now in effect.
 */
export type AdmissionState = { run: number | null; inbox: InboxItem[]; activeStart: number | null; steeringMode: QueueMode; followUpMode: QueueMode };

/**
 * One write of admission, applied by `admission.commit` in one commit, in order:
 * - queue: create a queued submission for `draft` and append its inbox item at the end ("write" for a write; for an
 *   input, "steer" when whenBusy is "steer", otherwise "followUp");
 * - boundary: place `selection` (see BoundarySelection) and, when it placed user items, start a run for them. ID 0 in
 *   the selection stands for the submission the queue operation of this commit created;
 * - write: append `draft.entry` as drafted and create a "done" write submission with that entry;
 * - stale: create an "unanswered" write submission with reason "stale"; no entry is appended;
 * - input: append a "pi.user" entry with `draft.content`, create a "placed" input submission with it, and start a run
 *   for it.
 */
export type AdmissionOp =
  | { op: "queue"; draft: SubmissionDraft }
  | { op: "boundary"; selection: BoundarySelection }
  | { op: "write"; draft: SubmissionDraft }
  | { op: "stale"; draft: SubmissionDraft }
  | { op: "input"; draft: SubmissionDraft };

/**
 * Guards of an admission commit: `run` and `inbox` as read (see AdmissionState), and `requestAbsent`, a request ID no
 * submission of the conversation may carry yet.
 */
export type AdmissionExpect = { run: number | null; inbox: number[]; requestAbsent?: string };

/** What admission concluded: the submission's ID, or the conversation is busy and rejects the input, or a request conflict. */
export type AdmitResult = { id?: number; busy?: boolean; conflict?: string };

// The companion (extensions/companion, COMPANION.md §§1, 4).

/** What the companion knows about one workspace file, valid while the file's hash is `hash`. */
export type FileKnowledge = {
  path: string;
  hash: string;
  /** What the file is for, in one or two sentences. */
  purpose: string;
  /** Its main definitions and exports. */
  symbols: string[];
  /** What someone changing or using the file must know: invariants, pitfalls, how it is tested. */
  notes: string[];
};

/** A file's summary as summarize returns it; the host adds the path and hash. */
export type FileSummary = { purpose: string; symbols: string[]; notes: string[] };

/** What the companion offers the agent for its next request. Every item is short and stands on its own. */
export type Briefing = {
  /** What the agent is working toward right now, in one sentence. */
  focus: string;
  /** Relevant facts the agent may not have in view, each naming its source (a file path or a command). */
  facts: string[];
  /** Problems in the agent's work: repeated failing steps, claims without evidence, edits whose dependents were not checked. */
  warnings: string[];
  /** Concrete next steps that would help: files to read, commands to run, checks to make. */
  suggestions: string[];
};

/** What one companion run sees. */
export type Observation = {
  /** The user's request that started the current work. */
  goal: string;
  /** The latest part of the conversation, compactly rendered: messages, tool calls and results (cut). */
  recent: string;
  /** Workspace files the recent tool calls read, edited or wrote, relative to the workspace. */
  touched: string[];
  /** The briefing the agent was shown last, or null. */
  previous: Briefing | null;
  /** Everything the companion already knows about workspace files (possibly stale for files changed since). */
  known: FileKnowledge[];
};

/** Which parts of a long tool output the agent sees, and what the rest holds. Lines are numbered from 1; ranges are inclusive. */
export type OutputShape = {
  keep: { from: number; to: number }[];
  /** What the output as a whole says, including the parts left out, in one or two sentences. */
  gist: string;
};
