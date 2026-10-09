# The companion: a living harness around the agent

Proposal, 2026-10-09, from the owner's direction: the harness is natlang's killer app. It is a domain that needs
semantic understanding everywhere, changes constantly, and rewards adaptation. Once the port works (PORT.md), the
harness should go well beyond the state of the art. It should be a second, always-present mind that accompanies the
main agent, works in the background, and keeps asking how it can help the agent do its job better.

This plan covers the harness side. The training side (teacher trajectories, a small Neuralese student, harness
support learned through Neuralese) is plans/neuralese/HARNESS_BENCH.md. The two share one design: whatever the
companion offers a live agent is exactly what the bench injects into rewritten trajectories.

## 1. What the companion does

The main agent works as today: pi's generation and tool tasks, with its own model. Beside it, the **companion**
watches the conversation and the workspace and does five kinds of work:

1. **Knows the repository.** It builds and keeps current a knowledge base of the workspace: what each file and
   directory is for, the symbols and how they connect, how to build and test, what is broken, what changed and why.
   It also keeps the session's own state: the goal, the hypotheses tried, the dead ends, what the agent believes.
2. **Researches ahead.** From the goal and the latest turn, it predicts what the agent will need next. It reads those
   files, greps, looks up documentation (local docs, installed packages' types, man pages, and the web where allowed),
   runs read-only commands and tests in a scratch copy, and has views of the results ready before the agent asks.
3. **Saves context.** It compresses large tool outputs into a summary plus a handle the agent can expand. It notices
   repeated reads and points to what is already known. It writes compaction summaries from its knowledge base instead
   of from the transcript alone, and it keeps exact text recallable after a cut.
4. **Offers, never commands.** Its results reach the agent through pi's own extension points (section 3): a briefing
   section in the system prompt, a `recall`/`ask` tool, shaped tool results, and, only for urgent findings, a steering
   message. The agent decides what to use.
5. **Watches the work.** It notices loops (the same failing command three times), regressions (a test that passed
   now fails), claims without evidence ("fixed", with no test run), and edits that miss related code. It says so in the
   briefing.

When the main model speaks Neuralese, the companion delivers in Neuralese: its briefing, its knowledge-base entries
and its compaction summaries become `Neuralese<T>` blocks spliced into the main agent's context through the read
port. The knowledge does not change, but it takes a fraction of the context (section 6).

## 2. Why natlang

Almost every companion decision is semantic: what matters for this goal, what the agent will need next, whether two
facts conflict, what to say and what to leave out. Each of these is a natural-language function with typed inputs and
outputs. The mechanical parts are crisp services: file hashing, the symbol index, the store, process sandboxes,
budgets.

Three runtime features make this practical, and the companion is their best showcase:

- **Durable background tasks.** Companion work runs as pi-durable tasks owned by the conversation (kind
  `pi.companion`). It is resumable, abortable and scheduled by the same scheduler as the agent's own tasks, so it
  survives restarts and never races the transcript.
- **The tracing JIT** (plans/TRACE_SPECIALIZATION.md). Every companion call is recorded in the machine's call store.
  Its hot, mechanical patterns (re-index after an edit to an unchanged-interface file, compress a test log of a known
  shape) are specialized into crisp cases under guards, with the agent as the fallback. The semantic parts stay with
  the model, and the specializer declines them as `semantic`. So the companion gets cheaper the longer it runs on a
  repository, without anyone writing the fast paths.
- **Pluggable hot paths** (owner decision). Policy is natural language. Where pi already has crisp code (pi's
  `estimateContext`), a setting selects crisp or natural language, as for the port's planning.

## 3. How it reaches the agent: pi's extension points, unchanged

The companion is a pi extension (`extensions/companion/`). It uses only mechanisms pi-durable already has, so a
conversation without the extension is exactly today's harness.

| pi mechanism | Companion use |
| --- | --- |
| A section (`addSection`, re-rendered at each prepare; planSystem appends only the delta) | `<companion>`: the current briefing. It is short and replaced each turn. An unchanged briefing costs nothing, and a changed one costs a section patch. |
| A tool | `recall(handle)` expands a compressed output or a knowledge-base entry; `ask(question)` queries the knowledge base synchronously (a natlang call with a small budget). |
| `afterTool` hook | Stores each long result (over 2,000 characters) as a call of `view` (§6): the exact result is kept in the companion's store (its `recall` handle), and the result message holds its text form with a reference to the call. Past 6,000 characters the text form is shaped: the agent sees part of the output plus a `recall` handle. Which part is a pluggable hot path (`shaping`, `--shaping`, modes `crisp`, `nl` and `shadow` through `pluggable()`). Crisp keeps the head and tail. Natural language (`shape.nl`) chooses the lines that matter for the next step and writes a gist of the rest; its choice is memoized for the call (`HookApi.memo`), and the host clamps it to the budget. If the natural-language call fails, the crisp shape stands in. With a Neuralese reader the hook also starts the view's block at once (latency). |
| `afterResponse` hook | With a Neuralese reader: records the intent of each tool call (the call and the turn's reasoning and text, as the harness bench's `intent()`), the view's instructions. |
| `beforeRequest` hook | Last-moment additions: a finished background result the briefing has not shown yet. |
| Owned tasks (`pi.companion`) | Background work: indexing, research, tests in a scratch copy, critique. |
| Steering submission (inbox, `steer` mode) | Urgent findings only, such as a loop or a destructive command about to run. A policy decides, and the default is rare. |
| Compaction | The summary is written from the knowledge base and the session state, and keeps handles to exact text. |
| Events (`watchEvents`) | The companion's triggers: turn end, tool end, new input, compaction. |

## 4. The knowledge base

One store per workspace (SQLite beside the session, `.pi/companion/kb.sqlite`), content-addressed by file hashes,
with provenance on every fact.

- **Files and directories:** path, hash, language, size, and a summary (purpose, main symbols, dependencies). The
  summary is written by a natural-language function and invalidated by the file's hash. Directory summaries are rolled
  up from their children, giving a hierarchical, wiki-like view.
- **Symbols:** a crisp index (TypeScript compiler API for TS/JS, tree-sitter or ctags for other languages): definitions,
  references, imports. It is ranked by the session's focus, Aider repo-map style (PageRank over the reference graph,
  personalized to the files in play).
- **Procedures:** how to build, test, lint and run, learned from manifests and from commands that worked, with their
  last results and durations.
- **Facts:** short typed statements with provenance (the entries, files and hashes they came from) and a staleness
  rule: a fact that cites a file is stale when the file's hash changes.
- **Session state:** the goal as understood, plan, hypotheses with evidence, attempts and their results, open
  questions. This is the companion's episodic memory, and the part a transcript cut loses first.
- **Documents:** fetched documentation and research notes, chunked, with where they came from.

Retrieval is a natural-language function over crisp candidates. Crisp search (symbols, paths, full text, and later
embeddings) proposes, and `relevant(goal, step, candidates)` chooses and explains. The same function picks what goes
into the briefing.

Inspirations, to borrow from deliberately: Aider's repo map; Cursor and Sourcegraph code indexes; DeepWiki- and
Devin-wiki-style generated repository documentation; Claude Code's CLAUDE.md memories; MemGPT/Letta tiered memory;
Reflexion-style lessons; SWE-agent's agent-computer interface (outputs shaped for the model); OpenHands' condenser;
speculative execution from CPUs (prefetch what the agent will probably ask for).

## 5. Scheduling and budgets

The companion must never slow the agent down. Its work runs while the agent's model is generating or a tool is
running, and it yields when they need the executor.

- The companion has its own executor setting. It may be the same model as the agent (the default today), a smaller
  local model, or the Neuralese student.
- A budget per turn (tokens and wall time) and per session. A natural-language scheduling policy chooses which
  pending work to run (index, research, critique), with a crisp default.
- Every offer is tracked: was it shown, did the agent's next actions use it (a later read of the same file, a cited
  symbol, a recall of the handle)? Usefulness statistics feed the policy and are training signal (HARNESS_BENCH.md).
- Results arrive asynchronously. A research result that is ready before the next prepare goes into that briefing. A
  later one waits for the following turn, or is dropped when it is no longer relevant.

## 6. Neuralese delivery

When the agent's model is Neuralese-capable (its driver advertises `neuralese: true` and a dialect):

- The briefing section renders a `Neuralese<Briefing>` written by the companion's writer from the same typed
  briefing. The text form stays recorded beside it, for logs and for agents without the port.
- Knowledge-base entries keep Neuralese encodings keyed by content hash and dialect. They are regenerated when the
  dialect changes, as with any stored value.
- Compaction summaries and large tool outputs can be Neuralese views: the Neuralese instance of the runtime's builtin
  `view(value, instructions?)` (DECISIONS.md 2026-10-09, "one summarizer family"), with the call's intent as its
  instructions; an explicit operator stage (TRAINING_RECIPE.md "The view operator stage"), qualified separately.
- **How `shape.nl` relates to `view`.** `view`'s crisp instance returns a string written from the value; `shape.nl`
  returns an `OutputShape`: line ranges the host copies verbatim from the exact output, plus a one-sentence gist. That
  keeps every shown line exact and lets the host clamp the selection to the budget, which a free-text view cannot
  promise, so `shape.nl` is not rewritten as `view`'s crisp instance; it stays the crisp shaping policy, and pi never
  runs `view`'s crisp instance (the shape is the text form text readers read; a second text form would be read by
  no one).
- **Stored calls, forced per reader (built, `host/views.ts`).** A long output is not converted when it arrives, since
  its readers are not known then. It is stored as a call of `view` with two inputs: the whole output (the `recall`
  store, `pi.companion.output`) and the agent's intent when it made the call (`pi.view.intent`, recorded after the
  response by the `afterResponse` hook while the reader is Neuralese; the same text as the harness bench's
  `intent()`). The result message keeps the text form plus the reference (`{ type: "text", text, stored: { function:
  "view", call } }`); pi messages stay plain JSON and reader-independent. Each consumer forces the call at its own
  representation:
  - the agent's requests (`ai.turn`, host/ai.ts): a Neuralese reader is sent the block of `view(output, intent)` in its
    dialect, written by its own server (`POST /v1/neuralese/view`), followed by the note `// view of the output;
    recall("ID") returns all of it` (the bench's `recall_note`). A text reader is sent the text form. The block is
    started by the `afterTool` hook as soon as the result exists, and the request joins or finds it;
  - compaction, the companion's transcript and `recall` read the text form or the raw output and force nothing.

  Forcing is memoized durably per (call, dialect) in the conversation's `pi.view.forced` documents (block ID, length,
  dialect), so turns, retries and restarts reuse the block. The block is archived in the runtime's Neuralese store,
  pinned on the server under the conversation's owner (its provider session ID, also sent with each request), and a
  generation's request collects the owner's blocks to the ones it references when the context's head moved (the
  first request, after a compaction or a reset), unpinning views that left the context. A block the server lost is
  restored from the archive. A view that cannot be written fails the turn, never falling back to text:
  `neuralese-view-unavailable` (server unreachable, 429, 5xx) is retried by pi's retry policy,
  `neuralese-view-failed` (no intent recorded, wrong dialect, a 4xx) is not.
- `recall` returns exact text when the agent asks for it. For a Neuralese reader the block is the default; it never falls back to text when a block cannot be written.

This depends on the runtime qualification gates of the Neuralese programme. Until a channel is qualified for those
exact weights, the companion delivers text. The bench (HARNESS_BENCH.md) is where these encodings are learned.

## 7. Order of work

1. **Port solid.** Conformance suite green on current code, the pure natural-language variant measured, live eval
   tasks (README Status).
2. **Companion skeleton.** The `extensions/companion/` registration, the `pi.companion` task kind (natural-language
   phases), the knowledge-base store with the file index and summaries, the briefing section, `recall`, and
   `afterTool` output shaping. Measured on eval tasks against the plain harness: turns, tokens in context, wall time,
   success.
3. **Research and critique.** Prediction of next needs, prefetch, scratch-copy test runs, the loop and regression
   watcher, steering policy.
4. **Session memory and compaction** from the knowledge base, with recall handles across cuts.
5. **Neuralese delivery**, once a channel is qualified for the agent's weights.
6. **JIT.** Watch which companion functions the specializer compiles and which it declines. Declines are the
   functions where natural language is essential. Use the specializer's reports to restructure companion functions
   so their mechanical parts can be specialized.

Each part gets its unit decision (function, instruction, crisp helper) before implementation (owner: port
granularity).

## 8. Open questions

- Web research needs network access and a policy for what may leave the machine. The default is local sources only,
  with web lookups as an opt-in setting.
- Scratch-copy execution (tests, speculative commands) needs a sandbox: a git worktree plus the existing environment
  abstraction. Commands that could have effects outside the copy are never run speculatively.
- How many companions? One per conversation to start. Subagent conversations get their own, sharing the workspace's
  knowledge base.
