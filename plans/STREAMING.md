# Streaming: model output, Neuralese, and user input as live streams

Owner direction, 2026-10-09: streaming end to end. Model output streams to the user and to background processing while
it arrives; user input streams too (edit deltas while typing), so the harness can help before a message is sent.

## 1. Model output

1. **Contract.** A model turn gains an optional delta channel (`onDelta` on the turn request options, ts-host
   `contracts.ts`): text, reasoning, tool-call fragments, and a Neuralese part once a block is complete. Deltas come
   from the one place that already assembles streamed chunks (`model/chat-completion.ts`), so a streamed turn and a
   whole turn cannot diverge. The final turn stays authoritative (the reference server's final `x_natlang_message`).
2. **Drivers.** The Neuralese driver streams when the server declares it in `/v1/neuralese/info`; no try-and-fall-back.
3. **Servers.** The llama.cpp fork serves SSE in the reference server's chunk format (text deltas, a content delta per
   written block, tool calls at the end with LFM's call marker held back, a final `x_natlang_message`). The browser
   build gets it through the rebuild, and its in-process endpoint streams responses. A conformance case: a streamed
   reply, assembled, equals the non-streamed one, on both servers.
4. **pi.** The natlang provider maps deltas to pi-ai's events, so pi-durable streams into `pi.live` and the user sees
   output as with any provider. Only the final message is stored. A Neuralese part streams as the provider's
   `neuralese` event (pi-ai has no part type for it). The final turn decides the message: streamed parts it extends
   are finished, others are replaced. A `reset` replaces the abandoned attempt's parts in the partial (they end as
   they stood; the re-sent request's parts start again at index 0) within the same pi-durable attempt: a new attempt
   is the generation's retry, which would store the abandoned partial as an aborted entry.
5. **Blocks shown to people.** A block streams as one unit when complete and shows as a block reference. Its content is
   shown by forcing its stored call at `string` (DECISIONS 2026-10-09) only when the user asks.
6. **Natlang programs.** The runtime passes a function's model deltas into its live events, so a host can show a natlang
   function working. Done (2026-10-10): `onLive` on the runtime or a task receives `model_delta` events (`LiveEvent`,
   `runtime/runtime.ts`) tagged with the task, call, caller, function and turn; a request that fails or is sent again
   after streaming gives a `reset`; without a subscriber the driver gets no turn options
   (`test/live-events.test.mjs`).

## 2. Semantic processing while output streams

Background helpers subscribe to the live stream instead of waiting for the finished message:

- the companion reads the agent's reasoning as it streams and starts research (files, docs, earlier context) for what
  the agent is about to need;
- a tool call's arguments are known before the turn ends: the harness can prepare (resolve paths, warm caches, start
  safe reads) and the digest's intent is available as soon as the call is complete;
- checks that would only produce a correction later (a wrong path, a forbidden command) can be prepared early.

Helpers act on deltas as hints; anything they commit is tied to the final message, so a turn that changes course
discards speculative work. Helper policy is natural language behind `pluggable()`.

Done (2026-10-10) for the companion (applications/pi COMPANION.md §7, `extensions/companion/stream.ts`): pi-durable's
generation hands each provider event of a streamed attempt to `onStream` observers (vendored generation.ts
`streamResponse`, synchronous, before pi.live's throttle). The companion's hint policy (`hints`, `--hints`:
crisp paths and quoted names, `hints.nl`, or shadow) turns streamed reasoning and text into lookups (files learned and
summarized, symbol definitions found); a tool call whose arguments parse as a whole object is prepared at once (its
view intent captured, a read's file learned, an edit's or write's file read once). The terminal message settles it:
work whose source text or call (same ID, name, arguments) it keeps is committed (file knowledge; research notes the
next request's companion section shows), the rest is aborted; a reset (parts restarting at a content index already
seen), a failed or aborted attempt and a new attempt discard it at once. Bounded: 2 jobs at a time, 8 per turn, 3
summaries per turn; crisp mode costs a regular expression when nothing is named. Tests:
applications/pi/test/stream-helpers.test.mjs. Not yet: early checks that would produce a correction (a forbidden
command), and helpers besides the companion.

## 3. User input as a stream

The harness UI sends edit deltas of the draft while the user types (a draft document per conversation, updated live,
never part of the transcript until sent). Helpers may react: show relevant context, completions, warnings, or
clarifying offers next to the draft. Sending the message is the only event that enters the transcript; help shown
while typing is an offer the user takes or ignores.

Done (2026-10-10), applications/pi `host/drafts.ts` (`openDrafts`, exported by index.ts; `BrowserPi.drafts` in the
browser host): the draft is a conversation document (`pi.draft`, `{ text, version }`) changed by edit deltas
(`{ from, to, insert }` or `{ text }`), never an entry. Helpers run debounced (400 ms; a newer edit aborts a running
pass) and write offers (`pi.draft.offers`: context, warning or question with the text taking it adds), committed only
for the draft version they read. `take` adds an offer to the draft; `send` submits the draft as input and clears the
draft and its offers in one commit (restored if the submission fails). The companion's helper
(`extensions/companion/draft.ts`) is pluggable (`offers`: crisp checks of what the draft names, `offers.nl`, or
shadow). Not yet: a UI that sends the deltas (the harness UI is the consumer; the API is in place on both hosts).

## 4. Order

1. Contract and delta assembly (§1.1), Neuralese driver (§1.2), pi mapping (§1.4).
2. Fork SSE and conformance (§1.3), with the fork parity fixes.
3. Live events for natlang functions (§1.6).
4. Background helpers on the stream (§2), starting with the companion. Done for the companion (2026-10-10).
5. Draft deltas and help while typing (§3). Done: the drafts API and the companion's offers (2026-10-10).

Driver wrappers forward `ModelTurnOptions` (the driver's third argument): ts-host evaluation/usage.ts
(`UsageGateway.request(..., role, options)`), self-play evaluation, the CLI's model-session wrappers (main.ts,
adaptation.ts, improvement.ts). A model session's `turn` takes `SessionTurnOptions` (contracts.ts): the turn options
plus `onProgress`, the aggregate stream progress that was its third argument; a Pi provider backend maps pi-ai's
events to `onDelta`, a managed local server passes `onDelta` to its HTTP driver.
