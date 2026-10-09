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
   output as with any provider. Only the final message is stored.
5. **Blocks shown to people.** A block streams as one unit when complete and shows as a block reference. Its content is
   shown by forcing its stored call at `string` (DECISIONS 2026-10-09) only when the user asks.
6. **Natlang programs.** The runtime passes a function's model deltas into its live events, so a host can show a natlang
   function working.

## 2. Semantic processing while output streams

Background helpers subscribe to the live stream instead of waiting for the finished message:

- the companion reads the agent's reasoning as it streams and starts research (files, docs, earlier context) for what
  the agent is about to need;
- a tool call's arguments are known before the turn ends: the harness can prepare (resolve paths, warm caches, start
  safe reads) and the digest's intent is available as soon as the call is complete;
- checks that would only produce a correction later (a wrong path, a forbidden command) can be prepared early.

Helpers act on deltas as hints; anything they commit is tied to the final message, so a turn that changes course
discards speculative work. Helper policy is natural language behind `pluggable()`.

## 3. User input as a stream

The harness UI sends edit deltas of the draft while the user types (a draft document per conversation, updated live,
never part of the transcript until sent). Helpers may react: show relevant context, completions, warnings, or
clarifying offers next to the draft. Sending the message is the only event that enters the transcript; help shown
while typing is an offer the user takes or ignores.

## 4. Order

1. Contract and delta assembly (§1.1), Neuralese driver (§1.2), pi mapping (§1.4).
2. Fork SSE and conformance (§1.3), with the fork parity fixes.
3. Live events for natlang functions (§1.6).
4. Background helpers on the stream (§2), starting with the companion.
5. Draft deltas and help while typing (§3).
