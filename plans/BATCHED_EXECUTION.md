# Batched execution of pending model calls

Status: design. The owner decided on 2026-10-09 to batch all pending model calls. Companions: [architecture
plan](ARCHITECTURE_IMPROVEMENT.md) E3, [refinement types](REFINEMENT_TYPES.md) (whose checks are the most batchable
calls we have).

## 1. Where we are

- **Concurrency exists, batching mostly does not.**
  - `Promise.all` over `nl` calls issues concurrent HTTP requests, limited by `requestLimit`
    (`model/chat-completion.ts`).
  - Whether they batch depends on the server. The managed local llama.cpp server starts with `--parallel 1` unless
    configured (`model/local-server.ts:168`), so the local default serializes every call.
- **Every turn of every call is its own request.** An `nl` call is a multi-turn tool loop (`native/agent.ts`). Calls
  that run side by side drift out of step, and nothing aligns their turns.
- **Decision readouts are one request per option** (`promptLogprobDecider`). A list of N classifications is N×K
  prefills. Each one recomputes a shared prefix unless the server's prefix cache catches it.
- **Prompt layout decides prefix reuse.** Calls of the same function share the system prompt and instructions. Calls
  of the same predicate share the predicate. Whether those come before or after the per-call arguments decides
  whether the server can reuse the prefix KV.
- **The runtime never sees the batch.** It cannot report occupancy, so we cannot tell whether a program ran
  batched.

## 2. Goal

Every model call that is ready at the same moment goes to the model in as few forward batches as the server allows:

- the turns of concurrent `nl` calls
- the options of decision readouts
- refinement checks
- the members of a directory-reducer fan-out
- independent `iterateOn` branches

Programs keep their meaning: logical order, effects and traces do not change. Only the schedule changes.

## 3. Design

**3.1 One scheduler per model backend** (`model/scheduler.ts`). Every request for a backend goes through it. It
replaces `requestLimit` as the place where requests wait, and keeps `requestLimit`'s contract as a special case.

- Ready requests collect in a queue. The scheduler sends them as one batch when the batch is full (the backend's
  `capabilities.batching.maxConcurrent`) or when a short coalescing window has passed. The default is the time to
  the next microtask drain plus 2 ms, measured and configurable.
- The scheduler serves turns of calls that are already running before new calls. That finishes work, frees
  KV/slots, and keeps latency bounded for programs that wait on one result.
- Backends declare how they batch:
  - **server-continuous**: llama.cpp with `--parallel N`, vLLM, the neuralese server. The scheduler sends up to N
    requests at once and the server batches them.
  - **explicit-batch**: one request carrying many prompts, e.g. a scoring endpoint that takes K options × M items.
    The scheduler packs.
  - **serial**: remote APIs where batching buys nothing. The scheduler only limits concurrency.

**3.2 Server slots by default.**

- The managed local server sizes `--parallel` from free memory and the model's KV need per slot. `-np` with a
  shared KV pool (`--kv-unified` where the build supports it) is better than one slot.
- `natlang doctor` reports the slot count.
- On DGX, GPU memory is system memory (`dgx-unified-memory-oom`), so the sizing goes through the ledger's
  memory admission.

**3.3 Batched decision readouts.**

- `DecisionScorer` gains `scoreMany(items: {prompt, options}[])`.
- **Explicit-batch scoring.** Where the backend supports it, one request scores every option of every item. The
  options become continuations of a shared prefix. Our neuralese server and the llama.cpp fork get an endpoint that
  takes a prefix and K continuations and returns K log-probabilities. That is one prefill for the prefix and K short
  decodes.
- **Server-continuous scoring.** Elsewhere the requests go out together so the server batches them.
- `readout: decision` calls arriving together from `Promise.all`, list refinements or classification loops are packed
  by the scheduler automatically. Authors write nothing new.

**3.4 Prefix-friendly prompt layout.**

- Requests are laid out as stable to variable: system prompt, then the function's instructions and types, then the
  callable-folder listing, then the arguments, then the transcript.
- `cache_prompt: true` (llama.cpp) or the server's prefix caching is on by default.
- Requests for the same function inside one batch are sent adjacent, so slots with a matching prefix are reused.
- A test fixes the layout. Changing the order is a measured decision.

**3.5 Loops that look sequential.** `for (const t of tickets) await classify(t)` is sequential by meaning. We do not
silently parallelize it, because effects and order would change. Instead:

- `natlang check` reports a loop whose body is one `nl` call with no effect and no carried state, and suggests
  `Promise.all(tickets.map(classify))`. It is taught in a diagnostic, not in the prompt.
- The specializer's mined call groups (`calls/groups.ts`) give evidence where such loops dominate run time.

**3.6 Observability.** Every request's trace event records its batch: batch id, batch size, queue wait, and prefix
tokens reused where the server reports them. `natlang traces` shows occupancy per run. The adaptation and
specialization reports include it.

## 4. Gates

1. **Mock-server unit tests.** Coalescing, priority of running calls, cancellation, `requestLimit` equivalence,
   packing for explicit-batch scoring, and an unchanged trace order.
2. **Triage throughput.** `examples/triage` over 1,000 tickets on the local student, comparing `--parallel 1` (today)
   with sized slots plus the scheduler. Report throughput, p50/p95 call latency and occupancy. Run through the
   ledger.
3. **Decision readouts.** 1,000 classifications as N×K requests versus prefix + K continuations. The scores must
   agree within tolerance, and throughput is reported.
4. **No semantic change.** The conformance suite (`test:conformance`) and application tests pass unchanged under the
   scheduler, with results identical to unbatched runs at temperature 0.

## 5. Steps

1. `model/scheduler.ts` with the capability declaration, routed through all backends (with B3 in the architecture
   plan), plus mock tests.
2. `scoreMany` and the prefix+continuations scoring endpoint in the neuralese server and the llama.cpp fork, with
   fallbacks.
3. Server slot sizing in `local-server.ts` and `doctor`.
4. Prompt layout audit and fixture test, with prefix caching on.
5. Trace fields and the `natlang traces` occupancy view.
6. The `natlang check` loop diagnostic.
7. Gates 2–3 on DGX through the ledger, in a window that does not starve training.
