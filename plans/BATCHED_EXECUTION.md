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
   fallbacks. Done: both servers have `POST /v1/natlang/score` (also `/v1/neuralese/decide_many`), and the Neuralese
   driver sends one batched request (section 9).
3. Server slot sizing in `local-server.ts` and `doctor`.
4. Prompt layout audit and fixture test, with prefix caching on.
5. Trace fields and the `natlang traces` occupancy view.
6. The `natlang check` loop diagnostic.
7. Gates 2–3 on DGX through the ledger, in a window that does not starve training.

## 6. Implementation status (TypeScript host)

Done: steps 1, 3, 4, 5, 6 and the TypeScript half of step 2. Tests: `test/scheduler.test.mjs`,
`test/prompt-layout.test.mjs`, `test/sequential-loops.test.mjs`.

**Scheduler (`model/scheduler.ts`).**
- `createScheduler({ mode, maxConcurrent, coalesceMs, priority, adjacency })`. Capabilities are
  `{ batching: { mode, maxConcurrent } }`.
- The call identity is `ModelTurnRequest.invocation_id`, which `chatCompletionModelTurn` hands to the transport as a
  third argument (`ChatRequestMeta`; plain transports ignore it). The scheduler counts requests per invocation: a
  second request of an invocation (or a malformed-call retry) is a running-call turn. No change to `native/runtime.ts`
  or `runtime/hooks.ts` was needed.
- Priority only reorders requests that are queued together: a slot that frees is refilled at once from the queue, so
  a running call's next turn that is not yet submitted does not hold a slot back for itself.
- Defaults deviate from 3.1: the coalescing window is 0 (next timer tick) for server-continuous and serial
  backends, 2 ms for explicit-batch ones. A continuous server batches by itself, so the window would only add
  latency. Release is immediate when a slot frees.
- Adjacency prefers the earliest-seen prompt prefix within a priority class; it can delay a late prefix while an
  earlier one keeps arriving. There is no aging yet.
- `requestLimit` is unchanged and still what a bare numeric `concurrency` gives. A scheduler with `priority: false`,
  `adjacency: false`, `coalesceMs: 0` admits like it (test). `openAICompatibleModelTurn({ concurrency })` accepts a
  number, a `RequestLimit` or a `Scheduler`; the managed session now always builds a scheduler, so its calls carry
  batch records. `model/config.ts` gains `batching` and `local.memoryBudgetMiB/kvBytesPerToken`.

**Scoring (`model/scoring.ts`, `native/decision.ts`).**
- `DecisionScorer.scoreMany?(items, signal)` returns one settled result per item; `scoreMany(scorer, items)` in
  `native/decision.ts` falls back to concurrent single scoring through the scheduler.
- `coalescingScorer` wraps the scorer of `openAICompatibleModelTurn`, so decision readouts that arrive in the same
  tick (from `Promise.all`, list refinements, classification loops) are scored by one `scoreMany`. `native/agent.ts`
  is unchanged for this.
- **Explicit-batch contract** (optional, capability-gated: `batching.mode: 'explicit-batch'` and
  `batching.scoreEndpoint`, `true` meaning `{endpoint}/v1/natlang/score`):

      POST /v1/natlang/score
      { "model": "...", "items": [ { "messages": [...], "continuations": ["\"a\"", "\"b\""], "adapters": [...]? } ] }
      200 { "results": [ { "log_probs": [..K numbers..], "tokens": [..K ints..]? } | { "error": "..." } ] }

  `continuations[k]` is option k as the closed final assistant message after `messages`. The server prefills each
  distinct `messages` prefix once and scores the continuations from its cache. `log_probs` and `tokens` have the
  meaning of `DecisionScores` (over the tokens where the options differ, end of message included), so results must
  agree with the one-request-per-option path within tolerance (gate 3). One request occupies one scheduler slot.
  404, 405 and 501 mean "no such endpoint": the client falls back to concurrent scoring and does not ask again. An
  item that fails returns `{ "error" }` and fails alone. The Neuralese server
  driver (`neuraleseServerModelTurn`) wires `scoreMany` to it, see section 9.

**Server slots (`model/server-slots.ts`, `local-server.ts`).** `slots = clamp(floor((budget - modelBytes) /
(contextTokens * kvBytesPerToken)), 1, 8)`, default KV budget = min(half of available memory, 4 GiB), shown by `natlang doctor` and the server start line (an explicit `local.memoryBudgetMiB`, which covers weights and KV, is not capped),
`kvBytesPerToken` 64 KiB by default; `local.parallel` wins. `-c` is the per-slot context times the slots. Requests to
the managed server carry `cache_prompt: true`. `natlang doctor` reports `serverSlots`, `slotPlan` and `batching`.
Note the default `--parallel` changes from 1 to the memory-sized value; on DGX set `local.memoryBudgetMiB` from the
ledger grant (the 64 KiB per token figure is a guess to be measured for the student model).

**Prompt layout (step 4).** Today's order is system, instructions with the signature and types (user message), the
pre-filled scope eval (callable declarations, host services, then the arguments, then captured variables), then the
folder listing for directory reducers, then the transcript. This matches 3.4 except that for directory reducers the
folder listing is a separate tool turn after the arguments, so the argument values precede the listing, and that
captured variables and the call's own earlier variables come after the arguments (all variable, so harmless for
prefix reuse). Reordering the listing before the arguments needs a prompt change and a live measurement; it was not
done. The fixture test checks the order and that two calls of one function share system, user and scope-eval
messages and differ first at the arguments.

**Observability (step 5).** `ModelTurn.scheduling` carries `batch_id`, `batch_size`, `in_flight`, `queue_wait_ms`,
`priority`; the `model_request` end event records them as `batch_id`, `batch_size`, `in_flight`, `queue_wait_ms`,
`schedule_priority`. Coalesced decision readouts record `batch_id`/`batch_size` on their `decision_readout` event. `natlang traces occupancy` summarises them per call and
in total. Prefix tokens reused are already in the turn stats (`cachedTokens`) but not in the event.

**Loop diagnostic (step 6).** `compiler/sequential-loops.ts`, called from `compiler/project.ts`. The diagnostic code
is `nl-sequential-loop` (severity warning). `natlang check` now prints warnings when the check passes.

## 7. Pending integration

- `compiler/inline.ts`: add `'nl-sequential-loop'` to the `NatlangDiagnostic['code']` union. Until then
  `sequential-loops.ts` casts the code. Detecting calls of inline `nl` functions bound to a local `const f = nl...` and
  using the type checker (rather than the `.nl` import and `nl` tag syntax) belongs in `compiler/eval-check.ts` or
  `inline.ts`.
- `native/runtime.ts` / `runtime/hooks.ts`: pass `turn` and the function name in the model request, so the scheduler
  can group by function instead of by system-message hash, and (decision-readout batch records are done).
- `improvement/services.ts`, `applications/pi/*`: construct their model drivers with the session's scheduler (they
  currently build their own `requestLimit`/drivers if they set `concurrency`).
- `skills/natlang-authoring/references/language.md`: one line telling authors that `Promise.all(items.map(f))`
  batches and a `for…of` that awaits does not.
- llama.cpp fork and `serve/grad.py`: the scoring endpoint above is done; `--kv-unified` where supported is not.

## 8. Gates 2 and 3 on DGX (through the ledger)

Gate 2 (triage throughput), baseline then scheduled. `tickets-1000.json` is a JSON array of 1,000 ticket strings:

    NATLANG_PROFILE=bench-p1 natlang run examples/triage -- tickets-1000.json
    NATLANG_PROFILE=bench-sized natlang run examples/triage -- tickets-1000.json
    natlang traces occupancy --limit 1200

with profiles in `~/.config/natlang/config.json`:

    "bench-p1":    { "local": { "parallel": 1 }, "batching": { "maxConcurrent": 1, "priority": false } }
    "bench-sized": { "local": { "memoryBudgetMiB": <ledger grant> } }

Time each run (`/usr/bin/time -v`); p50/p95 call latency are the `wall_ms` of `natlang traces list --definition classify
--json`; occupancy is the output of `natlang traces occupancy`. Gate 3 needs the server endpoint (pending) and compares
`batching: { mode: 'explicit-batch', scoreEndpoint: true }` with the same profile without `scoreEndpoint`, over
1,000 `readout: decision` classifications issued with `Promise.all`; compare the `decision_readout` events'
`log_probs` (events: `natlang traces show CALL --events`).

## 9. Results (2026-10-10, DGX, through the ledger)

Files: `runs/batched-execution-gates-20261010/` (the numbers are in `p1-classify.json`,
`sized-classify.json`, `p1-urgent.json`, `gate3-score.json`; launchers and profiles beside them). Scripts:
`scripts/bench_batched_gen.py`, `scripts/bench_batched_phase.py`, `scripts/bench_batched_score.mjs`.

**Wiring.** `neuraleseServerModelTurn(...).decide` is now a `coalescingScorer` whose `scoreMany` uploads the blocks
and resolves the adapters of each item (an item that fails there fails alone), sorts items so equal prompts are adjacent,
and posts one `/v1/natlang/score`. A 404, 405 or 501 falls back to per-item `/v1/neuralese/decide` and is not asked
again. Test: `test/neuralese-score-many.test.mjs` (fake server: one POST, adjacency, item error with its caller,
fallback). The OpenAI-compatible driver keeps its config gate (`batching.scoreEndpoint`); the llama.cpp `llama-server`
has no `prompt_logprobs` and no score endpoint, so decision readouts there are served only by the Neuralese servers.

**Setup.** One model at a time, each admitted by the ledger (budget 12 GB): gate 2 on the fork's `llama-server`
(`build-cuda`, commit 8bd8c95a0) with LFM2.5-2.6B Q6_K, `-ngl 99`, the GPU shared with training and a Mellum conversion
(14 to 25 tokens/s decode); gate 3 on the Python reference server with LFM2.5-350M on CPU (12 threads, also shared).
Reduced scale, because at these speeds 1,000 triage tickets would take days: gate 2 used 60 tickets.

**Gate 2, classify fan-out of `examples/triage`, 60 tickets** (`natlang run examples/triage --no-adaptation`; profiles
in `profiles.json`; server `--parallel 1 -c 8192` against `--parallel 4 -c 32768` with `batching.maxConcurrent` 4;
`cache_prompt: true` in both):

| | calls | wall (s) | calls/min | latency p50 / p95 (s) | model requests | mean in flight |
|---|---|---|---|---|---|---|
| bench-p1 | 60 | 1326 | 2.71 | 671 / 1261 | 158 | 1.00 |
| bench-sized | 60 | 589 | 6.11 | 268 / 530 | 132 | 3.91 |

Verdict: pass for the mechanism. Throughput is 2.25 times higher, latency 2.5 times lower, and the scheduler kept 3.9
of 4 slots busy. It is below the 4 times the slot count suggests because the decode is shared with training on the same
GPU (4 sequences decode in one step but each step is slower). Latencies include queue wait (mean 242 s and 111 s)
because all 60 calls start together. Occupancy comes from `scheduling` on the `model_request` events; `mean_batch_size`
1.09 is the scheduler's release batch, not the server's. The second stage (`is_urgent`, a boolean) was not
measured end to end: with the 2.6B model the `bench-p1` run needed 204 model requests for 12 calls (17 per call, 7631 s)
and was stopped by hand after 2 h 39 min, the sized run was stopped at the first `is_urgent` trace. The full-run
wall time of the plan's gate is therefore not reported. That `is_urgent` loop (a small model that keeps failing a
return-value contract) is a model-facing problem to look at separately, not a scheduling one.

**Gate 3, 1,000 decision readouts, 3 labels each** (`node scripts/bench_batched_score.mjs http://127.0.0.1:18743 1000`,
distinct prompts, shared rubric):

| path | requests | total (s) | decisions/s |
|---|---|---|---|
| A: N x K single-option `/v1/neuralese/decide` | 3000 | 2121 | 0.47 |
| B: N per-item `/v1/neuralese/decide` (K options, one prompt pass) | 1000 | 1173 | 0.85 |
| C: driver `decide` via `Promise.all` to `/v1/natlang/score` | 16 | 1302 | 0.77 |

Scores agree exactly (max |diff| 0 between A, B and C, argmax 1000/1000). Verdict: pass on agreement, and 1.63 times
the throughput of N x K requests (the prefix is prefilled once per item, not K times). C is 0.9 times B: with distinct
prompts per item the server does the same work, so batching only removes round trips, and on a loaded CPU the
difference is noise; the gain of one request per 64 items shows when requests carry a network or scheduler cost, and
when items share prompts. Not measured: the shared-prompt case, the fork's C++ `decide_many` (its working tree has
uncommitted changes owned by another session; a GGUF export of trained heads is needed to serve it), and GPU.

**Cleanup.** The three servers I started (`natlang-bench-llama-p1`, `natlang-bench-llama-s4`, `natlang-bench-ref`) are
stopped.
