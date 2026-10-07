# Termination, asynchrony and application state

Plan of 2026-10-07. It closes the places where eval code can run without bound, makes asynchronous
JavaScript idioms behave sensibly inside natlang calls, and adds a small host-side layer for
applications that serve users. Evidence comes from probes run against `ts-host/dist` (built
2026-10-07 09:21) and from reading the runtime. The probe script is reproduced as tests in Phase 0.

## Owner decisions this plan follows

1. Close the eval leaks and make the proposed async and state tweaks.
2. Do not be conservative with resource limits. Build a limit when a resource problem shows up, not
   in advance. Termination rules in this plan are structural; none is a budget, cap or default timeout.
3. Be circumspect with semantic and DSL extensions. The models executing natlang are small and must
   not need bespoke machinery they have to be prompted or trained to use.
4. Stay close to the semantics of the JavaScript runtimes we target.
5. Keep approximate Turing incompleteness, and say plainly where it ends.

## Design rules

**Two audiences.** The *model-facing surface* is eval code, callable-folder code a model reads or
edits, system prompts, tool descriptions and error messages. It gets no new names. Changes there either
make a standard JavaScript idiom behave correctly or refuse a construct with one sentence naming the
standard alternative. Relaxing a restriction is preferred to adding one. The *host surface* is
application TypeScript written by developers and coding agents. Small library additions are acceptable
there; nothing in it is a language primitive.

**Threat model.** The rules target code a small model writes in good faith. Deliberate evasion through
reflection (for example fetching `Symbol.iterator` with `Object.getOwnPropertySymbols`) is out of scope;
the Node backend is not a sandbox. Services and imported packages are host authority and may run
without bound, as they can today.

**What is guaranteed.** Restricted code (eval, callable folders, `natlang.d/`) cannot express an
unbounded loop or unbounded recursion by itself. The deliberate exceptions are model-controlled: a
natural-language stopping predicate for `iterateOn` and the agent loop of a call, which has no turn or
time limit unless the caller sets one. Applications run forever, but only by reacting to outside events
(users, webhooks, the clock), each reaction being bounded. With periodic wake-ups and growing state, an
application can still simulate a Turing machine over unbounded wall-clock time. No application model
avoids that, and this plan does not try to.

## Findings

Confirmed by probes (a scripted model running the eval code through `runtime.run`):

| # | Eval code | Observed |
|---|---|---|
| 1 | `for (let i = 0; i < Infinity; i++)`; a getter as the bound; pushing to an alias of the bounding array | Accepted. The bound is re-read every iteration (a getter bound ran 1002 times), so a counted `for` acts as `while`. |
| 2 | Recursion through object-literal methods, class methods, or `holder.go = () => …then(() => holder.go())` | Not guarded. Awaited recursion went as deep as the probe chose; unawaited versions kept running after the call returned (about 130 service calls in the next 300 ms). |
| 3 | `setInterval(() => counter.tick(), 5)` | Kept running after the call returned and the task closed (58 service calls in 300 ms). Also available to callable folders, which run with host globals. |
| 4 | `new Set({ [Symbol.iterator]() { return { next: () => ({ done: false, value: 1 }) } } })` | Hung the process. Only `for … of` is guarded; spread, `Array.from`, `new Set`/`new Map`, destructuring and `Promise.all(iterable)` consume user iterables directly, and eval has no default timeout. |
| 5 | Case 2 with `await null` instead of a timer | Starved the event loop: neither eval's `timeout_ms` nor any other timer in the process could fire. Collector workers share a process, so one such eval would stall all of them. |
| 6 | Prelude helpers `chunk(xs, 0)`, `windows(xs, n, 0)`, `range(1, Infinity)` | Loop until the heap is exhausted (checked under a 96 MB heap). |

What held: function declarations and `const f = …` functions are guarded, so self-reference through
`flatMap(f)`, interface-typed self-application and `w(w)` were rejected at run time; `iterateOn` with only
a deadline was rejected.

Found by reading:

- **Eval and callable folders differ.** Callable-folder lowering fixes the loop bound at entry and checks
  the counter advances (`ts-host/src/compiler/lower.ts:141`), and guards every authored function
  (`lower.ts:167`). Eval lowering guards only declarations and functions assigned in a variable
  declaration (`ts-host/src/scope-compiler.ts:525`) and does not rewrite `for` bounds.
- **Eval timeouts do not return early.** After an eval, the session waits for every child call it
  started (`ts-host/src/native/runtime.ts:1605`), so `timeout_ms` cannot return early while natural-language
  children run, and their later effects are invisible to the model. An eval that leaves children running
  is failed only after waiting for them (`runtime.ts:1456`). That also makes `Promise.race` over
  natural-language calls fail the eval after waiting for the loser.
- **Browser eval** runs in the page realm (`ts-host/src/browser/environment.ts:82`) with every page global.
  The task frame is restored after compiled `await`s only, so callbacks from `.then` and timers run with
  no frame and the recursion guard does not apply to them.
- **Rebinding can re-enter.** The recursion identity includes the context ID (`ts-host/src/runtime/kernel.ts:294`),
  so a definition rebound to a freshly built context may run below itself. Code that can build contexts can
  make one per level. No way to build a `Context` from eval was found; today this is host-level only.
- **Guard cost.** The recursion guard costs about 0.23 µs per call: a one-million-element `map` with a
  guarded `x => x * 2` took 240 ms instead of 13 ms.
- **Trees.** The recursion ban rules out walking trees. `conformance/programs/19-recursion-tree.yaml`
  shows the workaround: a loop over at most 64 levels.

## Execution constraints

- The live semantic collector (`ts-host/dist/teacher/cli.js`, run `semantic-teacher-20261006-s73`) imports the
  canonical `ts-host/dist`. `npm test` and `npm run build` clean and rebuild that directory. Develop and test
  in a separate git worktree with its own build. Rebuild the canonical `dist` only after s73 finishes, or
  with the owner's agreement.
- Frozen runtime copies (Pop's collections, the skill-authoring queue's `runtime-v3-…`) are unaffected until
  their owners refresh them.
- Small commits to `origin/main`; brief Pop after each landed phase with commit hashes and exact semantics.
- Tests that load a model go through the memory ledger. None of the tests below loads one; they use
  scripted drivers.

## Phase 0: Regression tests for the findings

Add `ts-host/test/termination.test.mjs` with one case per probe (findings 1–6, plus the cases that held).
Run cases that could hang, such as microtask loops, in a child process with a short timeout, so a regression
fails the test instead of hanging the suite. Each case asserts the behaviour after the fix; before the fix,
the cases fail. Keep `guard-bench` as `ts-host/scripts/guard-bench.mjs`.

## Phase 1: Close the eval leaks (Node)

**1.1 Same lowering for eval as for callable folders.** Factor the `for` and guard rewrites out of
`compiler/lower.ts` and apply them to eval snippets in `scope-compiler.ts`.

- **Counted `for`.** The bound is evaluated once at entry. `numericProgress` checks that the counter
  advances and rejects a non-finite bound (`RangeError: the loop bound must be a finite number`).
- **Guards.** Every authored function gets the guard, wherever it appears: arrows, function expressions,
  object-literal and class methods. Constructors and accessors stay unguarded, as in callable folders; they
  are synchronous, and anything they schedule is a guarded function.
- **Exemption for call-free functions,** in both eval and callable folders. A function whose body contains
  no call, `new`, tagged template, `await` or `yield` gets no guard. Re-entry needs a call. Implicit calls
  (getters, `valueOf`, proxy traps) can only form synchronous cycles of call-free functions, and those end
  at the engine's stack limit. This removes the guard cost from typical `map`, `filter` and `sort` callbacks.
- **Optional cheaper chain.** If the benchmark still shows a material cost, replace the array-copied chain
  in `Frame` with a linked list.
- **Static check unchanged.** `findRecursion` keeps catching direct recursion early in eval. Phase 5
  revisits it.

**1.2 Program-defined iterators.** In restricted code, refuse `Symbol.iterator` and
`Symbol.asyncIterator` as property accesses, `Iterator.from(…)` and classes extending `Iterator`
(`compiler/policy.ts`). Message: "Defining iterators is not available here; build an array, or use iterateOn
for an open-ended sequence." Built-in iterators (`map.entries()`, `matchAll`) are unaffected. Generators
are already refused.

**1.3 Timers.**

- **No `setInterval`.** Remove it from the eval context (`ts-host/src/environment.ts:186`) and refuse the
  identifier in callable-folder code. Message: "setInterval is not available; repeat with
  iterateOn(step, initial).withLimit({ maxSteps }) and wait inside the step with
  `await new Promise(r => setTimeout(r, ms))`."
- **Other timers stay.** `setTimeout`, `setImmediate` and `queueMicrotask` remain. Their callbacks carry
  the scheduling frame through `AsyncLocalStorage`, so a self-rescheduling function hits the guard; tested
  in Phase 0.
- **Timers end with their call.** The Node eval environment records the timer handles it creates, and
  `close()` clears them. The kernel already closes the environment when a call ends (`kernel.ts:450`).
  Record a trace event when timers were cleared. Callable-folder modules use host timers; scoping those
  waits for an observed problem.

**1.4 Prelude helpers** (`ts-host/prelude.js`). `range` requires finite bounds; `chunk` and `windows`
require a size and step of at least 1. Bad arguments throw `RangeError` with the rule in one sentence.

**1.5 Event-loop starvation** is closed by 1.1, because every such loop needs re-entry. There is no gas
counter and no new default timeout.

**1.6 Recursion message.** `NatlangRecursionError` keeps its text and adds the alternative: "use a loop
over a work list, or iterateOn."

**Gate.** All Phase 0 cases pass, and the full ts-host suite and native conformance pass in the worktree.
Call-free callbacks in `guard-bench` run within 1.5× of the unguarded baseline; the overhead of guarded
callbacks is recorded in this file.

## Phase 2: Browser parity

**2.1 Verify first.** Add the Phase 0 cases to the Chromium smoke (`npm run test:browser`) to confirm the
guard bypass found by reading.

**2.2 Evaluate in a separate realm.** Run eval code in a dedicated same-origin iframe realm per runtime,
the way Studio already isolates generated modules, with globals allowlisted to match the Node context
(no `setInterval`, `requestAnimationFrame` or `MessageChannel`).

- In that realm, wrap `Promise.prototype.then/catch/finally`, `queueMicrotask` and `setTimeout` so that
  callbacks restore the frame current when they were scheduled. This is what `AsyncLocalStorage` does in
  Node.
- Live values cross realms by reference, as they already do between Node's vm context and the host.
- Callable-folder modules stay in the page realm with their compiled `await` restoration. Extending the
  wrapping to them waits for an observed problem.
- When the TC39 AsyncContext proposal ships in browsers, the wrapping is replaced by it.

**Gate.** Phase 0 cases pass in Chromium; browser smoke and playground smoke pass.

## Phase 3: Cancellation within a call

No new API. Model drivers already accept an `AbortSignal`, and the runtime already stops at its checkpoints
when the signal aborts.

**3.1 A signal per call.** Each invocation gets an `AbortController`. Its frame signal is
`AbortSignal.any([parent signal, own controller])`, and the model driver and `NativeRuntime` receive that
signal instead of the task's (`kernel.ts:421`). Each eval runs under a frame whose signal also includes a
controller for that eval, so child calls started by the eval derive from it. `iterateOn` steps inherit the
frame.

**3.2 A failed eval stops its children.** When an eval fails (exception, compile rejection or `timeout_ms`),
the natural-language calls it started that are still running are aborted, then drained, which is quick. This
gives eval failure the same scope as its existing atomicity: the model is not left with invisible work
still running.

**3.3 An eval that ends with children running stops them.** Today the session waits for them and then fails
the eval. Instead, abort them and fail the eval with the existing message plus "they were stopped." There is
one exception: calls that lost a `Promise.race` or `Promise.any` are aborted silently and are not an error.

- To know which calls lost a race, the eval realm's `Promise.race` and `Promise.any` are wrapped to mark
  their inputs.
- Abort happens at eval end, not when the race settles. So the JavaScript timeout idiom
  `Promise.race([call(), new Promise((_, reject) => setTimeout(() => reject(new Error('slow')), ms))])` and
  first-of-several now work as written. Awaiting a race's loser later in the same eval also still works.
- `Promise.all` keeps JavaScript semantics: siblings of a rejected input keep running, because a program
  may still await them.

**3.4 A failed call stops its subtree.** When a call ends as failed, blocked or out of limits, its controller
aborts its remaining descendants.

**3.5 Model-facing text.** The `timeout_ms` description (`ts-host/src/native/agent.ts:518`, `:563`) changes to: "A
timeout stops the natural-language calls this eval started; service calls and other effects already made
remain." The spec's Eval and Model surface sections change to match.

**3.6 Host API unchanged.** `runtime.run(fn, { signal })` stays the handle for cancellation and deadlines.
Passing `AbortSignal.timeout(ms)` now propagates through every call in the task by 3.1.

**Gate.** Tests cover: a race between a call and a timer; first-of-two; race then awaiting the loser; a
failing `Promise.all` with a caught error; eval timeout with running children (the eval returns at the
timeout, the children's model requests are aborted, and the trace shows both); call failure aborting
grandchildren. The full suite passes.

## Phase 4: `for await` over host streams

**Allow `for await` over any source** (`compiler/policy.ts:24`). The lowering wraps the source:

- arrays, Maps and Sets go through `finite()`;
- objects with `Symbol.asyncIterator` pass unchanged;
- `iterateOn(...).streamUntil(...)` streams pass as today.

After 1.2, restricted code cannot define an async iterable, so every async iterable comes from the host:
`fetch` response bodies, service streams or packages. This covers `for await (const chunk of response.body)`,
streamed service results and paging through results with a cursor.

No `.take()` helper and no item limit are added. Consuming an outside stream is paced by the outside world,
and the host's streams are host authority.

**Gate.** Tests: a response body; a service returning an async generator; an array of promises; a refused
program-defined async iterable.

## Phase 5: Structural recursion (owner sign-off before implementing)

This changes the spec's rule that recursion is impossible by construction, for TypeScript functions only.

**Motivation.** Small models write recursive tree walks naturally. Trees are everywhere in applications:
comment threads, menus, file trees, JSON, syntax trees. The current alternative is a work-list loop with
`iterateOn` and a measure, or a fixed-depth loop like conformance program 19. Small models find both hard.

**Rule.** Re-entering a TypeScript function that is already active in its call chain is allowed when an
argument is smaller than in the nearest active call of the same function. Smaller means one of:

- a value reachable from the earlier argument (a part of it),
- a shorter array or string,
- a smaller non-negative integer.

The same argument position must keep decreasing along the chain, and a part may not repeat within it.
Anything else is still a `NatlangRecursionError`. Natural-language functions keep the context-graph rule
unchanged.

**Model-facing sentence:** "A function may call itself only on a smaller argument: a part of its input, a
shorter array or string, or a smaller non-negative integer."

**Termination.**

- Integers and lengths: they cannot keep decreasing forever.
- Parts: a chain of distinct parts of a finite value is finite.
- The documented gap: code that grows the structure while descending (attaching a new child, then
  recursing into it). That is deliberate generation, not an accidental loop.

**Implementation.**

- Guard entries keep references to their arguments.
- On re-entry, the guard compares positions with the nearest active entry of the same ID. The part check is
  a breadth-first search over own enumerable values that stops at the first match.
- Remove the static `findRecursion` error for TypeScript functions; the runtime check runs on the first
  re-entry. Conformance program 19 gets a recursive variant.
- Concurrent descent (`await Promise.all(node.children.map(walk))`) works because each async branch carries
  its own chain.

**Gate.** Tests: tree walk, mutual recursion over a tree, list recursion on `xs.slice(1)`, countdown on a
non-negative integer, and refused cases (same argument, parent from child, growing array, negative integer,
alternating positions). Then a small evaluation of model-written tree code with the student models before
and after, to confirm the change helps rather than confuses.

## Phase 6: Application layer (host library)

No language change; additions to `ts-host/src/app/event-loop.ts`. Model calls may stay inside `reduce` as
today. These pieces let an application move slow work out of the queue when it needs to.

**6.1 Per-key loops.** `KeyedEventLoop` takes `key(event)` and `initialState(key)`, creates one `EventLoop`
per key lazily, and passes the key to `onCommit` for per-key persistence. Events are serial within a key and
parallel across keys, so one user's model call no longer blocks every other user (Clojure agents, Erlang
processes, Durable Objects).

**6.2 Follow-up work.** `StepContext.after(work)`:

- `work` receives a signal and returns an event, or nothing. It runs after the commit is stored, and its event
  is dispatched to the same loop.
- Failures go to `onFailure` with stage `job`, and `close()` aborts pending work.
- This is Elm's `Cmd` in one function. It replaces the hand-rolled job pattern in `TERMINAL_APPLICATIONS.md`.

**6.3 Clock.** `StepContext.now` is fixed per step and stored in the commit record. Reducers use it instead
of `Date.now()`, so replaying a journal is deterministic.

**6.4 Wake-up.** Option `wakeAt(state) => number | null`.

- Each loop (each key) keeps one timer at that time and re-arms it after every commit and at start, so state
  restored after a restart re-arms itself.
- The timer dispatches `{ id: 'wake:<revision>:<time>', kind: 'wake', at }`.
- Deadlines, reminders and digests become state. This is the Durable Objects alarm model.

**6.5 Validation.** Port `applications/workflow` (per-order keys; `after` for external effects; `wakeAt` to
reconcile an unacknowledged charge). Add an example in which two users' events interleave while one user's
model call is pending.

**6.6 Docs.** `ts-host/FRONTEND_APPLICATIONS.md`, `ts-host/TERMINAL_APPLICATIONS.md`, `skills/natlang-integration`.

**Gate.** Unit tests for ordering within a key and parallelism across keys, `after` (commit before
dispatch, failure, abort on close), `now` stored in commits, and `wakeAt` re-arming after commit and
restart. The ported workflow application's tests pass.

## Phase 7: Bash and Python tools of directory reducers

Audit the same way: probes first, fixes only for confirmed cases.

- **Bash candidates:** `yes`, `seq` with huge bounds, pipelines that never close, recursion through
  indirection, `xargs`.
- **Python candidates:** recursion through lambdas, methods or higher-order calls (the static check sees
  only named calls); `iter(callable, sentinel)`; and infinite iterators consumed by `list`, `sum` or `max`
  rather than by `for` (only `for` and comprehensions are wrapped).

Python cells already have a 30-second default timeout (`ts-host/src/native/folder-python.ts:79`), so their
leaks are bounded in time. Report findings here before changing anything.

## Phase 8: Spec, skills and prompts (with each phase)

- `spec/SPEC.md`:
  - Iteration and termination: eval parity, iterators, timers, `for await`, and structural recursion if
    accepted.
  - Eval: timers end with the call; children are stopped when an eval fails or ends.
  - Model surface: the `timeout_ms` text.
  - A short statement of what termination guarantees and the threat model.
  - A note that rebinding to another context is a host capability, and that the recursion identity includes
    the context.
- Skills: `skills/natlang-authoring/SKILL.md`, `references/language.md` and `references/patterns.md`;
  `skills/natlang-integration/references/recovery.md`.
- The built-in docs in `ts-host/src/native/runtime.ts:130` and the tool descriptions in `ts-host/src/native/agent.ts`.
- Every new refusal message names the standard alternative in one sentence.

## Phase 9: Training data and coordination

- **Scan existing teacher records and corpora** for eval code that a phase newly refuses: `setInterval`,
  `Symbol.iterator`, `Iterator.from`, non-finite or changing loop bounds, recursion through methods, and
  `chunk`/`windows` with zero. Record the frequency here. Old trajectories remain valid records of the old
  runtime and are not rewritten; regeneration is considered only if the frequency is material.
- **New collections differ.** Prompt and refusal text changes alter new collections. Note the runtime
  revision in collection manifests as usual.
- **Brief Pop after each landed phase** with commit hashes. Pop's frozen runtimes adopt the changes at their
  next refresh.

## Deferred until a problem shows up

| Item | Trigger |
|---|---|
| Limit on concurrent calls per task | Memory or latency trouble with large `Promise.all` fan-out |
| Ordering sibling calls by shared prompt prefix; batching decision readouts | A measured throughput gap on the local server |
| Caching deterministic calls without services | Repeated identical calls visible in traces |
| A per-call deadline API | An application needing deadlines beyond the task signal |
| Recording clock and random reads for replay | Replay mismatches in training data |
| Effect operation IDs for idempotent retries | An application needing them beyond its own keys |
| Re-entry through rebinding (`kernel.ts:294`) | Program code, not host code, gaining the ability to build contexts |
| Per-event concurrency policies, derived-value types, state migrations, queries | A concrete application needing them |
| Default timeouts, gas counters, cascade limits, timer rate limits | Not planned |

## Order

1. Phases 0 and 1, then 3 and 4. These are small changes with Phase 8 text alongside.
2. Phase 6.
3. Phase 5 after sign-off.
4. Phase 2.
5. Phase 7.

Phase 9 runs throughout.
