# `iterateOn`: the sanctioned adaptive iteration operator

## Surface

Every compiled natlang callable, whether named `.nl` or inline `nl`, exposes an `iterateOn` method. A free function offers the same operation when a function value is already being passed around:

```ts
const deathStar = await improvePlans
  .iterateOn(deathStarPlans, engineeringConstraints)
  .until(nl`The power level exceeds the power of the Force.`);

// Exactly the same operator; the method is the usual spelling.
const same = await iterateOn(improvePlans, deathStarPlans, engineeringConstraints)
  .until(nl`The power level exceeds the power of the Force.`);
```

The callable is the **step**. On iteration `n`, invoke `improvePlans(stateN, engineeringConstraints)` and require a new value of the same state type `T`. `engineeringConstraints` and any other trailing arguments are evaluated once when the iteration is created and passed to every step call by normal JS identity/reference semantics. The result of one step is the state for the next. The `until` predicate receives the current state and returns `boolean` or `Promise<boolean>`. Check it on the initial state before taking a step and after every completed step; stop on `true` and return the checked `T`. A step or predicate error rejects the operation with its trajectory attached, without prompting the step model to fudge the result.

An approximate type contract is:

```ts
type Step<T, A extends unknown[]> = (state: T, ...args: A) => T | Promise<T>;
type Done<T> = (state: T) => boolean | Promise<boolean>;
type IterationEvent<T> = Readonly<{
  kind: 'initial' | 'step' | 'review' | 'done' | 'error';
  iteration: number;
  state?: T; // exposed as a read-only observation
}>;
type ProgressJudge<T> = (trajectory: IterationTrajectory<T>) =>
  Promise<{ verdict: 'continue' | 'divergent' | 'needs_help'; reason: string }>;

interface Iteration<T> {
  until(done: Done<T>): Promise<T>;
  streamUntil(done: Done<T>): AsyncIterable<IterationEvent<T>>;
  onStep(observer: (event: IterationEvent<T>) => void | Promise<void>): Iteration<T>;
  checkProgress(judge: ProgressJudge<T>): Iteration<T>;
  withSiteId(id: string): Iteration<T>;
}

declare function iterateOn<T, A extends unknown[]>(
  step: Step<T, A>, initial: T, ...args: A
): Iteration<T>;
```

The compiler gives all natlang callables a matching `.iterateOn(initial, ...args)` attribute. It is present as a standard method even when a specific callable cannot serve as a step; attempting to invoke it then gets a precise type error, for example when its return type is not assignable to its first input type. This includes directory reducers: they use the same operator if their typed return represents the next state and their folder authority is valid. No second reducer protocol is introduced. The Natlang function listing should mention `iterateOn` once as a common callable method, without repeating the method signature under every imported function. Reserve `iterateOn` as a child/export name on natlang callable objects so it cannot collide with the standard method.

The contextual callback type at `.until(...)` gives an inline `nl` predicate its input `T` and return target `boolean` before generation. The step already has an input `T` and checked return `T`. A named step may take additional typed arguments; a named or inline stopping predicate can read captures through its normal closure. The free and method forms lower to one runtime operator, not separate execution engines.

An `Iteration<T>` is a single-use plan. `until` and `streamUntil` each start it once; calling either again fails instead of duplicating effects. `onStep` observes completed, checked states without controlling progression. `streamUntil` yields initial/check, step, progress-review, completion, and failure events with sequence IDs and returns the same final state in its completion event. In constrained callable-folder TypeScript, consuming this **branded** stream with `for await` is the explicit exception to the general ban on open-ended async iteration. Arbitrary async iterables remain prohibited there. A caller may also consume the stream from unrestricted host application TypeScript.

## One stateful runtime path

This **replaces** the existing `IterateNode` / `$iterate` / `until` path. The current node has `init`, `step`, `check`, a mandatory `max`, recent states, and a repeated-state hash. Do not keep a parallel public node or compatibility alias. Move useful repeated-state detection and trace events into the library operator, then remove the old parser/type/runtime branches, source generators, docs, and prompt references after their callers migrate. Existing serialized Iterate traces need an explicit IR migration where the step, check, and state transitions are recoverable; regenerate those that are not. No default hard max-iteration count is imposed. A caller can opt into an explicit maximum or deadline for its own application.

The operator runs in the current Natlang task context: the same model driver, scoped callable namespace, live values/handles, folder authority, cancellation signal, and trace sink. A completed step is a normal child invocation and must pass the runtime `T` check before its state becomes current. The step calls are sequential and permitted by the no-recursion policy; a step that calls itself while active is still rejected. Optional directory reducer patch commits follow the existing folder semaphore. An observation callback cannot mutate the committed iteration state behind the operator; any mutation must be part of a step result or authorized folder transaction.

At each boundary record `iteration_id`, `site_id`, source revision, step/predicate definition IDs, initial/current state references, fixed argument references, child call IDs, elapsed active time, cumulative model turns/tokens, state hashes, check outcome, and any external effects. Retain the **whole** trajectory. A progress judge receives a read-only `IterationTrajectory<T>` with indexed access to all states, calls, traces, and effects plus a compact summary; do not paste the whole transcript into its opening. It can page or query any part, so it genuinely has access to the entire trajectory. Store its reasoning/verdict in trace and training IR. A judge return type can be `{ verdict: 'continue' | 'divergent' | 'needs_help'; reason: string }`; `divergent` ends with a typed `IterationDivergedError` carrying the last checked state and trajectory, while `needs_help` reports a blocker to the caller. The judge is not allowed to rewrite prior steps or results.

## Call-site identity and adaptive checks

The natlang compiler assigns each `iterateOn` expression a stable site ID from normalized project-relative module path, enclosing exported symbol, and source-node identity. The source revision is recorded separately, so ordinary code edits do not automatically discard a site's history. Multiple dynamic invocations of one expression share a site ID but receive distinct iteration IDs. If a site is moved or refactored, `.withSiteId('project/feature/plan-improvement')` gives an explicit durable identity; diagnose duplicate explicit IDs in a project. For uncompiled dynamic calls, require an explicit site ID before enabling persistent statistics rather than silently grouping unrelated loops.

Statistics are scoped to project + site ID + step/predicate revision + model/configuration cohort. Track successful run step counts and active step/predicate time separately, using online mean/variance and a few robust quantiles; exclude progress-judge time, model loading, and paused time so reviews do not cause more reviews. Record wall time as a separate diagnostic. Track divergent/error runs separately so failures cannot inflate the expected successful duration. Persist only aggregates and source/model fingerprints when the runtime is configured with an iteration-statistics store. The default is in-memory for the runtime lifetime. Node can use a local store adapter, browsers IndexedDB or an application adapter; neither storage format is part of the Natlang language. Expose read/reset/export APIs for diagnosis and deliberate migration after code changes.

On a new site, use permissive bootstrap thresholds and occasional checks after completed steps; do not presume a zero-variance baseline. After enough successful samples, crossing roughly one standard deviation above the site's typical step count **or** active time begins progress review. Increase frequency across further bands, for example every four completed steps after `mean + 1σ`, every two after `mean + 2σ`, and every step after `mean + 3σ`, with an analogous active-time trigger. Use minimum sample counts and absolute floors to prevent noisy short runs from prompting checks at every step. These thresholds trigger **review, not automatic termination**. A progress judge can affirm useful persistence even on an unusually long run, and that evidence becomes part of its trajectory. If the model is stuck inside one step, this boundary-based mechanism cannot inspect fresh iteration progress; cancellation or model-driver timeouts remain separate controls.

The default progress judge is a library-provided natlang lambda with read-only access to the whole trajectory and an instruction to distinguish meaningful progress from repetition, oscillation, unproductive tool use, and impossible goals. `checkProgress(customJudge)` replaces it for a call; runtime configuration may supply a project default. Include the original step instructions, stopping predicate instructions when natlang-authored, typed state/argument descriptions, statistics, and sampled trajectory index in the judge opening. Avoid repeatedly feeding the judge's warnings back to the step agent: a divergent step should surface as an error to the caller, while a `continue` judgment simply allows another step. This preserves the project's preference not to badger an interpreter into fudging.

Repeated states are evidence, not an unconditional failure: use identity/hash where available, report cycles to the progress judge, and let it account for legitimate revisits with new evidence. If the state is a live object or folder, compare checked observations/revisions rather than attempting a JSON dump. Let the caller optionally specify a stronger progress measure, but do not require one for the basic API.

## Compiler and training integration

- Add the method to the compiler-generated `NatlangFunction` type and its runtime callable object. Recognize free and method calls as the same operator. Infer `T` and trailing argument tuple from the step's actual TypeScript signature; reject incompatible return, optional/spread ambiguity, or a predicate target other than boolean.
- Generate the call-site manifest and diagnostics in `natlang check`, with Node/browser parity. Reject attempts to manufacture a branded `Iteration` or call the operator outside an authorized runtime context. Treat `iterateOn` as the only sanctioned open-ended iteration construct in constrained `.ts` and `eval` code.
- Implement the scheduler as a library over the shared invocation kernel, not as `NativeRuntime.runRoot` on an `IterateNode`. Factor model calls, stats, trajectory storage, observer delivery, cancellation, and the judge into testable components. The judge itself is a normal natlang child with a read-only trajectory handle.
- Trace and IR preserve the choice of step, fixed arguments, initial check, each state transition, every stopping check, statistic-triggered review, judge reasoning, and final outcome. Generate teacher trajectories with normal completions, long-but-progressing runs, repeated states, genuine divergence, and explicit caller recovery. Include contrastive samples where the judge correctly permits an unusual run to continue.
- Migrate examples, Studio generators, authoring/integration skills, specification, and training materializers from `$iterate` to `.iterateOn(...).until(...)`. Add a realistic folder reducer example and a streamed-progress UI example. Validate a small teacher collection and training export after cutover.

## Acceptance

An implementation is ready when the method and free forms produce the same state/check/review sequence apart from their site and invocation IDs; `nl` stopping predicates get `T → boolean` targets; extra arguments reach each step in order; initial-state success takes zero steps; runtime return errors never become new state; and streaming observes exactly the committed sequence. Node/browser parity covers call-site IDs, live objects, folder reducers, cancellation, and optional persistence. Fresh and statistically unusual sites trigger reviews at the intended boundaries; a progress judge can stop a divergent run or permit a slow but improving one. The old Iterate node has no supported public entry path after migration.
