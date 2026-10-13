# Effects, concurrency, and recovery

## Values, live handles, and native lifetime

Portable values (records, arrays, strings, finite numbers, booleans, null) are copied into an invocation and checked at its boundary. Everything else — functions, class instances, DOM nodes, database clients, processes, folders — is passed by reference as a live handle and stays owned by the application. The model sees a preview and can call methods; the handle is released with the task. Do not invent a universal serialization protocol for native objects.

For `nl.with<ResultType>({ fixedContext })`, the single type argument declares the child result and the object supplies captured context; pass changing values to the returned child function. If an eval reports `missing-fresh-result`, omit `finish` or set it to false while inspecting or staging; `finish:true` needs a fresh typed final expression or explicit `return`. Treat an incorrect result type as a rejected action and revise the value; do not coerce or wrap it to bypass the declared contract.

## What is and is not rolled back

| Change | When it takes effect | On failure |
|---|---|---|
| Captured `let` reassigned in eval | After the eval succeeds, if nobody else changed it | Discarded; a concurrent change is a `capture-conflict` and the eval retries |
| Property write on a live object | Immediately | Stays; traced |
| Service method call | Immediately | Stays; traced as an effect |
| Directory reducer file edits | On `commit`, installed by `folder.apply` | Discarded with the transaction |

For an operation that must not repeat:

1. Persist a stable operation ID and the exact request before executing it.
2. Record the observed result with that ID.
3. If interrupted without a result, keep the outcome unknown and reconcile with the external system before any retry.

A timeout can mean the operation is still running; a rejected return can follow a successful mutation. Neither justifies an unconditional retry. The workflow application (`applications/workflow/`) shows intents written before effects, idempotent receipts, and reconciliation after a lost acknowledgement.

## Concurrency

Independent tasks run concurrently, and sibling natlang calls in one task (`Promise.all`) run in parallel. Only folder writers serialize. A model backend may queue requests; that is resource scheduling, not a semantic limit. Order effectful work explicitly where it matters; several tool calls in one model response are an ordered, non-atomic batch.

For `no-such-function` while a named child tries to call the next stage, inspect callable-folder nesting. A child sees its own descendants, not the parent's other children: `root/inspect/decide.nl` makes `decide` callable from `inspect`, whereas `root/decide.nl` is only a sibling. Keep the existing named function and move the file into the caller's folder; use `read_code('decide')` inside `inspect` to confirm visibility before retrying collection.

When an intended per-item reducer produces one giant NL call over a serialized directory, inspect the named helper's argument type and root loop contract. Give the helper a single typed record with a source path, keep cross-record evidence in a separate typed context, and have the root advance one record per `iterateOn` step with a decreasing frontier. Review actual child invocation arguments and returned source IDs; a correct final answer alone does not establish the intended topology.

If that typed-record loop repeats a deep named chain for every record in a coherent source folder, a typed folder batch can retain the semantic work with fewer chain setups. Iterate over source folders with a shrinking frontier, pass an `EvidenceBatch` of `{path,text}` records to each stage, and require one typed finding per original path. Inspect the actual returned path multiset and citations: missing, duplicate, or fabricated source IDs mean the batch lost evidence and must be repaired before generation or training admission. A batch must not be serialized into an untyped string. If an authored scoped-literal plan names a context field absent from the emitted type, bind the plan to the actual field name before collection.

If a heterogeneous record array is declared as a union and the runtime says a field from one arm is missing on a valid value of another arm, inspect the union completeness check. Each element should be tested against every structurally compatible arm until one is complete. Preserve the authored union and source meaning; do not add synthetic fields merely to appease the checker.

## Cancellation and limits

`runtime.run(fn, { signal })` and `EventLoop.cancel()` abort the task; natlang calls in flight reject. Passing `AbortSignal.timeout(ms)` gives the whole task a deadline. Inside a call, cancellation follows the call tree: a call that fails stops the calls it started, and an eval that fails or times out stops the natural-language calls it started (the losers of a `Promise.race` are stopped when their eval ends). Native work already started keeps running unless the application cancels it. `limits` (`maxDepth`, `maxActions`, `maxToolCalls`, `maxEpisodes`) and model options are deployment policy; `maxDepth` counts nested natural-language calls.

A `return_result` that carries no usable result is rejected in one sentence that names the result type: `return_result needs the Judgment itself as its value argument.` (a result sent under another argument name, or none), or `return_result needs a complete Judgment as its value argument, with every field filled.` The fix is to call `return_result` again with the result in `value`, or to return it from an eval. The wording is the runtime setting `returnErrorText` (or `NATLANG_RETURN_ERROR_TEXT`): `sentence` (default), `steps`, `field` and `legacy` (`value: bad-action, expected ...`), kept for measurement; small models read the coded form as "the value was bad-action" and echo it back.

A call whose model keeps retrying the same failing eval is told so in the eval's error ("This is the code of your previous eval, and it failed the same way"). On LFM2 models, a reply that holds `<|tool_call_start|>` markup the server could not parse as a call (for example a `return_result(...]]` with a wrong bracket) is a malformed call: it is retried once with the malformed-call feedback and never read as the call's text answer. In traces, look for repeated identical `eval` codes or `model returned malformed tool arguments` when a call does not finish. An eval that declares a name the scope already binds fails with an `invalid-binding` error naming that binding's kind (function, service, input, captured variable or built-in) and the fix: keep using the existing binding and put a new value under another name (`shipmentsResult`). A `return_result` that sends its result under another argument name (for example `verdict`) is rejected with "expected the result in the value argument (this return_result put it in \"verdict\")"; the fix is to resend it as `value`, or to return it from an eval.

## Continuation and restart

Long invocations keep one conversation; past the model's `contextTokens` budget the oldest tool outputs and eval code are elided, while the eval scope carries all values. That is not process restart recovery: persist application state, event IDs, operation receipts, and seeds, and rebuild native resources through the application's own recovery path.

## Traces and reproducibility

Each invocation produces a trace (`trace` sink on the runtime or task): manifest, presented messages and tools, actions, observations, effects, state, outcome, and the call tree. Inspecting a trace, reconstructing its final value, and replaying effects are different operations; only the first two are implemented. Model seeds derive from invocation identity; keep world randomness in a separate seeded generator. Reproducibility needs the same source, model, template, sampling, inputs, ordering, and relevant environment, and still differs across inference backends.
