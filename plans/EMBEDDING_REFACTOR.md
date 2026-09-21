# Refactoring the prototype for small embeddings

Started 2026-09-19. Scope: implementation boundaries with unchanged language behavior. The Python executor and model seams, filesystem-independent definition graphs, TypeScript `EvalEnvironment`, platform-neutral native reducer, explicit seeds, trace data, streams, and host-specific Node/browser packages are implemented. Remaining work is recorded as such below.

The sections retain the original rationale and acceptance criteria. Current distribution boundaries are `@natlang/core`, `@natlang/node`, `@natlang/browser`, and `@natlang/cli`; see [native packages](../NATIVE_PACKAGES.md).

Follow-on design decisions are specified in [EXECUTION_INTERFACES.md](EXECUTION_INTERFACES.md): multiple engines, proposed required engine selection, shared-host environments, explicit seeds, streams and trace data. Keep the behavior-preserving extraction separate from those versioned changes.

For portfolio-wide priority and patch dependencies, follow [INFRASTRUCTURE_IMPLEMENTATION.md](INFRASTRUCTURE_IMPLEMENTATION.md). R0–R7 below describe refactor groups; the newer patch sequence is the delivery authority.

## Recommendation

Yes: separate a few prototype dependencies from the semantic core. Start with crisp execution, the actual model-turn interface, and source loading. Those are concrete replacement points. Keep the current implementations as defaults and preserve the existing model surface.

Use a small host-supplied engine mapping and callable protocol. No dynamic plugin discovery, general scheduler, portable VM or new intermediate language is needed for this work.

The desired result is that an embedding supplies a programme, an interpreter driver and the crisp implementations it needs. The Python/QuickJS/llama-server combination remains one convenient host. We can then learn which further changes a browser, a single-threaded game or a parallel batch runner actually requires.

## 1. Crisp execution: first extraction

**Evidence.** `natlang/runtime.py` imports `js` directly. `_run_crisp` and `Session._do_eval` construct JS-specific scope values with `js.to_js`, call `js.run`, and catch `js.JsError`. `_fx` raises `js.EffectError` for capability failures. `natlang/js.py` implements QuickJS setup, TypeScript preparation, limits and process execution.

**Refactor.** Inject a crisp executor into `Runtime`, defaulting to the existing adapter. Route both crisp function bodies and inline snippets through it. Move JS-specific value conversion and exception translation into that adapter. Use a small shared execution-failure type where necessary, rather than making authority errors depend on the JS implementation.

An illustrative internal contract is:

```text
execute(body, bindings, mode, environment, invocation) -> value or failure
```

This is not a new natlang tool. Preserve body versus expression/snippet semantics explicitly; do not collapse them accidentally. Inputs must preserve missing values, immutable snapshots and pending-value treatment according to the current contract. The core validates the returned value. Natlang-owned state remains behind checked boundaries. The adapter may intentionally share host-owned objects; direct host access and mediated effect checks have different enforcement guarantees, declared by the embedding.

First preserve the existing JS/TS evaluator and test the seam with a recording/fake executor. Then exercise a deliberately shared-host implementation and a second engine as specified in the execution-interface plan. A fake proves interface substitutability, not cross-engine semantic equivalence.

**Preserve.** Current return validation, diagnostics, effect order, evaluation limits, code preparation and transcript-visible results. This first extraction preserves the current tool. A subsequent versioned change proposes required `engine` selection on `run_code`, with a constrained set of available engines. No automatic translation of arbitrary JavaScript to another language.

**Benefit.** An embedding no longer needs to pretend every exact function is run by this particular Python QuickJS wrapper. It still must supply whatever evaluator the programme actually depends on.

## 2. Model execution: strengthen the seam we already have

**Evidence.** `Runtime` already takes `agent_factory`; retain that useful boundary. `ToolAgent` accepts a decoder and optional `ToolSurface`. However, the declared `Decoder` protocol specifies `format` and `generate`, while `ToolAgent` primarily calls `chat`. `NativeCallDecoder` inherits `LlamaServerDecoder`, connecting native syntax/constrained decoding to one server implementation. `ToolAgent` sets mutable decoder/runtime deadlines and currently supplies `seed=0` on ordinary turns.

**Refactor.** Define the small model-turn protocol that `ToolAgent` actually consumes, using the existing `ChatTurn` representation where suitable. Keep raw-generation/template interfaces for drivers that require them; a chat-only embedding should not implement unused llama-server endpoints. Keep tool aliases, transport conventions and token syntax in the relevant model/backend adapter.

Make invocation settings, including seed and deadline, explicit at the driver boundary. Preserve current defaults in the compatibility pass. Shared-seed merge execution can then supply a deliberate seed policy without editing the interpreter loop or teaching the model to manage randomness. Avoid shared mutable request state before allowing concurrent calls.

Optional token probabilities, raw responses and constrained generation remain optional capabilities with explicit unsupported behavior. A mode requiring review/probability data must not quietly claim the review ran when the backend cannot supply it. Do not create an elaborate negotiation protocol just to represent two known interfaces.

**Preserve.** The canonical actions, tool schemas, prompts, completion behavior and trace capture for a matched configuration. Changing when the seed varies is a separate experiment, not a hidden refactor.

**Benefit.** Local, remote and replay drivers fit the same episode loop without making llama.cpp's API or an LFM token format the language definition.

## 3. Source loading: separate packaging from programme meaning

**Evidence.** `codebase.py` already has `FunctionDef` and separate file/inline entry paths. The file loader combines YAML frontmatter, `Path` access, companion-folder discovery, relative links, named-type inheritance and graph checks. `host.py` binds input values from files and guesses whether short strings name existing paths.

**Refactor.** Consolidate validation/linking of already supplied function definitions. Keep directory/frontmatter loading as one frontend that produces those definitions. Expose an explicit value-binding route for embeddings alongside the existing CLI file-binding convenience. A text value in an embedded programme should not turn into a filesystem read because its spelling happens to match a path.

For a browser or packaged game, feed a preloaded codebase graph or a minimal source resolver. Retain lexical scope, named-type rules, effect declarations and cycle checks. Do not invent a source archive format, package resolver or new IR unless an actual target needs one. Reuse existing representations first and document identity/link semantics.

**Preserve.** Existing `.nl`/`.ts` layouts and `uses` resolution. The CLI can keep its current convenience with explicit documentation; a future CLI flag change should be separately reviewed. Embedded value loading and source linking must not depend on a working directory.

**Benefit.** Programmes can come from disk, bundled resources, editor buffers or an in-memory fixture without changing execution semantics or requiring every target to emulate a POSIX filesystem.

## 4. Execution context: small cleanup, then defer scheduling

**Evidence.** `Runtime` keeps mutable `_depth`, `_stack`, `_fn_stack`, a deadline and counters. `ToolAgent` temporarily modifies runtime/decoder deadlines. Trace entries use `id(self.lam)`. `runtime.py` changes the process-wide Python recursion limit at import time. These choices are workable for the current synchronous prototype but unsuitable as an implicit concurrency contract.

**Refactor when touched.** Make per-run/per-episode ownership explicit. Keep limits and model-request settings with the invocation that owns them. Avoid process-global changes on library import; preserve necessary recursion behavior through an explicit host setup or a tested implementation change. Use logical IDs where a trace actually needs identity beyond one Python object lifetime.

Do not turn this cleanup into a serialisable scheduler. Keep serial execution as the reference implementation. Before implementing parallel Map, isolate each child's execution context and define how results, budgets and diagnostics combine. A cooperative implementation can later suspend at existing call boundaries; it should not require model-visible futures.

Use streams consumed by Map/Fold for incoming events first. Complete waiting/closure/backpressure behavior at host boundaries; defer in-episode interrupts. This cleanup can support later experiments without requiring an inbox or coroutine protocol now.

**Benefit.** Reentrancy, controlled randomness and diagnostic clarity improve before concurrency is introduced. A single-threaded host retains a straightforward implementation.

## 5. Research options: keep them available without making them normative

**Evidence.** `ToolAgent` supports validation feedback, careful review, review prompts/order/scope, withdrawal policy and several logging/capture choices. Those experiments have value, but their constructor surface is not a good definition of the minimum interpreter.

**Refactor only if it simplifies a real change.** Name a compact default execution configuration and keep experimental configurations explicit. Separate recording from decisions where practical. If extraction is helpful, use a concrete review component rather than a general middleware/hook framework with arbitrary ordering. Preserve the experiments and their tests.

**Benefit.** An embedding can understand the default path without implementing every research feature. Training manifests still record the actual selected policy; architectural cleanup must not change the data's meaning.

## 6. Work order and stopping point

| Step | Change | Evidence before continuing |
|---|---|---|
| R0 | Characterise current semantic behavior and known spec drift | Representative fixtures for successful, invalid, quiesced and effectful calls |
| R1 | Inject crisp executor; translate JS-specific representation/errors inside adapter | Existing JS/TS and effect tests; recorded-adapter substitution; unchanged canonical outcomes |
| R2 | Correct the model-turn protocol; make request settings explicit | Existing backend paths and replay fake; same default action/completion behavior |
| R3 | Expose filesystem-independent definitions and explicit value binding | File/inline/in-memory inputs produce equivalent checked definitions and outcomes |
| R4 | Try minimal embeddings with these seams | No filesystem for an in-memory run; no network for recorded model turns; no unrelated host services |
| R5 | Version engine selection; support shared-host environments and explicit random API | Adapter contracts and seed test vectors; migrated tool/trace fixtures |
| R6 | Portable execution/reduction trace records | Common reader, state reconstruction and honest opaque-host limits |
| R7 | Complete stream Map/Fold behavior | Waiting, closure, failures, bounded buffering and output semantics |
| Later | In-episode interrupts or opaque host-type syntax | A concrete need beyond streams or environment-local native values |

Each step is a small independently reviewable change. Do not combine new semantics, transport changes, default seed changes and refactoring in the same patch. Keep the original implementation available until equivalent behavior is established.

Validation should compare typed outputs, failed operations, completion/quiescence, effects and canonical action histories. Timing and incidental Python identities are not useful equality criteria. Preserve teacher audit information; source-hash changes may require new manifests even when semantics are unchanged.

The first refactor milestone is an in-memory embedding with a supplied decision driver and replaceable crisp executor. The next planned milestones are R5–R7 above. That is enough to test whether the seams are real. It is not necessary to ship a second language, durable engine or browser UI before calling the refactor useful.

## 7. Explicitly deferred

No universal scheduler, event bus, durable state service, plugin discovery, effect algebra, mandatory asset type, or replacement of the typed tree. Engine selection extends the existing tool rather than adding a separate global tool. No rewrite of functioning loader/type logic merely to reduce Python-specific code. The conformance contract makes a future port possible; no amount of abstraction in Python makes a browser implementation automatic.

The criterion is reduced coupling with unchanged interpreter obligations. If an abstraction adds more concepts than the concrete dependency it removes, defer it until a second consumer makes the common boundary clear.
