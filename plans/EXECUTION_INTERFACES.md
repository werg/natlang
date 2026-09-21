# Execution interfaces: streams, evaluators, types, randomness and traces

Design proposal, 2026-09-19. Companion to the [architecture](AMBITIOUS_ARCHITECTURE.md), [refactor plan](EMBEDDING_REFACTOR.md) and [roadmap](AMBITIOUS_ROADMAP.md). Nothing in this document is implemented or normative yet. Examples show proposed interfaces, not commands accepted by the current runtime.

Application consumers and staged acceptance gates are specified in [the project implementation plans](projects/README.md).

Concrete delivery order and source touchpoints: [infrastructure implementation plan](INFRASTRUCTURE_IMPLEMENTATION.md). This document specifies interface intent; the implementation plan controls sequencing.

## 1. Decisions and remaining choices

| Area | Direction agreed in the design discussion | Concrete proposal / unresolved choice |
|---|---|---|
| External events | Streams consumed by Map/Fold first | Define waiting, closure, buffering, failures and partial results; defer in-episode interrupts |
| Crisp execution | Generalise engines and decouple sandboxing | Require `engine` on `run_code` in a versioned surface; a small host-supplied engine table |
| Host-owned data | Keep native references in crisp environment where possible | Return ordinary summaries; no mandatory natlang asset/reference service |
| Search and meta-capabilities | Initially expressed through crisp code | Host bindings for search, load/run and trace access; no additional global tools |
| Types | Prefer one TS-style natlang type language | Retain the existing structural subset; host-object escape hatch remains a design question |
| Trace | Specify and develop execution/reduction traces as data | Stable event vocabulary, logical IDs, optional sinks, portable value views and honest replay limits |
| Randomness | Explicit API support | Run seed plus per-call derivation, backend configuration and recorded support |

These decisions supersede earlier suggestions that every evaluator be sandboxed, that a general artifact store be foundational, or that engine selection should necessarily be hidden from the model. They also narrow eventful execution to streams for the initial implementation.

## 2. Crisp engine selection

### Model-facing surface

Proposed versioned tool:

```json
{"name":"run_code","arguments":{"engine":"ts","code":"return host.search(query);"}}
```

Retain current code-body/snippet semantics when migrating; the snippet above is illustrative and the adapter must specify its accepted form. The `engine` argument is required and constrained to the engines available to the current function. With one engine, the schema has one allowed value. Do not expose a long inventory or make the model select security policy on every invocation.

An engine name identifies an execution contract, not just a parser. A host binds `ts` to a specific implementation, environment and language version for a run. The same alias cannot silently change implementation mid-run. Other aliases could be `sql` or `bash`, but only on hosts/programmes that use them. Engine identity and bindings enter the run manifest.

For authored crisp functions, select the engine once at load time. Existing `.ts` files resolve to the declared TS engine binding. Add explicit engine metadata only where required for other source forms; that is a separate loader/spec change. Ordinary `call` does not need an engine argument: the callee already identifies its implementation.

No automatic fallback from an unavailable engine to a different dialect. A SQL engine specifies statement/query behavior and result conversion; it need not pretend to use JavaScript's expression mode. The model's schema/documentation and the adapter contract must describe the relevant behavior. Test engine-selection errors separately from code-generation errors.

### Internal seam

A minimal conceptual interface:

```text
prepare(source, mode) -> prepared_source or diagnostic
execute(prepared_source, bindings, environment, invocation) -> value or failure
```

An adapter may combine preparation and execution internally. The initial engine table is an ordinary host-supplied mapping, not a discovery/install/plugin framework. Parsing, compilation, environment creation and isolation belong to engine implementations; result validation and language state updates retain their natlang boundary.

The request names function-body versus snippet mode, logical evaluation ID, deadline/limits requested by the host and random context. The implementation reports failures without implying that external changes were rolled back. It may complete synchronously or suspend the embedding at that call boundary. That implementation choice does not introduce a future type or `await` tool.

### Environment lifetime and isolation are separate axes

| Choice | Meaning |
|---|---|
| Fresh environment | Bindings/runtime state created for the evaluation |
| Retained environment | Host elects to retain engine state across evaluations |
| Isolated implementation | Code runs behind the isolation boundary supplied by that adapter |
| Shared-host implementation | Code runs in a host context with the native objects/functions that embedding intentionally exposes |

A retained environment can be isolated; a fresh environment need not be a security sandbox. Describe the actual guarantees of the selected adapter instead of inferring them from the engine language.

Shared-host execution is an intended embedding option. Direct native objects mean the eval code really shares access to that host environment; do not describe this as the existing isolated sandbox. Proxy objects with similar-looking methods are a separate, optional bridge for isolation or remote access. Native sharing also requires a compatible engine/host: JS eval cannot directly share an arbitrary Python object without an explicit bridge. It may expose live application objects, database connections, a DOM, native arrays, process APIs or model services. Do not wrap all such access into a mandatory broker merely to fit the old sandbox. Equally, declared effects must not be represented as enforced if direct native access bypasses mediation. Record the adapter's authority/enforcement contract and mark shared access as effectful in the programme's declared requirements. Access through aliases to host objects may mutate them even when the returned natlang value is ordinary data.

Keep a separate boundary around natlang-owned state. Initially pass snapshots/views for args/locals and validate returned values before they enter the tree. Sharing a host environment does not automatically grant arbitrary mutation of the interpreter's internal tree. If a future meta-API supports runtime mutation, use checked operations and trace them explicitly.

Do not promise forcible cancellation or hard memory isolation for an in-process adapter that cannot provide it. The embedding may reject work requiring unsupported guarantees or choose an appropriate adapter before execution. This is an embedding contract, not an extra series of choices for a weak model.

## 3. Host data, search and meta-programming

Keep native resources in the crisp environment: for example `host.images`, `host.db`, `host.files`, or an application object supplied by the embedding. Exact names are application/library conventions, not natlang built-ins. A search implementation can iterate values, use a text index or call a database without changing the model's global tools.

When a programme needs to select a resource across function calls, pass an ordinary ID or query description and resolve it in the appropriate environment. Do not require the full native object to enter the typed tree. Bind environments deliberately: a key created in one environment must not silently refer to a different object in another.

Meta-operations follow the same route. A development host can expose checked functions to load source, validate a codebase, invoke a child run, inspect its trace and compare results. The ordinary eval code can call them. Public natlang library wrappers can make recurring operations easier for the student. Search/load/run/inspect are not new top-level model tools.

A nested run retains its own scope, limits and authority. Parent/child trace links and budget ownership must be explicit; an eval call must not accidentally start unlimited invisible interpreter work. These rules are properties of the supplied meta-API, not a new general metaprogramming language.

## 4. One natlang type system

### Recommended baseline

Keep one TS-style type syntax and one definition of natlang type fit across engines. Start with the existing structural subset: primitive values, records with optional fields, lists, dictionaries, unions, named types and current pending types. “TS-style” does not commit us to the full TypeScript compiler/type system, structural class instances or every TypeScript feature. Keep current spellings such as `Text` and `Num` during this work.

Engine-local types are an implementation detail until a value crosses the boundary. Every engine adapter maps representable results into the same natlang value model, then the existing validator checks the destination. SQL NULL maps to Null; missing fields remain distinct from Null. An engine's undefined, large integer, nonfinite number, date or binary object must have an explicit conversion or be rejected at the boundary. Avoid implicit lossy conversions and per-engine interpretations of `fits`.

Write these conversion rules down once, with small cross-engine fixtures. A common boundary type system supports similar developer experience even when execution engines differ.

### Host-specific escape hatch: prefer no new syntax initially

First try keeping native values inside their environment and passing typed descriptions/IDs. That gives most projects access to rich host types without extending natlang's type algebra.

If real programmes need native values in arguments, locals or returns, consider an opaque nominal host type, schematically `Host<"Image">`. This syntax is a candidate, not a decision. It should be supplied by an embedding declaration and have exact identity/ownership checks, not an arbitrary plugin-defined replacement for subtyping.

Before accepting it, specify:

- Which host/environment owns the value and how its type identity is declared.
- Whether it may be copied, compared, transported or persisted; aliasing of mutable host objects must not inherit the assumptions used for ordinary immutable values.
- How the model can refer to an existing value without fabricating a native object through `write(value=...)`.
- How inspection and tracing represent it without requiring serialization of the object itself.
- What happens when an object is disposed, its environment ends, or a run moves to another host.

An opaque host type would be a bounded exception to structural typing, not `Any`. Do not introduce it merely to type a database connection the model never needs to move. Avoid a second host-specific type language in the interpreter. If the exception becomes common, revisit whether more exact work should remain within the crisp environment instead.

## 5. Streams through Map and Fold

Use existing combinator concepts. Map handles independent items; Fold carries state through ordered items. An open input means the reduction has not received its final input yet. This does not require a new model tool or immediate `Stream<T>` source syntax. The current OpenList convention is only a partial implementation and must not define accidental semantics for every target.

Initial contract:

- **Item available:** consume a typed item with a stable position in that input stream.
- **Temporarily empty:** wait/suspend in the host; do not complete or ask the model to poll.
- **Closed:** no further items; finish after admitted work completes.
- **Source failed:** report the input failure without pretending normal closure.
- **Fold:** finish the current step before applying the next event.
- **Map:** allow bounded independent processing; retain input-order result identity. The initial output contract preserves order, so a slow earlier item can delay ordered delivery. Do not invent out-of-order semantics implicitly.
- **Backpressure:** bound admission and buffering in the embedding. No silent loss of semantic events.

For an unbounded stream, materialising all Map outputs or retaining the entire input forever is not viable. Separate consumption/progress from eventual completion: allow the host or a connected stream consumer to receive committed per-item outputs; retain a bounded active window. Finite closed Map still yields its ordinary list. Precisely how a model addresses evicted items, what history remains inspectable and whether stream-to-stream composition needs an explicit reference are open spec items. Prototype a host-driven Fold first; do not claim the current list-based runtime already supports bounded streaming Map.

State snapshots and source cursors can remain host concerns. Deterministic replay needs the actual consumed item sequence, not just a source URL. An event arriving during a Fold step waits; in-episode interrupts remain deferred until this demonstrably prevents useful behavior.

## 6. Explicit randomness API

Proposed host-level run configuration includes `seed`, model sampling settings and selected engine bindings. A new API should require an explicit seed policy: a supplied root seed, or an explicit request for a newly generated seed. Record the resolved seed. Preserve historical defaults only through an identified compatibility mode, not hidden constants in the interpreter loop.

Use a fixed-width unsigned root seed representation across hosts, for example a 64-bit seed encoded as 16 hexadecimal digits in manifests. The precise public Python/JS argument shape can differ; the manifest representation and derivation algorithm must be identical.

Derive streams from the root seed and stable logical identity: parent call path, call ordinal, Map item position or Fold step, attempt and model-turn index. Include a purpose tag such as interpreter, review, helper-model or crisp-random. Do not include wall time, process identity, run UUID or completion order. A repeated logical run with equal inputs can then request the same draws despite unrelated concurrent work.

Freeze a versioned derivation algorithm with published test vectors before implementation is called portable. A reasonable candidate is domain-separated SHA-256 over unambiguously encoded fields, with explicitly specified conversion to each backend's supported seed range. Record the derived backend seed and conversion version. Do not let Python and JavaScript use their native object hashes.

Trace replay uses recorded decisions. Restart/resume continues recorded logical counters. A deliberate resample advances a named attempt/sample index; it should not happen merely because a transport retry occurred. A request with uncertain remote completion needs an explicit policy rather than an unrecorded extra sample.

The crisp environment may expose an invocation-scoped RNG derived independently of model sampling. Shared-host code that uses ambient randomness is allowed where the embedding allows it, but the trace must not claim the run is fully controlled by the natlang seed. A model backend that ignores seeds or is numerically nondeterministic must report that limitation. The API guarantees explicit random configuration and provenance; output repeatability is tested for the chosen execution profile.

## 7. Execution and reduction traces as data

### Purpose and boundary

Define one small portable event vocabulary for what natlang did: function invocation, proposed/applied actions, value changes, combinator progress, evaluator calls, input consumption and completion/quiescence. The same trace reader should explain these events whether execution occurred in Python, a browser, a single-threaded host or a parallel worker system.

A trace is optional output from execution, not a required database or an event-sourced implementation of the runtime. Provide a sink accepting structured records; an embedding can ignore it, collect an array, stream JSONL, or persist it. A development tool can consume the records through its crisp environment. Event sinks should not acquire authority to change execution just by observing it.

### Manifest

A trace has a versioned manifest containing run identity, programme/source and semantic version, selected model/prompt/tool schema, engine bindings/environment modes, resolved seed policy and capture level. Record which required information is absent or opaque. Put backend-specific details in namespaced metadata rather than changing the core event vocabulary.

Initial capture levels:

- **Diagnostic:** lifecycle and action/evaluation summaries; useful navigation, no replay claim.
- **Reduction:** structured accepted changes and relevant observations sufficient to reconstruct captured natlang state from a known initial state.
- **Replay:** also records external results, delivered stream items and model decisions needed for replay, or explicitly marks missing dependencies that prevent it.

Captured host effects or native object mutations may remain incomplete at any level; “replay” is a request whose achieved coverage is reported, not an automatic guarantee. Full raw model responses and exposed reasoning remain optional linked audit records, as in teacher trajectory IR.

### Record shape

Illustrative record, not a committed schema:

```json
{
  "schema": "natlang.trace/1-proposed",
  "run": "run-17",
  "producer": "worker-2",
  "seq": 31,
  "event": "action.applied",
  "node": "root/call/2/map/3",
  "attempt": 1,
  "action": "a7",
  "causes": ["worker-2:30"],
  "payload": {
    "operation": "write",
    "path": "return/label",
    "value": {"kind":"value", "type":"Text", "value":"urgent"}
  }
}
```

Producer-local sequence numbers plus explicit causal links avoid a mandatory global sequencer. Node/action IDs are logical identities stable for the recorded run; cross-run comparison uses logical call paths and content identity, not Python `id()`. Parallel wall-clock timestamps are optional diagnostics and never the authority for causality. A sequential host uses one producer. A run's manifest identifies producers and initial state; stream input IDs/positions are distinct from trace collection order.

### Events and semantics

| Family | Minimum meaning |
|---|---|
| `run.started`, `run.ended` | Manifest/initial-state link; final outcome and capture completeness |
| `node.started`, `node.completed`, `node.quiesced` | Call identity, parent/binding, result or diagnostic; completion includes value replacement |
| `action.proposed`, `action.applied`, `action.rejected` | Structured tool/name arguments and validation outcome; proposal is never mistaken for execution |
| `reduction.committed` | Accepted tree changes, pending-node replacement or combinator cursor/state advancement |
| `eval.started`, `eval.ended` | Engine binding, source/mode identity, result/failure and host observation coverage |
| `stream.item`, `stream.closed`, `stream.failed` | Input observed by the reduction, position and causal consumer |

Use one linked record of each logical fact: an applied action links to its committed reduction; do not independently serialize contradictory copies of state changes. Rejected actions record diagnostics and no accepted tree mutation. Rejection of a result after eval does not imply the evaluation performed no host effects. Completion and quiescence retain current semantics. If capture records interruption without a terminal event, mark the run incomplete rather than inventing completion.

A model turn can propose an ordered, non-atomic batch of independent actions.
Validate, trace and report every action separately. A rejected action does not
erase an accepted action and does not suppress later independent actions in the
same batch. A terminal action stops the remainder. Any action that needs a
value or diagnostic produced by another action belongs in a later turn. An
embedding may run proven-pure independent actions concurrently while retaining
their logical order and stable action identities; effectful or potentially
conflicting actions remain ordered unless the environment explicitly provides
stronger semantics.

Define patches against an explicit initial state with stable node identities and path conventions. Patches need enough typed data to reconstruct partial values, missing fields, pending nodes, locals, marks and replacement/provenance links. Reuse the language's value model, but do not assume the current JSON dump already contains every necessary execution detail. Start with sequential fixtures and make reconstruction demonstrate what fields are necessary before freezing the schema.

### Values across hosts

Portable values have their natlang type and canonical data representation. Missing/unset is distinct from Null. Large values can be external records with content identity and an explicit availability flag. A host object can have a diagnostic descriptor: host type, run-local identity and optional preview. A descriptor is neither a serialized native object nor an authority token.

The trace reader handles ordinary natlang values uniformly. For an opaque value it shows the descriptor, whether a decoder/snapshot is available, and what cannot be reconstructed. Optional host renderers may improve presentation without being necessary to open the trace. Cross-platform developer experience means consistent navigation and honest information, not identical native engines or magical portability of live host objects.

### Replay boundaries

1. **Display:** navigate recorded calls, actions, reductions and diagnostics.
2. **State reconstruction:** apply captured reductions to the known initial natlang state.
3. **Decision replay:** feed recorded model actions and recorded external results through the validator/runtime.
4. **Live re-execution:** run source again in an environment satisfying its dependencies.

These are different operations. Shared-host execution may only support the first two unless the host captures effects/results and relevant native state. Purely recording eval source is insufficient to reproduce its mutations. Never run eval code while merely opening a trace. A debugger cannot undo external effects by moving its cursor backwards.

### Relationship to existing training IR

Keep semantic program IR and teacher trajectory IR. The reduction trace is evidence of execution, not a replacement for desired behavior or teacher decisions. Link records by run/node/action identity. Whole-program admission can use trace events to check bindings, intermediate states and effects, then generate the existing model-neutral training view. Avoid a second incompatible teacher transcript format.

## 8. Development order and concrete acceptance tests

1. **Extract engine boundary:** preserve current QuickJS behavior and validation; remove JS-specific representation/errors from the core where appropriate.
2. **Add explicit invocation configuration:** seed policy, model request settings and engine/environment bindings; test derivation vectors and independent concurrent calls.
3. **Version engine selection:** add required `engine` to the proposed tool surface and loader binding rules. Update native decoding, prompts, reference policies, generators and replay projections together. Preserve old traces; do not silently interpret missing engines in the new version.
4. **Exercise two contrasting adapters:** the existing isolated executor and a small explicitly shared-host executor. Demonstrate persistent native state visibility and that natlang state changes still pass validation. Use the same language initially to isolate the environment-policy difference; a SQL adapter can follow as a true second language.
5. **Implement trace data:** manifest, logical identity and lifecycle/action/eval/reduction records with an in-memory/JSONL sink. Reconstruct typed state for simple calls, failed writes, Map, Fold and quiescence; verify no live effects during replay. Keep diagnostic traces useful when replay is impossible.
6. **Complete stream semantics:** host-driven Fold waiting/closure/failure first, then bounded streaming Map and its history/output contract. Include out-of-order completion, backpressure, no-input-yet versus closed, and a failing child.
7. **Evaluate type escape hatches only if needed:** demonstrate a programme requiring native values across the natlang boundary before adding syntax. Write conversion fixtures for all actual engines regardless.

Cross-host acceptance includes: the same portable trace fixture opens in different readers; deterministic seed test vectors agree; a shared environment trace clearly reports uncaptured mutations; ordinary typed values validate identically across engines; unsupported host types/engines fail explicitly; and a single-threaded host needs no worker pool or persistence service.

This is a staged feature plan. Engine selection and stream behavior require deliberate surface/spec versions; they should not be smuggled into a supposedly behavior-preserving refactor. The first implementation batch can extract the seam and add invocation configuration/trace foundations without simultaneously delivering every evaluator, type extension and replay guarantee.
