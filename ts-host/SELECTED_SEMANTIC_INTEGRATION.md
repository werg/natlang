# Selected semantic execution in Natlang

`installSelectedSemanticEnvironment` installs the asynchronous executor at both
the task evaluator and callable-folder module loader. The implementation uses
the `semantic-trace/4` / `semantic-execution/4` contract supplied by the pinned
tracer. The JavaScript source is published before an explicit invocation. Model
selections drive the guest; declared native bindings run only after a committed
request, and their actual results become observations.

```ts
import { installSelectedSemanticEnvironment, EvalFailure } from '@natlang/node';
import { createNatlangSelectedEnvironment } from 'tracing-js-self/natlang-selected';

installSelectedSemanticEnvironment(mode => createNatlangSelectedEnvironment({
  mode,
  EvalFailure,
  select: selectedGroupAdapter,
  maxSelections: 4096,
}));
```

The embedding supplies `selectedGroupAdapter` from the serving integration. It
consumes the visible selected prefix and returns one committed semantic record
or the corresponding selected model group. Tests use explicitly controlled
selections through this same interface. They establish engineering behavior;
they do not establish that a trained model can produce those selections.

## Entry, module and callback boundaries

Use `executeAsync` and `moduleInstanceAsync` after this installation. Synchronous
`execute` and `evaluateModule` reject. Caller migration includes the actual
compiler-generated inline and guard helpers: their immediate calls are awaited,
and lexically resolved callers become asynchronous. Ordinary independent
Promise-producing calls retain the concurrency expressed in their source.

Each module has its own retained selected realm. Its initializer runs once,
with exact compiled and original source metadata. Static sibling dependencies
initialize before their importer. Cycles share the already allocated partial
namespace; statically declared export names are lazy cells before dependency
initialization, so cyclic imports remain live after initialization completes.
Concurrent roots serialize graph initialization and cannot wait on one another
forever. Exports are live namespace facades. Exported functions
retain their owned closure identity and capture cells across calls. An import
from another realm uses a declared native binding to that selected export.

Generated `bindLocal` accessors stay selected guest closures. The inline bridge
awaits their getters at the native boundary, presents synchronous snapshot cells
to the existing Natlang kernel, and awaits selected setters before returning a
successful child result. Calls sharing a capture bridge serialize refresh and
commit. All changed captures are checked for conflicting intervening writes
before the first setter runs. A failed child commits no capture writes. Native
effects performed by that child are still real effects. Typed native resources
retain their identity inside capture snapshots. Mutating a resource changes its
native state; replacing the captured handle changes the selected capture cell.

Native callbacks use an explicit guest wrapper in the published adapter source.
Native resource arguments arrive as typed envelopes; the selected wrapper
unwraps them and calls the original guest closure. The host never executes that
closure as JavaScript. A synchronous native operation may retain a callback for
later delivery. It cannot synchronously wait for a selected callback result.
The asynchronous `natlang.call` boundary supports awaited callback returns and
guest callback throws. Calls of a retained callback remain attached to its
original realm and source after its registering invocation has completed.

## Registered native surface

These operations share the tracer broker, graph codecs, source-frame checks,
dynamic invocation ownership, and observed-result channel.

| Operation | Result class | Arguments | Authority |
| --- | --- | --- | --- |
| `natlang.call` | Promise | callable handle, receiver, argument array | A declared asynchronous binding in the current program |
| `natlang.callSync` | Immediate | callable handle, receiver, argument array | A declared synchronous binding; Promise returns reject |
| `natlang.readCell` | Immediate | cell handle | A registered getter, invoked only after its request commits |
| `natlang.writeCell` | Immediate | cell handle, value | A registered writable cell |
| `platform.console.{log,info,warn,error}` | Immediate | one JSON value | Captured invocation logs and ordinary broker observations |

Natlang calls use `owned-graph/1`, preserving aliases, cycles, sparse arrays,
`undefined`, bigint and special numeric values. JSON-only APIs keep their
separate JSON codec. Nested owned callback markers are supported only by the
declared callable operations. Markers and handle strings do not grant authority
in another program or a closed environment.

The platform registers exact surfaces for `TranscriptView`, `FileHandle`,
`FolderHandle`, `Folder`, `FolderSnapshot`, `FolderProposal`, and
`FolderTransaction`. The code in `native/selected-surface.ts` records authority
by object identity; no class or capability is inferred from guest data keys.
File and folder operations expose their text/JSON/byte reads and writes,
directory lookup, scoped handles, snapshots, proposals and transactions through
their declared methods. Returned transaction resources release their writer
lock when the owning evaluator closes. Resources supplied by the embedding are
not disposed merely because an evaluator borrowed them.

Native `Uint8Array` values use an explicit byte-resource surface: `at`, `slice`,
`subarray`, `set`, `fill`, `length` and `byteLength`. Bytes remain native resource
state, rather than inflating a model prefix into one cell per byte. Pass the
resource back to `writeBytes` to preserve its exact bytes and identity. Raw
symbol iterators, direct typed-array indexing and arbitrary class instances are
outside this registered surface and require a corresponding explicit adapter.

Recorder-owned service proxies have a declared surface that preserves the
service receiver and requested/completed/failed effect recording. Imported
`natlang:services` methods resolve the current task on every call, so retained
modules do not capture another task's service object. Supply explicit named
service declarations and return classes. A function implemented without
`async` that returns a Promise must declare its asynchronous call kind.
An embedding can use tracer `selectedCallable` / `selectedCell` or the exact
`callKinds` binding-path option for custom APIs.

Arbitrary proxies, undeclared packages, nonliteral `require`, unregistered native
objects, implicit synchronous selected reentry, and foreign retained identities
are rejected. A refusal never switches to `TypeScriptEnvironment`. These are
the executable authority boundaries, not promises of general V8 compatibility.

## Lifecycle, cancellation and recovery

`fork()` creates a fresh selected environment with new handles and realm state.
`close()` is idempotent, aborts active selections, revokes native handles,
detaches callback scheduling and releases owned resource cleanup hooks. A late
callback cannot revive the environment. `EvalRequest.signal` and `timeoutMs`
reach the selected driver and its owned dependency cancellation path.

The installer wraps failures in Natlang's actual shared `EvalFailure` class.
Its `events` include observed invocation, effect-result and cancellation
records; `debug.logs` retains console output produced before failure. This
lets `NativeRuntime` record effects even when the scope transaction fails.

Changing `setModuleRealm` advances the module authority generation. Old handed-out
namespaces and callable wrappers reject instead of executing in the replacement
authority. Isolated task module realms close with their task. A module whose
initializer failed remains failed in its cache: an automatic retry could repeat
an uncertain native write. Repair the cause, reconcile external effects from
the broker journal, then explicitly start a new task/realm or a new source
revision. Do not treat evaluator failure as rollback of a file, service, or
resource mutation.

Durable selected execution checkpoints and dispatch reconciliation belong to the
shared tracer harness. Natlang does not invent a second effect ledger. Restore
with the exact compatible source, protocol, selected state and broker journal;
rebind native resources explicitly. A missing external resource is a recovery
failure, not evidence that an uncertain operation was never performed.

## Outside events: Fold and Map

`SelectedEventQueue`, `foldSelectedEvents`, and `mapSelectedEvents` are exported
from the runtime surface. Call a selected callable from the Fold step or Map
worker; the host waits for event arrival without asking the model to poll.

```ts
const input = new SelectedEventQueue('orders:epoch-3', 32, recordReceipt, 65536);
const result = foldSelectedEvents(input, initial, async (state, value, event) =>
  await selectedStep(state, value, event.id));
await input.push('delivery-17', payload);
input.close();
const finalState = await result;
```

Fold completes a step before consuming the next item. Map bounds independent
work and yields results in input order with the original source, event ID and
position. It can yield an already completed item while an open source is below
its concurrency window. Closing/cancelling a Map consumer cancels its source
and signals outstanding workers, including while `next()` is waiting for an
input or worker result. This does not undo an already dispatched native effect.

`push` applies producer backpressure. `post` reports explicit overflow for a
producer that cannot pause. Queue capacity, pending producer capacity and
delivery-history capacity are bounded. Exact repeated deliveries are duplicate
receipts; reusing an ID for different data rejects. When history capacity is
reached, archive a checkpoint and rotate the declared source epoch. No ledger
entry is silently forgotten.

Event payloads are finite JSON data, with descriptor inspection that rejects
getters, hidden/symbol properties, sparse arrays and cycles. This event profile
is deliberately distinct from the richer native-call graph codec. Checkpoints
are taken at settled admission boundaries and preserve the cursor, queued
payloads and duplicate ledger. Restore validates all payload digests and
positions. Normal closure drains accepted events; source failure drains them
then rejects; cancellation records discarded buffered items and rejects waits.
Receipt observers run after the accepted queue transition. If an observer
throws, the source fails and settles all blocked waits; already accepted events
remain deliverable and cannot disappear behind their duplicate receipt.

## Verification

`selected-semantic-real-driver.test.mjs` uses the actual compiler, NativeSession,
module loader, shared `EvalFailure`, and tracer selected driver. Its fixtures
cover generated `bindLocal` plus inline child execution and selected capture
writes, live sibling and cyclic imports, retained callable modules, FileHandle ordered
read/write/read, task-local imported service receivers and effect recording,
and console observations on failure. The tracer's dedicated
adapter tests additionally cover cross-realm retained callbacks, typed resource
callback inputs, late delivery after close, source-bound native actions and
captured module state. `selected-captures`, `selected-semantic-module`, and
`selected-events` tests cover conflicts, failed children, cache cycles, authority
cutover, isolation, backpressure, reversed completion and checkpoint tampering.

The root development dependency pins the tracer to an immutable GitHub source
tarball, with the package lock recording its integrity. The `owned-tracing-smoke`
workflow installs that package with `npm ci` and runs these tests through its
public selected-execution and training exports. No sibling checkout is needed.
The final roadmap integration pins tracer revision
[`77480bf3b85b34620dd07a3b12f223e45b827d9e`](https://github.com/werg/tracing-js-self/commit/77480bf3b85b34620dd07a3b12f223e45b827d9e).
Its public selected prefixes include the checked source and retained-state
context used by both the teacher and executing model. Native orchestration,
child completion, callback ownership and context-eviction verification use that
same versioned runtime; installing the development dependency does not install
or select a model checkpoint.

The older `OWNED_TRACING_INTEGRATION.md` documents the separate bounded
native evaluator reference; it is not the selected production path or a fallback.
Protocol migration, corpus admission and learned evaluation use the shared
cross-repository roadmap and its pinned v4 artifacts. No training or optimizer
run is needed to execute these engineering fixtures.

### Validation of the pinned package

The final 2026-10-25 validation used a clean `npm ci` installation of that
revision, then built Natlang against the installed package. The scoped runtime
suite passed all 208 tests without skips. Node compilation, browser export
types, dependency layering, the current source-review manifest, all 12 quick
Python ratchets and the quick TypeScript check also passed.

After `npm ci` and `npm run build:node`, reproduce the runtime gate from the
repository root with:

```sh
node --test \
  ts-host/test/selected-semantic-module.test.mjs \
  ts-host/test/external-module.test.mjs \
  ts-host/test/callable-types.test.mjs \
  ts-host/test/iteration-measure.test.mjs \
  ts-host/test/selected-semantic-real-driver.test.mjs \
  ts-host/test/selected-captures.test.mjs \
  ts-host/test/event-loop.test.mjs \
  ts-host/test/scoped-fs.test.mjs \
  ts-host/test/native-values.test.mjs \
  ts-host/test/scope-compiler.test.mjs \
  ts-host/test/native-files.test.mjs \
  ts-host/test/runtime.test.mjs \
  ts-host/test/selected-events.test.mjs \
  ts-host/test/owned-tracing-wiring.test.mjs \
  ts-host/test/interpreter.test.mjs
```

Run `scripts/check-main.sh --quick` for the source-review, Python and TypeScript
checks, and the host workspace's `check:browser-types` and `check:layering`
scripts for its export and dependency boundaries. The selected integration
workflow uses the smaller five-file production integration gate above. These
checks perform no optimizer updates; controlled selections retain their fixture
origin and are not evidence of learned model success.

## One canonical execution witness

Natlang's selected environment consumes the source-grounded `semantic-execution/4` / `semantic-trace/4` selected-harness trace from its **pinned tracer commit**, never the old group-shaped diagnostic producer. The model does not choose between explicit, compact or outcome-only alternatives. Its returned work must satisfy the same selected-state checks and external effect authorization. Private interpreter journals and independent reference outcomes are not model inputs. A schema change must update the package pin and lockfile together and pass the actual selected async module/callback tests.
