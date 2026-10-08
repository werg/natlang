# Runtime/compiler/tool idiom audit — 2026-10-08

Scope: review of agent-visible TypeScript, Neuralese/capture, result,
collection/file-reducer, and tool boundaries in `ts-host`. This continues the
V88–V97 review recorded in
`runs/luna-idiomatic-boundary-audit-20261007-v1/audit-packet-20261008-final.md`.
I read the machine coordination inbox and current handover first. The original
initial review made no source changes. Following owner approval, the `for...in`
finding below was implemented narrowly in policy and guidance with
compiler/runtime regressions; no jobs, dependencies, or shared build output
were changed.

## Findings

### 1. Existing narrow fixes close the observed V88–V97 issues

The reported V88/V90/V94/V96/V97 friction now has narrow, type-directed implementations in shared
source. Current commit history and the final packet document saved inline
callable `.with(record)` support (`4b2e414b`, `dc8031db`), finite typed capture
records (`13f22d41`), eval-local alias reuse (`b50228ee`), and typed result
normalization. The current source also contains typed `String`/`concat`/`join`
readout lowering, configured typed final-text writes, and current-invocation
visible preview resolution. These have explicit behavior and provenance; they
should not be proposed again as open work.

One closely related V95 issue was that `await nl\`...\`` was rejected even when
the model meant to save the callable. The source now distinguishes an awaited
tag in a variable initializer with no type or a callable context from a
non-callable context: `ts-host/src/scope-compiler.ts:240-252`. The observed
attempt to assign the callable to a result record still correctly fails. This
was a safe targeted correction; it is already implemented and is not a new
proposal.

### 2. Native `for...in` can be admitted without changing its semantics

**Priority: medium; implemented as a narrow shared policy change.** Before this
change, the policy rejected every `for...in` at `ts-host/src/compiler/policy.ts`
(then lines 215–216) with “iterate `Object.keys(value)` or
`Object.entries(value)`.” The intuitive expression is
`for (const key in record) { ... }`. This is distinct from `for...of`: native
`for...in` has a defined result, enumerable string keys on the object and its
prototype chain, with native enumeration order and behavior when properties
change during iteration. Keys/values/pairs ambiguity belongs to plain-object
`for...of`.

The constrained-source policy permits `for...of` over arrays, Maps, and Sets;
`ts-host/src/compiler/lower.ts:228-235` wraps only those sources. The wrapper
in `ts-host/src/runtime/lowered.ts:323-349` does not cap element count for an
array, Map, or Set; it rejects unsupported iterables and detects selected
collection growth. `for...in` is rejected before lowering, and the lowering has
no `ForInStatement` rewrite. Thus neither the current transformation nor the
documented finite-collection policy requires refusing native property
enumeration.

**Implemented behavior:** `for...in` now passes the constrained-source policy
unchanged; the compiler does not rewrite it to `Object.keys` or pre-materialize
the keys. This
preserves inherited keys and native mutation/order behavior. No timeout or
iteration budget was added; existing evaluator resource accounting applies as
it does to accepted collection loops.

**Risks:** an extremely large object/prototype chain can consume time and
memory, and Proxy traps can execute during enumeration. Eval timeout is not a
preflight key-count cap. Task inputs are frozen portable values, while local
code can create mutable objects and custom prototype chains. These are resource
concerns to observe, not reasons to substitute own-key-only behavior. I found
no historical rejected generation for `for...in`; this was a prospective
ergonomics improvement grounded in the blanket policy.

### 3. Cross-eval JavaScript function persistence remains a real boundary

**Priority: retain the boundary; consider only a clearer diagnostic if new
failures show it is still confusing.** V97 case 03, worker 2, trace sequence
27–29, declares `revise` in one `eval` and calls it in a later `eval`; the later
call reports `ReferenceError: revise is not defined`. The compiler marks
function-valued locals transient at
`ts-host/src/scope-compiler.ts:356-372` and excludes transient declarations from
the persisted names at `:673-680`. Failed evals separately discard new bindings
as described by `ts-host/src/native/runtime.ts:976-978`. The attempted
expression is ordinary JavaScript, but retaining a
closure would also retain its lexical locals, staged result state, invocation
facades, inline source plan, trace origin, and possibly authority-bearing
handles after the eval transaction ends.

**Prospective narrower design:** persist source for a top-level function or
arrow helper only when static free-variable analysis proves that it captures no
eval-local, input, service, mutable local, or opaque binding. Recheck and
recompile that source against each later eval's current aliases, imports,
callables, and capabilities instead of retaining its prior JavaScript closure.
For example, a helper such as
`const normalize = (x: Record<string, string>) => Object.keys(x).sort()` could
be usable in a later eval while every runtime value remains explicit in its
arguments.

**Priority: low/medium design candidate, no patch recommendation yet.** This
requires a source-backed helper registry and checks for implicit globals,
nested functions, imports, inline `nl` plans, call-site/trace provenance, and
mutable arguments. A helper with no lexical captures can still cause effects
through an explicitly passed `FileHandle`; those effects remain authorized by
the argument and should not be described as pure. Do not retain the old closure
object, captures, or auto-recreate a helper after a failed eval. Until such a
model exists, define and call helpers within one eval or save a named `.nl`
function in the program/context.

### 4. Plain-object `for...of` has no unique intuitive mapping

**Priority: no lowering proposed.** `ts-host/src/compiler/policy.ts:241-247`
restricts `for...of` to arrays, strings, maps, sets, and tuples. A plain object
has no built-in `Symbol.iterator`; treating it as keys, values, or key/value
pairs would invent behavior. Use an explicit `Object.keys(record)`,
`Object.values(record)`, or `Object.entries(record)`. I found no new actual
generation rejection that demonstrated an equivalent traversal was blocked.

### 5. File/reducer command refusals preserve authority and call shape

**Priority: retain strict behavior.** The folder shell requires a reducer name
and directory for `natlang apply` (`ts-host/src/native/folder-shell.ts:231-235`)
and a single named function plus explicit input mode for `natlang call`
(`:222-229`). Runtime checks require a callable function versus a
`directory-reducer`, enforce one argument for `--lines`, and require JSON
object-keyed arguments for multi-parameter calls
(`ts-host/src/native/runtime.ts:1197-1218`). These distinctions carry dispatch
and folder-write semantics. Guessing a reducer, mapping an array to multiple
parameters, or selecting a directory would be semantic/authority changes.

The recent handover reports no legitimate rejected file/path/reducer expression
in V93–V97. V17's `input.source.readText()` was a wrong nested path; missing
names such as `extracted`/`contract` were absent from scope. Those are not
coercion opportunities. Keep suggesting explicit handles and `folder.apply`.

### 6. A denied child write used to poison later clean evals in the same call

**Priority: high; fixed in this review.** V19 Luna case 3 provides direct
evidence. Its preserved partial is
`runs/neuralese-semantic-iterate-reducers-v19-20261008-v1/generation-review-v1/luna/luna-campaign-v1/slot-04/jobs/000003-64f57198009df9e9.partial.json`;
the corresponding evidence stream is the adjacent
`000003-64f57198009df9e9.partial.json.evidence-2e150bcc-87c8-46b7-8351-195f35250e1c.jsonl`.
It records 192 saved turns and ends with the 384-request whole-case budget
error. The evidence stream contains 58 `scope_failure` events repeating the
same denial, from 01:42:06Z through 01:54:19Z: writing `decision.json` was
outside the supplied `FileHandle` scope. At event 278 the child still returned
the supported eligible fields and a typed prose answer; subsequent clean
console/read/return attempts also failed at result completion with that same
write-scope error. This was not missing evidence or a reason to widen child
authority.

The scope decision itself was correct: a note child given one pass `FileHandle`
must not write the parent's `decision.json`. The incidental failure was late
enforcement. `Folder.beginFileTransaction()` created a one-file private folder,
but `FolderTransaction.rootedChanges()` detected extra paths only when the
whole child transaction committed. An out-of-scope write therefore mutated
the still-open overlay. Every later eval retried the same failing commit, even
after it returned a valid typed result.

**Implemented fix:** `ts-host/src/native/scoped-fs.ts:304-314,418-423,474-495,529-549`
places a one-path write fence on the private folder and checks it before
`writeBytes`, `remove`, `move`, and install mutate the overlay. Forks and
nested transactions retain the fence; the existing final transaction scope
check remains as defense in depth. The denied path and `FolderScopeError`
remain visible to the eval failure capture. The child can then read the
supplied file and return a typed result in a later eval of the same invocation.
No additional authority, timeout, or permanent per-call taint was added.

Regressions are in `ts-host/test/scoped-fs.test.mjs:136-161` and
`ts-host/test/interpreter.test.mjs:774-799`: they check rejected writes,
remove, and both move endpoints leave no out-of-scope overlay changes; a later
child eval still logs/reads its supplied file, returns a boolean, and preserves
the parent's existing result file. The scope failure remains in the trace.
The current builder also makes the note-child boundary explicit at
`ts-host/scripts/inline-curriculum/semantic-iterate-worlds-v15-soft-guided-builder.mjs:121-155`:
the child returns `Neuralese<string>` to its caller, while the parent owns the
final file write. Its captures pass decision context and the current pass
constraints, not the parent's artifact-writing instruction.

### 7. Spread arrays in `String.concat` can use the existing async readout

**Priority: medium; implemented as a narrow conversion convenience.** The
checker previously rejected `''.concat(...values)` when `values` was a statically
typed array or tuple containing a `Neuralese` element. Its diagnostic said the
array could not be spread into this “synchronous string conversion” at
`ts-host/src/compiler/neuralese.ts` (then lines 155–163), and
`ts-host/test/neuralese.test.mjs` asserted `neuralese-opaque-access` for that
expression. This is an actual current compiler refusal, although I did not find
a historical generation trace that attempted it.

The compiler already has an async lowering for direct `String.concat` values:
`ts-host/src/scope-compiler.ts:516-523` evaluates the receiver and method,
materializes the argument list (including spread expansion), then calls
`concatNeuralese`; `ts-host/src/runtime/lowered.ts:43-49` awaits each soft
read and applies JavaScript string coercion in order. The spread array is fully
expanded before any async read, matching JavaScript call evaluation order. The
checker now marks a spread as needing that readout only when its static array or
tuple element type includes a Neuralese arm, including a union of soft and crisp
element types. It does not accept an arbitrary
iterable, change native spread behavior, or grant a new capability.

Regressions in `ts-host/test/neuralese.test.mjs` check the diagnostics,
generated spread lowering, receiver/spread/later-argument/read order, ordinary
string coercion, and repeated typed reads. The focused Neuralese suite passed
19/19 in an isolated build. Other opaque uses and unsupported dynamic types
remain errors.

### 8. Array unions need the same typed join readout

**Priority: medium; confirmed and fixed narrowly.** A declared value of type
`Neuralese<string>[] | string[]` can call the ordinary built-in
`values.join('|')`. Before this fix, `arrayJoinKind()` checked only whether the
whole receiver type was an array or tuple
(`ts-host/src/compiler/neuralese.ts:95-116`); the valid union call had no
diagnostic but also no readout lowering. I reproduced the generated module
calling native `.join` directly and producing `[object Object]` for a soft
reference. No historical generation trace using this exact union shape was
found; this is a prospective type-directed correction based on a reproduced
runtime mismatch.

The checker now distributes over a union only if every alternative is an
array or tuple, then requests the existing async `joinNeuralese` readout when
any element alternative has a Neuralese arm. The runtime helper already preserves
array index order, holes, nullish elements, and typed read order. Tests cover
mutable and readonly array unions, readonly tuple unions, no-soft array unions,
the corresponding concat spread, and a generated module returning readable
text for a soft element (`ts-host/test/neuralese.test.mjs:163-179,235-244`).
Other union members such as `undefined` or a non-array are left to normal type
checking and are not lowered as arrays.

### 9. Text coercion of Neuralese unions should read only actual references

**Priority: medium; fixed with local, conditional lowering.** Before this change,
`neuraleseParts()` returned no result for a union, so text coercion did not
request a readout for `Neuralese<string> | Neuralese<number>`. TypeScript
accepted `String(value)` and template interpolation, but isolated generated code
kept the reference opaque and produced `[object Object]`. A mixed union such as
`Neuralese<string> | string` had the same issue. No reviewed Luna generation
trace attempted these exact union forms; this is a prospective fix based on a
reproduced generated-runtime mismatch, not a claimed historical failure.

The local `hasSoftAlternative()` check in
`ts-host/src/compiler/neuralese.ts` now detects a Neuralese arm only at string
coercion sites: `String(value)`, template interpolation, `toString()`, string
`+`, plus the existing `concat` and array `join` paths. It deliberately leaves
global `neuraleseParts()` unchanged because that function also drives type text,
callable inference, and target classification. For the direct readouts the
compiler calls `readNeuraleseIfReference()`; it asynchronously reads actual
typed references and returns crisp union arms untouched, so native coercion and
evaluation order still apply. The established `concatNeuralese` and
`joinNeuralese` helpers already use the same runtime brand check for their
elements.

Generated module regressions exercise all-soft and mixed unions with actual
references and ordinary strings. Eval analysis and lowering regressions verify
the conditional helper is selected for mixed values. The implementation does
not change JSON serialization or reinterpret structural objects as typed
values.

### 10. Eval exposes a read-only tool facade; other tool actions stay direct

The prompt says to invoke native tools directly but explicitly makes
`read_code(name)` available in eval when unshadowed (`ts-host/src/native/prompt.ts:6`).
`ts-host/src/native/runtime.ts:1650-1654,1702-1713,1758` binds that
visibility-checked read-only lookup and records trace events. V17v2 had an
actual attempted `read_code('decide')` inside eval that formerly received a
tool-routing correction; this expression is supported in current source.

**Priority: retain the scoped boundary; do not generalize by auto-routing.**
Other native tools remain actions, not eval functions. `runtime.ts:969-973`
adds a direct-tool correction for undefined native tool names, and
`:1461-1465` gives a precise namespace error when a native tool is requested
as a program function. Those tools have distinct arguments, completion paths,
and effect ledgers. A future explicitly named, visibility-checked read-only
facade could be considered if an actual rejected expression justifies it; no
other recent trace reviewed here supports one.

## Error/instruction candidates explicitly not reopened

- Validated typed JSON final values, exact `Neuralese<string>` writes, nested
  typed leaves, direct typed string operations, and saved callable `.with` were
  addressed by the 2026-10-07/08 work above.
- `nl.with` still requires a finite typed capture record (current diagnostic at
  `ts-host/src/compiler/inline.ts:419-437`). Unknown/dynamic keys, spreads, and
  property aliases with no checked finite shape cannot be snapshotted without
  inventing a capture set.
- Exact parameter arity and overlapping writable handles are deliberate
  runtime checks, not ergonomics defects; see the prior final audit for source
  locations and rationale.
- `while`, generator, `yield`, and dynamic-code refusals (`policy.ts:213-240`)
  protect bounded execution or source visibility. No universal transparent
  equivalent exists. The `for...in` exception above was prospective at audit
  time and is now implemented with native semantics.

## Recommendation

Native `for...in` is accepted unchanged, with compiler/runtime regressions for
ordinary property order and inherited enumerable keys versus `Object.keys`.
The focused compiler and interpreter tests passed 99/99 in an isolated build
copy. The denied-child-write regressions passed 94/94 focused file-system and
interpreter tests in an isolated build, and the Neuralese suite passed 19/19
after the concat-spread and array-union readout changes. The updated TypeScript
source compiled to a temporary isolated output directory. No dependencies or
shared `dist` were changed.
Cross-eval capture-free helper persistence remains a useful but larger
source-backed design candidate, grounded in the cited V97 error and requiring
a separate lifetime/provenance design.
