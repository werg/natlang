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
Cross-eval source-backed helper persistence is now implemented for top-level
function declarations and arrow/function-valued declarations whose free value
names resolve in the current call and whose referenced type aliases persist.
They are stored as authenticated source text, recompiled against each eval's
current inputs, locals, services, and folder handle, and receive fresh inline
`nl` analysis and execution origins. The runtime stores no function closures or
host handles. Nested functions in the helper are recreated from its source;
references to eval-local function bindings remain eval-local with an explicit
message. Failed evals do not replace the registry; successful same-name
declarations do. Source spans in helper plans point to the recorded helper
source hash and original declaration span, without claiming the helper was
written in the current eval. See
`ts-host/src/scope-compiler.ts`, `ts-host/src/native/runtime.ts`, and the
focused regressions in `ts-host/test/persistent-scope-helpers.test.mjs`.

The Bunny V20 index 13 trace provides direct evidence for the ergonomics
problem and distinguishes it from a later reasoning failure. In
`runs/neuralese-semantic-iterate-reducers-v20-20261008-v8/generation-review-v1/bunny/bunny-campaign-v1-native/attempt-02/case-000013/jobs/000013-000431ad1b41d8db.trace.jsonl`, root eval action seq 17 declares `Progress` and a top-level `const revise = async (...) => ...`; seq 28, at 102.302 seconds, calls `revise(firstCompleted)` and fails with `ReferenceError: revise is not defined`. The failure's own scope guide lists `task`, `decisionContext`, `seed`, `initialNotes`, and `firstCompleted`, but not `revise`. At seq 38 the model re-emits the full helper and successfully gets pass 2, showing that the source itself was usable and the omitted helper cost an avoidable model turn. Its free values (`task`, `folder`, `decisionContext`, `nl`) and type alias (`Progress`) are now rebound/revalidated at the current eval rather than retained as a closure.

This helper loss does not explain the final blocked result. The seq 50 and seq 63 pass actions receive and return accumulated notes containing the current-pass facts. The final interpreter child at seq 78 then states that ACT-R's risk score is missing. The campaign's source/evidence files and the earlier pass outputs contain ACT-R's measured score 89, ACT-S 94, and ACT-T 96; the pass-3 notes nevertheless repeat that ACT-R's score is absent. That is a child synthesis/evidence-propagation mistake after the helper had already been manually restored, not a typed-conversion, capture, or helper-rebinding error. Helper persistence should remove the concrete `revise` ReferenceError and repeated source declaration, but should not be described as fixing the later missing-score claim.

The helper implementation was checked with an isolated TypeScript output tree and
seven focused runtime regressions covering the `Progress` alias, fresh inline
`nl` compilation with source provenance, successful replacement, failed-eval
rollback, eval-local callable captures including parameter defaults, nested
lexical shadowing, and the current child-folder authority fence. The free-name
check uses TypeScript's symbol resolution rather than a hand-maintained global
value allowlist.

## Follow-up: Luna and Bunny runtime surfaces — 2026-10-08

### 11. Service openings must describe data as data

**Priority: high; fixed in `0fd2d175`.** Luna V20 index 6 repeatedly treated
`neuralese.bodies` and `neuralese.textReadSource` as callable methods. The
actual `StandardLibrary` fields in `ts-host/src/neuralese/combinators.ts`
are metadata values; callable operations live on the separately declared
`natlang:neuralese` facade. `ts-host/src/native/agent.ts` previously rendered
every enumerable service property as `Function`, so the opening itself taught
the wrong API shape. The fix renders callable values as callable and metadata
as readonly primitive/record types, never evaluates accessor properties, and
does not print field contents. It does not invent metadata methods or expose
private values. The V20 trace shows the repeated guesses and exhaustion; this
finding is tied to that actual surface mismatch.

### 12. Same-named inputs naturally shadow explicit captures

**Priority: medium; fixed in `f707f973`.** Luna V20 index 9 produced repeated
`nl-capture-parameter-collision` diagnostics for an ordinary
`nl.with({ input, ... })` child whose parameter was also named `input`. The
model's intended child input should use ordinary lexical shadowing. The
compiler now marks that declared slot as shadowed, evaluates its initializer
once in the original object-literal order, and leaves the input parameter as
the sole child-visible binding. A shadowed live wrapper is constructed as
written but its value is not dereferenced or installed in the child. A saved
`.with` still validates the complete declared capture shape, while its
shadowed values never enter child capture metadata or runtime snapshots. The
scope index keeps the authored field for source validation but creates no
binding for it. This change preserves authority and does not expose the old
handle. Isolated TypeScript compilation and seven focused compiler/runtime
regressions passed. A repository search found no remaining current
`nl-capture-parameter-collision` diagnostic or rename instruction; the older
dated handover entry describing the former help text is historical and is
retained as history.

### 13. The remaining Bunny result omission was synthesis, not conversion

The Bunny V20 index 13 helper error and later ACT-R omission are distinct.
The `revise` helper was declared in one eval and called in another, with a
recorded `ReferenceError`; source-backed, capture-checked helper persistence
already addresses that concrete friction (section 3 above). Later, the
preserved ACT-R score 89 was read and present in earlier pass outputs, but a
child note omitted it and a later child repeated the omission. The task files
remain intact. This is a child evidence-synthesis failure, not a text readout,
typed conversion, or missing source problem. No further coercion change is
supported by that trace.

### 14. Current native guidance review: await, optional arguments, and results

The scoped eval wrapper is async, and normal TypeScript async/await behavior
applies within it. The `neuralese-readout-sync` diagnostic is limited to a
Neuralese-to-text conversion inside a non-async function; its guidance to make
that function async or move conversion into async code follows from the
existing awaited readout implementation. Automatically making arbitrary
callbacks async would change their return type and call contract. The
existing `await nl\`...\`` normalization handles the separate common case of
awaiting the callable instead of invoking it; it does not relax ordinary
non-async callback semantics. I found no current await diagnostic with a
semantics-preserving implicit correction.

Optional parameters are represented in the opening as `T | undefined`, and
runtime invocation leaves omitted optional arguments absent; supplied values
are checked against the parameter type in `ts-host/src/runtime/kernel.ts`.
Exact parameter names, declared arity, and JSON-object dispatch remain
intentional because guessing omitted values or mapping positional arrays would
change call semantics. The paths reviewed do not support a new optional-arg
rewrite.

Typed result handling accepts an exact declared value or the configured
Neuralese string writer/readout. Runtime record coercion removes omitted
optional fields, rejects undeclared fields, and checks each declared field in
`ts-host/src/native/values.ts`; those shape errors preserve the caller's
result contract. The ordinary-string/Neuralese-string distinction is explicit
in the opening and covered by typed text preflight. The local conditional
readout fix above addresses all-soft and mixed Neuralese union coercion at
String/template/concat/join sites while preserving crisp alternatives. I found
no trace-backed case that justifies coercing arbitrary objects, arrays, or
malformed JSON into declared results.

**Disposition:** the source-backed helper loss, false callable service
descriptions, and capture-name collision were concrete ergonomics defects and
are fixed. The ACT-R omission was a child synthesis error. Other recent Luna
exhaustions reviewed here ended at the 384-request whole-case budget; they do
not establish a new language boundary failure. No additional evidenced
coercion or authority relaxation is proposed.

### 15. JSON serialization misses the existing conditional readout for unions

**Priority: low/medium; prospective reproduced compiler mismatch, no
historical trace found.** `checkNeuralese()` already requests readout of the
first argument to the standard `JSON.stringify` when that expression is a
single `Neuralese<T>` value. For a union, `neuraleseParts()` returns no value,
and the JSON branch currently tests only `soft(value)`. Reproduction through
`analyzeEvalSnippet('return JSON.stringify(value);', scope)` with
`value: Neuralese<string> | Neuralese<number>` produces no diagnostic and no
readout. The generated program directly calls `JSON.stringify(value)`, while
the adjacent `String(value)` expression correctly generates
`readNeuraleseIfReference(value)`. The former therefore serializes the opaque
transport reference (including its ID/metadata) instead of its typed payload.

A narrow candidate is to request conditional readout for the JSON value
argument when its static union has a Neuralese arm, reusing
`readNeuraleseIfReference`; crisp union arms would pass through unchanged and
then keep native `JSON.stringify` behavior. Do not alter the replacer or space
arguments, serialize arbitrary unknown values differently, or change JSON
input/output coercion. This should receive a focused generated-runtime test
before implementation because only direct type-checker reproduction currently
supports it; no generation failure has been attributed to this expression.

## Follow-up implementation: conditional JSON first-argument readout — 2026-10-08

The JSON union case above is now lowered as one typed call. The compiler captures
the JSON receiver/method and evaluates the argument expressions in source order
before awaiting the first argument's readout, then calls the captured method
with its original receiver and remaining arguments. Both module and eval scope
lowerings await the helper call itself, so a local assignment, type check,
concatenation, or nested argument receives the ordinary JSON result rather than
an implementation Promise. The conditional runtime brand check returns crisp
union arms unchanged. Replacer/space evaluation, replacer calls, ordinary JSON
serialization, and readout errors remain visible in focused generated module
and eval regressions. No replacer, space, or nested object field is
automatically read. No historical generation error is claimed.

### 16. Other implicit text/key conversions are candidates, not current fixes

The analyzer currently auto-reads typed values in explicit text contexts such
as `String(value)`, templates, string `+`, `concat`, `join`, and the JSON value
argument. Direct `analyzeEvalSnippet` reproductions on
`Neuralese<string>`-typed values produce no diagnostic and no readout for
computed keys (`record[key]`, `{ [key]: value }`), `new Error(message)`, and
standard string methods such as `label.includes(needle)`, `replace`, and
`split`. The current checker handles text conversion for a soft receiver's
`toString`, but does not inspect these argument/key positions
(`ts-host/src/compiler/neuralese.ts:130-184`). At runtime the opaque ref is an
ordinary object, so native JS coercion would use its object representation
instead of the typed payload.

Prospective priority: computed keys are medium because `ToPropertyKey` has a
clear value position; Error messages are medium/low because the default global
Error constructor applies `ToString`; string methods are low/medium because
support must enumerate standard library declarations and specific argument
positions. For each, a typed read of only the key/message/argument can preserve
crisp arms through the existing brand check; never recursively read arbitrary
objects or inspect transport metadata. Require the unshadowed standard global
or exact TypeScript-library declaration. Method and constructor rewrites must
capture receiver/constructor and every argument in native order before
awaiting, since an inline `await` can otherwise move effects past argument
evaluation. These were reproduced compiler gaps, but no recent generation
trace demonstrating failure was found.

`console.log(value)` is different: it receives a value for inspection and
does not apply JavaScript `ToString` to each argument. It should remain an
opaque diagnostic/trace display, not gain semantic readout simply because the
host eventually renders it as text. No reviewed generation trace demonstrates
failure in these computed-key, Error-message, or string-method-argument forms;
they are prospective, lower-priority candidates pending specific usage evidence
and focused ordering tests.

### 17. Dynamic text was put inside a static Neuralese marker in REL654

**Priority: medium; actual generation failure caused by an incorrect expression
choice, not compiler/readout behavior.** Luna V21 REL654
(`runs/neuralese-semantic-iterate-reducers-v21-20261008-v3/generation-review-v1/luna/luna-campaign-v1/slot-02/jobs/000001-b18c93d7fdbce3ce.trace.jsonl`, child invocation `task-1-t50gm2/10`, eval action seq 23) first computes `old = String(priorNotes)` and a correct ordinary `next` string, then assigns a literal marker to `out: Neuralese<string>`. The emitted marker body is seven characters, `${next}`. The child returns that block, and the caller's readout yields exactly `${next}`; it blocks because the notes contain no case facts. The returned reference/readout faithfully represent the written body.

This is the specified marker behavior: `writeLiterals()` extracts each body and calls `port.write(inner)` before eval compilation in `ts-host/src/native/neuralese.ts`; marker content does not run as JavaScript. The S0 specification likewise defines markers as vector payload and keeps host values opaque. Parsing expressions inside markers would need a new grammar and execution phase and could silently reinterpret prose. The clearer idiom already works: for a callable declared to return `Neuralese<string>`, compute ordinary text (`const next = ...`) and return `next`; `coerceReturn()` materializes that result through the configured writer (`ts-host/src/native/runtime.ts`). Existing general prompt text says plain text can satisfy this exact result type, but it does not contrast that path with static marker bodies. A focused prompt and guided note-child example now say marker bodies are literal and show returning computed text directly. This teaches the supported boundary behavior without adding a crisp-to-soft local conversion API or changing marker semantics.

### 18. `read_code` eval facade accepts the native tool's unambiguous argument shape

The eval scope already exposes a visibility-checked `read_code(name)` helper,
while the native tool schema takes `{ name }`. A test still called
`read_code({ name: "nl" })` inside eval and expected the pre-facade
“call it as a tool” diagnostic. This assertion failed the same way on committed
`9a43f4f3` and the then-current WIP, so it was a stale test expectation, not a
WIP regression. The implementation has now been normalized to accept either a
name string or exactly `{ name: string }` in eval. Both forms reach the same
visibility-checked `functionTool('read_code', ...)`; the separate native tool
surface remains unchanged. Other objects are rejected as a type mismatch
without stringifying or printing their contents. Focused tests exercise both
valid eval forms, the native tool form, and a malformed object. This is a
convenience for matching the same read operation's two existing call shapes,
not a namespace fallback or added read authority.

## Follow-up implementation: typed key, Error, and String argument conversions — 2026-10-08

The prospective candidates in §16 are now implemented in the compiler with
typed reads at only the native conversion position. Computed element-access
keys and computed object-literal keys use conditional `ToPropertyKey` readout;
record receivers and values stay opaque. The default global `Error` function
and constructor read the first argument when it is a soft union arm, capturing
the callee and all arguments before awaiting that read. Standard `String`
methods read only declared text positions for `includes`, `startsWith`,
`endsWith`, `indexOf`, `lastIndexOf`, `localeCompare`, `padStart`, and `padEnd`.
The lowering captures the receiver, method, and complete argument list in
native order, then calls the method with its original receiver. Crisp union
arms remain unchanged through `readNeuraleseIfReference`; custom methods with
matching names are not treated as standard String methods.

Focused scope and compiled-module regressions cover key lookup and object
construction, symbol/crisp arms, nested finite loops, failed read propagation,
Error options evaluation order, String search/padding argument positions, and
receiver/argument side effects. The full isolated `neuralese.test.mjs` suite
passed against `/tmp/natlang-json-union-check/dist`. Commits: `93ce10d8`
(computed keys), `ecd76d99` (Error messages), and `d04874af` (String method
arguments). These remain prospective ergonomics fixes: no recent model trace
was found that attempted any of these exact expressions.

The generated-guidance path also has a regression in
`ts-host/test/guided-soft-scaffold.test.mjs`: it builds a real guided case,
passes the generated root source through `compileModule()` and inline analysis,
asserts the prompt example is literal text with no interpolation nodes, then
compiles and executes the recommended `const next = String(priorNotes); return
next;` eval pattern against the isolated runtime output.

## Follow-up coverage check: array stringification and RegExp-capable methods — 2026-10-08

**Priority: medium; prospective generated-runtime mismatch, no historical
generation trace found.** A typed array of soft text has one working native
text path: `values.join()` and `values.join('|')` request `joinNeuralese()` when
the array/tuple's direct element type has a Neuralese arm
(`neuralese.ts:105-126`). The helper asynchronously reads direct reference
elements, applies commas or the supplied separator, and preserves index order
and nullish-element blanks (`runtime/lowered.ts:55-68`). However,
`values.toString()`, `String(values)`, and `` `${values}` `` produce no readout
for `(Neuralese<string> | string)[]`. An isolated compiled module returned
`[object Object],plain` for all three, while `values.join()` returned the
resolved payload. This is wrapper coercion, not a historical model failure.

Three implementation defects in the explicit join path were fixed in
`2b4e00d2`. The helper now uses asynchronous element readout only when the
captured runtime method is the standard `Array.prototype.join`; custom own or
prototype `join` methods are called with their original receiver and all
arguments. Receiver, method, and arguments are captured once in JavaScript
evaluation order before asynchronous reads. The helper no longer performs an
extra `in` check, which avoids a Proxy `has` trap absent from native join, and
it reads the array length once and each indexed value once. Holes and
nullish elements still yield blank fields, while inherited indexed properties
are read as native join would.

The soft separator gap was fixed after this audit was written. For the
standard intrinsic, `joinNeuralese()` now reads a Neuralese separator after
the single `length` read and before any indexed element gets. It preserves the
native default comma when the actual argument is `undefined`; a typed
separator that resolves to `undefined` follows ordinary `ToString` behavior.
Crisp union arms retain native coercion. Custom own/prototype join methods get
the original unmaterialized argument and are not read by the helper. Direct
and compiled-module tests cover mixed soft/crisp separators, effects order,
custom method behavior, and read failure before indexed gets in commit
`8758493b`.

Explicit `.toString()` now has a narrow typed readout in `d2dbe1a8`; this does
not rewrite the call blindly to `joinNeuralese`. The compiler selects it only
for statically typed array/tuple receivers whose direct element type has a
Neuralese arm. Runtime lowering captures the receiver, resolved `toString`
method, and arguments once in source order. If the resolved method is custom,
it calls that method with the original receiver and arguments and boxes the
result so a thenable is not accidentally awaited. For the native
`Array.prototype.toString`, it fetches the current `join` once: a custom join is
called unchanged, a non-callable join uses native `Object.prototype.toString`
fallback, and only intrinsic `Array.prototype.join` takes the existing typed
element-read path. The tests exercise sparse arrays, inherited indexes,
soft/crisp direct elements, custom methods/getters, thenables, thrown errors,
and opaque nested arrays. This preserves method lookup and native fallback
semantics without recursive reads.

The shared intrinsic-join path now also applies ECMAScript `ToLength` to its
single `length` read before resolving the separator. Unary plus preserves
abstract `ToNumber` conversion and its Symbol/BigInt errors; the result is
truncated and clamped to `[0, Number.MAX_SAFE_INTEGER]`. Bounded scratch chunks
avoid imposing JavaScript Array's `2^32-1` storage ceiling as an extra limit.
Tests use soft array elements behind Proxy lengths `2.9`, `'2'`, `undefined`,
and a negative value, and confirm direct/boxed BigInt and Symbol fail before
separator or index reads. They also cross the scratch-chunk boundary. No
arbitrary iteration cap was added; an extreme host-reported length can still
consume the same kind of time/output resources as native `join`.

`String(array)` and template interpolation remain deferred. Both first perform
string-hint `ToPrimitive`: lookup and call `Symbol.toPrimitive` when present,
otherwise try `toString` and then `valueOf`, preserving getter/call side effects,
the primitive-result requirement, and thrown errors. A safe narrow lowering
would need to emulate that whole protocol and only read direct array elements
if the selected path reaches the native Array `toString` → intrinsic `join`
sequence. The explicit `.toString()` coverage does not justify silently
changing those broader coercion paths. Neither path should recursively read
nested arrays or object graphs. Explicit `.join()` remains the direct choice
when callers want to control the separator.

The new String-method allowlist deliberately covers methods whose supported
argument slots are text-only (`includes`, `startsWith`, `endsWith`, `indexOf`,
`lastIndexOf`, `localeCompare`, plus the pad string slot). `replace` and
`replaceAll` have RegExp and string search arms, plus string or callback
replacement arms; `split`, `match`, and `search` also accept RegExp. They are
not inherently impossible to support: exact declaration/argument-position
checks and conditional readout can leave crisp RegExp or callback arms intact,
then let native `Symbol.match` and replacement protocols run. They were
excluded from the initial patch because a generic text-argument rule would
coerce these nontext alternatives and could change callback or RegExp
behavior. A separate follow-up should test overloads, mixed arms, custom
`Symbol.match`, and argument side-effect order before extending the allowlist.

## Follow-up: immutable evidence handles for guided note children — 2026-10-08

**Priority: high; grounded in V22 Luna RES-932 row 5.** The trace at
`runs/neuralese-semantic-iterate-reducers-v22-20261008-v5/generation-review-v1/luna/luna-campaign-v1/slot-03/jobs/000005-b9ae0c0066fb56c2.trace.jsonl`
shows the root task contract explicitly saying the four `pass-*.md` inputs are
read-only. The child received a writable `FileHandle` and successfully replaced
`pass-01-request.md` with a shortened summary at action sequence 23. The
subsequent evidence read therefore observed the child's paraphrase. The
parent's task was to write `decision.json`; this child write was outside the
explicit evidence contract, even though its file capability permitted it.

The existing APIs already express the intended boundary without a new
authority rule: `FolderHandle.snapshot()` returns a read-only
`FolderSnapshot`; `FolderSnapshot.file(path)` returns a `FileHandle` backed by
that snapshot; and `Folder.beginFileTransaction()` carries the backing
`access: 'read'` mode into the child's one-file transaction. The guided
builder now passes `folder.snapshot().file(current.evidence_path)` for the
current pass and states that the supplied handle is a read-only snapshot.
`ts-host/test/interpreter.test.mjs` proves the actual nested call only sees the
single file, reads it, rejects writes through both `FileHandle` and backing
folder, and leaves the parent free to write `decision.json`. The original
evidence bytes remain unchanged. The parent folder and output capability were
not made read-only, and no general write guard was introduced.
The guided builder change and integration regression are in `30d434ed`.

The snapshot materializes the parent folder's files before extracting one
handle. This is a bounded ergonomics cost for the current small evidence
folders; the child receives only the single file handle and the nested runtime
still constrains its scope to that file. No dedicated single-file snapshot
API was added.

## Final reviewed boundary disposition — 2026-10-08

The additional V20–V22 finish, result, tool-call, async, and capture traces did
not establish another transparent normalization. Three concrete Luna failures
remain honest contract errors. In
`runs/neuralese-semantic-iterate-reducers-v22-20261008-v5/generation-review-v1/luna/luna-campaign-v1/slot-01/jobs/000000-9d5661c7178ae3fc.trace.jsonl`,
seq 37 returns a `Draft` where the invocation requires `Neuralese<string>`;
treating it as text would discard the declared record shape. In
`runs/neuralese-semantic-iterate-reducers-v22-20261008-v5/generation-review-v1/luna/luna-campaign-v1/slot-04/jobs/000003-6b2c1b6239688ff2.trace.jsonl`,
seq 9 logs a local `answer` but finishes without a result expression or
explicit return; selecting the last logged/local value would invent a result
rule. In
`runs/neuralese-semantic-iterate-reducers-v22-20261008-v5/generation-review-v1/luna/luna-campaign-v1/slot-05/jobs/000012-346598160e2be594.trace.jsonl`,
seq 9 calls `await decide(input)` although `decide` is a record, not a
function. Choosing a reducer from that record would invent dispatch semantics.
These paths should retain clear type/finish diagnostics.

No recent trace reviewed here showed a legitimate error caused by omitted
optional arguments, implicit Promise awaiting, or an async collection callback;
no evidence supports making arbitrary callbacks async or automatically
awaiting tool returns. The source-backed helper persistence, explicit capture
shape, parameter-shadowing rule, typed return contract, and per-handle file
authority cover the actual cross-eval, capture, and effect boundaries. The
array `toString` case above is a reproduced prospective runtime mismatch, not
a claim about a past model failure. No recent actual generation failure was
found for `String(array)`, array template interpolation, or the excluded
RegExp/callback-capable `replace`, `split`, `match`, and `search` methods. Those
remain deferred protocol work: revisit them when a preserved trace shows a
specific typed value at a native conversion position, then require exact
declaration/overload recognition and ordering tests for crisp RegExp/callback
arms, `Symbol.match`, custom methods, getters, and errors. No missing function,
missing return, or absent capture in the examined traces is treated as an
implicit-value conversion opportunity. The only capture-name collision
evidence is recorded in §12's Luna V20/V34/V35 traces; the corrected
parameter-shadowing behavior is not a reason to infer unspecified captures.

## Final follow-up: scalar conversion and native string replacement — 2026-10-08

`Number(value)` and `Boolean(value)` were missing typed readout despite being
explicit scalar conversions. An isolated compiled-runtime reproduction showed
that the underlying JavaScript wrapper would give `Number(ref) === NaN` and
`Boolean(ref) === true`, regardless of the trained scalar text/value. No
preserved generation trace showed these exact conversion failures; this is a
prospective correction, not historical failure evidence. The compiler now
recognizes only unshadowed global `Number(...)` and `Boolean(...)` calls whose
Neuralese alternatives have declared `string`, `number`, or `boolean` payloads.
It evaluates the callee and every argument first, conditionally reads only an
actual reference in argument zero, then invokes the captured native function.
Crisp union arms retain native conversion, extra arguments still evaluate,
and read failures propagate. Opaque record/function payloads are not
crispified. `new Number(...)`/`new Boolean(...)` remain outside this change.
Focused eval and compiled-module tests cover soft and crisp arms, order,
truthiness, read errors, shadowing, and unsupported record payloads
(`ts-host/test/neuralese.test.mjs`; commit `c61f79ab`).

Mixed soft/crisp values in branch conditions previously escaped the pure-soft
condition diagnostic, allowing wrapper truthiness. The compiler now conditionally
reads only declared `Neuralese<boolean>` alternatives in actual `if`, `while`,
`do/while`, `for`, and ternary condition positions; crisp arms retain native
truthiness. In nested `&&`/`||`, operands are read immediately before their
short-circuit truthiness test, so a soft false skips the right side. Ternary
conditions and selected branch values are handled recursively when the whole
ternary result is itself a guard. Readout still requires an async function;
synchronous contexts receive the async-read diagnostic. Value-producing
logical expressions outside guards retain the existing error, and soft strings,
numbers, objects, and mixed payloads with nonboolean soft arms are not given
Boolean meaning. Eval/module regressions cover short-circuit order, crisp
alternatives, nested logical guards, ternaries, and sync rejection. This is a
prospective ergonomics change; no historical condition failure or generation
success-rate effect is claimed.

There is one actual `replace` attempt: Luna V20 trace
`runs/neuralese-semantic-iterate-reducers-v20-20261008-v8/generation-review-v3/luna/luna-campaign-v3/slot-05/jobs/000004-0bfdbbf2984e99ac.trace.jsonl`,
seq 19 calls `priorNotes.replace("Recomputing ...", "The selectedItems field ...")`
with `priorNotes: Neuralese<string>` and receives `neuralese-opaque-access`.
The nearby workaround explicitly uses `String(priorNotes).replace(...)`.
The trace supports only the literal string/string overload; it does not support
coercing RegExp search values or callback replacements. The compiler now
recognizes only the standard String `replace` declaration on a declared
`Neuralese<string>` receiver (or a union with crisp strings), with crisp
string-like search and replacement arguments. It reads the receiver first,
then looks up the actual `replace` method on the materialized string, evaluates
arguments, and invokes the captured method. This preserves a customized
`String.prototype.replace` getter/method, including getter/method errors and a
non-callable property. Tests cover the actual trace form through eval and a
compiled module, plus mixed crisp receivers and method/argument/read failure
order (`ts-host/test/neuralese.test.mjs`; source lowering in
`ts-host/src/compiler/neuralese.ts` and `ts-host/src/compiler/lower.ts`).
Soft search/replacement arguments remain unsupported in this specific path;
the standard string argument lowering can be considered separately. `replaceAll`,
`split`, `match`, and `search` remain deferred until a trace establishes a
concrete typed argument case and their RegExp, callback, and `Symbol.match`
behavior can be preserved exactly. This actual `replace` trace supersedes the
earlier statement that no recent trace had attempted it; no RegExp/callback
overload failure is claimed.

### Receiver-only standard String methods — 2026-10-08

The declared `Neuralese<string>` surface now includes `trim`, `trimStart`,
`trimEnd`, `toLowerCase`, and `toUpperCase`. The compiler accepts these only
when the payload arms are statically strings and the selected member resolves
to TypeScript's standard `String` declaration. Typed reads happen on the
receiver expression; property lookup then runs on the resulting primitive,
followed by the ordinary JavaScript method call. Crisp arms in a soft/crisp
string union pass through without a trained read. Other payloads and
same-named custom service methods remain opaque. These receiver-only methods
share the ordinary readout path and add no per-method call lowering.

Eval and module regressions cover all five recognized members, async-only
readout, crisp union arms, failure before method lookup, customized prototype
getter/method order, and a returned thenable object that remains unassimilated
as an expression value. The recent V20–V24 traces reviewed did not attempt
these direct receiver forms; this is prospective ergonomics support, not a
claimed generation recovery. The `.replace` implementation remains separate
because it also validates its current literal-string argument overload and
preserves complete call evaluation order.

The same declared receiver support now includes argument-taking standard
string operations: `includes`, `startsWith`, `endsWith`, `indexOf`,
`lastIndexOf`, `slice`, and `substring`. Only statically crisp arguments are
accepted, and the selected member must resolve to TypeScript's standard
`String` declaration. Lowering reads the receiver expression before native
property lookup and ordinary argument evaluation/invocation; crisp string
union arms remain native. This does not convert soft arguments or add
RegExp/callback overload handling. Eval/module tests cover all seven methods,
receiver read order, method getter and argument order, and return-value
preservation. There is no recent trace showing an attempted one of these seven
methods; this is a prospective consistent extension of the observed `.replace`
case, not a historical rejection claim (`ts-host/src/compiler/neuralese.ts`,
`ts-host/src/compiler/intrinsics.ts`, `ts-host/test/neuralese.test.mjs`).

### Portable target intersections and hybrid records — review only

`ts-host/src/compiler/targets.ts:166` currently rejects intersection types as
portable targets, and line 184 rejects records combining fixed properties
with a string index signature. A scan of the recent V20–V24 generation trace
and result JSONL files found no matching diagnostic, so I have no recent
generation failure to attribute to either boundary. Finite record
intersections such as `{ id: string } & { label?: string }` could plausibly
flatten through TypeScript's merged property view, while callable/host
intersections, conflicting properties that reduce to `never`, and intersections
with index signatures need distinct treatment. Hybrid records also have a
portable representation question: a broad index schema may erase the fixed
field contract. Keep these errors for now; only reconsider finite plain-record
intersections if an actual target-conversion attempt is preserved and a
focused assignability round-trip demonstrates that flattening keeps the same
accepted values. Do not treat this as a model refusal or widen host authority.

### V25 authority-pass note overreach — source-guidance correction

Two V25 rejected outcomes are semantic note errors with intact tool and file
scope. In TREE-710, the condition note selects TREE-A1 at 91 points and
excludes TREE-A2 because its permit covers a different block. The separate
pass-04 source says: “Parks duty officer PO-4 released TREE-710 for dispatch.”
The explicit rule requires the parks duty officer release for the work order,
but the authority note child adds an item-level grant requirement for TREE-A1.
The final Draft is therefore `TREE-A1`/`91`/`hold` despite the request-scoped
release. Trace: `runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-01/jobs/000000-7343a02dd8a3c8dc.trace.jsonl`
(condition child `task-1-wuwsv6/7`, authority child `/11`, final interpreter
`/12`).

In FISH-730, the condition note resolves VESSEL-TERN at 18 stations/day as the
highest eligible candidate; VESSEL-REEF's 23 is excluded by its stale observer
credential. The pass-04 source says: “Permit FP-73 is current for FISH-730.”
That matches the explicit current-fisheries-permit prerequisite, but the
authority note child says the current-permit requirement is satisfied and then
contradicts itself by claiming no recorded authorization. The final Draft is
`VESSEL-TERN`/`18`/`hold`. Trace:
`runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-05/jobs/000004-7306a12d697fdb93.trace.jsonl`
(authority child `task-1-555247/13`, eval action sequence 32).

A later V25 BURS-761 trace shows a related loss one stage earlier. The running
notes stringify to a Draft-shaped JSON object with candidate IDs, imprecise
measure prose, and `hold`, but no enrollment, documentation, duplicate-award,
or committee-approval facts. The final interpreter's early attempt is blocked
because the notes omit those facts; it also later reports that it cannot
determine them from the notes. The pass evidence itself is present in the
folder. Trace:
`runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-01/jobs/000011-c00d1c7716a0f364.trace.jsonl`
(notes and interpreter invocation `task-1-7d9y29/33`, especially actions
10, 20, 29, 46, and 54). The saved result later contains a correctly formatted
`APP-57; APP-52` / `2200; 1700` / `hold` result, so this is evidence of a
repeated recoverable note-retention failure, not evidence that the entire
request ended without a result.

VAX-720 is a useful contrast: its pass-04 release names request VAX-720, the
note records that as the required authority, and the final result correctly
selects SHIP-C7 and SHIP-C4 with measures 3100 and 2400 and `dispatch`.
Trace: `runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-03/jobs/000002-fbd8598a7433139b.trace.jsonl`
(authority child `task-1-x9lvni/11`, final interpreter `/13`).

The V25 source generator currently states the named authority in `decision_rule`
but uses generic “recorded authorization” language in the decision field and
pass-04 constraint. The shared `caseFrom` builder now supports per-task
`authorizationRule: { requirement, scope }` metadata. With scope `request`, it
repeats the precise named prerequisite and request ID in the decision rule,
decision-field guidance, and pass-04 instruction, and explicitly says not to
add item-level grants. With scope `each_selected_item`, the wording requires
coverage for every selected item and says a request-level record alone is
insufficient. Source-authored metadata declares the rule's scope; the helper
does not infer authority from source facts or alter gold outcomes. V25 frozen
source and result artifacts were not changed. This is a successor-pool source
authoring improvement, not a compiler/runtime repair and not evidence of an
agent tool boundary failure.

The shared guided note scaffold now asks each pass to keep a complete running
evidence record in prose, including observed eligibility findings, exact
metrics with units, and the named authority prerequisite and its observed
status. It explicitly says not to replace that record with only a Draft-shaped
JSON summary or selected IDs/measures. The final interpreter still decides
from accumulated evidence and the declared contract. The wording change is
prospective guidance based on the BURS-761 trace; it does not add file access,
infer missing facts, change authority, or rewrite frozen V25 artifacts.

### Separate `nl.with` capture and child types — 2026-10-08

Luna V24 slot 05 provides an actual refusal in
`runs/neuralese-semantic-iterate-reducers-v24-20261008-v3/generation-review-v1/luna/campaign-v1/slot-05/jobs/000004-d386e4a2238dcc2b.trace.jsonl`,
action sequence 20. The authored call used
`nl.with<{notes: Neuralese<string>; text: string; context: string; pass: string; constraint: string}, string>(captures)`
and then invoked the returned child with a separate record. The compiler returned `nl-type-arguments`, saying
`nl.with` accepted one type argument. The child input was a distinct single `input` parameter shape, so the two
schemas had been conflated by the diagnostic rather than describing an authority issue.

The direct `nl.with<CaptureRecord, Result>(record)` form is now accepted by the same inline capture analysis and
validation path as other explicit captures. `C` checks the finite listed record (unknown fields, missing required
fields, and nonassignable values are diagnosed); `Result` is the child result type. Invocation arguments still
determine the child parameter shape and are not inferred from or granted by the capture schema. Existing one-type
argument forms retain their result/full-callable meaning. A literal capture such as `'x'` is assignable to a
declared `string`; validation is one-way assignability from actual value to declared field type, not mutual
assignability. Shadowed capture handling remains unchanged, including evaluation of the listed initializer.

Eval-scope, compiler, and module regressions exercise the separate input and capture records, literal widening,
parameter shadowing, and missing/unknown/wrongly typed fields (`ts-host/test/compiler.test.mjs` and
`ts-host/test/neuralese.test.mjs`). This implementation addresses the observed refusal; no claim is made about
whether it changes generation success rates.

### `read_code` discovery for current native tools and visible host objects — 2026-10-08

V25 generated two honest but unhelpful inspection dead ends. Slot 04 trace
`runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-04/jobs/000003-1bcf5388b6b31200.trace.jsonl`, seq 46, child `task-1-thcshg/39`, asked `read_code({name:"neuralese"})` and got `no-such-function`. Slot 01 trace
`runs/neuralese-semantic-iterate-reducers-v25-20261008-v1/generation-review-v1/luna/campaign-v1/slot-01/jobs/000005-c42ca00e318cef1a.trace.jsonl`, seq 39, child `task-1-umqgdt/9`, asked to read `return_result` and received the same broad source-lookup dead end. The first name is a host metadata object; the second is a native action tool, not a program function.

`read_code` now returns truthful discovery metadata for both cases. The agent records the exact native tool schemas it offered to the current session turn; `read_code("return_result")` or `read_code({name:"return_result"})` returns that tool's description and JSON Schema, explicitly labeled as a separate action rather than function source. Schemas for tools not offered in the current call are not returned. For a host service object, `read_code("neuralese")` reports a type-only declaration using the same own-property descriptor formatter as the opening; it does not invoke getters or print values. The lookup uses `availableServices()` for the current call, so an out-of-scope service name receives no metadata. Declared external service/package lookup and actual program-function source lookup retain their existing meaning.

The tool schema cache is filled from `NativeToolAgent.tools()` using the definitions already sent to the model. The shared service type formatter lives in `ts-host/src/native/introspection.ts` and is used by both the opening and `read_code`. Eval and native `read_code` tests cover the real offered `return_result` schema, the `neuralese` member types, omitted getter/value contents, and an unoffered file tool and out-of-scope service. The isolated TypeScript build and `eval-guidance` suite passed (12/12). No canonical `dist` was rebuilt or changed.

### V27: typed callable inputs accept structural record views — 2026-10-08

All 30 V27 campaign cases reached terminal results. Their final typed values match the existing `review-facts.json` fields exactly; this is a consistency check against the source's two-provider consensus, not an independent re-adjudication of every source label. The actual traces contain 15 recoverable `unknown-field` failures in seven cases (05, 06, 09, 18, 19, 22, 24). The generated code repeatedly parsed a supplied question/state record, annotated only the fields needed by a child, and passed that value to `nl<T>(...)` or `nl.with<T>(...)(...)`. The checker then rejected unrelated source-envelope or state fields such as `source_path`, `source_question_instance_id`, `question.type`, `state.customer`, or `state.assistant_pending_proposal`.

Representative trace evidence:

- Case 05, `slot-05/jobs/000005-93a1a077923dc2c5.trace.jsonl`: root eval action 31 passes an `initialDraft` whose declared task value includes `triage__threat_chargeback_or_public`, although its local `InitialDraft` annotation omitted that field. Child eval actions 11, 22, and 32 also fail on extra `question.type` or `state.customer` properties.
- Case 09, `slot-01/jobs/000009-8844b5d2bc3e54b5.trace.jsonl`: child `task-1-16ilgl/6` fails at actions 20, 30, and 40 as its input changes from a narrowed question/state record to the complete `record`.
- Case 18, `slot-04/jobs/000018-1d840e16daad3079.trace.jsonl`: child `task-1-uij4pi/10` fails on extras at actions 9, 18, and 26; child `/14` fails on `source_question_instance_id` at action 11.

These are transparent TypeScript structural-view cases: the caller explicitly passed an existing object, declared the required fields and their types, and the rejected properties were unrelated source fields. Commit `fb409f2d` adds an input-only coercion option used at `definitionNode` invocation arguments. It recursively validates declared fields and required fields while preserving enumerable extra values and their existing handles. Default record coercion remains closed for declared results and captures, and no services or files are added to child authority. An isolated TypeScript build succeeded; focused unit and runtime regressions passed 2/2, including a scoped snapshot handle that remains read-only and cannot read its sibling file while the parent can still write its output.

Other V27 failures remain honest boundaries rather than candidates for fallback behavior. Case 01 action 24 is a recovered provider `WebSocket closed 1000`; it is not a runtime type failure. In case 03, `String(ev as any)` erased the declared `Neuralese<string>` type, yielding `[object Object]` and then a `JSON.parse` error; the typed readout path is available when the declaration is preserved. Cases 04, 10, 16, and 28 passed records/text to `decide(fn, ...args)` where its required first argument is a function. Case 17 has a malformed `Array<...>` annotation (`'>' expected`). Case 12 returns a boolean from a helper declared to return `Neuralese<string>`. Those are source/API/result-shape errors and do not justify inferring a function, erasing a declared return contract, or coercing arbitrary opaque values.
