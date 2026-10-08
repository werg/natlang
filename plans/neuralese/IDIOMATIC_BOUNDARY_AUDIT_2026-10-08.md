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

### 6. Eval exposes a read-only tool facade; other tool actions stay direct

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

Native `for...in` is now accepted unchanged, with compiler/runtime regressions
for ordinary property order and inherited enumerable keys versus `Object.keys`.
The focused compiler and interpreter tests passed 99/99 in an isolated build
copy. The isolated TypeScript build excluded `src/browser/**` because the
workspace's installed dependencies do not include
`@wllama/wllama/esm/index.js`; no dependencies or shared `dist` were changed.
Cross-eval capture-free helper persistence remains a useful but larger
source-backed design candidate, grounded in the cited V97 error and requiring
a separate lifetime/provenance design.
