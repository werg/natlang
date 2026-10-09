# natlang language specification

Version **0.5-draft**, 2026-10-03. This revision adds Neuralese (soft values), makes a
function's context an explicit argument, and replaces the caller-chain recursion
guard with structural termination rules.

natlang is TypeScript with natural-language functions. A natural-language
function is an asynchronous, typed function whose body is instructions; a model
executes them in a persistent TypeScript scope. Everything else is ordinary
TypeScript. The design rationale is in
[`docs/TS_INLINE_HOST_INTEGRATION_PLAN.md`](../docs/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[`docs/inline-natlang-lambdas.md`](../docs/inline-natlang-lambdas.md), and
[`docs/ITERATE_ON_PLAN.md`](../docs/ITERATE_ON_PLAN.md).

## Natural-language functions

A natural-language function has ordered typed parameters and one return type.
Calling it returns a promise of a value checked against that type; a failure
rejects with `NatlangCallError` carrying `outcome` (`quiesced` or `failed`),
`detail`, the call ID, and its trace.

There are two source forms.

**Inline.** A tagged template in TypeScript:

```ts
const priority: 'urgent' | 'normal' = await nl`Decide whether ticket is urgent.`(ticket);
const summarize: (text: string) => Promise<string> = nl`Summarize text in one sentence.`;
```

The compiler determines the signature before the model runs, from, in order:
an explicit type argument (`nl<F>`), the contextual type of the expression, an
immediate call's arguments and result context, and later uses of a local.
Conflicting evidence is a compile error. With no evidence (or an `any` or
`unknown` annotation) the result is open: the call may return any value, shaped
by the fields the code reads from it (`{ severity: unknown, … }`) or as a list. Template interpolations are
evaluated at call time and become part of the instructions.

In eval code, `nl` called like a function on literal instructions, as in
``await nl(`Is ${x} large?`)`` or `nl<T>("…")`, is the one-shot call it reads as:
``nl`…`()``, whose result is the answer. Instructions built at run time, and
other uses of `nl` as a value, are compile errors that show the template form.

**Named.** A `.nl` file with YAML frontmatter and instructions:

```yaml
---
description: Assess each observation against a criterion.
args:
  observations: string[]
  criterion: string
returns: Report
---
Assess every observation with assess, then summarize the assessments.
```

Frontmatter keys are `description`, `args`, `returns`, `types`, `kind`
(`function` or `directory-reducer`), `readout` (see Decision readout), `model`, and `uses`. `uses` lists package
items the function may call besides its companion folder, by path from the package root (`uses: [harness/cut]`); each
joins its context under its base name, with its own companion folder. `args`, `returns` and `types` hold
TypeScript type text, read verbatim rather than as YAML, so types need no
quoting: `rows: { title: string }[]`, `pick?: (x: string) => number`,
`returns: "yes" | "no"`. `args` and `types` are one `name: type` per indented
line (a type may continue on more deeply indented lines) or an inline object
type, `args: { q: string, n?: number }`. A value wrapped entirely in quotes is
the quoted text, as in YAML, so `"string[]"` is `string[]`; a lone string
literal type is written `'"yes"'`. The other keys are YAML. Application TypeScript imports a named
function as a module default export; the build generates its declaration
(`foo.d.nl.ts`).

## Contexts

Every natural-language function is a curried function of its **context**:
`(context) => (...args) => Promise<R>`. A context is an immutable,
content-addressed folder value holding `.nl` functions, TypeScript modules,
`.nz` files ([Neuralese files](NEURALESE_FILES.md)), data files and
subfolders. Its ID is a hash of its contents, so no context contains itself,
directly or indirectly; contexts form a directed acyclic graph.

**Default binding.** A named function `foo.nl` is bound to its companion folder
`foo/`. An inline `nl` inside a callable folder is bound to that folder; in
application code, to the nearest ancestor `natlang.d/` folder, which follows
the same rules. With no such folder its context is empty. `types.ts` in a folder
supplies type aliases to the functions in and below it.

**Calls.** A bound function calls exactly the items of its context. Items appear
as bindings and as properties (`helper(...)`, `group.child(...)`). Each item is
bound to its own context, so every call steps down the context graph. The
compiler and runtime enforce this.

**Context interface.** A function's free names (the items its instructions, code,
or explicit capture list refer to, with their types) are its context interface.

**Rebinding.** `foo.in(context)` returns `foo` bound to another context with the
same signature. The new context is checked structurally against `foo`'s context
interface (`context-interface-mismatch`). Rebinding selects skills, evaluates
candidate values of items, supplies fixtures to tests, and promotes revisions.

**Executable nodes come from files.** `.nl` functions and TypeScript functions are
executable nodes; everything else in a context is data. The executable nodes of
any bound context are selected from contexts loaded from files (the program's
tree, imported libraries, and staged trees written by a directory reducer and
compiled), with subsets and unions allowed. Rebinding never adds an executable
node (`context-new-executable`). An executable node may be replaced by an edited
definition that compiles against its signature and context interface. Data
entries may be added, removed, and replaced freely, type-checked where an
interface declares their type. A function-typed value stored as data calls only
the executable nodes of its own definition site's context.

**Authority.** Contexts hold definitions and data, never capabilities. Services
are supplied by the host (see Services and effects); rebinding cannot widen
authority.

Item names are identifiers and must not collide with function properties
(`call`, `apply`, `bind`, `name`, `length`, `prototype`, `constructor`, `then`,
`iterateOn`, `in`, `with`, and similar).

## Captures

An inline `nl` with a text body captures the visible bindings its instructions
mention by exact name. Data captures are read live at each call. A mentioned
`let` may be reassigned by the model: the write-back happens after a successful
eval and is version-checked, so a concurrent change fails that eval with
`capture-conflict` and the model retries it. Reassigning a captured `const` is a
compile error. Property writes on live objects take effect immediately.

`nl.with(captures)` lists captures explicitly. It is required for soft bodies,
which cannot be scanned for names (see Neuralese), and allowed for text bodies.
Explicit captures are snapshots taken when the function value is created;
`live(x)` marks a `let` capture that is read at each call and may be written
back as above. A function value with live captures belongs to its running scope
and cannot be saved to a file.

A binding whose type contains a function type is always captured by value, in
implicit and explicit captures alike.

## Callable-folder TypeScript

TypeScript files in callable folders are ordinary modules. A default-exported
function makes the module callable; named exports are callable attributes;
exported values are typed values. They may import sibling items, declared npm
packages, `natlang:services`, and the natlang surface module; other local files
are rejected. They follow the iteration and termination rules below. Application
TypeScript outside callable folders is unrestricted.

## Iteration and termination

Callable-folder TypeScript and eval code use finite iteration: `for...of`,
counted `for` loops (the condition may join the counter bound to an early exit with `&&`, as in `i < n && !found`), and array methods. A counted loop reads its bound once,
when it starts; the bound must be a finite number and the counter must advance
toward it. `while`, `do`, `for...in`, open `for(;;)`, generators, and code that
defines iterators (`Symbol.iterator`, `Symbol.asyncIterator`, `Iterator.from`, a
class extending `Iterator`) are rejected; `for...of` is guarded at run time
against iterating a growing collection. `for...of` takes an array, string, Map or
Set, and also `entries()`, `keys()` and `values()` of an array, Map or Set and
`string.matchAll(regex)`: each is a finite view of a collection that is fixed for
the loop (an array that grows during the loop throws; a Map or Set view iterates
a snapshot; `matchAll` advances through a fixed string). `natlang check` reports a
`for...of` source that would be refused by the same rules, with its location. `for await` consumes the async iterables
the host provides (a `fetch` response body, a service's stream, a package's,
an `iterateOn(...).streamUntil(...)` stream), and arrays of promises; it is paced
by whatever produces them. `setInterval` is not available: repeated
work is an `iterateOn` loop, whose step can wait with
`await new Promise(r => setTimeout(r, ms))`. Timers that eval code schedules
belong to the call; those still pending when it finishes are cleared.

Open-ended iteration uses `iterateOn(step, initial, ...args)` or
`fn.iterateOn(initial, ...args)`, which records each step and returns the first
state for which the stopping predicate holds (`until`, or `streamUntil` for a
stream of events):

- The predicate is an ordinary TypeScript function or a natural-language
  function returning `boolean`. It is checked on the initial state and after
  every step.
- `withMeasure(state => n)` supplies a non-negative integer that must decrease at
  every step that continues the loop. The step after which `until` is true ends
  the loop and need not lower it, so a measure that counts the work still to do
  needs no padding (`2*remaining + running`). A measure at 0 while `until` is
  still false ends the loop with `IterationLimitError` coded
  `iteration-measure-exhausted`, since a further step could not lower it; a
  continuing step that does not lower it is `IterationDivergedError`, whose
  trajectory ends at the last state that did. `withLimit({ maxSteps })` a hard step bound. A deadline alone is
  not a bound.
- With a TypeScript predicate, a measure or a step limit is required
  (`iteration-unbounded`). With a natural-language predicate none is required:
  the predicate runs under a system prompt that states it is a loop's stopping
  condition, shows the iteration count and whether recent states changed, and
  asks it to accept a reasonably met criterion rather than demand unattainable
  perfection.
- A progress judge reviews the trajectory at boundaries scheduled from per-site
  step-count and time statistics. Reviews never stop a run on their own; a
  `divergent` verdict ends it with `IterationDivergedError`.
  `checkProgress(judge)` replaces the judge; `checkProgress('off')` disables it
  and then requires a measure or step limit.

Natural-language functions cannot recurse, by construction:

- Calls step down the context graph (see Contexts), which is acyclic. A
  definition rebound to another context is another function; rebinding is a
  host capability.
- Function-typed bindings are captured by value (see Captures), so a closure
  cannot reach itself through a capture.
- Recursive function types are rejected: a type alias may not mention itself in
  a function parameter or result position, directly or through other aliases
  (`type-recursive-function`).
- Functions written at run time (inline `nl` and Neuralese function literals in
  eval) nest at most five active layers below a root.

TypeScript functions in eval and callable folders recurse structurally. A
function already running in its own call chain may run again only on a smaller
argument: a part of its input (reachable through its properties or elements), a
shorter array or string, or a smaller non-negative integer. The same argument
must keep getting smaller along the chain, and a part may not repeat, so every
chain of calls ends. Anything else fails when it happens
(`NatlangRecursionError`). Concurrent sibling calls and repeated sequential calls
are allowed, and host TypeScript callbacks into natlang keep a run-time check that
a definition is not re-entered from its own call.

What this guarantees: code in eval and callable folders cannot loop or recurse
without bound by itself. The deliberate exceptions are model-controlled: an
`iterateOn` whose stopping predicate is a natural-language function, and the
agent loop of a call, which has no turn or time limit unless the caller sets
one. Services and imported packages are the host's and may run as long as they
do. The rules address code written in good faith; reaching around them through
reflection is outside them, and the Node backend is not a sandbox. An application
runs as long as outside events (users, requests, the clock) keep arriving, each
handled in bounded work.

## Services and effects

The application supplies host capabilities as typed services when it creates a
runtime or a task. Callable-folder code imports them from `natlang:services`;
eval exposes them as named, read-only bindings. Every service call is traced as
an effect. Effects are not rolled back when a call fails; applications that must
not repeat an effect record operation identities and reconcile unknown outcomes.
Arguments reach a service as plain data of the host realm (eval runs in a realm
of its own), so a service can store or compare them like any host value.

A service lists the methods that are external effects to perform once under
`ONCE_EFFECTS` (`{ [ONCE_EFFECTS]: ['send'], send(...) {...} }`): a call with the
same arguments as an earlier call on the same service object returns the earlier
result instead of acting again, whether an eval ran again or another call made
it; a call that failed may run again. The service object's lifetime is the scope,
so a host hands a fresh object to each unit of work that may repeat the effect on
purpose. Reads are not listed: they must see the current state.

A service may come with a declaration: the TypeScript declaration of its members,
with their doc comments, as a declaration file would give them. The model is
shown it as `declare namespace name { … }` and can read it with `read_code`,
but the implementation runs in the host and is neither readable nor editable. A
task gives external modules (a board, a simulated world, a store, a checker) this
way, so that only code the program owns can be changed. Importable packages are
external in the same way: `read_code("pkg")` lists a package's exports from
its type declarations, and `read_code("pkg.name")` shows one with its docs.

A service can be scoped to functions: it is then usable only in their calls and
the calls they make, as a specialist can reach systems its caller cannot. Its
declaration stays readable everywhere, and a call that cannot use it is told
which functions can.

## Model surface

A natural-language invocation offers the model these tools:

- `eval(code, timeout_ms?)`: run TypeScript in the persistent scope, optionally with a time limit; a timeout stops the natural-language calls the eval started.
- `read_page(id, page)`: read the next part of a tool output that was cut off; the cut-off names the ID.
- `compact_history(note)`: shorten the conversation (see Conversation length).
- `return_result(status, value?, reason?)`: finish the call. Status `success` returns
  `value`, of the declared type; `blocked` stops because required information is
  absent, and `failed` because the instructions require an invalid or contradictory
  operation, each with a `reason` sentence instead of a value.
- `read_code`, `edit_code`, `diff_code`: inspect and edit the program's codebase
  (its callable items); `read_code` also reads the declaration of a service or an
  importable package, which cannot be edited, and the documentation of eval's
  built-ins (`nl`, `iterateOn`, `transcript`). It is offered on every call.

A directory reducer additionally receives `list_files`, `search_files`,
`read_file`, `write_file`, `edit_file`, and `diff_files`, and the conversation
opens with the folder's file listing.

Instructions do not all need code: an answer that takes only reading and
judgment is given directly.

**Decision readout.** A function whose result type is finite (a union of
literals, `boolean`, `null`) may declare `readout: decision`. Its call then
offers no tools: the runtime asks the model to score every allowed value as the
whole reply to the call's opening (signature and arguments, under a short
system prompt) and returns the most probable one. The normalised distribution
is the call's trace event `decision_readout` and its note, so a decision is a
probability vector that proper scoring rules (Brier, ranked probability score,
log loss) can grade and, on a Neuralese server, differentiate
(`objectives.decision` in `natlang:learning`). A model config with
`decisionReadout: 'finite-returns'` applies the readout to every finite-typed
call. A backend that cannot score replies falls back to the ordinary tool loop;
the loader rejects `readout: decision` on a type that is not finite.

## Eval

The scope holds the parameters, captures, callable items, services, and
persistent locals. Parameters are `const` and deeply frozen; derive a new
variable instead of changing one. A `let` capture's assignments are written
back to the caller. Top-level `const` and `let` declarations persist across eval
calls. The conversation opens with the runtime's own eval: ambient `declare`
lines for callable items and services, then the parameters and locals declared
with their current values (large values cut off; see Cut-offs). An
eval is atomic: a failed compilation or execution commits no local or capture
changes (effects already performed remain). A final expression is only shown.
Eval code may create inline `nl` functions like any TypeScript; they see the
callable items and the scope bindings their instructions mention.
An unannotated inline `nl` gets its parameter types from its arguments and its
result type from how the eval uses the result (a condition makes it `boolean`,
a typed variable gives that type); when no use says it, the result is open:
the fields the eval reads, or any value, as the call chooses.
Ad hoc natural-language calls may delegate through five active layers below a
root. An existing instruction function loaded from a `.nl` file starts a fresh
root budget, even when called by an ad hoc child. At the fifth layer the system
prompt and built-in help omit further ad hoc delegation; a sixth layer is
refused by the invocation kernel. Inline `nl`, Python `nl`, and the `delegate`
tool share this count. Sibling calls have independent budgets. The usual
recursion checks for reentering an existing function and task resource limits
still apply.
In eval, awaiting an inline `nl` without calling it (`await nl`...``) calls it
with no arguments: it judges the names its instructions mention and its
interpolated values.
Types an eval declares (`type State = …`, `interface Row { … }`, without type
parameters) annotate its locals like the call's own types, in that eval and in
later ones; a type the runtime cannot express leaves a local so annotated open.
What an eval leaves unawaited (a promise in a local, the final value or a
`return`, or arrays and objects holding promises) is awaited before it is kept.
Natural-language calls belong to the eval that starts them. When an eval fails
(an error, a rejected value, or its `timeout_ms`), the calls it started that are
still running are stopped; service calls and other effects already made remain,
and timers it scheduled are cleared. An eval that finishes while calls it
started are still running stops them and fails, saying so, except for calls that
lost a `Promise.race` or `Promise.any`, which are stopped without failing it.
`Promise.all` keeps JavaScript semantics: when one call rejects, the others keep
running. A call that fails stops the calls it started.
A top-level `return value` stages the value as the call's result if it has the
declared type; a later valid return replaces it. Values that are not portable
data (functions, class instances, handles) are passed by reference as live
values.

Every eval also sees `transcript`, the call's earlier tool calls with their full
outputs, unless a parameter or local takes that name. It is searched rather than
read through: `transcript.search(text or regex, { in, limit })` returns matching
lines of reasoning, code, and output as `{ entry, turn, tool, in, line }` (optionally only calls of a given
`status` or `tool`), and `transcript.entry(n)` returns one call as
`{ turn, reasoning, tool, code, arguments, status, value, console, output }`
(negative `n` counts from the end): `reasoning` is the model's reasoning in the
turn that made the call (on its first call), `status` is the outcome (`ok`, `rejected`, `error`, …),
`value` an eval's returned value as data when it is portable and of modest size,
`console` what it printed, and `output` the full text the model was shown. It has no array access or iteration, and printing it shows a summary.

## Cut-offs

Whatever is too long to show is shortened one way, with a note that is not
TypeScript: `<<cut off: 338 of 340 items not shown; customers holds all of it>>`.
The note says what was left out, where all of it is in the scope (a variable, or
`transcript.entry(n).output` for a tool output), and for tool output which
`read_page` call shows the next part. Values (the opening's declarations, eval
results, stored locals, the staged result) are cut at item and field boundaries;
text output (console, files) keeps its beginning and its end. A cut-off literal
copied into an eval fails to compile rather than running on part of the data.

## Conversation length

A call keeps one conversation. Past three quarters of its context budget
(`contextTokens`; by default the context window the model's server reports,
less an eighth for the reply, or 16,384 prompt tokens when it does not say) the next turn offers only
`compact_history`, with a tool call required. Its `note` (at most 600 characters:
what the model is doing, what it found, what is left) is kept after the opening,
replacing any earlier note, and every older tool output and eval code is replaced
by a note naming the `transcript` entry that holds it. The note is written so the
model can continue from it alone; with it, the model is told to look into the
history only when something specific matters, by searching it. The model may also compact
on its own. Compacting again waits until the conversation has grown by another
quarter of the budget, and a request never exceeds the budget: if needed, outputs
are elided without a note.

## Completion

A call finishes in one of three ways:

- `return_result` with status `success` and a value of the declared type;
- a reply without a tool call, which returns the staged result, or, for a call
  whose declared type accepts the reply's text as a string, that text (a bare
  "done" is never the text result);
- `return_result` with status `blocked` or `failed`, which ends the call without a result.

A reply without a tool call and without a result is answered with what is
missing, and the call continues. A directory reducer keeps the folder changes
that exist when it finishes. Turn, token, time, and repair limits apply only
when the caller sets them.

## Types

Signatures use TypeScript types: `string`, `number`, `boolean`, `null`,
records, arrays, `Record<string, T>` (also written `{ [key: string]: T }`),
intersections of object types (`A & { extra: string }`, merged into one record),
indexed access with a literal key (`State['status']`), literal unions, optional fields and
parameters, aliases, `Folder`, `Live<"T", kind, detail>` for host values, and
`Neuralese<T, D>` for soft values (see Neuralese). Values are checked at call
boundaries, after each eval, and at completion. Simple scalar mistakes may be
coerced when the declared type is unambiguous. A value eval computed and returns
(`return value;`, a final expression with `finish: true`) is assignable as in
TypeScript: fields its declared record does not list are kept, so a service's or a
callee's result is returned as it is; a literal written into `return_result` is
checked exactly. A value that misses a union of records is reported at the field
of the variant it matches (`checkpoint/cutoff: expected number`), not as the
whole union. Recursive function types are
rejected (see Iteration and termination).

## Is<T, P> refinement types

`Is<T, "predicate">` is `T` plus a natural-language predicate that its values satisfy:
`type Reply = Is<string, "a reply to the customer that is polite and does not blame them">`. `T` is any natlang
type; `P` is one nonempty string literal that describes the wanted values positively (a conjunction is one
sentence). `Is<Is<T, "a">, "b">` means both. Whitespace in `P` is normalized. An empty or non-literal `P` is
`refinement-predicate-invalid`, reported when the type is read.

Structure is unchanged: a value is checked against `T` as before, and a refined value is a `T` everywhere. The
predicate is an obligation checked where a value enters a refined slot:

- the return of an `nl` call or a `.nl` function, after the structural check (a failure goes back to the executing
  model as a tool error on `return_result` or its reply, and the model repairs it within the repair budget:
  `maxFailureRepairs`, else `refinements.repairs`, default 3; past it the call fails with the refinement's code);
- an argument into a refined parameter, checked at the caller (a failure throws and the callee does not start);
- a service result whose type is declared in `refinements.services`;
- `refine(value, predicate)`, which checks and returns an `Is<T, P>`, and `assume(value, predicate)`, which
  returns one without checking and records the assumption in the trace.

Records, arrays and dictionaries check each refined position; the checks of one value are issued together.

Fit: `Is<B, P>` fits `T` when `B` fits `T`; `T` fits `Is<B, P>` when `T` fits `B`, with the check as the
obligation of that site; `Is<B, P>` fits `Is<B', P'>` without a check when `B` fits `B'` and the predicates
are equal after whitespace normalization (for nested types, every predicate of the target is present).

**The check.** The judge is one `readout: decision` scoring pass on the call's model (or `refinements.judge`):
the value is shown as data, the predicate as the question, and P(true) is read from the scores of `true` and
`false`. A verdict is cached by `(sha256 of the canonical value, normalized predicate, judge id)`, so the same value
is not judged twice. The value passes at P(true) at or above `threshold` (default 0.5). Inside an optional
uncertainty `band` the `policy` applies: `accept`, `reject` (fail with `refinement-undecided`), or `escalate`
to the model named by `escalate`. A model that cannot score replies judges with an ordinary natural-language
call returning a boolean (traced as `judge: "call"`). A result eval returns or finishes, and a refined local, are
judged before they are kept, so a failure is repaired like any rejected eval. `refine<Is<T, "p">>(value)` and
`assume<R>(value)` read the predicate from the type.

**Errors.** `refinement-unsatisfied`: the predicate was judged false. `refinement-undecided`: the probability fell
inside the band under the `reject` policy, or no judge was available. `refinement-predicate-invalid`: `P` is empty
or not a string literal. Each message is one sentence that names the predicate and the fix. A call that ends on a
refinement rejects with `RefinementCallError` (a `NatlangCallError` with `code`); a rejected argument, service
result or `refine` throws `RefinementError`.

**Settings** (`natlang.json`, `refinements`): `threshold`, `band: { low, high }`, `policy`, `mode`, `judge`,
`escalate`, `repairs`, `predicates` (the same fields per normalized predicate) and `services`
(`"service.method": "Is<string, \"…\">"`). `mode` is `crisp` (default: a crisp checker that returns a boolean
decides, `undefined` defers to the judge), `nl` (judge only), or `shadow` (the judge decides; every crisp/judge
disagreement is traced as `refinement_shadow`). Crisp checkers come from the runtime option
`refinements.crisp` the `refinements` table exported by `native/types.ts`, or a `refinements.ts` module of the program that exports `refinements` (loaded by the launcher), keyed by the normalized predicate. With a call store, verdicts are kept in its `refinement_verdicts` table.

**Trace.** Every check is a `refinement_check` event: path, predicate, value, outcome, probability, judge and where
the verdict came from (`crisp`, `judge`, `cache`, `escalation`). `refinement_assumed` records an `assume`.

## Untrusted<T>

`Untrusted<T>` is `T` that came from outside the program: a file, an HTTP body, user text, a tool or service result.
It is structurally `T`; what it adds is how a model sees it and where it may go.

- **Entry.** A service method whose result type is declared in `refinements.services` as `Untrusted<...>` (or a type
  containing it, such as `{ evidence: { message: Untrusted<string> }[] }`), a parameter or captured variable typed
  `Untrusted<...>`, and `untrusted(value, source)` (exported by the runtime module; it returns the value typed
  `Untrusted<T>`). Only the strings at `Untrusted` positions are marked, so a record keeps its trusted fields as
  ordinary literals.
- **Rendering.** The one value renderer (arguments, scope, eval results, staged results) shows a marked string as a
  fenced data block with an info line naming the source, never as a bare literal and never inside instruction text:
  a line "```untrusted data from index.search", the text verbatim, and a closing fence. The fence is three backticks,
  or one more than the longest run of backticks in the text, so no text can close it. A long text is cut at the
  usual budget with the usual cut-off note. Ordinary values render as before.
- **Provenance.** The label is the origin: `service.method` for a declared service result, the label given to
  `untrusted(value, source)`, else `argument <name> of <function>` or `variable <name>`. The first source that marked a
  text is kept. Like refinement evidence it is kept by content (a string has no hidden tag): a registry per task, with a
  host-level registry behind it for text marked before any task runs, bounded in entries and characters. A string the
  program derives from an untrusted one (a slice, a concatenation) is a new string and is not tracked at run time; a
  trusted string equal to a marked one is shown as data too. Mark the exact fields that are outside text, not whole
  records.
- **Fit.** `Untrusted<B>` fits `T` when `B` fits `T`; `T` fits `Untrusted<B>` when `T` fits `B` (a plain value only
  loses trust). In TypeScript `Untrusted<T>` is `T & brand`, so it is a `T` everywhere, and a plain `T` is not an
  `Untrusted<T>` until `untrusted(value, source)` returns one. `Untrusted<Is<T, P>>` and `Is<Untrusted<T>, P>`
  carry both.
- **Instruction positions.** The text of a natlang function is the author's. In a `.nl` file the body is fixed text
  that refers to arguments by name; a value is never spliced into it, so a `.nl` body cannot receive untrusted text as
  instructions and the model reads each argument from the rendered scope. The one splice is the template literal of an
  inline `nl` call, where `${expression}` is evaluated at call time and becomes part of the instructions. There the
  compiler reports `untrusted-instruction` when the interpolated expression is `Untrusted<T>`, or is text built from
  an untrusted expression (`${message.slice(0, 20)}`, `${"[" + message + "]"}`); a number or boolean computed from one
  (`${message.length}`) is not text and passes. Names an instruction mentions (captured by mention or `nl.with`) and
  arguments of the call are not splices: they are rendered as data. The message is one sentence with the fix: pass the
  value as an argument instead, as in ``nl`Summarize the message.`(message)``. Interpolation inside eval code (a
  one-shot ``nl(`... ${x}`)``) is not typed by the project compiler and is not checked.

## Directory reducers

A directory reducer's first parameter is a `Folder` (or a handle from
`folder.dir(path)`), available as `folder` in eval. It works on an isolated
writable copy; paths are relative to the folder. `await reducer(folder, ...args)`
returns the typed result and discards file changes.
`await folder.apply(reducer, ...args)` retains the committed changes. A typed
result selects every change; `commit` selects changes by glob. Folder writers
serialize.

A reducer over a context folder is the way to compute a new context: its staged
tree, once compiled, is a file context from which new executable nodes may be
bound (see Contexts). Self-improvement is a function from a context and evidence
to a new context revision; promotion binds a program to that revision, or
`folder.apply` commits it to a real directory.

## Neuralese

`Neuralese<T, D = DefaultDialect>` is a soft value of type `T`: an immutable,
ordered block of vectors in dialect `D` that a model reads through its read
port. The model contract is in [NEURALESE_PORT.md](NEURALESE_PORT.md), the
declarations in [neuralese.d.ts](neuralese.d.ts).

**Type rules.**

- `Neuralese<T>` is not a `T` and a `T` is not a `Neuralese<T>`; moving between
  them is a computation (writing, or `read`).
- Host code may store, pass, and return soft values, but may not access fields
  or indices, compute with them, compare them, use them as conditions, spread
  them (`neuralese-opaque-access`, `neuralese-condition`). A JavaScript string
  conversion (`String(value)`, an untagged template interpolation, `+` with a
  string operand, or a direct `JSON.stringify(value)`) performs the existing typed
  `read<T>` computation first and then applies ordinary JavaScript formatting.
  For `JSON.stringify`, only its first argument is read; replacer and spacing
  arguments keep normal JavaScript behavior. The conversion is awaited
  at that expression, so it must be inside async code; synchronous functions
  and callbacks receive `neuralese-readout-sync` rather than silently returning
  promises. Tagged templates keep soft arguments as soft values.
- `T` is any natlang type. `Neuralese<Neuralese<T>>` is rejected
  (`neuralese-nested`). Records and arrays may hold soft fields.
- A `Neuralese<F>` with a function type `F` is callable with `F`'s parameters
  and result.
- Values of different dialects do not unify (`neuralese-dialect-mismatch`);
  `convert` moves between them. `DefaultDialect` is bound by configuration.
  Dialects are version tags ([NEURALESE_DIALECTS.md](NEURALESE_DIALECTS.md)).

**Literal.** At the model-token level a soft value is written
`<|neuralese|>⟦z1⟧…⟦zL⟧<|/neuralese|>`: two control tokens around the vectors and
nothing else. Its type comes from the contextual type (an annotation, a
parameter, a return position, `nl<F>`); a literal without one is a compile error
(`neuralese-untyped-literal`). When the model emits the opening token in eval
code or a tool argument, the server writes the block until its stop head closes
it. When the runtime shows the model a soft value (a parameter, a local, a
result, a soft body, a context item), it renders the literal and the server
splices the stored vectors in. The text `<|neuralese|>` in ordinary content is
text, never a literal.

**Reference form.** Everywhere outside the model (JSON, traces, logs, training
records, interfaces) a soft value is
`{ "$neuralese": { "type": "Neuralese<T>", "id": "nz1_…" } }`, referring to an
immutable, content-addressed store entry tagged with its dialect. Before type
checking eval code, the runtime replaces each emitted literal with a reference
expression. `gloss(v)` gives a diagnostic text rendering for people; it is not a
value form.

**Neuralese functions.** A function literal is `nl` with a soft body and explicit
captures:

```ts
const triage: Neuralese<(t: Ticket) => Promise<Label>> =
  nl.with({ rubric, history })`<|neuralese|>⟦…⟧<|/neuralese|>`;
```

Its signature comes from the contextual type, as for inline `nl`. Its calls go
only to its definition site's context and its captures; the vectors grant no
authority. A call renders the soft body as the instructions and lists the
captures in the opening scope. Text functions may take and return soft values;
soft arguments appear as literals in the opening declarations.

**Combinators.** `natlang:neuralese` exports `map`, `zip`, `ap`, `combine`,
`empty`, `split`, `splitList`, `read`, `convert`, and `gloss`. `read` is the only
way from a soft value to a `T`; it is validated like a call result and fails with
`NatlangCallError`. Implicit string conversions use this same `read` body and
require the task to provide a loaded standard library; a text provider may use an
explicit, digest checked implementation of the declared read source, recorded as
non-learned provenance. Without a configured read body they fail with
the structured `neuralese-readout-unavailable` capability error. The runtime
never reads vector payloads as text. `split` and `splitList` are the only way to soft parts of a
structured value. Each combinator except `empty` is a system natural-language
function with a soft body, trainable like any other; a program may bind its own
tuned bodies in its context. None takes a purpose argument: a value encodes what
its write site (the producing function's instructions, declared result, and
context) wrote it for.

**Laws and rewrites.** The combinators approximately satisfy map identity, map
fusion, read/map commutation, combine associativity and identity, and split of
zip. The compiler may rewrite programs with these laws when a rule is enabled
for the current model and dialect version
([NEURALESE_REWRITES.md](NEURALESE_REWRITES.md)); every applied rewrite is
traced.

**Files.** `.nz` files store typed named exports, exact and soft, in a
safetensors container ([NEURALESE_FILES.md](NEURALESE_FILES.md)). They are
imported like modules and are context items in callable folders.

**Learning.** `natlang:learning` exports `grad`, `valueAndGrad`, `stopGradient`,
objectives (`crossEntropy`, `selfDistill`, `conditionedDistill`, `logLikelihood`, `law`, `klPrior`, `decision`), optimisers,
`withAdapters`, `adapters.create`, and `save`, to callers given the `natlang:learning` service. `grad(f, a)`
differentiates a loss with respect to soft arguments by recording `f(a)` and
replaying it ([NEURALESE_GRAPH.md](NEURALESE_GRAPH.md)); discrete choices are
held fixed and trained through `logLikelihood`. Nested `grad` is first-order
unless `{ order: 2 }` is given. Training steps are ordinary step functions run
with `iterateOn`; learning produces new values, never mutates model weights, and
is promoted by binding a context that contains them.

**Adapters.** An `Adapter` is a soft value holding the coefficients of a tiny
weight adapter of the serving model. Its block's dialect states its structure
(kind, rank, layers, targets, base-weight hash; `model/tiny_adapters.py`), and it
runs only on that base. `withAdapters(adapters, fn)` makes adapters active for
every model turn inside `fn`: generation and decision readouts run the adapted
model, and recorded turns replay with the same adapters, so `valueAndGrad(f,
adapter)` trains an adapter like any soft value. A zero adapter is the base model.
An adapter is a value, not a change to the model: binding it is how it is
promoted, and it ships with the program that uses it
([LEARNING_CONTINUUM.md](../plans/neuralese/LEARNING_CONTINUUM.md) §6).

Writing a Neuralese value is stochastic, gated by a Neuralese temperature `τ`
separate from the text temperature: the payload is `μ + τ·σ⊙ε`, deterministic at
`τ = 0` (the inference default). `logLikelihood` includes the log-density of
sampled payloads, so encodings can be trained by sampling-based objectives as well
as by gradients through the payload; `klPrior` is the VAE-style regulariser. Stored
blocks may be Gaussian distributions ([NEURALESE_PORT.md](NEURALESE_PORT.md),
[NEURALESE_FILES.md](NEURALESE_FILES.md)).

**Backends.** A call that needs Neuralese on a backend without support fails with
`neuralese-unsupported-backend`. There is no text fallback.

## Runtime and tasks

Natural-language calls run in a task created by `runtime.run(fn)`; calls made
anywhere inside `fn` find it (Node: async context; browser: compiled code
restores the task across `await`, and `runtime.bind` wraps callbacks from
uncompiled code). A call with no task fails. Tasks run concurrently. Each
invocation produces a trace of messages, tools, actions, observations, effects,
and its outcome.

## Call records and compilations

The runtime records every call: the definition and its revision, the parent call
and the eval that started it, exact inputs and captures, the result, each service
call with its arguments and result, folder changes, the eval programs run, and
cost. Values beyond a bound are recorded by hash and type; a program may exclude
definitions or arguments, which are then recorded by type only.

A call may be served by a compilation of its definition's revision: an ordered
list of crisp cases, each a guard over the call's arguments and a body with the
function's signature, run with the function's context and services. The first
active case whose guard admits the arguments serves the call. A case that fails
or returns a value of the wrong type does not fail the call: the agent runs it,
told which effects the case already performed. A compilation never changes the
program's source, applies only while the definition's context interface is the
one it was compiled against, and is promoted or demoted by measured comparison
with the agent (plans/TRACE_SPECIALIZATION.md).

## Continuations

A long invocation may continue in a fresh model conversation. The runtime
carries the scope, staged result, child state, and folder overlay; earlier
conversation text is not copied. A short working note may carry unresolved
reasoning.
# Instruction adaptation

See [ADAPTATION.md](../docs/ADAPTATION.md) for the current TypeScript adaptation
APIs, portable artifacts, evaluation workflow, and implementation status. The
full acceptance contract and remaining gates are tracked in
[ADAPTATION_SYSTEM_IMPLEMENTATION.md](../plans/ADAPTATION_SYSTEM_IMPLEMENTATION.md).
