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

## Extensions

This document is the stable core. The research surface lives in versioned
extensions under [`spec/ext/`](ext/neuralese.md); each states its version and
scope, and the core does not depend on any of them.

| Extension | Version | Covers |
| --- | --- | --- |
| [Neuralese](ext/neuralese.md) | 0.5-draft | `Neuralese<T, D>` soft values, literals, combinators, `.nz` files, backends |
| [Learning](ext/learning.md) | 0.5-draft | `natlang:learning`: `grad`, objectives, optimisers, adapters |
| [Refinements and trust](ext/refinements.md) | 0.5-draft | `Is<T, P>`, `refine`/`assume`, `Untrusted<T>`, `refinements` settings |
| [Call records](ext/call-records.md) | 0.5-draft | recording, compilations, specialization |
| [Directory reducers](ext/directory-reducers.md) | 0.5-draft | `Folder` reducers, `folder.apply`, context revisions |
| [Continuations](ext/continuations.md) | 0.5-draft | continuing a long invocation in a fresh conversation |
| [Instruction adaptation](ext/instruction-adaptation.md) | pointer | adaptation APIs, defined in `docs/ADAPTATION.md` |

Error codes, the rule behind each, and when it fires are indexed in
[ERRORS.md](ERRORS.md).

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
(`function` or `directory-reducer`), `readout` (see Decision readout), `model`, and `uses`. `model: NAME` runs every
call of the function on the runtime's model of that name (the `models` runtime option), and on the default model
when the runtime has none. `uses` lists package
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
tree, imported libraries, and staged trees written by a [directory reducer](ext/directory-reducers.md) and
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
which cannot be scanned for names (see [Neuralese](ext/neuralese.md)), and allowed for text bodies.
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

A service may come with a declaration (the `serviceDeclarations` runtime option): the TypeScript declaration of its members,
with their doc comments, as a declaration file would give them. The model is
shown it as `declare namespace name { … }` and can read it with `read_code`,
but the implementation runs in the host and is neither readable nor editable. A
task gives external modules (a board, a simulated world, a store, a checker) this
way, so that only code the program owns can be changed. Importable packages are
external in the same way: `read_code("pkg")` lists a package's exports from
its type declarations, and `read_code("pkg.name")` shows one with its docs.

A service can be scoped to functions (`serviceScopes`): it is then usable only in their calls and
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

A [directory reducer](ext/directory-reducers.md) additionally receives `list_files`, `search_files`,
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
(`objectives.decision` in `natlang:learning`, [learning.md](ext/learning.md)). Host code reads it with
`runtime.decide(fn, ...args)` and eval code with `decide(fn, ...args)`, which return
`{ value, probabilities: [{ value, probability }], confidence, scored }`. A model config with
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
`Neuralese<T, D>` for soft values (see [Neuralese](ext/neuralese.md)). Values are checked at call
boundaries, after each eval, and at completion. Simple scalar mistakes may be
coerced when the declared type is unambiguous. A value eval computed and returns
(`return value;`, a final expression with `finish: true`) is assignable as in
TypeScript: fields its declared record does not list are kept, so a service's or a
callee's result is returned as it is; a literal written into `return_result` is
checked exactly. A value that misses a union of records is reported at the field
of the variant it matches (`checkpoint/cutoff: expected number`), not as the
whole union. Recursive function types are
rejected (see Iteration and termination). The refinement and trust types
`Is<T, P>` and `Untrusted<T>` belong to the
[refinements extension](ext/refinements.md).

## Runtime and tasks

Natural-language calls run in a task created by `runtime.run(fn)`; calls made
anywhere inside `fn` find it (Node: async context; browser: compiled code
restores the task across `await`, and `runtime.bind` wraps callbacks from
uncompiled code). A call with no task fails. Tasks run concurrently. Each
invocation produces a trace of messages, tools, actions, observations, effects,
and its outcome.
Recording and compilation of calls are the [call records extension](ext/call-records.md); resuming a long
invocation is [continuations](ext/continuations.md).

## Known discrepancies

Places where this document, the skills and the code disagree. They are recorded here, not resolved; each
needs an owner decision. Found while splitting the specification (2026-10-09).

1. **Quoting of frontmatter types.** The code (`ts-host/src/runtime/loader.ts`) and the frontmatter paragraph above
   read `args`, `returns` and `types` as TypeScript type text, so types need no YAML quoting. The authoring skill
   (`skills/natlang-authoring/references/language.md`, "Named functions and callable folders") says to quote YAML
   type strings that contain record syntax or YAML punctuation. Quoting is harmless (a fully quoted value is its
   contents) but the skill states it as a requirement.
2. **Reserved child names.** The list above names `constructor`, `in` and `with`; the skill's list omits them.
3. **`neuralese-dialect-mismatch`.** The Neuralese extension says values of different dialects do not unify, with
   this code. The only place that raises the code is the `.nz` file loader (`native/nz-file.ts`, when a file entry's
   dialect differs from its declared type). The compiler has no check of that name.
4. **`readout: decision` arity.** The text says a finite result type; the loader (`runtime/loader.ts`) also
   requires at least two values, so a single-literal or `null`-only return is rejected.
5. **`iteration-unbounded` and `iteration-measure-exhausted`.** Both are carried by `IterationLimitError` (the text
   says so only for the second); `iteration-unbounded` is thrown at run time, after the initial state fails the predicate and before the first step (a loop whose initial state already satisfies it returns without error), not when a limit is reached.
6. **Code comments cite the old layout.** `ts-host/src/neuralese/*.ts` and `ts-host/src/runtime/contexts.ts` cite
   "spec/SPEC.md, Neuralese chapter"; that text is now `spec/ext/neuralese.md` and `spec/ext/learning.md`.
