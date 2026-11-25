# natlang language specification

Version **0.5-draft**, 2026-10-03. This revision adds Neuralese (soft values), makes a
function's context an explicit argument, and replaces the caller-chain recursion
guard with structural termination rules.

natlang is TypeScript with natural-language functions. A natural-language
function is an asynchronous, typed function whose body is instructions; a model
executes them in a persistent TypeScript scope. Everything else is ordinary
TypeScript. The design rationale is in
[`docs/research/TS_INLINE_HOST_INTEGRATION_PLAN.md`](../docs/research/TS_INLINE_HOST_INTEGRATION_PLAN.md),
[`docs/research/inline-natlang-lambdas.md`](../docs/research/inline-natlang-lambdas.md), and
[`docs/research/ITERATE_ON_PLAN.md`](../docs/research/ITERATE_ON_PLAN.md).

## The core in one screen

`classify.nl`, a named natural-language function (YAML front matter, then instructions):

```yaml
---
description: Label one ticket.
args:
  ticket: string
returns: '"billing" | "technical" | "spam"'
---
Pick the label that fits ticket.
```

```ts
import { nl } from '@natlang/node';
import classify from './classify.nl';            // imported like a module; its type is (ticket: string) => Promise<Label>

const ticket = 'Checkout fails with a 500 error for every customer.';
const label = await classify(ticket);
const summary: string = await nl`Summarize ticket in one sentence.`(ticket);   // an inline function
```

- **Named functions** are `.nl` files: typed `args`, a `returns` type, instructions. The compiler checks every call.
- **Inline `nl`** is a tagged template whose signature the compiler infers from the call site; it captures the
  variables its instructions mention by name.
- **Callable folders.** `classify.nl` may call only what is in its companion folder `classify/`: other `.nl`
  functions and exact TypeScript helpers (`.ts`). Calls go down the folder tree, so natural-language functions cannot
  recurse (Contexts, Iteration and termination).
- **Services** are host capabilities (files, network, databases) passed to the runtime as typed objects; the model
  reaches the outside world only through them (Services and effects).
- **Types** are TypeScript types. A result is checked against its declared type before it reaches your code; a
  wrong-shaped answer goes back to the model as an error (Types, Completion).

The model executes each call in a persistent TypeScript scope with a few tools (Model surface, Eval). Everything
else in a program is ordinary TypeScript.

## Extensions

This document is the stable core. The research surface (soft Neuralese values, learning, tuned weights) lives in
versioned extensions under [`spec/ext/`](ext/); each states its version and scope, and the core does not depend on
any of them.

| Extension | Version | Covers |
| --- | --- | --- |
| [Neuralese](ext/neuralese.md) | 0.5-draft | `Neuralese<T, D>` soft values, literals, combinators, `.nz` files, backends |
| [Learning](ext/learning.md) | 0.5-draft | `natlang:learning`: `grad`, objectives, optimisers, adapters |
| [Tuned weights](ext/tuned-weights.md) | 0.5-draft | binding a program's tuned weight release to its calls |
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
rejects with `NatlangCallError` carrying `definition`, `outcome` (`quiesced` or `failed`), `detail`, the call ID
(`callId`), its `trace`, and a typed `reason`: `blocked`, `failed`, `type-mismatch`, `repeated-failed-action`,
`no-action`, `budget-exhausted`, `refinement`, `refused` or `incomplete` for the model's outcome, `cancelled` for a
cancelled call, and `error` for a failure outside the model's work (driver, arguments, host), which keeps the original
error as `cause`. `RefinementCallError` is the subclass for a result that fails a refined type.

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

Frontmatter keys are `description`, `args`, `returns`, `generic`, `types`, `kind`
(`function` or `directory-reducer`), `readout` (see Decision readout), `model`, `uses`, `withhold`, `protect`, `budget`,
`recursive` and `measure` (see Iteration and termination), `ensures` (see Postconditions) and `writes` (a directory
reducer's files, [Declared files](ext/directory-reducers.md#declared-files)). `generic` declares a
representation-generic result ([Neuralese](ext/neuralese.md#generic-results)): `generic: { R: string | Neuralese<string> }`
with `returns: R`. `model: NAME` runs every
call of the function on the runtime's model of that name (the `models` runtime option), and on the default model
when the runtime has none. `uses` lists package
items the function may call besides its companion folder: by path from the package root (the nearest folder with `natlang.json` or `package.json`, else the loaded project folder; `uses: [harness/cut]`), by
path relative to the function's file (`./shared/rubric`, `../lib/cut`, inside the package root), or by a specifier a
`natlang.json` `exports` map resolves (`@scope/pkg/item`, the package's own or an installed one's). The list form
binds each under its base name; the map form binds each under its key (`uses: { rubric: ./shared/rubric }`). Each
joins the function's context with its own companion folder; an item of the tree being loaded (a sibling, a cousin under
the same owner) is that very item, not a second copy. A `uses` chain that leads back to the function is a load
error naming the cycle; a recursive function calls itself by its own name. `withhold` lists optional tools the
function's own calls are not offered (`withhold: [delegate]`): `edit_code`, `diff_code` and the folder tools; eval,
return_result, read_page, compact_history and read_code are always offered. A withheld tool is absent from the call's
tool schema and prompt, and calling it anyway is refused with a one-sentence error. `protect` lists the items of the function's companion folder that judge its calls (verifiers, checkers, fixtures), by path from that folder (`protect: [checks/citations.ts]`; an entry without extension or a subfolder also matches): its calls can read them but `edit_code` refuses them (`protected`), as it refuses `types.ts` and an edit that changes a `protect` list. An entry that names no item is a load error. `budget` bounds the model work of each of
the function's calls in place of its model's own bounds: a map of `maxTurns`, `turnTokens` (reply tokens of one turn),
`maxTokens` and `contextTokens` to positive whole numbers (`budget: { maxTurns: 8, turnTokens: 1024 }`); a call that
exhausts it fails with reason `budget-exhausted`. No budget applies by default: termination is structural. `args`, `returns`, `generic` and `types` hold
TypeScript type text, read verbatim rather than as YAML, so types need no
quoting: `rows: { title: string }[]`, `pick?: (x: string) => number`,
`returns: "yes" | "no"`. `args` and `types` are one `name: type` per indented
line (a type may continue on more deeply indented lines) or an inline object
type, `args: { q: string, n?: number }`. A value wrapped entirely in quotes is
the quoted text, as in YAML, so `"string[]"` is `string[]`; a lone string
literal type is written `'"yes"'`. The other keys are YAML. Application TypeScript imports a named
function as a module default export; the build generates its declaration
(`foo.d.nl.ts`). A `uses:` item that is itself a named function of the build is
referenced there by an `import type` of its own declaration, so its type aliases
stay its own (`NATLANG_INLINE_USES_DECLARATIONS=1` inlines it instead).

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
interface (`context-interface-mismatch`). An inline `nl` value rebound this way
keeps its new context when `.with(...)` instantiates it again. Rebinding selects skills, evaluates
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

A program may also carry a tuned weight release, bound per callee; see the
[tuned weights extension](ext/tuned-weights.md).

Item names are identifiers and must not collide with function properties
(`call`, `apply`, `bind`, `name`, `length`, `prototype`, `constructor`, `then`,
`iterateOn`, `in`, `with`, and similar).

## Captures

An inline `nl` with a text body captures the visible bindings its instructions
mention by exact name, as an ordinary word too ("could not be played" captures
a variable `played`), and the bindings its interpolations use. `natlang check`
warns (`nl-word-capture`) when every mention of a captured name is a word in
verb position; quote the name in backticks if the call needs it, otherwise
reword or list the captures with `nl.with`. A capture is declared to the call
with its binding's type; a binding inferred from literals (`const aim = k ?
'a' : 'b'`) is declared as its base type (`string`), as for `let`. Data captures are read live at each call. A mentioned
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
exported values are typed values. They may import sibling items, other `.ts`,
`.js` and `.nl` files of their own package by relative path (inside the package
root), exports a `natlang.json` `exports` map names, declared npm packages,
`natlang:services`, and the natlang surface module; a relative import that leaves
the package root is rejected. `import type` of a package file (`import type { Lot }
from '../types.js'`) types the module's code, including the parameters of an inline
`nl` it builds, and is erased at run time. They follow the iteration and termination rules below. Application
TypeScript outside callable folders is unrestricted.

## Iteration and termination

Callable-folder TypeScript and eval code use finite iteration: `for...of`,
`for...in` (an object's enumerable string keys, inherited ones included), counted `for` loops (the condition may join the counter bound to an early exit with `&&`, as in `i < n && !found`), and array methods. A counted loop reads its bound once,
when it starts, so the bound may be a call (`p <= store.pages()`, or `p <= await store.pages()` for an async one); the bound must be a finite number and the counter must advance
toward it. `while`, `do`, open `for(;;)`, generators, and code that
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

Natural-language functions recurse only when they declare it, and then structurally:

- A `.nl` function with `recursive: structural` in its frontmatter sees itself
  under its own name in its calls' eval scope and may call itself on a smaller
  argument, by the rule for TypeScript functions below. An argument a call
  receives is a copy, so a part also counts when it is plain data equal to a
  proper part of the argument (its size is smaller, so the chain still ends).
- With `measure: <expression>` instead, a TypeScript expression over the
  arguments, a self-call is admitted where the expression's value, a
  non-negative whole number, is smaller than in the call it runs below
  (`measure: "items.length - done"`). A value that is not a non-negative whole
  number fails the call.
- Any other re-entry of a definition fails when it happens
  (`NatlangRecursionError`): without a declaration, "called itself, and its
  frontmatter does not declare recursion"; on an argument that is not smaller,
  "called itself without a smaller argument"; with a measure that does not fall,
  "called itself with measure M, which is not smaller than N".
- Apart from that, calls step down the context graph (see Contexts), which is
  acyclic. A definition rebound to another context is another function;
  rebinding is a host capability.
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

What this guarantees: code in eval and callable folders, and `.nl` functions, cannot loop or recurse
without bound by themselves. The deliberate exceptions are model-controlled: an
`iterateOn` whose stopping predicate is a natural-language function, and the
agent loop of a call, which has no turn or time limit unless the caller sets
one. Services and imported packages are the host's and may run as long as they
do. The rules address code written in good faith; reaching around them through
reflection is outside them, and the Node backend is not a sandbox. An application
runs as long as outside events (users, requests, the clock) keep arriving, each
handled in bounded work.

## Postconditions

`ensures` in a `.nl` function's frontmatter lists postconditions that relate its result to its arguments:

```yaml
ensures:
  - check: "result.quotes.every(quote => hunk.includes(quote))"
    message: every quote is copied from hunk
  - predicate: a faithful summary of hunk
    against: [hunk]
```

- A `check` is a TypeScript expression over `result`, the arguments and the items of the function's companion folder
  (by name), and standard globals; it holds when its value is `true` (an `await`ed helper is fine). Checks run on
  read-only copies, under the same finite-iteration rules as callable-folder TypeScript. `message` says what must
  hold, as a clause about the result.
- A `predicate` is a property of the result, judged like an `Is<T, P>` refinement (see
  [Refinements](ext/refinements.md)), with the arguments listed in `against` shown to the judge by name as context. An
  argument whose text came from outside the program (`Untrusted<T>`) is shown fenced, as data.
- Checks run first; predicates are judged only when every check holds. A failure is a `refinement-unsatisfied` (or
  `refinement-undecided`) of the return value: the call gets it as the result of its `return_result` or finishing
  eval, with the clause's message, and repairs within the refinement repair budget; past it the call fails with
  `RefinementCallError`. A result that reaches the caller satisfies every postcondition.
- The companion-folder items a `check` names judge the call: its `edit_code` refuses them (`protected`), as it refuses
  an edit that changes the function's `ensures`, `writes`, `measure` or `recursive` lines.
- A load error names an expression that does not parse, a name a `check` or `measure` uses that is not `result`, an
  argument, a companion item or a standard global, and an `against` entry that is not an argument.

`refine(value, predicate, { context: { draft } })` judges a relational predicate the same way from code.

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
  built-ins (`nl`, `iterateOn`, `transcript`, `decide`, `NatlangContext`, the package's
  `Context` under a name no program type takes, and the std combinators; see Eval). `read_code` is offered on every call;
  `edit_code` and `diff_code` where the call has code of its own and edits are not denied.
  An edit is a revision of the run's program view, never of the files: a call that
  succeeds records its revisions in its call record (`code_revisions`), and
  `natlang improve adopt-call CALL` installs them only after `natlang check` and
  `natlang test` pass on the proposed source and no protected item changed.

A [directory reducer](ext/directory-reducers.md) additionally receives `list_files`, `search_files`,
`read_file`, `write_file`, `edit_file`, and `diff_files`, and the conversation
opens with the folder's file listing.

Instructions do not all need code: an answer that takes only reading and
judgment is given directly.

**Decision readout.** A function whose result type is finite (a union of
literals, `boolean`, `null`; a record whose fields are all finite and none optional; a union of such records,
tagged or not) may declare `readout: decision`. Its call then
offers no tools. The model first replies to the readout's request (the call's opening, signature and arguments,
under a short system prompt) with its own reasoning, `decisionSamples` times (default 4, drawn at the call's
temperature; one at temperature 0); the runtime then scores every allowed value as the whole reply after each
reasoning and returns the value with the highest mean probability. A value's probability is that mean: the model's
chance of answering with the value when it reasons first, as it does when it generates. The scores are the model's own
log-probabilities (not tempered); temperature only shapes the reasoning drawn. `readout: instant` scores the values
right after the request, without reasoning: one pass for latency-critical calls, whose probabilities describe the
model answering without reasoning (`decisionSamples: 0` makes every readout instant). `decide(...)` reports
`reasoned`, the number of reasoning samples, when the call reasoned. The normalised distribution
is the call's trace event `decision_readout` and its note, so a decision is a
probability vector that proper scoring rules (Brier, ranked probability score,
log loss) can grade (the [learning extension](ext/learning.md) also differentiates them). Code reads it with
`decide(fn, ...args)`: eval code, and host code inside a running task (`import { decide } from '@natlang/node'`);
`runtime.decide(fn, ...args)` runs it in a task of its own. Both return
`{ value, probabilities: [{ value, probability }], confidence, scored }`, and for a record result also
`fields: { [field]: [{ value, probability }] }`. A record type of at most 32 values is scored jointly, each record (compact
JSON, fields in declaration order) one reply, and `fields` holds each field's marginal. A larger one is scored field by
field: the prompt shows the type's forms with each field's values written `<A | B>`, and each step scores the replies
that continue the fields chosen so far through the next field, as the start of the reply; `fields` holds each step's
distribution given the earlier choices, `probabilities` the chosen record alone, and `confidence` its probability (the
product of the steps). A backend that cannot score open replies generates the record instead. A model config with
`decisionReadout: 'finite-returns'` applies the readout to every finite-typed
call. A backend that cannot score replies generates one offered JSON value on the same short decision messages,
without tools. This is a sampled choice with `scored: false`, not a score distribution. A missing, truncated, malformed
or unoffered choice fails with `decision-generation-invalid`; return one complete offered JSON value. The model setting
`decisionGeneration: 'tool-loop'` explicitly restores the prior unscored tool loop for controlled diagnosis.
The loader rejects `readout: decision` and `readout: instant` on a type that is not finite.

## Eval

The scope holds the parameters, captures, callable items, services, and
persistent locals. Parameters are `const` and deeply frozen; derive a new
variable instead of changing one. A `let` capture's assignments are written
back to the caller. Top-level `const` and `let` declarations persist across eval
calls, and so does object identity, as in one JavaScript scope: an object two
locals hold, or a part of a parameter a local holds, is the same object in later
evals (`needs.indexOf(e)`, `item !== hit`), and a parameter's part stays frozen
through the local. The conversation opens with the runtime's own eval: ambient `declare`
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
parameters, recursive ones such as `type Node = { kids: Node[] }` included)
annotate its locals like the call's own types, in that eval and in later ones; a type the runtime cannot express leaves a local so annotated open.
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
An eval's memory is bounded where the host can measure it (Node: 256 MB of heap
growth that survives a full collection, the setting `NATLANG_EVAL_MEMORY_MB`, 0
for none). An eval past its bound fails like any failed eval with "eval stopped:
its data grew past the N MB memory limit of one eval; use a bounded or more
memory-efficient approach (prune the search, keep compact state, or process the
data in parts)"; the host and other evals keep running. The same setting bounds
each synchronous run of callable-folder module code, each folder `python` cell
(growth of the interpreter's WASM memory; Python sees MemoryError and the cell
fails with "python stopped: ..."), and each folder `bash` command (its strings,
inputs, outputs and the bytes it writes; "bash stopped: ..." follows the shell's
own limit message). Browser evals are not bounded: a page has no synchronous,
precise heap reading, and a runaway there ends only that user's tab.
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

Every eval also has the standard library's combinators by name, where the platform
serves the library (Node and browsers do): `semanticMap`, `semanticFilter`,
`semanticPartition`, `semanticReduce`, `semanticMatch`, `semanticRank`,
`semanticRepair`, `fixedPoint`, `semanticMerge`, `escalate`, `withContext` and
`compileRule`, and in a call with a folder `proposeAndVerify` and
`branchAndSelect`. They are the same functions callable-folder TypeScript imports
from `@natlang/node/std`, typed by its declarations, and run in the call's frame:
their `std:<name>` spans and the calls they make nest under the call. An input,
local, function, service, capture or declaration of the same name takes the name
instead. An eval `import { name } from '@natlang/node/std'` (or the browser or core
specifier) of one of them reads as the built-in; any other import from the library
fails with one sentence naming the built-ins. `read_code(name)` documents each.
The setting `NATLANG_EVAL_STD=0` leaves them out.

## Cut-offs

Whatever is too long to show is shortened one way, with a note that is not
TypeScript: `<<cut off: 338 of 340 items not shown; only this display is shortened, customers holds the complete value>>`.
A value whose whole literal is small (300 characters) is never shortened, also in one-line status texts.
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

A call finishes with `return_result` and status `success`, supplying a value of the declared type, or
omitting `value` to use the complete result staged by an earlier eval. An eval with `finish: true` completes the
call with its fresh typed final expression or explicit return. `return_result` with status `blocked` or `failed`
ends the call without a result.

A reply without a tool call does not complete the call, even when its text is a valid value or a result is staged.
The runtime tells the model to call `return_result` and continues. A directory reducer keeps the folder changes
that exist when it finishes. Turn, token, time, and repair limits apply only when the caller sets them.

A reply cut off at the request's token limit before any tool call can continue as the same turn when the backend
supports assistant prefill (the managed llama.cpp server). The continuation preserves the reply's text and reasoning;
it consumes the call's token budget and must fit in the context window. Other backends receive the cut-off reply
and a request to take the next step with a tool call in a new turn.

## Types

Signatures use TypeScript types: `string`, `number`, `boolean`, `null`,
records, arrays, `Record<string, T>` (also written `{ [key: string]: T }`),
structural intersections (`A & { extra: string }`, merged into one record, with overlapping fields narrowed conjunctively),
`never` for an impossible intersection,
indexed access with a literal key (`State['status']`), literal unions, optional fields and
parameters, aliases, `Folder`, `Live<"T", kind, detail>` for host values, and
`Neuralese<T, D>` for soft values (see [Neuralese](ext/neuralese.md)). Built-in host classes (`Date`, `Map`,
`URL`, `Request`, `Response`, ...) are checked as live values, but a program's own alias of the same name
shadows the built-in in its scope, as in TypeScript. Indexing a union selects the field
from each variant and requires the field to be available on every variant. Values are checked at call
boundaries, after each eval, and at completion. Simple scalar mistakes may be
coerced when the declared type is unambiguous. A value eval computed and returns
(`return value;`, a final expression with `finish: true`) is assignable as in
TypeScript: fields its declared record does not list are kept, so a service's or a
callee's result is returned as it is; a literal written into `return_result` is
checked exactly. When the declared result is a record, array or dictionary and
`return_result` receives a string, the string is read as JSON or as a JS
object/array literal (parsed, never run) and its data is checked instead; a
string that is neither is refused with the shape to write. A value that misses a union of records is reported at the field
of the variant it matches (`checkpoint/cutoff: expected number`), not as the
whole union. `Record<K, T>` whose key alias `K` is a union of string literals is the record with exactly those
fields. A function type `(x: A) => R` is the type of a function a caller awaits (`=> Promise<R>` is the same type).
Every natlang function carries its signature; a function value at a function-typed slot must fit it, parameters by
position (each takes what the slot passes there; a function may take fewer) and its result covariantly, or it is
refused with `function-signature-mismatch`. A function without a signature (TypeScript code) is wrapped at the slot so
that each call checks its arguments and its awaited result against the slot's types. `fn.partial(...values)` binds a
natlang function's first parameters, checked at once, and returns a natlang function of the rest that keeps `on`, `in`
and `iterateOn`. An alias of `types.ts` that this grammar cannot read is reported where a signature uses it.
Recursive function types are
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
A function value may outlive the invocation that created it while its owning task
remains live. Its lexical captures and service authority remain those of that
owner; calling it never adopts ambient services. Its causal parent is the actual
live caller when authenticated to the same recorder, otherwise it is an independent
root in the owning task. A completed creator is never a causal parent. A function
whose owning task has closed or been cancelled cannot be invoked implicitly in a
new task. The same rules apply to callbacks wrapped with `runtime.bind`.
A function value created by the top-level code of a callable-folder module whose
instance all tasks of a platform share (no task isolation) belongs to the module, not
to the task that happened to load it: it has no owning task, and each call runs in
its caller's task.

While a task runs, a host may subscribe to its live events (`onLive`, on the runtime or per task): each
`model_delta` event carries a piece of one of a call's model turns as it streams (text, reasoning, a tool-call
fragment, a written Neuralese block, or `reset` when the turn's request is sent again and what was shown of it is
void), tagged with the task, the call ID, the caller's call ID, the function's name and definition, and the turn
number its trace's `model_request` events carry. `call_start` and `call_end` (with the outcome and its detail) bracket
each call, and `trace` events carry the call's trace events as they are recorded (model requests with their token
counts, actions, failures; not scope snapshots). Live events are hints for showing a call at work; the trace and
the call's records do not depend on them, and without a subscriber the runtime does no live work.
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
   this code. The runtime raises it at `.nz` loading and wherever a value is coerced to a `Neuralese` type (the
   block's stored dialect against the type's, `DefaultDialect` bound to the runtime's reader dialect). A native
   state of another dialect at a `DefaultDialect` slot is instead rejected with `neuralese-native-consumer-unavailable`
   unless the reader declares a consumption mode that can take it (`native/values.ts`). The compiler has no check of
   either name.
4. **`readout: decision` arity.** The text says a finite result type; the loader (`runtime/loader.ts`) also
   requires at least two values, so a single-literal or `null`-only return is rejected.
5. **`iteration-unbounded` and `iteration-measure-exhausted`.** Both are carried by `IterationLimitError` (the text
   says so only for the second); `iteration-unbounded` is thrown at run time, after the initial state fails the predicate and before the first step (a loop whose initial state already satisfies it returns without error), not when a limit is reached.
