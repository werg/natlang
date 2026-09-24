# natlang language specification

Version **0.4-draft**, 2026-09-23.

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
Missing or conflicting evidence is a compile error. Template interpolations are
evaluated at call time and become part of the instructions.

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

Frontmatter keys are `description`, `args`, `returns`, `types`, and `kind`
(`function` or `directory-reducer`). Application TypeScript imports a named
function as a module default export; the build generates its declaration
(`foo.d.nl.ts`).

## Callable context

A named function `foo.nl` may call exactly the items of its companion folder
`foo/`: `.nl` functions, TypeScript modules, and subfolders, each with its own
companion folder. Items appear as bindings and as properties
(`helper(...)`, `group.child(...)`). A function cannot reach natural-language
functions outside its folder; this is enforced by the compiler and runtime.

An inline `nl` inside a callable folder sees that folder's items. An inline
`nl` in application code sees the items of the nearest ancestor `natlang.d/`
folder, which follows the same rules as a companion folder. With no such folder
its context is empty. `types.ts` in a folder supplies type aliases to the
functions in and below it.

Item names are identifiers and must not collide with function properties
(`call`, `apply`, `bind`, `name`, `length`, `prototype`, `constructor`, `then`,
`iterateOn`, and similar).

## Captures

An inline `nl` captures the visible bindings its instructions mention by exact
name. Captures are read live at each call. A mentioned `let` may be reassigned
by the model: the write-back happens after a successful eval and is
version-checked, so a concurrent change fails that eval with `capture-conflict`
and the model retries it. Reassigning a captured `const` is a compile error.
Property writes on live objects take effect immediately.

## Callable-folder TypeScript

TypeScript files in callable folders are ordinary modules. A default-exported
function makes the module callable; named exports are callable attributes;
exported values are typed values. They may import sibling items, declared npm
packages, `natlang:services`, and the natlang surface module; other local files
are rejected. They follow the iteration and recursion rules below. Application
TypeScript outside callable folders is unrestricted.

## Iteration and recursion

Callable-folder TypeScript and eval code use finite iteration: `for...of`,
counted `for` loops with a checked bound, and array methods. `while`, `do`,
`for...in`, open `for(;;)`, and generators are rejected; `for...of` is guarded
at run time against iterating a growing collection. Open-ended iteration uses
`iterateOn(step, initial, ...args)` or `fn.iterateOn(initial, ...args)`, which
records each step, reviews progress, and stops on a predicate (`until`,
`streamUntil`), a limit (`withLimit`), or a `divergent` progress verdict.

A definition may not appear in its own chain of callers. This forbids direct
and mutual recursion among natural-language functions and callable-folder
TypeScript, including through callbacks and captured functions. Concurrent
sibling calls and repeated sequential calls are allowed. The compiler rejects
cycles it can resolve; a runtime guard rejects the rest before the body runs.

## Services and effects

The application supplies host capabilities as typed services when it creates a
runtime or a task. Callable-folder code imports them from `natlang:services`;
eval exposes them as named, read-only bindings. Every service call is traced as
an effect. Effects are not rolled back when a call fails; applications that must
not repeat an effect record operation identities and reconcile unknown outcomes.

## Model surface

A natural-language invocation offers the model these tools:

- `eval(code, timeout_ms?)`: run TypeScript in the persistent scope, optionally with a time limit.
- `read_page(id, page)`: read the next part of a tool output that was cut off; the cut-off names the ID.
- `compact_history(note)`: shorten the conversation (see Conversation length).
- `return_result(status, value?, reason?)`: finish the call. Status `success` returns
  `value`, of the declared type; `blocked` stops because required information is
  absent, and `failed` because the instructions require an invalid or contradictory
  operation, each with a `reason` sentence instead of a value.
- `read_function`, `edit_function`, `diff_functions`: inspect and edit callable items.

A directory reducer additionally receives `list_files`, `search_files`,
`read_file`, `write_file`, `edit_file`, and `diff_files`, and the conversation
opens with the folder's file listing.

Instructions do not all need code: an answer that takes only reading and
judgment is given directly.

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
a typed variable gives that type); when no use says it, the eval is rejected
with a diagnostic that proposes `nl<T>`.
In eval, awaiting an inline `nl` without calling it (`await nl`...``) calls it
with no arguments: it judges the names its instructions mention and its
interpolated values.
A top-level `return value` stages the value as the call's result if it has the
declared type; a later valid return replaces it. Values that are not portable
data (functions, class instances, handles) are passed by reference as live
values.

Every eval also sees `transcript`, the call's earlier tool calls with their full
outputs, unless a parameter or local takes that name. It is searched rather than
read through: `transcript.search(text or regex, { in, limit })` returns matching
lines as `{ entry, turn, tool, in, line }`, and `transcript.entry(n)` returns one
call as `{ turn, tool, code, arguments, output }` (negative `n` counts from the
end). It has no array access or iteration, and printing it shows a summary.

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
(`contextTokens`, default 16,384 prompt tokens) the next turn offers only
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
records, arrays, `Record<string, T>`, literal unions, optional fields and
parameters, aliases, `Folder`, and `Live<"T", kind, detail>` for host values.
Values are checked at call boundaries, after each eval, and at completion.
Simple scalar mistakes may be coerced when the declared type is unambiguous.

## Directory reducers

A directory reducer's first parameter is a `Folder` (or a handle from
`folder.dir(path)`), available as `folder` in eval. It works on an isolated
writable copy; paths are relative to the folder. `await reducer(folder, ...args)`
returns the typed result and discards file changes.
`await folder.apply(reducer, ...args)` retains the committed changes. A typed
result selects every change; `commit` selects changes by glob. Folder writers
serialize.

## Runtime and tasks

Natural-language calls run in a task created by `runtime.run(fn)`; calls made
anywhere inside `fn` find it (Node: async context; browser: compiled code
restores the task across `await`, and `runtime.bind` wraps callbacks from
uncompiled code). A call with no task fails. Tasks run concurrently. Each
invocation produces a trace of messages, tools, actions, observations, effects,
and its outcome.

## Continuations

A long invocation may continue in a fresh model conversation. The runtime
carries the scope, staged result, child state, and folder overlay; earlier
conversation text is not copied. A short working note may carry unresolved
reasoning.
