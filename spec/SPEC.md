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

- `eval(code)`: run TypeScript in the persistent scope.
- `read_value(expression, start?, end?)`: inspect a value.
- `mark_lines(start, end?, skipped?)`: close completed lines and untaken branches.
- `report_blocker(missing)`: stop because required information is absent.
- `report_error(message)`: stop because the operation is invalid or failed.
- `read_function`, `edit_function`, `diff_functions`: inspect and edit callable items.

A directory reducer additionally receives `list_files`, `search_files`,
`read_file`, `write_file`, `edit_file`, `diff_files`, and
`commit(value, include?, exclude?)`.

## Eval

The scope holds the parameters, captures, callable items, services, and
persistent locals. Top-level `const` and `let` declarations persist across eval
calls. An eval is atomic: a failed compilation, execution, or type check commits
no local or capture changes (effects already performed remain). Assigning
`result`, or a final expression of the declared return type, supplies the
function result; a later compatible value may replace it until completion.
Values that are not portable data (functions, class instances, handles) are
passed by reference as live values.

## Completion

Every nonblank instruction line other than a comment is substantive. A function
completes when a valid typed result exists, every substantive line is closed,
and no call, blocker, or error remains unresolved. Completion is checked after
each tool call; there is no separate return action.

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
carries the scope, result, line marks, child state, and folder overlay; earlier
conversation text is not copied. A short working note may carry unresolved
reasoning.
