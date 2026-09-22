# natlang language specification

Version **0.3-draft**, 2026-09-22.

A natlang program is a set of typed functions. A function may contain TypeScript
or natural-language instructions. Natural-language functions are executed line
by line by a model in a persistent TypeScript scope.

## Functions

A function declares ordered parameters and one return type. Calls use ordinary
positional syntax and are asynchronous:

```ts
const summary = await summarize(document, rubric);
```

Crisp helpers are ordinary TypeScript modules with default-exported functions.
Their signatures use the same source types as natural-language functions.

Each call receives private parameter bindings and local state. Parameters are
ordinary mutable local bindings. Reassigning or mutating them affects the
current call only.

Crisp and natural-language functions share the same TypeScript value model and
call syntax. A crisp function returns a value normally. A natural-language
function returns when an `eval` call ends with a value that fits its declared
return type and every substantive instruction line has been closed.

## Model surface

Every natural-language function receives these tools:

- `eval(code)`: execute TypeScript in the persistent scope.
- `read_value(expression, start?, end?)`: inspect a scope value.
- `mark_lines(start, end?, skipped?)`: close completed lines and untaken
  branches.
- `report_blocker(missing)`: stop because required information is absent.
- `report_error(message)`: stop because the requested operation is invalid or
  failed.
- `read_function`, `edit_function`, and `diff_functions`: inspect and improve
  imported functions.

A directory reducer additionally receives `list_files`, `search_files`,
`read_file`, `write_file`, `edit_file`, `diff_files`, and
`commit(value, include?, exclude?)` for its writable input folder.

No other model-facing execution protocol exists.

## Eval

The runtime injects function parameters, persistent locals, and imported
functions. A directory reducer declares its `Folder` as an explicit first
parameter; in eval that parameter is available as `folder`. The reducer also
receives the path-oriented `fs` helper. Normal functions do not receive folder
or filesystem tools.

Top-level `const` and `let` declarations persist across eval calls. Parameter
assignment and mutation persist within the current function call. Eval is
atomic: failed compilation, execution, host calls, or type validation do not
commit parameter or local changes.

A final expression or top-level return is displayed as the eval tool result. If
that value fits the function return type, it also becomes the function result.
A later compatible eval value may replace it until the function completes.

Imported natlang and crisp functions are called with `await` and positional
values. Collection control flow uses normal TypeScript loops, array methods,
and `Promise.all`.

## Instruction lines and completion

Every nonblank instruction line other than a function signature or comment is
substantive. The model closes a line only after its work succeeds. An untaken
conditional branch is closed with `skipped=true`.

A function completes after:

1. a valid typed result exists;
2. every substantive line is closed; and
3. no child call, blocker, or error remains unresolved.

Blank lines and comments never block completion. Completion is checked after
each tool call; no separate done or return action exists.

## Files

Imported functions are exposed by name. `read_function`, `edit_function`, and
`diff_functions` provide source editing without presenting the codebase as an
unrelated filesystem. Existing function source may be changed and validated
edits become live at the next call boundary. The function set is fixed: source
files cannot be created, removed, renamed, or moved.

A directory reducer receives an isolated writable copy of its input folder.
All model-facing paths are relative to that folder, without a prefix. Files may
be created, changed, moved, or removed. Only directory reducers can call other
directory reducers. Calling `await reducer(someFolder, ...args)` passes the
folder explicitly, returns its typed value, and discards its folder changes.
Calling `await someFolder.apply(reducer, ...args)` applies it to an
automatically forked view rooted at that folder handle and retains the
committed changes there. A handle from `folder.dir("path")` therefore
delegates only that subdirectory, for example
`await folder.dir("packages/api").apply(reducer, input)`. The child receives
that selected folder as its first parameter and as `folder` in eval.

An eval expression that sets the typed result selects every change. `commit`
sets the same typed result while optionally selecting changes with relative
include and exclude glob patterns. Folder writers use a
semaphore: conflicting writes wait for the active reducer or fail if the active
operation fails.

## Types

Authored types use ordinary TypeScript: `string`, `number`, `boolean`, `null`,
records, arrays, `Record<string, T>`, named aliases, literal unions, `Folder`,
and `FileHandle`.
Values are checked at function boundaries, after eval transactions, and before
completion. Simple quoted scalar mistakes may be coerced when the intended
declared type is unambiguous. Invalid proposals remain invalid and are recorded
as such.

## Errors and effects

A failed imported call bubbles to its caller. The calling model may change its
approach or report an error; the runtime does not repeatedly coach it into a
different answer.

Declared external effects are journaled in order. A failed acknowledgement does
not prove that an effect did not occur. Retry policy belongs to the application
and must use operation identities where duplicate effects matter.

## Continuations

Long functions may continue in a fresh model conversation. The runtime carries
typed parameters, locals, result, line marks, child state, file overlays, and
effect journals. Earlier conversation text is not copied. A short working note
may carry unresolved reasoning that is not already represented in runtime
state.

## Directory reducers

A directory reducer has the same parameters, eval scope, imports, line marking,
and error behavior as any other natlang function. It additionally receives the
writable folder API and `commit`, because it may retain a folder patch alongside
its typed result. The reducer prompt defines the complete folder API and reducer
call behavior; normal lambdas are not shown reducers or these capabilities.
