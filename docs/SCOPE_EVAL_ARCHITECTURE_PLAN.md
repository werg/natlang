# Typed scope and `eval` orchestration

Status: proposed architecture, 2026-09-22.

This plan replaces the model-facing workspace-path and call-combinator protocol with a
persistent typed execution scope. The model interprets the natural-language program one
line at a time, executes ordinary TypeScript-like statements with `eval`, observes their
results, and explicitly closes every substantive source line. Natural-language lambdas,
crisp functions, and host-backed functions are imported and called through the same syntax.

The existing typed tree, pending nodes, reduction engine, effect journal, trace system, and
durable continuation machinery remain the execution substrate. The proposal changes how the
model expresses its decisions. It does not move program interpretation into deterministic
orchestration.

## 1. Goals and fixed decisions

The design must:

1. Encourage the model to execute natural-language instructions in source order, normally one
   substantive line at a time.
2. Require the model to mark every substantive instruction line `done` or `skipped` before a
   lambda can complete. Line closure remains an explicit model judgment.
3. Present ordinary lexical variables, assignment, control flow, function calls, and `return`
   expressions instead of slash paths and call destinations.
4. Put imported natural-language and crisp functions directly in lexical scope. Both use the
   same call syntax and may be awaited.
5. Let `eval` return a value to the model without completing the enclosing lambda. A final
   expression and an explicit `return` inside an eval snippet both produce a tool result.
6. Complete the enclosing lambda only through a typed value already present in scope.
7. Preserve explicit `report_blocker` and `report_error` exits and caller-level validation
   failure by default.
8. Keep scope value operations and codebase file operations unmistakably separate.
9. Give authorized developer agents a complete, familiar file editing workflow: discovery,
   search, reading, creation, exact or fuzzy editing, patches, moves, deletion, validation,
   and test execution.
10. Use static, revision-pinned imports. Editing a source file never changes bindings in an
    already-running lambda.
11. Serialize scope, suspended calls, marks, notes, effects, and the eval program counter so
    interruption and fresh-context continuation remain safe.
12. Keep Python and TypeScript behavior conformant while allowing each implementation to follow
    its language's internal naming conventions.
13. Define a distinct directory-reducer lambda subtype. It is applied asynchronously to a folder,
    receives an automatically forked folder at `project/`, can inspect and locally edit its module
    snapshot at `codebase/`, and can commit only selected `project/` changes.
14. Permit the same directory reducer to be called directly as
    `await reducer(folder, ...args)` when the caller wants only its typed value. Direct calls always
    discard the private project fork; only `folder.apply(reducer, ...args)` can retain changes.

## 2. Model-facing naming

There are two layers with different conventions.

### 2.1 Tool protocol

Tool and argument names use `snake_case`:

```text
eval
read_value
write_value
return_value
commit
mark_lines
report_blocker
report_error
read_file
edit_file
```

### 2.2 Code evaluated by `eval`

The evaluated language follows JavaScript and TypeScript conventions. Runtime-provided code
APIs use `camelCase`; native JavaScript APIs retain their standard spelling. Program-authored
exports and fields are preserved exactly. New generated codebases should use `camelCase` for
TypeScript-facing identifiers.

```ts
const nextState = await advance(state, chosenTask);
const grouped = groupBy(events, event => event.customerId);
const results = await Promise.all(items.map(item => classify(item)));
```

Python source uses idiomatic Python internally and TypeScript source uses idiomatic TypeScript.
Implementation identifiers do not leak into the model protocol.

## 3. Surface profiles

Tools are grouped by stable authority profiles. A profile does not change merely because a local
appeared or a line was marked; stable profiles reduce schema churn and make authorization clear.

### 3.1 Interpreter profile

Every natural-language interpreter episode receives:

```text
eval(code)
read_value(expression, start?, end?)
write_value(name, value, as_type?)
return_value(variable)
mark_lines(start, end?, skipped?)
report_blocker(missing)
report_error(message)
```

The ordinary path is `eval`, `mark_lines`, and finally `return_value`. `read_value` exists for
large, lazy, or truncated values. `write_value` exists for direct typed literal transfer; it is
not the preferred way to perform normal assignment.

### 3.2 Directory-reducer profile

A directory-reducer lambda receives the core interpreter tools, including `return_value`, plus:

```text
list_files(path?, pattern?)
search_files(query, path?, pattern?)
read_file(path, start_line?, end_line?)
write_file(path, content)
edit_file(path, find, replace_with, fuzzy?)
diff_files(path?)
commit(value, include?, exclude?)
```

It receives an injected `fs` binding inside eval and crisp code. Both `project/` and `codebase/`
are readable and locally editable. Only `project/` changes are eligible for `commit`; `codebase/`
edits are invocation-local and retained in the audit trace. This profile is fixed by the lambda
subtype, so its tool list does not fluctuate during the episode.

### 3.3 Developer profile

An agent authorized to modify a codebase additionally receives:

```text
apply_patch(patch)
move_file(source, destination)
delete_file(path)
validate_codebase(paths?)
run_program(entry, inputs?)
commit_files(paths?, message?)
```

The developer profile extends the directory file operations for agents authoring source outside a
directory-reducer invocation. File tools operate on files, never on execution-scope variables.

Read-only modules or developer tasks receive only the applicable discovery and read operations.
The distinction is declared module and host authority, not a prompt request.

File tools address the current execution's scoped filesystem view. `commit_files` is offered only
when the host grants authority to modify the backing working tree; ordinary runtime file writes
remain local to the execution overlay.

## 4. Interpreter tools

### 4.1 `eval(code)`

`eval` executes a TypeScript-like snippet in the lambda's persistent typed scope.

```json
{
  "type": "function",
  "function": {
    "name": "eval",
    "parameters": {
      "type": "object",
      "properties": { "code": { "type": "string" } },
      "required": ["code"],
      "additionalProperties": false
    }
  }
}
```

Semantics:

- Declarations and successful assignments persist across model turns and continuation segments.
- Inputs and imported bindings are immutable. Local `const` and `let` bindings behave normally.
- A final expression is returned as the tool result.
- `return expression` inside the snippet also returns that value as the tool result.
- An eval result does not write the enclosing lambda's return slot and does not complete it.
- A declaration-only snippet succeeds with no displayed value.
- The runtime infers and records local types. An explicit TypeScript annotation may narrow or
  disambiguate a new binding.
- Calls to imported functions use ordinary positional arguments. Every imported callable may be
  awaited; awaiting an immediate crisp result is valid.
- The model does not select an execution engine. The loaded target supplies a conformant engine.
- `eval`, dynamic `import()`, filesystem globals, network globals, process globals, prototype
  mutation, and undeclared capabilities are absent from the evaluated language.
- Host validation failures, child blockers, and type errors cannot be caught and suppressed by
  model code. They become structured eval failures and bubble to the caller under the configured
  validation policy.

Examples:

```ts
const ready = await readyTasks(state);
ready
```

returns a preview of `ready` and persists the binding.

```ts
const noneReady = ready.length === 0;
return noneReady;
```

returns a Boolean tool result. It does not finish the lambda.

The model should normally use one eval snippet for one natural-language instruction line. A
single source line may legitimately contain an authored bulk operation or bounded loop; the model
may express that operation in one snippet rather than unroll it across model turns.

### 4.2 `read_value(expression, start?, end?)`

`read_value` inspects execution state without executing arbitrary code.

```text
read_value(expression="ready")
read_value(expression="state.tasks", start=0, end=9)
read_value(expression='record["__proto__"]')
```

`expression` accepts only a root scope identifier followed by field or index selection. It cannot
call functions, assign values, or compute. Ranges address list items or text lines. The tool is
useful when the opening view or an eval result was truncated, and for host-backed lazy values that
cannot safely enter the crisp evaluator.

This tool never accepts a file path. Source text belongs to `read_file`.

### 4.3 `write_value(name, value, as_type?)`

`write_value` places a literal value into the execution scope.

- `name` is a top-level lexical identifier, not a slash path or member expression.
- Writing an existing mutable binding preserves and checks its declared type.
- A new binding normally gets a type inferred from the literal.
- `as_type` is required only when inference is ambiguous, such as an empty list, an empty object,
  or a value intended for a named or union type.
- Inputs, imports, and `const` locals cannot be overwritten.
- Nested updates use an eval statement, where ordinary syntax makes the operation visible.

Normal generated trajectories should prefer:

```ts
const zero: Num = 0;
```

over `write_value`. The separate tool remains useful when a model already knows a literal answer,
when a transport supplies a large structured value directly, and in error or calibration examples
where a typed proposal must remain fallible.

### 4.4 `return_value(variable)`

`return_value` copies one existing scope binding into the enclosing lambda's typed result.

```text
return_value(variable="nextState")
```

Rules:

- The argument is a variable name, not an arbitrary expression and not a path.
- The value must be complete, contain no pending child, and fit the declared return type.
- To return an expression or literal, bind it first with `eval` or `write_value`.
- A successful call writes the result but does not waive the line-closure requirement.
- If all substantive lines are already closed, the next natural end of turn completes the lambda.

This separates eval's local `return` from completion of the natural-language function.

In a directory reducer, `return_value(variable="report")` is a terminal alias for
`commit(value="report")`: both select all dirty project files by default. The invocation mode
determines whether that selected delta is merged or discarded. This terminal reducer behavior is
distinct from an ordinary lambda's write-then-end `return_value` behavior.

### 4.5 `commit(value, include?, exclude?)`

`commit` is the terminal result action of a directory-reducer lambda.

```text
commit(value="report")
commit(value="report", include=["src/**", "package.json"])
commit(value="report", exclude=["dist/**", "tmp/**"])
```

Rules:

- `value` names an existing typed scope variable. It must be complete and fit the reducer's
  declared return type.
- Paths and patterns are relative to `project/`. Naming `codebase/` is rejected.
- With neither selector, every change made under `project/` by this reducer is committed.
- `include` selects dirty files, deletions, and moves to commit. `exclude` subtracts from that
  selection. Excluded project edits are discarded when the reducer invocation closes, while their
  trace remains available.
- An empty selected delta is valid; a reducer may inspect a directory and return a typed value
  without changing it.
- All substantive instruction lines must already be closed. `commit` validates the typed return,
  line marks, pending calls, file selection, and target folder revision as one terminal boundary.
- On success, the selected delta merges atomically into the `Folder` on which `apply` was called,
  and `folder.apply(...)` resolves to the typed value.
- On rejection, blocker, error, cancellation, or merge conflict, the caller's folder remains
  unchanged. The fork stays attached to a resumable pending invocation when resumption is valid.
- `codebase/` edits never cross the commit boundary.

When the reducer was called directly, `commit` still validates the typed value, marks, pending
work, and file selection, but the complete project delta is discarded after validation. This makes
the reducer's internal behavior independent of how the caller chose to consume it while ensuring
that an ordinary function call cannot mutate the supplied folder.

The commit tool is not a version-control command and does not require a textual commit message.
The typed return may itself be a `Text` message, a structured report, or any other declared return
type.

### 4.6 `mark_lines(start, end?, skipped?)`

Line closure deliberately remains explicit and noncompact.

```text
mark_lines(start=2)
mark_lines(start=4, end=6)
mark_lines(start=8, skipped=true)
```

Rules:

- `start` alone marks one source line.
- `start` and `end` mark one inclusive contiguous range.
- `skipped=true` means the instruction did not apply on the executed path.
- Disjoint ranges require separate calls.
- Done and skipped ranges require separate calls.
- Blank lines, comments, declarations, and other non-substantive lines are not obligations.
- The model should observe the preceding operation's result before marking its line. An eval and
  the mark that depends on its success should therefore occur in separate model turns.
- The runtime validates line numbers, range shape, and conflicting marks. Because it does not
  parse natural-language control flow, whether a line truly applied remains a model assertion
  evaluated through traces, tests, judges, and training admission.

Completion requires a valid returned value and closure of every substantive line. Correct marks
cannot make an invalid result valid, and a correct result cannot bypass open lines.

### 4.7 Honest exits

```text
report_blocker(missing)
report_error(message)
```

`report_blocker` indicates absent information or an uncovered case. `report_error` indicates
contradictory instructions, an invalid required operation, or an incompatible result. Both end the
current attempt without fabricating a value. Their distinct names are preserved in traces and
training IR even if both quiesce the runtime.

Caller validation remains the production default: a rejected action or invalid eval bubbles to
the parent rather than inviting the same interpreter to alter requirements until something passes.

## 5. Persistent typed scope

The opening state presents a compact scope table:

```text
Scope:
  inputs (immutable)
    state: State = { tasks: 5 items, order: 0 items, ... }
  imports (immutable callables)
    readyTasks(state: State): Task[]
    choose(tasks: Task[]): Task
    advance(state: State, task: Task): State
  locals
    ready: Task[] = 2 items
  result: State — not written
```

A directory reducer additionally shows:

```text
Directory reducer:
  project/: fork of Folder revision 81ac… (3 changed)
  codebase/: pinned module revision 229f… (1 local edit)
  commit: not submitted
```

The model normally refers to input names directly. Record fields and list elements use ordinary
member or bracket syntax. Internally, the runtime may continue using canonical slash paths; the
model-facing layer maps bindings and selections to those paths.

Scope invariants:

- Input bindings are frozen.
- Imported bindings are immutable function handles.
- Locals are private to the current lambda and are serialized with it.
- A callee receives explicit positional arguments. It does not implicitly see caller locals.
- A natural-language function sees its own inputs and its own static imports.
- Values are copied across function boundaries unless a host type explicitly defines an opaque
  handle.
- Mutating a local record or list is permitted only when the binding is mutable; the runtime
  validates the resulting value against its fixed type before committing the statement.
- Scope changes from an eval snippet are staged and committed only when the snippet reaches a safe
  boundary. External effects and completed child calls remain journaled and are never pretended to
  have rolled back.
- In a directory reducer, `project`, `codebase`, and `fs` are immutable injected bindings backed by
  the same scoped overlays as the model-facing file tools.

## 6. Static imports and unified callables

Natural-language modules accept a leading ECMAScript-style import block before their existing
metadata and instruction body. Crisp `.ts` modules use ordinary TypeScript imports and exports.

```ts
import { readyTasks, choose, advance } from "./planning";
import { sortBy, groupBy } from "@natlang/std/collections";
import { classify as classifyTicket } from "./triage";
import * as money from "@natlang/std/money";
```

The loader resolves imports into one checked module graph. A named binding may refer to:

- a natural-language lambda;
- a crisp TypeScript function;
- a host-backed function with declared capabilities;
- a re-export of any of the above.

The calling syntax is identical:

```ts
const ready = await readyTasks(state);
const labels = await Promise.all(tickets.map(ticket => classifyTicket(ticket)));
const cents = await money.roundCents(amount);
```

Import rules:

- Imports are static. Dynamic `import()` is unavailable in eval.
- Relative imports resolve within the source package; package imports resolve through the package
  manifest and lock/revision data.
- Every import is included in the source revision digest.
- The complete graph is type checked and cycle checked before execution. Recursion remains
  unsupported until it has explicit, bounded semantics.
- Imported names and signatures appear in the opening scope; full source is not pasted into every
  prompt.
- Importing an effectful function does not grant authority. The host must provide every declared
  capability.
- All callables may be awaited. Crisp calls may resolve immediately; natural-language calls may
  suspend into child interpreter episodes.
- A source edit creates a new module revision. An already-running lambda retains its original
  import bindings and code. A new root run loads the new revision.

Editing an imported source file changes the current filesystem overlay but does not silently replace
an already-instantiated callable. `run_program` loads a fresh module graph from the selected overlay
revision. This makes edit-test cycles explicit and prevents source text from changing underneath a
suspended call.

Existing companion folders, `uses`, inline codebases, and link records become loader inputs for a
migration compiler. They should not remain permanent model-facing concepts.

## 7. Directory reducers and `Folder.apply`

A directory reducer is a distinct lambda subtype with a typed ordinary-argument list and typed
return, plus one implicit directory transaction. Its runtime type is conceptually:

```ts
interface DirectoryReducer<Args extends unknown[], Return> {
  readonly kind: "directoryReducer";
  readonly parameters: Args;
  readonly returns: Return;

  (folder: Folder, ...args: Args): Promise<Return>;
}

interface Folder<Access extends "read" | "write" | "overlay" = "write"> {
  apply<Args extends unknown[], Return>(
    this: Folder<"write"> | Folder<"overlay">,
    reducer: DirectoryReducer<Args, Return>,
    ...args: Args
  ): Promise<Return>;
}
```

The model-facing call is ordinary asynchronous code:

```ts
const report = await folder.apply(refactorProject, request, policy);
```

The folder is not repeated in the reducer's declared parameter list. It is supplied either by the
`Folder.apply` receiver or as the first argument of a direct call. Both forms transparently create
a private fork. An ordinary lambda cannot call `commit` because it has no reducer transaction.

Source metadata declares the subtype (`kind: directory-reducer` in the current source format).
Imports preserve its overloaded type, so both forms are statically valid:

```ts
const applied = await folder.apply(refactorProject, request);
const analysis = await refactorProject(folder, request);
```

Crisp libraries remain ordinary imports available inside the reducer.

### 7.1 Invocation mounts

Inside the reducer, the filesystem has exactly two standard roots:

```text
project/    the automatically forked input folder
codebase/   the reducer's revision-pinned module and associated files
```

Both roots can be read and edited through file tools and the injected `fs` library. Their commit
semantics differ:

- `project/` is the reducer's transactional output. Selected changes merge into the `Folder`
  receiver when `commit` succeeds.
- `codebase/` is a writable invocation-local overlay over the pinned module snapshot. Changes can
  support analysis, temporary adaptation, generation, and developer workflows, but cannot be
  included in the reducer commit.
- Active imported function bindings remain pinned even if their source under `codebase/` is edited.
  Testing changed code requires a fresh module load through the developer execution surface.

The reducer may refer to the implicit project folder as `project` in eval code when it needs to
apply another directory reducer:

```ts
const nestedResult = await project.apply(normalizeSources, options);
```

That nested apply forks the reducer's current project overlay. A successful nested commit merges
into the current overlay; it does not bypass the outer reducer's eventual commit boundary.

### 7.2 Apply lifecycle

Both invocation forms capture the folder revision, create the same mounts, and execute the same
reducer. Their finalization differs.

`await folder.apply(reducer, ...args)` performs:

1. Capture the receiver's current revision.
2. Fork a private project overlay.
3. Mount the reducer's pinned source snapshot at `codebase/` with its own local overlay.
4. Run the reducer line by line with its ordinary arguments and implicit folder context.
5. Require explicit marks for every substantive instruction line.
6. Receive one terminal `commit` action with a typed value and file selection.
7. Validate and atomically merge the selected project delta into the receiver.
8. Resolve the JavaScript promise to the commit's typed value.

`await reducer(folder, ...args)` performs steps 1 through 6, validates the same terminal action,
discards both writable overlays, leaves `folder` unchanged, and resolves to the typed value.

No separate caller-interest flag exists. The syntax is the signal:

- `await folder.apply(reducer, ...args)` requests the typed value and committed project changes;
- `await reducer(folder, ...args)` requests only the typed value and discards all tentative edits.

This supports pure analysis, analysis after speculative local edits, and mutation-producing
application with one reducer implementation.

If two applies run concurrently, each receives a fork of the stated receiver revision. Disjoint or
identical deltas can merge deterministically. Conflicting deltas reject with both revisions intact.

### 7.3 Root folders

A host creates a `Folder` from a real directory, package tree, virtual browser directory, empty
output directory, or existing overlay. The handle defines its own finalization policy:

- a transactional parent folder merges child commits into its overlay;
- a preview folder retains committed deltas for inspection or export;
- a host-backed writable folder applies the reducer commit with optimistic revision checks;
- a read-only folder supports a direct reducer call but rejects `folder.apply`, which could retain
  changes.

Thus the same reducer can drive an in-memory preview, a nested transformation, or an atomic update
of a real input folder without changing its model-facing implementation.

### 7.4 First-class folder and file handles

Folders and files are opaque, lazy capability values in the eval type system. They are not expanded
into JSON trees and their contents do not enter model context until explicitly read.

The primary code-side API is handle-oriented:

```ts
interface EntryHandle<Access> {
  readonly name: string;
  readonly relativePath: string;
  readonly parent: Folder<Access> | null;

  exists(): Promise<boolean>;
  stat(): Promise<EntryStat>;
  remove(): Promise<void>;
  moveTo(destination: Folder<Access> | FileHandle<Access>): Promise<void>;
}

interface FileHandle<Access> extends EntryHandle<Access> {
  readText(range?: LineRange): Promise<string>;
  readBytes(): Promise<Uint8Array>;
  readJson<T = unknown>(): Promise<T>;

  writeText(content: string): Promise<void>;
  writeBytes(content: Uint8Array): Promise<void>;
  writeJson(value: unknown): Promise<void>;
  editText(edit: TextEdit): Promise<void>;
}

interface Folder<Access extends "read" | "write" | "overlay"> extends EntryHandle<Access> {
  dir(path: string): Folder<Access>;
  file(path: string): FileHandle<Access>;
  entry(path: string): EntryHandle<Access>;

  entries(options?: ListOptions): Promise<EntryHandle<Access>[]>;
  files(pattern?: string): Promise<FileHandle<Access>[]>;
  folders(pattern?: string): Promise<Folder<Access>[]>;
  walk(options?: WalkOptions): AsyncIterable<EntryHandle<Access>>;
  diff(): Promise<ChangeSet>;

  apply<Args extends unknown[], Return>(
    this: Folder<"write"> | Folder<"overlay">,
    reducer: DirectoryReducer<Args, Return>,
    ...args: Args
  ): Promise<Return>;
}
```

Typical use is concise and keeps paths relative to a retained handle:

```ts
const src = project.dir("src");
const routes = src.dir("routes");
const manifest = project.file("package.json");

const packageJson: PackageJson = await manifest.readJson();
const sources = await src.files("**/*.ts");

for (const file of sources) {
  const text = await file.readText();
  await file.writeText(updateImports(text));
}

const analysis = await src.apply(analyzeSources, policy);
```

`dir`, `file`, and `entry` create lazy handles without reading the backing filesystem. The first
operation that needs existence, contents, metadata, or a listing captures the corresponding backing
observation. Handles retain their mount, normalized path, rights, overlay identity, and captured
revision, so they remain valid across model turns and resumable continuations.

`files`, `folders`, and `entries` return handles rather than path strings. This reduces invented paths and lets a
handle to a subfolder be passed directly to another reducer. A reducer applied to a subfolder sees
that subfolder as its complete `project/` root and cannot escape to its parent.

Read-only handles expose read methods. Direct reducer calls may accept a read-only folder because
their private fork is discarded. `apply` requires a writable or overlay folder. Rights can be
attenuated with `folder.readOnly()` when that method is added, but can never be escalated by model
code.

Dynamic property access such as `project.src.components` is not the primary API. It conflicts with
method names, cannot distinguish a missing file from a directory without I/O, and fails on ordinary
names containing dots, spaces, or hyphens. An implementation may offer `folder.files[path]` and
`folder.folders[path]` convenience proxies later, but `file(path)` and `dir(path)` define portable
semantics.

The model-facing file tools continue to use root-qualified textual paths because tool arguments
cannot safely capture arbitrary JavaScript object identity. Eval code uses first-class handles. Both
routes resolve through the same `ScopedFileSystem` and observe the same overlay.

## 8. Ordinary control flow

There are no model-facing `run_function`, `for_each`, `fold`, `repeat`, or `resume` tools.

Direct call:

```ts
const chosen = await choose(ready);
```

Parallel map:

```ts
const labels = await Promise.all(
  tickets.map(ticket => classifyTicket(ticket, rubric))
);
```

Sequential accumulation:

```ts
let total = 0;
for (const line of lines) {
  total = await addLine(total, line);
}
total
```

Bounded repeat-until:

```ts
let current = initial;
for (let attempt = 0; attempt < 16; attempt++) {
  if (await finished(current)) break;
  current = await step(current);
}
current
```

The eval compiler lowers recognizable constructs to the existing map, fold, and iterate nodes when
that preserves semantics. Otherwise it executes the explicit loop through a resumable program
counter. The model remains responsible for choosing the control flow and bound.

No special `mapCall`, `foldCall`, or `untilCall` vocabulary is required. Standard library helpers
may exist for ordinary data operations, but they are regular imports and not privileged model
protocol.

## 9. Scoped filesystem and developer file surface

Execution scope and source files are intentionally separate domains.

| Domain | Examples | Tools |
|---|---|---|
| Typed execution state | `state`, `ready`, `nextState` | `eval`, `read_value`, `write_value`, `return_value` |
| Reducer project | `project/package.json`, `project/src/index.ts` | file tools and `fs` |
| Reducer codebase | `codebase/refactor.nl`, `codebase/helpers.ts` | file tools and `fs`, never reducer-committable |

A value tool rejects filesystem paths. A file tool never treats its contents as a live scope value.
This removes the current ambiguity between workspace paths, function instruction text, and host
files.

### 9.1 Filesystem layers

Every directory-reducer invocation owns a `ScopedFileSystem` for each mounted root, with three
logical layers:

1. **Local overlay.** New files, replacements, moves, and tombstones created by the running
   program or agent.
2. **Inherited overlay.** The immutable view inherited from the caller or parent execution.
3. **Backing snapshot.** The applied `Folder` for `project/`, or the reducer's pinned module tree
   for `codebase/`, read lazily from the host, package archive, browser virtual filesystem, or
   another configured provider.

Resolution is copy-on-write:

```text
read(path)   = local overlay -> inherited overlay -> captured backing file
write(path)  = local overlay only
delete(path) = local tombstone
```

The backing layer may initialize lazily, but it remains deterministic. On first access the runtime
records the content hash and retains or references the captured bytes. A later host-filesystem
change cannot alter the view of a running or resumed execution. Directory listings similarly
capture their observed revision, so a newly appearing host file does not silently enter an old run.

Paths are POSIX and begin with `project/` or `codebase/`. `..`, absolute paths, symlink escapes,
device files, and undeclared mounts are rejected. Additional named mounts require explicit typed
capabilities and cannot silently participate in a project commit.

### 9.2 Lambda and child-call scoping

Each directory reducer receives the two-root view. Ordinary lambdas receive no ambient filesystem.
Ordinary crisp functions called by a reducer may use the reducer's injected `fs` capability only
when it is passed or declared in their effect contract; they do not gain arbitrary host access.

A child call forks an overlay at call start:

- successful completion merges its file delta into the parent overlay;
- quiescence retains the delta with the pending child so resumption continues from exactly that
  view, while the parent does not mistake it for a completed edit;
- a rejected proposal before execution creates no delta;
- parallel children merge disjoint changes and identical writes deterministically;
- conflicting writes to the same path produce a structured merge conflict instead of choosing a
  winner;
- external effects remain governed by the effect journal and are not rolled back with files.

The trace records every layer lookup, first-read backing hash, overlay mutation, merge, conflict,
and explicit backing commit.

### 9.3 Injected `fs` library

Inside a directory reducer, eval and authorized crisp code receive an immutable `fs` binding. It
exposes a portable asynchronous API rather than Node's process-wide filesystem:

```ts
await fs.exists("project/rules/policy.json")
await fs.list("project/rules", { pattern: "**/*.json" })
await fs.readText("project/rules/policy.json")
await fs.readText("codebase/templates/report.md", { startLine: 20, endLine: 40 })
await fs.readBytes("project/assets/icon.bin")
await fs.readJson("project/rules/policy.json")
await fs.writeText("project/generated/report.md", report)
await fs.writeBytes("project/generated/data.bin", bytes)
await fs.writeJson("project/generated/result.json", result)
await fs.editText("project/rules/policy.json", { find, replaceWith, fuzzy: true })
await fs.move("project/draft.md", "project/archive/draft.md")
await fs.remove("project/obsolete.json")
await fs.diff("project/")
```

Injected methods and model-facing file tools use the same host interface and emit the same trace
events. A write through `fs.writeText` is immediately visible to a later `read_file`; an
`edit_file` result is immediately visible to `fs.readText` in the same overlay.

Implementations may use a lazy proxy, but the proxy holds no ambient operating-system authority:
all resolution passes through `ScopedFileSystem`. Python, native TypeScript, and browser hosts may
implement the proxy differently while preserving these observable semantics.

File contents are not automatically inserted into model context or typed value scope. The model
chooses `read_file`, `read_value`, or eval with `fs` according to the task.

### 9.4 Discovery and reading

`list_files(path?, pattern?)` returns a bounded, sorted source inventory with type and size.
`search_files(query, path?, pattern?)` performs literal or explicitly requested regular-expression
search and returns file, line, and a short match context. `read_file(path, start_line?, end_line?)`
returns numbered source lines and a revision digest.

All reducer paths are root-qualified POSIX paths. Developer sessions outside a reducer use paths
relative to their explicitly opened workspace. Absolute paths and traversal outside the authorized
root are rejected.

### 9.5 Writing and editing

`write_file(path, content)` creates a file or replaces the entire current file. For replacement it
accepts an optional expected revision internally, so a host can reject stale writes.

`edit_file(path, find, replace_with, fuzzy?)` performs one replacement:

- exact mode requires one exact occurrence;
- fuzzy mode succeeds only when one unambiguous span is selected;
- the result reports the changed line range and new revision;
- ambiguous or absent selections fail without modifying the file.

`apply_patch(patch)` accepts a standard unified diff, validates all hunks first, and applies it
atomically across files. It is the preferred operation for coordinated developer changes.

`move_file` and `delete_file` provide complete refactoring capability. Hosts may require a stronger
write authority for deletion, but deletion is part of the developer surface rather than simulated
with empty writes.

### 9.6 Diffs, validation, execution, and backing commits

`diff_files(path?)` returns the overlay delta as a structured summary and unified diff without
changing the backing store. It includes changes made through both file tools and injected `fs`.

`validate_codebase(paths?)` parses source, resolves imports, validates types and effects, checks
cycles, and reports precise file diagnostics. It does not invoke a model.

`run_program(entry, inputs?)` starts a fresh, revision-pinned execution of an entry point. It
returns the typed outcome, trace identifier, effects summary, and validation diagnostics. A
developer agent can therefore inspect, edit, validate, and exercise a codebase without leaving the
agent surface.

`commit_files(paths?, message?)` is a developer-authoring operation that applies selected overlay changes to a mutable backing working tree
with optimistic revision checks. It is absent unless the host grants backing-write authority. If a
backing file changed since capture, the commit reports a conflict and leaves both overlay and
backing file intact. Immutable packages can export the diff as an artifact instead.

This developer operation is distinct from a directory reducer's typed `commit` action.

An edit never hot-patches an active lambda. To test edited source, the agent runs a new program
revision against the overlay. This provides familiar edit-test ergonomics while preserving
resumability and trace truth. After validation and tests pass, an authorized developer agent can
commit the same reviewed overlay to the backing tree.

## 10. Eval language and execution IR

`eval` must not be implemented as unrestricted JavaScript plus an ad hoc asynchronous callback.
Natural-language calls can suspend, resume, spawn children, perform effects, and survive process
restart. The portable design is a restricted TypeScript front end lowered to engine-independent
orchestration IR.

Initial supported syntax:

- literals, arrays, records, templates, and member/index selection;
- `const` and `let` declarations with optional type annotations;
- assignment to mutable bindings and their members;
- arithmetic, comparison, Boolean, nullish, and conditional expressions;
- `if`/`else`, bounded `for`, `for...of`, `while`, `break`, and `continue`;
- direct calls to imported bindings;
- `await`;
- synchronous arrow functions used by safe collection operations;
- `Promise.all` over a finite collection;
- local `return` for the eval tool result.

Unsupported initially:

- dynamic import, dynamic eval, `Function`, generators, classes, prototypes, reflection, and global
  object access;
- arbitrary exception catching around runtime validation or child-call failure;
- unbounded recursion;
- closures that escape the eval snippet;
- filesystem, network, timers, and process APIs except through imported typed host functions.

The scoped injected `fs` binding is the sole filesystem exception. It is a typed host API over the
execution overlay, never ambient filesystem access.

Core IR nodes include:

```text
Literal, ReadBinding, ReadMember, Declare, Assign
Block, If, ForOf, BoundedLoop, Break, Continue
Call, CallDirectoryReducerDiscarding, ApplyDirectoryReducer, Await, Parallel, ReturnEval
CommitDirectory(value, include, exclude)
```

Every node carries source spans and inferred types. Suspended evaluation serializes the IR digest,
program counter, lexical bindings, pending child call identifiers, and completed-effect journal.

The TypeScript parser should be shared or produce a canonical serialized AST consumed by both
runtimes. Python and TypeScript execute the same conformance fixtures and serialized IR. The
browser bundle uses the same grammar and lowering rules.

## 11. Turn and continuation behavior

The normal episode is intentionally incremental:

1. The model reads the next open instruction line and compact scope.
2. It calls `eval` or another operation needed by that line.
3. It observes the actual tool result.
4. It calls `mark_lines` for that line or contiguous completed range.
5. It proceeds to the next open line.
6. An ordinary lambda calls `return_value` when the required result exists, then ends naturally
   once the return and marks are valid.
7. A directory reducer calls terminal `commit` or its unfiltered alias `return_value` after its
   typed result exists and every substantive line is closed. Apply mode merges the selected project
   delta; direct-call mode discards it. Either terminal action resolves to the typed value.

The prompt should discourage speculative reading, line marking before observation, loop unrolling,
manual execution of imported semantic functions, and restating large values.

Continuation checkpoints retain:

- typed inputs and locals;
- imported module revision and binding table;
- captured backing-file manifest and copy-on-write filesystem overlays;
- the result draft;
- marks;
- pending eval IR and child calls;
- effect journal;
- a short model-authored note.

A continuation begins with a fresh conversation containing this durable state and note. It never
pastes the preceding conversation. The checkpoint is requested only at a safe boundary; a
suspended eval already has its own program counter and does not require the model to reconstruct it.

## 12. Failure semantics

Failures are classified rather than flattened into prose:

- **syntax failure:** the eval snippet cannot be parsed;
- **type rejection:** a declaration, assignment, call, or return does not fit its type;
- **binding rejection:** an unknown, immutable, or out-of-scope name was used;
- **child blocker:** a called lambda reported missing information;
- **child error:** a called lambda reported invalid instructions or failed validation;
- **directory conflict:** a reducer commit no longer applies cleanly to its `Folder` revision;
- **invalid directory commit:** a reducer selected `codebase/`, omitted required marks, or supplied
  a value that does not fit its declared return type;
- **effect failure:** an authorized external operation failed or has uncertain completion;
- **resource interruption:** an explicit embedding budget or cancellation stopped work.

With caller feedback, these failures quiesce the current lambda and bubble through the ordinary
call boundary. The interpreter that made the invalid proposal is not badgered into changing the
requirements. The optional careful-review fork may still withdraw a low-confidence proposal before
execution.

Eval code cannot catch validation failures, blockers, or effect uncertainty during the initial
implementation. Later typed domain errors may be represented as ordinary result unions and handled
by program logic without weakening runtime errors.

## 13. Tracing, audit, and provenance

Every model proposal records:

- the exact offered surface version;
- the eval source text and lowered IR digest;
- scope revision before and after execution;
- values read and bindings written;
- imported function calls, arguments, results, and child trace identifiers;
- folder revisions, automatic forks, reducer applies, complete dirty sets, commit selections,
  discarded edits, merges, and conflicts;
- effects and their stable identities;
- line marks;
- validation diagnostics;
- model reasoning, tool choice, confidence, and optional review decision for teacher traces.

The new surface receives a new explicit identifier such as `scope-eval-v1`. No runtime may hardcode
an older surface name in action events. Manifests, action events, materialized trajectories, and SFT
exports must agree on the identifier and source revision.

Final prose remains audit content. An ordinary lambda's program result is its typed value. A
directory reducer additionally produces the committed project delta recorded in its invocation
outcome and applied to the `Folder` receiver.

## 14. Training and data migration

### 14.1 Canonical trajectory IR

Training IR stores semantic actions independently of backend tool serialization:

```text
Eval(code, lowered_ir, reads, writes, displayed_value)
ReadValue(expression, range)
WriteValue(name, type, value)
ReturnValue(variable)
Commit(value, include, exclude, selected_delta)
MarkLines(start, end, status)
ReportBlocker(message)
ReportError(message)
FileAction(...)
```

Teacher reasoning, ordering choices, continuation notes, confidence, reviews, executions, and
validation results remain in the IR. Renderers decide the exact chat template and tool-call syntax.

### 14.2 Projection of existing data

Existing traces may be projected only when their semantics are unambiguous:

- direct positional calls become awaited assignments;
- `for_each` becomes `Promise.all(...map(...))` when parallel semantics are safe;
- folds and repeats become explicit bounded loops;
- `run_code` plus a faithful write becomes one eval declaration;
- scope copies become assignments;
- line marks and honest exits retain their meaning.

Trajectories involving guessed destination repair, incompatible writes, ambiguous function edits,
or obsolete local-feedback correction should be regenerated rather than cosmetically rewritten.

Every training assembly command must require one target surface and reject rows with another
surface unless an explicit, tested projection produced them. Old rendered v2/v3/v4 files must not
be discoverable through a broad glob in a new training run.

### 14.3 New curriculum

Synthetic and teacher data must cover:

- one line per eval-and-mark cycle;
- direct natural-language and crisp calls through identical imports;
- directory reducers invoked through awaited `folder.apply` for retained edits and called directly
  for value-only analysis with discarded edits;
- visible eval results from final expressions and local `return`;
- branches with explicit skipped lines;
- parallel map, sequential accumulation, and bounded repeat-until;
- type and syntax mistakes that bubble without fudging;
- blockers and explicit errors;
- interruption and resumption inside eval loops and child calls;
- large and lazy values read through `read_value`;
- first-class folder and file handles retained across turns, subfolder-relative operations, and
  reducers applied to subfolders;
- developer discovery, exact edits, fuzzy edits, patches, import refactors, validation, and tests;
- code-side `fs` reads and writes, lazy backing reads, child overlay merges, conflicts, diffs, and
  project-only reducer commits;
- reserved keys, unusual field names, empty collections, unions, optional fields, and effects;
- contrastive examples in which the model honestly persists when a careful review challenges a
  correct action and withdraws when it recognizes a real mistake.

Rare actions receive explicit quotas. In particular, no accepted corpus may contain zero
`report_error`, negligible resumption coverage, or only a handful of developer edits.

## 15. Evaluation plan

Compare v4 and `scope-eval-v1` on the same semantic programs and input seeds. Measure:

- whole-program correctness;
- line-mark correctness separately from result correctness;
- invalid binding, destination, and type proposals;
- child-call failures and honest error escalation;
- model turns and tool calls per completed source line;
- prompt, cached-prompt, and generated tokens;
- schema bytes and prefix-cache reuse;
- wall time and GPU throughput;
- continuation success after process interruption;
- Python/TypeScript trace conformance;
- developer task completion for inspect-edit-validate-run loops;
- overlay correctness across tool-side and code-side file operations.

Required probe groups include the current reconciliation, dependency planning, order saga,
NLProlog, webserver, shopkeeper, and architectural programs, plus deliberately malformed programs
and imports.

Admission remains semantic: exact expected value and effects, type-valid trace, correct required
line closure, no incompatible or superseded surface calls, and human review for novel teacher
behavior.

## 16. Implementation phases

### Phase 0: specification and fixtures

- Approve this surface and update the normative spec.
- Define `scope-eval-v1` JSON schemas and canonical action IR.
- Write cross-runtime fixtures before changing the production default.

### Phase 1: scope and imports

- Add durable lexical bindings over the existing typed tree.
- Add normal static imports and compile legacy `uses`/companion codebases into the new graph.
- Add branded `DirectoryReducer<Args, Return>` and `Folder` types with both accepted forms:
  `await reducer(folder, ...args)` for a discarded fork and
  `await folder.apply(reducer, ...args)` for an applied commit.
- Present imported signatures and compact values in the opening state.
- Implement revision-pinned module snapshots.
- Implement lazy captured backing snapshots and durable copy-on-write overlays.

### Phase 2: eval compiler and runtime

- Parse the restricted TypeScript subset into canonical orchestration IR.
- Implement declarations, assignments, displayed results, direct awaited calls, conditions, and
  bounded loops.
- Lower `Folder.apply` to an automatic project fork, resumable reducer invocation, typed commit,
  and atomic selected-delta merge.
- Lower a direct directory-reducer call through the same machinery with an unconditional discard
  finalizer.
- Serialize safe suspension points and program counters.
- Lower safe parallel map, fold-like loops, and repeat-until loops to existing runtime nodes.

### Phase 3: interpreter tools

- Implement `eval`, `read_value`, `write_value`, `return_value`, directory-reducer `commit`, and the
  unchanged noncompact `mark_lines` requirement in Python and TypeScript.
- Preserve caller validation, honest exits, careful review, and natural completion.
- Fix surface provenance at every trace layer.

### Phase 4: developer tools

- Implement source discovery, search, reading, writes, fuzzy edits, atomic patches, moves, deletes,
  validation, and fresh-revision execution.
- Inject the portable `fs` proxy into authorized eval and crisp scopes, backed by the same overlay.
- Implement child overlay fork/merge, diff export, optimistic backing commits, and browser virtual
  filesystem parity.
- Add authority profiles and browser-safe virtual-filesystem equivalents.
- Add edit conflict and source revision diagnostics.

### Phase 5: IR projection and corpus regeneration

- Add proven v4-to-scope projections.
- Regenerate ambiguous and failure-handling examples.
- Produce balanced synthetic data, then teacher trajectories, predominantly as short line-level
  segments with durable scope rather than repeated conversations.
- Keep teacher reasoning and choices in canonical trajectory IR.

### Phase 6: teacher validation and student training

- Run a small teacher comparison first and repair the surface rather than adding prompt bulk.
- Generate full coverage only after whole-program and line-closure probes pass.
- Train the efficient teacher adapter if useful, then train the student in phases: synthetic
  behavior followed by admitted teacher trajectories, with hard synthetic cases retained.

### Phase 7: default and retirement

- Make `scope-eval-v1` the default after Python, native TypeScript, browser, and teacher transports
  pass parity and interruption tests.
- Retain v4 only as an explicit replay and migration surface.
- Stop producing new v2/v3/v4 training rows and remove superseded rendered corpora once the new
  manifests are reproducible.

## 17. Risks and mitigations

### The model may write bad code

Keep the subset small, return precise parse/type diagnostics to the caller, train code rescue
behavior through careful review, and measure whole programs rather than token loss.

### One eval snippet could hide too much work

Prompt and train one natural-language line per snippet. Keep line closure separate and mandatory.
Record calls and binding changes within eval so a large snippet is still auditable.

### Persistent JavaScript state is difficult to serialize

Persist typed bindings and orchestration IR, not a JavaScript VM heap. Disallow escaping closures,
prototype mutation, and unsupported host objects.

### Async natural-language calls differ across hosts

Lower calls to existing pending nodes and interpret the same portable orchestration IR in both
runtimes. Do not depend on a QuickJS-specific Promise bridge as the semantic definition.

### Developer tools could enlarge the ordinary interpreter surface

Offer them only in stable developer authority profiles. Keep file and value operation names,
schemas, and namespaces distinct.

### Source edits could invalidate an active trace

Pin every execution to a module graph revision. Edits create a new revision and require a fresh
`run_program` invocation.

### Lazy fallback could make resumption depend on a changed host filesystem

Capture and hash a backing file or directory observation on first access, retain it in the run's
content-addressed snapshot, and replay that captured observation after interruption.

### Parallel lambdas could overwrite one another's files

Fork child overlays and merge only disjoint or identical deltas. Surface conflicts as typed merge
failures with both revisions preserved.

### Standard JavaScript loops could accidentally serialize safe parallel work

Support `Promise.all(...map(...))` and lower it to the runtime's parallel map where safe. Preserve
explicit sequential loops when order or effects matter.

## 18. Initial acceptance criteria

The new surface is ready for broad teacher generation when:

1. The Python and TypeScript tools, scope rendering, eval results, marking, errors, imports, and
   traces match on shared fixtures.
2. A process can stop inside a natural-language call or loop and resume without repeating a
   completed effect or preceding conversation.
3. Imported ordinary crisp and natural-language functions are indistinguishable at the call site;
   imported directory reducers support both their direct analysis call and `Folder.apply`.
4. A model can inspect, patch, validate, and run an authorized codebase without confusing a file
   path with a scope variable.
5. The same scoped file written through `fs` can be read through a file tool and vice versa, while
   unauthorized paths remain inaccessible.
6. `await folder.apply(reducer, ...args)` leaves the folder unchanged on failure and atomically
   applies only the reducer's selected `project/` delta on success; `codebase/` never crosses that
   boundary.
7. `await reducer(folder, ...args)` returns the same typed value while leaving the supplied folder
   unchanged even when the reducer wrote files or called `commit` internally.
8. Whole-program teacher probes remove the current wrong-destination and call-mode failure classes
   without replacing them with persistent scope or eval parsing failures.
9. Every successful program has a valid typed result and complete, explicit line marks.
10. Tool schemas remain small and stable across ordinary scope changes.
11. Training assembly rejects incompatible surface versions by construction.
