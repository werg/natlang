# natlang language specification

Version **0.2-draft**, 2026-09-19. Normative for the harness, the reference
policy, and all training data. Where this document and `PLAN.md` / `TYPES.md`
disagree, this document wins and the others should be corrected.

A natlang program is a **pseudocode algorithm**: functions with typed
signatures, subroutine calls, loops, conditions, local variables, organised as
a **code base** of files. The author states the structure. The interpreter, a
small language model, carries it out: it reads the pseudocode, decides the next
step, and performs it through the current `scope-eval-v1` surface. The harness provides memory, typing,
pluggable crisp execution and I/O. It parses no instructions, holds no cursor and owns no
control flow; every constraint it imposes is type-level and applies at write
time.

Key words: **must**, **must not**, **may** are used in their plain sense.

### Current agent surface: `scope-eval-v1`

The model receives a persistent TypeScript evaluation scope in the same Node host used by crisp code. `eval(code)`
handles declarations, assignments, control flow, exact expressions, and
ordinary awaited positional calls to imported natlang and crisp functions;
bindings remain available on later turns. The auxiliary actions are
`read_value(expression, start?, end?)`, `write_value(name, value, as_type?)`,
`return_value(variable)`, `mark_lines(start, end?, skipped?)`,
`report_blocker(missing)`, and `report_error(message)`. File access uses
`list_files`, `search_files`, `read_file`, `write_file`, `edit_file`, and
`diff_files`. Directory reducers additionally receive `commit`.

Execution values and files are separate. The recursive `codebase/` overlay has
a fixed file manifest: existing file contents may be edited and imports may be
relinked, but codebase files may not be created, moved, or deleted. A directory
reducer additionally receives writable `project/` semantics, where normal file
creation, editing, movement, and deletion are allowed; `commit` selects
changes and `folder.apply` retains them. These rules are normative for the
current runtime.

The slash-path `read`/`write`/`edit`/`run_code`/`call`/`mark_done` protocol
described in the historical section below is retained for trace compatibility
and design history only.

---

## 1. The object tree (historical v0.2 model)

The object-tree and slash-path sections below document the superseded protocol
for old traces. They do not override the current `scope-eval-v1` surface above.

A program, its state, its inputs, and its results are one tree of typed
nodes. Every node has a type, fixed when the node is created.

There are two families of node:

- **Value nodes**: scalars, records, lists, dicts, blobs.
- **Pending nodes**: a `Lambda`, `Map`, `Fold`, or `Iterate`. A pending node
  stands where a value will be. Reducing it replaces it with that value.
  Pending nodes come into existence only through `call` (§5.5).

### 1.1 Historical paths

A path is a sequence of segments separated by `/`. A segment is a record
field name, a dict key, or a list index (0-based). Inside an episode, paths
are **relative to the current lambda**, whose addressable parts are
`instructions`, `args`, `let`, and `return`; `codebase/<name>` is readable.

```
args/tickets/3/body
let/labels
return/summary
let/strict/instructions            # the text of an editable function copy (§5.3)
codebase/classify                  # read-only: the text of a function
args/files/docs/design.md/text     # leaf in a host-backed Dict<ProjectFile>
```

- **Ranges** are inclusive at both ends and use the numbers the listing
  shows. Text lines are 1-based, list items 0-based. `read` takes them as
  `start` and `end`.
- **Meta paths** use an `@` suffix and are read-only:

| Suffix | Gives |
|--------|-------|
| `@status` | `unreduced`, `running`, `quiesced`, `done` |
| `@note` | the closing note of a quiesced node |
| `@problems` | the validation report for the subtree |
| `@origin` | the provenance record of a value (the function instance that produced it, its `args`, its trace, earlier attempts) |
| `@effects` | the effect journal of the current lambda |
| `@dist` | the logged distribution of a finite-typed write |

### 1.2 Scope

A lambda can address only its own subtree. Anything it needs from outside
must have been passed to it as `inputs` by its caller. Its locals are private:
a callee never sees the caller's `let`. The functions it can call are those of
its own code base (§3.4), nothing else.

---

## 2. Types

Types are written in a subset of TypeScript type syntax.

```
Type     := Prim | Literal | Record | Type "[]" | "Dict<" Type ">"
          | Type "|" Type | Name | Pending | "(" Type ")"
Prim     := "Text" | "Num" | "Bool" | "Null" | "Blob"
Literal  := a double-quoted string, e.g. "urgent", or a number, e.g. 3
Record   := "{" Field ("," Field)* "}"
Field    := name ["?"] ":" Type
Pending  := "Lambda<" Record "," Type ">"
          | "Map<" Type "," Type ">" | "Fold<" Type "," Type ">"
          | "Iterate<" Type ">"
Name     := an identifier declared in a `types` block
```

- `Text` is Markdown text. `Num` is a finite number. Lists are `T[]`.
- **`Dict<T>`** is the string-keyed container. (It was called "Map" in early
  drafts; renamed so that `Map` means only the combinator.) A host may bind a
  read-only lazy implementation. This does not add a model-facing type: source
  still declares `Dict<T>` and uses ordinary `args/...` paths.
- A union of literals is an enum. String literals name options
  (`"sell_pear" | "decline"`); **numeric literals** give small
  numeric ranges (`1 | 2 | 3`). Enum narrowing (§2.1) applies to both.
- **Named types** are declared in a function's `types` frontmatter (or a
  folder's `types.ts`), are visible in that function and inherited by its code
  base, and may be recursive (a type may refer to itself; functions may not):
  `types: { Unit: '{ name: Text, reports: Unit[] }' }`.
- There is no `Any` and no type inference.
- v0.2 is **structural only**: no refinements, no postconditions.

### 2.1 Fit

Type `A` **fits** a slot of type `B` when:

1. `A` equals `B` structurally; or
2. `A` is a member of union `B`; or
3. `A` is an enum whose literals are a subset of enum `B`'s (**enum
   narrowing**), or `A` is a literal or enum of literals and `B` is their base
   type (`"spam"` fits `Text`, `3` fits `Num`); or
4. `A` is a record and `B` is a record, and each field fits (optional fields
   of `B` may be absent in `A`); or
5. **containers are covariant**: `A[]` fits `B[]`, and `Dict<A>` fits
   `Dict<B>`, when `A` fits `B`. (Values are never mutated in place through
   an alias, so this is sound here.) Or
6. **promise rule**: `A` is `Lambda<_, T>`, `Map<_, E>` with `T = E[]`,
   `Fold<_, T>`, or `Iterate<T>`, and `T` fits `B`.

Rule 6 is what allows a call in progress to sit wherever its result is needed.

### 2.2 Draft types

`Draft<T>` is `T` made deeply partial: any record field may be absent, any
list may be shorter than it will be. While a lambda is being reduced, its
`return` is checked against `Draft<T>`. At a commit point (§6.4) they are checked against `T`.

**Holes are fine, lies are not**: a missing field is accepted and reported as
a hole; a value of the wrong type is rejected at write and never enters the
tree.

---

## 3. Lambdas and code bases

### 3.1 Parts

A lambda is an **instance of a function**. Every zone of state an episode can
touch is a part of it:

| Part | Written by | Meaning |
|------|-----------|---------|
| `type` | author | `Lambda<P, T>`, from the function's signature |
| `instructions: Text` **or** `code: Text` | author; immutable while the lambda runs | the body. Pseudocode or prose for an ordinary function; TypeScript for a **crisp function**. Exactly one is present. |
| `args` | the caller | the bound parameters, typed by `P`. Frozen when the instance starts; read-only to the lambda itself |
| `let` | the interpreter | typed locals, `let/<name>` (§3.2) |
| `return` | the interpreter | the result so far, typed `Draft<T>` until commit |
| `codebase` | author | name -> function definition. Immutable, shared by reference (§3.4) |
| `types?`, `effects?` | author | named types; capabilities the function may use (§9.5). No declared effects; native host authority is separate (§9) |

and harness-owned metadata: status, step count, note, provenance. Swap-out
serializes interpreter state. Native objects, external resources and evaluator
bindings need a separate host recovery contract; a tree snapshot does not
recreate them.

### 3.2 Locals

- A local comes into existence by the first write to `let/<name>`. A plain
  `write` states its type; a `call` derives it from the callee's signature;
  a `write` with `source` takes the type of the source.
- A local holds a plain value, a call in progress (a pending node, §4), or an
  editable copy of a function (§5.3).
- Locals are private to the lambda, survive its completion as provenance, and
  are not part of the result. There is no fixed local-count limit; names are identifiers.

### 3.3 Lifecycle

```
unreduced --call--> running --+--> done      (node is replaced by its value)
     ^                        |
     |                        +--> quiesced  (node left exactly as it was)
     +---- call again (resume) +
```

- **Completion** of an ordinary lambda: the interpreter replies (§5.7) while
  `return` contains no pending nodes and fits `T`. There are **no tail
  calls**: completion always requires a value.
- **Completion** of a crisp function: the code returned a value that fits `T`.
- **Swap-out.** On completion the node is replaced by its value. The lambda
  record moves to provenance and is reachable at `path@origin`.
- **Quiescence.** A reduction that stops without completing leaves the node
  in place, partial `return` included, with status `quiesced` and a note.
  Causes: `report_blocker`; a reply without a valid `return`; a budget ran
  out; the process crashed; a crisp function threw. There is no separate
  failure state. The caller sees the note and may `call` again to resume.
- **Data is never destroyed.** Values are superseded by later writes or
  archived to provenance.

### 3.4 Code bases

A function is a file; its private functions live in a companion folder of the
same name:

```
triage/
  main.nl                 the entry point
  main/                   main's code base
    classify.nl
    is_urgent.nl
    summarize.nl
    summarize/            summarize's code base
      shorten.nl
      is_short.ts
std/
  count_true.ts           a crisp function
```

- `name.nl`: YAML frontmatter between `---` lines, then the body: pseudocode,
  or for a leaf a plain task description.
- `name.ts`: a crisp function. The same frontmatter inside a leading
  `/*--- ... ---*/` comment, then a TypeScript function body (§9.3).
- `types.ts` in a folder (optional): `type Name = ...;` declarations shared by
  every file in that folder.

| Frontmatter key | Meaning |
|---|---|
| `args` | name -> type, in signature order. `name?` marks an optional parameter |
| `returns` | type |
| `types` | name -> type; visible in this function and inherited by its code base |
| `description` | one line: what a caller sees in its listing |
| `uses` | name -> relative path of a function defined elsewhere (a link) |
| `effects` | capabilities (§9.5) |
| `engine` | selected crisp executor; must be provided by the embedding (§9.1) |

**Scope is lexical.** The code base of a function is the functions in its
companion folder plus its `uses`. Nothing else: not its siblings, not its
caller's code base. An instance of `main/summarize.nl` carries
`main/summarize/` (and its own `uses`) wherever it runs. Reuse is by linking:
`uses: { word_count: ../../std/word_count }`. There is no fixed function-count limit.

**Immutable, shared by reference.** Instantiating a function builds a fresh
lambda that points at the same definition; no body is duplicated, and no tool
can change a definition (§5.3 is how behaviour is varied).

**No recursion.** No function can reach itself, directly or through other
functions: the loader rejects a code base that contains a cycle (`recursion`).
All repetition goes through `call` with `over`, `over` + `init`, or `init` +
`until` + `max`, whose bounds the harness controls. The run-time guards of
§6.3 remain.

**Inline form.** In the YAML program format a `$lambda` may carry
`codebase:`, name -> function definition (the frontmatter keys plus
`instructions` or `code`, optionally a nested `codebase`) or `{ $link: path }`.

**There are no anonymous lambdas.** Nothing in the tool surface creates a
function from nothing. A program without a code base is a leaf: one judgment,
extraction or rewrite, answered whole.

---

## 4. Combinators

Combinators are pending node types with fixed semantics. They have no
instructions and get no episode; only their body instances do. The
interpreter never writes one: `call` with `over`, `over` + `init`, or
`init` + `until` + `max` (§5.5) builds the node from a code-base function,
places it where its result is needed, and runs it.

### 4.1 Map

`Map<A, B>` with parts `over: A[]` and `fn`, an instance template of a function `f(..., x: A, ...) -> B`.
Fits a slot of type `B[]`.

- The item goes to **the one required parameter the call left unbound**, under
  the author's own name for it (`ticket`, not `item`); the other parameters
  are bound by `inputs` (a rubric, a schema). If `f` declares `index: Num`,
  the harness binds it.
- **Run**: the node expands in place. Slot `i` of the node becomes an
  instance of `f` with that parameter bound to `over[i]`. Copies are copy-on-write. The
  harness reduces all slots, in parallel, batched. Result order is the order
  of `over`. Length is preserved by construction.
- **While expanded**, `<map>/i` addresses slot `i` (a value once reduced, a
  lambda otherwise), and `<map>/fn` and `<map>/over` remain readable.
- **Writing a slot by hand.** The caller may `write` into a slot of an
  expanded Map, typically to repair a quiesced one. The slot then accepts any
  value that fits the *element type of the slot the Map occupies*, which may
  be wider than `B`. (`B` constrains what `fn` returns, not what the parent
  may put there.) When every slot is a value, the node is the list.
- **Partial result**: if any slot quiesces, the Map quiesces. Finished slots
  hold values. Calling again (§5.5, resume) runs only the slots that are not
  yet values.
- Each slot has its own step budget.

### 4.2 Fold

`Fold<A, S>` with parts `over: A[]`, `init: S`, and `step`, an instance
template of a function with parameters named `acc: S` and `item: A`
returning `S`. Fits a slot of type `S`.

- **Trigger**: for each item in order, the harness instantiates `step` with
  `acc` and `item` bound, reduces it, and takes its value as the next `acc`.
  The model never threads the accumulator.
- **While running or quiesced**, `<fold>/acc` is the current accumulator,
  `<fold>/at` the index of the next item, `<fold>/current` the step lambda in
  progress.
- **Partial result**: if a step quiesces, the Fold quiesces at that index.
  Calling again resumes there.

### 4.3 Iterate

`Iterate<S>` with parts `init: S`, `step` (an instance template of a function
whose one unbound parameter takes `S` and which returns `S`), `check`, and
`max: Num` (required). Fits a slot of type `S`.

- `check` is a code-base function of one parameter returning `Bool`, applied
  to the state after every round, as a fresh episode (or crisp, when the
  stopping rule is exact). `true` ends the loop with the state as the value.
- Crisp guards, independent of the model: reaching `max` quiesces the node; a
  state whose content hash equals an earlier state's is degenerate and
  quiesces it.
- The harness also supports a check of the form
  `(recent: S[], iteration: Num) -> LoopVerdict` with
  `LoopVerdict = { reason: Text, verdict: "continue" | "done" | "degenerate" }`
  (reason written before verdict); it is not reachable from `call` in v0.2.
- `<iterate>/state` and `<iterate>/iteration` are readable while it runs or
  is quiesced.

### 4.4 Open lists

A list may be **open**: a harness attribute meaning an external source keeps
appending to it. Its type is still `A[]`. A `Fold` or `Map` over an open list
does not complete while the list is open. A long-lived reactive program is a
top-level `Fold` over an open list of events, with outputs emitted as effects
inside `step`. Open lists are bound only at the I/O boundary (§10): the host
builds the root `Fold` from a step function `f(acc: S, item: A) -> S`, an
initial state, and a source of events (`host.load_fold`); the run ends when
the source ends or yields `$close`. `codebases/webserver` (a listening HTTP
server) and `codebases/shopkeeper` are such programs.

---
---

## 5. Historical v0.2 tool protocol

The following is historical material, retained to document earlier traces. It
is not the current agent interface. The model worked through native tool calls. The opening user message was the
lambda's `instructions`, the line "Write the result to `return` (T).", and the
listing of its functions. **Data never shares a channel with instructions**:
the harness performs the first step on the interpreter's behalf,
`read(path="args")`, and the workspace arrives as a tool result (§8).

The historical eight standard tools; `call` and `mark_done` were present only when the lambda had
functions. Their argument schemas are regenerated from the tree and the types
every turn.

| Tool | Arguments | Purpose |
|------|-----------|---------|
| `read` | `path`, `start?`, `end?` | show a value, a range of it, `codebase/<name>`, or a host file |
| `write` | `path`, `type`, `value` \| `source` | put a plain value into `return` or a local; or copy a function (§5.3) |
| `edit` | `path`, `old`, `new` | replace text |
| `run_code` | `code`, `engine` in engine-selecting surfaces | exact work in the selected executor; the result comes back |
| `call` | `function`, `to`, `inputs?`, `values?`, `over?`, `init?`, `until?`, `max?` | run a function of the code base, result at `to` |
| `mark_done` | `start`, `end?`, `skipped?` | lines of the own program are finished, or did not apply (§7.3). `write` and `call` also take `done?`: a line, or `[first, last]`, marked if the action succeeds |
| `report_blocker` | `missing` | end without a result, saying what is missing |
| `report_error` | `message` | end without a result, explaining contradictory or impossible work |

Several calls may share a turn if they are independent. A call that depends on
what an earlier call created goes in a later turn: a turn's grammar is built
from the state before the turn. Batches are ordered and non-atomic: every action
receives a result, and successful actions remain applied after another action is
rejected. Terminal outcomes can stop the batch. Neither rejection nor cancellation
rolls back external effects.

### 5.1 read

Returns the whole value at `path`, or the range `start..end`. Meta paths are
read the same way. `codebase` lists the functions; `codebase/<name>` shows a
signature, description and body.

A host-backed `Dict<T>` lists its entries through its ordinary argument path.
For example, `args/files` lists a filesystem dictionary,
`args/files/docs` lists that branch, and
`args/files/docs/design.md/text` reads the text field of a typed leaf. Text
defaults to its first 200 lines and accepts inclusive, 1-based `start` and
`end` ranges. A standard project-file leaf is either
`{ kind: "text", text: Text, bytes: Num }` or
`{ kind: "binary", bytes: Num }`; binary bytes require an explicit host
capability.

Providers resolve observations on demand and cache every observed branch or
leaf for the lifetime of the bound dictionary. The value is read-only. A child
receives it through a normal compatible parameter binding; it is not ambient
or automatically inherited. The provider is host-owned and cannot be recreated
from serialized interpreter state. Reads remain ordinary tool actions in the
execution trace, and leaf values are checked against `T` when observed.

### 5.2 write

`write` changes the lambda's typed value tree; it never writes a host file. An
application can expose a declared filesystem effect, let trusted crisp code use
its host filesystem surface, or have natlang return a typed change plan such as
`{ path: Text, text: Text }[]` for the owning host to validate and commit. The
standard host libraries provide path-contained plan validators and Node/Python
commit helpers; these are library facilities rather than language primitives.

`write(path, type, value)` creates or replaces the node at `path`, which is
`return`, a part of it, or a local. `type` must fit the slot; for an existing
slot it is checked against the destination type; for a new local it is the
interpreter's statement of what the local holds. Typed decoding can constrain
proposals further; runtime validation remains authoritative. `value` is the
complete value. A single JSON-text layer can be parsed where the boundary
supports it; arbitrary `{"value": X}` wrappers are rejected unless the declared
type expects that record. Existing Text is preserved verbatim.

`write(path, type, source=<path>)` copies an existing value that fits instead
of re-emitting it. Copies are recorded in provenance.

A `type` naming a pending node (`Lambda<...>`, `Map<...>`, a task) is rejected
(`anonymous-lambda`).

### 5.3 Changing a function: copy, edit, call the copy

`write(path="let/strict", type="Function<is_urgent>")` puts an editable copy of
a code-base function into a local. `edit(path="let/strict/instructions", ...)`
changes it. `call(function="let/strict", ...)` instantiates from the copy,
which keeps the signature and the code base of the original. This is the only
form of authoring, and it is for adapting a function, not for inventing
structure: specialization belongs in arguments.

### 5.4 edit

`old` must occur exactly once in the text at `path` and is replaced by `new`;
an empty `new` deletes it (a whole line, if `old` was one). Editable texts: the
instructions of function copies held in locals (§5.3). A lambda's own
`instructions` are immutable while it runs; what is done is state (§7.3).

### 5.5 call

`call` places an instance of a function and runs it in one action, blocking
until it completes or quiesces.

```
call(function="summarize", to="return/summary", inputs={ tickets: "let/urgent" })
call(function="classify", to="let/labels", over="args/tickets", inputs={ rubric: "args/rubric" })
call(function="add_line", to="let/total", over="args/lines", init=0)
call(function="shorten", to="return", init="let/draft", until="is_short", max=3)
call(function="classify", to="let/labels")            # resume what did not finish
```

All rules are type-level and enforced before anything enters the tree:

- `function`: a name of the code base, or `let/<copy>`.
- `to`: `return`, a part of it, or a local. A new local gets the type derived
  from the signature (`R`, `R[]` for `over`, the type of `acc`, the state
  type); the interpreter states no type.
- `inputs` maps parameters to paths of existing values that fit them;
  `values` supplies typed literals instead. A parameter cannot appear in both.
  Artifact names and native IDs are literals unless they really are interpreter
  paths. Unknown, duplicate, and missing bindings are rejected; literals are
  validated against the callee signature.
- `over`: a Map (§4.1). `over` + `init`: a Fold (§4.2); `init` is a literal or
  a path. `init` + `until` + `max`: an Iterate (§4.3); `max` is mandatory.
- **Resume.** The same function, unfinished, already at `to`, and no other
  arguments: only what did not finish runs again. With other arguments, the
  call replaces what is there.
- The outcome is `done` with a one-line rendering of the value, or `quiesced`
  with the note (for a combinator also `<n> of <m>` and the stuck slots).

### 5.6 run_code

Runs code in the selected executor (§9) and returns the value of the final
expression as the tool result. Engine-selecting surfaces require `engine`;
Python's compatibility surface can retain an implicit QuickJS binding. It can read `args` and `locals`, and can cause declared effects.
It **cannot write to the tree**: the interpreter writes what it learned.

### 5.7 End of an episode

An episode (§6) ends when:

1. the interpreter **replies** in prose. If `return` is complete and fits, the
   lambda completes; the reply is kept as a note and is **never parsed as the
   result**. With caller validation feedback an incomplete return quiesces with
   diagnostics. With local feedback the model receives missing-return nudges
   (currently at most two), then quiesces if still incomplete. Or
2. the interpreter calls `report_blocker` or `report_error`: the lambda quiesces
   with the supplied diagnostic as its note. Or
3. a budget is exhausted, or the process crashes.

### 5.8 Decoding constraints

Where the inference engine allows it, each turn is decoded under a grammar
built from the turn's tool schemas, over the model's **native** call text:

```
root ::= "<|tool_call_start|>[" call (", " call)* "]"  |  reply
```

The model still chooses between calling and replying, and which call to make.
Constraints depend on the selected decoder policy; typed alternatives provide:

| Part | Constrained to |
|------|----------------|
| `read.path`, `start`, `end` | existing paths; positions that exist |
| `write.path` + `type` + `value` | one alternative per writable slot: the slot's type as a constant and the grammar of `Draft<type>`; or a new `let/<name>` |
| `write.source` | paths whose type fits the slot |
| `write` `Function<f>` | names of the code base |
| `edit.path` | editable texts |
| `call.function`, `until` | names of the code base and of function copies; check functions |
| `call.inputs` | parameter names and workspace-path strings; runtime checks path existence and type fit |
| `call.values` | literal schemas derived from parameter types; runtime checks complete bindings |
| `call.over` | paths of lists |
| `call.to` | writable slots, or a new `let/<name>` |

Tools carry `x-natlang-alternatives`: argument sets that belong together,
which JSON Schema cannot express at the top level. `None` in an optional field
means "does not apply" and is read as absent. An argument name must not be a
Python keyword. A call that fails validation returns a rejection with a one-line
hint. With local feedback it is delivered to the same trajectory; with caller
feedback the episode quiesces and reports it outward. Browser applications default
to local feedback; low-level hosts and Python ToolAgent default to caller.
Failures are never silently retried: an operation can have
performed effects before failing. Every attempt spends its budget. For engines that parse tool calls themselves, a decoder may rename tools
per model (`call` -> `call_function` for a server whose call format reserves
the word).

---

## 6. Episodes, context, and validation

### 6.1 Episode

One continuous agent run on one ordinary lambda. It begins when the instance
is started by `call` (or by the host, for the root) with the opening request
and listing (§8.1), proceeds as tool calls and results, and ends per §5.7.
Crisp functions and combinators have no episodes.

### 6.2 The context is a cache

Everything durable is in the tree. An episode's context may be dropped at any
time, and a new episode on the same node must be able to continue from the
tree alone. Such a **cold restart** receives the same opening observation as
any other episode, plus the effect journal if it is not empty.

The caller keeps its context while blocked in `call`.

### 6.3 Budgets

**There is no recursion (§3.4).** Because the instructions of a function copy
can be edited, run-time rules stand behind the load-time check: a function
that is already running above a call is not started again (`recursion`), and:

1. *No identical child.* A lambda is not started if its instructions, args,
   and type are identical to those of a lambda already being reduced above
   it; it quiesces with a note. (The same idea as cycle detection in
   `Iterate`.)
2. *Optional run budgets.*

Episode count, call depth, work actions, tool calls, model turns, generated
tokens, and elapsed time have no default cap. An embedding may set positive
limits explicitly. An exhausted explicit limit quiesces the affected function
with a budget note. `mark_done` spends a tool call but no work action. Backends
without token usage are charged their requested allowance when a token limit
is configured. Failures remain visible and count toward configured limits.
Nested native episodes inherit an explicit caller deadline. Host transports
may still fail independently; a transport failure is not a successful result.

### 6.4 Commit points

Full type conformance (not draft) is required at:

1. completion of a lambda;
2. a `call`: every required parameter bound, every input fitting;
3. an effectful call: arguments fit the capability's signature.

A failed commit is refused and changes nothing. Its blocking diagnostics are
listed in full in the result.

### 6.5 Validation report

`validate(tree, path, mode)` is a pure function run after every action and
available for any snapshot. A diagnostic is
`{ path, code, severity, expected, got }` with severities `reject`,
`blocks-commit`, `hole`. Messages have a fixed form:

```
return/3/confidence: expected Num, got Text "high"
return/7: hole, missing label
```

Diagnostic codes (v0.2):

| Code | Severity | Raised when |
|------|----------|-------------|
| `no-such-path`, `bad-range` | reject | the path or range does not exist |
| `out-of-scope` | reject | the path leaves the lambda's subtree |
| `not-writable` | reject | harness-owned, the lambda's own `args`, or the code base |
| `frozen` | reject | the `args` of a triggered lambda |
| `type-mismatch` | reject | the value does not fit the stated type |
| `type-does-not-fit-slot` | reject | the stated or source type does not fit the slot |
| `unknown-field` | reject | the record type has no such field |
| `reserved-key` | reject | a `$`-prefixed key used as data |
| `eval-cannot-write` | reject | `run_code` attempted to modify the tree |
| `effect-undeclared` | reject | capability not in the lambda's `effects` |
| `effect-wider-than-parent` | reject | a child declares a capability its parent lacks |
| `no-such-function` | reject | `call` or `Function<f>` names nothing in the code base |
| `bad-call` | reject | a required parameter is unbound, or `over` / `until` / `max` do not suit the signature |
| `anonymous-lambda` | reject | a `write` whose type is a pending node |
| `recursion` | reject (at load) | a function can reach itself |
| `unbound-param`, `unbound-part` | blocks-commit | starting a partial instance |
| `commit-holes` | blocks-commit | completing with holes in `return` |
| `commit-pending` | blocks-commit | completing with pending nodes in `return` |
| `hole` | hole | a missing required field while drafting |

---

## 7. How an interpreter should work (normative for the reference policy)

The harness does not enforce this section. The reference policy follows it,
and training data teaches it. The structure of the work is **given by the
pseudocode**; the interpreter's job is to carry it out faithfully, one step at
a time.

### 7.1 Reading the program

Take the statements in order. For each one, decide what kind it is:

| Pseudocode | What to do |
|------------|-----------|
| `x = f(a, b)` | `call` f with `inputs`, `to` the local `let/x`, or straight to the part of `return` that the value is for |
| `xs = for each t in list: f(t, ...)` | **one** `call` with `over`. Never one call per item, never the items' work by hand |
| a value carried along a list | `call` with `over` and `init` |
| `repeat ... until check(x)`, at most n times | one `call` with `init`, `until`, `max`. Never unroll |
| `if c: A else: B` | work out `c` (look at a value, think, or `run_code` when it is exact), then carry out **only the branch taken** |
| exact glue no function covers (a comparison over a list, a count, arithmetic, a filter by a computed value) | `run_code`, then `write` the result. Never estimate |
| a small prose step of this function ("write one sentence about ...") | do it and `write` the value |
| `return { a, b }` | make sure each part of `return` holds its value: results that are only returned are called straight `to` `return/<field>`; values that exist already are written with `source` |

A program without functions is a **leaf**: read what is needed, then `write`
the complete answer in one call. Leaves are where judgment lives: classify,
judge, extract, rewrite.

### 7.2 Binding

Bind every parameter by path, choosing among the values whose type fits. For
`over`, leave out exactly the parameter that takes the item. Pass values;
never restate data in a call.

### 7.3 Keeping the place

**What is done is state, not text.** The instructions of a running lambda
never change. The lambda carries `marks`: for each line of its instructions,
`done` or `skipped` (or nothing: still to do). Marks survive swap-out like
`let` and `return`.

- The request shows the body as a numbered listing with a box per line:
  `[ ]` still to do, `[x]` done, `[-]` did not apply (the branch of an `if`
  that was not taken; everything after a `return` that was). Blank lines and
  comment-only lines have a number and no box. Numbering parses nothing.
- `mark_done(start, end?, skipped?)` marks a line or a range. Marking is
  separate from acting: a line may need several actions before it is marked,
  and one mark may close several lines. Its result shows the finished
  stretch collapsed (`1-6 [x]`) and the next open lines, so what remains to
  read is what remains to do.
- Three equivalent ways to mark, all in the corpus: `mark_done` in a turn of
  its own; `mark_done` in the same turn as the *next* action (no extra turn);
  the `done` argument of the `write` or `call` that finishes a line (applied
  only if that action succeeds). A line is never marked in the same turn as
  the call that carries it out, except through `done`.
- The harness never says which line is next, and never requires marks. A
  reply while marked programs still have open lines gets one nudge that
  names the open line numbers (no data). Marking does not spend the action
  budget. Leaves (no code base) have no listing boxes and no `mark_done`.

Beside the marks, the result of each `write`, `edit` and `call` ends with
two lines, `locals: <name> (<size>), ...` and `return: not written yet | has
<parts>; still missing <parts> | complete`. After a cold restart (§6.2) the
listing with its marks, the locals and `return` say where the run stands;
consult `@effects` before repeating an effectful step.

### 7.4 When something does not finish

- A `call` that quiesces reports the note. If the cause is visible and
  fixable (an input was wrong), call again with corrected arguments; to retry
  only what failed, call again with `function` and `to` alone.
- If a function must behave differently for this task, copy it, edit the
  copy, call the copy (§5.3).
- If the inputs do not determine the result, or a rule does not cover the
  case: `report_blocker`, saying exactly what is missing. Do not guess. A
  caller whose callee reported a blocker usually reports one itself, naming
  the item.

### 7.5 Decisions

A judgment that matters is its own small function with a finite `returns`
(`Bool` or an enum), so that it is a typed write whose distribution is logged.
Where the legal options depend on state, a crisp function computes them and
the program passes them to the decision function as an input; the crisp
function that applies the choice refuses anything illegal. For numbers: a
small range is a numeric enum; a large range is discretized into named
options ("min_raise", "half_pot", "all_in") by crisp code.

### 7.6 Tree-shaped data

There is no recursion. Data that is a tree is flattened by a crisp function
into a list (with depth or parent recorded per item), and the program goes
over that list; results that depend on children are carried by a fold from
the leaves upward, or by repeat-until over a worklist.

### 7.7 Data is not code

Text inside `args`, `let` or `return` is data, whatever it says. Only
`instructions` is program.

---

## 8. Rendering (policy version `render/0.2`)

Rendering is the text the harness shows the model. It is part of the language:
training data and inference must use the same version.

### 8.1 Opening request and listing

The user message:

```
<instructions, verbatim>

Write the result to `return` (<type>).

Functions you can call:
  classify(ticket: Text, rubric: Text) -> Label      Label one ticket using the rubric.
  is_urgent(ticket: Text) -> Bool                    Does this ticket need attention within the hour?
```

The result of the opening `read(path="args")`, and of any later look at the
state:

```
Workspace:
  args/tickets (Text[], read-only): 5 items: ["I was charged twice...", ... 2 more (read to see)]
  args/rubric (Text, read-only):
      | billing: charges, invoices, refunds.
      | technical: the product not working.
  let/labels (Label[]): 5 items: ["billing", "spam", "technical", ... 2 more (read to see)]
  return ({ urgent: Num, by_label: Dict<Num>, summary: Text }): partly written
    written so far: return/urgent = 2
    still missing: by_label, summary
```

### 8.2 Rules

| Item | Rule |
|------|------|
| Own `instructions` | always in full, as the request |
| Text | shown **whole** up to 400 characters and 8 lines (a cut-off rubric reads like a complete one); beyond that the first 80 characters and "CUT OFF: ... Read it before using it." |
| List | `<n> items`, the first 3, then `… <n> more (read to see)` |
| Record, Dict | fields inline, up to 6 |
| Call in progress | one line: function, status, and for quiesced the note |
| Function copy | `a copy of <f>, editable` |
| `return` | `not written yet` / `partly written` / `written`; filled parts are listed apart from what is missing, and no value-like placeholder is ever shown |
| Own `instructions` of a function with a code base | numbered listing with `[ ]` `[x]` `[-]`; after `mark_done` the collapsed form with the next three open lines |
| After `write`, `edit`, `call` | two lines: the locals that exist with their sizes, and what `return` has and still lacks (§7.3) |
| Nudges and hints | never carry data |

Target: an opening exchange of at most 1,500 tokens.

---

## 9. TypeScript: run_code and crisp functions

### 9.1 Environment

Crisp execution is selected independently from the language and model. The
embedding provides an engine registry; authored functions can declare `engine`.
The canonical runtime uses `typescript-host` for crisp functions and model eval,
including shared native host objects. Historical Python traces may declare
`quickjs-isolated`; that compatibility label does not select the canonical executor.
Fresh or retained globals describe lifetime, not isolation. Shared-host evaluators are
trusted execution and can mutate native objects before validation fails. Their
portable results still cross the typed tree boundary; traces cannot replay
arbitrary native memory. The following limits describe the QuickJS binding only.

QuickJS executes JavaScript and the erasable TypeScript subset. Type annotations
are stripped with Node.js `module.stripTypeScriptTypes` (Node >=22.13 required
only for TypeScript syntax); enums and other syntax requiring code generation
are rejected. Stripping is not static type checking: the current runtime checks
values at the typed tree boundary and crisp function returns, not arbitrary
intermediate JavaScript expressions. The scope declarations below describe the
API; no declaration file or static checker is currently generated.

Pure execution has a 2-second QuickJS time limit. Effectful execution runs in a
killable worker with a 2-second wall-clock deadline, including host-effect waits.
Both use a 64 MB QuickJS heap limit. TypeScript parsing has a separate 5-second
limit and cached results. Capabilities run in the trusted Python host so their
state is preserved. A host callback that times out can still complete; its
outcome is uncertain, and hosts must supply cancellation or idempotency for
external effects. The interpreter never automatically retries failed effects.

### 9.2 run_code scope

```ts
declare const args: P;          // the bound parameters
declare const locals: { ... };  // the plain values in `let`
declare const self: {
  readonly instructions: string;
  readonly args: P;
  readonly return: DeepPartial<T>;
};
```

Pending nodes appear as descriptive `$pending` objects, not usable results.
The runtime copies scope values into QuickJS; lists are ordinary arrays. A result
must pass tree validation before it can be stored. The value of the final
expression is the tool result.

### 9.3 Crisp function body

`code` is the body of a function `(self, args, fx) => T`, where `args` is
typed by the signature. `run_code` has the same `args` and `self` in scope, so
there is one environment to learn.

```ts
/*---
description: How many flags are true.
args:
  flags: Bool[]
returns: Num
---*/
return args.flags.filter(Boolean).length
```

A crisp function runs without an episode; large values land in their slot
without passing through the model's tokens.

### 9.4 Standard library

Pure helpers available in both contexts. v0.2 set:

`count, sum, min, max, mean, sortBy, groupBy, countBy, uniq, uniqBy, zip,
range, chunk, windows, flatten, take, drop, partition, topK, indexBy,
splitLines, splitOn, joinWith, trim, lower, upper, contains, startsWith,
matchAll, replaceAll, wordCount, parseNum, parseDate, formatDate, daysBetween,
hash, enumOf`

### 9.5 Effects

**Capabilities are registered by the host** (§10). The current runtime does
not generate a static TypeScript declaration file for callbacks. `out.emit` is
provided by the core; an embedding registers other operations; a simulated
environment for tests or training data is simply a host that registers
deterministic in-memory fakes (a ticketing system, an inventory). The
language does not distinguish real capabilities from fake ones.

Effects are reached through `fx.<capability>.<function>(...)`. A call is
allowed only if the enclosing lambda declares the capability in `effects`,
and a callee may not declare a capability its caller lacks. `out.emit` writes a
record to the program's output stream. File, network, database and process
operations depend on host registration. Direct shared `host` access follows
its environment's authority and does not automatically enter the `fx` journal;
absence of declared effects alone cannot establish purity in such an environment.

Every effectful call is appended to the lambda's **effect journal** before it
executes: `{ seq, capability, function, args_hash, args_preview, status }`,
with status updated after the observation. The journal is readable at `@effects`.
A crash can leave an outcome unknown. Exactly-once delivery and safe retry require
host operation identity and reconciliation; the journal alone guarantees neither.

---

## 10. The I/O boundary

A **program** is a code base with an entry function (`main.nl` plus `main/`,
§3.4), or a single YAML file with an inline code base (§11). The host binds
the entry function's parameters and receives its value. A reactive program is
an entry function applied as a `Fold` step over an open list of events.

The reference Python host exposes this boundary directly (the installed native
`natlang` command instead accepts a single JSON object with `--inputs`):

```
python -m natlang run triage/main.nl --in tickets=./tickets/ --in rubric=./rubric.md \
                                      --out ./result.yaml
python -m natlang run shop/main.nl   --stream events=stdin:jsonl
```

Import is type-directed against the declared parameter type:

| Source | Parameter type | Result |
|--------|----------------|--------|
| `.md`, `.txt` file | `Text` | the text |
| `.json`, `.yaml` file | any | parsed against the type |
| directory | `T[]` | one item per file, sorted by name, each imported against `T` |
| directory | `Dict<T>` | keyed by file stem |
| `--stream NAME=SOURCE` | `T[]` | an **open list** fed by the source |

Export writes the root's value as YAML or JSON, or as a directory (`Text` to
`.md`, records to `.yaml`, lists to numbered files). Reading or writing the
outside world *during* a run is an effect (§9.5).

**The host** is whatever invokes a program: the command-line tool, or a client
library that embeds natlang inside another (crisp) program. The host owns
everything about a program's surroundings: binding parameters and streams,
registering capabilities, budgets, how long a long-lived `Fold` runs, how
large its state may grow, and what happens at shutdown. None of these are
language concerns.

Shared libraries are ordinary folders of functions, linked with `uses`.

---

## 11. Serialization

A tree is serialized as YAML, with the root wrapped:

```yaml
$lambda:
  type: 'Lambda<{ tickets: Text[], rubric: Text }, Label[]>'
  types:
    Label: '"urgent" | "normal" | "spam"'
  instructions: |
    labels = for each t in tickets: classify(t, rubric)
    return labels
  codebase:
    classify:
      description: Label one ticket using the rubric.
      args: { ticket: Text, rubric: Text }
      returns: Label
      instructions: Pick the label from `args/rubric` that fits `args/ticket`.
```

A quiesced or partially reduced tree serializes the same way, with `args`,
`let`, `return`, `function`, `status`, `note`, and `effects_journal` keys present (and `acc`/`at`
or `state`/`iteration` for combinators), so that a program in execution can
be saved and shipped. Resumption also needs the compatible runtime and evaluator
bindings; native resources and external effects are not reconstructed from YAML. The rest of provenance is
stored separately.

---

## 12. Provenance

For every completed pending node the harness keeps: the node as it was at
call time (function, body, `args`), its locals, the trace of tool calls and results, step and token
counts, the effect journal, the logged distributions of finite-typed writes,
and links to earlier attempts (resumed and repeated calls). Trace retention is an embedding concern; there is no automatic 1,000-record
retention guarantee. Configure storage and pruning for long-lived runs.

---

## 13. Decisions recorded in this document

| § | Decision |
|---|----------|
| intro, 3.4 | Programs are pseudocode code bases; the author states structure, the interpreter carries it out; **no anonymous lambdas** |
| 3.1 | The lambda holds every zone of state: type, body, `args`, `let`, `return`, `codebase` |
| 3.2 | Typed locals created by the first write; private; no fixed count limit |
| 3.4 | `.nl` / `.ts` files with frontmatter, companion folders, lexical scope, `uses` links, immutable and shared by reference; no recursion |
| 5 (historical) | Eight tools: `read`, `write`, `edit`, `run_code`, `call`, `mark_done`, `report_blocker`, `report_error`; retained for old traces only |
| 5.3 | Changing a function = copy into a local, edit, call the copy |
| 5.5 | `call` places and runs in one action; calling again resumes; Map / Fold / Iterate are reached only through `call` |
| 4.1, 4.3 | Combinators bind the author's parameter names; Iterate's check is a Bool function of the code base |
| 2 | TypeScript type syntax; `Dict<T>`; numeric literal types, no range refinement; literals widen to their base type |
| 2.1 | Lists and dicts are covariant |
| 5.8 | Constrained decoding over the model's native call text; grammar alternatives tie path, type and value |
| 6.3 | Agentic run budgets default to unlimited and may be explicitly configured; no fixed pending-node nesting limit |
| 6.5 | The diagnostic code list |
| 8 | `render/0.2`: short texts whole (400 characters, 8 lines), lists preview 3, no value-like placeholders |
| 9 | The bound-parameter part is **`args`**; `run_code` sees `args` and `locals` and cannot write |
| 9.5, 10 | Capabilities are host-registered with typed signatures; state size, lifetime, and shutdown are host concerns |

## Appendix A. Structured harness checks

`tests/test_structured_session.py` checks typed writes, incomplete drafts,
copy ranges, and read-only input through the tool surface in §5.
