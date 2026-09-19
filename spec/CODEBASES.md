# Code bases, calls and locals (proposal for SPEC v0.2)

Status: **agreed and implemented** (2026-09-19): `natlang/codebase.py`, `Session._place_call`,
`tests/test_codebase.py`. Not yet done: serializing `codebase` on swap-out, `max_depth` enforcement at run time,
`types.ts` beyond simple aliases. It
extends SPEC.md §3 (lambdas) and §5 (the tool surface); where they disagree,
this document is the proposal and SPEC.md is what runs today. A worked example
is in `examples/triage/`.

## 1. Why

The programs we have are one to three sentences of prose. Whatever structure
an execution needs, the interpreter must invent: the hardest thing to ask of a
350M model, and something a 27B has no reason to do. natlang programs are
meant to be **pseudocode**: functions with typed signatures, subroutine calls,
loops, conditions, local variables. The author states the structure; the
interpreter *carries it out*. The harness still parses no instructions and
holds no cursor: reading the pseudocode and deciding the next step remains
the model's job.

## 2. The lambda, complete

Every zone of state an episode can touch is a part of the lambda:

| part | who writes it | notes |
|---|---|---|
| `type` | author | `Lambda<Args, R>`, from the frontmatter |
| `instructions` \| `code` | author; the interpreter may edit its own `instructions` | pseudocode text, or TypeScript for a crisp function |
| `args` | the caller | read-only to the lambda itself, frozen on trigger |
| `let` | the interpreter | **new.** Typed locals, `let/<name>` |
| `return` | the interpreter | the result |
| `codebase` | author | **new.** Immutable. name -> function definition |
| status, note, provenance | the harness | |

Swap-out serializes the lambda and nothing else.

### 2.1 `let`

- A local comes into existence by the first `write` to `let/<name>`. The
  write carries its type, like every creating write (SPEC §5.2). When the
  write is a call (§4), the type is derived from the callee's signature and
  the interpreter does not state one.
- A local may hold a plain value or a pending sub-task, exactly like a slot
  inside `return`. `run` reduces it in place.
- Locals are private. A callee sees only what the caller passed as `inputs`.
- Locals survive completion (they are provenance), and are not part of the
  result.
- Bounds: at most 16 locals per lambda; names are identifiers.

## 3. Code bases

### 3.1 On disk

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
std/
  count_true.ts           a crisp function
```

- `name.nl`: natural-language function. YAML frontmatter, then the pseudocode.
- `name.ts`: crisp function. The same frontmatter inside a leading
  `/*--- ... ---*/` comment, then a TypeScript function body (SPEC §9.3).
- `types.ts` in a folder (optional): type declarations shared by every file
  in that folder.

Frontmatter:

| key | meaning |
|---|---|
| `args` | name -> type. Order is the order of the pseudocode signature |
| `returns` | type |
| `types` | name -> type, local to this function (and inherited by its code base) |
| `description` | one line; this is what a caller sees in its listing |
| `uses` | name -> relative path of a function defined elsewhere (a link) |
| `effects` | capabilities (SPEC §9.5) |
| `recursive`, `max_depth` | required when the function can reach itself (§3.3) |

### 3.2 Scope is lexical

The code base of a function is: the functions in its companion folder, plus
its `uses`. Nothing else: not its siblings, not its caller's code base. A
function instantiated from `main/summarize.nl` carries `main/summarize/` (and
its own `uses`) wherever it runs.

Reuse is by **linking**: `uses: { word_count: ../../std/word_count }`. OS
symlinks inside a companion folder are accepted on load and treated the same
way; `uses` is the canonical form (it survives git, archives and Windows).

In the tree, a code base is shared by reference. "Copying" a function
instantiates a fresh lambda that points at the same immutable definition; no
function body is duplicated.

### 3.3 Recursion

A function can reach itself only if the author linked a cycle. The loader
finds cycles and rejects the code base unless every function on a cycle
declares `recursive: true` and a `max_depth`. So recursion is visible,
deliberate and bounded at load time; everything else iterates through Map,
Fold and Iterate. The run-time guards (identical-child, nesting, run budgets)
remain: instruction edits make anything possible in principle.

### 3.4 Inline form

The YAML program format gains `codebase:` inside `$lambda`, name -> `$lambda`
or `{ $link: path }`, so conformance programs stay single files.

## 4. Calls

There are **no anonymous lambdas**. Every sub-task is an instance of a function
of the acting lambda's code base, started with one tool, `call`, which places
the instance and runs it in one action:

```
call(function="classify", to="let/labels", over="args/tickets", inputs={ rubric: "args/rubric" })
call(function="summarize", to="return/summary", inputs={ tickets: "let/urgent" })
call(function="add_line", to="let/total", over="args/lines", init=0)
call(function="shorten", to="return", init="let/draft", until="is_short", max=3)
call(function="classify", to="let/labels")            # again, arguments omitted: resume what did not finish
```

Rules (all type-level, enforced before anything enters the tree):

- `function`: a name of the code base, or `let/<copy>` (§4.2).
- `to`: `return`, a part of it, or a local. A new `let/<name>` is created
  with the type derived from the signature; the interpreter states no type.
- `inputs` maps parameters to paths of existing values that fit them.
  (`values` may give small literals instead.) Unknown and missing parameters
  are rejected with the signature in the message.
- `over` (Map): exactly one required parameter is left unbound; it receives
  the item and must accept the element type. The result is `R[]`.
- `over` + `init` (Fold): the function has parameters `acc` and `item` and returns the type of `acc`.
- `init` + `until` + `max` (Iterate): one parameter is left unbound and
  receives the state; `until` names a code-base function of one parameter
  returning Bool, applied to the state after every round; `max` is mandatory.
- The same function, unfinished, already at `to`, and no other arguments: the
  instance **resumes**. Only the failed items of a Map run again; a Fold or an
  Iterate continues where it stopped. There is no separate `run`.
- Under constrained decoding the function names, the parameter names, the
  candidate paths (only those whose type fits) and the check functions are enums.

What the model is shown, after its instructions:

```
Functions you can call:
  classify(ticket: Text, rubric: Text) -> Label      Label one ticket using the rubric.
  is_urgent(ticket: Text) -> Bool                    Does this ticket need attention within the hour?
```

`read(path="codebase/classify")` shows a body.

The tool set: `read, write, edit, run_code, call, report_blocker` (`call` is
absent when the lambda has no functions: such a lambda is a leaf).

### 4.1 What the interpreter still does itself

- Small steps of its own function ("write one paragraph about ...", a
  condition such as `if urgent is empty`): read, think, plain `write`.
- Exact glue the author did not name a function for (`l is not "spam"` over a
  list): `run_code` (inputs in `args`, locals in `locals`), then a plain `write`.
- Conditionals: evaluate the condition, then carry out only the branch taken.
  The harness knows nothing about branches.
- Bookkeeping in its own `instructions`: deleting finished steps, substituting
  results (SPEC §7.2). Unchanged.

### 4.2 Changing a function: copy, edit, call the copy

The code base is immutable. `write(path="let/strict", type="Function<is_urgent>")`
puts an editable copy into a local; `edit(path="let/strict/instructions", ...)`
changes it; `call(function="let/strict", ...)` instantiates from the copy.
The copy keeps the signature and the code base of the original. This is the
only form of authoring: there is no way to create a function from nothing,
which also keeps recursion where the author put it (§3.3).

## 5. Open points

1. Iterate's check is a function (decided). 
2. Listing budget: how many functions may a code base have before the listing must be paged? (Proposed: 12.)
4. A generic `std/`: which functions, and is it linked implicitly? (Proposed: explicit `uses` only.)
5. Pseudocode dialect: none is normative. The corpus renders the same program
   in several (Python-like, numbered steps, structured prose).
