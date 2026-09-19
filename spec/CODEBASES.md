# Code bases, calls and locals (proposal for SPEC v0.2)

Status: **draft for discussion**, 2026-09-19. Nothing here is implemented yet. It
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
| `authoring` | `free` allows model-written sub-tasks (§4.3). Default: off |

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

The interpreter creates a sub-task by **instantiating a function of its code
base**. The function is named in the type of the write; everything else is
derived and checked by the harness.

```
write(path="let/labels", type="Map<classify>",
      value={ over: "args/tickets", inputs: { rubric: "args/rubric" } })
write(path="let/summary", type="Call<summarize>", value={ inputs: { tickets: "let/urgent" } })
write(path="let/total",  type="Fold<add_line>",  value={ over: "args/lines", init: 0 })
write(path="let/draft2", type="Iterate<shorten>", value={ init: "let/draft", until: "word_count", limit: 60, max: 3 })
run(paths=["let/labels"])
```

Rules (all type-level, all enforced at write time):

- `Call<f>`: the slot must accept `f`'s return type. `inputs` maps each of
  `f`'s parameters to a path whose value fits it, or `params` declares it as
  produced later (SPEC §3, continuations). Unknown or missing parameters are
  rejected with the signature in the hint.
- `Map<f>`: the slot must accept `R[]`. Exactly one required parameter of `f`
  is left unbound by `inputs`; it receives the item, and must accept the
  element type of `over`.
- `Fold<f>`: `f` has parameters named `acc` and `item`, and returns the type of `acc`.
- `Iterate<f>`: `f` maps the state type to itself. The check is a function of
  the code base returning `Bool` or `LoopVerdict`, applied to the state.
  (Open: whether a one-line code expression is also allowed as the check.)
- Under constrained decoding the function names, the parameter names and the
  candidate paths are enums. The model chooses; it does not spell.

What the model is shown: a listing, not the bodies.

```
Functions you can call:
  classify(ticket: Text, rubric: Text) -> Label      Label one ticket using the rubric.
  is_urgent(ticket: Text) -> Bool                    Does this ticket need attention within the hour?
  select_by_flags(items: Text[], flags: Bool[]) -> Text[]   The items whose flag is true.
```

`read(path="codebase/classify")` shows a body when the interpreter needs it.

### 4.1 What the interpreter still does itself

- Small steps of its own function ("write one paragraph about ...",
  a condition such as `if urgent is empty`): by reading, thinking, and a plain
  `write` to a local or to `return`.
- Exact glue the author did not name a function for (`l is not "spam"` over a
  list): `run_code`, then a plain `write` of the result.
- Conditionals: evaluate the condition, then carry out only the branch taken.
  The harness knows nothing about branches.
- Bookkeeping in its own `instructions`: deleting finished steps, substituting
  results (SPEC §7.2). Unchanged.

### 4.2 Editing a copy

After instantiating `f`, the interpreter may edit the instructions of the copy
before running it. Allowed, discouraged: specialization belongs in arguments.
The reference policy never does it.

### 4.3 Free authoring

`Task<T>`, `Code<T>`, and `Map<A, B>` with model-written instructions (the
whole of today's surface) are available only in a lambda whose definition says
`authoring: free`. Default off: a prose program without a code base is a
leaf, and is answered whole. The capability stays in the language and in a
slice of the training corpus.

## 5. Open points

1. Iterate's check: function only, or also a code expression?
2. Does `authoring: free` inherit into the code base of the function that declares it? (Proposed: no.)
3. Listing budget: how many functions may a code base have before the listing must be paged? (Proposed: 12.)
4. A generic `std/`: which functions, and is it linked implicitly? (Proposed: explicit `uses` only.)
5. Pseudocode dialect: none is normative. The corpus renders the same program
   in several (Python-like, numbered steps, structured prose).
