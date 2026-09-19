# natlang language specification

Version **0.1-draft**, 2026-09-19. Normative for the harness, the reference
policy, and all training data. Where this document and `PLAN.md` / `TYPES.md`
disagree, this document wins and the others should be corrected.

Items marked **[proposed]** are Phase 0 decisions made here for the first
time and are the ones most worth reviewing. Section 13 lists them.

Key words: **must**, **must not**, **may** are used in their plain sense.

---

## 1. The object tree

A program, its state, its inputs, and its results are one tree of typed
nodes. Every node has a type, fixed when the node is created.

There are two families of node:

- **Value nodes**: scalars, records, lists, dicts, blobs.
- **Pending nodes**: a `Lambda`, `Map`, `Fold`, or `Iterate`. A pending node
  stands where a value will be. Reducing it replaces it with that value.

### 1.1 Paths

A path is a sequence of segments separated by `/`. A segment is a record
field name, a dict key, or a list index (0-based). Inside an episode, paths
are **relative to the current lambda**, whose parts are `instructions`, `args`,
and `return`.

```
args/tickets/3/body
return/summary
return/args/flags/fn/instructions      # into a pending lambda's parts
```

- `+` as the final segment of a list path means "append": `return/labels/+`.
- **Ranges** are inclusive at both ends and use the numbers the rendering
  shows. Text lines are 1-based: `instructions[3..9]`. List items are
  0-based: `args/tickets[0..9]`.
- **Meta paths** use an `@` suffix and are read-only:

| Suffix | Gives |
|--------|-------|
| `@status` | `unreduced`, `running`, `quiesced`, `done` |
| `@note` | the closing note of a quiesced node |
| `@problems` | the validation report for the subtree |
| `@origin` | the provenance record of a value (the lambda that produced it, its `args`, its trace, earlier attempts) |
| `@effects` | the effect journal of the current lambda |
| `@dist` | the logged distribution of a finite-typed write |

### 1.2 Scope

A lambda can address only its own subtree. Anything it needs from outside
must have been placed in its `args` by its parent.

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
  drafts; renamed so that `Map` means only the combinator.)
- A union of literals is an enum. String literals name options
  (`"sell_pear" | "decline"`); **numeric literals [proposed]** give small
  numeric ranges (`1 | 2 | 3`). Enum narrowing (§2.1) applies to both.
- **Named types** are declared in a lambda's `types` block, are visible in
  that lambda's subtree, and may be recursive:
  `types: { Unit: "{ name: Text, reports: Unit[] }" }`.
- There is no `Any` and no type inference.
- v0.1 is **structural only**: no refinements, no postconditions.

### 2.1 Fit

Type `A` **fits** a slot of type `B` when:

1. `A` equals `B` structurally; or
2. `A` is a member of union `B`; or
3. `A` is an enum whose literals are a subset of enum `B`'s
   (**enum narrowing**, the only subtyping rule); or
4. `A` is a record and `B` is a record, and each field fits (optional fields
   of `B` may be absent in `A`); or
5. **containers are covariant**: `A[]` fits `B[]`, and `Dict<A>` fits
   `Dict<B>`, when `A` fits `B`. (Values are never mutated in place through
   an alias, so this is sound here.) Or
6. **promise rule**: `A` is `Lambda<_, T>`, `Map<_, E>` with `T = E[]`,
   `Fold<_, T>`, or `Iterate<T>`, and `T` fits `B`.

Rule 6 is what allows a pending node to sit wherever its result is needed.

### 2.2 Draft types

`Draft<T>` is `T` made deeply partial: any record field may be absent, any
list may be shorter than it will be. While a lambda is being reduced, its
`return` and the `args` of its unreduced children are checked against
`Draft<T>`. At a commit point (§6.4) they are checked against `T`.

**Holes are fine, lies are not**: a missing field is accepted and reported as
a hole; a value of the wrong type is rejected at write and never enters the
tree.

---

## 3. Lambdas

### 3.1 Parts

A node of type `Lambda<P, T>` has:

| Part | Meaning |
|------|---------|
| `instructions: Text` **or** `code: Text` | the body. Natural language for an ordinary lambda; TypeScript for a **crisp lambda**. Exactly one is present. |
| `args` | the bound parameters, typed by `P` |
| `return` | the result so far, typed `Draft<T>` until commit |
| `types?` | named type declarations. May be declared on any pending node, combinators included. |
| `effects?` | capabilities this lambda may use (§9.3). Absent means pure. |

and harness-owned metadata: status, step count, note, provenance.

There is **no working-state zone**. A value needed by a later step is a
parameter of the lambda that will consume it (§7).

### 3.2 Lifecycle

```
unreduced --reduce--> running --+--> done      (node is replaced by its value)
     ^                          |
     |                          +--> quiesced  (node left exactly as it was)
     +------ edit / re-trigger -+
```

- **Binding.** A lambda whose `P` has a required field with nothing in the
  corresponding `args` slot is *partial*. Partial lambdas are well-typed and can
  be copied and further bound. Triggering one is rejected.
- **Freeze.** When a lambda is triggered, its `args` becomes immutable. `args` is
  always read-only to the lambda itself. It becomes writable by the parent
  again if the lambda quiesces.
- **Dependencies first [proposed].** Triggering a lambda first reduces every
  pending node in its `args` whose slot has a value type, in parallel where
  independent. If any of them quiesces, the lambda is not started and the
  trigger reports which dependency is stuck. A parameter whose declared type
  is itself a pending type (a higher-order parameter such as a `fn`) is not
  reduced.
- **Completion** of an ordinary lambda requires: `instructions` is empty;
  `return` contains no pending nodes; `return` fits `T`. There are **no tail
  calls**: a `return` that is itself a pending node blocks completion until
  reduced.
- **Completion** of a crisp lambda: the code returned a value that fits `T`.
- **Swap-out.** On completion the node is replaced by its value. The lambda
  record moves to provenance and is reachable at `path@origin`.
- **Quiescence.** A reduction that stops without completing leaves the node
  in place, partial `return` included, with status `quiesced` and a note.
  Causes: the agent ended its episode with a closing message; a budget ran
  out; the process crashed; a crisp lambda threw. There is no separate
  failure state.
- **Data is never destroyed.** Values are superseded by later writes or
  archived to provenance. What reduction consumes is instruction text.

### 3.3 Reopen

`reopen <path>` (§5.7) turns a value back into the lambda that produced it:
the origin lambda is restored from provenance, the previous value becomes the
draft `return`, and new instructions may be supplied. Provenance links the
attempts.

**Only a value produced by a natural-language lambda can be reopened.** A
value the agent wrote itself can simply be set again; reopening it would only
hand the agent's own task to a child. (Observed with an untuned model: it
reopened its own result with a stray code fence as the instructions, reduced
it, and the child did the same.) Post-processing that needs a new task is a
continuation built outside-in, not a reopen.

---

## 4. Combinators

Combinators are pending node types with fixed semantics. They have no
instructions and get no episode; only their body lambdas do. The interpreter
constructs them, places them where their result is needed, and triggers them
with `reduce`, exactly as for lambdas.

### 4.1 Map

`Map<A, B>` with parts `over: A[]` and `fn: Lambda<{ item: A, ... }, B>`.
Fits a slot of type `B[]`.

- `fn` may have further parameters already bound (a rubric, a schema): it is
  a partially applied lambda whose only unbound required parameter is `item`.
  If `fn` declares `index: Num`, the harness binds it.
- **Trigger**: the node expands in place. Slot `i` of the node becomes a copy
  of `fn` with `args/item` bound to `over[i]`. Copies are copy-on-write. The
  harness reduces all slots, in parallel, batched. Result order is the order
  of `over`. Length is preserved by construction.
- **While expanded**, `<map>/i` addresses slot `i` (a value once reduced, a
  lambda otherwise), and `<map>/fn` and `<map>/over` remain readable.
- **Writing a slot by hand.** The parent may `set` or `copy` into a slot of an
  expanded Map, typically to repair a quiesced one. The slot then accepts any
  value that fits the *element type of the slot the Map occupies*, which may
  be wider than `B`. (`B` constrains what `fn` returns, not what the parent
  may put there.) When every slot is a value, the node is the list.
- **Partial result**: if any slot quiesces, the Map quiesces. Finished slots
  hold values. The parent may edit stuck slots and re-trigger; only slots that
  are not yet values run.
- Each slot has its own step budget.

### 4.2 Fold

`Fold<A, S>` with parts `over: A[]`, `init: S`, and
`step: Lambda<{ acc: S, item: A, ... }, S>`. Fits a slot of type `S`.

- **Trigger**: for each item in order, the harness instantiates `step` with
  `acc` and `item` bound, reduces it, and takes its value as the next `acc`.
  The model never threads the accumulator.
- **While running or quiesced**, `<fold>/acc` is the current accumulator,
  `<fold>/at` the index of the next item, `<fold>/current` the step lambda in
  progress.
- **Partial result**: if a step quiesces, the Fold quiesces at that index.
  Re-triggering resumes there.

### 4.3 Iterate

`Iterate<S>` with parts `init: S`, `step: Lambda<{ state: S, ... }, S>`,
`check: Lambda<{ recent: S[], iteration: Num, ... }, LoopVerdict>`, and
`max: Num` (required). Fits a slot of type `S`.

```
LoopVerdict = { reason: Text, verdict: "continue" | "done" | "degenerate" }
```

- After every `step` the harness runs `check` as a fresh episode over the
  last **3** states (`recent`, oldest first) and the iteration count. The
  record's field order is significant: `reason` is written before `verdict`.
- `done`: the current state becomes the value. `degenerate`: the Iterate
  quiesces with `reason` as its note. `continue`: next iteration.
- Crisp guards, independent of the model: reaching `max` quiesces the node; a
  state whose content hash equals an earlier state's is `degenerate`.
- `check` may be a crisp lambda.
- `<iterate>/state` and `<iterate>/iteration` are readable while it runs or
  is quiesced.

### 4.4 Open lists

A list may be **open**: a harness attribute meaning an external source keeps
appending to it. Its type is still `A[]`. A `Fold` or `Map` over an open list
does not complete while the list is open. A long-lived reactive program is a
top-level `Fold` over an open list of events, with outputs emitted as effects
inside `step`. Open lists are bound only at the I/O boundary (§10).

---

## 5. Actions

> **Model-facing surface (2026-09-19).** What follows in this section is the
> harness's internal operations and the **trace notation** used by the
> conformance suite. The model does not see it. The model works through native
> tool calls, surface `tools-v2` in `natlang/surface.py`: the instructions as
> the user's request, the workspace delivered as a tool result (data never
> shares a channel with instructions); five fixed tools (`read`, typed
> `write`, `edit`, `run_code`, `run`; a sub-task is a `write` with a sub-task
> type such as `Task<T>` or `Map<A, B>`, all types derived and forced) whose argument schemas are
> derived from the tree and types each turn; the episode ends when the model
> replies; the result is what was written to `return`, never the reply. Tool
> names map onto the operations below (`write` → set, `run` → reduce,
> `run_code` → eval, substitution `edit` → edit). This section will be
> rewritten around the tool surface once the surface has settled.

An agent turn is either one action or a closing message.

An action is wrapped in the model's tool-call tokens and consists of a
**header line** and, for some tools, a **body** that runs to the end token.
Nothing in a body is quoted or escaped.

```
<|tool_call_start|>
HEADER
BODY
<|tool_call_end|>
```

| Tool | Header | Body |
|------|--------|------|
| read | `read PATH` or `read PATH[RANGE]` | none |
| edit | `edit PATH[RANGE]` | replacement text; empty body deletes the lines |
| set | `set PATH : TYPE` | the value (§5.2) |
| unset | `unset PATH` | none |
| copy | `copy SRC to DST` | none |
| reduce | `reduce PATH PATH ...` | none |
| reopen | `reopen PATH` | optional new instructions |
| eval | `eval` | TypeScript |

### 5.1 read

Returns the rendering (§8) of the subtree or range. Meta paths are read the
same way.

### 5.2 set

Creates or replaces the node at `PATH`. `TYPE` is the node's type and **must**
fit the slot. The body is:

- for `Text`: the raw text, verbatim;
- for other value types: YAML in the subset of §5.3, parsed **against
  `TYPE`** (type-directed), so `label: no` in a `Text` field is the string
  "no";
- for `Lambda<P, T>`: a YAML mapping with keys `instructions` or `code`, and
  optionally `args`, `types`, `effects`. `P` and `T` come from the header;
- for a combinator: a YAML mapping of its parts.

**A node's type is stated exactly once.** For the node named in the header it
is the header's `TYPE`. A pending node nested inside a body is written with a
reserved wrapper key carrying its own `type`:

```
<|tool_call_start|>
set return : Lambda<{ flags: Bool[] }, Text>
instructions: |
  1. Count how many of `args/flags` are true.
  2. If more than 5, write an alert naming the count. Otherwise write "ok".
args:
  flags:
    $map:
      type: 'Map<Text, Bool>'
      fn:
        $lambda:
          type: 'Lambda<{ item: Text }, Bool>'
          instructions: Is this ticket urgent? Answer true or false.
<|tool_call_end|>
```

Reserved wrapper keys are `$lambda`, `$map`, `$fold`, `$iterate`. Keys
beginning with `$` are never valid record fields or dict keys. Parts left
unbound (here `over`) are filled later, usually by `copy`.

### 5.3 The YAML subset

Block and flow mappings and sequences; plain, single-quoted, double-quoted
scalars; literal block scalars (`|`). **Not allowed**: anchors, aliases,
tags, multiple documents, merge keys, folded scalars (`>`), complex keys.
JSON is valid input. A type written as a YAML value (the `type` of a nested
pending node, entries of `types`) is **always single-quoted**: types may
contain double-quoted literals and never contain a single quote.

### 5.4 unset

Removes an optional field, a dict entry, or a list item, or clears a draft
slot. Rejected on frozen or harness-owned paths.

### 5.5 edit

Line-addressed replacement in a `Text` node. **[proposed]** v0.1 has only
this form; search-and-replace and whole-text rewrite (`set PATH : Text`) are
the alternatives to measure in Phase 2.

### 5.6 copy

`copy SRC to DST`. `SRC` is a path or a range; `DST` is a path, possibly
ending in `+`.

- The type of `SRC` must fit the slot at `DST`. A line range of `Text` is
  `Text`; a slice of `A[]` is `A[]`; a record field has its field's type.
- Copies are O(1), copy-on-write. The copy's provenance records its source.
- Copying a pending node copies it as `unreduced`, keeping any partial
  `return`. A running node cannot be copied.
- There is **no move**. To carve a subroutine out of one's own instructions:
  copy the lines into the child, reduce the child, then delete the lines.

### 5.7 reduce and reopen

`reduce` triggers one or more pending nodes and **blocks** until every one
has quiesced or completed. Several paths in one action are reduced in
parallel. **[proposed]** v0.1 has no non-blocking form and no `wait`; the
parent's episode is suspended, so parent and children never write
concurrently.

The result reports, per path, one of:

| Outcome | Meaning |
|---------|---------|
| `done` | replaced by its value; a one-line rendering follows |
| `quiesced` | left in place; the note follows. For a combinator, also `<n> of <m> reduced` and the stuck slots |
| `replaced` | a crisp lambda returned a pending node (§9.3), which now occupies the slot, `unreduced`. The agent decides whether to trigger it |
| `refused` | the trigger failed a commit check (unbound part, stuck dependency) |

### 5.8 eval

Runs TypeScript (§9) and returns the value of the final expression as the
tool result, rendered under §8 limits. `eval` can read the tree and can cause
declared effects. It **cannot write to the tree**.

### 5.9 Closing message and end of episode

An episode (§6) ends when:

1. the agent empties `instructions` and the commit check passes: the harness
   ends the episode at once; or
2. the agent says it is stuck: header `stuck`, with the closing message as the
   body. The message is stored as the node's note and the node is `quiesced`.
   (Under constrained decoding every turn is an action, so this is one too.
   It is not one of the eight tools: it changes nothing in the tree.) Or
3. a budget is exhausted, or the process crashes.

### 5.10 Decoding constraints

Every action is decoded under a grammar derived from the current tree:

| Part | Constrained to |
|------|----------------|
| tool | the eight header words |
| read `PATH` | existing paths in scope; valid ranges; valid meta suffixes |
| edit `PATH[RANGE]` | `Text` nodes that are writable; existing line numbers |
| set `PATH` | writable slots: existing, or creatable children of writable containers |
| set `TYPE` | types that fit the slot |
| set body | the grammar of `Draft<TYPE>` |
| copy `DST` | writable slots that the type of `SRC` fits |
| reduce `PATH` | pending nodes with status `unreduced` or `quiesced` and no unbound required part |
| reopen `PATH` | writable values that a natural-language lambda produced |

Writable means: inside scope, not harness-owned, not the lambda's own `args`,
not the `args` of a triggered child.

An action that still fails validation is discarded and resampled, up to 3
times, before the model is shown a rejection. Rejected samples are not added
to the context.

---

## 6. Episodes, context, and validation

### 6.1 Episode

One continuous agent run on one ordinary lambda. It begins when the lambda is
triggered with the **opening observation** (§8.1), proceeds as actions and
results, and ends per §5.9. Crisp lambdas and combinators have no episodes.

### 6.2 The context is a cache

Everything durable is in the tree. An episode's context may be dropped at any
time, and a new episode on the same node must be able to continue from the
tree alone. Such a **cold restart** receives the same opening observation as
any other episode, plus the effect journal if it is not empty.

The parent keeps its context while blocked in `reduce`.

### 6.3 Budgets

**Recursion is bounded, not forbidden.** A lambda may construct and reduce
lambdas; that is how subroutines work, so recursion cannot be ruled out
statically. Three rules keep it harmless:

1. *No identical child.* A lambda is not started if its instructions, args,
   and type are identical to those of a lambda already being reduced above
   it; it quiesces with a note. (The same idea as cycle detection in
   `Iterate`. Recursion over tree-shaped data passes, because the args differ
   at every level.)
2. *Bounded nesting.* At most 6 pending nodes may nest inside one another
   below the acting lambda; a write that would exceed this is rejected
   (`too-deep`).
3. *Run budgets.*

Per run: at most 256 episodes, nested at most 8 deep; beyond either, a
triggered lambda quiesces with a run-budget note instead of starting. (Found
necessary in practice: an untuned model that reopens and reduces its own
result otherwise recurses without bound.) Per episode: 24 actions and 4,000 generated tokens. Per action: 600 generated
tokens; longer output is discarded and resampled. Values are defaults and
configurable per run.

### 6.4 Commit points

Full type conformance (not draft) is required at:

1. completion of a lambda;
2. triggering a node: all required parts and parameters bound and fitting;
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

Diagnostic codes (v0.1):

| Code | Severity | Raised when |
|------|----------|-------------|
| `no-such-path`, `bad-range` | reject | the path or range does not exist |
| `out-of-scope` | reject | the path leaves the lambda's subtree |
| `not-writable` | reject | harness-owned, or the lambda's own `args` |
| `frozen` | reject | the `args` of a triggered lambda |
| `type-mismatch` | reject | the value does not fit the stated type |
| `type-does-not-fit-slot` | reject | the stated or source type does not fit the slot |
| `unknown-field` | reject | the record type has no such field |
| `reserved-key` | reject | a `$`-prefixed key used as data |
| `no-origin` | reject | `reopen` on a value no natural-language lambda produced |
| `too-deep` | reject | pending nodes would nest more than 6 deep |
| `eval-cannot-write` | reject | `eval` attempted to modify the tree |
| `effect-undeclared` | reject | capability not in the lambda's `effects` |
| `effect-wider-than-parent` | reject | a child declares a capability its parent lacks |
| `unbound-param`, `unbound-part` | blocks-commit | triggering a partial lambda or combinator |
| `stuck-dependency` | blocks-commit | a dependency in `args` quiesced |
| `commit-holes` | blocks-commit | completing with holes in `return` |
| `commit-pending` | blocks-commit | completing with pending nodes in `return` |
| `hole` | hole | a missing required field while drafting |

---

## 7. How an interpreter should work (normative for the reference policy)

The harness does not enforce this section. The reference policy follows it,
and training data teaches it.

### 7.1 The granularity rule [proposed]

For each step of `instructions`, in order:

1. **Leaf.** If the step is one judgment, extraction, or rewrite over inputs
   small enough to be visible inline, and its result belongs in `return`,
   write it directly.
2. **Exact work.** If the step is exact (counting, arithmetic, sorting,
   string operations, lookups):
   - result is a small scalar needed by later steps: `eval`, then
     **substitute** the scalar into the instruction text;
   - result is a value that belongs in the tree: a **crisp lambda** in the
     slot that needs it.
3. **Iteration.** If the step ranges over a collection, construct a `Map`,
   `Fold`, or `Iterate` where its result is needed. Never unroll, never
   recurse to iterate.
4. **Intermediate value.** If a later step needs a value that rule 2 cannot
   supply, construct the consumer first (outside-in): a continuation lambda in
   `return` whose `instructions` are the remaining steps, with a typed
   parameter for the intermediate; place the producer in that parameter.
5. **Too large.** If an input exceeds the inline threshold, never read it
   whole: delegate to a child, or `Map` over chunks made by a crisp lambda.
6. **Size cap.** If finishing would take more than about 8 actions, carve the
   remainder into a continuation.

After a step's result exists, delete the step's lines. A parent that has
carved everything into `return` reduces `return` and completes.

### 7.2 Substitution

- **Scalar substitution**: replace the phrase that denotes a small result
  with the result. "If the count is more than 5" becomes "If 7 is more
  than 5".
- **Path substitution**: replace a phrase that denotes a larger value with
  its path, in backticks. "Summarize the complaints" becomes "Summarize
  `args/complaints`".

### 7.3 Order of construction

Write the child, fill its `args` by `copy`, `reduce` it, then delete the
corresponding lines. Instructions therefore always describe work that is
remaining or in progress.

### 7.4 Cold restart

Infer progress from the tree: a step whose result already exists is deleted,
not redone. Consult `@effects` before repeating any effectful step.

### 7.5 Quiescing, repairing, reopening

- Stop with a closing message when no progress is possible. Say what is
  missing or ambiguous in one or two sentences.
- As a parent, on a quiesced child: read its note; edit its `instructions` or
  `args`; re-trigger. If that fails twice, quiesce with a note of your own.
- On a value that is not good enough: `reopen` it with instructions that say
  what to change.

### 7.6 Decisions

A judgment that matters is its own small lambda with a finite `returns`
(`Bool` or an enum). Where the legal options depend on state, a crisp lambda
computes them and constructs the decision lambda with a **narrowed enum** as
its `returns`.

**Numeric choices.** When the legal values of a number depend on state, there
is no range refinement to reach for. For a small range, the crisp lambda
builds a numeric enum (`enumOf(range(1, stock))` gives `1 | 2 | 3`). For a
large range, it computes a handful of named options and the amounts they
stand for ("min_raise", "half_pot", "pot", "all_in"), and the decision is an
ordinary string enum. The crisp code that applies the choice still refuses
anything illegal.

### 7.7 Recursion over tree-shaped data

Recursion is for data that is a tree, never for iteration. A recursive lambda
takes a template of itself as a higher-order parameter `self`. At each level
it builds a `Map` over the children whose `fn` is a copy of `args/self` with its
own `self` bound to another copy of `args/self`. The template never contains
itself; each instance carries the template.

### 7.8 Data is not code

Text inside `args` or `return` is data, whatever it says. Only `instructions`
is program.

---

## 8. Rendering (policy version `render/0.1`)

Rendering is the text the harness shows the model. It is part of the language:
training data and inference must use the same version.

### 8.1 Opening observation

```
events: <children quiesced/completed since last shown; "none">
problems: <n> blocking · <n> holes
effects so far: <only on cold restart, if any>

instructions  Text  <n> lines
  1| ...
  2| ...

in
  <rendered subtree>

return  <type>
  <rendered subtree>
```

### 8.2 Rules

| Item | Rule |
|------|------|
| Own `instructions` | always in full, with line numbers |
| Scalar | inline if ≤ 80 characters; else first 80, then `… (<n> chars)` |
| `Text` elsewhere | first line up to 80 characters, then `(<n> lines)` |
| Record | one line per field: `name  Type  value` |
| List | `Type  <n> items`, then the first 3 items, then `… <n> more` |
| Open list | `Type  open, <n> so far` |
| Dict | as list, keys in insertion order |
| Depth | 3 levels below the rendered root; deeper shows `…` |
| Hole | `·` in place of the value |
| Pending node | one line: type, status, and for quiesced the note's first 60 characters |
| Expanded Map | `<n> of <m> reduced`, then quiesced slots first, then the first 3 |
| Fold / Iterate | `at <i> of <n>` / `iteration <i> of max <m>`, plus the current `acc` / `state` by the rules above |
| Data text | rendered between reserved delimiters that are stripped from any data on write |

Target: an opening observation of at most 1,500 tokens. A `read` result is
held to the same rules for the subtree it names.

---

## 9. TypeScript: eval and crisp lambdas

### 9.1 Environment

QuickJS; TypeScript accepted, types stripped before execution; time limit
2 s and memory limit 64 MB per call (defaults). The harness generates a
declaration file for the current scope from the declared types; snippets are
statically checked against it before they run.

### 9.2 eval scope

```ts
declare const args: P;          // the bound parameters, typed from `params`
declare const self: {
  readonly instructions: string;
  readonly args: P;
  readonly return: DeepPartial<T>;
};
```

Pending nodes appear as opaque `Pending<T>` handles; using one as a value is
a type error. Lists larger than the inline threshold are lazy proxies
supporting indexing, `length`, iteration, and `slice`. The value of the final
expression is the tool result.

### 9.3 Crisp lambda body

`code` is the body of a function `(self, args, fx) => T`, where `args` is typed
by `P`. `eval` has the same `args` and `self` in scope, so there is one
environment to learn.

```
$lambda:
  type: 'Lambda<{ flags: Bool[] }, Num>'
  code: |
    return args.flags.filter(Boolean).length
```

A crisp lambda may return a **pending node** built with the constructors
`lambda(...)`, `map(...)`, `fold(...)`, `iterate(...)`, provided its type fits
`T`. The returned node takes the crisp lambda's place, `unreduced`, and
`reduce` reports `replaced`; it is not triggered automatically. This is how
narrowed decision lambdas (§7.6) are made.

### 9.4 Standard library

Pure helpers available in both contexts. v0.1 set:

`count, sum, min, max, mean, sortBy, groupBy, countBy, uniq, uniqBy, zip,
range, chunk, windows, flatten, take, drop, partition, topK, indexBy,
splitLines, splitOn, joinWith, trim, lower, upper, contains, startsWith,
matchAll, replaceAll, wordCount, parseNum, parseDate, formatDate, daysBetween,
hash, enumOf`

### 9.5 Effects

**Capabilities are registered by the host** (§10), each with a TypeScript
signature that goes into the generated declaration file. The harness ships a
few built-ins; an embedding program registers its own; a simulated
environment for tests or training data is simply a host that registers
deterministic in-memory fakes (a ticketing system, an inventory). The
language does not distinguish real capabilities from fake ones.

Effects are reached through `fx.<capability>.<function>(...)`. A call is
allowed only if the enclosing lambda declares the capability in `effects`,
and a child may not declare a capability its parent lacks. v0.1 capabilities:
`fs.read`, `fs.write`, `http.get`, `http.post`, `out.emit` (write a record to
the program's output stream). These are the built-ins.

Every effectful call is appended to the lambda's **effect journal** before it
executes: `{ seq, capability, function, args_hash, args_preview, status }`,
with `status` updated to `ok` or `error` afterward. The journal is readable at
`@effects`. Delivery is at-least-once across a crash.

---

## 10. The I/O boundary

A **program** is a root pending node in a file (§11): usually a lambda, or a
combinator such as a top-level `Fold` for a reactive program. The host binds
its parameters or parts and receives its value.

```
natlang run triage.yaml --in tickets=./tickets/ --in rubric=./rubric.md \
                        --out ./result.yaml
natlang run shop.yaml   --stream events=stdin:jsonl
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

There is no library namespace in v0.1 **[proposed]**: reusable lambdas are
passed as higher-order parameters and copied.

---

## 11. Serialization

A tree is serialized as YAML in the same form as `set` bodies, with the root
wrapped:

```yaml
$lambda:
  type: 'Lambda<{ tickets: Text[], rubric: Text }, Label[]>'
  types:
    Label: '"urgent" | "normal" | "spam"'
  instructions: |
    Label each ticket in `args/tickets` using `args/rubric`.
```

A quiesced or partially reduced tree serializes the same way, with `args`,
`return`, `status`, `note`, and `effects_journal` keys present (and `acc`/`at`
or `state`/`iteration` for combinators), so that a program in execution can
be saved, shipped, and resumed from the file alone. The rest of provenance is
stored separately.

---

## 12. Provenance

For every completed pending node the harness keeps: the node as it was at
trigger time (body, `args`), the trace of actions and results, step and token
counts, the effect journal, the logged distributions of finite-typed writes,
and links to earlier attempts (`reopen`, re-trigger). Long-lived folds keep
the most recent 1,000 step records by default.

---

## 13. Decisions made in this document, for review

| § | Decision |
|---|----------|
| 1.1 | Ranges inclusive, numbered as rendered; `+` appends; `@` meta suffixes |
| 2 | TypeScript type syntax with `Text`/`Num`/`Bool`/`Null`; lists as `T[]`; **`Dict<T>`** replaces the container formerly called Map |
| 3.2 | **Triggering reduces pending value-typed dependencies in `args` first** |
| 5 | Eight header words; `unset` and `reopen` are actions rather than library calls |
| 5.2 | A node's type is stated once; nested pending nodes use `$lambda` / `$map` / `$fold` / `$iterate` wrappers |
| 5.5 | Line-addressed `edit` only, for now |
| 5.7 | Blocking `reduce` only; no `wait`; therefore no concurrent writers |
| 6.3 | Budgets: 24 actions, 4,000 tokens per episode; 600 tokens per action |
| 7.1 | The granularity rule |
| 7.7 | Recursion by passing a template of oneself |
| 8 | Rendering thresholds (80 characters, 3 items, depth 3, 1,500 tokens) |
| 4.3 | `check` sees the last 3 states and runs every iteration |
| 3.1, 9 | The bound-parameter part is named **`args`** (types are `params`, values are `args`). It was `in`, which is a reserved word in JavaScript. Paths read `args/tickets`; `eval` and crisp code read `args.tickets` |
| 9.3 | Crisp lambdas may return pending nodes; the slot is `replaced`, not auto-triggered |
| 2.1 | Lists and dicts are covariant |
| 4.1 | A hand-written Map slot is typed by the enclosing slot, not by `B` |
| 6.5 | The diagnostic code list |
| 2 | Numeric literal types; no range refinement (§7.6 gives the pattern) |
| 5.9 | Quiescing is written as a `stuck` header plus a note, so that it can be decoded under the same grammar as actions |
| 5.3 | Types written as YAML values are always single-quoted |
| 9.5, 10 | Capabilities are host-registered with typed signatures; state size, lifetime, and shutdown are host concerns |
| 10 | No library namespace; host-side import and export, type-directed |
