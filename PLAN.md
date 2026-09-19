# natlang: a small-model interpreter for natural-language programs

Design and execution plan. Status: draft, 2026-09-18.
Training, use cases, and datasets are covered in depth in `TRAINING.md`.
Detailed designs for the synthesized datasets are in `SYNTHETIC_DATA.md`.
The type system, validation, and validation feedback are specified in `TYPES.md`.

## 1. Thesis

Small language models (LFM2.5 230M / 350M) are not capable enough to act as
general agents, but they are fast: a 200-token step costs well under 100 ms on
a consumer GPU, and thousands of independent steps can be batched into one
forward pass. The bet is that *algorithmic structure* can keep every individual
model decision inside the competence of a small model, while composition,
explicit state, and cheap verification supply everything else. The result is a
system that runs programs no conventional language can express (fuzzy leaves:
judge, classify, extract, rewrite, route) at a speed and cost no frontier model
can match.

The model **is the interpreter**. The harness provides memory, I/O, a sandbox,
and a scheduler. It imposes no syntax, no cursor, and no control flow. All
interpretive discipline (consuming instructions top-down, unrolling loops,
deciding when to reduce a child, recovering from errors) is instilled through
training data, not enforced by code.

Stated as arithmetic: with per-step success `p` and `n` steps, naive success is
`p^n`. A 97 % model fails half of all 25-step programs. Decomposition alone
does not rescue a small model; decomposition plus **error containment** does.
Every design decision below either shrinks a step or contains an error.

What this favors: shallow semantics × heavy structure × lots of data. Inbox
triage, normalizing messy collections into schemas, fuzzy dedup, extraction
over thousands of short texts, iterative refinement with judged loops. What it
does not favor: leaves that need knowledge or synthesis beyond the model
(structure cannot shrink a leaf below the model's knowledge floor).

## 2. The language

### 2.1 The object tree

The program, its state, its inputs, and its outputs are one typed object tree.
There is no filesystem in the semantics; the filesystem is one serialization.
Every node has a path (slash-separated, numeric indices for lists) and a type.

Type algebra (deliberately minimal; every type is something the model must
learn to create, read, and edit):

| Type      | Notes                                                              |
|-----------|--------------------------------------------------------------------|
| `Text`    | Markdown. Also holds code, instructions, prose.                    |
| `Num`, `Bool`, `Null` | Scalars.                                               |
| `List[T]` | Ordered. Optional element schema.                                  |
| `Dict<T>` | String-keyed. (Called "Map" in early drafts; renamed so `Map` is only the combinator.) |
| `Lambda<T>` | A typed promise with provenance. See 2.2.                        |
| `Blob`    | Bytes. Rare; only for opaque inputs.                                |

No `Ref` type. Sharing is copy-on-write in the store, so copying stays free
and semantics stay value-based.

Type/schema syntax is TypeScript type literals (compact, well known to the
base model). Runtime validation happens on every write to the tree, whether
from a tool call or from eval. Type errors are returned to the model as
ordinary tool errors and are a trained recovery case.

**Surface syntax (decided, to be confirmed by measurement).** Types are
written in TypeScript type syntax. **Values, including nested lambda literals,
are written in a strict YAML subset**, the same form the tree is rendered in,
so what the model reads and what it writes look alike. YAML being untyped is
not a problem, because parsing is **type-directed**: the declared type of the
slot decides how a scalar is read, which removes YAML's classic ambiguities
(`no`, `1.0`, `null`, dates). Block scalars make multi-line instruction text
painless, which JSON does not. The subset excludes anchors, tags, and
multiple documents; flow style (`{ id: 3, label: urgent }`) is allowed for
short values, and since JSON is valid flow YAML it remains a fallback.
Indentation can be grammar-constrained because the grammar is derived from the
type, whose nesting depth is known. Actions are written as a header line plus
a body (§3.1), so YAML never has to be embedded in a quoted string.
Alternatives to measure: TypeScript object literals, JSON.

### 2.2 Lambda: a typed promise with provenance

A `Lambda<P, T>` node has exactly three agent-visible parts, plus
harness-owned status. **There is no working-state zone** (decided 2026-09-18;
see 2.2.1).

- `instructions: Text` – the natural-language program. Non-empty means
  unreduced.
- `args` – the bound parameters, typed by `params: P`. Writable by the *parent*
  while the lambda is unreduced; **frozen when `reduce` is triggered**, and
  read-only to the lambda itself. This is what lets a parent accumulate and
  edit a child's inputs before shipping it.
- `return` – typed by `returns: T`, written during reduction under the draft
  rules (`TYPES.md` §3). Any slot of type `T'` inside it accepts a
  `Lambda<_, T'>`.
- status (harness-owned): `unreduced | running | quiesced | done`, step
  count, and for `quiesced` a reason kind and the agent's closing note.

The same shape is what the eval scope sees: `inputstructions`, `args`,
`self.return`, with the generated `.d.ts` typed from `params` and `returns`.

**Every node is created with an explicit type chosen by the agent.** There is
no type inference. `Lambda` is a constructible type like any other: a `set`
whose literal is a Lambda (instructions, `params`, `returns`, optional initial
`args`) is how subroutines come into existence.

Semantics:

- **Abstraction** is a Lambda node. **Application** is copying it and filling
  `args`. **Partial application** is filling some of `args`; a lambda with unbound
  required params cannot be reduced.
- **Reduction** is explicit and agent-triggered: the agent calls `reduce` on a
  path. `reduce` means *run the interpreter at this node until it is a
  value*. Reduction is in place; the agent copies first if it wants to keep
  the abstraction.
- **Completion**: `instructions` is empty, and `return` is a fully reduced
  value that type-checks against `returns`. Pending lambdas nested *inside* a
  record or list in `return` block completion until the agent reduces them.
- **No tail calls** (decided). Completion always requires a value. Iteration
  is provided by the `Map` and `Fold` combinator types (§2.4), not by
  recursion; a weak interpreter should never have to thread loop state by
  hand. Recursion remains possible for tree-shaped data, bounded by the depth
  of the data.
- **Swap-out**: on completion the node's path resolves to the plain value `T`.
  The lambda record (instructions, `args`, trace, steps, cost) moves to a shadow
  layer reachable through a meta path. The value tree stays pure data.
- **Typing discipline**: reading a `Lambda<_, T>` where `T` is expected is a
  type error until reduced.
- **Quiescence without completion.** A reduction may stop without reaching a
  value, and that is a normal outcome, not an exception. The agent ends its
  episode by stopping tool calls and writing a closing message ("I can't make
  progress: the rubric doesn't say how to treat forwarded messages"). The
  harness then checks the node: if `instructions` is empty and `return`
  validates, it is `done` and swaps out; otherwise it is `quiesced`, **left
  exactly as it was**, partial `return` included, with the closing message
  stored as its note. Budget exhaustion, a crash, and repeatedly refused
  commits produce the same state with a harness-written reason. There is no
  separate failure representation. The parent sees the note in its event
  line; the child's `args` and `instructions` are editable again; the parent may
  edit and re-trigger, replace it, or quiesce itself. A lambda that stops
  because it lacks information is a *residual program* awaiting more inputs,
  which is the same thing as a partial application.
- **Re-opening a value (proposed).** `reopen(path, instructions?)` in the eval
  stdlib turns a computed value back into a lambda: the provenance record is
  restored (or, for a value with no lambda origin, a `Lambda<{}, T>` is
  wrapped around it), **the previous value becomes the editable draft
  `return`**, and new instructions may be supplied ("shorten to 50 words";
  "the checker says: missing a date"). The type is unchanged, so the parent
  slot stays well-typed. Provenance links the attempts. This is the idiom for
  "the parent is not satisfied" and for judged-refinement loops: reduce,
  check, reopen with the checker's message, reduce. Post-processing that
  *changes* the type is not a reopen; it is a continuation built outside-in,
  and the type system forces that.
- **Scope** is the subtree. A lambda sees its own node and nothing above. The
  parent must copy anything the child needs into the child's `args`. This is
  what makes a lambda self-contained, shippable, and resumable.

#### 2.2.1 Where intermediate data lives

Hewing to the lambda calculus: there are no variables to assign, only
applications. `let x = e1 in e2` is `(λx. e2) e1`. So an intermediate result
is never stored in a scratch area; it is **a parameter of the lambda that
will consume it**, placed where that lambda's result needs to go.

- *Sequencing.* "Count the urgent ones; if more than 5, write an alert"
  becomes: `return` holds a continuation `Lambda<{ n: Num }, Text>` with the
  rest of the instructions, and its `args/n` slot holds a
  `Lambda<{ inbox }, Num>` that does the counting. Reduce the inner one, its
  value lands in `args/n`, then reduce the continuation.
- *Scalar and path substitution* (decided: a taught style). A small scalar
  result may be substituted into the instruction text, which is β-reduction
  proper; a larger result is referred to by substituting its **path** into the
  text ("summarize the complaints" becomes "summarize `args/complaints`"). A
  parameter binding is the same substitution delayed, and the only option for
  large or structured values.
- *Iteration* uses the `Map` and `Fold` combinators (§2.4). The accumulator of
  a fold is threaded by the harness, exactly typed, never by the model.

Why this rather than a work zone: it forces decomposition into small
subroutines, which is the regime where a small model is reliable; every
intermediate value is some lambda's return, so **every intermediate has a
type, a declared consumer, and provenance** (re-run, memoization, and
dataflow apply to all of it, not just to final results); the per-lambda
rendering stays at instructions + `args` + `return`; and nothing needs type
inference or tidying up. The cost is heavier structural steps: the
interpreter must carve continuations out of its own instructions and declare
param types. That is what the `copy` tool (3.1) is for.

**Fallback if the discipline proves too hard at 350M:** add a typed `work`
zone later. It is purely additive, whereas a model trained to lean on a
scratch pad is hard to wean off one; hence this order.

Provenance-derived idioms (design for, implement later): `rerun(path,
changed_inputs)`; staleness tracking when inputs change (the tree as a
dataflow graph); memoization by content hash of (instructions, inputs).

### 2.3 Type safety (summary; full design in `TYPES.md`)

- **Every lambda can be typed entirely**: `params` (the closure), `returns`,
  `effects` (side-effect capabilities; none = pure), and
  `ensures` (crisp postconditions such as `return.length == inbox.length`).
  Typing is gradual: `Any` is allowed, and an elaboration pass can propose
  types for hand-written untyped lambdas.
- **Partial application is typed.** A lambda with unbound required params is
  partial; reducing it is a type error.
- **Type preservation is enforced by the harness**: every accepted action
  takes a draft-well-typed tree to a draft-well-typed tree, and completion
  takes `Lambda<P, T>` to `T`. A slot of type `T` accepts a `T` or a
  `Lambda<_, T>`.
- **Holes are fine, lies are not.** While drafting, typed nodes are checked
  against `Draft<T>` (deeply partial `T`): incompleteness is accepted and
  recorded; contradictions are rejected at write. There is no untyped scratch
  area and no type inference: every node is created with an explicit type.
- **Commit points** demand full conformance: completion, `reduce` of a child,
  side-effecting calls, explicit `check` steps. A failed commit is refused.
- **The validator is a pure function of the tree**, run after every action
  and on demand for any snapshot. Its report is always logged and drives
  error containment, trace filtering, metrics, and RL reward, whether or not
  the model sees it.
- **Feedback is state, not interruption**: holes render inline as progress, a
  one-line problems count heads the observation, the full report is a meta
  path (`path@problems`), and blocking diagnostics are listed only after a
  refused commit. The balance between drafting and fixing is set by training
  data (`TYPES.md` §8), not by the harness.

### 2.4 Crisp lambdas, combinators, and control flow

The harness still reads no instructions and owns no program-level control
flow. It does provide a small set of **typed node kinds with fixed
semantics**, which the interpreter constructs, places, and triggers exactly
like lambdas.

**Crisp lambdas** (decided). A `Lambda<P, T>` whose body is TypeScript instead
of natural language: `code: Text` in place of `instructions`. It is typed,
editable, copyable, and sits in the result tree like any lambda. Triggering it
runs the code in the sandbox with `args` as its typed input, and the result
becomes the node's value. No model episode is involved. Crisp results get
provenance and memoization like everything else, and large values land in
their slot without passing through the model's tokens. A failing crisp lambda
quiesces like any other, with the error as its note, and its code can be
edited and re-run. Crisp lambdas may have side effects, subject to the
lambda's declared `effects`.

**`eval`** (decided). `eval` runs TypeScript in the sandbox and **returns its
result to the agent as a tool result**, truncated like any rendering. It can
read the tree and **can cause external side effects** (files, APIs, the mock
tools of simulated environments), subject to the enclosing lambda's declared
`effects`. It does not write to the tree: the agent writes whatever it learned
into the result structure itself. Rule of thumb taught by data: *eval to look
and to act on the world; crisp lambda to produce a value that belongs in the
tree*.

*Effects and restart.* The parent keeps its context while waiting, so context
is lost after an effect only on a crash or a budget stop, which is accepted as
at-least-once. As a cheap aid the harness journals every effectful eval call
in provenance and lists "effects already performed in this lambda" in a
cold-restart observation.

**Combinators** (decided; names provisional). Node types, not tools. They have
no instructions and get no model episode of their own; only their body
lambdas do.

| Type | Fields | Reduces to |
|------|--------|-----------|
| `Map<A, B>` | `over: List[A]`, `fn: Lambda<{ item: A }, B>` | `List[B]` |
| `Fold<A, S>` | `over: List[A]`, `init: S`, `step: Lambda<{ acc: S, item: A }, S>` | `S` |

- *Triggering a `Map`* expands it in place into one copy of `fn` per item,
  each with `args/item` bound (copy-on-write, so free). The expansion is visible
  in the tree as a list of pending lambdas. The harness reduces them in
  parallel, batched. When all are values the node is a `List[B]`. Length and
  order are preserved **by construction**.
- *Triggering a `Fold`* runs `step` once per item in order; the harness
  threads `acc`. Each step is a fresh small lambda with a fresh context.
- *Partial results*: if some body lambdas quiesce unreduced, the combinator
  quiesces with values in the finished slots and the stuck lambdas left in
  place. The parent may edit those and re-trigger; only unreduced slots run.
- `fn` and `step` may be crisp lambdas, natural-language lambdas, or contain
  further combinators. `fn` may carry extra bound params (a rubric, a schema).
- `Fold` is the name of the reduce operator (decided); the `reduce` tool that
  triggers reduction keeps its name.
- **Open lists and long-lived programs** (decided). A list may be *open*: a
  harness-level attribute meaning an external source keeps appending to it
  (rendered as `List[Event]  open, 1,204 so far`). It is still `List[A]` on the
  model-facing type surface, so **a reactive system is an ordinary top-level
  `Fold` over an open list of events**: `step: Lambda<{ acc: S, item: E }, S>`,
  state threaded by the harness, outputs emitted as `eval` side effects inside
  the step. A `Map` over an open list is a service that handles each arriving
  item independently (ticket routing as a daemon). Harness-only concerns: the
  combinator does not complete while the list is open, its current `acc` is
  readable in the tree, `S` needs a size bound, and per-step provenance needs a
  retention policy.
- **Narrowed types for legal actions** (decided). Hard constraints on choices
  are expressed by construction, not by after-the-fact validation: a crisp
  lambda computes the legal options from the state and constructs the decision
  lambda with a narrowed enum as its `returns`. A narrower enum fits a slot
  declared with the wider one (the only subtyping rule needed). The crisp
  transaction that applies the choice still refuses illegal input.

**`Iterate`: unbounded iteration, with ceremony** (decided). Judged loops and
fixpoints ("refine until the checker passes", "apply rules until nothing new
follows") need a loop whose length is not known in advance, which makes
runaway loops possible. The combinator therefore carries its own supervision.

| Type | Fields | Reduces to |
|------|--------|-----------|
| `Iterate<S>` | `init: S`, `step: Lambda<{ state: S }, S>`, `check: Lambda<{ recent: List[S], iteration: Num }, LoopVerdict>`, `max: Num` (mandatory) | `S` |

```
type LoopVerdict = { reason: Text, verdict: "continue" | "done" | "degenerate" }
```

- After every iteration the harness runs `check` as a **fresh episode**,
  independent of the step's context. It sees the last few states and the
  iteration count, must **write its reasoning first and then a verdict** from
  a finite set. This is the targeted place where the model states whether the
  loop is proceeding in an orderly way. It replaces a separate `until`.
- `done` ends the loop with the current state as the value. `degenerate`
  quiesces the `Iterate` with the reason as its note, state preserved, for the
  parent to inspect, edit, and re-trigger or abandon.
- Two crisp guards cost nothing and do not depend on the model: the mandatory
  `max`, and **cycle detection by content hash** (a state identical to an
  earlier one is degenerate by definition).
- `check` may be a crisp lambda when the stopping rule is exact.
- The verdict is a finite-typed write, so its distribution is logged like any
  other decision.

Patterns the interpreter is trained on:

- **Sequential steps**: consume `instructions` top-down; delete a step once
  its result exists. Use scalar and path substitution.
- **Subroutine call**: construct a child lambda outside-in, fill its `args` by
  `copy`, trigger it.
- **Iteration**: construct a `Map`, `Fold`, or `Iterate` where its result is
  needed, trigger it. Never unroll loops by hand, never recurse to iterate.
- **Exact work**: `eval` to look and to act on the world, crisp lambda to
  produce values for the tree. Arithmetic, sorting, counting, string
  operations, and schema transforms never go through the model's own
  generation.
- **Async**: non-blocking trigger + `wait` (may be dropped from v1).

## 3. The agent interface

### 3.1 Tools (six)

| Tool     | Signature (sketch)                                | Notes |
|----------|---------------------------------------------------|-------|
| `read`   | `read(path, range?)`                              | Renders a subtree (see 3.2). Range = lines for Text, index range for List. |
| `edit`   | `edit(path, old, new)`                            | Search/replace on a `Text` node. The trained file-editing skill. |
| `set`    | `set(path, type, literal)`                        | Create / replace / append / delete (literal `null` deletes). **The type is explicit at creation**; `Lambda` is a constructible type. |
| `copy`   | `copy(src, dst)`                                  | **Type-aware path-to-path copy.** See below. |
| `reduce` | `reduce(paths, blocking=true)` + `wait(paths, any\|all)` | Blocking by default. Non-blocking = spawn. Freezes the target's `args`. |
| `eval`   | `eval` + TypeScript body                          | Runs in the sandbox; the result comes back as the tool result. May cause external side effects (within declared `effects`); does not write to the tree (§2.4). |

**`copy` is a first-class tool and likely the most used one.** With no
working-state zone and subtree scoping, nearly all data movement is copying:
passing inputs into a child's `args`, carving a continuation out of one's own
instructions, moving a reduced value into the slot that needs it. Doing this
through `set` would force values back through the model's tokens, which is
slow, error-prone, and impossible for large values.

- *Sources and destinations* are paths, including **sub-ranges**: a line range
  of a `Text` node (`instructions[3..9]`), a slice of a `List`, a field of a
  record.
- *Type-aware*: the source's type must fit the destination slot (under the
  draft rules while drafting). Under type-directed decoding, once the source
  path is chosen **the destination grammar is restricted to type-compatible
  slots**, so an ill-typed copy cannot be emitted.
- *Text into a new Lambda*: copying a line range into a child's `instructions`
  is how continuations and subroutines are carved out. **There is no move
  option** (decided). The order is: write the child, reduce it, *then* delete
  the corresponding lines from one's own instructions. The parent's
  instructions therefore always describe work that is remaining or in
  progress, and deleting a step is the acknowledgment that it is done.
- *Construction is outside-in.* Build the continuation first, with a typed
  hole in its `args`; put the producer lambda in that hole. Lambda literals may
  nest, so this can be one `set`. The type system enforces the order: a
  `Lambda<_, List[Label]>` cannot sit in a `Text` slot.
- **Data is never destroyed**, only superseded by a later write or archived to
  provenance. What reduction consumes is the instruction text.
- *Cost*: O(1), copy-on-write in the content-addressed store. Provenance
  records the origin of copied values.
- Also available in eval as `copy(src, dst)` for bulk use (`spawnEach` is
  built on it).

**Action format: a header line plus a body** (decided as the default; to be
confirmed by measurement). One action per turn, wrapped in the model's
tool-call special tokens so the learned "now acting" mode is reused, but with
our own content inside instead of the native Pythonic call:

```
<|tool_call_start|>
set return/3 : { id: Num, label: Label }
id: 3
label: urgent
<|tool_call_end|>
```

```
<|tool_call_start|>
set return/summary : Lambda<{ items: List[Text] }, Text>
instructions: |
  Read the items and write a two-sentence summary.
  Mention the most common complaint.
<|tool_call_end|>
```

The header carries the tool, the path(s), and for `set` the TypeScript type.
The body runs to the end token, so **nothing is ever quoted or escaped**: for
`set` it is YAML (or raw text when the slot's type is `Text`), for `edit` the
raw replacement text, for `eval` raw TypeScript. `read`, `copy`, and `reduce`
are header-only. This is available because we control both fine-tuning and
parsing: the harness reads raw model output under a grammar and does not rely
on an inference engine's tool-call parser. Decoding is multi-phase: header
grammar first, then the body grammar derived from the type in the header. The
risk is departing from the model's Pythonic prior; the published fine-tuning
result suggests it adapts quickly. Measure against the native wrapper in
Phase 2.

Every model output is grammar-constrained (GBNF via llama.cpp, or an
equivalent). Malformed actions cannot occur.

### 3.2 Context model, observation, and rendering

**The lambda is the unit of context** (revised 2026-09-18; earlier drafts said
"stateless per step", which was too strong). Reducing a lambda is an ordinary
multi-turn agent episode: the agent makes several tool calls, sees their
results, and remembers what it just did. A `read` is useful because its
result stays in context. Each child lambda gets a fresh context of its own.

What is required is not statelessness but **restartability: the context is a
cache, never the state.** Everything durable lives in the tree, so a context
can be dropped at any moment (crash, budget, context limit, a long wait on
children, a re-trigger after quiescence) and a fresh episode can pick the
lambda up cold from the tree alone. Consequences:

- Cold-starting on a partially reduced lambda is a core trained skill, not an
  edge case: it is also what happens whenever a parent edits and re-triggers a
  quiesced child. The agent infers progress from the remaining instructions
  and what already exists in `return` (a step whose result is present gets
  deleted, not redone).
- Episodes stay short because lambdas stay small (no working state; work is
  decomposed). That limits the known degradation from long contexts and from
  a model seeing its own earlier errors. Rejected actions that were silently
  resampled are **not** added to the context.
- Whether a parent keeps or drops its context while blocked on children is a
  runtime choice to measure; restartability makes both valid.

**Episode.** One continuous agent run on one lambda. It starts when that
lambda is triggered or re-triggered, with the opening observation below. The
agent then alternates tool calls and tool results. It ends in one of three
ways: (a) the agent empties `instructions` and the commit validates, at which
point the harness ends the episode at once and swaps the value in; (b) the
agent stops calling tools and writes a closing message, leaving the lambda
`quiesced` with that note; (c) the harness stops it on a turn or token
budget, or it crashes. Waiting on triggered children happens inside the
episode, which is suspended, not ended; the parent keeps its context by
default. The context is discarded when the episode ends; the tree persists.
Crisp lambdas and combinators have no episodes.

An episode's opening observation, and any cold restart, contains:

1. An **event line**: children that quiesced or failed since the last step,
   and a **problems count** (`0 blocking · 3 holes`).
2. The **instructions** node of the current lambda (it *is* the program).
3. A **rendering of the lambda subtree**: YAML-like, type-annotated, with
   line numbers on Text, inline values under a size threshold, truncation with
   a "read to expand" hint above it, Lambda status inline.
4. After that, the episode proceeds as tool calls and tool results, each
   result including the event line when it has changed.

Rendering policy (inline thresholds, depth limits, list previews) shapes
model behavior as much as the tools do. It is part of the language
definition: fix it early, version it, do not tune it per experiment.

Example rendering:

```
inbox/        List[Text]  40 items
rubric        Text        12 lines
labels/       Lambda<List[{id:Num,label:Text,confidence:Num}]>  running  step 9
count         Num         40
summary/      Map
  tone        Text        "mostly complaints about ..."  (1.8k chars)
  themes/     List[Text]  6 items
```

### 3.3 Decisions and write-time typing

There is no predicate construct. A yes/no or categorical decision is simply a
write to a node whose declared type is finite (`Bool`, an enum). Decisions
that matter are reified as small child lambdas with a finite `returns`, a
convention taught by data; trivial branches are decided inline as part of the
interpreter's next action.

**Decided:** the primary reliability mechanism is the type system applied in
the harness at write time, built out fully (`TYPES.md` §6): every part of an
action is decoded under a grammar derived from the current tree and its
types, so ill-formed and ill-typed actions cannot be emitted; what still
fails validation is discarded and resampled before the model sees it. The
harness constrains what is well-typed, never which well-typed action to take.

The distribution over members for finite-typed writes is logged to provenance
at no cost. **Deferred**, pending evidence from those logs: voting,
confidence thresholds, calibration, escalation to a larger model, surfaced
confidence, speculative execution. `TYPES.md` §6.4 lists a simpler fallback
for each deep component.

## 4. The runtime

### 4.1 Components

- **Tree store**: SQLite. Rows = path → node (type, content hash, schema,
  status, provenance pointer). Content-addressed blobs alongside, giving
  copy-on-write, dedup across children, and memoization by hash.
- **Shadow layer**: provenance records keyed by path; per-lambda action log
  (observation, action, result) as JSONL – the raw material for training.
- **Scheduler**: a queue of pending reductions. Batches all live lambdas'
  next steps across all runs into shared forward passes. Blocking `reduce`
  simply suspends the parent until its children quiesce. Enforces step
  budgets, records status, emits quiescence events.
- **Model server**: llama.cpp (or vLLM) with grammar-constrained decoding and
  logprob access. Batched.
- **Validator and type layer** (`TYPES.md`): type parser, `Draft<T>`
  derivation, refinements, `ensures`, effect checks, preservation check;
  `.d.ts` generation per lambda scope and a warm TypeScript checker for eval
  snippets; **two-phase type-directed decoding** for `set` literals (path
  first, then a grammar derived from the declared type at that path); a
  silent-resample loop that discards rejected actions before the model sees
  them.
- **Eval sandbox**: QuickJS embedded in the harness process. TypeScript
  accepted, types stripped with esbuild, run as JS. Memory and time limits per
  call. The tree is exposed through a small typed API plus a crisp standard
  library (count, sortBy, filter, groupBy, regexExtract, join, spawnEach,
  reduceAll, collect, ...), so that most evals are one-line library calls. A
  `Lambda` node surfaces as a Promise. Writes are validated against node
  types. Large values come back as lazy handles with slicing.
- **Serializer**: export/import the tree to a directory (Text → `.md`,
  containers → `.yaml` or dirs, Lambda → dir with `instructions.md`). For
  human inspection and shipping only.
- **Snapshots**: at lambda boundaries, cheap (content-addressed). Fork,
  rewind, and mid-state RL rollouts use these.

**Note from the model card (see `TRAINING.md` §0.1):** LFM2.5 natively emits
Pythonic tool calls and is not recommended for code generation. TypeScript
stays the default eval language. The response is to keep evals small: a rich
crisp standard library makes the common case a single call, and free-form
multi-line code is a natural point for escalation. A Python-shaped surface
(Starlark) with the identical tree API is the fallback to measure in Phase 2.

### 4.2 What the harness does *not* do

Parse instructions. Impose a cursor. Own control flow. Decide when a child
runs. Collapse or clean up on the agent's behalf beyond the defined
swap-out. Everything interpretive is the model's job.

### 4.3 Speed levers

Batching across parallel branches and concurrent runs (the advantage that
does not shrink end to end). (Speculative execution of both conditional branches is deferred with the
other confidence-based features.) Memoization by content hash.
Per-step prompts kept to a few hundred tokens.

## 5. Training data generation

Because the tree is the state and episodes are short (one per lambda, §3.2),
**one agent turn = one training example**: `(opening observation + episode
prefix → next action)`. Prefixes are a handful of turns, so no long-context
trajectories are needed. Every example also has a cold-restart twin with the
prefix dropped, since the reference policy can label any tree state.

### 5.1 Oracles

- **Reference policy** (Python): a scripted interpreter over an abstract
  program representation. Given an AST and a rendered tree, it emits the
  canonical action sequence a good interpreter would take: instruction edits
  that delete completed steps, `set`s that create child lambdas, `reduce`
  calls, eval snippets for crisp work, and the final `return`. This is the
  oracle for **everything algorithmic** and where cursor discipline, loop
  unrolling, and crisp-routing conventions are instilled.
- **Crisp leaf library**: each leaf has an NL template and a hidden exact
  implementation (e.g. "count the words in x" ↔ `x.split().length`). Executed
  by the reference interpreter for exact labels.
- **Big-model oracle** for fuzzy leaves (summarize, judge tone, classify by
  rubric): one leaf at a time, inside otherwise crisp programs.
- **Existing instruction datasets** (FLAN-style, classification, extraction)
  reformatted directly as fuzzy-leaf examples: instruction + input → output.

### 5.2 Program generation

Generate programs as ASTs, not text. Random control flow (sequence, if,
while, for-each/map, fold, call, recursion) over typed leaves, with typed
inputs sampled to match. Render each AST with **varied surface forms**:
paraphrased keywords, paraphrased leaves, different instruction styles
(numbered steps, prose, bullets). The surface varies, the AST and its
semantics do not, so labels stay exact. Depth/length curriculum from
single-leaf to deep recursion and large collections.

### 5.3 Recovery data

Perturb states with realistic errors and label the fix: stale path,
ill-typed set, malformed edit that does not match, failed child, missing
return, wrong type in return, oversized inline value. The model must learn to
repair from the harness's error messages.

### 5.4 Distillation on messy input

Run a strong model as interpreter on hand-written, realistic pseudocode
through the identical tools and rendering. Keep runs whose final `return`
passes assertions or a judge. This is where semantic tolerance for real
instruction styles comes from. The protocol must be byte-identical to what
the small model will see.

### 5.5 Verification signals (for filtering and later RL)

Crisp-backed programs: exact match against the reference interpreter at every
step (a free process reward) and on the final tree. Fuzzy programs:
assertions the program author writes into instructions, type-checked
returns, and an LLM judge on the final value.

### 5.6 Dataset registry

Families refer to the use cases in `TRAINING.md` §1 (A triage, B extraction,
C entity resolution and cleaning, D long-input aggregation, E rules and
policies, F human-written procedures, G decomposed QA, H summarization,
I judged refinement, J segment-wise transformation). Names are recorded from
memory: **verify license, availability, and version before use**, and track
each in a data manifest with its license.

**Existing datasets: natural programs with verifiable outcomes**

| Dataset | Family | Role | Verification |
|---------|--------|------|--------------|
| SPoC | F | human pseudocode programs | test cases |
| Django, CoNaLa | F | NL ↔ code lines, crisp-routing phrasing | code execution |
| NAPS, NL2Bash, tldr-pages | F | algorithm and shell descriptions | execution / exact match |
| BREAK (QDMR) | G | NL step programs with back-references | source QA gold answers |
| MuSiQue, StrategyQA, HotpotQA, 2WikiMultiHop | G | multi-hop with decompositions | gold answers |
| DROP | D | extract + count/sort/add | exact answers |
| FinQA, TAT-QA, TabFact, WikiTableQuestions | D | table + text, gold arithmetic programs | exact answers |
| GSM8K | – | arithmetic routing (small share) | exact answers |
| ProofWriter / RuleTaker, CLUTRR | E | forward chaining over NL rules | gold proofs by depth |
| bAbI, ProPara | F | state tracking | gold states |
| BIG-bench / BBH algorithmic | F | small exact procedures | exact answers |
| OOLONG, S-NIAH, RULER, BABILong | D | long-input aggregation; **held-out eval first** | exact answers |

**Existing datasets: leaves and collections with gold labels**

| Dataset | Family | Role |
|---------|--------|------|
| Super-NaturalInstructions | all | task definition = lambda instructions; instances = collection |
| FLAN collection, P3, Tülu 3 SFT mix | all | broad leaf coverage, replay |
| Tülu 3 RLVR / IFEval-style constraints | I | verifiable constraints with checkers |
| Banking77, CLINC150, HWU64, MASSIVE | A | intent routing; out-of-scope for abstention |
| Bitext support, Twitter support, Enron, GitHub issues, StackExchange | A | triage and routing |
| AG News, DBpedia-14, 20 Newsgroups, Yahoo Answers | A | topic |
| GoEmotions, TweetEval, SST-2, Yelp/Amazon reviews, SemEval ABSA | A | sentiment and aspects |
| Civil Comments / Jigsaw | A, E | policy-as-rubric moderation |
| CLEF eHealth TAR, SYNERGY | A | abstract screening against criteria |
| PubMed-RCT, LEDGAR, LexGLUE | A | domain classification |
| CoNLL-2003, OntoNotes, Few-NERD, WNUT-17, Pile-NER, DocRED, REBEL | B | entity and relation extraction |
| Schema-Guided Dialogue, MultiWOZ, ATIS, SNIPS | B | slot filling to typed records |
| SROIE, CORD, FUNSD, Kleister | B | document text → record |
| CaseReportBench, JSONSchemaBench | B | eval only |
| SQuAD 2.0, NewsQA, Qasper | B | span extraction with abstention |
| Magellan/DeepMatcher suite, WDC Products | C | entity matching |
| Hospital, Adult, Restaurant, Buy (data-wrangling suite) | C | error detection, imputation |
| SOTAB, Sherlock/VizNet, TURL | C | column-type annotation |
| Quora Question Pairs, PAWS, MRPC | C | text dedup |
| CUAD, ContractNLI, LegalBench subsets | E | clause × rule grids |
| XSum, CNN/DM, SAMSum, QMSum, MultiNews, GovReport, BookSum | H | summarization, hierarchical |
| ASSET, JFLEG, W&I, GYAFC, FLORES, WMT | J | segment-wise transformation |
| Prometheus Feedback Collection, HelpSteer2, UltraFeedback, SummEval | I | rubric judging |
| xLAM-60k / APIGen, ToolACE, Hermes FC, Glaive | – | tool-format retention |
| BFCL, τ²-bench, IFEval | – | external eval only |
| Spider, WikiSQL, BIRD | D | source programs for back-translation (Y3) |

**Datasets we synthesize** (summary in `TRAINING.md` §3.5; full designs, prior art, and build order in `SYNTHETIC_DATA.md`)

| ID | Dataset | Gold comes from | Serves |
|----|---------|-----------------|--------|
| Y1 | Latent-world corpora: hidden database rendered as messy text | the hidden database | A, B, C, D, G |
| Y2 | Program-first synthetic programs | reference policy | all algorithmic skills |
| Y3 | Back-translated SQL / Python → NL procedures | executing the original | D, F |
| Y4 | Label-first rubrics and policies | construction | A, E |
| Y5 | Schema-first extraction, with corruption for repair | sampled value | B, repair |
| Y6 | Rule worlds in domain dress | exact solver | E, loops |
| Y7 | Simulated environments with mock side-effect tools | simulator state | F, K |
| Y8 | Interpreter drills, one per algorithmic skill | reference policy | S1–S10 |
| Y9 | Oversize and decomposition sets | per-item gold | S11 |
| Y10 | Dataset algebra: chained real datasets | composed gold | pipelines |
| Y11 | Minimal pairs and calibration sets | construction, vote splits | S2, S12 |
| Y12 | Data-is-not-code robustness (embedded instructions) | unaffected behavior | safety, correctness |
| Y13 | Verifier-backed generation constraints | generated checkers | I |
| Y14 | Long-horizon stress sets | Y1/Y2 machinery | eval, some training |
| Y15 | Style corpus for the renderer (unlabeled) | – | surface diversity |

## 6. Training

1. **Base model**: LFM2.5-350M for development. Shrink to 230M only after the
   harness, rendering, and data pipeline are stable; the size difference will
   show mostly in fuzzy-leaf quality, not in algorithmic rewriting.
2. **Formats**: use the base model's native tool-calling format rather than
   inventing one. Grammar-constrain outputs from day one so SFT and inference
   see the same distribution.
3. **Stage A – per-op SFT**: train on the per-step pairs, mixing algorithmic
   (reference-policy) and fuzzy (oracle/dataset) examples. Target: algorithmic
   ops in the high nineties per step. This is the number the whole thesis
   rides on.
4. **Stage B – whole-program SFT**: distilled trajectories and long synthetic
   programs, still as per-step pairs but drawn from full runs so
   distributions match.
5. **Stage C – RL**: GRPO or similar on programs with verifiable outcomes.
   Reward: final-state correctness, step-level agreement with the reference
   interpreter where available, penalties for step count and oversized reads.
   Rollouts fork from snapshots.
6. **Escalation policy**: train the logprob-margin thresholds for voting and
   escalation on held-out data, separately from the model.

## 7. Evaluation

Measure before training anything: run the harness with (a) a strong model
and (b) the base LFM with no fine-tuning, and record:

- **Per-op accuracy by op type** (edit-to-advance, set-child, reduce
  timing, crisp routing, predicate, fuzzy leaf, return typing).
- **Whole-program success vs. program length** and vs. nesting depth.
- Effect of voting and escalation on both curves.

Held-out families that stress what small models lose: long loops, deep
recursion, maps over large collections, values too big to inline, failed
children requiring repair, instructions in unfamiliar styles.

Speed: steps/second batched, end-to-end latency vs. a frontier model on the
same programs, cost per 1k items mapped.

## 8. Phases

**Phase 0 – Language spec (1–2 weeks).** Write the spec: type algebra,
Lambda semantics, combinators, tool signatures and action format, rendering
policy (versioned), eval API, serialization format. Hand-write the ~20-program
conformance suite (§10.2), organized by language construct, with expected
outputs.

**Phase 1 – Harness with a strong model (2–4 weeks).** SQLite tree store,
shadow layer, scheduler with batching, QuickJS eval with the tree API and stdlib v0,
llama.cpp client with grammars and logprobs, directory export. Type layer stages A–C (`TYPES.md` §6.3). Run the 20
programs end-to-end with a frontier model as executor. No fine-tuning. Fix the
rendering policy. Deliverable: a working system and the exact protocol the
small model will be trained on.

**Phase 2 – Baselines (1 week).** Swap in base LFM2.5-350M. Record the
per-op and program-length curves from §7. Decide JS vs. Starlark from the
crisp-op numbers. Decide which ops need the most data.

**Phase 3 – Data pipeline (3–5 weeks).** AST generator, surface renderer,
reference policy, crisp leaf library, fuzzy-leaf oracle, instruction-dataset
adapter, perturbation/recovery generator, distillation runner, filters. Target
a few million per-step pairs with a curriculum by depth.

**Phase 4 – SFT (2–3 weeks).** Stages A and B. Re-run §7. Iterate on data
mix until algorithmic ops are in the high nineties.

**Phase 5 – RL and reliability (3–4 weeks).** Stage C. Memoization. Review
the logged finite-type write distributions and decide whether any deferred
confidence feature (voting, thresholds, escalation, speculative execution) is
worth building. Shrink to 230M if warranted.

**Phase 6 – Applications and speed (ongoing).** Build three real programs
(e.g. inbox triage over thousands of messages, collection normalization to a
schema, judged iterative refinement). Measure against a frontier model on
quality, latency, and cost. Provenance idioms: rerun, staleness, dataflow.

## 9. Risks and mitigations

| Risk | Mitigation |
|------|------------|
| The type-directed harness grows too complex or slow | Staged build (`TYPES.md` §6.3); a named simpler fallback per component (§6.4); language and data format unaffected by falling back. |
| Grammar constraints distort the model's outputs | Train under the same constraints; monitor the mask-bind rate (`TYPES.md` §6.5). |
| Algorithmic per-step accuracy plateaus below ~98 % | Thesis fails. Measure in Phase 2/4 before investing further; shrink action space; more recovery data; voting on structural decisions. |
| Fuzzy leaves exceed the model's knowledge floor | Escalation per leaf; choose task families with small-judgment leaves; decompose leaves in data. |
| Rendering policy drifts between data generation and inference | Version it; byte-identical protocol across strong-model runs, generator, and inference; test equality in CI. |
| Instruction-editing loops grow the text | Convention (taught by data) to collapse completed iterations into a one-line summary. |
| Copy costs for large inputs across many children | Content-addressed store, copy-on-write. |
| The base model is weak at code and natively Pythonic; TypeScript evals may be error-prone | Crisp stdlib so evals are one-line calls; train S6 routing heavily; escalate free-form code; Starlark fallback with identical tree API, measured in Phase 2. |
| Overfitting to synthetic surface forms | Distillation on messy hand-written programs; held-out instruction styles. |

## 10. Open questions register

Consolidated 2026-09-18. Type-system specifics are also listed in `TYPES.md`
§10; items to verify for datasets are in `SYNTHETIC_DATA.md`.

### 10.0 Decided so far

- **Research artifact.** Non-commercial corpora are usable as style seeds;
  keep share-alike-derived data in a separable shard; check `lfm1.0` terms.
- **English first.** Multilingual is a later wave.
- **`reopen`** is in (§2.2).
- **`eval` returns its result to the agent as a tool result, may cause
  external side effects, and does not write to the tree; crisp lambdas**
  produce values for the tree (§2.4). At-least-once effects on a crash are
  accepted; an effect journal aids cold restarts.
- **`Map`, `Fold`, and `Iterate` combinator types; no tail calls** (§2.4).
  `Iterate` carries a mandatory bound, crisp cycle detection, and a
  model-based `check` that writes its reasoning and a verdict every iteration.
- **Action format**: header line plus body, nothing quoted or escaped (§3.1).
- **`Fold` is the operator's name. Reactive systems are a top-level `Fold` over
  an open list of events**; legal actions are enforced by narrowed enum types
  built by crisp lambdas, not by postconditions (§2.4).
- **Teacher candidates: K2 Horizon 7B and Ternary Bonsai 2 27B**, evaluated on
  the conformance suite first; if both struggle, diagnose whether to tweak the
  approach or the models (`TRAINING.md` §3.4).
- **Postconditions and refinements are deferred**; v1 typing is structural
  only (`TYPES.md`).
- **Authoring-time help from a larger model is out of scope.** Larger models
  are used for generating training data and tasks only.
- **Scalar and path substitution** into instruction text is a taught style.
- **Types in TypeScript syntax, values in a strict YAML subset**, parsed
  type-directed (§2.1).
- No working-state zone; explicit types at creation; no `move`; the lambda is
  the unit of context; write-time typing as the primary reliability
  mechanism; TypeScript as the eval language.

### 10.1 Needs a decision from the project owner

Nothing blocking. One later choice:

- **Demonstration task** (not blocking): the one task with an external
  benchmark and a frontier-model comparison that tests the thesis. Default
  candidate: long-input aggregation (a `Map` of small judgments plus a crisp
  aggregate; OOLONG as the benchmark; direct comparison with RLM). It also
  decides which fuzzy-leaf data to prioritize.

### 10.2 Phase 0 status

**Done, under review.** `spec/SPEC.md` v0.1-draft specifies the tree, types,
lambdas, combinators, the eight actions and their decoding constraints,
episodes, validation and diagnostic codes, the interpreter conventions, the
rendering policy, the TypeScript environment, the I/O boundary, serialization,
and provenance. Its §13 lists every decision first made there; the ones most
worth reviewing are the **granularity rule** (§7.1), **dependencies reduce
first on trigger** (§3.2), **blocking `reduce` only** (§5.7), and the
**rendering thresholds** (§8). `conformance/` holds 22 programs and 6 harness
scripts, organized by construct; `tools/check_conformance.py` parses every
type expression in them against the grammar.

Renamed by the spec: the string-keyed container is **`Dict<T>`**, so that
`Map` means only the combinator.

**Not language questions** (settled 2026-09-19)

- *State size of a long-lived `Fold`, and shutdown*: concerns of the host that
  invokes the program (the CLI, or a client library embedding natlang in
  another program). SPEC §10.
- *Numeric range choices*: no range refinement. Numeric literal types cover
  small ranges; large ranges are discretized into named options by crisp
  code. SPEC §2, §7.6.
- *Mock capabilities*: capabilities are registered by the host with typed
  signatures, so a simulated environment is just a host registering fakes.
  Which simulators to build is synthetic-data work (`SYNTHETIC_DATA.md` Y7),
  not harness work. SPEC §9.5.

**Remaining engineering for Phase 1** (work to do, not decisions)

- The action grammar in the inference engine's format: a static part (tool
  words, path and type syntax) and a per-turn part generated from the tree
  (existing paths, types that fit the slot, the body grammar of the draft
  type). Large lists need a compact path pattern or restriction to rendered
  paths. Multi-phase decoding is done as separate generation calls joined by
  prefix caching.
- The capability registration interface.

**To measure in Phase 2**

- Search-and-replace and whole-text rewrite as alternatives to line-addressed
  `edit`.

### 10.2b Phase 1 status (harness)

Branch `phase-1-harness`. Package `natlang/`:

| Module | Contents |
|--------|----------|
| `types.py` | type grammar parser, environments with named and recursive types, the fit relation |
| `values.py` | type-directed construction of values and pending nodes, holes, unbound parts, serialization |
| `nodes.py`, `refs.py`, `paths.py` | node kinds, typed slot references, path resolution with scope and writability |
| `actions.py` | the header-plus-body action parser |
| `runtime.py` | sessions that apply the eight actions with validation and commit checks; triggering of lambdas, crisp lambdas, `Map`, `Fold`, `Iterate`; dependencies first; swap-out; quiescence; effect journal; open lists |
| `js.py`, `prelude.js` | QuickJS sandbox and the crisp standard library |
| `render.py` | rendering policy `render/0.1` |
| `agents.py` | stub, oracle, and replay agents for tests |

Tests (62): the 6 harness conformance scripts; a replay of the 8 canonical
traces end to end with small Python oracles standing in for the model; the
generated grammars accepting every canonical action at its state and refusing
ill-typed ones; the model agent's two-phase loop with a scripted decoder.

**Found by building it** (spec corrected): `in` is a reserved word in
JavaScript, so the bound-parameter part is now `args`; unbound parts of pending
nodes in a draft count as holes.

**Generation control** (decided 2026-09-19, from a survey of current backends;
see `natlang/decoder.py`). An OpenAI-style chat endpoint is not enough: the
harness needs a different grammar on every call, raw prompts with special
tokens, continuation of one sequence under a second grammar, top-token
probabilities, cheap resampling, and prefix-cache reuse.

- The harness owns a small `Decoder` interface; backends plug in behind it.
- **Grammar format: portable GBNF** with no token-id terminals and no `{m,n}`.
  It is native to llama.cpp, accepted by XGrammar (the default in vLLM and
  SGLang), and by llguidance. `natlang/grammar.py` derives a header grammar
  from the tree (existing paths, writable slots, types that fit each slot,
  reducible nodes) and a body grammar from the header's type (`Draft<T>`).
  `natlang/gbnf.py` is a recognizer used to test them without an engine.
- **Local development: llama.cpp `llama-server`, native `/completion`.** Per
  the server README it takes a per-request grammar, raw or token-array
  prompts, `n_probs`, `cache_prompt`, slots with continuous batching, `seed`,
  and multiple completions. Two-phase decoding is two requests on one slot.
- **Bulk generation and RL rollouts: SGLang first** (official LFM2.5 cookbook,
  hybrid radix cache), vLLM where TRL requires it, via TRL's experimental
  `rollout_func` for per-request grammars.
- **What would force in-process bindings** (llama-cpp-python low level, or
  transformers with llguidance): needing full unconstrained logits at every
  step; token healing or mid-token grammar switches; misbehaving hybrid-state
  caching that needs explicit state save/load; stateful masks GBNF cannot
  express; or RL needing exactly mask-consistent log-probabilities.
- **To verify on a running server**: that `n_probs` with
  `post_sampling_probs=false` reports probabilities *before* the grammar mask
  (needed for the mask-bind rate); that prefix reuse works for this hybrid
  architecture (two open llama.cpp issues suggest it may not for sibling
  requests; at 350M re-prefill is cheap either way); how the action's end is
  signalled under a grammar (`<|tool_call_end|>` versus the end-of-turn token);
  the exact LFM2.5 chat template for tool results.

**Verified on a running server** (2026-09-19; llama.cpp 0.4.1 from Homebrew,
CPU build; `LFM2.5-350M-Q8_0.gguf` from the official LiquidAI GGUF repo):

- The model's chat template is plain ChatML with roles passed through and
  `<|im_end|>` as end of turn; `ChatTemplate` matches it.
- **`n_probs` with `post_sampling_probs=false` reports probabilities before
  the grammar mask** (model wanted "Paris" at 0.987 while the grammar forced
  "Berlin"). The mask-bind rate is measurable over HTTP.
- A grammar-complete action ends by itself (stop type `eos`).
- Prefix cache: continuing on the same slot reuses nearly everything
  (1,214 of 1,219 tokens); a sibling that branches mid-prefix reuses part
  (702 of 1,218); another slot reuses nothing. Consequence: keep an episode on
  one slot; for `Map` siblings either pin them to one slot in sequence or
  accept re-prefill.
- CPU speed: ~360 tok/s prefill, ~116 tok/s decode. A GPU build is the obvious
  next improvement (the laptop has an 8 GB RTX 4060).

**First baseline with the untuned model** (4 conformance programs):

- An untuned 350M model **parrots prompt text**: if the prompt names the
  give-up action (`stuck`, earlier `close`), it emits it at ~0.95 on a task it
  can do; with that word absent it writes the correct `set return : Bool`.
  Hence two prompts: `interpreter.md` for a teacher, `interpreter_small.md`
  (short, worked examples, never names `stuck`) for an untuned small model.
- The first runs spent most turns on out-of-range `read instructions[a..b]`
  and empty container bodies, all rejected by validation. Those were **grammar
  gaps**, now closed: `read` is offered only where the rendering hides
  something and never for the lambda's own instructions; ranges are limited
  to positions that exist; container bodies must be non-empty. Result:
  **rejections went from dozens per program to zero**, and a run dropped from
  minutes to seconds.
- Program 01 (one judgment) now completes correctly, first action
  `set return : Bool` / `true`. Programs 02, 03, 06 do not: the untuned model
  does not know the protocol. That is the expected starting point for
  fine-tuning, and the per-skill baseline the plan asks for.

**Decisions of 2026-09-19 (second session)**

- **Template-agnostic by default.** The prompt is rendered by the loaded
  model's own chat template (`/apply-template`), an action is plain text in the
  assistant turn, and results return as user turns so roles strictly
  alternate. Teachers need no special casing. Per-model adaptations are
  opt-in `Wrapper`s for the models we fine-tune (`lfm`: the native tool-call
  token; `reasoning`: think freely up to `</think>`, then the constrained
  action). This also fixed a bug: hand-built prompts carried two
  beginning-of-text tokens.
- **GPU serving through Docker.** llama.cpp's prebuilt CUDA binaries need
  glibc 2.38 and this machine has 2.35; the system CUDA toolkit (11.5)
  predates the GPU. The official `server-cuda` image needs neither.
  `scripts/serve.sh`. Measured on the RTX 4060: ~26,000 tok/s prefill
  (CPU: 360), 224 tok/s single-stream decode, under 1 GB of VRAM.
- **Scope of this machine: prove out the whole system.** Teachers and real
  training run on a larger GPU elsewhere.
- **A memory-constrained training mode is a goal**, so that people can
  fine-tune on their own machines: LoRA/QLoRA, 8-bit optimizer states,
  gradient checkpointing, short sequences (episodes are short by design),
  and a documented configuration that fits in 8 GB.
- **Recursion is bounded, not forbidden** (SPEC §6.3): no identical child;
  at most 6 nested pending nodes; run budgets. `reopen` applies only to values
  a natural-language lambda produced. All three came out of a runaway observed
  with the untuned model (171 reopens and 153 reduces in one run), which also
  intermittently overflowed Python's recursion limit.

**Wrapper comparison, untuned model, six programs**: both wrappers solve none
beyond the single judgment; `lfm` solved it and `generic` did not in this
sample. Too little data to conclude anything; the comparison that matters is
after fine-tuning.

Also built: `model_agent.py` (two-phase constrained decoding, discard-and-
resample of rejected actions without showing them to the model, `close`),
`prompts/interpreter.md` (the interpreter's instructions, for a teacher or an
untuned model), `host.py` and `python -m natlang run` (type-directed import,
export, trace output).

**Tool surface refactor (2026-09-19, third session).** The model-facing surface
is now native tool calls, in one swappable module (`natlang/surface.py`,
surface `tools-v1`): a fixed list of eleven familiarly named tools (`write`,
`done`, `define`, `run`, `run_code`, `edit`, `copy`, `read`, `retry`, `delete`,
`give_up`) whose argument schemas are regenerated from the tree and types each
turn; `define` wires a sub-task's inputs in the same call (`args_from`,
`over_from`) and is all-or-nothing; `edit` is classic substitution (`old` must
occur exactly once); `done` finishes and is refused with a hint while anything
is missing; failures carry a one-line hint; results come back with a refreshed
state; state rendering follows the anti-parroting rules (no value-like
placeholders, filled values apart from what is still to fill). There is no
`answer` tool: a one-shot leaf is `[write(...), done()]` in one turn, which is
the model's native multi-call habit, so incremental work and sub-tasks stay on
the same path. `natlang/tool_agent.py` drives it; a prose turn is kept and
followed by a nudge. The internal operations and the text trace notation of
the conformance suite are unchanged; the tool names map onto them.

**What the untuned 350M taught us (measured, small samples)**

| Finding | Evidence |
|---------|----------|
| Its native strength is a complete typed answer in one shot | whole record as schema-constrained JSON: 8/8 exact, `"0077"` kept a string; Bool via a tool: 8/8 |
| Field-by-field filling is weak | 0–2 of 8 in every rendering tried |
| It parrots nearby text | wrote "(empty)", a recently shown number, "the customer name", the give-up word, and its own call as plain text |
| History must be in the model's **native** call format | with a neutral text rendering it replied `[read(...)]` as prose; LFM2.5's template also silently drops `tool_calls` from history, so the call has to be written into the assistant content |
| After a tool result it tends to report in prose | ~2 of 3 turns; the trained call → result → answer pattern |
| **Tool-set size dominates** | with `write` + `done` only: correct `write return true` every time; with eleven tools: picks `read` every time, regardless of order |
| **llama.cpp does not enforce argument schemas for LFM's native format** | it accepted `path` values outside the enum and wrongly typed fields; earlier conforming outputs were the model cooperating (it sees the schema in its prompt) |
| Free-form `edit(old, new)` on a task document is not a natural strength | no-op edits, or replaced the wrong text: 0/6 useful |

**The natural agent loop (surface `tools-v2`, same day).** The earlier surfaces
presented an artificial task: a state dump as the user message, inputs pasted
inline so that reading looked like a mistake, a result that had to go into a
slot by a special call, and completion by `done`. Much of what looked like
model weakness was that framing. `tools-v2` follows the loop these models are
trained on:

- The user message is the lambda's **instructions as a request**, followed by a
  workspace listing (small values inline, long ones to be read).
- Six tools, always the same: `read`, `write`, `edit`, `run_code`, `define`,
  `run`. Reading first is normal and `read` returns the whole value.
- **The reply ends the episode.** The result is whatever was written to
  `return`; **the reply is never parsed as a result** (decided), it is kept as
  the note. A reply with `return` missing gets one line saying what is missing,
  twice at most, then the lambda quiesces with the reply as its note. There is
  no `done` and no `give_up`: both are what a reply already is.
- History is sent as standard `tool_calls` messages.

**The GGUF's chat template was the broken part, not llama.cpp.** The template
embedded in LiquidAI's GGUF is a reduced one that ignores `tool_calls` and has
no tool-call tokens; the official `chat_template.jinja` in the model repo
renders `<|tool_call_start|>[read(path='args/note')]<|tool_call_end|>`.
`scripts/serve.sh` now serves with the official template
(`models/templates/`). With it, standard history works and no per-model
wrapper is needed.

**Still true with the right template: llama.cpp does not enforce argument
schemas for this model's format** (asked to violate an enum and an integer
type, it did so 10 of 10 times and the server accepted it).

**Untuned 350M under `tools-v2`, seven programs**: two correct (the judgment in
one action; the program that must stop, with a sensible note), extraction
right in every required field but with an invented empty optional field,
whole trajectories of the form read → write → reply in well under a second.
The Map and crisp-lambda programs are not solved: `define` is beyond the
untuned model.

**Native constrained decoding (built, `natlang/native.py`).** The prompt is
rendered by the model's own template (`/apply-template`, with tools); the
completion runs under a grammar built from the turn's tool schemas, over the
model's native call text:

    root ::= "<|tool_call_start|>[" call (", " call)* "]"  |  reply

The model still chooses between calling and replying. A call cannot name a
path that does not exist, put a wrongly typed value into a slot, invent or
omit a field, read a range that is not there, or leave an optional text field
empty. Tools may carry `x-natlang-alternatives`, argument sets that belong
together (a path **and the value type of that path**; a path and its valid
range), which JSON Schema cannot express at the top level. `None` in an
optional field means "does not apply" and is read as absent. Verified on the
server: the tool-call token can be named inside a grammar; a grammar-complete
call ends by itself; special tokens are invisible in `content` but present in
the returned token ids; the first-token distribution gives **P(the model
starts a call)** before the grammar, logged per turn.

*Bug found on the way:* the range arguments were called `from`/`to`; `from` is
a Python keyword, so the model's valid native call failed to parse, was filed
as prose, entered the history in a non-native form, and was then imitated.
Argument names are now checked against Python keywords, and an unparseable
call counts as a rejected call, never as a reply.

**Untuned 350M, `tools-v2` + native decoding, seven programs:** every call
valid (0 rejections), runs take 0.2–0.4 s. Judgment correct. Extraction right
in all required fields but it invents a value for the optional `phone` even
when `None` is offered (8 of 8): a competence gap, left to fine-tuning. The
arithmetic program guesses instead of using `run_code`; the Map program writes
a well-typed but short list by hand after a failed `define`. Mean P(call) at
the start of a turn is ~0.35: on question-shaped tasks the untuned model
prefers to answer in prose, is told what `return` still needs, and then
usually writes.

**Typed `write`, no `define`; data only through the tool channel (same day).**

- Five tools: `read`, `write`, `edit`, `run_code`, `run`. `define` is gone:
  **`write(path, type, value)`** is typed, and a sub-task type gives the
  semantics `define` had. Model-facing sub-task types: `Task<T>`
  (instructions), `Code<T>` (TypeScript), `Map<A, B>`, `Fold<A, S>`,
  `Iterate<S>`. Every one is derivable: `T`/`B`/`S` from the slot, `A` from
  the list chosen in `over`, a task's parameter types from the `inputs` wired
  into it (`Task<T>` stands for `Lambda<{derived}, T>`). Under constrained
  decoding each (path, type, value shape) is one grammar alternative with the
  type as a constant, so **the type tokens are forced and appear in the
  transcript as if the model had chosen them**; a wrong type, or `over`
  pointing at something that is not a list, cannot be written. This relaxes
  "the agent chooses the type at creation" exactly where the type is
  determined; narrowing a result type remains a later, explicit option.
- **Instructions and data travel in different channels.** The user message is
  the instructions and nothing else. The harness performs the first step on
  the agent's behalf, `read(path="args")`, and the workspace arrives as a tool
  result; nudges carry no data either. Before this, the opening user message
  quoted the beginning of each input, so an injected "ignore your
  instructions" sat a few lines under the real instructions: the earlier
  injection result tested that rendering, not the model.
- Worked examples are back in the small prompt, in the native call format.
  They raise the rate at which the model starts a call (0.40 → 0.52 in one
  comparison) and are copied literally, which is now harmless for types.

Untuned 350M, seven programs, after these changes: judgment and the
must-stop program correct; no rejected calls; new failure shape: the cheapest
well-typed value (`write return Label[] = []`) ends a list task at once. Not
patched: an empty list is sometimes the right answer, and this is what
fine-tuning is for.

**Reference policy and first synthetic generator (built, `natlang/gen/`,
`scripts/generate.py`).** A miniature latent world (tickets, reviews, notes
with hidden fields) feeds six program families: `judge`, `classify`,
`extract`, `crisp_scalar`, `map_leaf`, `map_then_count` (the outside-in
continuation: a `Code` consumer with a declared param, a `Map` written into
that param, one `run`). The reference policy follows a plan, never the
natural-language text; it performs every call through the real harness and
records each assistant turn as one sample `(messages so far, the turn's
tools) → target`, in exactly the ToolAgent's message structure. Three checks
run on everything it emits: every call is accepted by the harness, every
target is accepted by **the grammar of its own turn**, and the final value
equals the expected one (for exact work the oracle is the code itself).
120 programs → 344 episodes → 897 samples in a few seconds on the CPU.

*Found by the grammar self-check:* a turn's grammar is built from the state
before the turn, so a call that depends on what an earlier call created
(writing into the param slot of a task just written; running it) must go into
the next turn. Only independent calls may share a turn. The reference policy
and the prompt example follow that rule.

**Teacher: Ternary Bonsai 2 27B** (`scripts/serve_bonsai.sh`), set up from
the lab's own materials: Prism ML's llama.cpp fork (mainline cannot read the
`PTQ1_0` quantization), the official `chat_template.jinja` rather than the
GGUF's, q4_0 KV cache, a capped thinking budget, the lab's sampling
settings. The fork's prebuilt binaries run on this glibc but do not bundle
the CUDA runtime, so they run inside `nvidia/cuda:12.8.1-runtime-ubuntu22.04`.
Tool calls are Qwen-coder-style XML, parsed by the server.
`scripts/paraphrase.py` uses the teacher for surface diversity with a
round-trip check: a paraphrase of an instruction is kept only if the teacher
itself, executing the paraphrased program through the harness, reaches the
known answer on fresh instances.

**Bonsai as teacher: what happened (2026-09-19).** It runs: Prism ML's fork
inside `natlang-prism-runtime` (`docker/prism.Dockerfile`: CUDA 12.8 runtime
plus `libgomp1`, which the prebuilt binaries need and do not bundle), the
lab's official template, PTQ1_0 weights, q4_0 KV cache at 8K context.
**GPU: 6.0 of 8.2 GB. Decode 23.5 tok/s, 6–8 s per agent turn** with a
256-token thinking budget. Tool calling through the server works (Qwen-coder
XML, parsed server-side); standard `tool_calls` history works with the
official template.

Two teacher-side quirks, both now absorbed by the harness rather than fought:
- it wrapped a plain value as `{"value": true}`. Cause: my regression, `value`
  had an empty JSON schema after `define` was folded into `write`. `value` now
  lists every shape it may take; a `{"value": X}` wrapper is unwrapped when the
  slot is not a record with that field; the type-mismatch hint gives examples.
- its XML call format delivers every parameter as text, so a record arrived
  as a JSON string. A string is tried as text first; if the slot does not take
  it but it is JSON for a value the slot does take, the parsed value is used.

With those fixes Bonsai solved programs 01 and (as far as the run got) 02
cleanly: `read` → whole-record `write` → reply.

**Host memory: solved (2026-09-19).** The first attempt exhausted this
laptop's 14 GB of RAM. Two causes, both llama-server defaults: the mapped
weights stayed in host RAM, and the host-side prompt cache (default 8 GB)
grew without bound. `scripts/serve_bonsai.sh` now passes `--no-mmap` and
`--cache-ram 1024` and caps the container at 5 GB; Bonsai then costs ~0.8 GB
of host RAM and runs for hours beside a desktop session.
`scripts/watch_bonsai.sh` restarts it if the health check fails.

**Local data generation, first full pass (2026-09-19), all on this machine.**
- Reference policy, CPU only: 2000 programs -> 14,365 verified samples
  (`data/ref-v1.jsonl`), about a minute.
- Teacher paraphrase round trip: 16 base texts -> 61 paraphrases kept, 1
  dropped for a wrong answer (8 more were lost to the context bug below and
  re-asked). ~80 s per candidate; the whole pass took ~1.5 h. 32% of samples
  in `ref-v1` carry a verified paraphrase as the root instruction.
- Bug found by the run: sub-task alternatives were offered for every element
  of a filled list, so a four-item `return` grew the `write` schema from 6k
  to 21k characters and the request past the teacher's context. They are now
  offered for named slots only; context raised to 12k.

**Bonsai on the conformance suite** (server tool calling, thinking 512,
temperature 0.6): 15 of 20 correct, **0 rejected calls in 94 turns**, 25-210 s
per program. The five misses are not interpreter failures: 04, 10, 12, 17
have free-text answers that the baseline compares by equality, and 16 expects
a quiesce where the teacher chose to answer. Notable: the 27B almost never
decomposes (19 of 20 programs in a single episode; only the crisp fold spawned
children). It reads, answers whole, replies. So the teacher is a good
*oracle* and paraphraser, but a poor source of *decomposition* trajectories:
those must keep coming from the reference policy, with the teacher checking
answers. The baseline needs a judged comparison for free-text expectations.

**Consequences**
- Write-time typing cannot be delegated to the server for this model. The
  harness must constrain the **native call text itself** (raw completion under
  our own grammar, emitting `[write(path="return", value=True)]`); the JSON
  schemas of `surface.py` are the source, the GBNF machinery of `grammar.py`
  the target. For teachers whose formats llama.cpp does constrain, the server
  path stays.
- The consistent tool surface should be **small**. Candidates to fold away:
  `read` (into `run_code`, which can inspect anything), `copy` (into
  `args_from`/`over_from`), `delete` and `retry` (rare). To be decided by a
  sweep over a few fixed tool sets, not by filtering per turn.
- A lambda-as-document surface driven by `edit` would need the same
  constraints (`old` limited to placeholder lines) to work at this size; it
  is a candidate for the sweep, not a shortcut.

**Known limitations of the current code**
- Not yet run against a real model. Large lists get an index pattern in the
  path grammar that is validated afterwards, not bounded by the grammar.
- Children run sequentially; no batching.
- Copies are deep copies, not copy-on-write; the tree is in memory, not SQLite.
- TypeScript type annotations are not stripped and eval is not statically
  checked (type layer stage D).
- The QuickJS binding cannot call into Python while a time limit is set, so
  code in effect-declaring lambdas runs without the time limit. A subprocess
  worker should replace this.
- Provenance is minimal: origins for `reopen` and `@origin`, the effect
  journal, and a flat action trace.

### 10.3 To be measured, not argued

- Per-step accuracy of the base model on each skill, and whole-program success
  against length. Go/no-go for the thesis.
- Whether the model can sustain the no-working-state discipline (fallback: an
  additive `work` zone).
- TypeScript versus a Python-shaped eval surface for this model.
- Header-plus-body action format versus the native Pythonic tool-call wrapper.
- Mask-bind rate and any quality loss from grammar constraints.
- Edit reliability per format.
- Keep (default) or drop the parent's context while blocked on children.
- Base versus instruction-tuned checkpoint; 350M versus 230M.
- Paraphrases per program; real-text share of the mix; cross-lingual transfer;
  forward chaining at this scale.
- Latency: dynamic grammar compilation per turn, static checking of eval
  snippets.
- **Inference engine fit for LFM2's hybrid conv/attention architecture**:
  dynamic per-request grammars, logprob access, and prefix caching across the
  turns of an episode (now relevant, since episodes are multi-turn). Verify
  early in llama.cpp, vLLM, and SGLang.

### 10.4 Deferred

- **On-device versus server.** Not needed: the language, harness, type layer,
  and data are the same either way, escalation is already deferred, and the
  scheduler batches whatever is pending in both settings. Default: develop on
  a single GPU workstation, keep both batch and reactive programs
  demonstrable, and let the engine-fit check (§10.3) choose the inference
  engine.
- **Budget.** Default assumption: one GPU and time, no API spend. Both teacher
  candidates are self-hostable, the 350M fine-tune is under a day on one GPU,
  and algorithmic labels come free from the reference policy. Becomes a real
  question only if both teachers fail the conformance suite and a hosted model
  is needed.

- Voting, confidence thresholds, calibration, escalation, surfaced confidence,
  speculative execution (pending logged finite-type write distributions).
- Postconditions (`ensures`) and type refinements, crisp or judged.
- Provenance idioms: `rerun`, staleness, dataflow.
- Content-addressed store choice underneath SQLite.
- Language tags on Text nodes holding code.
- Security model for eval with real file and network access.
