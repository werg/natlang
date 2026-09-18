# natlang: type safety and validation

> The normative grammar, fit rules, and diagnostic codes are in
> `spec/SPEC.md`. This document gives the rationale and the training-side
> design. Where they differ, the spec wins.

Companion to `PLAN.md` §2–4. Defines how lambdas are typed, what the harness
validates and when, and how validation feedback reaches the model without
turning it into a compulsive error-fixer. Status: draft, 2026-09-18.

## 1. Goals

1. Every lambda **can** be typed entirely: inputs, return, working state,
   effects, and postconditions.
2. The harness can validate **any** state, intermediate or final, against
   those types, as a pure function of the tree.
3. Validation is first an *error-containment mechanism* for the runtime and a
   *labelling and filtering signal* for training, and only then feedback to
   the model.
4. The model keeps the ability to **draft**: to leave things incomplete, work
   in the order the instructions suggest, and finish later.

> **Scope note (2026-09-18).** v1 typing is **structural only**. Refinements
> (`where` clauses) and postconditions (`ensures`) described below are
> **deferred** to avoid over-complicating the first build. `Map` and `Fold`
> already guarantee length, order, and accumulator typing by construction,
> which covers the most common checks they were meant for. Wherever this
> document lists `ensures` or refinements as commit conditions, read them as
> "when enabled".

## 2. The type language

A subset of TypeScript type syntax, as already used for `returns`.

| Form | Example |
|------|---------|
| Primitives | `Text`, `Num`, `Bool`, `Null`, `Blob` |
| Literal unions (enums) | `"urgent" \| "normal" \| "spam"` |
| Records, optional fields | `{ id: Num, label: Text, note?: Text }` |
| Lists | `T[]` (canonical; `List[T]` in older text means the same) |
| Dicts | `Dict<T>` (string keys) |
| Unions | `Num \| Null` |
| Lambda | `Lambda<P, T>`: params record `P`, return `T` |
| Gradual escape | `Any` |
| Named types | `type Label = { ... }` declared on a lambda, visible to its subtree |

**Refinements.** Crisp predicates attached to a type, written as short eval
expressions: `Num where (x) => x >= 0 && x <= 1`, `Text where maxLen(200)`,
`List[T] where nonEmpty`. A small library of named refinements covers the
common cases so they rarely need code. Refinements are structural checks'
slower cousins: always computed, but only *enforced* at commit points (§4).

### 2.1 A fully typed lambda

```
labels/   Lambda<{ inbox: List[Text], rubric: Text },
                 List[{ id: Num, label: "urgent"|"normal"|"spam", confidence: Num }]>
  instructions   Text
  params         { inbox: List[Text], rubric: Text }
  returns        List[{ id: Num, label: ..., confidence: Num where unit }]
  effects?       []                                        # pure; or ["ticketing.write"]
  ensures?       [ "return.length == inbox.length",
                   "ids(return) == range(inbox.length)" ]  # crisp postconditions
  inbox, rubric  (bound inputs)
  return         (written during reduction)
```

- **`params`** types the closure. A lambda with a required param unbound is
  *partial*. Partial application is well-typed; **reducing a partial lambda is
  a type error.** Currying is now visible to the type checker.
- **`returns`** as before.
- **No working state, no inference.** A lambda has `instructions`, `args`, and
  `return`, nothing else (`PLAN.md` §2.2.1). Intermediate data is a parameter
  of the lambda that will consume it. **Every node is created with an explicit
  type chosen by the agent**; `Lambda<P, T>` is constructible like any other
  type. There is no `Any` scratch area: prose drafts live in `Text`,
  structured drafts are covered by `Draft<T>`.
- **`args` is frozen on reduce.** The parent may write a child's `args` while the
  child is unreduced (accumulate, edit, then ship). Once `reduce` is
  triggered, `args` is immutable, and it is always read-only to the lambda
  itself. It unfreezes if the child fails.
- **Crisp lambdas and combinators are typed too** (`PLAN.md` §2.4). A crisp
  lambda's `code` is statically checked against its `params` and `returns`
  when written. `Map<A, B>` requires `over: List[A]` and
  `fn: Lambda<{ item: A }, B>` and fits any slot of type `List[B]`;
  `Fold<A, S>` requires `step: Lambda<{ acc: S, item: A }, S>` and fits a slot
  of type `S`. `Iterate<S>` requires `step: Lambda<{ state: S }, S>`, a
  `check` returning `LoopVerdict`, and a `max`, and fits a slot of type `S`.
- **Enum narrowing** is the one subtyping rule: a lambda whose `returns` is a
  narrower enum fits a slot declared with the wider enum. It lets crisp code
  construct a decision lambda restricted to exactly the currently legal
  options (`PLAN.md` §2.4).
- **`effects`** lists the side-effect capabilities the lambda may use. A
  lambda with no `effects` is pure. A child cannot have effects its parent
  lacks. Effects can be caused by `eval` calls and by crisp lambdas; both are
  checked against the enclosing lambda's declared `effects`. `eval` never
  writes to the tree.
- **`ensures`** are optional crisp postconditions over inputs and `return`:
  checks the harness runs at commit that relate the result to the inputs,
  which types alone cannot express. `Map` and `Fold` now guarantee length,
  order, and accumulator typing by construction, so `ensures` matters mainly
  for hand-built results. They are
  the cheapest high-value check in the system for fuzzy maps: length
  preserved, ids are a permutation, labels inside the enum, sums match.
- **Higher-order.** `Lambda<P, T>` is a type, so params can be lambdas:
  `classify: Lambda<{ item: Text }, Label>`. Generics live only in stdlib
  signatures (`spawnEach<A, B>(xs: A[], f: Lambda<{item: A}, B>): Lambda<{}, B>[]`),
  where TypeScript handles them; the tree's type language stays monomorphic.

### 2.2 Gradual typing and elaboration

Typing is available everywhere and required nowhere: a user can write an
untyped lambda (`params`, `returns` = `Any`). The more that is declared, the
more the harness can contain errors.

**Type elaboration** closes the gap for hand-written programs: before
reduction, a typing pass proposes `params`, `returns`, and `ensures` for an
untyped lambda. The pass is itself a lambda (run by the interpreter model, or
escalated to a larger model), and its output is checked by the harness for
consistency with the bound inputs. Synthetic training programs are generated
fully typed, with a minority untyped or partially typed for robustness.

### 2.3 The invariant: type preservation

The harness enforces *subject reduction* for the object tree:

> Every accepted action takes a draft-well-typed tree to a draft-well-typed
> tree. Completion takes a node of type `Lambda<P, T>` to a node of type `T`.

Consequently a slot declared `T` accepts either a `T` or a `Lambda<_, T>`
(a typed promise), and the swap-out on completion can never break the parent's
types. *Progress* is not guaranteed, since leaves are fuzzy; *preservation*
is. This is a CI property for the harness and a filter for all training
traces: a trace that violates preservation at any step is discarded.

## 3. Draft types: holes are fine, lies are not

The balance between type safety and drafting rests on one distinction.

- **Incompleteness** (a *hole*): a required field not yet written, a list
  shorter than it will be, a refinement not yet satisfied because the value is
  still being built. Legitimate mid-draft.
- **Contradiction**: a `Text` where a `Num` is declared, a label outside the
  enum, a field the type does not have. Never legitimate; it is wrong now and
  will be wrong later.

Every declared type `T` has a derived draft type `Draft<T>`: `T` made deeply
partial. During reduction, typed nodes are checked against `Draft<T>`; at
commit points against `T`.

| | During drafting | At commit |
|---|---|---|
| Contradiction | **write rejected** | n/a (cannot exist) |
| Hole | accepted, recorded as diagnostic | **blocks commit** |
| Refinement / `ensures` violation | accepted, recorded | **blocks commit** |
| Pending `Lambda` inside a record or list in `return` | accepted (typed promise) | **blocks commit** until reduced |
| Pending `Map` / `Fold` node in a slot | accepted (typed promise for `List[B]` / `S`) | **blocks commit** until reduced |

Rejecting contradictions does not bias the model toward premature fixing,
because a rejected write never enters the state: the "fix" is simply the
correct version of the action the model was already taking.

## 4. Commit points

Full conformance is demanded only where an error would *escape*:

1. **Completion**: emptying `instructions`. Requires `return : T`, all
   `ensures` true, refinements satisfied, no unreduced children the return
   depends on.
2. **`reduce` of a child**: the child's `params` must be fully bound and
   well-typed; its declared `returns` must fit the slot it occupies.
3. **Side-effecting calls**: arguments fully typed, effect declared.
   Hard-checked, never draft.
4. **Explicit checks**: an `assert`/`check` step written in the instructions.

A failed commit is refused, nothing changes, and the diagnostics that blocked
it become the salient part of the next observation. At that moment, and only
then, fixing is the required next action.

## 5. The validator

A pure function in the harness, run after every action on the affected
subtree and on demand for any path:

```
validate(tree, path, mode: "draft" | "commit") -> Report
Report = { ok: Bool, diagnostics: Diagnostic[] }
Diagnostic = { path, code, severity: "reject"|"blocks-commit"|"hole"|"info",
               expected, got, message }
```

Messages are short, stable, and structured for a small model:
`return/3/confidence: expected Num, got Text "high"`. Fixed codes, fixed
ordering, a cap on items shown, no stack traces.

The report is **always computed and logged**, whatever the model is shown. It
feeds: runtime error containment (§6), trace filtering, per-skill metrics,
DAgger labelling, RL reward at commit, and debugging. "Validate any
intermediate or final state" is this function applied to any snapshot.

**Eval code.** Because eval is TypeScript, the harness generates a `.d.ts`
for the current lambda's scope from the declared types, and eval snippets are
type-checked *statically* before they run, using a warm, persistent checker
process. Snippets are a few lines against a small declaration file, so the
budget is small; measure in Phase 1, and fall back to runtime validation at
the tree boundary (which always happens anyway) if latency is a problem.
Static errors are real TypeScript diagnostics, which is what the type-repair
drills train on.

## 6. Decision: go deep on write-time typing

**Decided 2026-09-18.** With a weak model, the type system applied *at write
time in the harness* is the primary reliability mechanism, and it is built
out fully. The statistical machinery discussed earlier (voting, confidence
from token probabilities, calibration, escalation to a larger model) is
**deferred**: it complicates the harness, part of it rests on an unmeasured
assumption, and it can be added later without touching the language. Simpler
options are recorded in §6.4 for each deep component in case one hits a wall.

**The boundary that keeps this consistent with "the model is the
interpreter":** the harness constrains what is *well-formed and well-typed
against the current tree*. It never chooses *which* well-typed action to
take, never reads the instructions, and never owns control flow.

### 6.1 What "deep" means, tool by tool

Every action is decoded under a grammar derived from the **current tree and
its types**, so that ill-formed and ill-typed actions cannot be emitted.

| Part of the action | Constrained to |
|--------------------|----------------|
| Tool name | the six tools |
| `read(path, range)` | paths that exist in the lambda's scope, plus meta suffixes (`@problems`, provenance); ranges within bounds |
| `edit(path, …)` | paths of type `Text`; line ranges that exist |
| `set(path, type, literal)` — path | existing paths, or a creatable child of an existing container, inside scope, not harness-owned |
| `set(path, type, literal)` — type, literal | for a new node, a type literal that fits the parent slot, then the grammar of `Draft<T>` for it; for an existing node, the declared type at that path (`T` for side-effect arguments); enums to their members; records to their fields; a new `Lambda` literal to the Lambda schema, with `returns` required to fit the slot |
| `copy(src, dst)` | `src`: existing paths and valid sub-ranges; `dst`: **only slots whose type accepts the source's type**, inside scope, not frozen |
| `reduce(paths)` | paths whose type is `Lambda`, status unreduced or failed, params fully bound |
| `eval` body | statically type-checked against the generated `.d.ts` before running (§5); optionally *type-constrained decoding* of the snippet itself (§6.3, stage F) |

Decoding is multi-phase: tool name, then path, then the harness selects the
grammar for the remainder from the type at that path. Grammars are cached by
type hash. JSON-schema-to-grammar exists in llama.cpp; XGrammar and Outlines
provide the equivalent for vLLM and SGLang.

Path constraint matters as much as literal constraint: in the RLM training
data, 13 % of turns referenced variables that did not exist
(`SYNTHETIC_DATA.md` Y9). Under a tree-derived path grammar that class of
error cannot occur.

### 6.2 The layers that remain, in order

1. **Unrepresentable**: type- and tree-directed decoding (§6.1).
2. **Resample on reject**: whatever still fails validation (a contradiction
   reachable only through `eval`, an over-long action, an effect violation)
   is discarded and resampled up to k times before the model sees anything.
   This is a loop around the sampler, and the only piece of the "statistical"
   family kept in v1.
3. **Reject with message** when resampling is exhausted.
4. **Diagnostics as state** (§7).
5. **Commit gate** (§4).
6. **Quiesce upward**: repeated commit failure leaves the lambda `quiesced`,
   as it was, with a reason, for the parent to handle (`PLAN.md` §2.2).

**Logged, with no policy attached.** When a write targets a finite type
(`Bool`, an enum), the probability distribution over the allowed members is
recorded in provenance. It costs nothing under type-directed decoding, and it
is the data needed to decide later whether confidence-based features are
worth building. A "predicate" in this system is nothing more than a write to
a finite-typed node; there is no predicate construct. For the record to be
interpretable, a judgment that matters should be its own small lambda with a
finite `returns`, a convention taught by data.

**Deferred.** Voting, thresholds on the recorded distributions, calibration,
escalation to a larger model, a surfaced `path@confidence` meta path,
speculative execution of both branches.

### 6.3 Build stages for the type layer

| Stage | Contents |
|-------|----------|
| A | Type parser; structural validator; static tool-call grammar; resample-on-reject; preservation check in CI |
| B | Tree-derived path grammars; type-directed literals for `set`; grammar cache |
| C | `Draft<T>`, commit gates, hole rendering, `@problems` |
| D | `.d.ts` generation per scope; warm TypeScript checker for eval snippets |
| E | Refinements, `ensures`, effects |
| F | Type-constrained decoding of eval snippets (research-grade; prior work exists for TypeScript: Mündler et al., "Type-Constrained Code Generation with Language Models", 2025, *cited from memory, verify*) |

Stages A–C deliver most of the value. D–E deepen it. F is optional.

### 6.4 Simpler options if we hit a wall

Each rung is independent, and none changes the language or the training data
format beyond what the model is shown.

| If this is the wall | Fall back to |
|---------------------|--------------|
| Per-step dynamic grammars are too slow or too complex to build | Static tool-call grammar + validate + resample (stage A only) |
| Path grammars are too large on big trees | Constrain to paths present in the current rendering; or validate + resample |
| The constraint distorts the model (see below) | Loosen to structural JSON only; rely on validate + resample |
| Static checking of eval is too slow | Runtime validation at the tree boundary only (always present anyway) |
| `Draft<T>` and commit semantics confuse the model | *Strict mode*: typed nodes accept only complete values; or *lenient mode*: validate only at commit |
| Structuring all intermediate data as lambda params is too hard for the model | Add a typed `work` zone per lambda (additive) |
| Refinements and `ensures` cost more than they catch | Structural types only |
| Effects typing is overkill | A per-run tool allowlist |
| The whole layer is too heavy | The original design: a schema on `returns`, checked once at completion |

### 6.5 Known risk: constraint distortion

Forcing a grammar can push a model into continuations it assigns low
probability, and some studies report quality loss from format restriction
(*from memory: Tam et al. 2024, "Let Me Speak Freely?", with later
rebuttals; verify*). Token-boundary effects between grammar and tokenizer are
a second known nuisance. Mitigations:

- **Train under the same constraints used at inference**, so the mask rarely
  binds. All SFT targets are valid under the grammar by construction.
- Track the **mask-bind rate**: how often the model's unconstrained top
  choice was masked out. It should fall toward zero with training. Where it
  stays high, the model misunderstands that type or path, which is a precise
  pointer to missing training data. Report it per tool part and per type.

## 7. How the model sees validation

**Feedback is state, not interruption.** The context is a cache and the tree
is the state (`PLAN.md` §3.2), so diagnostics
are rendered as part of the observation, the way an editor shows squiggles:
present, not demanding.

- **Holes are rendered inline as progress**, not as errors:

  ```
  return/     List[{id,label,confidence}]   37 of 40 complete
    37        { id: 37, label: ·, confidence: · }
  ```

  A typed, partly filled `return` is a to-do list. For a cursor-free
  interpreter this is a *help*: the type tells the model what remains.
- **A one-line problems summary** at the top of the observation:
  `problems: 0 blocking · 3 holes`. Counts, not lists.
- **Pull for detail.** The full report is a meta path, like provenance:
  `read("return@problems")`. No new tool. The model decides when to look,
  and *when to look* is a trainable policy (before commit; after a bulk
  write).
- **Blocking diagnostics are listed in full only after a refused commit.**

## 8. Training the balance

The harness imposes no control flow, so the balance is set by data. The
reference policy defines the canonical behavior and the data makes sure that
*seeing a diagnostic does not predict "fix now"*; the **kind and phase** do.

**Policy the reference interpreter demonstrates**

| Situation | Correct next action |
|-----------|---------------------|
| Holes present, instructions not finished | **continue with the next instruction step** |
| Hole concerns the node the current step is about | fill it (that *is* the step) |
| Unmet refinement on a value a *later* step will consume | fix before that step |
| Write rejected | retry the same intent, corrected |
| Commit refused | fix the blocking diagnostics, then commit again |
| Commit refused repeatedly on the same node | mark uncertain / fail upward |
| Nothing blocking, work done | commit |

**Data constructions**

- **Near-miss twins on diagnostics** (extends `SYNTHETIC_DATA.md` Y8): same
  problems panel, different phase, different correct action. One twin
  continues drafting; the other fixes. This is the direct antidote to
  over-bias.
- **Controlled base rate.** Across the mix, P(fix is the next action | a
  diagnostic is visible) stays well below 1. Holes: continue nearly always.
  Blocking at commit: fix always. Refinements: mixed by whether a consumer is
  imminent. Track this rate as a dataset statistic.
- **Diagnostics dropout.** In a fraction of examples the summary line is
  hidden, so the model neither depends on it nor treats its presence as a
  cue. At evaluation, compare three modes: none, summary, full.
- **Commit drills**: refused commit plus blocking list → targeted fix;
  repeated refusal → fail upward. **Draft drills**: long fills of a typed
  `return` with holes visible throughout, where every correct action is
  "next item".
- **Elaboration examples**: untyped lambda + bound inputs → proposed
  `params`/`returns`/`ensures`.

**Reward shaping (RL).** No per-step reward for reducing the diagnostic
count; that is exactly the gradient that produces compulsive fixing. Small
penalty for a refused commit, large for a final invalid state, reward for
valid completion with correct content. Rejected writes that were silently
resampled carry a small cost so the model still learns to avoid them. (RL
is itself a late stage; nothing here depends on the deferred confidence
features.)

**Metrics that detect imbalance**

- *Detour rate*: steps where the model leaves instruction order to fix a hole
  a later step would have filled anyway. Over-bias shows up here first.
- *Fix thrash*: more than k writes to the same node.
- *Commit-failure rate* and *steps-to-valid-completion*. Under-bias shows up
  here.
- *Rejected-write rate* before and after type-directed decoding, and the
  *mask-bind rate* (§6.5) per tool part and per type.
- Completion accuracy under the three visibility modes.

The target is a model whose detour rate and commit-failure rate are both low;
moving one at the expense of the other means the data mix, not the harness,
needs adjusting.

## 9. What this adds to the build

- Harness, in the stage order of §6.3: type parser, structural validator,
  static grammar, resample loop; tree-derived path grammars and type-directed
  literals; `Draft<T>`, commit gates, hole rendering, `@problems`; `.d.ts`
  generation and warm TS checker; refinements, `ensures`, effects. Finite-type
  write distributions logged to provenance.
- Reference policy: the §8 behavior table.
- Data: diagnostic twins, commit and draft drills, elaboration examples,
  dropout; preservation as a trace filter.
- Evaluation: the §8 metrics, reported per visibility mode.

## 10. Open questions

- Judged (fuzzy) postconditions, e.g. "the summary mentions every theme":
  useful, but they cost model calls. Probably opt-in, commit-only.
- How much of the declared type to render inline versus on demand, given the
  observation token budget.
- Exact typing rules for `copy` with sub-ranges (a `Text` line range is
  `Text`; a `List[T]` slice is `List[T]`; a record field is its field type),
  and how `Draft<T>` sources copy into `T` slots.
- Latency of static checking per eval; whether to check only evals that
  write to typed nodes.
