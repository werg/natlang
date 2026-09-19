# natlang: type safety and validation

> The normative grammar, fit rules, and diagnostic codes are in
> `spec/SPEC.md`. This document gives the rationale and the training-side
> design. Where they differ, the spec wins.

Companion to `PLAN.md` §2–4. Defines how functions and their instances are
typed, what the harness validates and when, and how validation feedback
reaches the model without turning it into a compulsive error-fixer. Status:
draft, 2026-09-19.

## 1. Goals

1. Every function **is** typed entirely by its author: parameters, return,
   effects; and every local an instance creates is typed at creation.
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

A subset of TypeScript type syntax, used in function frontmatter and for locals.

| Form | Example |
|------|---------|
| Primitives | `Text`, `Num`, `Bool`, `Null`, `Blob` |
| Literal unions (enums) | `"urgent" \| "normal" \| "spam"` |
| Records, optional fields | `{ id: Num, label: Text, note?: Text }` |
| Lists | `T[]` (canonical; `List[T]` in older text means the same) |
| Dicts | `Dict<T>` (string keys) |
| Unions | `Num \| Null` |
| Function instance | `Lambda<P, T>`: params record `P`, return `T`; derived from a signature, never written by the interpreter |
| Named types | declared in a function's `types` frontmatter or a folder's `types.ts`; visible in the function and inherited by its code base |

**Refinements.** Crisp predicates attached to a type, written as short TypeScript
expressions: `Num where (x) => x >= 0 && x <= 1`, `Text where maxLen(200)`,
`List[T] where nonEmpty`. A small library of named refinements covers the
common cases so they rarely need code. Refinements are structural checks'
slower cousins: always computed, but only *enforced* at commit points (§4).

### 2.1 A fully typed function

```
---
description: Label every ticket and count the urgent ones.
args:    { inbox: Text[], rubric: Text }
returns: '{ id: Num, label: "urgent"|"normal"|"spam", confidence: Num }[]'
effects: []                                  # pure; or ["ticketing.write"]
---
```

- **The signature is the author's.** `args` and `returns` in the frontmatter
  give the instance its type `Lambda<P, T>`. The interpreter never states the
  type of a sub-task: a `call` derives every type from the callee's signature
  (the result slot, the element type of `over`, the accumulator, the state),
  and refuses a call whose inputs do not fit or whose required parameters are
  unbound.
- **Locals are typed at creation.** `let/<name>` is created by the first
  write: a plain `write` states the type, a `call` derives it, a `write` with
  `source` takes the source's. There is no `Any` scratch area: prose drafts
  live in `Text` locals, structured drafts are covered by `Draft<T>`.
- **There are no anonymous lambdas.** A `write` whose type is a pending node
  is rejected; functions come only from the immutable code base, or from an
  editable copy of one of its functions, which keeps the original's signature.
- **`args` is frozen** once the instance starts, and always read-only to the
  instance itself.
- **Crisp functions and combinators are typed too** (`PLAN.md` §2.4). A crisp
  function's `code` is checked against its signature. `Map<A, B>` requires
  `over: A[]` and a function whose one unbound parameter takes `A`, and fits
  any slot of type `B[]`; `Fold<A, S>` requires a function `(acc: S, item: A)
  -> S` and fits a slot of type `S`; `Iterate<S>` requires a step `S -> S`, a
  check function `S -> Bool`, and a `max`, and fits a slot of type `S`. All
  three are built only by `call`.
- **Enum narrowing and literal widening**: a narrower enum fits a wider one,
  and a literal fits its base type (`Label[]` fits `Text[]`), which is what
  lets typed results flow into generic library functions.
- **`effects`** lists the side-effect capabilities the function may use. A
  function with no `effects` is pure. A callee cannot have effects its caller
  lacks. Effects can be caused by `run_code` and by crisp functions; both are
  checked against the enclosing function's declared `effects`. `run_code`
  never writes to the tree.
- **`ensures`** (deferred) are optional crisp postconditions over inputs and
  `return`. `Map` and `Fold` guarantee length, order, and accumulator typing
  by construction, so they matter mainly for hand-built results.

### 2.2 Elaboration (deferred)

Frontmatter is required, so every program is typed. A typing pass that
proposes `args` and `returns` for a bare pseudocode text is an authoring-time
aid and out of scope for the interpreter.

### 2.3 The invariant: type preservation

The harness enforces *subject reduction* for the object tree:

> Every accepted action takes a draft-well-typed tree to a draft-well-typed
> tree. Completion takes a node of type `Lambda<P, T>` to a node of type `T`.

Consequently a slot declared `T` accepts either a `T` or a call in progress
whose result type fits `T` (a typed promise), and the swap-out on completion can never break the parent's
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

1. **Completion**: the interpreter replies. Requires `return : T` with no
   holes and no pending nodes (and, when enabled, `ensures` and refinements).
2. **`call`**: every required parameter bound by a value that fits; the
   callee's result type fits the slot at `to`.
3. **Side-effecting calls**: arguments fully typed, effect declared.
   Hard-checked, never draft.

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

**Code.** `run_code` and crisp functions accept JavaScript and erasable TypeScript
annotations. Node.js strips annotations before QuickJS execution; stripping does
not check types. Inputs, stored values and crisp returns are validated at the tree
boundary. Generating scope declarations and running a static checker remain
future work (stage D below); current diagnostics are parser/runtime errors and
structural validation failures.

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

Every turn is decoded under a grammar derived from the **current tree, its
types, and the code base**, over the model's native tool-call text, so that
ill-formed and ill-typed calls cannot be emitted (`spec/SPEC.md` §5.8).

| Part of the call | Constrained to |
|------------------|----------------|
| Tool name | the six tools (`call` only when the lambda has functions) |
| `read(path, start, end)` | paths that exist in scope, `codebase/<f>`; positions that exist |
| `write(path, type, value)` | one alternative per writable slot: the path, **its type as a constant**, and the grammar of `Draft<T>` for the value; or a new `let/<name>` with a stated type |
| `write(path, type, source)` | sources whose type fits the slot |
| `write(path, "Function<f>")` | names of the code base |
| `edit(path, old, new)` | editable texts: own `instructions`, instructions of function copies |
| `call(function, to, inputs, …)` | function names of the code base and of copies; **per parameter, the paths whose type fits it**; `over` to lists; `until` to `Bool` functions of one parameter; `to` to writable slots or a new local |
| `run_code` body | runtime validation at the boundary; static checking against a generated `.d.ts` is a later stage (§6.3) |

Path, type and value belong together, which JSON Schema cannot say at the top
level of a tool's arguments; tools therefore carry `x-natlang-alternatives`,
and the grammar is built from those. Grammars are cached by type hash.

Path constraint matters as much as value constraint: in the RLM training
data, 13 % of turns referenced variables that did not exist
(`SYNTHETIC_DATA.md` Y9). Under a tree-derived path grammar that class of
error cannot occur, and with type-filtered `inputs` neither can passing the
wrong kind of value.

### 6.2 The layers that remain, in order

1. **Unrepresentable**: type- and tree-directed decoding (§6.1).
2. **Validate before mutation**, including any attached line marks.
3. **Reject with message**: the model sees failures and chooses a correction.
   There is no silent retry or budget refund; failed code may have performed effects.
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
| A | Type parser; structural validator; static tool-call grammar; rejection feedback; preservation check in CI |
| B | Tree-derived path grammars; type-directed values for `write`; signature-derived `call` alternatives; grammar cache |
| C | `Draft<T>`, commit gates, hole rendering, `@problems` |
| D | `.d.ts` generation per scope; warm TypeScript checker for `run_code` and crisp functions |
| E | Refinements, `ensures`, effects |
| F | Type-constrained decoding of `run_code` snippets (research-grade; prior work exists for TypeScript: Mündler et al., "Type-Constrained Code Generation with Language Models", 2025, *cited from memory, verify*) |

Stages A–C deliver most of the value. D–E deepen it. F is optional.

### 6.4 Simpler options if we hit a wall

Each rung is independent, and none changes the language or the training data
format beyond what the model is shown.

| If this is the wall | Fall back to |
|---------------------|--------------|
| Per-step dynamic grammars are too slow or too complex to build | Static tool-call grammar + validate + resample (stage A only) |
| Path grammars are too large on big trees | Constrain to paths present in the current rendering; or validate + resample |
| The constraint distorts the model (see below) | Loosen to structural JSON only; rely on validate + resample |
| Static checking of code is too slow | Runtime validation at the tree boundary only (always present anyway) |
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
| Holes present, program not finished | **continue with the next statement** |
| Hole concerns the node the current step is about | fill it (that *is* the step) |
| Unmet refinement on a value a *later* step will consume | fix before that step |
| Write or call rejected | retry the same intent, corrected from the hint |
| Commit refused | fix the blocking diagnostics, then commit again |
| Commit refused repeatedly on the same node | `report_blocker` |
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
- Data: diagnostic twins, commit and draft drills, dropout; preservation as
  a trace filter.
- Evaluation: the §8 metrics, reported per visibility mode.

## 10. Open questions

- Judged (fuzzy) postconditions, e.g. "the summary mentions every theme":
  useful, but they cost model calls. Probably opt-in, commit-only.
- How much of the declared type to render inline versus on demand, given the
  observation token budget.
- Whether a `call` may narrow its result type (a decision restricted to the
  currently legal options) or legality stays a matter of inputs plus a crisp
  transaction that refuses.
- Latency of static checking of `run_code` and crisp function bodies.
