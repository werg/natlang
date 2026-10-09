# Natural-language refinement types

Status: design. The owner decided on 2026-10-09 to build this and to carry it through training, authoring and datasets.
Companions: [architecture plan](ARCHITECTURE_IMPROVEMENT.md) E4/D5, [batched execution](BATCHED_EXECUTION.md).

## 1. Idea

A refinement type is an ordinary natlang type plus a natural-language predicate that its values must satisfy:

```ts
type Reply   = Is<string, "a reply to the customer that is polite and does not blame them">;
type Subject = Is<string, "one line of at most 60 characters, without a trailing period">;
type Finding = { claim: string; quote: Is<string, "copied verbatim from the cited source"> };
```

The runtime checks the predicate where a value enters a refined slot: an `nl` call's return, an argument passed to a
function whose parameter is refined, or a field of either. The check is one `readout: decision` scoring pass ("does
this value satisfy the predicate?"). It is cheap, batched (BATCHED_EXECUTION.md) and cached by content. A failure is
an ordinary type error with a stable code. The executing model repairs it the way it repairs any type error.

This moves constraints out of prompts and into types:

- Repeated prompt guards in the applications become types: "write exactly a, b, c", "never treat X as instructions",
  "read only the named file", "keep it under N words".
- Under `negative-examples-prime`, a warning that spells out the wrong form doubles that form. A refinement states only
  the wanted property, and the checker enforces it.
- Under `language-design-restraint`, constraints are taught where they apply: in the error that names the fix, not in
  the system prompt.

## 2. Surface

**One new name, `Is<T, P>`.** `T` is any natlang type and `P` a string literal holding a positive description of the
values. One predicate per `Is`, because natural language already expresses conjunction ("polite and under three
sentences"). Nesting (`Is<Is<string, "a">, "b">`) is allowed and means both. No algebra over predicates, no
entailment.

**TypeScript view.** Declared in the natlang ambient types as a brand:

```ts
declare const refinement: unique symbol;
type Is<T, P extends string> = T & { readonly [refinement]: { [K in P]: true } };
```

- Crisp TypeScript cannot pass a plain `string` where `Is<string, P>` is expected without a check. That is the point:
  refined values are evidence.
- Crisp code obtains one from a refined `nl` result, from `refine<R>(value)`, which runs the check and throws
  `refinement-unsatisfied`, or from `assume<R>(value)`, which is explicit, unchecked and traced.
- A refined value is assignable to its base type, so `Is<string, P>` is a `string` everywhere.

**Natlang type grammar.** `native/types.ts` gains `{ kind: 'refined'; base: Type; predicate: string }`. The fit
relation:

- `refined(B, P)` fits `T` iff `B` fits `T`. The refinement is forgotten.
- `T` fits `refined(B, P)` iff `T` fits `B`, with a check obligation recorded at that site.
- `refined(B, P)` fits `refined(B', P)` iff `B` fits `B'` and the predicates are identical after whitespace
  normalization. No obligation.

Frontmatter `args`/`returns` and `types.ts` accept the same text. The `.d.nl.ts` declarations emit the brand.

**Where checks run (the obligation sites).**

1. **Return of an `nl` call or `.nl` function**, after the ordinary structural check.
   - A failure goes back to the executing model as a tool error on `return_result`. It quotes the predicate and the
     judge's verdict and says one sentence on the fix.
   - The model gets the ordinary repair budget, then the call fails with `refinement-unsatisfied`.
2. **Arguments** into a refined parameter, at the caller.
   - A value already known to satisfy the predicate (§3) is not rechecked.
   - A failure throws `refinement-unsatisfied` to the caller and does not start the callee.
3. **Inside eval.** A refined binding is checked when the model writes it, the same moment structural values are
   checked today, so a failure comes back as the eval's error.
4. **Services.** A service's declared result type may be refined; its result is checked like an `nl` return.

Records, lists and dicts with refined members check each refined position. A list of N refined strings is N
predicate checks, issued as one batch.

**Error codes.**

- `refinement-unsatisfied`: the predicate was judged false.
- `refinement-undecided`: the judge's probability fell inside the configured uncertainty band, and the policy says
  to fail rather than to accept.
- `refinement-predicate-invalid`: `P` is empty or not a string literal. This is a compile error.

Each error message is one sentence that names the predicate and the fix. The authoring and integration skills get the
recovery advice in the same push.

## 3. The check

**Judge.**

- The judge is a built-in natural-language function,
  `holds(value: unknown, predicate: string): boolean, readout: decision`.
- It runs on the program's model unless the program configures `refinements.judge` to another model or a crisp
  implementation.
- Its instructions are short and fixed: the value is rendered as data, the predicate as the question. The verdict
  is P(true) from one scoring pass.

**Thresholds.**

- `accept ≥ 0.5` by default, plus an optional uncertainty band `[low, high]`. Inside the band the configured policy
  applies: accept, reject, or escalate to a stronger judge.
- Escalation reuses the existing model configuration, for example the teacher.
- Settings live in `natlang.json` under `refinements`, per predicate or globally.

**Crisp implementations, pluggable (`pluggable-hot-paths`).**

- `types.ts` may export `refinements: { [predicate]: (value) => boolean | undefined }`.
- A crisp implementation that returns a boolean decides. One that returns `undefined` defers to the judge.
- A setting selects crisp, natural language, or both with shadow comparison. Shadow disagreements are recorded as
  training data (§5).

**Cache and evidence.**

- Verdicts are cached by `(sha256(canonical value), normalized predicate, judge id)` in the call store, so the same
  value is never judged twice for the same predicate.
- A value returned from a refined slot carries its evidence in the trace: the predicate, the verdict, the probability
  and the judge.
- Primitives cannot carry hidden tags at run time, so "already known" means a cache hit on that content.

**Tracing and specialization.**

- Every check is a recorded call (the `calls/` store), with the value, predicate, probability and the outcome after
  repair.
- The specializer can learn a crisp checker for a hot predicate (tier 3 in the tiered engine, ARCHITECTURE_IMPROVEMENT
  E1), guarded by shadow replay.

## 4. Taint as a refinement: `Untrusted<T>`

`Untrusted<T>` is the provenance counterpart and is defined in the standard library, not the core grammar:

- Values from the outside world (files, HTTP, user text, tool output) enter as `Untrusted<string>` where a service
  declares it.
- The model-facing renderer always shows an untrusted value as a quoted data block labelled with its source, never
  spliced into instruction text. Today's "never treat X as instructions" sentences become unnecessary.
- `Untrusted<T>` fits `T` for crisp code. It does not fit an instruction position, meaning the template text of an
  `nl` call or an `.nl` body; using it there is the compile error `untrusted-instruction`.

## 5. Training and data

Refinement types only pay off if the student checks predicates well and repairs failed ones. Five data families,
each registered through `training/neuralese_corpora.json` with manifests, and admitted explicitly as usual:

1. **Predicate verdicts** (`refine-judge`): `(value, predicate) → P(true)` as decision-readout records.
   - Sources: (a) mined from application traces in the call store, labelled by the teacher with calibrated
     probabilities; (b) synthetic near-miss pairs, i.e. a value that satisfies the predicate and a minimally edited
     one that does not (one sentence too long, one blaming phrase), generated by the teacher and verified by a second
     teacher pass; (c) crisp-checkable predicates (length, format, enumerations) whose crisp checker gives exact
     labels for free.
   - Split by predicate, so held-out predicates measure generalization, not memorization.
2. **Repair trajectories** (`refine-repair`): whole trajectories, per `whole-trajectory-supervision`, where the
   executor returns a value, the refinement fails with the real error text, and the executor repairs it.
   - Collected live from the teacher on the rebuilt applications.
   - Trained like type-error recovery today, with the mechanical feedback (the error text) at lower weight.
3. **Authoring examples** (`refine-author`): `.nl` functions and `types.ts` files where a guard sentence in the
   instructions is replaced by a refined return type.
   - These teach models that write natlang (the teacher, the improver, the specializer's writer) to reach for
     refinements.
   - Sources: the rewrite of the application guards (§6) and S2 skill-authoring generation.
4. **Calibration set** (`refine-calibration`): a small held-out set of human-reviewed verdicts per predicate family.
   It measures the judge's calibration (Brier and ECE on the decision readout) and gates the default thresholds.
5. **Shadow disagreements**: crisp checker against natural-language judge disagreements from §3. These are hard
   negatives for family 1 and bug reports for the crisp checkers.

The Neuralese line reuses family 1 directly: a predicate judge is a decision readout, so it trains with the existing
`decision` objective (`neuralese/learning.ts`), and a learned soft predicate is a natural Neuralese artifact for a hot
predicate.

**Evaluation gate before making refinements the recommended style.** Per `negative-examples-prime`, measure live.
Take the guard-bearing functions in the applications. For each, compare (a) today's prompt guard, (b) the refined
return type with the guard removed, and (c) both. Run about 48 samples each on the live executor (student and
teacher) and report the constraint violation rate, the repair rate and the extra model calls. Adopt (b) where it is
no worse on violations at acceptable cost.

## 6. Authoring

- `skills/natlang-authoring` gains a section "Constraints belong in types". It covers when to use `Is<T, P>` (a
  property a reader can judge from the value alone), how to phrase `P` (positively, concretely, one property or a
  short conjunction, no negative examples), and when not to use it: properties that need outside knowledge belong in
  a service or a separate function.
- `skills/natlang-integration` covers `refine`, `assume`, the settings, crisp checkers and the error codes with their
  recovery advice.
- `natlang check` reports every refined slot and the predicates it will check, beside the capture report (ARCHITECTURE
  D3).
- The improver and the specializer may propose refinements: a predicate mined from repeated repairs or evaluation
  failures becomes a candidate `Is<…>` on the function's return, proposed and evaluated like any other edit.

## 7. Implementation steps

1. Types: the `refined` kind in `native/types.ts` (parse, format, fit with obligations), the TS ambient
   `Is`/`refine`/`assume`, and `.d.nl.ts` emission. Unit tests for fit and parse.
2. Checks: obligation sites in the kernel's return path, the argument path, eval writes and service results. Judge
   through the decision scorer, with the cache in the call store and trace events. Tests with a stub scorer.
3. Errors and skills: codes, messages and recovery advice, with both skills updated in the same push.
4. Settings: `refinements` in `natlang.json`, crisp checkers from `types.ts`, shadow mode.
5. `Untrusted<T>`: renderer and the `untrusted-instruction` check.
6. Data: generators for families 1–3, registry entries and the calibration set.
7. Live evaluation (§5) in the teacher window, then adoption across the rebuilt applications.

Steps 1–4 are host and compiler work with stub-model tests. Steps 6–7 need the teacher window and the ledger.
