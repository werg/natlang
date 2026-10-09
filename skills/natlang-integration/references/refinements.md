# Refinements (`Is<T, P>`): settings, errors and recovery

`Is<T, "predicate">` slots are checked by the runtime; see the [spec](../../../spec/SPEC.md) ("Is<T, P> refinement types") for the rules and [constraints belong in types](../../natlang-authoring/references/refinements.md) for authoring.

## What the host does

- The judge runs on the call's own model when its driver can score replies (`decide`: the same readout as `readout: decision`). `refinements.judge` names another entry of `models`. A driver that cannot score judges nothing: every uncovered predicate is `refinement-undecided`, so pair refined types with a scoring driver or a crisp checker.
- Verdicts are cached in memory by content. Pass `refinements: { cache }` to the runtime to keep them elsewhere (`get(key)`, `set(key, { probability, judge })`, both may be async); the key already includes the judge id, so one cache can serve several judges.
- Thresholds apply after the cache, so changing them never needs a new cache.
- Crisp checkers: `refinements: { crisp: { "<normalized predicate>": value => boolean | undefined } }` in the runtime options, or entries of the `refinements` table exported by the package.

## `natlang.json`

```json
{ "refinements": {
    "threshold": 0.5,
    "band": { "low": 0.35, "high": 0.65 },
    "policy": "escalate",
    "escalate": "teacher",
    "mode": "crisp",
    "repairs": 3,
    "predicates": { "one line of at most 60 characters": { "mode": "shadow" } },
    "services": { "mail.draft": "Is<string, \"a draft that is polite\">" } } }
```

`policy` decides a probability inside `band`: `accept`, `reject` (fails as `refinement-undecided`) or `escalate` (judge again on the `escalate` model, then accept at `threshold`). `mode: "shadow"` records every crisp/judge disagreement as a `refinement_shadow` trace event; those events are the bug reports for a crisp checker and hard negatives for the judge. `services` declares the refined result type of a service method, which the runtime checks like an `nl` return.

## Errors and what to do

| Code | Raised as | Recovery |
| --- | --- | --- |
| `refinement-unsatisfied` | `RefinementCallError` (a call's result), `RefinementError` (an argument, a service result, `refine`) | The value does not hold the predicate. The executor has already had its repair budget on a return. Fix the producer: tighten the instruction or the predicate wording, give the function the evidence the predicate refers to, or check the argument before calling. Do not retry the identical call. |
| `refinement-undecided` | the same | The probability is inside the uncertainty band under `reject`, or no judge could score (the driver cannot score replies). Use a scoring driver, add a crisp checker, widen the band policy to `escalate`, or make the predicate sharper. |
| `refinement-predicate-invalid` | `TypeSyntaxError` when the type is read; `RefinementError` from `refine`/`assume` | `P` must be a nonempty string literal, as in `Is<string, "one line of at most 60 characters">`. |

On a return the executing model gets the failure as a tool error naming the predicate and the fix, so a refined return that fails once usually costs one more turn, not a failed call. A call that must not spend extra turns can set `repairs: 0` and catch the error.

## Trace

Each check emits `refinement_check` (`phase`: `return`, `argument`, `service`, `refine`; `outcome`: `pass`, `fail`, `undecided`; `source`: `crisp`, `judge`, `cache`, `escalation`; predicate, value, probability, judge). `assume` emits `refinement_assumed`. A result an eval finishes (`finish: true`) skips the in-loop repair and is checked when the call completes, so its failure is final.
