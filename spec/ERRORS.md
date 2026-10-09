# Error codes

Every error code that [SPEC.md](SPEC.md), the [extensions](ext/neuralese.md) and the skills name, with the rule it
enforces, when it fires and the `ts-host/src` file that raises it. Each code was checked with `grep` against
`ts-host/src`; the mismatches are listed at the end.

**When it fires.**

- **Plan time**: reported by `natlang check` / `natlang build` before any model runs (the compiler plans every `nl`
  expression), or when a type is read.
- **Load/bind time**: when a context, file or `.nl` source is loaded, bound or saved.
- **Eval time**: when the model's eval code is checked or committed, before or while it runs. The model sees the
  rejection and repairs it.
- **Call time**: while a call or an `iterateOn` loop runs, or when a value crosses a refined slot.

## Codes

| Code | Rule | Fires | File |
| --- | --- | --- | --- |
| `nl-ambiguous-signature` | Conflicting evidence about an `nl` expression's parameters or result ([SPEC](SPEC.md#natural-language-functions)) | plan time | `compiler/inline.ts` |
| `nl-unknown-parameter` | The type found for a parameter cannot be used as a signature type; annotate it | plan time | `compiler/inline.ts` |
| `nl-unknown-return` | The return type of the `nl` expression cannot be determined from its target; annotate it or write `nl<T>` | plan time | `compiler/inline.ts` |
| `nl-sync-callback` | An `nl` function (async) put in a callback slot that expects a synchronous result | plan time | `compiler/inline.ts` |
| `nl-sequential-loop` | Warning: a loop of independent `nl` calls awaited one at a time; use `Promise.all` | plan time (`natlang check`, warning) | `compiler/sequential-loops.ts` |
| `type-recursive-function` | A type alias may not mention itself in a function parameter or result position ([SPEC](SPEC.md#iteration-and-termination)) | plan time | `compiler/neuralese.ts`, `native/types.ts` |
| `undeclared-field` | Eval reads a field the declared type lacks ([SPEC](SPEC.md#types)) | eval time (checked before the eval runs) | `compiler/eval-check.ts` |
| `invalid-binding` | An eval binding that would shadow or read itself before it exists (`const plan = plan(...)`) | eval time (before the eval runs) | `scope-compiler.ts` |
| `capture-conflict` | A `let` capture changed under the eval; the write-back is version-checked ([SPEC](SPEC.md#captures)) | eval time (write-back) | `native/runtime.ts` |
| `context-interface-mismatch` | A rebound or replaced context does not satisfy the function's context interface ([SPEC](SPEC.md#contexts)) | load/bind time | `runtime/contexts.ts` |
| `context-new-executable` | Rebinding may not add an executable node | load/bind time | `runtime/contexts.ts` |
| `iteration-unbounded` | An `iterateOn` with a TypeScript predicate needs a measure or step limit ([SPEC](SPEC.md#iteration-and-termination)) | call time, before the first step (`IterationLimitError`) | `runtime/iterate.ts` |
| `iteration-measure-exhausted` | The measure reached 0 with `until` still false (`IterationLimitError`) | call time | `runtime/iterate.ts` |
| `refinement-unsatisfied` | The predicate was judged false ([refinements](ext/refinements.md)) | call time | `native/refinement.ts` |
| `refinement-undecided` | The probability fell in the band under the `reject` policy, or no judge was available | call time | `native/refinement.ts` |
| `refinement-predicate-invalid` | `P` in `Is<T, P>` is empty or not a string literal | plan time (type read); call time for `assume` | `native/types.ts`, `native/refinement.ts`, `runtime/surface.ts` |
| `untrusted-instruction` | An `Untrusted<T>` value interpolated into an inline `nl` template | plan time | `compiler/inline.ts` |
| `neuralese-untyped-literal` | A Neuralese literal or soft function with no contextual type ([neuralese](ext/neuralese.md)) | plan time | `compiler/neuralese.ts`, `compiler/inline.ts`, `scope-compiler.ts` |
| `neuralese-opaque-access` | Host code accessing fields or indices of, or computing with, a soft value | plan time | `compiler/neuralese.ts` |
| `neuralese-condition` | A soft value used as a condition | plan time | `compiler/neuralese.ts` |
| `neuralese-readout-sync` | Implicit string conversion of a soft value in a synchronous function or callback | plan time | `compiler/neuralese.ts` |
| `neuralese-nested` | `Neuralese<Neuralese<T>>` | plan time (type read) | `compiler/neuralese.ts`, `native/types.ts` |
| `neuralese-dialect-mismatch` | A `.nz` entry's block dialect differs from its declared type; a block coerced to a `Neuralese` type (argument, result, `let`, `read`) was stored in another dialect than the type names (`DefaultDialect`: the runtime's reader dialect); text at a slot of another dialect than the port's; a runtime whose declared dialect differs from its port's | load time, call time, runtime creation | `native/nz-file.ts`, `native/values.ts`, `native/runtime.ts`, `neuralese/combinators.ts`, `native/neuralese.ts` |
| `neuralese-block-integrity` | A `.nz` block does not hash to its ID | load time | `native/nz-file.ts` |
| `neuralese-live-capture-save` | A Neuralese function with live captures cannot be saved to a file ([SPEC](SPEC.md#captures)) | save time | `native/nz-file.ts`, `runtime/contexts.ts` |
| `neuralese-readout-unavailable` | Implicit string conversion with no configured read body | call time | `neuralese/combinators.ts` |
| `neuralese-unsupported-backend` | A call needs Neuralese on a backend without support; there is no text fallback | call time | `native/neuralese.ts` |

## Error classes

These are exceptions rather than codes. `NatlangCallError` carries `outcome` (`quiesced` or `failed`) and `detail`
([SPEC](SPEC.md#natural-language-functions)); `RefinementCallError` is a `NatlangCallError` with a refinement code;
`RefinementError` is thrown for a rejected argument, service result or `refine`; `NatlangRecursionError`
(call time) is a recursive call that is not on a smaller argument; `IterationDivergedError`, `IterationStepError`
and `IterationLimitError` (call time) carry `lastState` and the trajectory; `NzFileError` carries the
`neuralese-*` file codes above. All exist in `ts-host/src` (`runtime/kernel.ts`, `runtime/index.ts`,
`runtime/iterate.ts`, `native/nz-file.ts`).

## Mismatches found by the grep

1. `neuralese-dialect-mismatch`: resolved 2026-10-09 for the runtime (stored block dialects are checked at every
   coercion to a `Neuralese` type, DECISIONS.md "representation chosen by use"). The compiler still has no check of
   that name: it sees `DefaultDialect` as its own literal, not the configured dialect.
2. `TypeSyntaxError` and `EvaluationInfrastructureError` are named in the skills but carry no code.
3. Backticked kebab-case words in the docs that are not error codes and have no source in `ts-host/src` as errors:
   `value-not-printed`, `crisp-value`, `single-use`, `full-value-unavailable`, `producer-missing`, `dynamic-text`
   (training-data conversion reasons in `spec/NEURALESE_DATA.md`, found in `compiler/neuralese-conversion.ts`);
   `guidance-only` (an evaluation category in `evaluation/types.ts`); and the rewrite-rule names
   `map-identity`, `map-fusion`, `read-map`, `combine-identity`, `combine-reassociate`, `split-zip`
   ([NEURALESE_REWRITES.md](NEURALESE_REWRITES.md)).
4. Codes in the source that the docs do not name: `nl-not-called`, `nl-not-tag`, `nl-shadowed`, `nl-explicit-captures`,
   `nl-type-arguments`, `nl-parameter-collision`, `nl-unknown-name`, `nl-spread`, `nl-const-capture-write`, `reserved-property`, `duplicate-site`, `neuralese-interpolation`, `neuralese-file`, `context-conflict`, `neuralese-ref-cycle`,
   `neuralese-unknown-block`, `neuralese-file-format`, `neuralese-file-type`, `forbidden-control`,
   `forbidden-loop`, `forbidden-prototype-mutation`, `recursion`. They are outside this index until a document names
   them.
