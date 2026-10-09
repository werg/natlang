# Refinement and trust types: Is<T, P> and Untrusted<T>

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** Natural-language refinement types (`Is<T, P>`, `refine`, `assume`, the `refinements` settings and the judge) and the `Untrusted<T>` data-versus-instruction marking, including the `untrusted-instruction` compile check.

## Is<T, P> refinement types

`Is<T, "predicate">` is `T` plus a natural-language predicate that its values satisfy:
`type Reply = Is<string, "a reply to the customer that is polite and does not blame them">`. `T` is any natlang
type; `P` is one nonempty string literal that describes the wanted values positively (a conjunction is one
sentence). `Is<Is<T, "a">, "b">` means both. Whitespace in `P` is normalized. An empty or non-literal `P` is
`refinement-predicate-invalid`, reported when the type is read.

Structure is unchanged: a value is checked against `T` as before, and a refined value is a `T` everywhere. The
predicate is an obligation checked where a value enters a refined slot:

- the return of an `nl` call or a `.nl` function, after the structural check (a failure goes back to the executing
  model as a tool error on `return_result` or its reply, and the model repairs it within the repair budget:
  `maxFailureRepairs`, else `refinements.repairs`, default 3; past it the call fails with the refinement's code);
- an argument into a refined parameter, checked at the caller (a failure throws and the callee does not start);
- a service result whose type is declared in `refinements.services`;
- `refine(value, predicate)`, which checks and returns an `Is<T, P>`, and `assume(value, predicate)`, which
  returns one without checking and records the assumption in the trace.

Records, arrays and dictionaries check each refined position; the checks of one value are issued together.

Fit: `Is<B, P>` fits `T` when `B` fits `T`; `T` fits `Is<B, P>` when `T` fits `B`, with the check as the
obligation of that site; `Is<B, P>` fits `Is<B', P'>` without a check when `B` fits `B'` and the predicates
are equal after whitespace normalization (for nested types, every predicate of the target is present).

**The check.** The judge is one `readout: decision` scoring pass on the call's model (or `refinements.judge`):
the value is shown as data, the predicate as the question, and P(true) is read from the scores of `true` and
`false`. A verdict is cached by `(sha256 of the canonical value, normalized predicate, judge id)`, so the same value
is not judged twice. The value passes at P(true) at or above `threshold` (default 0.5). Inside an optional
uncertainty `band` the `policy` applies: `accept`, `reject` (fail with `refinement-undecided`), or `escalate`
to the model named by `escalate`. A model that cannot score replies judges with an ordinary natural-language
call returning a boolean (traced as `judge: "call"`). A result eval returns or finishes, and a refined local, are
judged before they are kept, so a failure is repaired like any rejected eval. `refine<Is<T, "p">>(value)` and
`assume<R>(value)` read the predicate from the type.

**Errors.** `refinement-unsatisfied`: the predicate was judged false. `refinement-undecided`: the probability fell
inside the band under the `reject` policy, or no judge was available. `refinement-predicate-invalid`: `P` is empty
or not a string literal. Each message is one sentence that names the predicate and the fix. A call that ends on a
refinement rejects with `RefinementCallError` (a `NatlangCallError` with `code`); a rejected argument, service
result or `refine` throws `RefinementError`.

**Settings** (`natlang.json`, `refinements`): `threshold`, `band: { low, high }`, `policy`, `mode`, `judge`,
`escalate`, `repairs`, `predicates` (the same fields per normalized predicate) and `services`
(`"service.method": "Is<string, \"…\">"`). `mode` is `crisp` (default: a crisp checker that returns a boolean
decides, `undefined` defers to the judge), `nl` (judge only), or `shadow` (the judge decides; every crisp/judge
disagreement is traced as `refinement_shadow`). Crisp checkers come from the runtime option
`refinements.crisp` the `refinements` table exported by `native/types.ts`, or a `refinements.ts` module of the program that exports `refinements` (loaded by the launcher), keyed by the normalized predicate. With a call store, verdicts are kept in its `refinement_verdicts` table.

**Trace.** Every check is a `refinement_check` event: path, predicate, value, outcome, probability, judge and where
the verdict came from (`crisp`, `judge`, `cache`, `escalation`). `refinement_assumed` records an `assume`.

## Untrusted<T>

`Untrusted<T>` is `T` that came from outside the program: a file, an HTTP body, user text, a tool or service result.
It is structurally `T`; what it adds is how a model sees it and where it may go.

- **Entry.** A service method whose result type is declared in `refinements.services` as `Untrusted<...>` (or a type
  containing it, such as `{ evidence: { message: Untrusted<string> }[] }`), a parameter or captured variable typed
  `Untrusted<...>`, and `untrusted(value, source)` (exported by the runtime module; it returns the value typed
  `Untrusted<T>`). Only the strings at `Untrusted` positions are marked, so a record keeps its trusted fields as
  ordinary literals.
- **Rendering.** The one value renderer (arguments, scope, eval results, staged results) shows a marked string as a
  fenced data block with an info line naming the source, never as a bare literal and never inside instruction text:
  a line "```untrusted data from index.search", the text verbatim, and a closing fence. The fence is three backticks,
  or one more than the longest run of backticks in the text, so no text can close it. A long text is cut at the
  usual budget with the usual cut-off note. Ordinary values render as before.
- **Provenance.** The label is the origin: `service.method` for a declared service result, the label given to
  `untrusted(value, source)`, else `argument <name> of <function>` or `variable <name>`. The first source that marked a
  text is kept. Like refinement evidence it is kept by content (a string has no hidden tag): a registry per task, with a
  host-level registry behind it for text marked before any task runs, bounded in entries and characters. A string the
  program derives from an untrusted one (a slice, a concatenation) is a new string and is not tracked at run time; a
  trusted string equal to a marked one is shown as data too. Mark the exact fields that are outside text, not whole
  records.
- **Fit.** `Untrusted<B>` fits `T` when `B` fits `T`; `T` fits `Untrusted<B>` when `T` fits `B` (a plain value only
  loses trust). In TypeScript `Untrusted<T>` is `T & brand`, so it is a `T` everywhere, and a plain `T` is not an
  `Untrusted<T>` until `untrusted(value, source)` returns one. `Untrusted<Is<T, P>>` and `Is<Untrusted<T>, P>`
  carry both.
- **Instruction positions.** The text of a natlang function is the author's. In a `.nl` file the body is fixed text
  that refers to arguments by name; a value is never spliced into it, so a `.nl` body cannot receive untrusted text as
  instructions and the model reads each argument from the rendered scope. The one splice is the template literal of an
  inline `nl` call, where `${expression}` is evaluated at call time and becomes part of the instructions. There the
  compiler reports `untrusted-instruction` when the interpolated expression is `Untrusted<T>`, or is text built from
  an untrusted expression (`${message.slice(0, 20)}`, `${"[" + message + "]"}`); a number or boolean computed from one
  (`${message.length}`) is not text and passes. Names an instruction mentions (captured by mention or `nl.with`) and
  arguments of the call are not splices: they are rendered as data. The message is one sentence with the fix: pass the
  value as an argument instead, as in ``nl`Summarize the message.`(message)``. Interpolation inside eval code (a
  one-shot ``nl(`... ${x}`)``) is not typed by the project compiler and is not checked.
