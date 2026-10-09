# Neuralese

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** Soft values: the `Neuralese<T, D>` type and its rules, the model-token literal and reference form, Neuralese functions, the `natlang:neuralese` combinators and their laws, `.nz` files, stochastic writing and the unsupported-backend rule. Training and adapters are in [learning.md](learning.md).

`Neuralese<T, D = DefaultDialect>` is a soft value of type `T`: an immutable,
ordered block of vectors in dialect `D` that a model reads through its read
port. The model contract is in [NEURALESE_PORT.md](../NEURALESE_PORT.md), the
declarations in [neuralese.d.ts](../neuralese.d.ts).

**Type rules.**

- `Neuralese<T>` is not a `T` and a `T` is not a `Neuralese<T>`; moving between
  them is a computation (writing, or `read`).
- Host code may store, pass, and return soft values, but may not access fields
  or indices, compute with them, compare them, use them as conditions, spread
  them (`neuralese-opaque-access`, `neuralese-condition`). A JavaScript string
  conversion (`String(value)`, an untagged template interpolation, `+` with a
  string operand, or a direct `JSON.stringify(value)`) performs the existing typed
  `read<T>` computation first and then applies ordinary JavaScript formatting.
  For `JSON.stringify`, only its first argument is read; replacer and spacing
  arguments keep normal JavaScript behavior. The conversion is awaited
  at that expression, so it must be inside async code; synchronous functions
  and callbacks receive `neuralese-readout-sync` rather than silently returning
  promises. Tagged templates keep soft arguments as soft values.
- `T` is any natlang type. `Neuralese<Neuralese<T>>` is rejected
  (`neuralese-nested`). Records and arrays may hold soft fields.
- A `Neuralese<F>` with a function type `F` is callable with `F`'s parameters
  and result.
- Values of different dialects do not unify (`neuralese-dialect-mismatch`);
  `convert` moves between them. `DefaultDialect` is bound by configuration: it
  is the runtime's reader dialect (`neuralese.dialect`, else the write port's;
  `NatlangRuntime.readerDialect()`), so `Neuralese<T>` and
  `Neuralese<T, "nd:x@1">` are one type in a runtime that reads `nd:x@1`. A
  text-only runtime leaves it unbound.
- A block's stored dialect is checked wherever a value is coerced to a
  `Neuralese` type (call arguments, results, `let` variables, `read`): a block
  written in another dialect is rejected with `neuralese-dialect-mismatch`,
  naming both dialects. A block whose dialect the runtime's store cannot tell
  without I/O (one it does not hold) is checked by its reference type only.
  Text at a `Neuralese<string, D>` slot is written only when `D` is the port's
  dialect.
  Dialects are version tags ([NEURALESE_DIALECTS.md](../NEURALESE_DIALECTS.md)).

**Generic results.** A named function may declare a representation-generic
result: a type parameter constrained to a crisp type `T` and its soft form,

```yaml
generic:
  R: string | Neuralese<string>
returns: R
```

Each call site instantiates `R` from the type its result is used as, as an
inline `nl` takes its signature from context: a typed `const`, a parameter it
is passed to, a return position. A call whose context names no type, or one
that also admits `T`, gets `T`. The compiler records the chosen instance on the
call; the runtime runs the one body at it: `T` as ordinary execution, and
`Neuralese<T, D>` as a template write (`readout: template`: the first reply is
`return_result`, its value written as a block). The generated declaration is a
TypeScript generic, `NatlangGenericFunction<[args], T | Neuralese<T>, T>`. A
function declared with a plain result keeps that one representation; where its
crisp result is used as `Neuralese`, the compiler reports
`neuralese-crisp-result`. The Neuralese instance needs a runtime with a reader
dialect (`neuralese-unsupported-backend`, naming the call site) and writes in
the dialect its constraint names (`DefaultDialect`: the reader's). Call records
keep the definition and the representation it ran at; a host runs a call at a
given representation with `invokeAt(fn, args, { kind: "crisp" | "neuralese" })`.

**Literal.** At the model-token level a soft value is written
`<|neuralese|>⟦z1⟧…⟦zL⟧<|/neuralese|>`: two control tokens around the vectors and
nothing else. Its type comes from the contextual type (an annotation, a
parameter, a return position, `nl<F>`); a literal without one is a compile error
(`neuralese-untyped-literal`). When the model emits the opening token in eval
code or a tool argument, the server writes the block until its stop head closes
it. When the runtime shows the model a soft value (a parameter, a local, a
result, a soft body, a context item), it renders the literal and the server
splices the stored vectors in. The text `<|neuralese|>` in ordinary content is
text, never a literal.

**Reference form.** Everywhere outside the model (JSON, traces, logs, training
records, interfaces) a soft value is
`{ "$neuralese": { "type": "Neuralese<T>", "id": "nz1_…" } }`, referring to an
immutable, content-addressed store entry tagged with its dialect. Before type
checking eval code, the runtime replaces each emitted literal with a reference
expression. `gloss(v)` gives a diagnostic text rendering for people; it is not a
value form.

**Neuralese functions.** A function literal is `nl` with a soft body and explicit
captures:

```ts
const triage: Neuralese<(t: Ticket) => Promise<Label>> =
  nl.with({ rubric, history })`<|neuralese|>⟦…⟧<|/neuralese|>`;
```

Its signature comes from the contextual type, as for inline `nl`. Its calls go
only to its definition site's context and its captures; the vectors grant no
authority. A call renders the soft body as the instructions and lists the
captures in the opening scope. Text functions may take and return soft values;
soft arguments appear as literals in the opening declarations.

**Combinators.** `natlang:neuralese` exports `map`, `zip`, `ap`, `combine`,
`empty`, `split`, `splitList`, `read`, `convert`, and `gloss`. `read` is the only
way from a soft value to a `T`; it is validated like a call result and fails with
`NatlangCallError`. Implicit string conversions use this same `read` body and
require the task to provide a loaded standard library; a text provider may use an
explicit, digest checked implementation of the declared read source, recorded as
non-learned provenance. Without a configured read body they fail with
the structured `neuralese-readout-unavailable` capability error. The runtime
never reads vector payloads as text. `split` and `splitList` are the only way to soft parts of a
structured value. Each combinator except `empty` is a system natural-language
function with a soft body, trainable like any other; a program may bind its own
tuned bodies in its context. None takes a purpose argument: a value encodes what
its write site (the producing function's instructions, declared result, and
context) wrote it for.

**Laws and rewrites.** The combinators approximately satisfy map identity, map
fusion, read/map commutation, combine associativity and identity, and split of
zip. The compiler may rewrite programs with these laws when a rule is enabled
for the current model and dialect version
([NEURALESE_REWRITES.md](../NEURALESE_REWRITES.md)); every applied rewrite is
traced.

**Files.** `.nz` files store typed named exports, exact and soft, in a
safetensors container ([NEURALESE_FILES.md](../NEURALESE_FILES.md)). They are
imported like modules and are context items in callable folders.

Writing a Neuralese value is stochastic, gated by a Neuralese temperature `τ`
separate from the text temperature: the payload is `μ + τ·σ⊙ε`, deterministic at
`τ = 0` (the inference default). `logLikelihood` includes the log-density of
sampled payloads, so encodings can be trained by sampling-based objectives as well
as by gradients through the payload; `klPrior` is the VAE-style regulariser. Stored
blocks may be Gaussian distributions ([NEURALESE_PORT.md](../NEURALESE_PORT.md),
[NEURALESE_FILES.md](../NEURALESE_FILES.md)).

**Backends.** A call that needs Neuralese on a backend without support fails with
`neuralese-unsupported-backend`. There is no text fallback.
