# Neuralese

Extension version **0.5-draft**, extending the [core specification](../SPEC.md) (0.5-draft). An extension is optional: a program or host that does not use it is unaffected.

**Scope.** Soft values: the `Neuralese<T, D>` type and its rules, the model-token markers and reference form, Neuralese functions, the `natlang:neuralese` combinators and their laws, `.nz` files, writing and consuming native state, and the unsupported-backend rule. Training and adapters are in [learning.md](learning.md).

`Neuralese<T, D = DefaultDialect>` is a soft value of type `T`: an immutable,
content-addressed native state that a model wrote. A native state is the text of
one producer span, its token IDs and token embeddings, and the positional KV and
hybrid (convolution and recurrent) state the writing model computed for that
span, bound to the writer's checkpoint, tokenizer, embedding and architecture
identity. A model reads it by splicing that state into its own context at the
correct positions. The model contract is in [NEURALESE_PORT.md](../NEURALESE_PORT.md), the
declarations in [neuralese.d.ts](../neuralese.d.ts).

**Type rules.**

- `Neuralese<T>` is not a `T` and a `T` is not a `Neuralese<T>`; moving between
  them is a computation (writing, or `read`).
- Host code may store, pass, and return soft values, but may not access fields
  or indices, compute with them, compare them, spread or iterate them
  (`neuralese-opaque-access`), or use them as conditions (`neuralese-condition`).
  A JavaScript conversion of a soft value to ordinary data performs the existing
  typed `read<T>` computation first and then applies ordinary JavaScript
  behaviour. The conversions are: `String(value)`, an untagged template
  interpolation, `+` or `+=` with a string operand, a direct
  `JSON.stringify(value)` (only its first argument is read; replacer and spacing
  arguments keep normal JavaScript behaviour), `Error(value)` and
  `new Error(value)`, a soft value used as a computed property name or element
  index, `join`, `toString` and `String.prototype.concat` over soft elements, the
  `String` methods `includes`, `startsWith`, `endsWith`, `indexOf`,
  `lastIndexOf`, `localeCompare`, `padStart` and `padEnd` with a soft string
  argument, the receiver methods `trim`, `trimStart`, `trimEnd`, `toLowerCase`,
  `toUpperCase`, `includes`, `startsWith`, `endsWith`, `indexOf`, `lastIndexOf`,
  `slice`, `substring` and `replace` (with string arguments) on a
  `Neuralese<string>`, `Number(value)` and `Boolean(value)` on a soft string,
  number or boolean, and a `Neuralese<boolean>` used as a condition. Any other
  soft value used as a condition is `neuralese-condition`. The conversion is
  awaited at that expression, so it must be inside async code (an `Array.map`
  callback inside async code is lowered for it); synchronous functions and
  callbacks receive `neuralese-readout-sync` rather than silently returning
  promises. Tagged templates keep soft arguments as soft values.
- `T` is any natlang type. `Neuralese<Neuralese<T>>` is rejected
  (`neuralese-nested`). Records and arrays may hold soft fields.
- A `Neuralese<F>` with a function type `F` is callable with `F`'s parameters
  and result.
- Values of different dialects do not unify (`neuralese-dialect-mismatch`);
  `convert` writes a value again. `DefaultDialect` is bound by configuration: it
  is the runtime's reader dialect (`neuralese.dialect`, else the write port's;
  `NatlangRuntime.readerDialect()`). For a native server that is the first
  dialect its `/v1/neuralese/info` declares: `native-state/<architecture>`, or
  `semantic-state/<adapter SHA-256>` for a server with a configured semantic
  adapter. `Neuralese<T>` and `Neuralese<T, "native-state/lfm2">` are one type in
  a runtime that reads `native-state/lfm2`. A text-only runtime leaves it unbound.
- A block's stored dialect is checked wherever a value is coerced to a
  `Neuralese` type (call arguments, results, `let` variables, `read`). A block
  written in another dialect is rejected with `neuralese-dialect-mismatch`,
  naming both dialects, when the slot names an explicit dialect. At a
  `DefaultDialect` slot, a native state from another native or semantic dialect
  is accepted only when the reader's `/v1/neuralese/info` declares a consumption
  mode that can take it (`best-effort-text-retokenization`, or
  `learned-semantic-state-deltas` with the matching adapter); otherwise the
  value is rejected with `neuralese-native-consumer-unavailable`. A block whose
  dialect the runtime's store cannot tell without I/O (one it does not hold) is
  checked by its reference type only. Text at a `Neuralese<string, D>` slot
  (a direct string argument to a `Neuralese<string>` parameter, or plain text
  returned for a declared `Neuralese` result) is written through the configured
  native writer only when `D` is the port's dialect.
  Dialects are described in [NEURALESE_DIALECTS.md](../NEURALESE_DIALECTS.md).

<a id="generic-results"></a>**Generic results.** A named function may declare a representation-generic
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
`return_result`, its value written as a native state). The generated declaration is a
TypeScript generic, `NatlangGenericFunction<[args], T | Neuralese<T>, T>`. A
function declared with a plain result keeps that one representation; where its
crisp result is used as `Neuralese`, the compiler reports
`neuralese-crisp-result`. The Neuralese instance needs a runtime with a reader
dialect (`neuralese-unsupported-backend`, naming the call site) and writes in
the dialect its constraint names (`DefaultDialect`: the reader's). Call records
keep the definition and the representation it ran at; a host runs a call at a
given representation with `invokeAt(fn, args, { kind: "crisp" | "neuralese" })`.

**View.** The builtin `view(value: string, instructions?: string): R` (with
`R: string | Neuralese<string>`) is the one summarizer: without instructions a
faithful compression (everything needed to reproduce the value; trained so that
`read(view(x))` reproduces `x`), with instructions what their purpose needs.
Its `string` instance is the crisp view; its `Neuralese<string>` instance is
the template write of the same body, which a Neuralese server runs at its
declared write site (`POST /v1/neuralese/view`, chunking values longer than
the site's window). Where a call's opening listing would cut off an argument,
a runtime with a Neuralese server lists that argument's view instead, written
for the receiving call, and the variable keeps the whole value.

**Markers.** A model uses two pairs of control tokens, distinct for producing
and consuming. A model writes a soft value as a producer span,
`<|neuralese_produce|>…<|/neuralese_produce|>`: ordinary text tokens that the
model samples with its own token head, ended by the model emitting the end
marker. The server captures the span's text, token IDs, token embeddings and
positional KV and hybrid state, stores them as one native state and returns a
`native_state` part in its place. Producer spans do not nest; a reply that ends
inside one fails (`neuralese-native-state-open`). When the runtime shows the
model a soft value (a parameter, a local, a result, a soft body, a context
item), the server inserts `<|neuralese_consume|>`, splices the stored state at
the following positions, and inserts `<|/neuralese_consume|>`. A model never
emits consume markers (`neuralese-native-state-consume-marker`). The text of
these markers in message content or tool-call arguments is ordinary text: the
server escapes special-token text in content before tokenizing it.

**Literal.** In model output that the runtime handles as text (eval code,
assistant text, or a value at a slot explicitly typed `Neuralese`), the runtime
also accepts the text form `<|neuralese|>…<|/neuralese|>`: it writes the enclosed
text as a native state by forcing it through the configured native writer
(`native.text-write`), and an unclosed literal is written as truncated. Its type
comes from the contextual type (an annotation, a parameter, a return position,
`nl<F>`); a literal without one is a compile error
(`neuralese-untyped-literal`). A runtime without a write port that meets this
text fails with `neuralese-unsupported-backend`.

**Reference form.** Everywhere outside the model (JSON, traces, logs, training
records, interfaces) a soft value is
`{ "$neuralese": { "type": "Neuralese<T>", "id": "nzs1_…" } }`, referring to an
immutable store entry. The ID is `nzs1_` followed by 52 base32 characters, the
SHA-256 of the state's canonical metadata and tensors. Two optional fields
carry per-use derivative metadata: `stop_gradient: true` marks a constant use,
and `source_occurrence` names the producing graph occurrence. Named tensor
artifacts (adapter coefficients, gradients, optimizer state) use `nnt1_` IDs and
their own forms, `{ "$nativeAdapter": { "id" } }` and
`{ "$nativeTensor": { "id", "kind" } }`; they are not soft values
(`neuralese-native-tensor-as-state`). Before type checking eval code, the
runtime replaces each written state with a reference expression,
`__neuralese("nzs1_…")`, or `__neuralese.body("nzs1_…")` for a soft function
body. `gloss(v)` gives a diagnostic text rendering for people; it is not a
value form.

**Neuralese functions.** A function literal is `nl` with a soft body and explicit
captures:

```ts
const triage: Neuralese<(t: Ticket) => Promise<Label>> =
  nl.with({ rubric, history })`<|neuralese|>…<|/neuralese|>`;
```

Its signature comes from the contextual type, as for inline `nl`. Its calls go
only to its definition site's context and its captures; the state grants no
authority. A call renders the soft body as the instructions and lists the
captures in the opening scope. Text functions may take and return soft values;
soft arguments appear as consumed state in the opening declarations.

**Combinators.** `natlang:neuralese` exports `map`, `zip`, `ap`, `combine`,
`empty`, `split`, `splitList`, `read`, `convert`, `gloss`, and the query operator
`ask(v, question)`, defined as `read(map(v, question))`: the answer to `question`
from what `v` holds, a string. `read` is also an eval global, and `read` of an
ordinary string returns it unchanged. `read` is the only typed way from a soft
value to a `T`; it is validated like a call result and fails with
`NatlangCallError`. Implicit conversions use this same `read` body and
require the task to provide a loaded standard library whose read body is in the
task's native archive; a text provider may use an explicit, digest checked
implementation of the declared read source, recorded as non-learned
provenance. Without a configured read body they fail with the structured
`neuralese-readout-unavailable` capability error. The runtime does not turn a
soft value into text by itself, with one exception: when a model returns native
state for a call whose declared result is an ordinary type, the runtime puts
each state's stored text in its place. `split` and `splitList` are the only way
to soft parts of a structured value. `empty()` is the library's archived
zero-token native state, the identity of `combine`; a library without it fails
with `neuralese-native-empty-unavailable`. Each combinator except `empty` and
`ask` is a system natural-language function whose body is a native state,
trainable like any other; a program may bind its own tuned bodies in its
context. A combinator call answers in the normal call trajectory (it may
evaluate code before `return_result`); only `read` keeps the forced template
readout, because its value must be a validated `T`. A body initialised from text
is initialised **in context**: the runtime renders the actual call with a
placeholder state written from the text, and the server writes the body's state
in that call's context (`POST /v1/neuralese/native-states/init-body`). The
result keeps the server's source capture and import receipts; it is not
claimed to equal the text-instructed call, and no agreement gate is applied.
None takes a purpose argument: a value encodes what its write site (the
producing function's instructions, declared result, and context) wrote it for.

**Laws and rewrites.** The combinators approximately satisfy map identity, map
fusion, read/map commutation, combine associativity and identity, and split of
zip; `ask` approximately satisfies `ask(v, q) ≈ q` answered from `read(v)` (answering
from the state matches answering from the full text). The compiler may rewrite programs with these laws when a rule is enabled
for the current model and dialect version
([NEURALESE_REWRITES.md](../NEURALESE_REWRITES.md)); every applied rewrite is
traced.

**Files.** `.nz` files store typed named exports, exact data and soft values,
with the native state and named tensor artifacts they reference, in a
safetensors container ([NEURALESE_FILES.md](../NEURALESE_FILES.md)). They are
imported like modules and are context items in callable folders.

**Writing and consuming.** Writing a soft value is ordinary decoding of the
producer span under the call's text sampling settings. There is no separate
Neuralese temperature, no sampled payload noise, no learned stop decision, and
no distributional state. A server rejects the retired projected-write request
fields `neuralese_temperature`, `neuralese_max_length`, `neuralese_length`,
`neuralese_passes` and `x_natlang_forced` (`unsupported-request-field`), and
projected `neuralese` content parts (`neuralese-retired-part`). A reader
consumes a stored state in one of the modes its `/v1/neuralese/info` declares:

- `exact-native-state`: the state's checkpoint, tokenizer, embedding and
  architecture identity equal the reader's; the stored KV and hybrid state are
  spliced unchanged.
- `same-base-adapter-state`: requested explicitly per state; raw state from
  another adapter actor on the exact same base is spliced with its source
  identity kept.
- `learned-semantic-state-deltas`: a server configured with a learned semantic
  adapter decodes the state's shared per-token codes into deltas for its own
  state. A mismatch fails with `neuralese-semantic-adapter-mismatch`. No such
  adapter is trained yet; the shared-space adapter is a later training stage.
- `best-effort-text-retokenization`: the reader discards the foreign tensors
  and re-encodes the stored text with its own tokenizer. Matching across
  tokenizers is best effort, not exact state interchange.

**Backends.** A call that needs Neuralese on a backend without native state
support fails with `neuralese-unsupported-backend`. There is no text fallback
for such a backend; text re-encoding is a consumption mode of a native reader.
