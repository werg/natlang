# S0: Neuralese specification and foundations

Revision 2, 2026-10-03, ready for owner review. Detailed plan for stage S0 of the [Neuralese programme](README.md). Every rule here follows the decisions table in the README. Where this draft proposes something the owner has not yet decided, it is marked **Proposed** and listed in §14.

S0 produces the specification that every later stage builds against. It writes no model code. Its outputs are normative text in `spec/`, type declarations, conformance cases that fail until S3 and S4 make them pass, and the record formats that S1 and S5 rely on.

Revision 2 changes: mutable parameters are replaced by immutable values, `grad` and iteration (§9); termination rules replace fuel and the caller-chain guard (§8); a function's codebase becomes an explicit, curried context argument (§7); explicit literal captures are snapshots by default (§6); a file format for Neuralese-bearing data is added (§5); per-literal budgets are dropped (generation limits are a runtime matter); compiler rewrites justified by trained laws are kept and specified (§4.3).

## 1. Outputs

| Output | Location |
| --- | --- |
| Neuralese chapter of the language specification | `spec/SPEC.md` (new section "Neuralese"), summary in `TYPES.md` |
| Context-as-argument revision of callable folders, directory reducers and self-improvement | `spec/SPEC.md` (revised "Callable context" and "Directory reducers" sections) |
| Surface declarations: `Neuralese<T, D>`, the combinator library, `grad` | `spec/neuralese.d.ts`, exported as `natlang:neuralese` |
| File format | `spec/NEURALESE_FILES.md`, header schema `spec/neuralese-file.schema.json` |
| Port contract (tokens, literal at token level, wire protocol, cache rules) | `spec/NEURALESE_PORT.md` |
| Dialect version tags and the compatibility check | `spec/NEURALESE_DIALECTS.md` (short) |
| Law-based compiler rewrites | `spec/NEURALESE_REWRITES.md` |
| Execution-graph record and replay rules | `spec/NEURALESE_GRAPH.md`, schema `spec/neuralese-graph.schema.json` |
| Training-data rewrite contract (eager typing, explicit captures) | `spec/NEURALESE_DATA.md` and a `training/api-migrations/` entry |
| Failing conformance cases | `conformance/neuralese/` |
| Decision log | `plans/neuralese/DECISIONS.md` |
| Documentation and iteration fixes | `TRAINING.md` §0 (LFM2.5-350M has 16 layers, not 14); `spec/SPEC.md` "Iteration and recursion" rewritten to the termination rules (§8, §13); the predicate system prompt and limit rules in `ts-host/src/runtime/iterate.ts` |

## 2. The type

```ts
type Neuralese<T, D extends Dialect = DefaultDialect> = …;
```

`Neuralese<T, D>` is a soft value of type `T` in dialect `D`: an ordered block of vectors in `D`'s space that a model reads through the read port. It is a real type of the language, checked by the compiler and by the runtime at call boundaries, after each eval, and at completion, like every other natlang type. Every Neuralese value is immutable.

- **Not a subtype in either direction.** A `Neuralese<T>` is not a `T`, and a `T` is not a `Neuralese<T>`. Moving between them is always a computation: writing in one direction, `read` in the other.
- **Opaque to host code.** The compiler rejects field and index access, arithmetic, comparison, use as a condition, spreading, template interpolation into text, and serialising the payload. Diagnostics: `neuralese-opaque-access`, `neuralese-condition`, `neuralese-interpolation`. Host code may store, pass and return Neuralese values as ordinary values.
- **Any `T`.** `T` may be any natlang type, including records, arrays, unions and function types. `Neuralese<Neuralese<T>>` is rejected (`neuralese-nested`).
- **Applicable.** When `T` is a function type `(…a: A) => Promise<R>`, a `Neuralese<T>` is callable with the same parameters and result. Calling it runs the soft body (§6).
- **Composition with ordinary types.** Records, arrays and dictionaries may hold Neuralese fields: `{ summary: Neuralese<Summary>; path: string }`. Exact host content stays an ordinary value or an existing `Folder`/`Live` reference. The source document's `Handle<T>`, `Structured<S>`, `NeuraleseFunction<F>` and `Crisp<T>` are not needed.
- **Dialect parameter.** `D` defaults to `DefaultDialect`, which a program or runtime configuration binds to a concrete dialect. Values of different dialects do not unify; `convert` (§4) is the only route between them.

**Declaration sketch.** The TypeScript checker enforces opacity and applicability with a branded type:

```ts
declare const brand: unique symbol;
interface SoftValue<T, D> { readonly [brand]: { readonly type: T; readonly dialect: D } }
type Neuralese<T, D extends Dialect = DefaultDialect> =
  T extends (...args: infer A) => infer R
    ? SoftValue<T, D> & ((...args: A) => Promise<Awaited<R>>)
    : SoftValue<T, D>;
```

The native type checker (`ts-host/src/native/types.ts`) gets a new kind, `{ kind: 'neuralese'; element: Type; dialect: string }`, parsed from `Neuralese<…>` in frontmatter, inline signatures and eval-declared types.

## 3. The literal

### 3.1 Model form

A Neuralese literal is a block of vectors between two control tokens:

```ts
<|neuralese|>⟦z1⟧ … ⟦zL⟧<|/neuralese|>
```

The markers wrap only the vectors. Everything that describes the value (its type, a function's signature, explicit captures) is plain TypeScript around the literal, and therefore precedes the opening marker in the token stream, where the writer and its stop head see it:

```ts
const plan: Neuralese<Plan> = <|neuralese|>⟦…⟧<|/neuralese|>;
const triage: Neuralese<(t: Ticket) => Promise<Label>> =
  nl.with({ rubric, history })`<|neuralese|>⟦…⟧<|/neuralese|>`;
```

The literal is an expression of type `Neuralese<T, D>`, where `T` comes from the contextual type: a declaration's annotation, a parameter's type, a return position, or `nl<F>`. A literal with no contextual type is a compile error (`neuralese-untyped-literal`). Because training data is eagerly typed (§11.4), the model learns to write the annotation first.

The same form appears in two directions:

- **Writing.** When the model emits `<|neuralese|>` inside eval code or a tool argument, the server switches to the write procedure: shallow sketch generation, learned stop, blockwise completion, then readback ([port document](sources/port-mechanics-and-training.md) §2). The writer continues until its stop head closes the block. The closing marker is emitted by the write procedure, not sampled as text.
- **Reading.** When the runtime shows the model a Neuralese value (a parameter, a local, an eval result, a staged result, a function's soft body, a context item), it renders the same literal and the server splices the stored vectors into the marked positions.

### 3.2 Reference form

Outside the model, a Neuralese value is an ordinary data value that refers to stored content by unique ID. It follows the existing `$`-tagged portable-value convention:

```json
{ "$neuralese": { "type": "Neuralese<Plan>", "id": "nz1_b7k2…" } }
```

This is the only form in JSON, traces, logs, training records and user interfaces. Files carry Neuralese data in the file format (§5), whose value tree uses the same reference form. There are exactly two renderings: reference form everywhere, and model form at the token level. The runtime converts between them when rendering a conversation (reference → literal) and when parsing a model turn (literal → stored entry → reference). Within compiled eval code, the runtime replaces each model-emitted literal with a reference expression before type checking, so the checker sees ordinary TypeScript.

**Glosses.** `gloss(v)` returns a diagnostic text rendering for people. It is not a third form, never round-trips, and is never shown to a model as the value.

### 3.3 Escaping

The control tokens are registered special tokens. Only the runtime and the write procedure produce them. The text `<|neuralese|>` occurring in ordinary strings, files or tool output is tokenized as text and is never a literal. The renderer must tokenize content with special-token splitting disabled and enable it only for runtime-inserted structure; a conformance case checks this.

### 3.4 Store entries

Every Neuralese value is an immutable, content-addressed entry:

| Field | Meaning |
| --- | --- |
| `id` | `nz1_` + base32 SHA-256 over dialect, shape, dtype and payload bytes. |
| `dialect` | Dialect ID and version, e.g. `nd:natlang@1`. |
| `type` | Canonical type string of `T`. |
| `length`, `width`, `dtype` | Shape of the payload in the dialect's space (§12). |
| `producer` | Invocation and turn that wrote it, the file it was loaded from, the combinator call, or the gradient step that produced it. |
| `truncated` | Whether writing hit the runtime's hard maximum instead of stopping. |

The model server holds a working store. The runtime holds references. Files (§5) are the durable home of entries that a program keeps. For remote servers, the wire protocol (§10) puts, gets and pins entries by ID. An entry is portable across models that speak its dialect and is never silently reused across dialects.

## 4. The combinator library

`natlang:neuralese` exports library functions. None of them is syntax, and none takes a purpose argument (§4.4).

```ts
map<A, B, D>(v: Neuralese<A, D>, f: (a: A) => Promise<B>): Promise<Neuralese<B, D>>;
zip<A, B, D>(a: Neuralese<A, D>, b: Neuralese<B, D>): Promise<Neuralese<[A, B], D>>;
ap<A, B, D>(f: Neuralese<(a: A) => Promise<B>, D>, a: Neuralese<A, D> | A): Promise<Neuralese<B, D>>;
combine<T, D>(...vs: Neuralese<T, D>[]): Promise<Neuralese<T, D>>;
empty<T, D>(): Neuralese<T, D>;
split<T extends object, D>(v: Neuralese<T, D>): Promise<{ [K in keyof T]: Neuralese<T[K], D> }>;
splitList<E, D>(v: Neuralese<E[], D>): Promise<Neuralese<E, D>[]>;
read<T, D>(v: Neuralese<T, D>): Promise<T>;
convert<T, D1, D2>(v: Neuralese<T, D1>, to: D2): Promise<Neuralese<T, D2>>;
gloss(v: Neuralese<unknown, Dialect>): Promise<string>;
```

The learning helpers (`grad`, `valueAndGrad`, `stopGradient`, objectives, optimisers, `save`) are in §9 and are exported from `natlang:learning`.

### 4.1 Meaning

- **`map(v, f)`** runs `f`'s instructions with `v` as its soft argument and writes the result as `Neuralese<B>` instead of reading it out. `f` may be a text or soft natural-language function.
- **`zip`** packs two soft values into one block. **`ap`** applies a soft function to a soft or exact argument and keeps the result soft. Calling a `Neuralese<F>` directly gives `F`'s declared result.
- **`combine`** merges views of the same `T`; **`empty`** is its zero-length identity. Associativity makes parallel tree folds (and directory reducers over soft values) well defined.
- **`split` and `splitList`** are the only way to get soft parts of a structured soft value. They are explicit because each costs a model call; field access stays a compile error.
- **`read`** is the only way out. It produces a `T`, validated like any call result, and fails with `NatlangCallError` on an invalid readout.
- **`convert`** moves a value between dialects.
- **Kleisli composition** needs no combinator. A natural-language function with `Neuralese` parameters and a `Neuralese` result is a Kleisli arrow, and calling such functions in sequence is `flatMap`.

Each combinator except `empty` is a **system natural-language function** whose instructions are a soft body stored in the standard library's own Neuralese file (§5). The operators are therefore trainable like any other function. The library's bodies are the default; a model or program may keep its own tuned bodies as values in its context (per program, promoted to the pool when they help broadly). Tensor registration, the ports, type validation and store access remain runtime primitives under them.

### 4.2 Laws

The laws hold approximately and are trained to some degree in S5 as consistency objectives:

| Law | Form |
| --- | --- |
| Map identity | `read(map(v, x => x)) ≈ read(v)` |
| Map fusion | `map(map(v, g), f) ≈ map(v, x => f(g(x)))` |
| Read/map commutation | `read(map(v, f)) ≈ f(read(v))` |
| Combine associativity | `combine(combine(a, b), c) ≈ combine(a, combine(b, c))` |
| Combine identity | `combine(v, empty()) ≈ v` |
| Split/zip | `split(zip(a, b)) ≈ [a, b]` |

"≈" is measured through `read` and through downstream consumers, never as vector equality.

### 4.3 Law-based compiler rewrites

The natlang compiler (`ts-host/src/compiler/`, which already lowers `nl`, `iterateOn` and eval code) gains an optimisation pass, in the spirit of GHC's rewrite rules, that uses the laws to make programs cheaper:

| Rule | Rewrite | Saves |
| --- | --- | --- |
| Map fusion | `map(map(v, g), f)` → `map(v, compose(f, g))` | one model call and one stored block |
| Read/map commutation | `read(map(v, f))` → `f(read(v))`, or the reverse | the cheaper of an exact call and a soft call |
| Map identity | `map(v, x => x)` → `v` | one model call |
| Combine reassociation | a chain of `combine` → a balanced tree | latency, through parallel calls |
| Combine identity | `combine(v, empty())` → `v` | one model call |
| Split of zip | `split(zip(a, b))` → `{ a, b }` | two model calls |

- **Enabling.** A rule is enabled automatically for the current model and dialect version once a whole-program comparison of rewritten against unrewritten executions shows it does no harm. The comparison is re-run whenever the model or dialect version changes; until then the rule stays off.
- **Composition.** Fusing two natural-language functions requires a composed function. For text functions, `compose(f, g)` is a system natural-language function with a soft body that is itself trained (it is the instruction-level analogue of function composition). For soft functions it is the same operator applied to soft bodies.
- **Traceability.** The trace records every applied rewrite with its rule and the comparison that enabled it, so replay and evaluation can compare rewritten and unrewritten executions.
- **Control.** Rewrites can be disabled per program or call site, for evaluation and debugging.

### 4.4 Purpose

Purpose is not part of the type and not an argument of any combinator. A value is written for a purpose at its write site: the producing function's instructions, its declared result type and its causal context determine what the block encodes. A separate "re-encode this value for purpose X" call is an anti-pattern. If a program needs a different view, it writes one from the exact source with a function whose instructions say what the view is for.

## 5. File format

Neuralese-bearing data is stored in one general file format. Soft functions, skills, data blocks and the standard library's operators are all values in it. The extension is `.nz`.

- **Container.** A safetensors file. Its JSON header (stored in safetensors metadata) holds the format version, the type declarations, the value tree and the dialect of every block. The tensor section holds the blocks, keyed by content ID, so identical blocks are stored once and standard tooling can open the file.
- **Module of named exports.** A file declares named exports, each with a declared type, like a TypeScript module:

  ```json
  {
    "format": "natlang.neuralese-file/1",
    "dialect": "nd:natlang@1",
    "types": "type Rubric = { criteria: string[] }; type Label = 'urgent' | 'normal' | 'spam';",
    "exports": {
      "rubric":  { "type": "Neuralese<Rubric>", "value": { "$neuralese": { "type": "Neuralese<Rubric>", "id": "nz1_…" } } },
      "triage":  { "type": "Neuralese<(t: Ticket) => Promise<Label>>",
                   "value": { "$neuralese-fn": { "body": "nz1_…", "captures": { "rubric": { "$ref": "rubric" } } } } },
      "limits":  { "type": "{ maxItems: number; note: Neuralese<string> }",
                   "value": { "maxItems": 20, "note": { "$neuralese": { "type": "Neuralese<string>", "id": "nz1_…" } } } }
    }
  }
  ```

- **Types.** The header's `types` are natlang type declarations, and every export carries its type in natlang type syntax. Loading checks each value against its declared type, exactly as call boundaries do. Mixed exact and soft data (the `limits` example) is ordinary.
- **Soft functions.** An export of function type stores its body block and its snapshot captures (§6). Captures can refer to other exports or to other files.
- **Imports.** Application and callable-folder code import exports by name: `import { rubric, triage } from './support.nz'`. The build generates a declaration file from the header, as `.nl` files get `foo.d.nl.ts`, so the TypeScript checker sees the declared types.
- **Context items.** A `.nz` file in a callable folder is a context item: its exports become bindings for the functions of that folder (§7).
- **Reading by people.** `natlang nz show file.nz` prints the header with types and references. Glosses are optional.
- **Immutability.** A file's content is a function of its header and blocks. A changed value means a new file content; the codebase's revision history is the version history.

## 6. Neuralese functions

A Neuralese function literal is the existing `nl` construct with a soft body:

```ts
const triage: Neuralese<(t: Ticket) => Promise<Label>> =
  nl.with({ rubric, history })`<|neuralese|>⟦…⟧<|/neuralese|>`;
```

- **Signature.** From the contextual type, as for inline `nl`. Missing evidence is `nl-unknown-return` or `nl-unknown-parameter`, as today.
- **Captures are explicit, snapshots by default.** A soft body cannot be scanned for names, so the capture object lists every binding the body may use. By default captured values are taken when the literal is evaluated, so the function is a self-contained value that can be stored in a `.nz` file. A `let` binding wrapped in `live(x)` is read at each call and may be written back under the existing version-checked rule; a function with live captures belongs to its running scope and cannot be saved. Immutability is used where it helps, not as a rule: text `nl` keeps its implicit live captures and `let` write-back unchanged.
- **Function-typed captures are snapshots.** A binding whose type contains a function type is always captured by value, in text `nl` as well as in literals. Live capture of function values is the one way a closure could reach itself, so this keeps recursion impossible without a runtime guard (§8).
- **Callable context.** A soft function sees its explicit captures and the context of its definition site (§7). A soft body never grants authority: no service, folder or callable item becomes reachable because of what the vectors contain.
- **Recursion and iteration.** The existing rules apply unchanged. A soft function is a definition in the caller chain like any other.
- **Text bodies with soft arguments.** Ordinary text functions may take and return `Neuralese` types. Their soft arguments appear as literals in the opening declarations of the call.
- **Named soft functions** are exports of `.nz` files (§5). They need no separate `.nl` form.

**Executing a call.** The runtime renders the call like a text call, except that the instruction section contains the soft body literal and the opening scope lists explicit captures. The model then works through the usual tools (`eval`, `return_result`, …). New soft values it writes in eval code or in `return_result` go through the write port.

## 7. Context as a curried argument

This section replaces the existing "Callable context", "Captures" (for bound items), "Directory reducers" and recursion rules of `spec/SPEC.md`. The owner has approved changing existing semantics and migrating data where the result is better formulated and more capable.

### 7.1 What is implicit today

A named function `foo.nl` may call exactly the items of its companion folder `foo/`. An inline `nl` sees the nearest `natlang.d/` folder. A directory reducer's first parameter is a `Folder` that it works on in an isolated writable copy, and `folder.apply(reducer)` commits its changes. Self-improvement edits a program's codebase through `edit_code`. Skills, once added, would be further items in such folders.

All of these are the same thing: a function whose behaviour depends on a folder of definitions, which is bound before the call.

### 7.2 The explicit form

Every natural-language function is a curried function of its context:

```ts
type NlFunction<A extends unknown[], R> = (context: Context) => (...args: A) => Promise<R>;
```

- **`Context`** is an immutable, content-addressed folder value holding `.nl` functions, TypeScript modules, `.nz` files and subfolders. Its ID is a hash of its contents, so no context can contain itself, directly or indirectly. Contexts form a DAG.
- **Calls go to the function's own context only.** A bound function can call exactly the items of the context it is bound to. Each item is itself bound to its own context, which existed before the enclosing one. Every call therefore steps down the context DAG.
- **Default binding.** The compiler binds each function's context at its definition site, as the companion-folder rule does today.
- **Rebinding.** `foo.in(context)` returns `foo` bound to a different context with the same signature. It is used for: selecting the skills a task gets; evaluating a candidate value of an item (for `grad`, candidate revisions and A/B evaluation); evaluating revised contexts on separate query cases in meta-learning; substituting fixtures in tests; and promotion.
- **Context interface.** A function's free names (the items its instructions, code or explicit capture list refer to, with their types) form its context interface. Rebinding checks the new context against that interface structurally, as arguments are checked against parameter types.
- **Authority.** Contexts contain definitions and values, not capabilities. Services stay host-supplied and function-scoped. Rebinding cannot widen authority.
- **Directory reducers** become the general way to compute a new context from an old one: a reducer over a context returns a new, immutable context revision. `folder.apply` remains the commit step for real directories.
- **Self-improvement** is a function from a context (and evidence) to a new context. Crisp skill authoring writes new `.nl` and `.ts` items; soft authoring writes `.nz` items. Promotion is binding the program to the new context revision.

### 7.3 Executable nodes come from files

The rule that keeps the call graph known in advance is about **executable nodes**: `.nl` functions and TypeScript functions, the definitions that can be called and that call others. Everything else is data.

- **No new executable nodes outside files.** The executable nodes of any bound context are selected from contexts loaded from files: the program's tree and the libraries it imports, with subsets and unions allowed. Rebinding never introduces a new executable node. A new helper function exists only by being written to a file (for example, by a directory reducer writing a staged tree that is then compiled and loaded).
- **Executable nodes may be edited.** A node's definition may be replaced, provided the result compiles against the node's declared signature and context interface. It cannot gain calls to nodes outside its context.
- **Data may be added and edited freely.** Skills can gain files: procedures, structured knowledge, examples, `.nz` data, `Neuralese<T>` values. Data entries can be added, removed and replaced at will, subject only to type checks where an interface declares their type.
- **Function-typed values are data with fixed reach.** A soft function or inline `nl` value stored in a context is data in this sense: it can be called by code that holds it, but its own calls go only to the executable nodes of its definition site's context, through its declared interface. It cannot add call edges between executable nodes. Runtime-written functions stay under the existing nesting limit for ad hoc definitions.

Termination does not depend on this rule (it follows from §8), but the rule keeps the set of code that can run reviewable and compiler-checked, while leaving skills, data and learned values open.

### 7.4 Why this matters for learning

Tuning a block that lives inside a program becomes differentiation with respect to an argument: the block is a value item of the context, and the context is an argument (§9). No hidden global parameters and no compiler-side lambda lifting are needed.

### 7.5 Migration

Existing spec sections, the self-improvement engine, `natlang.adaptation/v1` artifacts and the compiler's folder rules change; observable behaviour of existing programs should not. Training data built under the old semantics is migrated through an API-migration entry. Content-addressed context revisions need efficient storage and caching for large codebases; folder overlays and continuations already handle similar state.

## 8. Termination

Natlang stays computationally weaker than a general-purpose language, by construction where possible and semantically elsewhere. There is no general fuel parameter.

| Source of unbounded work | Rule |
| --- | --- |
| Recursion through definitions | Impossible by construction: contexts are an acyclic DAG and calls go only to the bound context's items (§7.2). This replaces today's identity-based caller-chain guard. Host TypeScript callbacks into natlang keep a runtime check. |
| Recursion through captured closures | Impossible: function-typed bindings are always captured by value, so a closure cannot reach itself through a capture, even when data `let` captures are live (§6). |
| Self-application | Impossible: recursive function types are rejected. A type alias may not mention itself in a function parameter or result position, directly or through other aliases (`type-recursive-function`). |
| Runtime-written definitions | The existing hard nesting limit on ad hoc definitions (three active layers below a root) stays. |
| `iterateOn` with a TypeScript predicate | Requires a hard bound: `withLimit({ maxSteps })` or a decreasing `withMeasure`. A deadline alone is insufficient. |
| `iterateOn` with a natural-language predicate | No hard bound required. Termination is semantic: the predicate runs under a dedicated system prompt that tells it it is a loop's stopping condition and biases it against divergence (accept a criterion that is reasonably met, do not hold out for unattainable perfection, notice when the state has stopped changing), and the progress judge stays on. `checkProgress('off')` is allowed only together with a hard bound. |

Today's implementation requires a measure or step limit for every iteration, including natural-language predicates, while the built-in help shows unbounded examples. S0 replaces that with the table above.

## 9. Learning as iteration over immutable values

There is no mutable parameter type. A trainable block is an ordinary Neuralese value that lives in a context, typically in a `.nz` file. Training produces new values; promotion binds a new context. The learning surface is exported from `natlang:learning` and is available only to callers given that service.

```ts
grad<A>(f: (a: A) => Promise<Loss>, a: A): Promise<Gradient<A>>;
valueAndGrad<A>(f: (a: A) => Promise<Loss>, a: A): Promise<{ loss: Loss; grad: Gradient<A> }>;
stopGradient<T>(v: T): T;
objectives: {
  crossEntropy(output: Promise<unknown>, expected: unknown): Promise<Loss>;
  selfDistill(output: Promise<unknown>, withFullSource: () => Promise<unknown>): Promise<Loss>;
  logLikelihood(trajectory: Trajectory, weight?: number): Promise<Loss>;
  law(name: LawName, …): Promise<Loss>;
};
optimizers: { sgd(opts): Optimizer; adam(opts): Optimizer };   // init(a) → state; step(state, grad) → state
save(path: string, exports: Record<string, unknown>): Promise<NzFile>;
```

`A` is a Neuralese value or a record or array of Neuralese values, possibly mixed with exact data that is treated as constant. `Loss` and `Gradient<A>` are opaque.

### 9.1 `grad` and `valueAndGrad`

`grad` differentiates a loss with respect to soft arguments, in the manner of JAX's `grad`. It runs `f(a)`, records the execution graph, and replays it with gradients flowing to `a` (§11.3). `valueAndGrad` returns the loss from the same replay, which every training step needs. Because contexts are arguments (§7), tuning a skill inside a program is `grad` with respect to a context item:

```ts
const loss = (s: Neuralese<Skill>) => objective(program.in(ctx.with({ skill: s })), support);
const { loss: l, grad: g } = await valueAndGrad(loss, ctx.skill);
```

### 9.2 `stopGradient`

`stopGradient(v)` makes `v` a constant for differentiation. A learned updater uses it to state whether a gradient digest or trajectory summary is an observed feature or something to differentiate through.

### 9.3 Nested `grad`

`grad` composes: a loss may itself call `grad`, as in meta-learning through update steps. The default is the first-order approximation, which treats inner gradients as constants. Exact second-order differentiation is available on request (`grad(f, a, { order: 2 })`).

### 9.4 Discrete choices

Sampled tokens, tool choices and stop decisions are not differentiable. `logLikelihood(trajectory, weight)` is differentiable: the log-probability of a recorded trajectory under the current soft values. Supervised imitation is `grad` of the log-likelihood of good trajectories. Policy gradient is `grad` of reward-weighted log-likelihood. No separate reinforcement-learning primitive is needed.

### 9.5 Steps, optimisers and iteration

```ts
const opt = optimizers.adam({ lr: 1e-3 });
const step = async (s: { value: Neuralese<Skill>; opt: OptState }, cases: Case[]) => {
  const { grad: g } = await valueAndGrad(loss, s.value);
  return opt.step(s, g);
};
const tuned = await iterateOn(step, { value: ctx.skill, opt: opt.init(ctx.skill) }, support)
  .withLimit({ maxSteps: 200 })
  .until(s => s.loss < target);
```

Optimiser state is an opaque immutable value carried in the iteration state. Every step produces new immutable values; content addressing and garbage collection of unreferenced entries keep this cheap. Batching needs no primitive: a loss that maps over cases with `Promise.all` is batched by the runtime during replay.

### 9.6 Learned updaters and meta-learning

A learned updater is a step function too: a natural-language function that reads the current value, trajectory digests, feedback and a structured gradient digest (identity, role, shape, direction, scale, optimiser state) and writes the next value. Gradient descent and learned updates have the same shape and are interchangeable in `iterateOn`. Meta-learning trains the updater by evaluating the iteration's final value on separate query cases, differentiating through the steps (first-order by default) and training discrete choices with `logLikelihood` objectives.

### 9.7 Promotion and authority

- `save` writes values to a `.nz` file; promotion binds the program to a context containing it.
- A learned update cannot widen any capability.
- The backbone is never changed by `grad` or by updaters; backbone training is a separate, offline process. Small adapters are an exception by decision 37: `grad` and updaters may produce `Adapter<Base, Kind>` values, which act only inside calls whose context binds them ([LEARNING_CONTINUUM.md](LEARNING_CONTINUUM.md) §6).

## 10. Port contract and wire protocol

Full detail goes in `spec/NEURALESE_PORT.md`. S0 fixes:

- **Control tokens.** `<|neuralese|>` and `<|/neuralese|>` take two of LFM2.5's unused reserved token IDs (`<|reserved_7|>` and `<|reserved_8|>` are proposed) with their own trained embeddings. Other backbones register the same token strings.
- **Where literals occur in the template.** Inside message content and inside tool-call arguments (eval code, `return_result` values), within the backbone's native chat template. No new role. The closing marker ends the block, not the message, the tool call or the function.
- **Length.** The writer's stop head alone decides length. There is no budget in the language. The runtime enforces a hard maximum per block to bound runaway generation, with room for the closing marker; reaching it sets `truncated` and is recorded, not treated as learned stopping.
- **Cache and readback.** As in the port document §3: after writing, restore the cache to the block start and prefill the completed payload before continuing. This covers attention KV and short-convolution state.
- **Wire protocol.** The existing model-turn request and response (`ts-host/src/contracts.ts`) gain content parts of the form `{ type: "neuralese", id }`, in both directions, analogous to image parts in chat APIs. A server exposes `PUT/GET /v1/neuralese/blocks/{id}` and declares the dialects it speaks. A backend without Neuralese support rejects such requests, and the runtime fails the call with `neuralese-unsupported-backend`. There is no text fallback.

## 11. Execution graph, replay and training-data rewriting

### 11.1 Graph record

The native trace (`ts-host/src/native/trace.ts`, `reduction-trace/1`) is extended to a graph record. Nodes:

- invocations (definition, context revision, signature, bindings)
- model turns (rendered input references, sampled tokens, seeds)
- block writes (input context, stop decisions, final length, truncation, resulting entry)
- block reads (entry and position)
- `read` readouts and validations
- combinator calls, applied rewrites, effects with their recorded results, `grad` calls and iteration steps

Edges connect each soft value to its producer and to every consumer.

### 11.2 What counts as discrete

Sampled text tokens, tool choices, stop positions, readout choices and host branches are discrete. Replay holds them fixed. Gradients flow only through continuous payloads, projections and soft values. Discrete choices are trained by supervision or outcome-based objectives in S5 to S7.

### 11.3 Replay

The trainer, or the server running `grad`, replays a recorded graph on the GPU:

- It restores the definitions and context revisions used at each node.
- It supplies recorded effect results rather than repeating effects.
- It recomputes everything differentiable, and accumulates gradients at every producer from all of its consumers.

An old observation is not valid evidence after a changed discrete choice. Such cases need fresh rollouts.

### 11.4 Training-data rewrite contract

S1 and S5 render natlang training data through two compiler passes, specified in `spec/NEURALESE_DATA.md` and registered as an API migration:

- **Eager typing.** Insert the checker's inferred types into model-written eval code, so that every declaration, and in particular every Neuralese literal, carries its annotation.
- **Explicit captures.** Rewrite inline `nl` calls that capture by name mention into the `nl.with({ … })` form with the same bindings, using `live(x)` for a captured `let` the call writes back. This teaches the model to name captures, and gives paired text and soft function literals with checked captures.

Both passes are compiler-driven and checked by recompiling the rewritten trajectory. The rendering of literals into model form happens only in the per-model render step, never in model-neutral records.

## 12. Dialects

Dialect stability is not a concern until Neuralese has a large user base. A dialect is a lightweight version tag, specified briefly in `spec/NEURALESE_DIALECTS.md`:

- A dialect `nd:<name>@<version>` names the space a model reads and writes: width, dtype, normalisation and control-token strings. `nd:natlang@1` is the tag of the first S3-trained model.
- When training changes the model's space, the version is bumped. There is no holding of a dialect and no conformance-preservation objective.
- Every stored value and `.nz` file carries its dialect tag. Loading a value with a mismatched tag is rejected; the value is regenerated from its exact source or converted with `convert`.
- Adapters (maps between a model's hidden width and the dialect's space) exist only where a model of a different width must speak an existing dialect, for example a small browser model and a larger server model sharing one.
- `DefaultDialect` is bound by the program's configuration.

## 13. Iteration changes

`spec/SPEC.md` "Iteration and recursion" is rewritten to the rules of §8 and states what the implementation does: `until` accepts TypeScript or natural-language predicates; `withMeasure` supplies a decreasing measure; `withLimit` a hard step bound; `checkProgress(judge | 'off')` replaces or disables the progress judge; reviews are scheduled from per-site step-count and time statistics and never stop a run on their own, only a `divergent` verdict does.

Implementation changes in `ts-host/src/runtime/iterate.ts` (specified in S0, implemented in S4):

- Require `withLimit` or `withMeasure` only when the stopping predicate is TypeScript, or when the progress judge is off. Natural-language predicates run without a hard bound.
- Run natural-language predicates under a dedicated system prompt: it says the call is a loop's stopping condition, shows the iteration count and whether recent states changed, and biases toward accepting a reasonably met criterion over demanding unattainable perfection.
- Generate training data for predicates under that prompt, including contrastive cases where the right answer is to accept a good-enough state.

## 14. Open points for the owner

None at revision 2. Dialects are lightweight version tags (§12). Resolved on 2026-10-03: executable nodes come from files, are editable, and data is open (§7.3); text `nl` keeps live captures and `let` write-back, with function-typed captures always by value (§6); the predicate system prompt (§13).

## 15. Exit criteria

- The spec chapters, declarations and schemas above exist and agree with the README's decisions table, with no contradictions.
- `spec/neuralese.d.ts` type-checks against the examples in `conformance/neuralese/`, including the expected compile errors (opacity, untyped literal, nested type, dialect mismatch, context-interface mismatch on rebinding).
- Failing conformance cases exist for: literal round-trip between reference and model form; escaping of marker text in ordinary content; write-port invocation inside eval code and `return_result`; readback agreement with recomputation; each combinator's type and failure behaviour; each rewrite rule's gating and trace record; `.nz` load, type check, import and capture resolution; soft-function capture and authority rules; context rebinding and its interface check; `grad` through a context item; iteration with a TypeScript predicate and reviews off; graph replay with fixed discrete choices; the eager-typing and explicit-capture rewrites.
- `TRAINING.md` §0 and the iteration section of `spec/SPEC.md` are corrected.
- Conformance cases also cover: a context that would contain itself is unrepresentable; rebinding that adds an executable node not present in any loaded file context is rejected; adding data entries is accepted; an edited executable node that does not compile against its interface is rejected; `type-recursive-function`; a TypeScript-predicate iteration without a bound is rejected; `valueAndGrad`, `stopGradient`, first-order nested `grad`, and `logLikelihood` over a recorded trajectory.
- `DECISIONS.md` records every decision with date and reason.

## 16. Work items

1. Write the Neuralese chapter in `spec/SPEC.md` and the summary in `TYPES.md` (§§2–4, 6, 9).
2. Write the context revision of "Callable context", "Directory reducers", captures and recursion (§§7–8), including the rebinding provenance rule and the `type-recursive-function` check, with a migration note.
3. Write `spec/neuralese.d.ts` and check it with `tsc` against example programs.
4. Add the `neuralese` type kind to the native type parser and checker as declarations only, with compile errors and no runtime behaviour.
5. Write `spec/NEURALESE_FILES.md` and the header schema (§5).
6. Write `spec/NEURALESE_PORT.md`: token registration, template placement, write-mode switching, hard limit, readback, wire protocol.
7. Write `spec/NEURALESE_REWRITES.md`: rules, gating, composition operator, trace records.
8. Write the short `spec/NEURALESE_DIALECTS.md` (tags, compatibility check, `convert`).
9. Write `spec/NEURALESE_GRAPH.md` and the graph-record schema as an extension of the native trace.
10. Write `spec/NEURALESE_DATA.md` for the two rewrite passes and register the API migration.
11. Write the conformance cases in `conformance/neuralese/`.
12. Start `DECISIONS.md`, fix `TRAINING.md` §0 and the iteration section of `spec/SPEC.md`.
