# Inline natural-language lambdas in `eval`

For the project-wide TypeScript compiler, host, portability, application, and documentation migration, see [TypeScript-native natlang integration](TS_INLINE_HOST_INTEGRATION_PLAN.md).

## Surface and intended use

`nl` is a compiler-recognized tagged template expression that creates an anonymous asynchronous natural-language function. It needs no codebase file or generated name. It can be called immediately, assigned to a local, or passed where a typed asynchronous callback is expected. Its child runs through the ordinary lambda interpreter, with its own line marks, checked return, trace, and training segment.

```ts
const limit = 0.8;
const policy = 'Hold a release when evidence conflicts.';

// note is explicit callback input. limit and policy come from the closure.
const verdict: Verdict = await nl`
  Judge note against limit and policy. Return a Verdict.
`(note);

// Context supplies both parameter and return types.
const judge: (note: string) => Promise<Verdict> = nl`
  Judge note against limit and policy. Return a Verdict.
`;
const next = await judge(otherNote);

// A known callback parameter can supply the signature without an intermediate name.
await reviewEach(notes, nl`Judge note against limit and policy.`);

// Only the return type needs spelling when the call determines input types.
const verdicts: Verdict[] = await Promise.all(
  notes.map(note => nl<Verdict>`Judge note against policy.`(note))
);

// Entirely unannotated at the call site: the enclosing lambda's result type
// supplies Verdict, and note's checked type supplies the input type.
result = await nl`Judge note against limit and policy.`(note);
```

Here `reviewEach` must declare its callback parameter as `(note: string) => Promise<Verdict>` for the anonymous argument to be inferred. `nl<R>` names the child's value return type; the TypeScript callable returns `Promise<R>`. `nl<(note: string) => Verdict>` supplies a full value-level signature when desired. Type arguments on tagged templates are already TypeScript syntax. `Array.filter` expects a synchronous predicate, so an async `nl` callable cannot be passed directly to it; use an async map followed by an ordinary filter or loop.

## Lexical closure and implicit mention

The compiler enumerates value bindings visible at the `nl` expression: parent parameters, persistent eval locals, earlier declarations in the current eval, and bindings in surrounding blocks. Normal lexical shadowing applies. A later declaration, an out-of-scope block local, and a type-only alias cannot be captured. When created during a named `foo.nl` invocation or in a callable TypeScript file under `foo/`, the anonymous lambda sees the same callable items under `foo/` as its parent. Callable TypeScript files there may import sibling `.ts` and `.nl` functions and call anonymous natlang lambdas in their own code. The explicit positional parameters of a call shadow captured names, just as function parameters shadow outer variables.

**Exact mentions in the English instruction select captures.** First infer the explicit parameter names, then tokenize the raw template text against the remaining names in the lexical symbol table, using case-sensitive JavaScript identifier boundaries. In `Judge note against threshold and policy`, `threshold` and `policy` are captured when those exact names are in scope; `note` is the explicit callback parameter and therefore is not captured. Backtick-delimited identifiers are also exact mentions and can produce a useful unknown-name diagnostic. References inside `${expression}` are found through the TypeScript AST. Ordinary interpolation still evaluates at template creation, as JavaScript requires; the referenced binding can also be captured for the child's use by name. Unknown ordinary English words remain prose, not unresolved-variable errors. Prompt the agent to use exact variable names when it wants an implicit capture. This intentionally accepts an occasional false match; the capture list is visible in the child opening and trace.

Callback inputs are different from captures. A contextual function type supplies parameter names. In an immediate call, a bare identifier argument can supply its name (`nl`...`(note)` gives the child `note`); other expressions get stable names `input`, `input2`, etc. The child opening and trace show this mapping. If two inferred names collide, the compiler diagnoses the collision instead of silently changing the child's scope.

Capture is late, at **each invocation**, not frozen when `nl` is created:

```ts
let threshold = 0.7;
const judge: (item: Item) => Promise<boolean> = nl`
  Decide whether item clears threshold.
`;
threshold = 0.9;
const passed = await judge(item); // child sees 0.9
```

The lowered `nl` expression creates an actual JavaScript callable that closes over these selected bindings. A `let` changed after creation is read at its current value when the child is called. The Node and browser evaluators invoke their host effect handlers in the same process, so a private inline-call bridge can retain closure accessors by reference. The present `scopeBridgeValue` happens to copy and flatten functions; that is an implementation choice to change for this internal operation, not a fundamental portability constraint. The parent conversation is never pasted into a child.

The target semantics are normal closure semantics: a child can read captured bindings; writes to a captured `let` or a property of a captured object become visible to the parent after the child action succeeds; rebinding a captured `const` is rejected. The runtime must expose typed getter/setter cells or live object handles to the child instead of constructing an independent input snapshot. Child edits must still respect the existing atomic action boundary. Calls that write the same captured state need simple serialization so that two model-driven children do not race. This is runtime work following the compiler task, not a reason to freeze captures in the language design.

Arbitrary JavaScript objects are usable inside the parent's `eval` today, but the current natlang value boundary is narrower. `scopeBridgeValue` flattens object prototypes and functions; `portable()` and `coerce()` accept JSON-like trees and a few special handles, and ordinary scope snapshots reject `Date`, `Map`, `Set`, class instances, functions, and cycles. Inline calls can support these as **live host references** in the same process, with typed property/method access through a handle rather than JSON cloning. Method calls must preserve their receiver, and handles need a session lifetime and identity. The model-facing opening can show type, identity, and bounded observations; `eval` and `read_value` can inspect through the handle. A `WeakMap` or closure can be passed by identity even though its internals cannot be enumerated; only its exposed operations are observable. The trace records operations and observations, with a separate replay strategy for objects that cannot be reconstructed from source inputs. Supporting arbitrary JSO *return targets* also requires extending runtime checking beyond the current portable type grammar. None of this requires excluding a mentioned JSO from the proposed inline-lambda API.

Capturing `folder` preserves its existing authority and does not itself make the child a directory reducer. `result`, debug state, `nl`, and tool plumbing are not accidental captures. An inline callable can be reused within its creating `eval`. Persistence across eval turns needs an opaque handle that retains a live frame while the session is active; a durable or replayed session additionally needs a reconstruction record. The compiler records identity and captures so those runtime choices do not change source syntax.

## Inference rules

Instruction prose does not have a TypeScript return type. The compiler must infer a **target type before running the child**, then put that type and any needed aliases into the child's ordinary typed scope opening and use the existing runtime return-value check. The current opening already displays `result: <type>`; the inline child should use that same mechanism. Inferring the type from the child's produced value would be too late to guide it and could bless an incorrect value. The compiler uses a real `ts.Program` and `TypeChecker` over a virtual eval file containing typed parent parameters, locals, codebase declarations, type aliases, `result`, and an intrinsic declaration for `nl`. The present `transpileModule` path parses and emits but has no type checker.

For each tag, collect constraints from these sources:

1. **Explicit annotation.** `nl<(item: Item) => Verdict>` gives names, input types, and value return type. `nl<Verdict>` gives the return type only.
2. **Contextual callable type.** An annotated variable or known callback parameter can supply a function signature. Use `checker.getContextualType(tagNode)` and resolved overloads. Require one compatible call signature. A synchronous callback slot must not silently accept the promised natlang return.
3. **Immediate-call arguments.** For `nl`...`(expression)`, ask the checker for each argument type. These determine positional input types. A spread needs a statically known tuple or an explicit signature.
4. **Immediate-call result.** An annotated assignment, typed function argument or return, or the enclosing lambda's typed `result` slot can supply the value return type. Walk through call and `await` nodes and unwrap the `Promise`. Checking only the tag misses this context.
5. **Later uses within one eval.** For `const judge = nl`...`` followed by calls, or `const verdict = await nl`...`(item)` later assigned to typed `result`, a small constraint solver may propagate a unique type through that local. This is extra analysis; TypeScript does not generally infer backward from arbitrary later uses. Conflicting uses or branches require an annotation.

The compiler type-checks the surrounding eval so ordinary generic helpers and callbacks contribute their real types. It converts resolved `ts.Type` into a target descriptor. For primitives, literals, unions, arrays, records, optional fields, and resolvable aliases, that descriptor feeds the existing portable return check. For class or host-object targets, it carries the checked TypeScript name and the runtime validator or live-handle contract that must be added for that target. It unwraps the callable's promise. It rejects `any`, unconstrained `unknown`, and unresolved generics rather than pretending to have inferred a useful target. `typeToString()` is useful for the child prompt and diagnostics but is not by itself a runtime validator.

Prefer the enclosing result type and actual call site: `result = await nl`...`(item)` has the parent lambda's return target and the checked type of `item`, with no inline annotations. Propagate through a uniquely typed local when needed. If no unique target exists anywhere, emit a source-span diagnostic: “Return type of this `nl` expression is unknown; annotate the target or write `nl<Verdict>`.” Do not guess from a sample output or the first model response. Ambiguous overloads and union call signatures get analogous diagnostics. Explicit types remain a short local escape hatch.

I tested the project's TypeScript 5.9 checker with a virtual file. In `const judge: Judge = nl`...``, `getContextualType` on the tag returned `Judge`. In `const answer: boolean = await nl`...`(item)`, the tag itself had no context, while the `await` and call nodes carried the `boolean` expectation. This supports implementing both forms now, but requires walking the containing expression rather than querying only the tag.

## Lowering, identity, and trace

The compiler should emit one plan per source expression:

```ts
type InlineLambdaPlan = {
  sourceSpan: { start: number; end: number };
  definitionId: string; // source revision + AST span; not a user-facing name
  instructions: string;
  parameters: { name: string; type: TargetDescriptor }[];
  returns: TargetDescriptor; // inferred before the child starts; shown in its prompt
  captures: { name: string; type: TargetDescriptor; mutable: boolean;
              source: 'input' | 'local' | 'block' | 'handle'; mentionSpan: number }[];
  inheritedCodebaseRevision: string;
};
```

Lower the tag to a real JS callable with lexical accessors for the mentioned captures. At invocation it sends the definition ID, positional arguments, and live accessors through a private inline-call bridge. The host creates a normal child lambda with arguments and closure bindings in distinct recorded slots, and prompts it with the already-inferred return target. A tag evaluated repeatedly in a loop retains one source definition ID but gets distinct dynamic instance and call IDs. Live codebase edits resolve at invocation as they do for named lambdas.

The present `scopeBridgeValue` copies enumerable data and flattens a function into its properties. The inline-call operation must bypass that conversion for its private closure accessors in both Node and browser hosts. Public tool arguments and completed `eval` results can retain their existing validation until live JSO handles are implemented. The child uses named-lambda completion and error behavior. A child error rejects the awaited call; any parent recovery occurs in ordinary TypeScript and is visible in the trace.

The parent trace records the source span, inferred signature, capture manifest, and invocation IDs. Child IR records explicit arguments, typed capture references or versions, parent call ID, source revision, reasoning/actions, checked return, and errors. Training export preserves the parent's decision to construct and call the child; flattening the child into a final value would lose that behavior.

## Relationship to TypeScript

This can be a small **semantic TypeScript extension**. `nl` uses existing tagged-template, generic type-argument, function-call, and `await` syntax, so there is no new grammar or need to fork the TypeScript parser. A declaration for `nl` can make ordinary editors recognize its approximate callable shape. Our compiler pass then gives it the extra meaning TypeScript cannot derive from a string: English-name capture, closure manifests, target inference tied to the natlang parent result slot, and lowering to child invocations. A transformer alone handles lowering but cannot supply the checker context and diagnostics; the virtual `Program`/`TypeChecker` pass must run first. An optional language-service plugin could show our diagnostics and inferred signatures in editors later, but it cannot change `tsc` typechecking or emit. Our compiler remains authoritative. The source remains ordinary `.ts` wherever it does not use `nl`.

## Compiler task: ready to start

This task can start independently of the runtime bridge:

1. Add a shared in-memory TypeScript program for one eval, with virtual declarations for the typed scope and codebase. Keep the compiler host behind an adapter for native/browser parity.
2. Find `nl` tags and enclosing calls; use checker symbols and AST scopes to enumerate visible bindings, shadowing, and source spans. Match exact identifier mentions in raw instruction text to those bindings, record mention spans, and distinguish explicit inputs.
3. Implement call-site and result-context constraints, target descriptors, and precise diagnostics. Return `InlineLambdaPlan[]` without executing children yet.
4. Test nested blocks, shadowing, English mentions and false matches, reassignment before invocation, immediate calls, contextual callbacks, `result` context, generic helper context, return-only and full annotations, missing/ambiguous target types, spreads, JSOs as captured names, and identical native/browser plans.
5. After the plans are stable, add lowering, the `inlineCall` bridge operation, child traces/IR, and end-to-end teacher probes.

Acceptance for this compiler task is a stable typed plan, correct source locations, and no successful inference through `any`. It requires no teacher GPU or prompt experiment. Context extraction and TypeScript-to-runtime target conversion are the main implementation risks and can be tested in isolation.

## Prior art

- [TypeScript contextual typing](https://www.typescriptlang.org/docs/handbook/type-inference) and [function/callback contracts](https://www.typescriptlang.org/docs/handbook/2/functions.html) provide the inference model.
- The [TypeScript Compiler API](https://github.com/microsoft/TypeScript/wiki/Using-the-Compiler-API) provides `Program`, `TypeChecker`, symbols, and type queries. The [Language Service API](https://github.com/microsoft/TypeScript/wiki/Using-the-Language-Service-API) offers an incremental virtual-file host if repeated eval compilation becomes expensive.
- [TypeScript language-service plugins](https://github.com/microsoft/TypeScript/wiki/Writing-a-Language-Service-Plugin) can augment the editor but cannot change core typechecking or `tsc` output; the natlang compiler pass is the language extension.
- TypeScript already supports [type arguments on generic tagged templates](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-2-9.html).
- [ECMAScript lexical environments](https://tc39.es/ecma262/2024/multipage/ecmascript-language-functions-and-classes.html) define the model for visible names, shadowing, and call-time reads. The host boundary above states the intentional difference for child writes.
