# Inline natural-language lambdas in `eval`

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

The compiler enumerates value bindings visible at the `nl` expression: parent parameters, persistent eval locals, earlier declarations in the current eval, and bindings in surrounding blocks. Normal lexical shadowing applies. A later declaration, an out-of-scope block local, and a type-only alias are not runtime captures. Named codebase functions remain available through the existing import bridge; they are not copied as values. The explicit positional parameters of a call shadow captured names, just as function parameters shadow outer variables.

Every eligible lexical binding is available to the child under its original name. The compiler **does not infer capture from English words**. Thus the instruction “compare item with threshold” may mention `threshold` without listing it as a parameter or interpolating it into the template. Exact identifier mentions can make a binding prominent in the child's opening; other available names remain in a bounded closure listing and accessible through `read_value`. A false word match can affect presentation only. Backtick-delimited names are unambiguous display hints. Ordinary `${expression}` follows tagged-template semantics: its value is interpolated when the template is evaluated; it is not required for capture.

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

The generated callable retains a reference to the lexical frame and materializes current typed values immediately before crossing the host bridge. Two calls can see different values. Captured large values use the usual bounded preview and `read_value` paging. The parent conversation is never pasted into a child.

There is a deliberate host-boundary difference from a JavaScript closure. The first runtime version gives the child readable captures; child edits to those captures remain local, while parent mutations before a call are visible. The child returns a typed value. Write-through capture mutation would require typed mutable cells, commit deltas, and serialization of concurrent calls sharing a cell. Object-property mutation has the same issue. This must be described honestly until implemented. The child can mutate its explicit inputs under the existing lambda rules.

Unrelated nonportable locals, such as a `Set`, package namespace, or arbitrary JavaScript function, do not make lambda creation fail. The capture manifest marks them unavailable; an instruction that uses one gets a binding-specific diagnostic when statically detectable, or on access. Supported folder/file handles can cross with their existing authority. Capturing `folder` does not turn an ordinary child into a directory reducer. `result`, debug state, `nl`, tool plumbing, and host globals are not accidental captures. Codebase imports remain live according to the named-function invocation rules.

An inline callable can be used repeatedly within its creating `eval`. Persisting it across eval turns needs an opaque handle and a serializable closure-frame reference in `lam.let`; it must not masquerade as a portable JS function. The compiler plan below records identity and captures so that extension does not change the source syntax.

## Inference rules

Instruction prose is opaque to TypeScript. No compiler can infer `Verdict` merely from “judge the note.” The compiler uses a real `ts.Program` and `TypeChecker` over a virtual eval file containing typed parent parameters, locals, codebase declarations, type aliases, `result`, and an intrinsic declaration for `nl`. The present `transpileModule` path parses and emits but has no type checker.

For each tag, collect constraints from these sources:

1. **Explicit annotation.** `nl<(item: Item) => Verdict>` gives names, input types, and value return type. `nl<Verdict>` gives the return type only.
2. **Contextual callable type.** An annotated variable or known callback parameter can supply a function signature. Use `checker.getContextualType(tagNode)` and resolved overloads. Require one compatible call signature. A synchronous callback slot must not silently accept the promised natlang return.
3. **Immediate-call arguments.** For `nl`...`(expression)`, ask the checker for each argument type. These determine positional input types. A spread needs a statically known tuple or an explicit signature.
4. **Immediate-call result.** An annotated assignment, typed function argument or return, or the enclosing lambda's typed `result` slot can supply the value return type. Walk through call and `await` nodes and unwrap the `Promise`. Checking only the tag misses this context.
5. **Later uses within one eval.** For `const judge = nl`...`` followed by calls, or `const verdict = await nl`...`(item)` later assigned to typed `result`, a small constraint solver may propagate a unique type through that local. This is extra analysis; TypeScript does not generally infer backward from arbitrary later uses. Conflicting uses or branches require an annotation.

The compiler type-checks the surrounding eval so ordinary generic helpers and callbacks contribute their real types. It converts resolved `ts.Type` structurally to the portable type tree: primitives, literals, unions, arrays, supported records, optional fields, and resolvable aliases. It unwraps the callable's promise. It rejects `any`, unconstrained `unknown`, unresolved generics, unsupported native objects, and types the runtime cannot check. `typeToString()` is useful for diagnostics but is not a sound substitute for structural conversion.

If no unique return constraint exists, emit a source-span diagnostic: “Return type of this `nl` expression is unknown; annotate the target or write `nl<Verdict>`.” Do not guess from prose, a sample output, or the first model response. Ambiguous overloads and union call signatures get analogous diagnostics. Explicit types remain a short local escape hatch.

I tested the project's TypeScript 5.9 checker with a virtual file. In `const judge: Judge = nl`...``, `getContextualType` on the tag returned `Judge`. In `const answer: boolean = await nl`...`(item)`, the tag itself had no context, while the `await` and call nodes carried the `boolean` expectation. This supports implementing both forms now, but requires walking the containing expression rather than querying only the tag.

## Lowering, identity, and trace

The compiler should emit one plan per source expression:

```ts
type InlineLambdaPlan = {
  sourceSpan: { start: number; end: number };
  definitionId: string; // source revision + AST span; not a user-facing name
  instructions: string;
  parameters: { name: string; type: PortableType }[];
  returns: PortableType;
  captures: { name: string; type: PortableType; mutable: boolean;
              source: 'input' | 'local' | 'block' | 'handle' }[];
  inheritedCodebaseRevision: string;
};
```

Lower the tag to a JS callable. At invocation it sends the definition ID, positional arguments, and current capture values through the existing scope bridge. The host creates a normal child lambda with arguments and closure values in distinct recorded slots. A tag evaluated repeatedly in a loop retains one source definition ID but gets distinct dynamic instance and call IDs. Live codebase edits resolve at invocation as they do for named lambdas.

The present `scopeBridgeValue` copies enumerable data and cannot transport a JS closure/getter. Generated code must therefore read lexical variables on the JS side for each invocation and send checked values. A host-only frame reference could optimize large immutable captures later. The child uses named-lambda completion and error behavior. A child error rejects the awaited call; any parent recovery occurs in ordinary TypeScript and is visible in the trace.

The parent trace records the source span, inferred signature, capture manifest, and invocation IDs. Child IR records explicit arguments, typed capture references or versions, parent call ID, source revision, reasoning/actions, checked return, and errors. Training export preserves the parent's decision to construct and call the child; flattening the child into a final value would lose that behavior.

## Compiler task: ready to start

This task can start independently of the runtime bridge:

1. Add a shared in-memory TypeScript program for one eval, with virtual declarations for the typed scope and codebase. Keep the compiler host behind an adapter for native/browser parity.
2. Find `nl` tags and enclosing calls; use checker symbols and AST scopes to enumerate visible bindings, shadowing, and source spans. Keep all eligible captures available and distinguish explicit inputs.
3. Implement the constraint rules, structural portable-type conversion, and precise diagnostics. Return `InlineLambdaPlan[]` without executing children yet.
4. Test nested blocks, shadowing, reassignment before invocation, immediate calls, contextual callbacks, `result` context, generic helper context, return-only and full annotations, missing/ambiguous return types, spreads, unsupported captures, and identical native/browser plans.
5. After the plans are stable, add lowering, the `inlineCall` bridge operation, child traces/IR, and end-to-end teacher probes.

Acceptance for this compiler task is a stable typed plan, correct source locations, and no successful inference through `any`. It requires no teacher GPU or prompt experiment. Context extraction and portable-type conversion are the main implementation risks and can be tested in isolation.

## Prior art

- [TypeScript contextual typing](https://www.typescriptlang.org/docs/handbook/type-inference) and [function/callback contracts](https://www.typescriptlang.org/docs/handbook/2/functions.html) provide the inference model.
- The [TypeScript Compiler API](https://github.com/microsoft/TypeScript/wiki/Using-the-Compiler-API) provides `Program`, `TypeChecker`, symbols, and type queries. The [Language Service API](https://github.com/microsoft/TypeScript/wiki/Using-the-Language-Service-API) offers an incremental virtual-file host if repeated eval compilation becomes expensive.
- TypeScript already supports [type arguments on generic tagged templates](https://www.typescriptlang.org/docs/handbook/release-notes/typescript-2-9.html).
- [ECMAScript lexical environments](https://tc39.es/ecma262/2024/multipage/ecmascript-language-functions-and-classes.html) define the model for visible names, shadowing, and call-time reads. The host boundary above states the intentional difference for child writes.
