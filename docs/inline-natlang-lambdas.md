# Inline natural-language lambdas in `eval`

## Purpose

An interpreter may write one `eval` that performs ordinary TypeScript work and delegates a semantic judgment to a child natural-language lambda. The child is a normal lambda invocation: it receives typed inputs, can call the parent's available codebase functions, marks its own instructions, and returns a checked value. Its model turns and actions remain visible in the trace and training IR.

```ts
type Assessment = {
  trend: 'improved' | 'stable' | 'regressed';
  context: 'linked' | 'uncertain' | 'unrelated';
  conclusion: 'proceed' | 'hold' | 'rollback';
};

const change = mean(focus_after) - mean(focus_before)
             - (mean(control_after) - mean(control_before));
const note = lookup_context(case_id);
const assess = nl<(change: number, note: string) => Assessment>`
  Interpret the investigation note and the adjusted change together.
  Decide whether the release should proceed, be held, or be rolled back.
`;
result = await assess(change, note);
```

`nl` is a tagged template provided by `eval`. Its type argument is the *value-level* function signature; invoking the resulting function is asynchronous, so the call uses `await`. This avoids requiring `Promise` in the natlang value type grammar.

## Closure and invocation

- The compiler turns each `nl` template into a definition with a stable identity within the parent invocation, instruction text, a function signature, and a snapshot of visible portable bindings. The snapshot includes the current values of parameters and locals. A binding named in the instructions is therefore available to the child without the author spelling it again as a parameter. Explicit call parameters take precedence over captured names.
- Capture occurs when the `nl` expression executes, like an ordinary JavaScript closure. Subsequent parent mutations do not silently change a child already created. Calls can use different explicit arguments with the same definition.
- The child inherits callable codebase imports and their current definitions. Live edits to codebase files are resolved at invocation, as with named functions. It does not receive unrelated host globals or a stringified parent conversation.
- Captures must be serializable typed values or supported handles. An unsupported capture is reported at the `nl` definition, with the binding name. Function values created only inside an `eval` are not automatically transported into the child; named codebase functions remain available through the call bridge.
- The child has the ordinary lambda tools and completion rules. A directory reducer parent may pass a folder explicitly to an inline directory reducer in a later extension; the first implementation should cover ordinary value-returning lambdas.

## Types

The initial surface requires an explicit signature, as in `nl<(change: number, note: string) => Assessment>`. The runtime checks each positional argument and the child return against that signature. The compiler should resolve local type aliases from the same `eval` and named codebase types, then lower the signature into the portable type tree.

TypeScript contextual inference can be added for unannotated `nl` expressions where a unique expected function or return type exists: an annotated variable, an argument to a known typed function, or the enclosing lambda's typed `result` assignment. If the expected type is absent or ambiguous, the compiler asks for an explicit signature. The current scope compiler uses `transpileModule` plus its own portable annotation parser; it does not have a TypeScript `TypeChecker`, so arbitrary call-site inference is not presently implementable by a small runtime change.

## Trace and IR

The parent `eval` action records the emitted inline definition, declared type, instruction text, capture names and values or their typed references, and an invocation identifier. Each child call records explicit arguments, parent call ID, child action sequence, checked return or error, and the source definition revision. The child trajectory is materialized as its own bounded training segment while retaining its parent link. Training export must preserve the model's choice to create and call the semantic lambda; flattening everything into the returned value would lose that decision.

## Completion and errors

An inline lambda cannot be returned or stored as an ordinary portable value; it is a callable local to the current `eval`. Its checked return can be stored. A child error rejects the awaited call and follows the same caller-visible error path as a named lambda. The parent may catch that error in TypeScript when the task permits recovery. The child does not receive validation feedback solely to make a bad instruction appear successful.

## Implementation order

1. Add typed `nl` syntax extraction and free-binding capture to the scope compiler, with diagnostics for unresolved signatures and unsupported captures.
2. Add an inline-definition operation to the existing scope bridge, then invoke the resulting child through the ordinary native runtime call path. Keep browser and native `eval` syntax and observations aligned.
3. Record definition and child-call provenance in reduction traces and shared IR, then test native and browser replay and training export.
4. Use the semantic follow-up corpus to compare an explicit multi-`eval` solution with an inline-child solution. Success means both return the checked value and retain the semantic child reasoning and actions in training data.

The first implementation should keep explicit signatures. General contextual inference is a separate compiler task after the runtime and IR contract is proven.
