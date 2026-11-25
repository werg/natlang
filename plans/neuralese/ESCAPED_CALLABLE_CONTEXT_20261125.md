# Escaped callable context and causal ownership

The real bound-lifetime proof executed a named callable after its creator invocation
had completed, recording the child under that completed creator. The owner's task
remained live. Expiring the function would violate higher-order function semantics;
lexical authority and dynamic causal ancestry are different lifetimes.

Inventory before edits:

- `runtime/callable.ts`: named and inline callables share `startCall`; folder apply
  directly invoked metadata; TS module exports entered a raw bound frame. Partial,
  model and context derivations carry the same metadata and pass through these paths.
- `runtime/runtime.ts`: `runtime.bind` entered a raw saved frame, while unbound inline
  creation fallback could instead adopt the ambient caller before its lexical owner.
- `runtime/iterate.ts`: stored iteration frames and bound stopping predicates bypassed
  the common start path. These now resolve the binding at invocation time.
- `runtime/contexts.ts` attaches rebinding's children through callable constructors;
  it does not implement a separate invocation entry or authority owner.
- `runtime/kernel.ts`: crisp admission honored frame services, but agent execution
  and crisp execution selected all task services. Actual lexical-only service tests
  found the mismatch; both execution paths and child frames now retain that subset.
- `runtime/native-entry.ts` and `std/trace.ts` already use the frame's task and parent
  for records, native scopes and graph custody. Task graph/sibling helpers now route
  authenticated closure callers through associations keyed by actual parent ID,
  preserving parallel callers without a mutable task-wide parent slot.
- Node async context and Workerd/browser synchronous slots share these modules;
  no model-serving engine implements a separate JS callable binding. Platform tests
  execute Node and Workerd slot paths. A full browser worker process is not exercised
  by this focused check; source adoption is shared, not claimed as a browser test.

`resolveBoundFrame` preserves the lexical task, captured cells, services, skill files,
owner and weights. It derives parent identity, recursion stack, cancellation and
readout from the actual live caller. A completed creator contributes no dynamic
parent. With no admissible caller the call is an independent root in its live owner
task. A different task sharing the authenticated recorder can own the causal edge
and graph occurrence, without supplying any ambient service grants. Unknown or
foreign recorder owners cannot supply that edge. Closed/cancelled lexical tasks
cannot be silently replaced by a new authority context.

The test uses scripted local agent evals, not any model/provider. A creator retains
a named function with a lexical-only service grant; after the creator ends, a later
natural-language consumer invokes it and obtains the scalar native effect result.
Actual CallCapture records must retain the true parent and child custody, exact
scalar input/output hashes and lexical service effect, with zero model requests.
The escaped function itself remains honestly `nonportable` in the existing value
snapshot codec. This does not claim portable function-value training examples or
invent a JSON representation for the intermediate closure.

Other controls cover callback binding, inline capture identity, partial application,
no-caller roots, same-recorder external caller graph/ordinal ownership, parallel
caller separation, foreign recorder roots and cancelled/closed owner rejection.
Existing nested runtime, authored source service-scope and invocation lifetime
controls remain part of the focused run. Fresh packaged SDK/installed integration
belongs to Assembly's successor source; no frozen production source is edited.

Validation: final Node build and all 26 focused controls passed under ledger
`escaped-callable-module-20261125i` (1.2 GiB peak, 53 s), including bound module
constant access while live and rejection after owner closure. The quick push gate
passed under `escaped-callable-final-20261125h`: source review, 16 ratchets and
TypeScript checks. The focused checks do not exercise learned crisp specialization
with an installed case, a browser worker process, or portable closure serialization;
those are not claimed as passing. The shared agent/crisp service input correction
is applied to both paths, preserving their existing recording wrappers.
