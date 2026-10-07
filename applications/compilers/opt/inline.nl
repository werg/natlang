---
description: Middle end, inlining.
args:
  fn: string
  callees: string
  context: string
  problem?: string
returns: string
---
Inline calls in fn as LLVM's inliner does. callees holds the definitions of the functions fn calls. Inline a call
when the callee is small (about 30 instructions or fewer) or called only here, and it is not recursive and not fn
itself. To inline: copy the callee's blocks into fn with every value and block renamed uniquely, substitute the
arguments for its parameters, split the calling block at the call, turn each `ret` into a branch to the
continuation block, and replace the call's result with a `phi` of the returned values there.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
