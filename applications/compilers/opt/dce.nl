---
description: Middle end, dead code elimination and CFG cleanup.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Remove dead code from fn, as LLVM's ADCE and SimplifyCFG do: instructions whose results nobody uses and that have
no effect (not a store, a call that may have effects, or a volatile access); blocks no path reaches; phis with a
single incoming value or all-equal values; blocks that only branch onward (retarget their predecessors and fix the
phis that named them).

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
