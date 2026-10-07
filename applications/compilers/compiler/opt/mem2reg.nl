---
description: Middle end, SSA construction (mem2reg/SROA).
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Promote memory to registers in fn, as LLVM's mem2reg and SROA do. Each `alloca` whose address is only loaded from
and stored to (never passed to a call, stored, compared or offset) becomes SSA values: insert `phi` nodes where
definitions meet (the iterated dominance frontier), replace each load with the value that reaches it, and delete
the alloca with its loads and stores. A struct or array alloca accessed only at constant offsets splits into
scalars first. Keep every other alloca.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
