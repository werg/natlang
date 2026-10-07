---
description: Middle end, loop optimization (induction variables, strength reduction, rotation, unrolling).
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Optimize the loops of fn as LLVM's loop passes do. Rotate each loop so its exit test is at the bottom, guarded once
before entry. Simplify induction variables and reduce strength: an expression that grows by a constant each
iteration (`i * stride`, an address `base + i * size`) becomes its own phi incremented by the step. Unroll an
innermost loop with a small constant trip count completely; unroll other small innermost loops by 4, with a
remainder loop for the leftover iterations. Leave a loop alone when its trip count or effects make a change unsafe.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
