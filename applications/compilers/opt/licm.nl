---
description: Middle end, loop-invariant code motion.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Hoist loop-invariant code in fn, as LLVM's LICM does. Find each natural loop (a back edge to a header that
dominates it). An instruction whose operands are all defined outside the loop moves to the loop's preheader (create
one if the header has several entries from outside) when executing it there is safe: arithmetic that cannot trap,
address computations, and loads from memory that nothing in the loop may write (no store to it, no call). Keep
division by a value that might be zero, and everything with side effects, in place. Work from outer loops inward.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
