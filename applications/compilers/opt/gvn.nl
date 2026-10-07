---
description: Middle end, global value numbering and redundant load elimination.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Eliminate redundancy in fn as LLVM's GVN does: an instruction that computes the same value from the same operands
as one that dominates it is replaced by that earlier value (commutative operands match either way round); a load is
replaced by an earlier load or store of the same address when no store or call that may write memory lies between
them on any path. Delete what becomes unused.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
