---
description: Middle end, simplification (InstCombine, SCCP, constant folding, SimplifyCFG).
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Simplify fn as LLVM's InstCombine, SCCP and SimplifyCFG do, without changing what it computes: fold constant
expressions; propagate constants through phis and branches, keeping only what is reachable; replace a branch on a
constant with an unconditional one and delete unreachable blocks; merge a block into its only predecessor; fold
algebraic identities (`x + 0`, `x * 1`, `x * 2^k` to `shl`, double negation, comparisons of a value with itself);
turn small if-then-else diamonds that only choose a value into `select`. Never rely on behavior C leaves undefined
beyond the `nsw`/`inbounds` flags already present.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
