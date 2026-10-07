---
description: Middle end, inlining. Replace calls to small functions by copies of their bodies.
args:
  fn: string
  callees: string
  context: string
  problem?: string
returns: string
---
Inline calls in fn as LLVM's inliner does. callees holds the definitions of the functions fn calls.

1. Cost. Take each call site in order. Inline it when the callee is defined in callees, is not fn itself, and does not
   call itself, and when its cost fits. The cost is its count of instructions, less 5 for each argument that is a
   constant, and less 15 when this is its only call. It fits at 30 or less.
2. Clone. Copy the callee's blocks into fn. Rename every value and block with a suffix unique to this call site
   (`.i1`). Replace each parameter by the call's argument. Move the callee's allocas to fn's entry block.
3. Splice. Split the call's block at the call. The first part branches to the copy of the callee's entry. Each `ret` in
   the copy branches to the second part. The call's value becomes a phi there of the returned values (or the one
   returned value), and the call is deleted.
Inlining can expose more calls in the copied code; do not inline those in this run.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
