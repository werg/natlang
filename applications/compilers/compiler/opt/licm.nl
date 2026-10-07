---
description: Middle end, loop-invariant code motion. Hoist computations that do not change inside a loop to its preheader.
args:
  fn: string
  context: string
  flow: Flow
  problem?: string
returns: string
---
Hoist loop-invariant code in fn, as LLVM's LICM does. flow gives each natural loop, innermost first.

For each loop, innermost first:
1. Preheader. If the loop has none, create one. A new block takes every entering edge from outside the loop and
   branches to the header, and the header's phis take one incoming value from it, a phi of the outside values when
   there were several.
2. Invariants. Visit the loop's instructions in dominator order. An instruction is invariant when each operand is a
   constant, is defined outside the loop, or is an invariant instruction. Stores, calls with effects, phis and
   terminators never are.
3. Hoisting. Move an invariant instruction to the end of the preheader, before its branch, when doing so is safe. It
   cannot trap (no division by a value that may be zero, and no load from an address that may be invalid), or its
   block dominates every exit of the loop. A load is invariant only when no store or call in the loop may write its
   address.
4. Sinking and promotion are not needed. Leave everything else in place.
After an inner loop, its hoisted instructions may be invariant in the outer loop too: the outer loop's turn sees them.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
