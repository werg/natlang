---
description: Middle end, dead code elimination. Aggressive DCE and CFG cleanup.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Remove dead code from fn, as LLVM's ADCE and SimplifyCFG do.

1. Live roots: terminators, stores, calls that may have effects (any call not known to be pure), and volatile
   accesses. Mark them live.
2. Propagate, until nothing changes: an instruction is live when a live instruction uses its value. Every
   terminator is live.
3. Sweep. Delete every instruction that is not live.
4. Clean up the CFG, until nothing changes:
   - delete blocks that no path from the entry reaches, and their phi entries;
   - a phi with a single incoming value, or with all the same, is that value;
   - a block that only branches on is removed: its predecessors branch to its target, and the target's phis take its
     incoming values from them;
   - a conditional branch whose targets are the same block becomes unconditional.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
