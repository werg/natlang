---
description: Middle end, SSA construction (mem2reg and SROA). Promote stack slots to SSA values with phi nodes.
args:
  fn: string
  context: string
  flow: Flow
  problem?: string
returns: string
---
Promote memory to registers in fn, as LLVM's SROA and mem2reg do. flow is fn's control flow.

1. Split aggregates (SROA). An `alloca` of a struct or array that is accessed only through `getelementptr` with
   constant indices, then loaded or stored, becomes one alloca per accessed field or element.
2. Promotable allocas: those whose address is only the pointer operand of loads and stores of one type. It is never
   stored, passed to a call, compared, cast or offset. Keep every other alloca as it is.
3. Phi placement, per promotable alloca. Its definition blocks are the blocks that store to it. Place a phi for it at
   every block in the iterated dominance frontier of those blocks: their frontiers, then the frontiers of the blocks
   added, until nothing new comes in (use flow's frontiers).
4. Renaming. Walk the dominator tree from the entry, keeping a stack of current values per alloca. The initial value is
   `undef` (`poison` for a pointer). In each block, a phi of the alloca pushes its result, a store pushes its stored
   value, and a load is replaced everywhere by the top of the stack and then deleted. At the end of the block, give each
   successor's phis the incoming value for this block. After its dominator-tree children, pop what the block pushed.
5. Delete the promoted allocas and their stores. Then delete phis that nothing uses, and a phi whose incoming values are
   all the same value (or itself), replacing its uses by that value, until none is left.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
