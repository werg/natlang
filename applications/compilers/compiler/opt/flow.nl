---
description: Middle end, analysis. Compute one IR function's control-flow graph, dominator tree, dominance frontiers and natural loops.
args:
  fn: string
returns: Flow
---
Analyze fn, one LLVM IR function, as LLVM's DominatorTree, DominanceFrontier and LoopInfo analyses do. Compute it
exactly, in eval, from the function's text.

1. CFG. Each label starts a block; the first block is the entry. A block's successors are the labels its terminator
   names (`br`, `switch`; none for `ret` and `unreachable`). Predecessors are the reverse.
2. Dominators, by the iterative algorithm of Cooper, Harvey and Kennedy. Number the blocks in reverse postorder from
   the entry. The entry's idom is itself during the computation. Repeat until nothing changes: for each other block in
   reverse postorder, intersect the processed predecessors' idoms by walking up both chains (by number) to their
   common ancestor. Report the entry's idom as null. Unreachable blocks have none and are left out.
3. Dominance frontiers. For each block with two or more predecessors, walk up from each predecessor to the block's
   idom, adding the block to the frontier of every block passed.
4. Natural loops. An edge from a latch to a header that dominates it is a back edge. The loop holds the header and every
   block that reaches the latch without passing through the header. Loops with the same header are one loop. A loop's
   exits are the successors of its blocks outside it. Its preheader is the header's only predecessor outside the loop,
   if that predecessor's only successor is the header; otherwise it is null. Depth is how many loops contain it. List
   the loops innermost first.
