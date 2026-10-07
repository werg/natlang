---
description: Back end, analysis. Compute the live intervals of a machine function's virtual registers.
args:
  code: MachineCode
returns: Liveness
---
Compute the live intervals of code's virtual registers (%x, %w and %d with the same number are the same register),
as LLVM's LiveIntervals analysis does. Compute it exactly, in eval.

1. Number the instructions from 0 in order, leaving out labels, comments and frame index lines. Note each instruction's
   defined and used registers. Note its block, and its successors: the next block, and the labels it branches to;
   `ret` has none.
2. Liveness by blocks. Iterate backwards until nothing changes. A block's live-out is the union of its successors'
   live-ins. Its live-in is its uses before any definition in it, plus its live-out minus its definitions.
3. Intervals. A register's interval runs from its first definition to its last use. It is extended to cover every
   block where it is live-in or live-out, from that block's first instruction to its last, so that a value live around
   a loop's back edge covers the whole loop.
4. crossesCall: a `bl` or `blr` lies strictly inside the interval. calls: the numbers of those instructions.
