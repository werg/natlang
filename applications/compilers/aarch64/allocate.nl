---
description: Back end, register allocation for one AArch64 function.
args:
  assembly: string
  problem?: string
returns: string
---
assembly is a correct AArch64 function that keeps its values in stack slots. Allocate registers, as a linear-scan
or graph-coloring allocator does: compute each value's live range; keep values that are not live across a call in
caller-saved registers (x9–x15, d16–d31) and values live across calls in callee-saved ones (x19–x28, d8–d15, saved
and restored in the prologue and epilogue in pairs); spill only when registers run out. Remove the loads and stores
this makes unnecessary and shrink the frame, keeping `sp` 16-byte aligned and the calling convention intact.

Answer with the whole function's assembly only, directives included.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
