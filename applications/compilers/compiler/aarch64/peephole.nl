---
description: Back end, machine peephole optimization of one emitted AArch64 function.
args:
  assembly: string
  problem?: string
returns: string
---
Improve the AArch64 function assembly with peephole rewrites, as LLVM's AArch64 peephole and load/store optimizers
do, keeping what it computes. Slide a window of two or three instructions over each block and apply these until none
fits:
- a `mov` of a register to itself goes; a branch to the label right after it goes;
- `str` to a slot followed by `ldr` of the same slot into the same register keeps only the store;
  into another register, it becomes a `mov`;
- `add xA, xB, #k` followed by an access at `[xA]` becomes an access at `[xB, #k]`, when xA is not used again;
- `cmp x, #0` and `b.eq` or `b.ne` become `cbz` or `cbnz`; a test of one bit becomes `tbz` or `tbnz`;
- `mul` followed by `add` of its result becomes `madd`, and followed by `sub` from it becomes `msub`, when the product
  is not used again;
- two loads or two stores of adjacent 8-byte slots from the same base become `ldp` or `stp`;
- a multiplication by a power of two becomes a shift.
A rewrite must not change a value that is used later. Keep the directives as they are.

Answer with the whole function's assembly only, directives included.

problem, when given, says why an earlier answer was rejected (how the generated program misbehaved); make sure your
answer does not have it.
