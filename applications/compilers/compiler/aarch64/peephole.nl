---
description: Back end, machine peephole optimization for one AArch64 function.
args:
  assembly: string
  problem?: string
returns: string
---
Improve the AArch64 function assembly with machine peepholes, keeping what it computes: fold address arithmetic
into load and store addressing modes (`[x1, x2, lsl #3]`, pre/post-index); turn compare-with-zero branches into
`cbz`/`cbnz`/`tbz`; use `madd`/`msub` for multiply-add, `csel`/`cset` for small conditional moves, shifts for
multiplication by powers of two, immediate forms where they fit; delete moves of a register to itself, stores
followed by a load of the same slot, and branches to the next instruction; pair adjacent loads and stores into
`ldp`/`stp`.

Answer with the whole function's assembly only, directives included.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
