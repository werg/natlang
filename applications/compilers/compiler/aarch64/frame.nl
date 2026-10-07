---
description: Back end, frame lowering. Lay out a machine function's stack frame and insert its prologue and epilogue.
args:
  code: MachineCode
  problem?: string
returns: string
---
Lower the frame of code, a machine function after register allocation, as LLVM's PrologEpilogInserter and
AArch64FrameLowering do.

1. Layout. The frame record (x29, x30) sits at the bottom of the frame, at [sp]. Above it come the preserved registers
   the `; saved:` line lists, in pairs, then each frame index at an offset aligned to its alignment. Round the total up
   to a multiple of 16.
2. Prologue at the entry: `stp x29, x30, [sp, #-SIZE]!`, then `mov x29, sp`, then `stp` of the saved registers in pairs
   (`str` for one left over) at their offsets. With no calls, no saved registers and no frame indices, the function is a
   leaf and needs no frame at all.
3. Epilogue before each `ret`: reload the saved registers, then `ldp x29, x30, [sp], #SIZE`.
4. Replace every `[fi#k]` by `[sp, #offset]`. When an offset does not fit the instruction's immediate, compute the
   address into x16 first.
5. Delete the frame index lines and the saved line.

Answer with the instructions and labels only.

problem, when given, says why an earlier answer was rejected (how the generated program misbehaved); make sure your
answer does not have it.
