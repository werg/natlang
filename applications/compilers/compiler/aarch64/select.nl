---
description: Back end, instruction selection. Translate one LLVM IR function into AArch64 machine code over virtual registers.
args:
  fn: string
  context: string
  problem?: string
returns: MachineCode
---
Select AArch64 instructions for fn, as LLVM's instruction selector does, producing machine code over virtual registers
(the MachineCode format).

1. Values. Each SSA value of integer or pointer type gets a virtual register, %x<n> (i64, ptr) or %w<n> (i32, i16,
   i8, i1), and each double gets %d<n>. Each `alloca` becomes a frame index, with its size and alignment.
2. Calling convention, AAPCS64. On entry, copy the arguments out of x0–x7 (w0–w7) and d0–d7 into their virtual
   registers. For a call, move the arguments into x0–x7 and d0–d7 in order, `bl` the callee, and copy the result out
   of x0/w0 or d0. Variadic calls such as printf pass their arguments the same way on Linux. A `ret` moves its value
   into x0/w0 or d0 and ends with `ret`.
3. Patterns, one or two instructions per IR instruction:
   - `add`, `sub`, `mul`, `and`, `orr`, `eor`, `lsl`, `asr` and `lsr`, with immediates where they fit (12 bits,
     optionally shifted);
   - `sdiv` and `udiv`; a remainder is a division and then `msub`;
   - `icmp` is `cmp` and `cset`; a branch on it is `cmp` and `b.<cond>` (eq, ne, lt, le, gt, ge, lo, ls, hi, hs);
     `select` is `csel`;
   - sign and zero extension are `sxtw`, `sxtb` and `uxtb`, or `and` with a mask; `trunc` reads the w register;
   - `sitofp` and `fptosi` are `scvtf` and `fcvtzs`; double arithmetic is `fadd`, `fsub`, `fmul`, `fdiv`, `fsqrt`,
     `fcmp`;
   - `load` and `store` are `ldr`/`str` (`ldrb`/`strb`, `ldrsw`) at an address: a frame index, or a register plus an
     offset;
   - `getelementptr` adds each index times its element size, from LLVM's layout of the type;
   - a global `@name` is `adrp` plus `add :lo12:name`, and a private constant `@.str.N` is the label `.L.str.N`.
4. Phi elimination. For each phi, at the end of each predecessor, before its branch, copy the incoming value into a
   fresh register that holds the phi's value (`mov`, or `fmov` for doubles). Where a predecessor has several successors
   with phis, split the edge with a block of its own.
5. Blocks keep their order. Each block is a label `.L<function>_<n>`, the entry block unlabeled.

context holds the module's types, globals and declarations. Answer with the machine code only: the frame index lines,
then the instructions.

problem, when given, says why an earlier answer was rejected (how the generated program misbehaved); make sure your
answer does not have it.
