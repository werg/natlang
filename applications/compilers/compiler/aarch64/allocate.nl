---
description: Back end, register allocation. Assign physical registers to a machine function's virtual registers by linear scan, spilling when they run out.
args:
  code: MachineCode
  liveness: Liveness
  problem?: string
returns: MachineCode
---
Allocate registers for code by linear scan (Poletto and Sarkar), with liveness its live intervals.

1. Pools. Integer registers that a call may clobber: x9–x15. Integer registers a call preserves: x19–x28. For doubles,
   d16–d31 and d8–d15. x0–x7, d0–d7, x8, x16–x18, x29, x30 and sp are never allocated: the calling convention, the
   platform and the frame use them. An interval that crosses a call must get a preserved register.
2. Scan the intervals by start. Before each, expire the active intervals that ended before it starts, freeing their
   registers. Give it a free register from its pool. When none is free, spill the active interval that ends last: it
   or the new one, whichever ends later, goes to a new frame index of 8 bytes, and the other takes the register.
3. Rewrite. Replace each virtual register by its physical register, as x<n>, w<n> or d<n> as the instruction needs.
   A spilled register is reloaded into x16, x17 or d17 just before each use, with `ldr` from its frame index, and
   stored back with `str` just after each definition.
4. Delete a `mov` whose source and destination are the same register. List the preserved registers you used in a
   comment line `; saved: x19, x20, d8`, which frame lowering reads.

Answer with the machine code only: the frame index lines (the spill slots added), the saved line, then the
instructions.

problem, when given, says why an earlier answer was rejected (how the generated program misbehaved); make sure your
answer does not have it.
