---
description: Middle end, pass manager. Choose the optimization pipeline for one function.
args:
  fn: string
  level: Level
returns: Pass[]
---
fn is one LLVM IR function. Choose the passes that will make it faster at this level, in order; a pass may appear
more than once (a cleanup after a pass that exposes more). Leave out a pass that cannot apply to this function:
mem2reg without allocas, licm and loops without loops, inline without calls to defined functions.

- O1: promote memory, simplify, remove dead code.
- O2: also inline small callees, eliminate redundancy, hoist loop invariants, strength-reduce and unroll loops, and
  clean up after them.
- O3: as O2, with a second round of simplification and redundancy elimination after the loop passes.

The passes: mem2reg (allocas to SSA), simplify (folding, constant propagation, peepholes, CFG simplification),
inline (calls to small functions), gvn (redundant computations and loads), licm (loop-invariant code motion),
loops (induction variables, strength reduction, rotation, unrolling), dce (dead code and empty blocks).
