---
description: Middle end, simplification. Sparse conditional constant propagation, instruction combining and CFG simplification.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Simplify fn as LLVM's SCCP, InstCombine and SimplifyCFG do, without changing what it computes.

1. SCCP. Give each SSA value a lattice cell: unknown, a constant, or overdefined. Mark edges executable starting from
   the entry. Work two lists until both are empty:
   - an edge newly executable: visit its target's phis, and the target's instructions on its first visit;
   - a value whose cell changed: revisit its users in executable blocks.
   An instruction with constant operands folds to a constant. A phi meets only its executable incoming values. A
   branch on a constant marks one edge; on overdefined, all. Then replace every constant value by its constant, make
   each branch on a constant unconditional, and delete blocks that no executable edge reaches, along with their phi
   entries.
2. InstCombine. Visit instructions until none changes. Fold identities: `x + 0`, `x - 0`, `x * 1`, `x | 0`, `x & -1`,
   `x ^ 0` and `x` shifted by 0 are x; `x * 0` and `x & 0` are 0; `x - x` and `x ^ x` are 0. Turn `x * 2^k` into `shl`,
   and unsigned or exact division and remainder by a power of two into shifts and masks. Reassociate constants
   (`(x + 1) + 2` is `x + 3`). Fold a comparison of a value with itself, a `zext` of a comparison compared with 0, and
   double negation. Keep `nsw`, `nuw` and `inbounds` only where they still hold, and add only assumptions the source
   language makes.
3. SimplifyCFG. Merge a block into its only predecessor when that predecessor has no other successor. Thread a branch
   to a block that only branches on. Turn a small diamond (an if-then-else whose arms only compute one value each,
   without side effects) into a `select`. Delete blocks no path reaches.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make your answer free of it.
