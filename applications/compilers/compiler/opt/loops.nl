---
description: Middle end, loop optimization. Rotation, induction-variable simplification, strength reduction and unrolling.
args:
  fn: string
  context: string
  flow: Flow
  problem?: string
returns: string
---
Optimize the loops of fn as LLVM's loop passes do. flow gives each natural loop, innermost first; work on each in
that order.

1. Rotation. A loop whose header tests the exit condition and whose body follows becomes a guard before the loop
   (the same test, branching past it) and a bottom test in the latch. The body then runs without a test at the top.
   Fix the phis: values that came into the header now come from the guard's block or from the latch.
2. Induction variables. A basic induction variable is a header phi whose latch value is the phi plus a constant step.
   An expression that is a linear function of one (`i * c + d`, or an address `base + i * size`) becomes a phi of its
   own: it starts at its value for the start, and the step adds `step * c` each iteration. Its old computation is
   deleted (strength reduction). When the exit test compares the variable with a bound, and an equivalent test on the
   derived variable is cheaper, compare that instead.
3. Trip count. When the start, step and bound are constants, compute how many times the body runs.
4. Unrolling. A loop with a constant trip count of at most 8 and a body of at most 20 instructions is fully unrolled.
   Copy the body once per iteration with the induction variables' values substituted, then remove the loop. A loop
   with a body of at most 12 instructions and an unknown trip count is unrolled by 2. The body runs twice per
   iteration while at least two iterations remain, and a remainder copy runs the last one.

context holds the module's types, globals and function declarations. Check the result with
`toolchain.verify(context + "\n" + result)` and fix what the verifier reports. Answer with the transformed function
only (same signature and behavior); if nothing applies, answer fn unchanged.

problem, when given, says why an earlier answer was rejected (the verifier's message, or how the program's output
changed); make sure your answer does not have it.
