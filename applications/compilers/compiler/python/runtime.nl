---
description: Front end, the Python subset's runtime library as LLVM IR.
args:
  header: string
  problem?: string
returns: string
---
header declares the runtime functions (`@rt_…`) this program needs, each with a comment stating its contract, and
the `%list` type they work on. Write an LLVM 22 IR definition for each declared runtime function, exactly as its
contract says, using malloc, realloc, printf, scanf and exit from the C library (declare those you use that the
header does not, at the top of your answer). Lists grow by doubling their capacity.

Check your definitions with `toolchain.verify(header + "\n" + yourDefinitions)` with the runtime `declare` lines
removed, and fix what the verifier reports. Answer with the IR only: any extra `declare` lines, then the
definitions.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
