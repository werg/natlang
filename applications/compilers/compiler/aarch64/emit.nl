---
description: Back end, emission. Print one lowered AArch64 function as GNU assembler text, as LLVM's AsmPrinter does.
args:
  name: string
  body: string
  problem?: string
returns: string
---
Emit the function name, whose body (instructions and block labels) is final, as an assembler file's text section.
- Start with `.text`, `.globl name` (unless the IR made it internal or private), `.p2align 2`, and
  `.type name, %function`, then the label `name:`.
- Then the body as it is: one instruction per line, indented by a tab, with a tab between the mnemonic and its
  operands. Labels sit at the start of a line. Keep every local label unique to this function (`.L<name>_<n>`), and
  every reference to a constant as `.L.str.N`.
- End with `.size name, .-name`.

Answer with the assembly only.

problem, when given, says why an earlier answer was rejected (how the generated program misbehaved); make sure your
answer does not have it.
