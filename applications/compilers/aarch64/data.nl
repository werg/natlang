---
description: Back end, data emission for AArch64 Linux.
args:
  header: string
  problem?: string
returns: string
---
header is the LLVM IR of a module's types, globals and constants (no function bodies). Write the GNU assembler text
for its data on AArch64 Linux: initialized globals in `.data`, zero-initialized ones in `.bss`, constants and
string literals in `.section .rodata`, each aligned to its type, with LLVM's layout of every value (struct
padding, little-endian integers, IEEE doubles as `.xword`/`.8byte` bit patterns or `.double`). Name symbols as the
back end does everywhere: a global `@name` is `name` (`.globl` unless it is private or internal), and a private
constant such as `@.str.3` is the local label `.L.str.3`. Declarations of external functions produce nothing.

Answer with the assembly only.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
