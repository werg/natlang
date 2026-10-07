---
description: Back end, instruction selection and frame layout for one function on AArch64 Linux.
args:
  fn: string
  context: string
  problem?: string
returns: string
---
Translate the LLVM IR function fn into GNU assembler text for AArch64 Linux that a straightforward compiler would
produce: correct first, simple second. Give every SSA value its own 8-byte stack slot; follow the AAPCS64 calling
convention (integer and pointer arguments in x0–x7, floating-point ones in d0–d7, results in x0 or d0, variadic
calls such as printf pass arguments exactly like fixed ones on Linux, `sp` 16-byte aligned, save x29/x30 and set the
frame pointer); resolve each `phi` with copies at the end of its predecessors; lower `getelementptr` to address
arithmetic with LLVM's struct layout; reach a global `@name` as `name` and a private constant such as `@.str.3` as
`.L.str.3`, through `adrp` and `:lo12:`; name local labels `.L<function>_<n>`; compare and branch with `cmp`/`b.cond`; use `sdiv` and `msub` for signed
remainder, `scvtf`/`fcvtzs` for conversions.

context holds the module's types, globals and declarations. Start with `.text`, `.globl`, `.type name, %function`
and `.p2align 2`; end with `.size`. Answer with the assembly only.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
