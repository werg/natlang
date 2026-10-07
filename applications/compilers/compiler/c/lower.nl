---
description: Front end, IR generation for one C function, unoptimized.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Translate the C function fn into one LLVM 22 IR function definition that starts with fn.signature, the way clang
does at -O0: an `alloca` for each parameter and local variable in the entry block, a store of each parameter, loads
and stores for every use, one basic block per control-flow region, and C's semantics exactly (integer promotions
and conversions, signed `nsw` arithmetic, short-circuit `&&` and `||`, `getelementptr` for arrays and struct
fields, `sdiv`/`srem` for signed division).

context holds the module's types, globals, string constants (`@.str.N`, commented with the literal they hold) and
the declarations of every function; use those names. Check your function with
`toolchain.verify(context + "\n" + yourFunction)` before answering, and fix what the verifier reports.

Answer with the function definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
