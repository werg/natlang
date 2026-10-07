---
description: Front end, IR generation for one function of a typed Python program, unoptimized.
args:
  fn: SourceFunction
  context: string
  problem?: string
returns: string
---
Translate the Python function fn into one LLVM 22 IR function definition that starts with fn.signature, the way an
unoptimized compiler does: an `alloca` for each local in the entry block, loads and stores for every use, one block
per control-flow region. Keep Python's semantics within the subset: `range` loops count with the step's sign;
`//` and `%` use the runtime's floor operations; `and`/`or` short-circuit; `len`, indexing and `append` on lists
go through the runtime functions; `print(a, b)` prints the values separated by spaces with a newline (ints with
`%ld`, strings with `%s`, booleans as `True`/`False`); `main` returns 0.

context holds the module's types, globals, string and format constants (`@.str.N`, commented), and the
declarations of the runtime and of every function, each with its contract; use those names. Check your function
with `toolchain.verify(context + "\n" + yourFunction)` before answering, and fix what the verifier reports.

Answer with the function definition only, from `define` to its closing brace.

problem, when given, says why an earlier answer to this same request was rejected (the verifier's message, or how the
program's output changed); make sure your answer does not have it.
